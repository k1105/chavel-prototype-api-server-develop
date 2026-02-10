# 吾輩は猫である - 対話 API サーバー

小説ベースの対話 API（RAG + ネタバレ防止）

## セットアップ

### 前提条件

- Python 3.8 以上
- Docker（Qdrant 用）
- OpenAI API キー

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

仮想環境を使う場合：

```bash
python3 -m venv myenv
source myenv/bin/activate  # macOS/Linux
# または
myenv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定してください：

```env
OPENAI_API_KEY=your_api_key_here
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

### 3. Qdrant（ベクトル DB）の起動

Qdrant を Docker で起動します：

```bash
# フォアグラウンドで起動
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# または、バックグラウンドで起動
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
```

**ポート情報**：
- `6333`: Qdrant REST API
- `6334`: Qdrant gRPC
- ストレージ: `qdrant_storage` ディレクトリに永続化

**接続確認**：

```bash
curl http://localhost:6333/collections
```

正常に起動している場合、JSON レスポンスが返ります。

### 4. データのインデックス作成

**重要**: 初回セットアップ時に必須の手順です。

チャンクデータを Qdrant にインデックスします：

```bash
python app/index_chunks_qdrant.py
```

このスクリプトは以下を実行します：
1. `data/chunks.jsonl` を読み込み（1,002 チャンク）
2. OpenAI Embeddings API で各チャンクをベクトル化
3. Qdrant の `neko_scenes` コレクションにインデックスを作成

**注意**:
- 初回セットアップ時に必須
- `data/chunks.jsonl` を更新した際にも再実行が必要
- OpenAI API を使用するため、API 使用料が発生します
- 既存のコレクションがある場合は削除されます

## サーバーの起動

### 開発環境

**重要**: プロジェクトルート（`api-server`ディレクトリ）で実行してください。

```bash
# 方法1: uvicornを直接使用（推奨）
python -m uvicorn app.dialog_api:app --reload --host 0.0.0.0 --port 8000
```

または、仮想環境を使用している場合：

```bash
# 仮想環境をアクティベート
source myenv/bin/activate  # macOS/Linux
# または
myenv\Scripts\activate  # Windows

# サーバー起動
python -m uvicorn app.dialog_api:app --reload --host 0.0.0.0 --port 8000
```

または、`app/dialog_api.py` を直接実行：

```bash
python app/dialog_api.py
```

### 本番環境

```bash
python -m uvicorn app.dialog_api:app --host 0.0.0.0 --port 8000
```

サーバー起動後、以下の URL でアクセスできます：

- API: http://localhost:8000
- ヘルスチェック: http://localhost:8000/health
- API ドキュメント: http://localhost:8000/docs

### トラブルシューティング

#### 400 Bad Request エラーが発生する場合

1. **サーバーのログを確認**

   - サーバーのコンソールにエラーメッセージが表示されます
   - バリデーションエラーの場合は、詳細なエラー情報がログに出力されます

2. **リクエスト形式の確認**

   - `/chat` エンドポイントは以下の形式を期待します：
     ```json
     {
       "book_id": "wagahai",
       "character": "吾輩",
       "character_id": 1,
       "question": "質問内容",
       "pos": 1217,
       "k": 8,
       "temperature": 0.4,
       "history": [
         {
           "character_id": null,
           "message": "ユーザーのメッセージ"
         },
         {
           "character_id": 1,
           "message": "キャラクターの返答"
         }
       ]
     }
     ```

   **フィールド説明**：
   - `book_id` (Optional): 書籍ID（現在は未使用）
   - `character` (Optional): キャラクター名（`character_id` がない場合に使用）
   - `character_id` (Optional): キャラクターID（`null` = ユーザー、数値 = キャラクター）
   - `question` (Required): ユーザーの質問
   - `pos` (Required): 本文の現在位置（文字オフセット）
   - `k` (Optional): 取得チャンク数（デフォルト: 8）
   - `temperature` (Optional): LLM の temperature（デフォルト: 0.7）
   - `history` (Optional): 会話履歴の配列（デフォルト: []）

3. **チャンクが見つからない場合**

   - `pos` の値が範囲外の可能性があります
   - サーバーログで「該当するチャンクが見つかりませんでした」というメッセージを確認してください

4. **モジュールインポートエラーの場合**
   - プロジェクトルートから起動していることを確認してください
   - 仮想環境が正しくアクティベートされているか確認してください

## API エンドポイント

### `GET /health`

ヘルスチェック。サーバーが正常に稼働しているかを確認します。

**レスポンス**:
```json
{
  "ok": true
}
```

---

### `POST /context`

会話コンテクストを取得します。キャラクター選択時に呼び出し、初回メッセージと現在の読了位置における状況を取得します。

`/chat` の前に呼び出すことで、状況要約（LLM 呼び出し結果）をサーバー側にキャッシュし、後続の `/chat` を高速化できます。ただし `/context` の呼び出しは必須ではなく、省略しても `/chat` は単独で動作します。

**リクエストボディ**:

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `book_id` | `string` | - | 書籍ID（将来の複数作品対応用） |
| `character_id` | `int` | **必須** | キャラクターID |
| `pos` | `int` | **必須** | 本文の現在位置（文字オフセット） |

```json
{
  "character_id": 1,
  "pos": 1217
}
```

**レスポンス**:

| フィールド | 型 | 説明 |
|---|---|---|
| `character_name` | `string` | 解決されたキャラクター名 |
| `first_message` | `string` | キャラクターの初回挨拶メッセージ |
| `situation` | `string` | 現在の読了位置におけるキャラクターの状況要約（LLM 生成） |
| `scene` | `object \| null` | 現在のシーン情報 |
| `scene.scene_id` | `string` | シーンID（例: `"scene_00042"`） |
| `scene.chapter` | `int` | 章番号 |
| `scene.scene_index` | `int` | シーン通し番号 |

```json
{
  "character_name": "吾輩",
  "first_message": "吾輩は猫である。名前はまだ無い。……で、君は何者だ？",
  "situation": "吾輩: 珍野苦沙弥の書斎の縁側で丸くなっている。...",
  "scene": {
    "scene_id": "scene_00042",
    "chapter": 3,
    "scene_index": 42
  }
}
```

**エラーレスポンス**:

| ステータス | 条件 |
|---|---|
| `404` | `character_id` に対応するキャラクターが見つからない |
| `500` | 内部エラー（LLM 呼び出し失敗等） |

---

### `POST /chat`

キャラクターとの対話を行います。RAG によるコンテクスト検索とキャラクターペルソナに基づいた応答を生成します。

**リクエストボディ**:

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `book_id` | `string` | - | 書籍ID |
| `character` | `string` | △ | キャラクター名（`character_id` がない場合に使用） |
| `character_id` | `int` | △ | キャラクターID。`character` と `character_id` のいずれかが必須 |
| `pos` | `int` | **必須** | 本文の現在位置（文字オフセット） |
| `question` | `string` | **必須** | ユーザーの質問 |
| `k` | `int` | - | 取得チャンク数（デフォルト: `8`） |
| `temperature` | `float` | - | LLM の temperature（デフォルト: `0.7`） |
| `history` | `HistoryItem[]` | - | 会話履歴（デフォルト: `[]`） |

`history` 配列の各要素:

| フィールド | 型 | 説明 |
|---|---|---|
| `character_id` | `int \| null` | `null` = ユーザーの発言、数値 = キャラクターの発言 |
| `message` | `string` | メッセージ内容 |

```json
{
  "character_id": 1,
  "pos": 1217,
  "question": "ここは何処だ？",
  "history": [
    {
      "character_id": 1,
      "message": "吾輩は猫である。名前はまだ無い。……で、君は何者だ？"
    },
    {
      "character_id": null,
      "message": "通りすがりの者です"
    },
    {
      "character_id": 1,
      "message": "通りすがり？怪しい奴だな。"
    }
  ]
}
```

**レスポンス**:

| フィールド | 型 | 説明 |
|---|---|---|
| `answer` | `string[]` | キャラクターの返答（行ごとに分割） |

```json
{
  "answer": [
    "ここは苦沙弥先生の書斎だ。",
    "君のような見知らぬ人間が来る場所ではないのだがな。"
  ]
}
```

**エラーレスポンス**:

| ステータス | 条件 |
|---|---|
| `400` | `character_id` と `character` の両方が未指定 |
| `404` | キャラクターが見つからない |
| `422` | リクエストバリデーションエラー |
| `500` | 内部エラー（LLM 呼び出し失敗等） |

---

### フロントエンドからの想定利用フロー

```
ユーザーがキャラクターを選択
    │
    ├─→ POST /context { character_id, pos }     ← fire-and-forget（完了を待たなくてよい）
    │     → first_message を画面に表示
    │     → situation をサーバー側にキャッシュ
    │
ユーザーがメッセージを入力
    │
    └─→ POST /chat { character_id, pos, question, history }
          → situation はキャッシュヒット → 高速応答
```

`/context` が未完了または未呼び出しの場合でも、`/chat` は自己完結して動作します。同一の `(pos, character_id)` に対する並行リクエストは、サーバー内部のロック機構により LLM の重複呼び出しを防止します。

詳細は http://localhost:8000/docs の Swagger UI でも確認できます。
