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
1. `data/chunks.jsonl` を読み込み（約 300 チャンク）
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
   - `temperature` (Optional): LLM の temperature（デフォルト: 0.4）
   - `history` (Optional): 会話履歴の配列（デフォルト: []）

3. **チャンクが見つからない場合**

   - `pos` の値が範囲外の可能性があります
   - サーバーログで「該当するチャンクが見つかりませんでした」というメッセージを確認してください

4. **モジュールインポートエラーの場合**
   - プロジェクトルートから起動していることを確認してください
   - 仮想環境が正しくアクティベートされているか確認してください

## API エンドポイント

### `POST /chat`

キャラクターとの対話を行います。

**リクエストボディ**:
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
    }
  ]
}
```

**レスポンス**:
```json
{
  "answer": [
    "キャラクターの返答（1行目）",
    "キャラクターの返答（2行目）"
  ]
}
```

### `GET /health`

ヘルスチェック。

**レスポンス**:
```json
{
  "ok": true
}
```

詳細は http://localhost:8000/docs の Swagger UI で確認できます。
