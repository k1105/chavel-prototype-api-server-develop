# 吾輩は猫である - 対話 API サーバー

小説ベースの対話 API（RAG + ネタバレ防止）

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定してください：

```env
OPENAI_API_KEY=your_api_key_here
CHAT_MODEL=gpt-4o-mini
EMBED_MODEL=text-embedding-3-small
```

## データベースの起動

### Qdrant（ベクトル DB）

Qdrant は Docker で起動します：

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

または、バックグラウンドで起動する場合：

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
```

- ポート `6333`: Qdrant API
- ポート `6334`: Qdrant gRPC
- ストレージ: `qdrant_storage` ディレクトリに永続化されます

### SQLite（会話履歴）

SQLite は自動的に初期化されます。データは `data/dialog.sqlite` に保存されます。

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
       "character": "吾輩",
       "question": "質問内容",
       "pos": 1217,
       "k": 8,
       "temperature": 0.4,
       "session_id": "optional"
     }
     ```

3. **チャンクが見つからない場合**

   - `pos` の値が範囲外の可能性があります
   - サーバーログで「該当するチャンクが見つかりませんでした」というメッセージを確認してください

4. **モジュールインポートエラーの場合**
   - プロジェクトルートから起動していることを確認してください
   - 仮想環境が正しくアクティベートされているか確認してください

## API エンドポイント

- `POST /chat`: 対話エンドポイント
- `GET /sessions/{session_id}`: セッション履歴取得
- `GET /health`: ヘルスチェック

詳細は http://localhost:8000/docs の Swagger UI で確認できます。
