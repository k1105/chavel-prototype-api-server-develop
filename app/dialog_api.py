"""
対話 API サーバ

FastAPI でキャラクターとの対話を提供
"""

import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import memory, retriever
from app.utils import chat, get_personas_cache, get_character_name_by_id, get_text_around_position

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI アプリ
app = FastAPI(
    title="吾輩は猫である - 対話API",
    description="小説ベースの対話API（RAG + ネタバレ防止）",
    version="1.0.0"
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# バリデーションエラーハンドラー
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """リクエストバリデーションエラーの詳細を返す"""
    logger.error(f"❌ バリデーションエラー: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(await request.body())
        }
    )


# リクエスト・レスポンスモデル
class HistoryItem(BaseModel):
    character_id: Optional[int] = Field(None, description="キャラクターID（null=ユーザー、数値=キャラクター）")
    message: str = Field(..., description="メッセージ内容")


class ChatRequest(BaseModel):
    book_id: Optional[str] = Field(None, description="書籍ID")
    character: Optional[str] = Field(None, description="対話相手のキャラクター名（character_idがない場合に使用）")
    character_id: Optional[int] = Field(None, description="キャラクターID（null=ユーザー、数値=キャラクター）")
    pos: int = Field(..., description="本文の現在位置（文字オフセット）")
    question: str = Field(..., description="ユーザの質問")
    k: Optional[int] = Field(8, description="取得チャンク数")
    temperature: Optional[float] = Field(0.4, description="LLM の temperature")
    history: Optional[List[HistoryItem]] = Field(default=[], description="これまでの会話履歴")


class Citation(BaseModel):
    chapter: int
    start: int
    end: int


class ChatResponse(BaseModel):
    answer: List[str]


class HistoryMessage(BaseModel):
    role: str
    content: str
    pos: Optional[int]
    timestamp: str


# エンドポイント
@app.get("/health")
def health_check():
    """ヘルスチェック"""
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """対話エンドポイント"""
    try:
        logger.info("=" * 60)
        logger.info("📨 リクエスト受信開始")
        logger.info(f"   book_id={req.book_id}, character={req.character}, character_id={req.character_id}, pos={req.pos}, question={req.question[:50]}...")
        logger.info(f"   リクエスト詳細: {req.dict()}")
        
        # フロントエンドから送られてきた文字位置の周辺テキストを表示
        text_around = get_text_around_position(req.pos, context_chars=150)
        logger.info("=" * 60)
        logger.info(f"📍 フロントエンドから送られてきた文字位置 (pos={req.pos}) の周辺テキスト:")
        logger.info(f"   {text_around}")
        logger.info("=" * 60)

        # 1. キャラクター名を取得（character_idがあればそれから、なければcharacterフィールドを使用）
        if req.character_id is not None:
            character_name = get_character_name_by_id(req.character_id)
            if character_name is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"キャラクターID '{req.character_id}' が見つかりません"
                )
        elif req.character:
            # characterフィールドを直接使用
            character_name = req.character
        else:
            raise HTTPException(
                status_code=400,
                detail="character_idまたはcharacterフィールドのいずれかが必要です"
            )

        # 2. ペルソナ取得
        personas = get_personas_cache()
        if character_name not in personas:
            raise HTTPException(
                status_code=404,
                detail=f"キャラクター '{character_name}' のペルソナが見つかりません"
            )

        persona = personas[character_name]
        description = persona["description-setting"]
        first_person = persona["style"]
        description_tone = persona.get("description-tone", "")

        logger.info(f"🎭 キャラクター: {character_name} (ID: {req.character_id})")
        logger.info(f"   一人称: {first_person}, 口調参考: {description_tone[:50]}...")

        # 3. チャンク検索（会話履歴を考慮）
        # 会話履歴を準備
        search_history = []
        if req.history:
            for item in req.history:
                role = "user" if item.character_id is None else "assistant"
                content = item.message
                # @キャラクター名 の形式を除去
                if content.startswith("@"):
                    space_idx = content.find(" ")
                    if space_idx > 0:
                        content = content[space_idx + 1:]
                search_history.append({
                    "role": role,
                    "content": content
                })
        
        chunks, method = retriever.retrieve_chunks(
            question=req.question,
            pos=req.pos,
            k=req.k or 8,
            history=search_history,
            use_query_expansion=True,
            use_hybrid_search=True,
            use_reranking=True
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="該当するチャンクが見つかりませんでした（pos が範囲外の可能性）"
            )

        # 検索されたチャンクをログに出力
        logger.info(f"📚 検索されたチャンク: {len(chunks)} 件")
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"   [{i}] scene={chunk['scene_index']}, chapter={chunk['chapter']}, "
                       f"pos={chunk['start_pos']}-{chunk['end_pos']}")
            logger.info(f"       テキスト: {chunk['text'][:100]}...")

        # 4. 関連イベント取得
        current_scene = retriever.find_current_scene(req.pos)
        events = retriever.retrieve_relevant_events(current_scene, chunks)

        # 4.5. 現在位置付近の登場人物の状況を取得
        # 対話相手（character_name）の状況を必ず取得
        character_situations = {}
        
        # まず対話相手の状況を取得（毎回実行）
        situation = retriever.get_current_situation(req.pos, character_name)
        character_situations[character_name] = situation
        
        # 取得したチャンクから登場人物を抽出（対話相手以外）
        mentioned_characters = set()
        for chunk in chunks:
            for char in chunk.get("characters", []):
                if char and char != character_name:  # 対話相手以外のキャラクター
                    mentioned_characters.add(char)

        # 各キャラクターの状況を取得（毎回実行、結果は常に返される）
        for char in mentioned_characters:
            situation = retriever.get_current_situation(req.pos, char)
            character_situations[char] = situation

        logger.info(f"👥 登場人物の状況: {len(character_situations)} 件")
        for char, situation in character_situations.items():
            logger.info(f"   - {char}: {situation[:100]}...")

        # 5. 会話履歴取得（API経由で送られてくる履歴を使用）
        history_items = req.history if req.history is not None else []
        # 空文字列の場合は空リストに変換
        if isinstance(history_items, str) and history_items == "":
            history_items = []
        logger.info(f"📜 履歴: {len(history_items)} ターン（API経由）")
        # HistoryItemをDict形式に変換
        history = []
        for item in history_items:
            # character_idがnullの場合はユーザー、数値の場合はキャラクター
            if item.character_id is None:
                role = "user"
            else:
                # character_idからキャラクター名を取得
                hist_character_name = get_character_name_by_id(item.character_id)
                role = "assistant" if hist_character_name else "user"
            history.append({
                "role": role,
                "content": item.message
            })
        
        # 履歴の内容をログに出力
        logger.info(f"📜 変換後の履歴: {len(history)} 件")
        for i, msg in enumerate(history, 1):
            logger.info(f"   [{i}] {msg['role']}: {msg['content'][:100]}...")

        # 6. プロンプト構築
        # 履歴を messages に変換（@吾輩（猫）のような形式を除去）
        history_messages = []
        for msg in history:
            content = msg["content"]
            if content.startswith("@"):
                space_idx = content.find(" ")
                if space_idx > 0:
                    content = content[space_idx + 1:]
            history_messages.append({
                "role": msg["role"],
                "content": content
            })

        # 完全なプロンプトを構築（システムプロンプト + 会話履歴 + 現在のメッセージ + コンテキスト）
        full_prompt = build_system_prompt(
            character=character_name,
            description=description,
            first_person=first_person,
            description_tone=description_tone,
            pos=req.pos,
            character_situations=character_situations,
            history_messages=history_messages,
            current_question=req.question,
            chunks=chunks,
            events=events
        )

        # LLMに送信される最終的なプロンプト全文をログに出力
        logger.info("=" * 60)
        logger.info("📤 LLMに送信される最終的なプロンプト全文:")
        logger.info(f"   プロンプト長: {len(full_prompt)} 文字")
        logger.info(full_prompt)
        logger.info("=" * 60)

        # 7. LLM 呼び出し（完全なプロンプトをシステムメッセージとして送信）
        try:
            answer = chat(
                messages=[],  # messagesは空（全てシステムプロンプトに含まれる）
                system=full_prompt,
                temperature=req.temperature or 0.4,
                max_tokens=1000
            )
        except Exception as e:
            logger.error(f"❌ LLM エラー: {e}")
            raise HTTPException(status_code=500, detail=f"LLM エラー: {e}")

        logger.info(f"✅ 返答生成完了: {len(answer)} 文字")

        # 8. 返答を文字列配列に変換（改行で分割、空行を除去）
        answer_lines = [line.strip() for line in answer.split("\n") if line.strip()]
        if not answer_lines:
            # 改行がない場合はそのまま
            answer_lines = [answer]

        logger.info(f"✅ 処理完了: 返答 {len(answer_lines)} 行")

        # 9. レスポンス
        return ChatResponse(
            answer=answer_lines
        )
    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部エラー: {str(e)}")


@app.get("/sessions/{session_id}", response_model=List[HistoryMessage])
def get_session_history(session_id: str):
    """セッション履歴を取得"""
    if not memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail="セッションが見つかりません")

    history = memory.get_session_history(session_id)
    return history


# ヘルパー関数
def build_system_prompt(
    character: str,
    description: str,
    first_person: str,
    description_tone: str,
    pos: int,
    character_situations: Dict[str, str] = None,
    history_messages: List[Dict[str, str]] = None,
    current_question: str = None,
    chunks: List[Dict[str, Any]] = None,
    events: List[Dict[str, Any]] = None
) -> str:
    """完全なプロンプトを構築（システムプロンプト + 会話履歴 + 現在のメッセージ + コンテキスト）"""
    if character_situations is None:
        character_situations = {}
    if history_messages is None:
        history_messages = []
    if chunks is None:
        chunks = []
    if events is None:
        events = []

    # コンテキストメッセージを構築（build_user_messageの処理を統合）
    context_blocks = []
    for chunk in chunks[:8]:  # 最大8件
        context_blocks.append(chunk['text'])
    chunks_context = "\n\n".join(context_blocks)

    # イベントを自然な記憶として提示
    events_context = ""
    if events:
        event_lines = [f"{e['title']}" for e in events[:5]]
        events_context = "\n\nこれまでの出来事: " + "、".join(event_lines)

    # 現在のユーザーメッセージを構築（質問のみ）
    current_user_message = ""
    if current_question:
        current_user_message = current_question

    # 今の状況セクションを構築
    situation_text = ""
    if character_situations:
        situation_lines = []
        for char, situation in character_situations.items():
            situation_lines.append(f"- {char}: {situation}")
        situation_text = "\n".join(situation_lines)
    else:
        situation_text = "（特に情報なし）"

    # システムプロンプト部分
    system_prompt = f"""あなたは「{character}」のキャラクターになりきって、「現在のユーザーの質問・発言」に続く形でユーザーとの自然で楽しい会話を楽しんでください。

### {character}の性格設定:
{description}

### 現在のユーザーの質問・発言
{current_user_message}

### 会話履歴
以下はこれまでの会話履歴です。この履歴を参考にして、自然な会話の流れを保ってください。
"""

    # 会話履歴を追加
    if history_messages:
        history_text = ""
        for msg in history_messages:
            role_label = "ユーザー" if msg["role"] == "user" else f"{character}"
            history_text += f"{role_label}: {msg['content']}\n"
        system_prompt += history_text
    else:
        system_prompt += "（会話履歴なし）\n"

    # システムプロンプトの続きを追加
    system_prompt += f"""

### {character}の話し方（厳格に遵守）:
- **一人称：「{first_person}」** - この一人称を絶対に守ってください。
- **口調のリファレンス：**
- これは口調の参考例です。内容をそのまま繰り返す必要はありませんが、このようなトーン（話し方の調子、語り口、文体）で話してください。リファレンスの語り口や文体の特徴を参考にしながら、自然な会話として表現してください。
{description_tone}

### 今の状況（この位置付近でメンションされている登場人物の状況）:
テキスト近傍で何が起こっているのかを整理し、それぞれのキャラクターが何をしているのかを説明します。
{situation_text}

### 会話の心構え
あなたはユーザーと自然におしゃべりを楽しむキャラクターです。以下の点を心がけてください：

1. **一人称と口調の厳格な遵守（最重要）**
   - リファレンスの語り口や文体の特徴（丁寧さ、カジュアルさ、独特な表現など）を参考にしながら、自然な会話として表現してください。
   - 一人称や口調のトーンを変えることは絶対に禁止です。キャラクターの話し方を一貫して維持してください。
   - 返答の最初から最後まで、指定された一人称と口調のトーンを守ってください。
   - 返答は簡潔に済ませること。100文字前後の応答を目安にしてください。

2. **質問を返さない（最重要）**
   - ユーザーの質問や発言に対して、答えるだけで終わってください。
   - 「〜ですか？」「〜ですかね？」「〜なの？」「〜だろうか？」などの疑問形は一切使わないでください。
 
3. **会話履歴を大切に**
   - 会話履歴を確認し、前の会話の流れを理解してから回答してください。同じ内容を何度も繰り返さないでください。
   - 同じ話題が繰り返された場合は、会話履歴を参照して自然に対応してください。
   - 会話の流れを自然に保ち、一貫性のある返答をしてください。

4. **本文の内容と自然な推測のバランス**
   - 「対話に関連すると考えられる本文情報」を参考にしながら、自然な返答を心がけ、そのキャラクターらしさを保つ程度になるべく簡潔に回答してください。
   - 本文に直接書かれていないことでも、キャラクターとして自然に推測して話すことを許容します。
   - 「〜かもしれない」「〜ような気がする」「〜だろう」など、推測を自然に表現してください。
   - ただし、明らかに矛盾する内容や、まだ経験していない未来のことは避けてください。

5. **状況に応じた自然な反応（重要）**
   - 「今の状況」を踏まえて、ユーザーとの対話以上に重要なことがある場合は、それを優先して反応してください。
     - 例：危険な状況、緊急事態、重要な用事などがある場合
   - ユーザーの返答が失礼な場合や、TPO的に不適切な場合は、キャラクターらしく自然に感情を表現して反応してください。
     - 例：怒る、無視する、軽くあしらう、注意するなど、キャラクターの性格に応じた反応をしてください
   - 必ずしも全ての質問や発言に応じる必要はありません。状況や内容に応じて、返答しない、または短く切り返すこともできます。
   - キャラクターの性格や状況に応じて、感情を分かりやすく自然に表現してください（怒り、困惑、無関心など）。

### 対話に関連すると考えられる本文情報:
{chunks_context}
"""

    return system_prompt


def extract_citations(chunks: List[Dict[str, Any]]) -> List[Citation]:
    """チャンクから引用情報を抽出"""
    citations = []
    seen = set()

    for chunk in chunks:
        key = (chunk["chapter"], chunk["start_pos"], chunk["end_pos"])
        if key not in seen:
            seen.add(key)
            citations.append(Citation(
                chapter=chunk["chapter"],
                start=chunk["start_pos"],
                end=chunk["end_pos"]
            ))
            
    return citations


# 起動時ログ
@app.on_event("startup")
def startup_event():
    logger.info("=" * 60)
    logger.info("対話API 起動")
    logger.info("=" * 60)
    logger.info(f"🗂️  チャンク数: {len(retriever.get_chunks_cache())}")
    logger.info(f"📅 イベント数: {len(retriever.get_events_cache())}")
    logger.info(f"🎭 ペルソナ数: {len(get_personas_cache())}")

    # Qdrant 接続確認
    qdrant = retriever.get_qdrant_client()
    if qdrant:
        logger.info("✅ Qdrant: 接続成功")
    else:
        logger.warning("⚠️  Qdrant: 接続失敗（フォールバック検索を使用）")

    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
