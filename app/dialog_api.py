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

from app import retriever
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

        logger.info("=" * 60)
        logger.info(f"🎭 現在の対話相手: {character_name} (ID: {req.character_id})")
        logger.info(f"   一人称: {first_person}")
        logger.info(f"   口調参考: {description_tone[:50]}...")
        logger.info("=" * 60)

        # 同じ位置・同じキャラクターでの会話回数をカウント
        # 会話履歴を逆順に見て、現在のキャラクターの返答（ターン）をカウント
        same_position_count = 0
        if req.history:
            logger.info(f"📊 会話履歴の解析開始 (履歴アイテム数: {len(req.history)})")
            for i, item in enumerate(reversed(req.history)):
                logger.info(f"   履歴[{i}]: character_id={item.character_id}, message={item.message[:30]}...")

                # キャラクターの返答（character_idが有効な値）の場合のみ処理
                if item.character_id is not None and item.character_id > 0:
                    if item.character_id == req.character_id:
                        # 現在のキャラクターの返答をカウント
                        same_position_count += 1
                        logger.info(f"      → カウント+1 (現在: {same_position_count})")
                    else:
                        # 異なるキャラクターの返答が出てきたら終了
                        logger.info(f"      → 異なるキャラクター (ID={item.character_id}) なので終了")
                        break
                else:
                    # character_id=None or 0 (ユーザーメッセージ) の場合はスキップ
                    logger.info(f"      → ユーザーメッセージ (character_id={item.character_id}) なのでスキップ")

        # 今回の質問も含めるため+1
        same_position_count += 1

        logger.info(f"📊 同じ位置・同じキャラクターでの会話回数: {same_position_count} 回")
        logger.info("=" * 60)

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
            character_name=character_name,  # キャラクター名を渡す
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
        # HistoryItemをDict形式に変換（キャラクター名を保持）
        history = []
        for item in history_items:
            # character_idがnullの場合はユーザー、数値の場合はキャラクター
            if item.character_id is None:
                role = "user"
                char_name = None
            else:
                # character_idからキャラクター名を取得
                char_name = get_character_name_by_id(item.character_id)
                role = "assistant" if char_name else "user"

            history.append({
                "role": role,
                "content": item.message,
                "character_name": char_name  # キャラクター名を追加
            })

        # 履歴の内容をログに出力
        logger.info(f"📜 変換後の履歴: {len(history)} 件")
        for i, msg in enumerate(history, 1):
            char_label = f" ({msg['character_name']})" if msg.get('character_name') else ""
            logger.info(f"   [{i}] {msg['role']}{char_label}: {msg['content'][:100]}...")

        # 6. プロンプト構築
        # 履歴を messages に変換（キャラクター名を保持）
        history_messages = []
        for msg in history:
            content = msg["content"]
            # @キャラクター名 の形式を除去
            if content.startswith("@"):
                space_idx = content.find(" ")
                if space_idx > 0:
                    content = content[space_idx + 1:]

            history_messages.append({
                "role": msg["role"],
                "content": content,
                "character_name": msg.get("character_name")  # キャラクター名を保持
            })

        # システムプロンプトを構築（性格設定 + ルール + 参考情報のみ）
        system_prompt = build_system_prompt(
            character=character_name,
            description=description,
            first_person=first_person,
            description_tone=description_tone,
            character_situations=character_situations,
            chunks=chunks,
            same_position_count=same_position_count  # 会話回数を渡す
        )

        # 会話履歴をmessages配列に変換（キャラクター名を明示）
        messages = []
        for msg in history_messages:
            content = msg["content"]
            char_name = msg.get("character_name")

            # assistantの場合はキャラクター名をcontentに含める
            if msg["role"] == "assistant" and char_name:
                content = f"[{char_name}]: {content}"

            messages.append({
                "role": msg["role"],
                "content": content
            })

        # 現在のユーザー発言を追加
        messages.append({
            "role": "user",
            "content": req.question
        })

        # LLMに送信される最終的なプロンプト全文をログに出力
        logger.info("=" * 60)
        logger.info("📤 LLMに送信される情報:")
        logger.info(f"   システムプロンプト長: {len(system_prompt)} 文字")
        logger.info(f"   会話履歴: {len(history_messages)} ターン")
        logger.info("   --- システムプロンプト ---")
        logger.info(system_prompt)
        logger.info("   --- 会話履歴 ---")
        for msg in messages:
            logger.info(f"   {msg['role']}: {msg['content']}")
        logger.info("=" * 60)

        # 7. LLM 呼び出し（systemとmessagesを正しく分離）
        # Structured Outputsのスキーマを定義（Hidden Inner Monologueパターン）
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "character_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": f"{character_name}としての内面の思考。感情、動機、戦略、意図などを自由に記述する。この思考はユーザーには見えないが、返答を考えるための重要な推論プロセス。"
                        },
                        "response": {
                            "type": "string",
                            "description": f"{character_name}本人としての一人称の返答文。thoughtで考えた内容に基づいて、実際にユーザーに向けて発話する内容。他のキャラクターについて説明するのではなく、{character_name}自身の経験や考えを語る。キャラクター名のプレフィックスは付けない。"
                        }
                    },
                    "required": ["thought", "response"],
                    "additionalProperties": False
                }
            }
        }

        try:
            answer_json = chat(
                messages=messages,  # 会話履歴 + 現在の質問
                system=system_prompt,  # 性格設定とルール
                temperature=req.temperature or 0.4,
                max_tokens=1000,
                response_format=response_format  # Structured Outputs
            )

            # JSON をパース
            import json
            answer_data = json.loads(answer_json)
            thought = answer_data.get("thought", "")
            answer = answer_data.get("response", "")

            # 内面の思考をログに出力（デバッグ用）
            logger.info(f"💭 内面の思考: {thought}")
        except Exception as e:
            logger.error(f"❌ LLM エラー: {e}")
            raise HTTPException(status_code=500, detail=f"LLM エラー: {e}")

        logger.info(f"✅ 返答生成完了: {len(answer)} 文字")
        logger.info(f"   返答内容: {answer[:100]}...")

        # キャラクター名プレフィックス（[キャラクター名]:）を除去（念のため）
        import re
        original_answer = answer
        answer = re.sub(r'^\[.+?\]:\s*', '', answer.strip())
        if original_answer != answer:
            logger.info(f"⚠️  プレフィックスを除去: '{original_answer[:50]}...' → '{answer[:50]}...'")

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




# ヘルパー関数
def build_system_prompt(
    character: str,
    description: str,
    first_person: str,
    description_tone: str,
    character_situations: Dict[str, str] = None,
    chunks: List[Dict[str, Any]] = None,
    same_position_count: int = 1
) -> str:
    """システムプロンプトを構築（性格設定 + ルール + 参考情報のみ）"""
    if character_situations is None:
        character_situations = {}
    if chunks is None:
        chunks = []

    # コンテキストメッセージを構築
    context_blocks = []
    for chunk in chunks[:3]:  # 最大3件（会話履歴重視のため削減）
        context_blocks.append(chunk['text'])
    chunks_context = "\n\n".join(context_blocks)

    # 今の状況セクションを構築
    situation_text = ""
    if character_situations:
        situation_lines = []
        for char, situation in character_situations.items():
            situation_lines.append(f"- {char}: {situation}")
        situation_text = "\n".join(situation_lines)
    else:
        situation_text = "（特に情報なし）"

    # 読み進め促進メッセージの構築（同じ位置での会話回数に応じて）
    reading_encouragement = ""
    if same_position_count >= 3:
        if same_position_count == 3:
            reading_encouragement = """
**特別な状況:**
ユーザーは同じ位置で3回目の会話をしています。そろそろ先を読み進めるよう、優しく促してください。
- 自然な会話の流れで「続きを読んでみたらどうか」と提案する
- キャラクターの性格に合った言い方で促す
- 強制的ではなく、さりげなく"""
        elif same_position_count == 4:
            reading_encouragement = """
**特別な状況:**
ユーザーは同じ位置で4回目の会話をしています。もう少し強めに先を読み進めるよう促してください。
- 「話はこれくらいにして、先を読んでほしい」という趣旨を伝える
- キャラクターの性格に応じて、少し呆れた様子や困った様子を見せる
- それでも威圧的にならず、キャラクターらしく"""
        else:  # 5回以上
            reading_encouragement = f"""
**特別な状況:**
ユーザーは同じ位置で{same_position_count}回目の会話をしています。明確に苛立ちを表現し、先を読むよう強く促してください。
- 「いい加減にして先を読め」という趣旨を強めに伝える
- キャラクターの性格に応じた苛立ち方をする（怒る、無視する、冷たくあしらうなど）
- 長い返答は避け、短く切り上げる
- 「もう答えぬ」「先を読め」など、端的に"""

    # システムプロンプト部分
    system_prompt = f"""# あなたの役割

あなたは「{character}」です。

**重要な指示:**

1. **Hidden Inner Monologue（内面の思考）:**
   - まず `thought` フィールドで、{character}としての内面の思考を自由に記述してください
   - 思考には以下を含めてください:
     * ユーザーの発言に対する感情（興味、退屈、苛立ち、共感など）
     * 返答の動機（なぜそう答えるのか）
     * 会話の戦略（どう答えるべきか、話題を変えるべきか、など）
   - この思考はユーザーには見えませんが、より適切な返答を生成するために重要です

2. **あなたの視点で話す:**
   - 一人称（{first_person}）で話してください
   - 「{character}本人」として経験や考えを語ってください
   - 他のキャラクターの代わりに話してはいけません

3. **会話履歴の活用:**
   - 会話履歴には他のキャラクターの発言も含まれています
   - ユーザーが「〇〇はこう言っていたけど、あなたはどう思う？」と聞いた場合:
     → 会話履歴を参照し、そのキャラクターの発言を踏まえて、あなた（{character}）の意見を述べてください
   - 単に「〇〇はどう？」と聞かれた場合:
     → あなた（{character}）の視点で〇〇について説明してください

4. **返答の形式:**
   - 常に{character}として話す
   - 他のキャラクターになりすまさない
   - ただし、他のキャラクターの発言を引用したり、それに対する意見を述べることは可能です
{reading_encouragement}

### {character}の性格設定:
{description}

### {character}の話し方（厳格に遵守）:
- **一人称：「{first_person}」** - この一人称を絶対に守ってください。
- **口調のリファレンス：**
- これは口調の参考例です。内容をそのまま繰り返す必要はありませんが、このようなトーン（話し方の調子、語り口、文体）で話してください。リファレンスの語り口や文体の特徴を参考にしながら、自然な会話として表現してください。
{description_tone}

### 今の状況:
{situation_text}

### 参考情報（本文の一部）:
{chunks_context}

### 会話のルール（厳守）:
1. **一人称と口調**: 一人称「{first_person}」を使い、リファレンスの文体で話す

2. **簡潔だが内容のある返答**:
   - 長々と説明しない（50-100文字程度）
   - 聞かれたことには簡潔に答える
   - 余計な質問は返さない

3. **会話履歴への対応**:
   - 過去の発言を思い出して答える
   - 「覚えてる？」と聞かれたら、実際に要約して答える
   - 同じ話題なら「さっき言った」と指摘する

4. **返答の例**:
   良い例:
   - ユーザー「まあそうだね」→ {character}「そうか」
   - ユーザー「最近どう？」→ {character}「私は変わらず...」（一人称で自分のことを話す）
   - ユーザー「〇〇は△△と言っていたけど、あなたはどう思う？」→ {character}「〇〇はそう言っていたのか。私としては...」（他キャラの発言を踏まえて自分の意見を述べる）

   悪い例:
   - ユーザー「最近どう？」→ {character}「主人は...」❌（自分ではなく他人について語っている）
   - ユーザー「〇〇は△△と言っていたけど、どう思う？」→ {character}「〇〇の言う通りである」❌（自分の意見を言わず、他キャラに同調するだけ）
   - 何でも「面倒だ」「知らぬ」で済ませる❌

5. **会話の終わらせ方**:
   - 「〜である」「〜だ」で断定的に終わる
   - 質問形（「〜かい？」）は控えめに
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
