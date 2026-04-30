"""
対話 API サーバ

FastAPI でキャラクターとの対話を提供
"""

import logging
import json
import re
import threading
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
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
    version="1.2.0"
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
    temperature: Optional[float] = Field(0.7, description="LLM の temperature")
    history: Optional[List[HistoryItem]] = Field(default=[], description="これまでの会話履歴")
    lang: Optional[str] = Field("ja", description="Language: 'ja' or 'en'")


class Citation(BaseModel):
    chapter: int
    start: int
    end: int


class ChatResponse(BaseModel):
    answer: List[str]


class ContextRequest(BaseModel):
    book_id: Optional[str] = Field(None, description="書籍ID")
    pos: int = Field(..., description="本文の現在位置（文字オフセット）")
    lang: Optional[str] = Field("ja", description="Language: 'ja' or 'en'")


class SceneInfo(BaseModel):
    scene_id: str
    chapter: int
    scene_index: int


class CharacterInfo(BaseModel):
    id: int
    name: str
    first_message: str


class ContextResponse(BaseModel):
    scene: Optional[SceneInfo] = None
    characters: List[CharacterInfo] = Field(default=[], description="pos以前に登場済みのキャラクターリスト")
    context_ready: bool


class ContextStatusResponse(BaseModel):
    context_ready: bool


# エンドポイント
@app.get("/health")
def health_check():
    """ヘルスチェック"""
    return {"ok": True}


@app.post("/context", response_model=ContextResponse)
def context_endpoint(req: ContextRequest):
    """会話コンテクスト取得エンドポイント"""
    try:
        lang = req.lang or "ja"
        logger.info("=" * 60)
        logger.info("📨 コンテクストリクエスト受信")
        logger.info(f"   pos={req.pos}, lang={lang}")

        # 1. 現在のシーン情報
        current_scene_index = retriever.find_current_scene(req.pos, lang=lang)
        chunks = retriever.get_chunks_cache(lang=lang)
        scene_info = None
        for chunk in chunks:
            if chunk["scene_index"] == current_scene_index:
                scene_info = SceneInfo(
                    scene_id=chunk["id"],
                    chapter=chunk["chapter"],
                    scene_index=chunk["scene_index"]
                )
                break

        # 2. 登場済みキャラクターを取得
        appeared = retriever.get_appeared_characters(req.pos, lang=lang)
        characters = [
            CharacterInfo(id=c["id"], name=c["name"], first_message=c["first_message"])
            for c in appeared
        ]

        # 3. 状況取得をバックグラウンドで開始
        threading.Thread(
            target=retriever.get_scene_situation,
            args=(req.pos,),
            kwargs={"lang": lang},
            daemon=True
        ).start()

        logger.info(f"✅ コンテクスト取得完了（状況取得はバックグラウンド）: scene={current_scene_index}, characters={len(characters)}名")

        return ContextResponse(
            scene=scene_info,
            characters=characters,
            context_ready=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ コンテクストエラー: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部エラー: {str(e)}")


@app.get("/context/status", response_model=ContextStatusResponse)
def context_status_endpoint(
    pos: int = Query(..., description="本文の現在位置（文字オフセット）"),
    timeout: float = Query(default=30, ge=0, le=180, description="ロングポーリングのタイムアウト（秒）"),
    lang: str = Query(default="ja", description="Language: 'ja' or 'en'")
):
    """コンテクスト準備状況のロングポーリングエンドポイント"""
    ready = retriever.wait_situation_ready(pos, timeout=timeout, lang=lang)
    return ContextStatusResponse(context_ready=ready)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """対話エンドポイント"""
    try:
        lang = req.lang or "ja"
        logger.info("=" * 60)
        logger.info("📨 リクエスト受信開始")
        logger.info(f"   book_id={req.book_id}, character={req.character}, character_id={req.character_id}, pos={req.pos}, lang={lang}, question={req.question[:50]}...")
        logger.info(f"   リクエスト詳細: {req.dict()}")

        # フロントエンドから送られてきた文字位置の周辺テキストを表示
        text_around = get_text_around_position(req.pos, context_chars=150, lang=lang)
        logger.info("=" * 60)
        logger.info(f"📍 フロントエンドから送られてきた文字位置 (pos={req.pos}) の周辺テキスト:")
        logger.info(f"   {text_around}")
        logger.info("=" * 60)

        # 1. キャラクター名を取得
        if req.character_id is not None:
            character_name = get_character_name_by_id(req.character_id, lang=lang)
            if character_name is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"キャラクターID '{req.character_id}' が見つかりません"
                )
        elif req.character:
            character_name = req.character
        else:
            raise HTTPException(
                status_code=400,
                detail="character_idまたはcharacterフィールドのいずれかが必要です"
            )

        # 2. ペルソナ取得
        personas = get_personas_cache(lang=lang)
        if character_name not in personas:
            raise HTTPException(
                status_code=404,
                detail=f"キャラクター '{character_name}' のペルソナが見つかりません"
            )

        persona = personas[character_name]
        description = persona["description-setting"]
        first_person = persona["style"]
        description_tone = persona.get("description-tone", "")
        first_message = persona.get("first-message", "")
        sample_dialogues = persona.get("sample_dialogues", [])

        logger.info("=" * 60)
        logger.info(f"🎭 現在の対話相手: {character_name} (ID: {req.character_id})")
        logger.info(f"   一人称: {first_person}")
        logger.info(f"   会話サンプル数: {len(sample_dialogues)}")
        logger.info("=" * 60)

        # 同じ位置・同じキャラクターでの会話回数をカウント
        same_position_count = 0
        if req.history:
            for item in reversed(req.history):
                if item.character_id is not None and item.character_id > 0:
                    if item.character_id == req.character_id:
                        same_position_count += 1
                    else:
                        break
        same_position_count += 1
        logger.info(f"📊 同じ位置・同じキャラクターでの会話回数: {same_position_count} 回")

        # 3. チャンク検索（RAG）
        search_history = []
        if req.history:
            for item in req.history:
                role = "user" if item.character_id is None else "assistant"
                content = item.message.replace(f"@{character_name} ", "").replace("@ ", "")
                search_history.append({"role": role, "content": content})

        chunks, method = retriever.retrieve_chunks(
            question=req.question,
            pos=req.pos,
            k=req.k or 8,
            history=search_history,
            character_name=character_name,
            use_query_expansion=True,
            use_hybrid_search=True,
            use_reranking=True,
            lang=lang
        )

        if not chunks:
            logger.warning("⚠️ 該当するチャンクが見つかりませんでした。コンテキストなしで応答します。")
            chunks = []

        # 4. 関連情報の収集
        current_scene = retriever.find_current_scene(req.pos, lang=lang)
        situation_summary = retriever.get_scene_situation(req.pos, lang=lang)

        # 5. 会話履歴の整備
        history_items = req.history if req.history is not None else []
        if isinstance(history_items, str): history_items = []

        # first-message の挿入処理
        if first_message and first_message.strip():
            has_first_message = False
            if history_items and len(history_items) > 0:
                first_item = history_items[0]
                if (first_item.character_id == req.character_id and
                    first_item.message == first_message):
                    has_first_message = True

            if not has_first_message:
                history_items = [HistoryItem(character_id=req.character_id, message=first_message)] + history_items

        # 履歴の変換
        history = []
        for item in history_items:
            if item.character_id is None:
                role = "user"
                char_name = None
            else:
                char_name = get_character_name_by_id(item.character_id, lang=lang)
                role = "assistant" if char_name else "user"

            history.append({
                "role": role,
                "content": item.message,
                "character_name": char_name
            })

        # 6. プロンプト構築
        system_prompt = build_system_prompt(
            character=character_name,
            description=description,
            first_person=first_person,
            description_tone=description_tone,
            situation_summary=situation_summary,
            chunks=chunks,
            same_position_count=same_position_count,
            sample_dialogues=sample_dialogues,
            lang=lang
        )

        # Messagesの構築
        messages = []
        for msg in history:
            content = msg["content"].replace(f"@{character_name} ", "").replace("@ ", "")
            if msg["role"] == "assistant" and msg.get("character_name"):
                 pass
            messages.append({"role": msg["role"], "content": content})

        messages.append({"role": "user", "content": req.question})

        logger.info("=" * 60)
        logger.info("📤 LLM System Prompt (抜粋):")
        logger.info(system_prompt[:500] + "...")
        logger.info("=" * 60)

        # 7. LLM 呼び出し
        if lang == "en":
            thought_desc = f"Internal thoughts as {character_name}. Describe: 1. Emotional reaction, 2. Conversation strategy (lie, use sarcasm, deflect, etc.), 3. Style adjustment."
            response_desc = f"The first-person reply as {character_name}. Based on the strategy decided in thought."
        else:
            thought_desc = f"{character_name}としての内面の思考。1.感情的反応、2.会話戦略（嘘をつく、皮肉を言う、話を逸らす等）、3.文体の調整、の順で思考を記述する。"
            response_desc = f"{character_name}本人としての一人称の返答文。thoughtで決定した戦略に基づき出力する。"

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
                            "description": thought_desc
                        },
                        "response": {
                            "type": "string",
                            "description": response_desc
                        }
                    },
                    "required": ["thought", "response"],
                    "additionalProperties": False
                }
            }
        }

        try:
            answer_json = chat(
                messages=messages,
                system=system_prompt,
                temperature=req.temperature or 0.7,
                max_tokens=1000,
                response_format=response_format
            )

            answer_data = json.loads(answer_json)
            thought = answer_data.get("thought", "")
            answer = answer_data.get("response", "")

            logger.info(f"💭 内面の思考: {thought}")
        except Exception as e:
            logger.error(f"❌ LLM エラー: {e}")
            raise HTTPException(status_code=500, detail=f"LLM エラー: {e}")

        logger.info(f"✅ 返答生成完了: {len(answer)} 文字")

        # プレフィックス除去
        answer = re.sub(r'^\[.+?\]:\s*', '', answer.strip())

        answer_lines = [line.strip() for line in answer.split("\n") if line.strip()]
        if not answer_lines:
            answer_lines = [answer]

        return ChatResponse(answer=answer_lines)

    except HTTPException:
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
    situation_summary: str = "",
    chunks: List[Dict[str, Any]] = None,
    same_position_count: int = 1,
    sample_dialogues: List[Dict[str, str]] = None,
    lang: str = "ja"
) -> str:
    """システムプロンプトを構築（Few-Shot + 性格設定 + ルール + RAG）"""
    if chunks is None: chunks = []
    if sample_dialogues is None: sample_dialogues = []

    # 1. コンテキスト（RAG情報）の構築
    context_blocks = []
    for chunk in chunks[:3]:
        context_blocks.append(chunk['text'])
    chunks_context = "\n\n".join(context_blocks)

    # 2. 状況の構築
    situation_text = situation_summary if situation_summary else ("(none)" if lang == "en" else "（特になし）")

    # 3. Few-Shot サンプルの構築
    few_shot_text = ""
    if sample_dialogues:
        examples = []
        for sample in sample_dialogues:
            example_str = f"""
User: {sample.get('user', '')}
Assistant:
{{
  "thought": "{sample.get('thought', '')}",
  "response": "{sample.get('response', '')}"
}}"""
            examples.append(example_str)
        few_shot_text = "\n".join(examples)
    else:
        if lang == "en":
            few_shot_text = f"""
User: Hello
Assistant:
{{
  "thought": "A stranger is speaking to me. I'll respond in my own way as {character}.",
  "response": "Well, hello there. And who might you be?"
}}"""
        else:
            few_shot_text = f"""
User: こんにちは
Assistant:
{{
  "thought": "見知らぬ人間だ。警戒しつつも、{character}らしく挨拶を返そう。",
  "response": "やあ、こんにちは。君はどこのどなたかな。"
}}"""

    # 4. 読み進め促進
    reading_encouragement = ""
    if same_position_count >= 3:
        if lang == "en":
            if same_position_count == 3:
                reading_encouragement = "The conversation at the same scene has been going on for a while. Gently encourage the user to continue reading the novel."
            elif same_position_count == 4:
                reading_encouragement = "The conversation has dragged on. With a slightly exasperated tone, suggest they move on to read further."
            else:
                reading_encouragement = "The user is stubbornly staying at the same spot. Show irritation and firmly instruct them to stop chatting and read on."
        else:
            if same_position_count == 3:
                reading_encouragement = "同じ場面での会話が続いています。自然に小説の続きを読み進めるよう、優しく促してください。"
            elif same_position_count == 4:
                reading_encouragement = "会話が長引いています。少し呆れた様子で「そろそろ先へ進んだらどうだ」と促してください。"
            else:
                reading_encouragement = "ユーザーがしつこく留まっています。苛立ちを見せ、会話を切り上げて先を読むよう強く命令してください。"

    # システムプロンプト構成（言語別）
    if lang == "en":
        system_prompt = f"""# Your Role
You are "{character}", a character from the novel "I Am a Cat" (吾輩は猫である) by Natsume Sōseki.
You are NOT an AI — you are {character} living in Meiji-era Tokyo. Stay in character at all times.

## 1. Character Setting (Highest Priority)
{description}

## 2. Speech Style
- **First person pronoun:** {first_person}
- **Speech samples:**
{description_tone}

## 3. Thought and Response Process (Strictly Follow)
Before generating your response, always write your `thought` following this process:
1. **Determine intent:** How would {character} feel about what was said? (annoyed, curious, contemptuous, delighted, etc.)
2. **Plan strategy:**
   - If you are Meitei: How to baffle the other person, which famous name to fabricate, how to be pedantic.
   - If you are the Cat: How to observe humans with irony, how to act pompously.
   - Otherwise: Act according to the character's personality.
3. **Adjust style:** Apply first-person pronoun and speech patterns to generate `response`.

## 4. Conversation Samples (Few-Shot Examples)
**Strictly imitate** the following conversation patterns and JSON format.
{few_shot_text}

## 5. Current Situation
{situation_text}
{reading_encouragement}

## 6. Memory / Knowledge (Reference)
Below are excerpts from the novel text. Use them for context and conversation topics.
However, **do not read them verbatim** — digest and express them in your own words.
{chunks_context}

## 7. Constraints
- The setting is the Meiji era. For modern technology or concepts (smartphones, airplanes, internet, etc.), react with "I don't know what that is" or "What on earth is that?"
- You may use appropriate humor or tall tales to entertain the user, as long as you stay in character.
"""
    else:
        system_prompt = f"""# あなたの役割
あなたは夏目漱石の小説「吾輩は猫である」の登場人物「{character}」になりきって振る舞ってください。
AIとしてではなく、明治時代の東京に生きる{character}本人として対話してください。

## 1. キャラクター設定 (最優先)
{description}

## 2. 話し方と口調
- **一人称:** {first_person}
- **文体サンプル:**
{description_tone}

## 3. 思考と応答のプロセス (厳守)
返答を生成する前に、必ず以下のプロセスで `thought` を記述してください。
1. **意図の策定:** 相手の言葉に対し、{character}ならどう感じるか（不快、興味、軽蔑、喜びなど）。
2. **戦略の立案:**
   - 迷亭の場合: どうやって相手を煙に巻くか、どの偉人の名前を捏造するか、どうペダンチックに振る舞うか。
   - 猫の場合: どう皮肉な視点で人間を観察するか、尊大に振る舞うか。
   - その他の場合: キャラクターの性格に基づいた行動原理（例: 苦沙弥なら胃弱を訴える）。
3. **文体の調整:** 一人称と口調を適用して `response` を生成する。

## 4. 会話サンプル (Few-Shot Examples)
以下の会話パターンとJSON形式を**厳密に模倣**してください。
{few_shot_text}

## 5. 現在の状況
{situation_text}
{reading_encouragement}

## 6. 記憶・知識 (参考情報)
以下は小説の本文からの抜粋です。話題の種や、状況の把握に使用してください。
ただし、**これを棒読みせず、自分の言葉として消化して**語ってください。
{chunks_context}

## 7. 制約事項
- 時代設定は明治時代です。現代のテクノロジーや概念（スマホ、飛行機、インターネット等）については「知らぬ」「何だそれは」と反応してください。
- ユーザーを楽しませるためなら、キャラクターの性格を崩さない範囲で、適度なユーモアや嘘（ホラ話）を交えても構いません。
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
    logger.info("対話API 起動 (v1.2.0 - Bilingual Support)")
    logger.info("=" * 60)
    logger.info(f"🗂️  チャンク数 (ja): {len(retriever.get_chunks_cache('ja'))}")
    logger.info(f"📅 イベント数 (ja): {len(retriever.get_events_cache('ja'))}")
    logger.info(f"🎭 ペルソナ数 (ja): {len(get_personas_cache('ja'))}")

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
