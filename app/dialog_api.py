"""
対話 API サーバ

FastAPI でキャラクターとの対話を提供
"""

import logging
import json
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
    version="1.1.0"
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
    # 【変更】創造性を高めるため、デフォルトの temperature を 0.4 -> 0.7 に変更
    temperature: Optional[float] = Field(0.7, description="LLM の temperature")
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

        # 1. キャラクター名を取得
        if req.character_id is not None:
            character_name = get_character_name_by_id(req.character_id)
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
        first_message = persona.get("first-message", "")
        # 【追加】会話サンプルを取得
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
            use_reranking=True
        )

        if not chunks:
            # チャンクが見つからない場合でも会話は成立させるため、空リストで続行（あるいはエラー）
            logger.warning("⚠️ 該当するチャンクが見つかりませんでした。コンテキストなしで応答します。")
            chunks = []

        # 4. 関連情報の収集
        current_scene = retriever.find_current_scene(req.pos)
        
        # 登場人物の状況取得
        character_situations = {}
        situation = retriever.get_current_situation(req.pos, character_name)
        character_situations[character_name] = situation
        
        mentioned_characters = set()
        for chunk in chunks:
            for char in chunk.get("characters", []):
                if char and char != character_name:
                    mentioned_characters.add(char)

        for char in mentioned_characters:
            situation = retriever.get_current_situation(req.pos, char)
            character_situations[char] = situation

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
                char_name = get_character_name_by_id(item.character_id)
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
            character_situations=character_situations,
            chunks=chunks,
            same_position_count=same_position_count,
            sample_dialogues=sample_dialogues  # 【追加】サンプルを渡す
        )

        # Messagesの構築
        messages = []
        for msg in history:
            content = msg["content"].replace(f"@{character_name} ", "").replace("@ ", "")
            # assistantの場合は誰の発言か明記（マルチキャラ対応の布石）
            if msg["role"] == "assistant" and msg.get("character_name"):
                 # ここではLLMに「会話の流れ」として認識させるため、自分の発言には名前を付けず、
                 # 他のキャラの発言があれば付ける等の制御が可能だが、
                 # シンプルに history として渡す。
                 pass
            messages.append({"role": msg["role"], "content": content})

        messages.append({"role": "user", "content": req.question})

        logger.info("=" * 60)
        logger.info("📤 LLM System Prompt (抜粋):")
        logger.info(system_prompt[:500] + "...")
        logger.info("=" * 60)

        # 7. LLM 呼び出し
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
                            "description": f"{character_name}としての内面の思考。1.感情的反応、2.会話戦略（嘘をつく、皮肉を言う、話を逸らす等）、3.文体の調整、の順で思考を記述する。"
                        },
                        "response": {
                            "type": "string",
                            "description": f"{character_name}本人としての一人称の返答文。thoughtで決定した戦略に基づき出力する。"
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
                temperature=req.temperature or 0.7, # デフォルト高め
                max_tokens=1000,
                response_format=response_format
            )

            import json
            answer_data = json.loads(answer_json)
            thought = answer_data.get("thought", "")
            answer = answer_data.get("response", "")

            logger.info(f"💭 内面の思考: {thought}")
        except Exception as e:
            logger.error(f"❌ LLM エラー: {e}")
            raise HTTPException(status_code=500, detail=f"LLM エラー: {e}")

        logger.info(f"✅ 返答生成完了: {len(answer)} 文字")
        
        # プレフィックス除去
        import re
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
    character_situations: Dict[str, str] = None,
    chunks: List[Dict[str, Any]] = None,
    same_position_count: int = 1,
    sample_dialogues: List[Dict[str, str]] = None # 【追加】
) -> str:
    """システムプロンプトを構築（Few-Shot + 性格設定 + ルール + RAG）"""
    if character_situations is None: character_situations = {}
    if chunks is None: chunks = []
    if sample_dialogues is None: sample_dialogues = []

    # 1. コンテキスト（RAG情報）の構築
    # 少し量を絞って、LLMが混乱しないようにする
    context_blocks = []
    for chunk in chunks[:3]:
        context_blocks.append(chunk['text'])
    chunks_context = "\n\n".join(context_blocks)

    # 2. 状況の構築
    situation_text = ""
    if character_situations:
        situation_lines = [f"- {char}: {sit}" for char, sit in character_situations.items()]
        situation_text = "\n".join(situation_lines)
    else:
        situation_text = "（特になし）"

    # 3. Few-Shot サンプルの構築（重要）
    # JSON構造を提示し、模倣させる
    few_shot_text = ""
    if sample_dialogues:
        examples = []
        for sample in sample_dialogues:
            # 実際のJSON出力形式に近い形で提示
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
        # デフォルトのサンプル（万が一データがない場合）
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
        if same_position_count == 3:
            reading_encouragement = "同じ場面での会話が続いています。自然に小説の続きを読み進めるよう、優しく促してください。"
        elif same_position_count == 4:
            reading_encouragement = "会話が長引いています。少し呆れた様子で「そろそろ先へ進んだらどうだ」と促してください。"
        else:
            reading_encouragement = "ユーザーがしつこく留まっています。苛立ちを見せ、会話を切り上げて先を読むよう強く命令してください。"

    # システムプロンプト構成
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
    logger.info("対話API 起動 (v1.1.0 - Improved Persona Support)")
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