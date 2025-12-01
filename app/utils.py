"""
共通ユーティリティ

.env のロード、埋め込み生成、LLM 呼び出し、データ読み込み
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

# .env をプロジェクト直下からロード
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

# 設定
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# データパス
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EVENTS_FILE = DATA_DIR / "events.jsonl"
PERSONA_FILE = DATA_DIR / "character.jsonl"
MAIN_TXT_FILE = DATA_DIR / "main.txt"

# OpenAI クライアント
_client = None


def get_openai_client() -> OpenAI:
    """OpenAI クライアントをシングルトンで取得"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed(text: str) -> List[float]:
    """テキストを埋め込みベクトルに変換"""
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 1000,
    response_format: Optional[Dict] = None
) -> str:
    """LLM でチャット補完"""
    client = get_openai_client()

    # システムメッセージを先頭に追加
    if system:
        messages = [{"role": "system", "content": system}] + messages

    # API呼び出しのパラメータを構築
    params = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # response_formatが指定されている場合は追加
    if response_format:
        params["response_format"] = response_format

    response = client.chat.completions.create(**params)

    return response.choices[0].message.content


def load_chunks() -> List[Dict[str, Any]]:
    """chunks.jsonl を読み込み"""
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    chunks.sort(key=lambda x: x["scene_index"])
    return chunks


def load_events() -> List[Dict[str, Any]]:
    """events.jsonl を読み込み"""
    events = []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line.strip()))
    return events


def load_personas() -> Dict[str, Dict[str, Any]]:
    """character.jsonl を読み込み、name でインデックス"""
    personas = {}
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            persona = json.loads(line.strip())
            personas[persona["name"]] = persona
    return personas


def get_persona(character: str) -> Optional[Dict[str, Any]]:
    """キャラクターのペルソナを取得"""
    personas = load_personas()
    return personas.get(character)


# キャッシュ（起動時に読み込む）
_chunks_cache = None
_events_cache = None
_personas_cache = None
_main_text_cache = None


def get_chunks_cache() -> List[Dict[str, Any]]:
    """キャッシュされたチャンクを取得"""
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = load_chunks()
    return _chunks_cache


def get_events_cache() -> List[Dict[str, Any]]:
    """キャッシュされたイベントを取得"""
    global _events_cache
    if _events_cache is None:
        _events_cache = load_events()
    return _events_cache


def get_personas_cache() -> Dict[str, Dict[str, Any]]:
    """キャッシュされたペルソナを取得"""
    global _personas_cache
    if _personas_cache is None:
        _personas_cache = load_personas()
    return _personas_cache


def get_persona_by_id(character_id: int) -> Optional[Dict[str, Any]]:
    """character_idからペルソナを取得"""
    personas = get_personas_cache()
    for persona in personas.values():
        if persona.get("id") == character_id:
            return persona
    return None


def get_character_name_by_id(character_id: int) -> Optional[str]:
    """character_idからキャラクター名を取得"""
    persona = get_persona_by_id(character_id)
    return persona.get("name") if persona else None


def get_main_text() -> str:
    """main.txtのテキストを取得（キャッシュ付き）"""
    global _main_text_cache
    if _main_text_cache is None:
        if MAIN_TXT_FILE.exists():
            with open(MAIN_TXT_FILE, "r", encoding="utf-8") as f:
                _main_text_cache = f.read()
        else:
            _main_text_cache = ""
    return _main_text_cache


def get_text_around_position(pos: int, context_chars: int = 100) -> str:
    """
    指定された位置の前後のテキストを取得
    
    Args:
        pos: 文字位置
        context_chars: 前後に取得する文字数
    
    Returns:
        位置付近のテキスト（位置マーカー付き）
    """
    text = get_main_text()
    if not text:
        return f"[テキストファイルが見つかりません] pos={pos}"
    
    if pos < 0 or pos >= len(text):
        return f"[位置が範囲外] pos={pos}, text_length={len(text)}"
    
    start = max(0, pos - context_chars)
    end = min(len(text), pos + context_chars)
    
    before = text[start:pos]
    at_pos = text[pos] if pos < len(text) else ""
    after = text[pos + 1:end] if pos + 1 < len(text) else ""
    
    # 位置マーカーを追加
    result = f"{before}【{at_pos}】{after}"
    
    return result
