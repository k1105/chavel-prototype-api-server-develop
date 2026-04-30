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


def get_data_dir(lang: str = "ja") -> Path:
    """言語別のデータディレクトリを返す"""
    return DATA_DIR / lang


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


def load_chunks(lang: str = "ja") -> List[Dict[str, Any]]:
    """chunks.jsonl を読み込み"""
    path = get_data_dir(lang) / "chunks.jsonl"
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    chunks.sort(key=lambda x: x["scene_index"])
    return chunks


def load_events(lang: str = "ja") -> List[Dict[str, Any]]:
    """events.jsonl を読み込み"""
    path = get_data_dir(lang) / "events.jsonl"
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def load_personas(lang: str = "ja") -> Dict[str, Dict[str, Any]]:
    """character.json を読み込み、name でインデックス"""
    path = get_data_dir(lang) / "character.json"
    personas = {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for persona in data:
            personas[persona["name"]] = persona
    return personas


def get_persona(character: str, lang: str = "ja") -> Optional[Dict[str, Any]]:
    """キャラクターのペルソナを取得"""
    personas = load_personas(lang)
    return personas.get(character)


# キャッシュ（言語別 dict）
_chunks_cache: Dict[str, List[Dict]] = {}
_events_cache: Dict[str, List[Dict]] = {}
_personas_cache: Dict[str, Dict] = {}
_main_text_cache: Dict[str, str] = {}


def get_chunks_cache(lang: str = "ja") -> List[Dict[str, Any]]:
    """キャッシュされたチャンクを取得"""
    if lang not in _chunks_cache:
        _chunks_cache[lang] = load_chunks(lang)
    return _chunks_cache[lang]


def get_events_cache(lang: str = "ja") -> List[Dict[str, Any]]:
    """キャッシュされたイベントを取得"""
    if lang not in _events_cache:
        _events_cache[lang] = load_events(lang)
    return _events_cache[lang]


def get_personas_cache(lang: str = "ja") -> Dict[str, Dict[str, Any]]:
    """キャッシュされたペルソナを取得"""
    if lang not in _personas_cache:
        _personas_cache[lang] = load_personas(lang)
    return _personas_cache[lang]


def get_persona_by_id(character_id: int, lang: str = "ja") -> Optional[Dict[str, Any]]:
    """character_idからペルソナを取得"""
    personas = get_personas_cache(lang)
    for persona in personas.values():
        if persona.get("id") == character_id:
            return persona
    return None


def get_character_name_by_id(character_id: int, lang: str = "ja") -> Optional[str]:
    """character_idからキャラクター名を取得"""
    persona = get_persona_by_id(character_id, lang)
    return persona.get("name") if persona else None


def get_main_text(lang: str = "ja") -> str:
    """main.txtのテキストを取得（キャッシュ付き）"""
    if lang not in _main_text_cache:
        path = get_data_dir(lang) / "main.txt"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _main_text_cache[lang] = f.read()
        else:
            _main_text_cache[lang] = ""
    return _main_text_cache[lang]


def get_text_around_position(pos: int, context_chars: int = 100, lang: str = "ja") -> str:
    """
    指定された位置の前後のテキストを取得

    Args:
        pos: 文字位置
        context_chars: 前後に取得する文字数
        lang: 言語コード

    Returns:
        位置付近のテキスト（位置マーカー付き）
    """
    text = get_main_text(lang)
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
