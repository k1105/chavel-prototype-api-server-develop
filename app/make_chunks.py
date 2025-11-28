# app/make_chunks.py
# -*- coding: utf-8 -*-
import os, re, json, argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable, Set
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
print("DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # LLM未使用でも動く

load_dotenv()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ========= チャンク長の目安（日本語） =========
TARGET_CHUNK = 1000  # 目標
MIN_CHUNK    = 600   # これ未満は次にくっつける
MAX_CHUNK    = 1400  # これを超えたら必ず分割

# ========= パス =========
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
MAIN_TXT   = DATA_DIR / "main.txt"
OUT_JSONL  = DATA_DIR / "chunks.jsonl"
NER_CACHE  = DATA_DIR / "ner_cache.json"  # LLM抽出のキャッシュ

# ========= 章見出し検出 =========
KANJI_NUM   = "〇一二三四五六七八九十百千"
CHAPTER_PAT = re.compile(rf"^\s*第\s*([0-9{KANJI_NUM}]+)\s*章\s*$")

# ========= 許可キャラクター（正規名） =========
ALLOWED_NAMES: List[str] = [
    "吾輩",
    "三毛子",
    "車屋の黒",
    "白",
    "珍野 苦沙弥",
    "迷亭",
    "水島 寒月",
    "越智 東風",
    "八木 独仙",
    "甘木先生",
    "金田",
    "金田 鼻子",
    "金田 富子",
    "鈴木 籐十郎",
    "多々良 三平",
    "牧山",
    "珍野夫人",
    "珍野 とん子",
    "珍野 すん子",
    "珍野 めん子",
    "御三",
    "雪江",
    "二絃琴の御師匠さん",
    "古井 武右衛門",
    "吉田 虎蔵",
    "泥棒陰士",
    "八っちゃん",
]

# ========= 表記ゆれ → 正規名 =========
ALIASES: Dict[str, str] = {
    "苦沙弥": "珍野 苦沙弥",
    "苦沙弥先生": "珍野 苦沙弥",
    "先生（苦沙弥）": "珍野 苦沙弥",
    "寒月": "水島 寒月",
    "越智": "越智 東風",
    "東風": "越智 東風",
    "独仙": "八木 独仙",
    "甘木": "甘木先生",
    "鼻子": "金田 鼻子",
    "富子": "金田 富子",
    "籐十郎": "鈴木 籐十郎",
    "藤十郎": "鈴木 籐十郎",
    "三平": "多々良 三平",
    "とん子": "珍野 とん子",
    "すん子": "珍野 すん子",
    "めん子": "珍野 めん子",
    "清": "御三",
    "おさん": "御三",
    "御師匠さん": "二絃琴の御師匠さん",
    "古井": "古井 武右衛門",
    "武右衛門": "古井 武右衛門",
    "吉田": "吉田 虎蔵",
    "虎蔵": "吉田 虎蔵",
    "やっちゃん": "八っちゃん",
}

# ========= データモデル =========
@dataclass
class ChunkRec:
    id: str
    chapter: int
    scene_index: int
    start_pos: int
    end_pos: int
    characters: List[str]
    text: str

# ========= ユーティリティ =========
def split_chapters(text: str) -> List[Dict]:
    """章見出しがあれば分割、無ければ全体を第1章に"""
    lines = text.splitlines()
    chapters = []
    cur_title = "第1章"
    cur_buf = []
    for ln in lines:
        if CHAPTER_PAT.match(ln):
            if cur_buf:
                chapters.append({"title": cur_title, "text": "\n".join(cur_buf)})
                cur_buf = []
            cur_title = ln.strip()
        else:
            cur_buf.append(ln)
    if cur_buf:
        chapters.append({"title": cur_title, "text": "\n".join(cur_buf)})

    out = []
    for i, ch in enumerate(chapters, 1):
        m = CHAPTER_PAT.match(ch["title"])
        num = i
        if m:
            try:
                num = int(m.group(1))
            except:
                num = i
        out.append({"chapter": num, "text": ch["text"]})
    return out or [{"chapter": 1, "text": text}]

def sentence_split(s: str) -> List[str]:
    """句点などで文に分割。無い場合は1文扱い。"""
    parts = re.split(r'(?<=[。！？…])', s)
    out = [p.strip() for p in parts if p and p.strip()]
    return out if out else [s.strip()]

def split_oversize(text: str, max_len: int = MAX_CHUNK) -> List[str]:
    """
    大きすぎる塊を“なるべく自然に”分割する。
    1) 「、」「，」「。」「空白」「改行」「・」などの区切りでmax_lenを超えない最後の位置で切る
    2) それでも難しければ強制スライス
    """
    res = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_len, n)
        window = text[i:end]
        if end < n:
            cut = max(
                window.rfind("、"),
                window.rfind("，"),
                window.rfind("。"),
                window.rfind(" "),
                window.rfind("\n"),
                window.rfind("・"),
            )
            if cut >= 0 and cut >= MIN_CHUNK // 2:
                end = i + cut + 1
        res.append(text[i:end])
        i = end
    return [r.strip() for r in res if r.strip()]

def pack_sentences_to_chunks(sentences: List[str]) -> List[str]:
    """
    文列を MAX_CHUNK を超えないように順次パック（非再帰）
    """
    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        if len(s) > MAX_CHUNK:
            flush()
            parts = split_oversize(s, MAX_CHUNK)
            chunks.extend(parts)
            continue

        if not buf:
            buf = s
            continue

        if len(buf) + 1 + len(s) <= MAX_CHUNK:
            buf = f"{buf} {s}"
        else:
            if len(buf) < MIN_CHUNK:
                flush()
                buf = s
            else:
                flush()
                buf = s

    flush()

    # 念のため最終チェック
    final = []
    for c in chunks:
        if len(c) > MAX_CHUNK:
            final.extend(split_oversize(c, MAX_CHUNK))
        else:
            final.append(c)
    return final

def load_cache() -> Dict[str, List[str]]:
    if NER_CACHE.exists():
        try:
            return json.loads(NER_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    NER_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

# ========= 許可名/エイリアス検出 =========
# 日本語の"ゆるい境界"：前が和文・英数の連続文字でないこと（後ろは助詞などがつくので制限しない）
_JP_BOUND = r"(?<![一-龥ぁ-んァ-ンA-Za-z0-9]){name}"

def _contains_name(text: str, name: str) -> bool:
    if name == "白":
        # 1文字は誤検出が多いので境界厳しめ（後方チェック維持）
        pat = re.compile(rf"(?<![一-龥ぁ-んァ-ンA-Za-z0-9])白(?![一-龥ぁ-んァ-ンA-Za-z0-9])")
        return bool(pat.search(text))
    pat = re.compile(_JP_BOUND.format(name=re.escape(name)))
    return bool(pat.search(text))

def detect_allowed_names(text: str) -> Set[str]:
    found: Set[str] = set()
    # 1) 正規名の直接ヒット
    for nm in ALLOWED_NAMES:
        if _contains_name(text, nm):
            found.add(nm)
    # 2) エイリアス → 正規化
    for alias, canon in ALIASES.items():
        if _contains_name(text, alias):
            found.add(canon)
    return found

# ========= LLM補助：許可リスト制約付き =========
def llm_characters(client: OpenAI, text: str, allowed_names: List[str]) -> List[str]:
    """
    allowed_names にあるキャラクター一覧の“中からのみ”、
    本文に登場する人物名を返すよう LLM に指示。
    """
    char_json = json.dumps(allowed_names, ensure_ascii=False)
    prompt = (
        "以下は登場人物リストです。この中で、次の本文に実際に登場している人物のみを選び、"
        "名前だけの配列JSONで返してください。説明は禁止。本文にいない人物は出力しないでください。\n\n"
        f"登場人物リスト: {char_json}\n\n本文:\n{text[:1200]}"
    )
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    content = r.choices[0].message.content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            # 念のため許可名でフィルタ
            return [n for n in arr if n in allowed_names]
    except Exception:
        pass
    return []

# ========= メイン処理 =========
def build_chunks(
    text: str,
    use_llm_ner: bool=False,
    llm_limit:int=999999,
    ner_min_heuristic:int=2
) -> List[ChunkRec]:
    client = OpenAI() if (use_llm_ner and OpenAI is not None) else None
    cache = load_cache()
    llm_calls = 0

    chapters = split_chapters(text)
    chunks: List[ChunkRec] = []
    scene_idx = 1
    global_pos = 0

    for ch in chapters:
        ch_text = ch["text"]
        sents = sentence_split(ch_text)
        blocks = pack_sentences_to_chunks(sents)

        for block in blocks:
            # start/end は「本文中の出現位置」を前方探索で近似
            start = text.find(block, global_pos)
            if start == -1:
                start = global_pos
            end = start + len(block)
            global_pos = end

            scene_id = f"scene_{scene_idx:05d}"

            # 1) まず許可名/エイリアスのみで検出
            names_from_rules = sorted(detect_allowed_names(block))

            # 2) キャッシュ or LLM補助（不足時のみ）
            if scene_id in cache:
                names = cache[scene_id]
            else:
                need_llm = (
                    client is not None
                    and llm_calls < llm_limit
                    and len(names_from_rules) < ner_min_heuristic
                )
                if need_llm:
                    llm_names = llm_characters(client, block, ALLOWED_NAMES)
                    llm_calls += 1
                    # ルール検出とLLM検出の和集合（どちらも正規名）
                    names = sorted(set(names_from_rules) | set(llm_names))
                else:
                    names = names_from_rules
                cache[scene_id] = names  # キャッシュ

            chunks.append(ChunkRec(
                id=scene_id,
                chapter=ch["chapter"],
                scene_index=scene_idx,
                start_pos=int(start),
                end_pos=int(end),
                characters=names[:8],  # 多すぎるとノイズなので上限
                text=block.strip()
            ))
            scene_idx += 1

    save_cache(cache)
    return chunks

def write_jsonl(chunks: Iterable[ChunkRec], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate chunks.jsonl from main.txt with character detection.")
    parser.add_argument("--llm-ner", action="store_true", help="LLMで人物名抽出を補助（最終的には許可リストでフィルタ）")
    parser.add_argument("--llm-ner-limit", type=int, default=999999, help="LLMに投げる最大シーン数")
    parser.add_argument("--ner-min-heuristic", type=int, default=2, help="ルール検出がこの人数未満ならLLM補助を使う")
    args = parser.parse_args()

    text = MAIN_TXT.read_text(encoding="utf-8")
    chunks = build_chunks(
        text,
        use_llm_ner=args.llm_ner,
        llm_limit=args.llm_ner_limit,
        ner_min_heuristic=args.ner_min_heuristic
    )
    write_jsonl(chunks, OUT_JSONL)
    print(f"✅ wrote {OUT_JSONL} ({len(chunks)} chunks)")

if __name__ == "__main__":
    main()
