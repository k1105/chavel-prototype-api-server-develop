#!/usr/bin/env python3
"""
本文翻訳スクリプト

data/ja/main.txt を章単位で読み込み、OpenAI API で英語に翻訳して
data/en/main.txt に出力する。

Usage:
    python app/translate_text.py --chapters 3
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# .env をロード
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# 章見出し検出
KANJI_NUM = "〇一二三四五六七八九十百千"
CHAPTER_PAT = re.compile(rf"^\s*第\s*([0-9{KANJI_NUM}]+)\s*章\s*$")

# レジュームファイル
PROGRESS_FILE = DATA_DIR / "en" / ".translate_progress.json"

# キャラクター名マッピング（翻訳プロンプトに含める）
CHARACTER_NAME_MAP = """
Character Name Mapping (use these romanized names consistently):
- 吾輩 → I (narrator cat, refer to self as "I"; when other characters refer to him, use "the cat")
- 珍野 苦沙弥 / 苦沙弥 / 苦沙弥先生 → Kushami (Mr. Kushami when addressed formally)
- 迷亭 → Meitei
- 水島 寒月 / 寒月 → Kangetsu
- 越智 東風 / 東風 → Tofu (note: read as "Kochi" in Japanese, but use "Tofu" for the English version)
- 八木 独仙 / 独仙 → Dokusen
- 甘木先生 → Dr. Amaki
- 金田 → Kaneda
- 金田 鼻子 → Mrs. Kaneda (Hanako)
- 金田 富子 → Tomiko Kaneda
- 鈴木 籐十郎 → Suzuki Tojuro
- 多々良 三平 → Tatara Sampei
- 牧山 → Makiyama
- 珍野夫人 → Mrs. Kushami
- 珍野 とん子 → Tonko
- 珍野 すん子 → Sunko
- 珍野 めん子 → Menko
- 御三 / おさん → Osan
- 三毛子 → Mikeko
- 車屋の黒 → Kuro (the rickshaw man's black cat)
- 白 → Shiro
- 雪江 → Yukie
- 二絃琴の御師匠さん → the koto teacher
- 古井 武右衛門 → Furui Buemon
- 吉田 虎蔵 → Yoshida Torazo
- 泥棒陰士 → the thief
- 八っちゃん → Hatchan
"""

TRANSLATION_SYSTEM_PROMPT = f"""You are a professional literary translator specializing in Japanese-to-English translation of classical Japanese literature.

You are translating "吾輩は猫である" (I Am a Cat) by Natsume Sōseki.

{CHARACTER_NAME_MAP}

Translation Guidelines:
1. Maintain the narrative voice: The narrator is a cat speaking in first person ("I").
2. Preserve the literary style: Sōseki's writing is characterized by wit, irony, and philosophical observations. Maintain these qualities.
3. Use the character name mappings above consistently throughout.
4. Preserve paragraph breaks and formatting from the original.
5. Translate idioms and cultural references naturally, adding brief context in parentheses only when absolutely necessary for comprehension.
6. Maintain the Meiji-era setting — do not modernize language.
7. For poetry, haiku, or literary quotations embedded in the text, provide a natural English rendering.
8. Keep chapter headings in the format "Chapter N" (e.g., "Chapter 1", "Chapter 2").
9. Do NOT add any translator notes, commentary, or explanations outside the translation itself.
10. Translate ALL text provided — do not skip or summarize any portion."""


def split_chapters(text: str) -> list:
    """テキストを章単位で分割"""
    lines = text.splitlines()
    chapters = []
    cur_num = 1
    cur_buf = []

    for ln in lines:
        m = CHAPTER_PAT.match(ln)
        if m:
            if cur_buf:
                chapters.append({"num": cur_num, "text": "\n".join(cur_buf)})
                cur_buf = []
            try:
                cur_num = int(m.group(1))
            except ValueError:
                cur_num = len(chapters) + 1
            # 章見出し自体は翻訳時に Chapter N として出力
        else:
            cur_buf.append(ln)

    if cur_buf:
        chapters.append({"num": cur_num, "text": "\n".join(cur_buf)})

    return chapters


def split_into_blocks(text: str, max_chars: int = 3000) -> list:
    """テキストを段落単位でブロックに分割（max_chars目安）"""
    paragraphs = text.split("\n")
    blocks = []
    current_block = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current_block:
            blocks.append("\n".join(current_block))
            current_block = []
            current_len = 0
        current_block.append(para)
        current_len += para_len + 1  # +1 for newline

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def load_progress() -> dict:
    """翻訳進捗を読み込み"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(progress: dict):
    """翻訳進捗を保存"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_block(client: OpenAI, block_text: str, chapter_num: int, block_idx: int) -> str:
    """1ブロックを翻訳"""
    user_message = (
        f"Translate the following text from Chapter {chapter_num} of '吾輩は猫である'.\n"
        f"IMPORTANT: Do NOT include any chapter headings (e.g. 'Chapter 1') in your output. "
        f"Just translate the text itself.\n\n{block_text}"
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=4000
    )

    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Translate main.txt from Japanese to English")
    parser.add_argument("--chapters", type=int, default=3, help="Number of chapters to translate (default: 3)")
    parser.add_argument("--block-size", type=int, default=3000, help="Max characters per translation block (default: 3000)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress")
    args = parser.parse_args()

    # 入出力パス
    ja_main = DATA_DIR / "ja" / "main.txt"
    en_main = DATA_DIR / "en" / "main.txt"
    en_main.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📖 本文翻訳: 吾輩は猫である → I Am a Cat")
    print(f"   対象章数: {args.chapters}")
    print("=" * 60)

    # 入力テキスト読み込み
    text = ja_main.read_text(encoding="utf-8")
    chapters = split_chapters(text)
    print(f"✓ {len(chapters)} 章を検出")

    # 翻訳対象の章を制限
    target_chapters = chapters[:args.chapters]
    print(f"✓ 翻訳対象: 第1章〜第{target_chapters[-1]['num']}章")

    # 進捗の読み込み
    progress = load_progress() if args.resume else {}

    # OpenAI クライアント
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 翻訳実行
    translated_chapters = []
    for ch in target_chapters:
        ch_num = ch["num"]
        print(f"\n📝 第{ch_num}章を翻訳中...")

        blocks = split_into_blocks(ch["text"], max_chars=args.block_size)
        print(f"   {len(blocks)} ブロックに分割")

        translated_blocks = []
        for i, block in enumerate(blocks):
            progress_key = f"ch{ch_num}_block{i}"

            if progress_key in progress:
                print(f"   ⏭️  ブロック {i+1}/{len(blocks)} はスキップ（翻訳済み）")
                translated_blocks.append(progress[progress_key])
                continue

            print(f"   🔄 ブロック {i+1}/{len(blocks)} を翻訳中 ({len(block)} 文字)...")
            try:
                translated = translate_block(client, block, ch_num, i)
                translated_blocks.append(translated)

                # 進捗保存
                progress[progress_key] = translated
                save_progress(progress)

                print(f"   ✅ ブロック {i+1}/{len(blocks)} 完了 ({len(translated)} 文字)")

                # レート制限対策
                time.sleep(1)
            except Exception as e:
                print(f"   ❌ ブロック {i+1}/{len(blocks)} エラー: {e}")
                print("   ⚠️  中断します。--resume で再開してください。")
                # ここまでの結果を保存
                save_progress(progress)
                return

        chapter_text = f"Chapter {ch_num}\n\n" + "\n\n".join(translated_blocks)
        translated_chapters.append(chapter_text)
        print(f"✅ 第{ch_num}章 翻訳完了")

    # 出力
    output_text = "\n\n".join(translated_chapters)
    en_main.write_text(output_text, encoding="utf-8")
    print(f"\n✅ 翻訳完了: {en_main}")
    print(f"   総文字数: {len(output_text)}")


if __name__ == "__main__":
    main()
