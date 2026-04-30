#!/usr/bin/env python3
"""
キャラクター・イベントデータ翻訳スクリプト

data/ja/character.json → data/en/character.json
data/ja/events.jsonl   → data/en/events.jsonl

Usage:
    python app/translate_characters.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# .env をロード
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# キャラクター名マッピング
NAME_MAP = {
    "吾輩": "the Cat",
    "おさん": "Osan",
    "珍野 苦沙弥": "Kushami",
    "珍野夫人": "Mrs. Kushami",
    "迷亭": "Meitei",
    "車屋の黒": "Kuro",
    "水島 寒月": "Kangetsu",
    "三毛子": "Mikeko",
    "二絃琴の師匠": "the koto teacher",
    "越智 東風": "Tofu",
    "八木 独仙": "Dokusen",
    "甘木先生": "Dr. Amaki",
    "金田": "Kaneda",
    "金田 鼻子": "Mrs. Kaneda",
    "金田 富子": "Tomiko Kaneda",
    "鈴木 籐十郎": "Suzuki Tojuro",
    "多々良 三平": "Tatara Sampei",
    "牧山": "Makiyama",
    "珍野 とん子": "Tonko",
    "珍野 すん子": "Sunko",
    "珍野 めん子": "Menko",
    "御三": "Osan",
    "雪江": "Yukie",
    "二絃琴の御師匠さん": "the koto teacher",
    "古井 武右衛門": "Furui Buemon",
    "吉田 虎蔵": "Yoshida Torazo",
    "泥棒陰士": "the thief",
    "八っちゃん": "Hatchan",
    "白": "Shiro",
}

# 一人称マッピング
STYLE_MAP = {
    "吾輩": "I",
    "私": "I",
    "僕": "I",
    "おれ": "I",
    "あたくし": "I",
}


def translate_text_field(client: OpenAI, text: str, context: str = "") -> str:
    """テキストフィールドを英訳"""
    if not text or not text.strip():
        return ""

    # キャラクター名の置換を含めた翻訳指示
    name_instructions = "\n".join([f"- {ja} → {en}" for ja, en in NAME_MAP.items()])

    prompt = f"""Translate the following Japanese text to English. This is from the novel "I Am a Cat" (吾輩は猫である) by Natsume Sōseki.

Character name mappings (use these consistently):
{name_instructions}

Context: {context}

Text to translate:
{text}

Translated text (English only, no explanations):"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


def translate_character_json(client: OpenAI):
    """character.json を英訳"""
    ja_path = DATA_DIR / "ja" / "character.json"
    en_path = DATA_DIR / "en" / "character.json"
    en_path.parent.mkdir(parents=True, exist_ok=True)

    print("🎭 character.json を翻訳中...")

    with open(ja_path, "r", encoding="utf-8") as f:
        characters = json.load(f)

    translated = []
    for char in characters:
        ja_name = char["name"]
        en_name = NAME_MAP.get(ja_name, ja_name)
        en_style = STYLE_MAP.get(char.get("style", ""), "I")

        print(f"  📝 {ja_name} → {en_name}...")

        # description-setting の翻訳
        desc_setting = translate_text_field(
            client,
            char.get("description-setting", ""),
            context=f"Character description for {en_name}"
        )
        time.sleep(0.5)

        # description-features の翻訳
        desc_features = translate_text_field(
            client,
            char.get("description-features", ""),
            context=f"Physical description of {en_name}"
        )
        time.sleep(0.5)

        # description-tone の翻訳
        desc_tone = translate_text_field(
            client,
            char.get("description-tone", ""),
            context=f"Speech style samples for {en_name}. These are example lines of dialogue."
        )
        time.sleep(0.5)

        # first-message の翻訳
        first_message = translate_text_field(
            client,
            char.get("first-message", ""),
            context=f"The first thing {en_name} says when meeting a new person"
        )
        time.sleep(0.5)

        # sample_dialogues の翻訳
        en_dialogues = []
        for dialogue in char.get("sample_dialogues", []):
            user_msg = translate_text_field(
                client,
                dialogue.get("user", ""),
                context=f"User's question to {en_name}"
            )
            time.sleep(0.3)

            thought = translate_text_field(
                client,
                dialogue.get("thought", ""),
                context=f"{en_name}'s internal thoughts (not spoken aloud)"
            )
            time.sleep(0.3)

            response = translate_text_field(
                client,
                dialogue.get("response", ""),
                context=f"{en_name}'s spoken response in character"
            )
            time.sleep(0.3)

            en_dialogues.append({
                "user": user_msg,
                "thought": thought,
                "response": response
            })

        translated.append({
            "id": char["id"],
            "name": en_name,
            "style": en_style,
            "description-setting": desc_setting,
            "description-features": desc_features,
            "description-tone": desc_tone,
            "first-message": first_message,
            "sample_dialogues": en_dialogues
        })

        print(f"  ✅ {en_name} 完了")

    # 出力
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    print(f"✅ character.json 翻訳完了: {en_path}")


def translate_events_jsonl(client: OpenAI):
    """events.jsonl を英訳"""
    ja_path = DATA_DIR / "ja" / "events.jsonl"
    en_path = DATA_DIR / "en" / "events.jsonl"
    en_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n📅 events.jsonl を翻訳中...")

    events = []
    with open(ja_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print(f"  {len(events)} 件のイベントを処理")

    # バッチ翻訳（効率化のため複数イベントをまとめて翻訳）
    batch_size = 10
    translated_events = []

    for i in range(0, len(events), batch_size):
        batch = events[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(events) + batch_size - 1) // batch_size

        print(f"  🔄 バッチ {batch_num}/{total_batches} ({len(batch)} 件)...")

        # バッチの翻訳リクエスト構築
        items_text = ""
        for j, ev in enumerate(batch):
            items_text += f"[{j}] title: {ev.get('title', '')}\n    summary: {ev.get('summary', '')}\n"

        name_instructions = "\n".join([f"- {ja} → {en}" for ja, en in NAME_MAP.items()])

        prompt = f"""Translate the following event titles and summaries from Japanese to English.
These are events from the novel "I Am a Cat" (吾輩は猫である).

Character name mappings:
{name_instructions}

Events to translate:
{items_text}

Return a JSON array where each element has "index", "title", and "summary" keys.
Example: [{{"index": 0, "title": "...", "summary": "..."}}]"""

        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000
            )
            content = response.choices[0].message.content.strip()

            # マークダウンコードブロックを除去
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            translations = json.loads(content)

            for trans in translations:
                idx = trans["index"]
                if 0 <= idx < len(batch):
                    ev = batch[idx].copy()
                    ev["title"] = trans["title"]
                    ev["summary"] = trans["summary"]
                    translated_events.append(ev)

        except Exception as e:
            print(f"  ❌ バッチ {batch_num} エラー: {e}")
            # フォールバック: 個別に翻訳
            for ev in batch:
                ev_copy = ev.copy()
                try:
                    ev_copy["title"] = translate_text_field(client, ev.get("title", ""), "Event title")
                    time.sleep(0.3)
                    ev_copy["summary"] = translate_text_field(client, ev.get("summary", ""), "Event summary")
                    time.sleep(0.3)
                except Exception as e2:
                    print(f"    ❌ 個別翻訳エラー: {e2}")
                translated_events.append(ev_copy)

        time.sleep(0.5)

    # 出力
    with open(en_path, "w", encoding="utf-8") as f:
        for ev in translated_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"✅ events.jsonl 翻訳完了: {en_path} ({len(translated_events)} 件)")


def main():
    print("=" * 60)
    print("🌐 キャラクター・イベントデータ翻訳")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    translate_character_json(client)
    translate_events_jsonl(client)

    print("\n" + "=" * 60)
    print("✅ すべての翻訳が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
