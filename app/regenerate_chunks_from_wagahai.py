#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wagahai.json から chunks.jsonl を再生成するスクリプト

paragraph型、conversation型のtext部分を積み重ねて、
400文字を超えるまで積み重ね、超えた場合に別のチャンクとして分割します。
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Set
from dataclasses import dataclass, asdict
import re

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# .env を api-server 直下からロード
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

# 設定
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "neko_scenes"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100
TARGET_CHUNK_SIZE = 400  # 目標チャンクサイズ（文字数）

# データファイルパス
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WAGAHAI_JSON = DATA_DIR / "wagahai.json"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"

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


# ========= キャラクター検出 =========
_JP_BOUND = r"(?<![一-龥ぁ-んァ-ンA-Za-z0-9]){name}"

def _contains_name(text: str, name: str) -> bool:
    if name == "白":
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


# ========= チャンク生成 =========
def load_wagahai_json(filepath: Path) -> Dict[str, Any]:
    """wagahai.json を読み込み"""
    print(f"📖 {filepath} を読み込み中...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ 読み込み完了")
    return data

def split_oversize(text: str, max_len: int = TARGET_CHUNK_SIZE) -> List[str]:
    """
    大きすぎるテキストを分割する。
    句点や読点などの区切りでmax_lenを超えない最後の位置で切る。
    """
    if len(text) <= max_len:
        return [text]
    
    res = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_len, n)
        window = text[i:end]
        if end < n:
            # 区切り文字で分割
            cut = max(
                window.rfind("。"),
                window.rfind("、"),
                window.rfind("，"),
                window.rfind("\n"),
                window.rfind(" "),
            )
            if cut >= 0 and cut >= max_len // 2:
                end = i + cut + 1
        res.append(text[i:end])
        i = end
    return [r.strip() for r in res if r.strip()]

def build_chunks_from_wagahai(data: Dict[str, Any]) -> List[ChunkRec]:
    """wagahai.json のデータからチャンクを生成"""
    print(f"\n🔨 チャンク生成中...")
    
    chunks: List[ChunkRec] = []
    scene_idx = 1
    global_pos = 0
    
    chapters = data.get("content", {}).get("chapters", [])
    
    for chapter_data in chapters:
        chapter_id = chapter_data.get("id", 1)
        blocks = chapter_data.get("blocks", [])
        
        current_chunk_text = ""
        current_chunk_start = global_pos
        
        for block in blocks:
            block_type = block.get("type", "")
            block_text = block.get("text", "")
            
            # paragraph型とconversation型のみ処理
            if block_type not in ["paragraph", "conversation"]:
                continue
            
            if not block_text.strip():
                continue
            
            # ブロックが長すぎる場合は分割
            if len(block_text) > TARGET_CHUNK_SIZE:
                # 現在のチャンクがあれば確定
                if current_chunk_text.strip():
                    scene_id = f"scene_{scene_idx:05d}"
                    current_chunk_end = global_pos
                    
                    characters = sorted(detect_allowed_names(current_chunk_text))[:8]
                    
                    chunks.append(ChunkRec(
                        id=scene_id,
                        chapter=chapter_id,
                        scene_index=scene_idx,
                        start_pos=current_chunk_start,
                        end_pos=current_chunk_end,
                        characters=characters,
                        text=current_chunk_text.strip()
                    ))
                    
                    scene_idx += 1
                    global_pos = current_chunk_end
                    current_chunk_text = ""
                
                # 長いブロックを分割
                block_parts = split_oversize(block_text, TARGET_CHUNK_SIZE)
                for part in block_parts:
                    scene_id = f"scene_{scene_idx:05d}"
                    part_start = global_pos
                    part_end = global_pos + len(part)
                    
                    characters = sorted(detect_allowed_names(part))[:8]
                    
                    chunks.append(ChunkRec(
                        id=scene_id,
                        chapter=chapter_id,
                        scene_index=scene_idx,
                        start_pos=part_start,
                        end_pos=part_end,
                        characters=characters,
                        text=part.strip()
                    ))
                    
                    scene_idx += 1
                    global_pos = part_end
                
                current_chunk_text = ""
                current_chunk_start = global_pos
                continue
            
            # 現在のチャンクに追加した場合の文字数を計算
            if current_chunk_text:
                test_text = current_chunk_text + block_text
            else:
                test_text = block_text
            
            # 400文字を超える場合は、現在のチャンクを確定して新しいチャンクを開始
            if len(test_text) > TARGET_CHUNK_SIZE and current_chunk_text:
                # 現在のチャンクを確定
                scene_id = f"scene_{scene_idx:05d}"
                current_chunk_end = global_pos
                
                # キャラクター検出
                characters = sorted(detect_allowed_names(current_chunk_text))[:8]
                
                chunks.append(ChunkRec(
                    id=scene_id,
                    chapter=chapter_id,
                    scene_index=scene_idx,
                    start_pos=current_chunk_start,
                    end_pos=current_chunk_end,
                    characters=characters,
                    text=current_chunk_text.strip()
                ))
                
                scene_idx += 1
                global_pos = current_chunk_end
                
                # 新しいチャンクを開始
                current_chunk_text = block_text
                current_chunk_start = global_pos
                global_pos += len(block_text)
            else:
                # 現在のチャンクに追加
                if current_chunk_text:
                    current_chunk_text += block_text
                else:
                    current_chunk_text = block_text
                    current_chunk_start = global_pos
                global_pos += len(block_text)
        
        # 章の最後に残っているチャンクを確定
        if current_chunk_text.strip():
            scene_id = f"scene_{scene_idx:05d}"
            current_chunk_end = global_pos
            
            # キャラクター検出
            characters = sorted(detect_allowed_names(current_chunk_text))[:8]
            
            chunks.append(ChunkRec(
                id=scene_id,
                chapter=chapter_id,
                scene_index=scene_idx,
                start_pos=current_chunk_start,
                end_pos=current_chunk_end,
                characters=characters,
                text=current_chunk_text.strip()
            ))
            
            scene_idx += 1
    
    print(f"✓ {len(chunks)} 件のチャンクを生成")
    return chunks

def get_overlap_text(text: str, overlap_size: int, from_end: bool = False) -> str:
    """
    テキストから指定サイズのオーバーラップ部分を取得
    from_end=True: 末尾から取得、False: 先頭から取得
    句読点の区切りで適切に分割
    """
    if not text or len(text) <= overlap_size:
        return text
    
    if from_end:
        # 末尾から取得: 最後の句読点の位置を探して、そこから50文字程度を取得
        start_pos = max(0, len(text) - overlap_size - 100)  # 少し余裕を持たせる
        search_text = text[start_pos:]
        
        # 最初の句読点の位置を探す
        best_pos = 0
        for punct in ["。", "、", "，", "\n"]:
            idx = search_text.find(punct)
            if idx >= 0 and idx < overlap_size:
                best_pos = max(best_pos, idx + 1)
        
        if best_pos > 0:
            overlap_text = search_text[best_pos:]
        else:
            overlap_text = text[-overlap_size:]
        
        # 50文字を超えないように調整
        if len(overlap_text) > overlap_size:
            overlap_text = overlap_text[:overlap_size]
            # 最後の句読点まで戻る
            for punct in ["。", "、", "，", "\n"]:
                idx = overlap_text.rfind(punct)
                if idx >= 0:
                    overlap_text = overlap_text[:idx + 1]
                    break
        
        return overlap_text.strip()
    else:
        # 先頭から取得: 最初の50文字程度を取得し、最後の句読点まで
        overlap_text = text[:overlap_size + 100]  # 少し余裕を持たせる
        
        # 最後の句読点の位置を探す
        best_pos = len(overlap_text)
        for punct in ["。", "、", "，", "\n"]:
            idx = overlap_text.rfind(punct)
            if idx >= 0 and idx >= overlap_size // 2:  # 最低でも半分は確保
                best_pos = min(best_pos, idx + 1)
        
        if best_pos < len(overlap_text):
            overlap_text = overlap_text[:best_pos]
        else:
            overlap_text = text[:overlap_size]
        
        return overlap_text.strip()

def add_overlap_to_chunks(chunks: List[ChunkRec], overlap_size: int = 50) -> List[ChunkRec]:
    """
    チャンク間にオーバーラップを追加
    先頭と末尾のチャンクを除く各チャンクに、前後のチャンクのテキストを追加
    """
    if len(chunks) <= 1:
        return chunks
    
    print(f"\n🔗 オーバーラップ追加中 (サイズ: {overlap_size}文字)...")
    
    result_chunks = []
    
    for i, chunk in enumerate(chunks):
        original_text = chunk.text
        new_text = original_text
        
        # 先頭のチャンクでない場合、前のチャンクの最後50文字を追加
        if i > 0:
            prev_text = chunks[i - 1].text
            prev_overlap = get_overlap_text(prev_text, overlap_size, from_end=True)
            if prev_overlap:
                new_text = prev_overlap + new_text
        
        # 末尾のチャンクでない場合、次のチャンクの最初50文字を追加
        if i < len(chunks) - 1:
            next_text = chunks[i + 1].text
            next_overlap = get_overlap_text(next_text, overlap_size, from_end=False)
            if next_overlap:
                new_text = new_text + next_overlap
        
        # 新しいチャンクを作成（テキストのみ変更、その他は同じ）
        new_chunk = ChunkRec(
            id=chunk.id,
            chapter=chunk.chapter,
            scene_index=chunk.scene_index,
            start_pos=chunk.start_pos,
            end_pos=chunk.end_pos,
            characters=chunk.characters,
            text=new_text
        )
        result_chunks.append(new_chunk)
    
    print(f"✓ オーバーラップ追加完了")
    return result_chunks

def write_chunks_jsonl(chunks: List[ChunkRec], filepath: Path):
    """chunks.jsonl に書き込み"""
    print(f"\n💾 {filepath} に書き込み中...")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"✓ 書き込み完了")


# ========= Qdrant インデックス作成 =========
def get_embedding_dimension(client: OpenAI, model: str) -> int:
    """埋め込みモデルの次元数を取得"""
    print(f"\n📏 埋め込みモデル '{model}' の次元数を取得中...")
    response = client.embeddings.create(
        model=model,
        input="test"
    )
    dimension = len(response.data[0].embedding)
    print(f"✓ 次元数: {dimension}")
    return dimension

def create_collection(qdrant: QdrantClient, collection_name: str, dimension: int):
    """コレクションを作成（既存がある場合は削除して再作成）"""
    print(f"\n🗂️  コレクション '{collection_name}' を準備中...")
    
    # 既存コレクションの確認
    collections = qdrant.get_collections().collections
    exists = any(col.name == collection_name for col in collections)
    
    if exists:
        print(f"⚠️  既存コレクション '{collection_name}' を削除中...")
        qdrant.delete_collection(collection_name)
        print("✓ 削除完了")
    
    # 新規作成
    print(f"✓ コレクション '{collection_name}' を作成中...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
    print("✓ 作成完了")

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """テキストリストを埋め込みベクトルに変換"""
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [data.embedding for data in response.data]

def index_chunks_to_qdrant(
    qdrant: QdrantClient,
    openai_client: OpenAI,
    chunks: List[ChunkRec],
    collection_name: str,
    embed_model: str,
    batch_size: int
):
    """チャンクをバッチ処理で Qdrant にインデックス"""
    print(f"\n🚀 インデックス作成開始 (バッチサイズ: {batch_size})")
    
    total = len(chunks)
    indexed_count = 0
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"📦 バッチ {batch_num}/{total_batches} (件数: {len(batch)}) を処理中...")
        
        # テキストを埋め込み
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(openai_client, texts, embed_model)
        
        # ポイントを作成
        points = []
        for chunk, embedding in zip(batch, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "scene_id": chunk.id,
                "scene_index": chunk.scene_index,
                "chapter": chunk.chapter,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "characters": chunk.characters,
                "text": chunk.text
            }
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Qdrant にアップサート
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        
        indexed_count += len(batch)
        print(f"✓ {indexed_count}/{total} 件完了")
        
        # APIレート制限を避けるため少し待機
        if i + batch_size < total:
            time.sleep(0.5)
    
    print(f"\n✅ インデックス作成完了: {indexed_count} 件")

def print_statistics(chunks: List[ChunkRec]):
    """統計情報を出力"""
    print("\n📊 統計情報:")
    print(f"  総件数: {len(chunks)}")
    
    if chunks:
        first = chunks[0]
        last = chunks[-1]
        print(f"  最初の scene_index: {first.scene_index} (章 {first.chapter})")
        print(f"  最後の scene_index: {last.scene_index} (章 {last.chapter})")
        
        # 章ごとの件数
        chapters = {}
        for chunk in chunks:
            ch = chunk.chapter
            chapters[ch] = chapters.get(ch, 0) + 1
        
        print(f"  章の数: {len(chapters)}")
        print(f"  章ごとの件数: {dict(sorted(chapters.items()))}")
        
        # チャンクサイズの統計
        sizes = [len(chunk.text) for chunk in chunks]
        if sizes:
            print(f"  平均チャンクサイズ: {sum(sizes) / len(sizes):.1f} 文字")
            print(f"  最小チャンクサイズ: {min(sizes)} 文字")
            print(f"  最大チャンクサイズ: {max(sizes)} 文字")


def main():
    print("=" * 60)
    print("🐱 吾輩は猫である - チャンク再生成 & Qdrant インデックス作成")
    print("=" * 60)
    
    # wagahai.json を読み込み
    data = load_wagahai_json(WAGAHAI_JSON)
    
    # チャンクを生成
    chunks = build_chunks_from_wagahai(data)
    
    # オーバーラップを追加
    chunks = add_overlap_to_chunks(chunks, overlap_size=50)
    
    # 統計情報を表示
    print_statistics(chunks)
    
    # chunks.jsonl に書き込み
    write_chunks_jsonl(chunks, CHUNKS_JSONL)
    
    # OpenAI クライアント初期化
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Qdrant クライアント初期化
    print(f"\n🔌 Qdrant に接続中 ({QDRANT_HOST}:{QDRANT_PORT})...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ 接続成功")
    
    # 埋め込み次元数を取得
    dimension = get_embedding_dimension(openai_client, EMBED_MODEL)
    
    # コレクション作成
    create_collection(qdrant, COLLECTION_NAME, dimension)
    
    # インデックス作成
    index_chunks_to_qdrant(
        qdrant=qdrant,
        openai_client=openai_client,
        chunks=chunks,
        collection_name=COLLECTION_NAME,
        embed_model=EMBED_MODEL,
        batch_size=BATCH_SIZE
    )
    
    # 最終確認
    print("\n🔍 コレクション情報:")
    collection_info = qdrant.get_collection(COLLECTION_NAME)
    print(f"  コレクション名: {COLLECTION_NAME}")
    print(f"  ベクトル数: {collection_info.points_count}")
    print(f"  ベクトル次元: {collection_info.config.params.vectors.size}")
    
    print("\n" + "=" * 60)
    print("✅ すべての処理が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()

