#!/usr/bin/env python3
"""
Qdrant インデックス作成スクリプト

data/{lang}/chunks.jsonl を読み込み、Qdrant にベクトルインデックスを構築します。
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any

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
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH_SIZE = 100

# データファイルパス
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def get_collection_name(lang: str = "ja") -> str:
    """言語別の Qdrant コレクション名を返す"""
    if lang == "ja":
        return "neko_scenes"
    return f"neko_scenes_{lang}"


def get_embedding_dimension(client: OpenAI, model: str) -> int:
    """埋め込みモデルの次元数を取得"""
    print(f"📏 埋め込みモデル '{model}' の次元数を取得中...")
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


def load_chunks(filepath: Path) -> List[Dict[str, Any]]:
    """chunks.jsonl を読み込み"""
    print(f"\n📖 {filepath} を読み込み中...")
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    print(f"✓ {len(chunks)} 件のチャンクを読み込み")
    return chunks


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
    chunks: List[Dict[str, Any]],
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
        texts = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(openai_client, texts, embed_model)

        # ポイントを作成
        points = []
        for chunk, embedding in zip(batch, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "scene_id": chunk["id"],
                "scene_index": chunk["scene_index"],
                "chapter": chunk["chapter"],
                "start_pos": chunk["start_pos"],
                "end_pos": chunk["end_pos"],
                "characters": chunk["characters"],
                "text": chunk["text"]
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


def print_statistics(chunks: List[Dict[str, Any]]):
    """統計情報を出力"""
    print("\n📊 統計情報:")
    print(f"  総件数: {len(chunks)}")

    if chunks:
        first = chunks[0]
        last = chunks[-1]
        print(f"  最初の scene_index: {first['scene_index']} (章 {first['chapter']})")
        print(f"  最後の scene_index: {last['scene_index']} (章 {last['chapter']})")

        # 章ごとの件数
        chapters = {}
        for chunk in chunks:
            ch = chunk["chapter"]
            chapters[ch] = chapters.get(ch, 0) + 1

        print(f"  章の数: {len(chapters)}")
        print(f"  章ごとの件数: {dict(sorted(chapters.items()))}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant インデックス作成")
    parser.add_argument("--lang", type=str, default="ja", help="Language: 'ja' or 'en'")
    args = parser.parse_args()

    lang = args.lang
    collection_name = get_collection_name(lang)
    chunks_file = DATA_DIR / lang / "chunks.jsonl"

    print("=" * 60)
    print(f"🐱 吾輩は猫である - Qdrant インデックス作成 (lang={lang})")
    print("=" * 60)

    # OpenAI クライアント初期化
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Qdrant クライアント初期化
    print(f"\n🔌 Qdrant に接続中 ({QDRANT_HOST}:{QDRANT_PORT})...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("✓ 接続成功")

    # 埋め込み次元数を取得
    dimension = get_embedding_dimension(openai_client, EMBED_MODEL)

    # コレクション作成
    create_collection(qdrant, collection_name, dimension)

    # チャンクを読み込み
    chunks = load_chunks(chunks_file)

    # 統計情報を表示
    print_statistics(chunks)

    # インデックス作成
    index_chunks_to_qdrant(
        qdrant=qdrant,
        openai_client=openai_client,
        chunks=chunks,
        collection_name=collection_name,
        embed_model=EMBED_MODEL,
        batch_size=BATCH_SIZE
    )

    # 最終確認
    print("\n🔍 コレクション情報:")
    collection_info = qdrant.get_collection(collection_name)
    print(f"  コレクション名: {collection_name}")
    print(f"  ベクトル数: {collection_info.points_count}")
    print(f"  ベクトル次元: {collection_info.config.params.vectors.size}")

    print("\n" + "=" * 60)
    print("✅ すべての処理が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
