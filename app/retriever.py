"""
検索ロジック（RAG）

pos 以前のチャンクのみを検索対象とし、ネタバレを防止
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import get_chunks_cache, get_events_cache, embed, chat

logger = logging.getLogger(__name__)

# Qdrant クライアント（オプショナル）
_qdrant_client = None
COLLECTION_NAME = "neko_scenes"


def get_qdrant_client():
    """Qdrant クライアントを取得（利用可能な場合）"""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            _qdrant_client = QdrantClient(host="localhost", port=6333)
            # 接続テスト
            _qdrant_client.get_collections()
            logger.info("✓ Qdrant 接続成功")
        except Exception as e:
            logger.warning(f"⚠️  Qdrant 接続失敗: {e}")
            _qdrant_client = False  # 失敗を記録
    return _qdrant_client if _qdrant_client is not False else None


def find_current_scene(pos: int) -> Optional[int]:
    """pos を含む/最も近いチャンクの scene_index を返す"""
    chunks = get_chunks_cache()

    # pos を含むチャンクを探す
    for chunk in chunks:
        if chunk["start_pos"] <= pos <= chunk["end_pos"]:
            return chunk["scene_index"]

    # 含まれない場合、最も近いチャンクを探す
    closest = min(chunks, key=lambda c: abs(c["start_pos"] - pos))
    return closest["scene_index"]


def retrieve_nearby(scene: int, window: int = 3) -> List[Dict[str, Any]]:
    """scene の前後 window のチャンクを取得"""
    chunks = get_chunks_cache()

    nearby = []
    for chunk in chunks:
        if scene - window <= chunk["scene_index"] <= scene + window:
            nearby.append(chunk)

    return nearby


def search_semantic_qdrant(
    query_vec: List[float],
    k: int,
    max_pos: int
) -> Optional[List[Dict[str, Any]]]:
    """Qdrant でベクトル検索（pos フィルタ付き）"""
    client = get_qdrant_client()
    if client is None:
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, Range

        # max_pos を含むチャンク、または max_pos 以前で終わるチャンク
        # start_pos <= max_pos のフィルタ（end_pos <= max_pos または start_pos <= max_pos <= end_pos）
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="start_pos",
                    range=Range(lte=max_pos)
                )
            ]
        )

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            query_filter=query_filter,
            limit=k
        )

        chunks = []
        for hit in results:
            chunk_data = {
                "scene_index": hit.payload["scene_index"],
                "chapter": hit.payload["chapter"],
                "start_pos": hit.payload["start_pos"],
                "end_pos": hit.payload["end_pos"],
                "text": hit.payload["text"],
                "characters": hit.payload.get("characters", []),
                "score": hit.score
            }
            chunks.append(chunk_data)

        logger.info(f"✓ Qdrant 検索: {len(chunks)} 件取得")
        return chunks

    except Exception as e:
        logger.error(f"❌ Qdrant 検索エラー: {e}")
        return None


def search_semantic_fallback(
    query_vec: List[float],
    k: int,
    max_pos: int
) -> List[Dict[str, Any]]:
    """フォールバック: ローカルでコサイン類似度検索"""
    chunks = get_chunks_cache()

    # max_pos を含むチャンク、または max_pos 以前で終わるチャンク
    candidates = [c for c in chunks if c["start_pos"] <= max_pos <= c["end_pos"] or c["end_pos"] <= max_pos]

    if not candidates:
        logger.warning("⚠️  max_pos 以前のチャンクがありません")
        return []

    # 各チャンクを埋め込み
    logger.info(f"📊 フォールバック検索: {len(candidates)} 件から埋め込み計算中...")
    chunk_vecs = []
    for chunk in candidates:
        try:
            vec = embed(chunk["text"][:500])  # 先頭500文字
            chunk_vecs.append(vec)
        except Exception as e:
            logger.error(f"埋め込みエラー: {e}")
            chunk_vecs.append([0.0] * len(query_vec))  # ダミー

    # コサイン類似度計算
    query_vec_np = np.array(query_vec).reshape(1, -1)
    chunk_vecs_np = np.array(chunk_vecs)

    similarities = cosine_similarity(query_vec_np, chunk_vecs_np)[0]

    # 上位 k 件
    top_indices = np.argsort(similarities)[::-1][:k]

    results = []
    for idx in top_indices:
        chunk = candidates[idx].copy()
        chunk["score"] = float(similarities[idx])
        results.append(chunk)

    logger.info(f"✓ フォールバック検索: {len(results)} 件取得")
    return results


def expand_query_with_history(question: str, history: List[Dict[str, str]] = None) -> str:
    """
    会話履歴を考慮して質問を拡張・リライト
    
    Args:
        question: 現在の質問
        history: 会話履歴 [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        拡張された質問文
    """
    if not history or len(history) == 0:
        return question
    
    # 直近の会話履歴を取得（最大3ターン）
    recent_history = history[-6:] if len(history) > 6 else history
    
    # 履歴からキーワードや文脈を抽出
    history_context = ""
    for msg in recent_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            history_context += f"{content}\n"
    
    # LLMを使って質問を拡張・リライト
    try:
        system_prompt = """あなたは検索クエリを改善するアシスタントです。
会話履歴と現在の質問を考慮して、より検索に適した質問文にリライトしてください。

以下の点を重視してください：
- 会話履歴の文脈を考慮する
- 質問の意図を明確にする
- 具体的なキーワードを含める
- 簡潔で検索に適した形にする（50文字以内を目安）
- 元の質問の意図を変えない"""
        
        user_message = f"""会話履歴:
{history_context}

現在の質問: {question}

上記の会話履歴と現在の質問を考慮して、より検索に適した質問文にリライトしてください。"""
        
        expanded = chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0.3,
            max_tokens=100
        )
        
        expanded = expanded.strip()
        if expanded and len(expanded) > 10:  # 有効な結果の場合
            logger.info(f"📝 質問拡張: '{question}' → '{expanded}'")
            return expanded
    except Exception as e:
        logger.warning(f"⚠️  質問拡張エラー: {e}")
    
    return question


def search_keyword(
    query: str,
    k: int,
    max_pos: int
) -> List[Dict[str, Any]]:
    """
    キーワード検索（簡易版：テキスト内のキーワードマッチング）
    
    Args:
        query: 検索クエリ
        k: 取得件数
        max_pos: 最大位置
    
    Returns:
        検索結果チャンクのリスト
    """
    chunks = get_chunks_cache()
    
    # max_pos以前のチャンクをフィルタ
    candidates = [c for c in chunks if c["start_pos"] <= max_pos]
    
    if not candidates:
        return []
    
    # クエリからキーワードを抽出（日本語の単語境界を考慮）
    keywords = []
    # 名詞や重要そうな単語を抽出（簡易版：2文字以上の連続文字）
    words = re.findall(r'[一-龥ぁ-んァ-ン]{2,}', query)
    keywords.extend(words)
    
    # スコア計算（キーワードの出現回数）
    scored_chunks = []
    for chunk in candidates:
        text = chunk.get("text", "")
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            count = text.count(keyword)
            if count > 0:
                score += count
                matched_keywords.append(keyword)
        
        if score > 0:
            chunk_copy = chunk.copy()
            chunk_copy["score"] = float(score)
            chunk_copy["matched_keywords"] = matched_keywords
            scored_chunks.append(chunk_copy)
    
    # スコアでソート
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    logger.info(f"🔍 キーワード検索: {len(scored_chunks)} 件 (キーワード: {keywords})")
    return scored_chunks[:k]


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    LLMを使って検索結果を再ランキング
    
    Args:
        query: 検索クエリ
        chunks: 検索結果チャンク
        top_k: 上位k件を返す（Noneの場合は全て）
    
    Returns:
        再ランキングされたチャンクのリスト
    """
    if not chunks or len(chunks) <= 1:
        return chunks
    
    try:
        # チャンクのテキストを準備
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:300]  # 先頭300文字
            chunk_texts.append(f"[{i+1}] {text}")
        
        system_prompt = """あなたは検索結果をランキングするアシスタントです。
検索クエリに関連性が高い順に、チャンクの番号を並び替えてください。

以下の点を重視してください：
- 検索クエリの意図に最も関連するチャンクを上位に
- 関連性の低いチャンクは下位に
- 番号のみをカンマ区切りで返す（例: 3,1,2,4）"""
        
        user_message = f"""検索クエリ: {query}

検索結果:
{chr(10).join(chunk_texts)}

上記の検索結果を、検索クエリに関連性が高い順に並び替えてください。
番号のみをカンマ区切りで返してください（例: 3,1,2,4）。"""
        
        result = chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        # 番号を抽出
        numbers = re.findall(r'\d+', result)
        if numbers:
            indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(chunks)]
            if len(indices) == len(chunks):
                # 重複除去して順序を保持
                seen = set()
                reranked = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        reranked.append(chunks[idx])
                # 残りを追加
                for i, chunk in enumerate(chunks):
                    if i not in seen:
                        reranked.append(chunk)
                
                logger.info(f"🔄 再ランキング: {len(reranked)} 件")
                if top_k:
                    return reranked[:top_k]
                return reranked
    except Exception as e:
        logger.warning(f"⚠️  再ランキングエラー: {e}")
    
    # エラー時は元の順序を返す
    if top_k:
        return chunks[:top_k]
    return chunks


def retrieve_chunks(
    question: str,
    pos: int,
    k: int = 8,
    window: int = 3,
    history: List[Dict[str, str]] = None,
    use_query_expansion: bool = True,
    use_hybrid_search: bool = True,
    use_reranking: bool = True
) -> Tuple[List[Dict[str, Any]], str]:
    """
    質問と位置に基づいてチャンクを取得（改善版）

    Args:
        question: 質問文
        pos: 現在位置
        k: 取得チャンク数
        window: 近傍ウィンドウサイズ
        history: 会話履歴 [{"role": "user/assistant", "content": "..."}]
        use_query_expansion: 質問拡張を使用するか
        use_hybrid_search: ハイブリッド検索を使用するか
        use_reranking: 再ランキングを使用するか

    Returns:
        (chunks, method): チャンクリストと使用した検索方法
    """
    # 現在のシーンを特定
    current_scene = find_current_scene(pos)
    logger.info(f"📍 現在位置: pos={pos}, scene={current_scene}")

    # 近傍ウィンドウ取得
    nearby = retrieve_nearby(current_scene, window=window)
    logger.info(f"📦 近傍チャンク: {len(nearby)} 件")

    # 質問の拡張・リライト
    search_query = question
    if use_query_expansion and history:
        search_query = expand_query_with_history(question, history)
    
    logger.info(f"🔍 検索クエリ: '{search_query}'")

    # セマンティック検索
    query_vec = embed(search_query)
    semantic_results = search_semantic_qdrant(query_vec, k=k*2, max_pos=pos)  # 多めに取得
    method = "qdrant"

    # Qdrant検索が失敗した、または結果が0件の場合はフォールバック
    if semantic_results is None or len(semantic_results) == 0:
        logger.info("⚠️  Qdrant検索結果が0件のため、フォールバック検索に切り替えます")
        semantic_results = search_semantic_fallback(query_vec, k=k*2, max_pos=pos)
        method = "fallback"

    # ハイブリッド検索：キーワード検索も実行
    keyword_results = []
    if use_hybrid_search:
        keyword_results = search_keyword(search_query, k=k, max_pos=pos)
        logger.info(f"🔑 キーワード検索: {len(keyword_results)} 件")

    # 近傍 + セマンティック + キーワード結果を統合（重複除去）
    seen_scenes = set()
    combined = []
    scene_to_chunk = {}  # シーンごとの最良チャンクを保持

    # 近傍を優先
    nearby_added = 0
    for chunk in nearby:
        if chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos:
            scene_idx = chunk["scene_index"]
            if scene_idx not in seen_scenes:
                seen_scenes.add(scene_idx)
                chunk["source"] = "nearby"
                chunk["score"] = chunk.get("score", 1.0) + 0.5  # 近傍はボーナス
                scene_to_chunk[scene_idx] = chunk
                nearby_added += 1
    logger.info(f"📌 近傍チャンク追加: {nearby_added}/{len(nearby)} 件")

    # セマンティック検索結果を追加
    for chunk in semantic_results:
        scene_idx = chunk["scene_index"]
        if scene_idx not in seen_scenes:
            seen_scenes.add(scene_idx)
            chunk["source"] = "semantic"
            scene_to_chunk[scene_idx] = chunk
        else:
            # 既存のチャンクがある場合、スコアが高い方を保持
            existing = scene_to_chunk[scene_idx]
            if chunk.get("score", 0) > existing.get("score", 0):
                chunk["source"] = "semantic"
                scene_to_chunk[scene_idx] = chunk

    # キーワード検索結果を追加
    for chunk in keyword_results:
        scene_idx = chunk["scene_index"]
        if scene_idx not in seen_scenes:
            seen_scenes.add(scene_idx)
            chunk["source"] = "keyword"
            scene_to_chunk[scene_idx] = chunk
        else:
            # 既存のチャンクがある場合、スコアを加算
            existing = scene_to_chunk[scene_idx]
            existing["score"] = existing.get("score", 0) + chunk.get("score", 0) * 0.3
            existing["source"] = existing.get("source", "") + "+keyword"

    combined = list(scene_to_chunk.values())
    
    # スコアでソート
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 再ランキング
    if use_reranking and len(combined) > 2:
        combined = rerank_chunks(search_query, combined, top_k=k*2)

    # 上位 k 件に制限
    combined = combined[:k]

    logger.info(f"✅ 最終取得: {len(combined)} 件 (method={method})")
    return combined, method


def retrieve_relevant_events(
    current_scene: int,
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    現在のシーンと取得したチャンクに関連するイベントを取得

    pos より未来のイベントは除外
    """
    events = get_events_cache()

    # 取得したチャンクの scene 範囲
    scene_indices = {c["scene_index"] for c in chunks}
    min_scene = min(scene_indices) if scene_indices else current_scene
    max_scene = current_scene  # 現在位置まで

    relevant = []
    for event in events:
        first = event.get("first_scene")
        last = event.get("last_scene")

        if first is None or last is None:
            continue

        # イベントの範囲が max_scene 以前で、かつ取得範囲と重なる
        if last <= max_scene and first <= max_scene:
            # 重なりチェック
            if first <= max_scene and last >= min_scene:
                relevant.append(event)

    logger.info(f"📅 関連イベント: {len(relevant)} 件")
    return relevant[:5]  # 最大5件


def get_current_situation(
    pos: int,
    character_name: str,
    window: int = 5
) -> str:
    """
    指定された位置付近で、指定されたキャラクターが何をしているかを取得し、要約して返す

    Args:
        pos: 現在の文字位置
        character_name: キャラクター名
        window: 検索する前後のシーン数

    Returns:
        キャラクターの状況を要約したテキスト（常に何かしらの結果を返す）
    """
    chunks = get_chunks_cache()
    current_scene = find_current_scene(pos)

    # 現在位置付近のチャンクを取得（pos以前のみ）
    nearby_chunks = []
    for chunk in chunks:
        # posを含むチャンク、またはpos以前で終わるチャンク
        if (chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos):
            # 現在シーンの前後window内
            if abs(chunk["scene_index"] - current_scene) <= window:
                # 指定されたキャラクターが登場しているチャンク
                if character_name in chunk.get("characters", []):
                    nearby_chunks.append(chunk)

    # チャンクが見つからない場合、windowを広げて再検索
    if not nearby_chunks:
        logger.info(f"⚠️  {character_name}の近傍チャンクが見つからないため、windowを拡大して再検索")
        expanded_window = window * 2
        for chunk in chunks:
            if (chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos):
                if abs(chunk["scene_index"] - current_scene) <= expanded_window:
                    if character_name in chunk.get("characters", []):
                        nearby_chunks.append(chunk)

    # それでも見つからない場合、現在位置付近のチャンクを取得（キャラクター指定なし）
    if not nearby_chunks:
        logger.info(f"⚠️  {character_name}のチャンクが見つからないため、現在位置付近のチャンクを使用")
        for chunk in chunks:
            if (chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos):
                if abs(chunk["scene_index"] - current_scene) <= window:
                    nearby_chunks.append(chunk)

    # 最も近いチャンク（posに最も近い）を優先してソート
    nearby_chunks.sort(key=lambda c: abs(c["start_pos"] - pos))

    # 最大3つのチャンクのテキストを集める（近いものから）
    relevant_texts = []
    for chunk in nearby_chunks[:3]:
        text = chunk.get("text", "").strip()
        if text:
            relevant_texts.append(text)

    # テキストが見つからない場合でも、現在位置付近のテキストを取得
    if not relevant_texts:
        logger.warning(f"⚠️  {character_name}の関連テキストが見つかりません。現在位置付近のテキストを使用")
        # 現在位置を含むチャンクを探す
        for chunk in chunks:
            if chunk["start_pos"] <= pos <= chunk["end_pos"]:
                text = chunk.get("text", "").strip()
                if text:
                    relevant_texts.append(text[:500])  # 先頭500文字
                    break

    # チャンクテキストを結合
    combined_text = "\n\n".join(relevant_texts) if relevant_texts else ""

    # 要約プロンプトを作成（特定のキャラクターを指定せず、純粋に現在の状況を要約）
    system_prompt = """あなたは小説のテキストを分析するアシスタントです。
テキスト近傍で何が起こっているのかを整理し、登場している人物ごとに、誰が何をしているのかを説明してください。

以下の点を重視してください：
- テキスト近傍で何が起こっているのかを整理する（場面の状況、出来事の流れなど）
- 登場している人物を特定し、それぞれの人物が何をしているのか、どのような行動を取っているのかを具体的に説明する
- 各人物の状況や心理状態も含めて説明する
- 人物ごとに分けて説明する（例：「人物A: ...」「人物B: ...」のような形式）
- 簡潔すぎず、必要な情報を含めて説明する（100-200文字程度を目安）"""

    user_message = f"""以下のテキストから、テキスト近傍で何が起こっているのかを整理し、登場している人物ごとに、誰が何をしているのかを説明してください。

{combined_text}"""

    # 要約に使用されるプロンプト全文をログに出力
    logger.info("=" * 60)
    logger.info(f"📝 {character_name}の状況要約プロンプト全文:")
    logger.info(f"   システムプロンプト:")
    logger.info(f"   {system_prompt}")
    logger.info(f"   ユーザーメッセージ:")
    logger.info(f"   {user_message}")
    logger.info("=" * 60)

    try:
        # 軽量なモデルで要約（gpt-4o-miniを使用、max_tokensを増やして詳細な説明を可能にする）
        summary = chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0.3,
            max_tokens=300
        )
        result = summary.strip()
        if not result:
            result = "現在位置付近の状況は不明"
        logger.info(f"✓ {character_name}の状況要約結果: {result[:50]}...")
        return result
    except Exception as e:
        logger.error(f"❌ 要約エラー ({character_name}): {e}")
        # エラー時はフォールバック：最初のチャンクの一部を返す
        if relevant_texts:
            fallback_text = relevant_texts[0]
            if len(fallback_text) > 200:
                fallback_text = fallback_text[:200] + "..."
            return fallback_text
        else:
            # それでも見つからない場合は、デフォルトメッセージを返す
            return "現在位置付近の状況は不明"
