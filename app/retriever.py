"""
検索ロジック（RAG）

pos 以前のチャンクのみを検索対象とし、ネタバレを防止
"""

import logging
import re
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import get_chunks_cache, get_events_cache, get_personas_cache, embed, chat

logger = logging.getLogger(__name__)

# Qdrant クライアント（オプショナル）
_qdrant_client = None


def get_collection_name(lang: str = "ja") -> str:
    """言語別の Qdrant コレクション名を返す"""
    if lang == "ja":
        return "neko_scenes"
    return f"neko_scenes_{lang}"


# 状況要約のキャッシュ（(lang, pos) -> 要約テキスト）
_situation_cache: Dict[Tuple[str, int], str] = {}
_situation_locks: Dict[Tuple[str, int], threading.Event] = {}
_situation_locks_guard = threading.Lock()

# character.json の名前 → chunks.jsonl での名前の揺れマッピング
_CHARACTER_NAME_ALIASES: Dict[str, List[str]] = {
    "おさん": ["御三"],
    "二絃琴の師匠": ["二絃琴の御師匠さん"],
    "珍野夫人": [],  # chunks.jsonl に登場しない
}


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


def find_current_scene(pos: int, lang: str = "ja") -> Optional[int]:
    """pos を含む/最も近いチャンクの scene_index を返す"""
    chunks = get_chunks_cache(lang)

    # pos を含むチャンクを探す
    for chunk in chunks:
        if chunk["start_pos"] <= pos <= chunk["end_pos"]:
            return chunk["scene_index"]

    # 含まれない場合、最も近いチャンクを探す
    closest = min(chunks, key=lambda c: abs(c["start_pos"] - pos))
    return closest["scene_index"]


def retrieve_nearby(scene: int, window: int = 3, lang: str = "ja") -> List[Dict[str, Any]]:
    """scene の前後 window のチャンクを取得"""
    chunks = get_chunks_cache(lang)

    nearby = []
    for chunk in chunks:
        if scene - window <= chunk["scene_index"] <= scene + window:
            nearby.append(chunk)

    return nearby


def search_semantic_qdrant(
    query_vec: List[float],
    k: int,
    max_pos: int,
    lang: str = "ja"
) -> Optional[List[Dict[str, Any]]]:
    """Qdrant でベクトル検索（pos フィルタ付き）"""
    client = get_qdrant_client()
    if client is None:
        return None

    collection_name = get_collection_name(lang)

    try:
        from qdrant_client.models import FieldCondition, Filter, Range

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="start_pos",
                    range=Range(lte=max_pos)
                )
            ]
        )

        results = client.search(
            collection_name=collection_name,
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

        logger.info(f"✓ Qdrant 検索: {len(chunks)} 件取得 (collection={collection_name})")
        return chunks

    except Exception as e:
        logger.error(f"❌ Qdrant 検索エラー: {e}")
        return None


def search_semantic_fallback(
    query_vec: List[float],
    k: int,
    max_pos: int,
    lang: str = "ja"
) -> List[Dict[str, Any]]:
    """フォールバック: ローカルでコサイン類似度検索"""
    chunks = get_chunks_cache(lang)

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


def expand_query_with_history(
    question: str,
    history: List[Dict[str, str]] = None,
    character_name: str = None,
    pos: int = None,
    lang: str = "ja"
) -> str:
    """
    会話履歴、キャラクター情報、テキスト位置を考慮して質問を拡張・リライト
    """
    if not question or len(question.strip()) == 0:
        return question

    # 直近の会話履歴を取得（最大3ターン＝6メッセージ）
    recent_history = []
    if history and len(history) > 0:
        recent_history = history[-6:] if len(history) > 6 else history

    # 履歴からコンテキストを構築（キャラクター名を含める）
    history_context = ""
    for msg in recent_history:
        char_name = msg.get("character_name")
        if msg.get("role") == "user":
            role_label = "User" if lang == "en" else "ユーザー"
        elif char_name:
            role_label = char_name
        else:
            role_label = "Character" if lang == "en" else "キャラクター"

        content = msg.get("content", "")
        if content:
            history_context += f"{role_label}: {content}\n"

    # キャラクター情報の取得
    from app.utils import get_personas_cache
    personas = get_personas_cache(lang)
    character_info = ""
    if character_name and character_name in personas:
        persona = personas[character_name]
        character_info = f"\n{'Conversation partner' if lang == 'en' else '対話相手'}: {character_name}\n"
        if lang == "ja":
            if character_name == "吾輩":
                character_info += "（吾輩の家主は「苦沙弥先生」、嫌いな人物は「おさん」、友人に「車屋の黒」などがいる）"
        else:
            if character_name == "the Cat":
                character_info += "(The Cat's owner is 'Kushami', dislikes 'Osan', friends include 'Kuro')"

    # 現在位置付近のテキストを取得
    position_context = ""
    if pos is not None:
        from app.utils import get_text_around_position
        nearby_text = get_text_around_position(pos, context_chars=100, lang=lang)
        if nearby_text:
            label = "Current text position" if lang == "en" else "現在のテキスト位置付近"
            position_context = f"\n{label}: {nearby_text}\n"

    # LLMを使って質問を拡張・リライト
    try:
        if lang == "en":
            system_prompt = """You are an assistant that improves search queries.
Considering the conversation history, character info, and text position, rewrite the question for optimal novel text retrieval.

Key tasks:
1. **Resolve pronouns/omitted subjects**: e.g., "the owner" → "Mr. Kushami"
2. **Specify time expressions**: e.g., "recently" → "near the current text position"
3. **Resolve context references**: e.g., "What specifically?" → include the topic from conversation
4. **Search-friendly format**: Use words from the novel text, concise (10-30 words)

Output only the rewritten question."""
        else:
            system_prompt = """あなたは検索クエリを改善するアシスタントです。
会話履歴、キャラクター情報、テキスト位置を考慮して、小説本文の検索に最適な質問文にリライトしてください。

重要な処理：
1. **代名詞・省略された主語の解決**:
   - 「家主」→「苦沙弥先生」のように、キャラクター関係を考慮
   - 「それ」「あれ」→ 会話履歴から具体的な対象を特定

2. **時間表現の具体化**:
   - 「最近」→「現在のテキスト位置付近で」
   - 「その後」→「その出来事の後で」

3. **文脈参照の解決**:
   - 「具体的には？」→ 直前の話題を含めた質問に変換
   - 「なぜ？」→ 何についての「なぜ」かを明確化

4. **検索に適した形式**:
   - 本文中に登場する言葉を使う
   - 簡潔で具体的（30-60文字程度）
   - 検索キーワードを含める

出力は拡張された質問文のみを返してください。"""

        user_message = f"""{'Rewrite the following question for search:' if lang == 'en' else '以下の情報をもとに、質問を検索に適した形にリライトしてください。'}

{character_info}
{position_context}
{'Conversation history' if lang == 'en' else '会話履歴'}:
{history_context if history_context else ('(none)' if lang == 'en' else '（なし）')}

{'Current question' if lang == 'en' else '現在の質問'}: {question}

{'Rewritten question' if lang == 'en' else 'リライトされた質問'}:"""

        expanded = chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0.2,
            max_tokens=150
        )

        expanded = expanded.strip()
        expanded = expanded.strip('"').strip("'").strip("「").strip("」")

        if expanded and len(expanded) > 5:
            logger.info(f"📝 質問拡張: '{question}' → '{expanded}'")
            return expanded
        else:
            logger.warning(f"⚠️  質問拡張結果が短すぎる: '{expanded}'")
    except Exception as e:
        logger.warning(f"⚠️  質問拡張エラー: {e}")

    return question


def search_keyword(
    query: str,
    k: int,
    max_pos: int,
    lang: str = "ja"
) -> List[Dict[str, Any]]:
    """
    キーワード検索（簡易版：テキスト内のキーワードマッチング）
    """
    chunks = get_chunks_cache(lang)

    # max_pos以前のチャンクをフィルタ
    candidates = [c for c in chunks if c["start_pos"] <= max_pos]

    if not candidates:
        return []

    # クエリからキーワードを抽出
    keywords = []
    if lang == "en":
        # 英語: 単語境界ベースの抽出（3文字以上の単語）
        words = re.findall(r'\b\w{3,}\b', query.lower())
        # ストップワードを除外
        stop_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                      "can", "her", "was", "one", "our", "out", "has", "had",
                      "his", "how", "its", "may", "who", "did", "get", "let",
                      "say", "she", "too", "use", "what", "when", "where",
                      "which", "this", "that", "with", "from", "have", "been",
                      "will", "they", "were", "about", "would", "there", "their"}
        keywords = [w for w in words if w not in stop_words]
    else:
        # 日本語: 2文字以上の連続文字
        words = re.findall(r'[一-龥ぁ-んァ-ン]{2,}', query)
        keywords.extend(words)

    # スコア計算（キーワードの出現回数）
    scored_chunks = []
    for chunk in candidates:
        text = chunk.get("text", "")
        if lang == "en":
            text_lower = text.lower()
        score = 0
        matched_keywords = []

        for keyword in keywords:
            if lang == "en":
                count = text_lower.count(keyword)
            else:
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
    top_k: int = None,
    lang: str = "ja"
) -> List[Dict[str, Any]]:
    """LLMを使って検索結果を再ランキング"""
    if not chunks or len(chunks) <= 1:
        return chunks

    try:
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")[:300]
            chunk_texts.append(f"[{i+1}] {text}")

        if lang == "en":
            system_prompt = """You are a search result ranking assistant.
Reorder the chunks by relevance to the search query (most relevant first).

Focus on:
- Chunks most related to the query intent go first
- Less relevant chunks go last
- Return only comma-separated numbers (e.g.: 3,1,2,4)"""
        else:
            system_prompt = """あなたは検索結果をランキングするアシスタントです。
検索クエリに関連性が高い順に、チャンクの番号を並び替えてください。

以下の点を重視してください：
- 検索クエリの意図に最も関連するチャンクを上位に
- 関連性の低いチャンクは下位に
- 番号のみをカンマ区切りで返す（例: 3,1,2,4）"""

        if lang == "en":
            user_message = f"""Search query: {query}

Search results:
{chr(10).join(chunk_texts)}

Reorder the results by relevance. Return only comma-separated numbers (e.g.: 3,1,2,4)."""
        else:
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

        numbers = re.findall(r'\d+', result)
        if numbers:
            indices = [int(n) - 1 for n in numbers if 1 <= int(n) <= len(chunks)]
            if len(indices) == len(chunks):
                seen = set()
                reranked = []
                for idx in indices:
                    if idx not in seen:
                        seen.add(idx)
                        reranked.append(chunks[idx])
                for i, chunk in enumerate(chunks):
                    if i not in seen:
                        reranked.append(chunk)

                logger.info(f"🔄 再ランキング: {len(reranked)} 件")
                if top_k:
                    return reranked[:top_k]
                return reranked
    except Exception as e:
        logger.warning(f"⚠️  再ランキングエラー: {e}")

    if top_k:
        return chunks[:top_k]
    return chunks


def retrieve_chunks(
    question: str,
    pos: int,
    k: int = 8,
    window: int = 3,
    history: List[Dict[str, str]] = None,
    character_name: str = None,
    use_query_expansion: bool = True,
    use_hybrid_search: bool = True,
    use_reranking: bool = True,
    lang: str = "ja"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    質問と位置に基づいてチャンクを取得（改善版）
    """
    # 現在のシーンを特定
    current_scene = find_current_scene(pos, lang=lang)
    logger.info(f"📍 現在位置: pos={pos}, scene={current_scene}")

    # 近傍ウィンドウ取得
    nearby = retrieve_nearby(current_scene, window=window, lang=lang)
    logger.info(f"📦 近傍チャンク: {len(nearby)} 件")

    # 質問の拡張・リライト
    search_query = question
    if use_query_expansion:
        search_query = expand_query_with_history(
            question=question,
            history=history,
            character_name=character_name,
            pos=pos,
            lang=lang
        )

    logger.info(f"🔍 検索クエリ: '{search_query}'")

    # セマンティック検索
    query_vec = embed(search_query)
    semantic_results = search_semantic_qdrant(query_vec, k=k*2, max_pos=pos, lang=lang)
    method = "qdrant"

    if semantic_results is None or len(semantic_results) == 0:
        logger.info("⚠️  Qdrant検索結果が0件のため、フォールバック検索に切り替えます")
        semantic_results = search_semantic_fallback(query_vec, k=k*2, max_pos=pos, lang=lang)
        method = "fallback"

    # ハイブリッド検索
    keyword_results = []
    if use_hybrid_search:
        keyword_results = search_keyword(search_query, k=k, max_pos=pos, lang=lang)
        logger.info(f"🔑 キーワード検索: {len(keyword_results)} 件")

    # 近傍 + セマンティック + キーワード結果を統合（重複除去）
    seen_scenes = set()
    combined = []
    scene_to_chunk = {}

    # 近傍チャンクの類似度を計算
    nearby_with_scores = []
    for chunk in nearby:
        if chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos:
            chunk_text = chunk.get("text", "")
            chunk_vec = embed(chunk_text)
            similarity = cosine_similarity([query_vec], [chunk_vec])[0][0]
            chunk["nearby_similarity"] = similarity
            nearby_with_scores.append((chunk, similarity))

    nearby_with_scores.sort(key=lambda x: x[1], reverse=True)

    nearby_threshold = 0.7
    nearby_added = 0
    for i, (chunk, similarity) in enumerate(nearby_with_scores):
        if i < 2 or similarity >= nearby_threshold:
            scene_idx = chunk["scene_index"]
            if scene_idx not in seen_scenes:
                seen_scenes.add(scene_idx)
                chunk["source"] = "nearby"
                chunk["score"] = similarity + 0.2
                scene_to_chunk[scene_idx] = chunk
                nearby_added += 1
                logger.info(f"   📌 近傍チャンク追加 (similarity={similarity:.3f}): scene={scene_idx}")
        else:
            logger.info(f"   ⏭️  近傍チャンクをスキップ (similarity={similarity:.3f} < {nearby_threshold}): scene={scene_idx}")

    logger.info(f"📌 近傍チャンク追加: {nearby_added}/{len(nearby)} 件")

    # セマンティック検索結果を追加
    for chunk in semantic_results:
        scene_idx = chunk["scene_index"]
        if scene_idx not in seen_scenes:
            seen_scenes.add(scene_idx)
            chunk["source"] = "semantic"
            scene_to_chunk[scene_idx] = chunk
        else:
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
            existing = scene_to_chunk[scene_idx]
            existing["score"] = existing.get("score", 0) + chunk.get("score", 0) * 0.3
            existing["source"] = existing.get("source", "") + "+keyword"

    combined = list(scene_to_chunk.values())

    # スコアでソート
    combined.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 再ランキング
    if use_reranking and len(combined) > 2:
        combined = rerank_chunks(search_query, combined, top_k=k*2, lang=lang)

    # 上位 k 件に制限
    combined = combined[:k]

    logger.info(f"✅ 最終取得: {len(combined)} 件 (method={method})")
    return combined, method


def retrieve_relevant_events(
    current_scene: int,
    chunks: List[Dict[str, Any]],
    lang: str = "ja"
) -> List[Dict[str, Any]]:
    """
    現在のシーンと取得したチャンクに関連するイベントを取得
    """
    events = get_events_cache(lang)

    scene_indices = {c["scene_index"] for c in chunks}
    min_scene = min(scene_indices) if scene_indices else current_scene
    max_scene = current_scene

    relevant = []
    for event in events:
        first = event.get("first_scene")
        last = event.get("last_scene")

        if first is None or last is None:
            continue

        if last <= max_scene and first <= max_scene:
            if first <= max_scene and last >= min_scene:
                relevant.append(event)

    logger.info(f"📅 関連イベント: {len(relevant)} 件")
    return relevant[:5]


def wait_situation_ready(pos: int, timeout: float = 30, lang: str = "ja") -> bool:
    """状況要約の完了をロングポーリングで待機する"""
    cache_key = (lang, pos)

    if cache_key in _situation_cache:
        return True

    with _situation_locks_guard:
        event = _situation_locks.get(cache_key)

    if event is not None:
        event.wait(timeout=timeout)
        return cache_key in _situation_cache

    return cache_key in _situation_cache


def get_scene_situation(pos: int, window: int = 3, lang: str = "ja") -> str:
    """
    指定された位置付近の場面全体の状況を取得し、要約して返す
    """
    cache_key = (lang, pos)
    if cache_key in _situation_cache:
        logger.info(f"✓ 状況要約をキャッシュから取得 (pos={pos}, lang={lang})")
        return _situation_cache[cache_key]

    with _situation_locks_guard:
        if cache_key in _situation_cache:
            return _situation_cache[cache_key]
        if cache_key in _situation_locks:
            event = _situation_locks[cache_key]
        else:
            event = threading.Event()
            _situation_locks[cache_key] = event
            event = None

    if event is not None:
        logger.info(f"⏳ 状況要約を別リクエストが処理中、待機します (pos={pos}, lang={lang})")
        event.wait()
        if cache_key in _situation_cache:
            return _situation_cache[cache_key]

    chunks = get_chunks_cache(lang)
    current_scene = find_current_scene(pos, lang=lang)

    nearby_chunks = []
    for chunk in chunks:
        if (chunk["start_pos"] <= pos <= chunk["end_pos"] or chunk["end_pos"] <= pos):
            if abs(chunk["scene_index"] - current_scene) <= window:
                nearby_chunks.append(chunk)

    nearby_chunks.sort(key=lambda c: abs(c["start_pos"] - pos))

    scene_characters = set()
    for chunk in nearby_chunks[:3]:
        for char in chunk.get("characters", []):
            scene_characters.add(char)

    relevant_texts = []
    for chunk in nearby_chunks[:3]:
        text = chunk.get("text", "").strip()
        if text:
            relevant_texts.append(text)

    if not relevant_texts:
        logger.warning(f"⚠️  関連テキストが見つかりません。現在位置付近のテキストを使用 (pos={pos})")
        for chunk in chunks:
            if chunk["start_pos"] <= pos <= chunk["end_pos"]:
                text = chunk.get("text", "").strip()
                if text:
                    relevant_texts.append(text[:500])
                    for char in chunk.get("characters", []):
                        scene_characters.add(char)
                    break

    combined_text = "\n\n".join(relevant_texts) if relevant_texts else ""

    if lang == "en":
        characters_str = ", ".join(sorted(scene_characters)) if scene_characters else "(unknown)"

        system_prompt = """You are an assistant that summarizes scenes from a novel concisely.

Read the text and output in the following format:

[Scene] (location, time of day, situation in one sentence)
- Character name: What they are doing, their state (1-2 sentences)

Rules:
- The [Scene] line should be under 80 characters, stating location and situation briefly
- Only describe characters that appear in the character list
- Keep each character's description concise (1-2 sentences), including actions and mental state
- Do not infer information not explicitly in the text"""

        user_message = f"""Summarize the following scene from the novel "I Am a Cat" (Natsume Sōseki).

Characters: {characters_str}

Text:
{combined_text}"""
    else:
        characters_str = "、".join(sorted(scene_characters)) if scene_characters else "（不明）"

        system_prompt = """あなたは小説の場面を簡潔に整理するアシスタントです。

テキストを読み、以下の形式で出力してください：

【場面】（場所・時間帯・状況を1文で）
- 人物名: その人物が何をしているか、どのような状態か（1-2文）

ルール：
- 「【場面】」行は50文字以内で、場所と状況を端的に述べる
- 登場人物リストに含まれる人物のみ記述する
- 各人物の説明は簡潔に（各30-60文字）、行動と心理状態を含める
- テキストに明示されていない情報は推測しない"""

        user_message = f"""以下のテキスト（小説「吾輩は猫である」の一部）から、場面を整理してください。

登場人物: {characters_str}

テキスト:
{combined_text}"""

    logger.info("=" * 60)
    logger.info(f"📝 状況要約プロンプト (pos={pos}, lang={lang}):")
    logger.info(f"   登場人物: {characters_str}")
    logger.info(f"   テキスト長: {len(combined_text)} 文字")
    logger.info("=" * 60)

    try:
        summary = chat(
            messages=[{"role": "user", "content": user_message}],
            system=system_prompt,
            temperature=0.2,
            max_tokens=400
        )
        result = summary.strip()
        if not result:
            result = "The situation near the current position is unknown" if lang == "en" else "現在位置付近の状況は不明"
        logger.info(f"✓ 状況要約結果 (pos={pos}): {result[:80]}...")

        _situation_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"❌ 要約エラー (pos={pos}): {e}")
        if relevant_texts:
            fallback_text = relevant_texts[0]
            if len(fallback_text) > 200:
                fallback_text = fallback_text[:200] + "..."
            _situation_cache[cache_key] = fallback_text
            return fallback_text
        else:
            default_msg = "The situation near the current position is unknown" if lang == "en" else "現在位置付近の状況は不明"
            _situation_cache[cache_key] = default_msg
            return default_msg
    finally:
        with _situation_locks_guard:
            ev = _situation_locks.pop(cache_key, None)
        if ev is not None:
            ev.set()


def get_appeared_characters(pos: int, lang: str = "ja") -> List[Dict[str, Any]]:
    """
    pos以前に登場済みの character.json 登録キャラクターをリストアップする
    """
    chunks = get_chunks_cache(lang)
    personas = get_personas_cache(lang)

    # character.json の名前 → チャンク内での名前（逆引き）を構築
    chunk_name_to_persona_name: Dict[str, str] = {}
    for persona_name in personas:
        chunk_name_to_persona_name[persona_name] = persona_name
        for alias in _CHARACTER_NAME_ALIASES.get(persona_name, []):
            chunk_name_to_persona_name[alias] = persona_name

    # pos以前の全チャンクから characters をunion集合
    appeared_chunk_names: set = set()
    for chunk in chunks:
        if chunk["end_pos"] <= pos or (chunk["start_pos"] <= pos <= chunk["end_pos"]):
            for char in chunk.get("characters", []):
                appeared_chunk_names.add(char)

    # character.json に登録されている名前のみフィルタ
    appeared_persona_names: set = set()
    for chunk_name in appeared_chunk_names:
        persona_name = chunk_name_to_persona_name.get(chunk_name)
        if persona_name:
            appeared_persona_names.add(persona_name)

    # 結果を構築（id順にソート）
    result = []
    for persona_name in appeared_persona_names:
        persona = personas[persona_name]
        result.append({
            "id": persona["id"],
            "name": persona["name"],
            "first_message": persona.get("first-message", ""),
        })

    result.sort(key=lambda x: x["id"])
    logger.info(f"📋 登場済みキャラクター (pos={pos}, lang={lang}): {[c['name'] for c in result]}")
    return result
