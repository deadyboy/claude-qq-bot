"""Retrieval and debug helpers for owner-style generation."""

from .qce_io import *
from .phrases import *
from .turns import *
from .taxonomy import *
from .reports import *


def infer_scene_label(
    latest_message: str,
    *,
    chat_type: str = "private",
    current_context: str | Sequence[Dict[str, Any]] | None = None,
) -> str:
    context_text = _normalize_current_context(current_context, latest_message)
    latest_text = str(latest_message or "").strip()
    latest_features = _features_for_text(latest_text)
    context_features = _features_for_text(context_text or latest_text)
    if _contains_any(context_text, FORMAL_WORK_HINTS) or context_features["has_url"]:
        return "formal_or_worklike"
    if chat_type == "group":
        if "@" in context_text or "[reply]" in context_text or "回复" in context_text:
            return "group_mentioned_or_reply"
        return "group_interjection"
    if (
        _contains_any(latest_text, EMOTIONAL_HINTS)
        or latest_features["has_exclamation"]
        or latest_features["emoji_count"] > 0
        or (_contains_any(context_text, EMOTIONAL_HINTS) and len(latest_text) <= 80)
    ):
        return "private_emotional"
    if _contains_any(latest_text, EXPLAIN_HINTS):
        return "private_long_explain"
    if (
        len(latest_text) >= 80
        or latest_features["line_count"] > 1
        or (context_features["line_count"] >= 3 and len(context_text) >= 160)
    ):
        return "private_long_explain"
    return "private_short_casual"

def _query_embedding_index_for_retrieval(
    query: str,
    *,
    run_dir: str | Path,
    limit: int,
) -> Dict[str, Any]:
    """Query the optional local embedding index without making it a hard runtime dependency."""
    try:
        from .embedding import query_stage5b_embedding_index

        return query_stage5b_embedding_index(
            query,
            run_dir=run_dir,
            limit=limit,
            include_text=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "message": f"embedding 检索不可用：{type(e).__name__}",
            "error_type": type(e).__name__,
        }

def _retrieval_sort_key(
    item: Dict[str, Any],
    *,
    preferred_source_file_id: str | None = None,
    preferred_chat_type: str | None = None,
) -> tuple:
    source_file_id = str(item.get("source_file_id") or "")
    chat_type = str(item.get("chat_type") or "")
    return (
        not bool(preferred_source_file_id and source_file_id == preferred_source_file_id),
        not bool(preferred_chat_type and chat_type == preferred_chat_type),
        -_safe_float(item.get("similarity")),
        -_safe_int(item.get("quality_score")),
        str(item.get("sample_id") or item.get("pair_id") or ""),
    )

def retrieve_dialogue_pair_samples(
    latest_message: str,
    *,
    current_context: str | Sequence[Dict[str, Any]] | None = None,
    run_dir: str | Path | None = None,
    chat_type: str = "private",
    target_id: str | int | None = None,
    scene_label: str | None = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Retrieval-first raw dialogue-pair lookup for offline draft generation."""
    query_text = _normalize_current_context(current_context, latest_message)
    if not query_text:
        return {"ok": False, "message": "latest_message 不能为空。"}
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return {"ok": False, "message": "还没有找到 Stage 5B 离线蒸馏结果。"}
    target_mapping = find_source_for_target(target_id, chat_type=chat_type, run_dir=run_path) if target_id else {"matched": False}
    preferred_source = str(target_mapping.get("source_file_id") or "") if target_mapping.get("matched") else ""
    inferred_scene = scene_label or infer_scene_label(latest_message, chat_type=chat_type, current_context=current_context)
    pairs = _load_dialogue_pairs(run_path)
    query_ngrams = _text_ngrams(query_text)
    query_keywords = _keyword_tokens(query_text)
    query_intent = detect_message_intent(query_text)
    ranked = []
    for pair in pairs:
        if chat_type and pair.get("chat_type") != chat_type:
            continue
        context_turns = pair.get("context") or []
        pair_context = _turns_text(context_turns, roles={"other"}) or _turns_text(context_turns)
        if not pair_context:
            continue
        overlap = _jaccard(query_ngrams, _text_ngrams(pair_context))
        keyword_overlap = _jaccard(query_keywords, _keyword_tokens(pair_context))
        intent_bonus = _intent_similarity(query_intent, detect_message_intent(pair_context))
        source_bonus = 0.35 if preferred_source and pair.get("source_file_id") == preferred_source else 0.0
        scene_bonus = 0.22 if inferred_scene and pair.get("scene_label") == inferred_scene else 0.0
        fallback_bonus = 0.04 if not preferred_source else 0.0
        total = round(
            overlap * 0.5
            + keyword_overlap * 0.3
            + intent_bonus
            + source_bonus
            + scene_bonus
            + (int(pair.get("score") or 0) / 1200)
            + fallback_bonus,
            4,
        )
        if total <= 0 and not source_bonus and not scene_bonus:
            continue
        ranked.append({
            "pair_id": pair.get("pair_id"),
            "source_file_id": pair.get("source_file_id"),
            "relationship_id": pair.get("relationship_id"),
            "chat_type": pair.get("chat_type"),
            "scene_label": pair.get("scene_label"),
            "score": int(pair.get("score") or 0),
            "retrieval_score": total,
            "context_overlap": round(overlap, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "same_relationship": bool(preferred_source and pair.get("source_file_id") == preferred_source),
            "same_scene": bool(inferred_scene and pair.get("scene_label") == inferred_scene),
            "taxonomy": pair.get("taxonomy") or {},
            "context": context_turns[-12:],
            "target": pair.get("target") or {},
        })
    ranked.sort(
        key=lambda item: (
            not item["same_relationship"],
            not item["same_scene"],
            -float(item["retrieval_score"]),
            -int(item["score"]),
            str(item["pair_id"]),
        )
    )
    selected = ranked[: max(0, min(8, int(limit)))]
    return {
        "ok": True,
        "run_id": run_path.name,
        "scene_label": inferred_scene,
        "target_mapping": {
            "matched": bool(target_mapping.get("matched")),
            "source_file_id": preferred_source,
            "chat_type": target_mapping.get("chat_type") if target_mapping.get("matched") else chat_type,
        },
        "result_count": len(selected),
        "results": selected,
        "raw_text_policy": "Returns local raw dialogue-pair snippets for offline draft generation only.",
    }

def _source_target_id(source: Dict[str, Any]) -> str:
    """Best-effort target id from QCE export filenames/metadata."""
    text = " ".join(
        str(source.get(key) or "")
        for key in ("relative_path", "file_name", "name")
    )
    chat_type = str(source.get("chat_type") or "")
    patterns = []
    if chat_type == "private":
        patterns = [r"private_(\d{5,12})", r"friend_[^\\/_]*_(?:\d+_)?private_(\d{5,12})"]
    elif chat_type == "group":
        patterns = [r"group_(\d{5,12})", r"troop_(\d{5,12})"]
    else:
        patterns = [r"(?:private|group|troop)_(\d{5,12})"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""

def find_source_for_target(
    target_id: str | int,
    *,
    chat_type: str | None = None,
    run_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Map a live QQ user/group id to a local source_file_id without returning names or text."""
    target = str(target_id).strip()
    if not target:
        return {"matched": False}
    try:
        run_path, catalog, _ = _resolve_run_paths(run_dir)
    except Exception as e:
        return {"matched": False, "error": type(e).__name__}

    expected_chat_type = str(chat_type or "").strip()
    for source in catalog.get("sources") or []:
        if not isinstance(source, dict):
            continue
        source_chat_type = str(source.get("chat_type") or "")
        if expected_chat_type and source_chat_type != expected_chat_type:
            continue
        if _source_target_id(source) != target:
            continue
        return {
            "matched": True,
            "run_id": run_path.name,
            "source_file_id": str(source.get("source_file_id") or ""),
            "chat_type": source_chat_type,
            "raw_identifier_policy": "QQ/group ids are used locally for mapping but are not included in prompts.",
        }
    return {"matched": False, "run_id": run_path.name}

def retrieve_similar_style_samples(
    query: str,
    *,
    run_dir: str | Path | None = None,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    preferred_source_file_id: str | None = None,
    preferred_chat_type: str | None = None,
    use_embedding: bool = True,
) -> Dict[str, Any]:
    """Retrieve similar indexed samples using local raw QCE text transiently.

    Return value intentionally contains no raw historical text.
    """
    query_text = query.strip()
    if not query_text:
        return {"ok": False, "message": "用法：/风格 检索 <当前对方消息>"}
    if len(query_text) > 500:
        return {"ok": False, "message": "检索文本太长，先支持 500 字以内。"}

    run_path, catalog, samples = _resolve_run_paths(run_dir)
    query_features = _features_for_text(query_text)
    query_ngrams = _text_ngrams(query_text)
    query_keywords = _keyword_tokens(query_text)
    query_intent = query_features.get("intent") or detect_message_intent(query_text)
    cache: Dict[str, List[Dict[str, Any]]] = {}
    results = []

    for sample in samples:
        source_file_id = str(sample.get("source_file_id") or "")
        messages = _load_source_messages(catalog, source_file_id, cache)
        if not messages:
            continue
        context_texts = []
        for ref in (sample.get("context") or {}).get("messages") or []:
            if ref.get("role") != "other":
                continue
            try:
                record_index = int(ref.get("record_index"))
            except (TypeError, ValueError):
                continue
            if record_index < 0 or record_index >= len(messages):
                continue
            text = _message_text(messages[record_index])
            if _is_useful_text(text):
                context_texts.append(text)
        if not context_texts:
            continue
        context_text = "\n".join(context_texts[-3:])
        overlap = _jaccard(query_ngrams, _text_ngrams(context_text))
        keyword_overlap = _jaccard(query_keywords, _keyword_tokens(context_text))
        context_intent = detect_message_intent(context_text)
        intent_bonus = _intent_similarity(query_intent, context_intent)
        if overlap <= 0 and keyword_overlap <= 0 and intent_bonus < 0.08:
            continue
        if query_intent.get("game_invitation") and overlap <= 0 and keyword_overlap <= 0:
            continue
        reply = sample.get("reply") or {}
        feature_bonus = 0.0
        if bool(query_features["has_question"]) == bool(reply.get("features", {}).get("has_question")):
            feature_bonus += 0.04
        if query_features["length_bucket"] == reply.get("length_bucket"):
            feature_bonus += 0.03
        chat_type_bonus = 0.0
        if preferred_chat_type:
            if sample.get("chat_type") == preferred_chat_type:
                chat_type_bonus += 0.12
            else:
                chat_type_bonus -= 0.08
        source_bonus = 0.0
        if preferred_source_file_id and source_file_id == preferred_source_file_id:
            source_bonus += 0.1
        quality_bonus = int(sample.get("score") or 0) / 1000
        total_score = round(
            (overlap * 0.58)
            + (keyword_overlap * 0.28)
            + intent_bonus
            + feature_bonus
            + chat_type_bonus
            + source_bonus
            + quality_bonus,
            4,
        )
        pair_id = str(sample.get("pair_id") or sample.get("sample_id") or "")
        results.append({
            "sample_id": sample.get("sample_id"),
            "pair_id": pair_id,
            "source_file_id": source_file_id,
            "chat_type": sample.get("chat_type"),
            "similarity": total_score,
            "rule_similarity": total_score,
            "embedding_similarity": 0.0,
            "retrieval_source": "rules",
            "context_overlap": round(overlap, 4),
            "keyword_overlap": round(keyword_overlap, 4),
            "intent_bonus": round(intent_bonus, 4),
            "chat_type_bonus": round(chat_type_bonus, 4),
            "source_bonus": round(source_bonus, 4),
            "quality_score": int(sample.get("score") or 0),
            "reply_length_bucket": reply.get("length_bucket"),
            "reply_char_length": reply.get("char_length"),
            "context_count": (sample.get("context") or {}).get("count"),
            "time_bucket": reply.get("time_bucket"),
            "score_reasons": sample.get("score_reasons") or [],
        })

    embedding_status = {"enabled": False, "ok": False, "message": "未启用 embedding 检索。"}
    if use_embedding:
        embedding_limit = max(20, min(100, int(limit or DEFAULT_RETRIEVAL_LIMIT) * 8))
        embedding_result = _query_embedding_index_for_retrieval(
            query_text,
            run_dir=run_path,
            limit=embedding_limit,
        )
        embedding_status = {
            "enabled": True,
            "ok": bool(embedding_result.get("ok")),
            "model": embedding_result.get("model"),
            "result_count": int(embedding_result.get("result_count") or 0),
            "message": embedding_result.get("message") or "",
        }
        if embedding_result.get("ok"):
            by_pair_id = {
                str(item.get("pair_id") or item.get("sample_id") or ""): item
                for item in results
                if str(item.get("pair_id") or item.get("sample_id") or "").strip()
            }
            samples_by_pair_id = {
                str(sample.get("pair_id") or sample.get("sample_id") or ""): sample
                for sample in samples
                if isinstance(sample, dict)
            }
            for rank, hit in enumerate(embedding_result.get("results") or [], start=1):
                metadata = hit.get("metadata") or {}
                pair_id = str(hit.get("pair_id") or metadata.get("pair_id") or "").strip()
                if not pair_id:
                    continue
                embedding_similarity = _safe_float(hit.get("embedding_similarity"))
                if embedding_similarity <= 0:
                    continue
                sample = samples_by_pair_id.get(pair_id)
                source_file_id = str(metadata.get("source_file_id") or (sample or {}).get("source_file_id") or "")
                hit_chat_type = str(metadata.get("chat_type") or (sample or {}).get("chat_type") or "")
                source_bonus = 0.08 if preferred_source_file_id and source_file_id == preferred_source_file_id else 0.0
                chat_type_bonus = 0.06 if preferred_chat_type and hit_chat_type == preferred_chat_type else 0.0
                if preferred_chat_type and hit_chat_type and hit_chat_type != preferred_chat_type:
                    chat_type_bonus -= 0.04

                existing = by_pair_id.get(pair_id)
                if existing:
                    rule_similarity = _safe_float(existing.get("rule_similarity") or existing.get("similarity"))
                    hybrid = max(
                        rule_similarity,
                        round(rule_similarity * 0.78 + embedding_similarity * 0.38 + source_bonus + chat_type_bonus, 4),
                    )
                    existing.update({
                        "similarity": hybrid,
                        "embedding_similarity": round(embedding_similarity, 4),
                        "embedding_distance": hit.get("distance"),
                        "embedding_rank": rank,
                        "retrieval_source": "hybrid",
                    })
                    continue

                quality_score = _safe_int(metadata.get("score") or (sample or {}).get("score"))
                reply = (sample or {}).get("reply") or {}
                context = (sample or {}).get("context") or {}
                embedding_only_score = round(
                    embedding_similarity * 0.78
                    + source_bonus
                    + chat_type_bonus
                    + (quality_score / 1500),
                    4,
                )
                item = {
                    "sample_id": (sample or {}).get("sample_id") or pair_id,
                    "pair_id": pair_id,
                    "source_file_id": source_file_id,
                    "chat_type": hit_chat_type,
                    "similarity": embedding_only_score,
                    "rule_similarity": 0.0,
                    "embedding_similarity": round(embedding_similarity, 4),
                    "embedding_distance": hit.get("distance"),
                    "embedding_rank": rank,
                    "retrieval_source": "embedding",
                    "context_overlap": 0.0,
                    "keyword_overlap": 0.0,
                    "intent_bonus": 0.0,
                    "chat_type_bonus": round(chat_type_bonus, 4),
                    "source_bonus": round(source_bonus, 4),
                    "quality_score": quality_score,
                    "reply_length_bucket": reply.get("length_bucket") or metadata.get("length_bucket"),
                    "reply_char_length": _safe_int(reply.get("char_length") or metadata.get("target_char_length")),
                    "context_count": context.get("count") or metadata.get("context_turn_count"),
                    "time_bucket": reply.get("time_bucket"),
                    "score_reasons": ["embedding_hit"],
                    "scene_label": metadata.get("scene_label"),
                    "taxonomy": {
                        "scope": metadata.get("scope"),
                        "grounding_type": metadata.get("grounding_type"),
                        "learning_value": metadata.get("learning_value"),
                    },
                }
                by_pair_id[pair_id] = item
                results.append(item)

    results.sort(
        key=lambda item: _retrieval_sort_key(
            item,
            preferred_source_file_id=preferred_source_file_id,
            preferred_chat_type=preferred_chat_type,
        )
    )
    return {
        "ok": True,
        "run_id": run_path.name,
        "retrieval_strategy": "hybrid_rules_embedding" if embedding_status.get("ok") else "rules_only",
        "embedding_status": embedding_status,
        "raw_text_policy": "Historical text was read transiently for rule scoring only; embedding index stores no raw text.",
        "query_features": {
            "char_length": query_features["char_length"],
            "length_bucket": query_features["length_bucket"],
            "has_question": query_features["has_question"],
            "has_exclamation": query_features["has_exclamation"],
            "has_ellipsis": query_features["has_ellipsis"],
            "intent": query_intent,
        },
        "result_count": min(len(results), max(0, int(limit))),
        "results": results[: max(0, int(limit))],
    }

def _select_relationship_profiles(
    relationship_data: Dict[str, Any],
    retrieval_results: Sequence[Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    profiles = relationship_data.get("profiles") or []
    by_source = {
        str(item.get("source_file_id") or ""): item
        for item in profiles
        if isinstance(item, dict)
    }
    selected = []
    seen = set()
    for result in retrieval_results:
        source_file_id = str(result.get("source_file_id") or "")
        item = by_source.get(source_file_id)
        if not item or source_file_id in seen:
            continue
        seen.add(source_file_id)
        selected.append(item)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in profiles:
            source_file_id = str(item.get("source_file_id") or "")
            if not source_file_id or source_file_id in seen:
                continue
            selected.append(item)
            seen.add(source_file_id)
            if len(selected) >= limit:
                break

    sanitized = []
    for item in selected:
        sanitized.append({
            "source_file_id": item.get("source_file_id"),
            "chat_type": item.get("chat_type"),
            "owner_text_messages": _safe_int(item.get("owner_text_messages")),
            "candidate_samples": _safe_int(item.get("candidate_samples")),
            "avg_length": _safe_float(item.get("avg_length")),
            "median_length": _safe_int(item.get("median_length")),
            "labels": list(item.get("labels") or [])[:8],
            "length_buckets": dict(item.get("length_buckets") or {}),
            "element_types": dict(item.get("element_types") or {}),
        })
    return sanitized

def _scene_match_score(
    scene: Dict[str, Any],
    *,
    preferred_chat_type: str,
    preferred_length_bucket: str,
    query_features: Dict[str, Any],
) -> float:
    score = _safe_int(scene.get("sample_count")) * 0.01 + _safe_float(scene.get("avg_score")) * 0.01
    scene_id = str(scene.get("scene_id") or "")
    chat_types = scene.get("chat_types") or {}
    length_buckets = scene.get("length_buckets") or {}
    if preferred_chat_type and _safe_int(chat_types.get(preferred_chat_type)) > 0:
        score += 2.0
    if preferred_length_bucket and _safe_int(length_buckets.get(preferred_length_bucket)) > 0:
        score += 1.5
    if query_features.get("has_question") and "question_reply" in scene_id:
        score += 0.4
    if query_features.get("has_exclamation") and "emotional_reply" in scene_id:
        score += 0.4
    if query_features.get("has_ellipsis") and "pause_reply" in scene_id:
        score += 0.4
    return score

def _select_scene_profiles(
    scene_data: Dict[str, Any],
    retrieval_results: Sequence[Dict[str, Any]],
    query_features: Dict[str, Any],
    *,
    chat_type: str | None = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    profiles = [
        item for item in (scene_data.get("profiles") or [])
        if isinstance(item, dict)
    ]
    if not profiles:
        return []

    top_result = retrieval_results[0] if retrieval_results else {}
    preferred_chat_type = str(chat_type or top_result.get("chat_type") or "private")
    preferred_length_bucket = str(top_result.get("reply_length_bucket") or query_features.get("length_bucket") or "")
    ranked = sorted(
        profiles,
        key=lambda item: _scene_match_score(
            item,
            preferred_chat_type=preferred_chat_type,
            preferred_length_bucket=preferred_length_bucket,
            query_features=query_features,
        ),
        reverse=True,
    )

    selected = []
    for item in ranked[:limit]:
        selected.append({
            "scene_id": item.get("scene_id"),
            "sample_count": _safe_int(item.get("sample_count")),
            "chat_types": dict(item.get("chat_types") or {}),
            "length_buckets": dict(item.get("length_buckets") or {}),
            "avg_reply_length": _safe_float(item.get("avg_reply_length")),
            "quick_reply_ratio": _safe_float(item.get("quick_reply_ratio")),
            "question_ratio": _safe_float(item.get("question_ratio")),
            "exclamation_ratio": _safe_float(item.get("exclamation_ratio")),
            "ellipsis_ratio": _safe_float(item.get("ellipsis_ratio")),
            "recommended_style": str(item.get("recommended_style") or ""),
        })
    return selected

def _derive_generation_guidance(
    query_features: Dict[str, Any],
    retrieval_results: Sequence[Dict[str, Any]],
    scene_profiles: Sequence[Dict[str, Any]],
    relationship_profiles: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    strong_results = [
        item for item in retrieval_results
        if _safe_float(item.get("similarity")) >= 0.32
        or _safe_float(item.get("context_overlap")) >= 0.12
        or _safe_float(item.get("keyword_overlap")) >= 0.12
    ]
    length_source = strong_results or retrieval_results
    reply_lengths = [
        _safe_int(item.get("reply_char_length"))
        for item in length_source
        if _safe_int(item.get("reply_char_length")) > 0
    ]
    if not reply_lengths:
        reply_lengths = [
            round(_safe_float(item.get("avg_reply_length")))
            for item in scene_profiles
            if _safe_float(item.get("avg_reply_length")) > 0
        ]
    if reply_lengths:
        target_length = round(sum(reply_lengths[:5]) / min(5, len(reply_lengths)), 1)
    else:
        target_length = max(6, min(30, _safe_int(query_features.get("char_length"), 12)))

    labels = Counter()
    for item in relationship_profiles:
        labels.update(item.get("labels") or [])

    intent = query_features.get("intent") or {}
    if intent.get("game_invitation"):
        target_length = min(target_length, 12)
        length_instruction = "优先 3-12 字短回复，像自然邀约接话"
    elif intent.get("invitation"):
        target_length = min(target_length, 14)
        length_instruction = "优先 6-14 字短回复，不直接承诺"
    elif target_length <= 8:
        length_instruction = "优先 3-8 字短回复"
    elif target_length <= 18:
        length_instruction = "优先 8-18 字中短回复"
    elif target_length <= 40:
        length_instruction = "优先 15-40 字，给一点上下文"
    else:
        length_instruction = "可以稍微展开，但保持口语化"

    if intent.get("game_invitation"):
        stance = "对方在发游戏邀约时，优先短句接话；不知道主人是否能玩时不要替主人承诺上线。"
    elif intent.get("availability_query") or intent.get("reality_state_query"):
        stance = "对方在问主人现实状态或可用性时，不要替主人确认忙闲、位置或进度；优先自然追问或模糊过渡。"
    elif intent.get("help_request"):
        stance = "对方在求助时，先接住请求；信息不足时让对方发具体内容或说明卡在哪一步。"
    elif intent.get("invitation"):
        stance = "对方在邀约时，不要直接承诺主人会去；优先用模糊过渡或确认细节。"
    elif intent.get("task_request"):
        stance = "对方在安排任务时，先确认收到或询问关键信息，不要编造已经完成。"
    elif query_features.get("has_question"):
        stance = "对方在提问时，先自然回应；涉及现实状态或承诺时不要替主人确认。"
    elif query_features.get("has_exclamation"):
        stance = "对方情绪较强时，先短句接住语气，再给轻量回应。"
    else:
        stance = "普通消息优先自然短句，不要过度解释。"

    return {
        "target_reply_length": target_length,
        "length_instruction": length_instruction,
        "stance_instruction": stance,
        "dominant_relationship_labels": dict(labels.most_common(8)),
        "draft_policy": "draft_only_no_auto_send",
        "reality_policy": "do_not_invent_owner_state_or_commitments",
        "intent_summary": {
            "question_type": intent.get("question_type") or "unknown",
            "availability_query": bool(intent.get("availability_query")),
            "help_request": bool(intent.get("help_request")),
            "invitation": bool(intent.get("invitation")),
            "game_invitation": bool(intent.get("game_invitation")),
            "task_request": bool(intent.get("task_request")),
            "image_reference": bool(intent.get("image_reference")),
            "reality_state_query": bool(intent.get("reality_state_query")),
        },
    }

def format_style_debug_report(
    query: str,
    *,
    run_dir: str | Path | None = None,
    chat_type: str | None = None,
    target_id: str | int | None = None,
    limit: int = 5,
) -> str:
    """Build an owner-only debug report with visible raw historical snippets."""
    from .generation import build_style_generation_context

    context = build_style_generation_context(
        query,
        run_dir=run_dir,
        limit=limit,
        chat_type=chat_type,
        target_id=target_id,
        include_raw_fewshot=True,
        raw_fewshot_limit=min(5, max(1, int(limit))),
    )
    if not context.get("ok"):
        return str(context.get("message") or "风格调试失败。")

    features = context.get("query_features") or {}
    intent = features.get("intent") or {}
    mapping = context.get("target_mapping") or {}
    guidance = context.get("guidance") or {}
    lines = [
        "Stage 5B-RAG 风格调试：",
        f"- run_id：{context.get('run_id')}",
        f"- readiness：{context.get('readiness')}",
        f"- 检索策略：{context.get('retrieval_strategy') or 'rules_only'}",
        f"- 当前对象映射：{'已匹配 ' + str(mapping.get('source_file_id')) if mapping.get('matched') else '未匹配'} ({mapping.get('chat_type') or chat_type or 'unknown'})",
        f"- 问句：{features.get('has_question')}；类型：{intent.get('question_type')}",
        (
            "- 意图："
            f"可用性/现实={bool(intent.get('availability_query') or intent.get('reality_state_query'))}，"
            f"求助={bool(intent.get('help_request'))}，"
            f"邀约={bool(intent.get('invitation'))}，"
            f"游戏邀约={bool(intent.get('game_invitation'))}，"
            f"任务={bool(intent.get('task_request'))}，"
            f"图片={bool(intent.get('image_reference'))}"
        ),
        f"- 策略：{guidance.get('stance_instruction')}",
        f"- 长度：{guidance.get('length_instruction')}，目标约 {guidance.get('target_reply_length')} 字",
    ]
    embedding_status = context.get("embedding_status") or {}
    if embedding_status.get("enabled"):
        lines.append(
            "- 向量检索："
            f"{'可用' if embedding_status.get('ok') else '不可用'}"
            f"，model={embedding_status.get('model') or 'unknown'}"
            f"，hits={embedding_status.get('result_count', 0)}"
        )

    samples = context.get("similar_samples") or []
    if samples:
        lines.append("相似样本：")
        for index, item in enumerate(samples[:limit], start=1):
            lines.append(
                f"{index}. sim={item.get('similarity')} rule={item.get('rule_similarity')} "
                f"emb={item.get('embedding_similarity')} src={item.get('retrieval_source')} "
                f"text={item.get('context_overlap')} kw={item.get('keyword_overlap')} intent={item.get('intent_bonus')} "
                f"q={item.get('quality_score')} {item.get('chat_type')} "
                f"{item.get('source_file_id')} reply_len={item.get('reply_char_length')}"
            )
    else:
        lines.append("相似样本：无。")

    examples = context.get("few_shot_examples") or []
    if examples:
        lines.append("真实历史样本（owner 私聊调试可见；凭据类文本已跳过）：")
        for index, example in enumerate(examples[:limit], start=1):
            lines.append(
                f"样本 {index}：sim={example.get('similarity')} q={example.get('quality_score')} "
                f"{example.get('chat_type')} {example.get('source_file_id')} src={example.get('retrieval_source')}"
            )
            for item in example.get("context") or []:
                role = item.get("role") or "对方"
                lines.append(f"- {role}：{item.get('text')}")
            lines.append(f"- 主人：{example.get('owner_reply')}")
    else:
        lines.append(
            f"真实历史样本：无。可能是相似度低于 {MIN_RAW_FEWSHOT_SIMILARITY}、"
            f"文本/关键词命中低于 {MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD}、"
            "私聊/群聊类型不匹配，或命中样本含凭据/无可用文本。"
        )
    return "\n".join(lines)

def format_similar_sample_results(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return str(result.get("message") or "检索失败。")
    lines = [
        "相似样本检索结果：",
        f"- run_id：{result.get('run_id')}",
        f"- 检索策略：{result.get('retrieval_strategy') or 'rules_only'}",
        f"- 命中：{result.get('result_count', 0)} 条",
        "- 原文策略：规则检索会本地临时读取历史文本；向量索引只存 embedding 和元数据",
    ]
    embedding_status = result.get("embedding_status") or {}
    if embedding_status.get("enabled"):
        lines.append(
            f"- 向量检索：{'可用' if embedding_status.get('ok') else '不可用'}，"
            f"hits={embedding_status.get('result_count', 0)}"
        )
    features = result.get("query_features") or {}
    lines.append(
        f"- 查询特征：{features.get('length_bucket')}，"
        f"问句={features.get('has_question')}，"
        f"感叹={features.get('has_exclamation')}"
    )
    for index, item in enumerate(result.get("results") or [], start=1):
        lines.append(
            f"{index}. sim={item.get('similarity')} rule={item.get('rule_similarity')} "
            f"emb={item.get('embedding_similarity')} src={item.get('retrieval_source')} q={item.get('quality_score')} "
            f"{item.get('chat_type')} {item.get('source_file_id')} "
            f"reply_len={item.get('reply_char_length')} ctx={item.get('context_count')}"
        )
    if not result.get("results"):
        lines.append("没有找到足够相似的历史上下文。")
    return "\n".join(lines)


__all__ = [name for name in globals() if not name.startswith("__")]
