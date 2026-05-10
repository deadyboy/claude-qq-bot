"""Learning-value taxonomy artifacts for QCE style distillation."""

from .qce_io import *
from .phrases import *
from .turns import *


def _phrase_terms(section: Dict[str, Any], *, limit: int = 80) -> set[str]:
    terms: set[str] = set()
    for field in (
        "high_freq_short_replies",
        "confirmation_templates",
        "negative_templates",
        "rhetorical_templates",
    ):
        for item in section.get(field) or []:
            text = str((item or {}).get("text") or "").strip()
            if 2 <= len(text) <= 24 and not contains_sensitive_content(text):
                terms.add(text)
            if len(terms) >= limit:
                return terms
    return terms

def _relationship_phrase_lookup(phrase_profile: Dict[str, Any]) -> Dict[str, set[str]]:
    global_terms = _phrase_terms(phrase_profile.get("global") or {}, limit=200)
    lookup: Dict[str, set[str]] = {}
    for key, section in (phrase_profile.get("per_relationship") or {}).items():
        terms = _phrase_terms(section or {}, limit=120)
        relationship_only = {term for term in terms if term not in global_terms}
        if relationship_only:
            lookup[str(key)] = relationship_only
    return lookup

def _turn_media_elements(turn: Dict[str, Any]) -> set[str]:
    elements = {str(item) for item in (turn.get("element_types") or []) if item}
    media = {item for item in elements if item in MEDIA_DEPENDENT_ELEMENTS}
    if not media:
        media = {item for item in elements if item not in TEXT_ONLY_ELEMENTS}
    text = str(turn.get("text") or "")
    if TEXT_PLACEHOLDER_RE.match(text):
        media.add("placeholder")
    return media

def _pair_target_text(pair: Dict[str, Any]) -> str:
    return str((pair.get("target") or {}).get("text") or "").strip()

def _pair_context_text(pair: Dict[str, Any], *, roles: set[str] | None = None) -> str:
    return _turns_text(pair.get("context") or [], roles=roles)

def _is_low_information_reply(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) <= 2:
        return True
    return bool(re.fullmatch(r"[?？!！.。,…，、~～哈嗯啊哦额]+", compact))

def _classify_learning_pair(
    pair: Dict[str, Any],
    *,
    relationship_profiles_by_source: Dict[str, Dict[str, Any]],
    relationship_phrase_sets: Dict[str, set[str]],
) -> Dict[str, Any]:
    target_text = _pair_target_text(pair)
    context_turns = pair.get("context") or []
    other_context_text = _pair_context_text(pair, roles={"other"})
    full_context_text = _pair_context_text(pair)
    score = int(pair.get("score") or 0)
    chat_type = str(pair.get("chat_type") or "")
    scene_label = str(pair.get("scene_label") or "")
    source_id = str(pair.get("source_file_id") or "")
    relationship_id = str(pair.get("relationship_id") or "")
    rel_key = _phrase_scope_key(chat_type, relationship_id)
    source_profile = relationship_profiles_by_source.get(source_id) or {}
    source_labels = set(source_profile.get("labels") or [])
    context_media_turns = sum(1 for turn in context_turns if _turn_media_elements(turn))
    target_media_elements = _turn_media_elements(pair.get("target") or {})
    context_has_text_grounding = len(re.sub(r"\s+", "", other_context_text)) >= 6
    low_information = _is_low_information_reply(target_text)
    state_sensitive = _contains_any("\n".join([full_context_text, target_text]), REALITY_STATE_HINTS)
    relationship_terms = sorted(
        term
        for term in relationship_phrase_sets.get(rel_key, set())
        if term and (target_text == term or (len(term) >= 3 and term in target_text))
    )[:8]
    media_dependent = (
        bool(target_media_elements)
        or context_media_turns > 0 and len(re.sub(r"\s+", "", other_context_text)) < 24
    )
    grounding_types: List[str] = []
    if media_dependent:
        grounding_types.append("media_dependent")
    if state_sensitive:
        grounding_types.append("state_sensitive")
    if relationship_terms:
        grounding_types.append("relationship_inside_joke")
    if context_has_text_grounding and not media_dependent:
        grounding_types.append("text_grounded")
    if not grounding_types:
        grounding_types.append("weak_text_grounding")

    if media_dependent or relationship_terms or chat_type == "group":
        scope = "same_relationship_only"
    elif "high_familiarity_private" in source_labels and scene_label in {"private_short_casual", "private_emotional"}:
        scope = "familiar_private"
    else:
        scope = "global_style"

    reply_len = len(target_text)
    sft_ok = (
        score >= 85
        and context_has_text_grounding
        and not media_dependent
        and not state_sensitive
        and not low_information
        and 3 <= reply_len <= 160
        and scope in {"global_style", "familiar_private"}
    )
    use_for: List[str] = []
    if score >= 70:
        use_for.append("rag")
    if sft_ok:
        use_for.append("sft_candidate")
    if low_information or relationship_terms or reply_len <= 14:
        use_for.append("phrase_profile")
    if low_information or media_dependent or state_sensitive or relationship_terms:
        use_for.append("rerank_only")
    if not use_for:
        use_for.append("exclude")

    if sft_ok and score >= 95:
        learning_value = "high"
    elif "rag" in use_for and score >= 75 and not state_sensitive:
        learning_value = "medium"
    else:
        learning_value = "low"

    reasons = []
    if context_has_text_grounding:
        reasons.append("has_text_grounding")
    if media_dependent:
        reasons.append("media_or_expression_context")
    if state_sensitive:
        reasons.append("reality_state_sensitive")
    if low_information:
        reasons.append("low_information_short_reply")
    if relationship_terms:
        reasons.append("relationship_specific_phrase")
    if "high_familiarity_private" in source_labels:
        reasons.append("high_familiarity_source")
    if sft_ok:
        reasons.append("sft_ready")

    return {
        "learning_value": learning_value,
        "grounding_type": grounding_types[0],
        "grounding_types": grounding_types,
        "scope": scope,
        "use_for": use_for,
        "reasons": reasons,
        "quality_flags": {
            "text_grounded": context_has_text_grounding,
            "media_dependent": media_dependent,
            "state_sensitive": state_sensitive,
            "relationship_specific": bool(relationship_terms),
            "low_information": low_information,
            "target_has_media_elements": bool(target_media_elements),
        },
        "relationship_terms": relationship_terms,
        "target_char_length": reply_len,
        "context_turn_count": len(context_turns),
        "context_media_turn_count": context_media_turns,
    }

def _taxonomy_record(pair: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": pair.get("pair_id"),
        "source_file_id": pair.get("source_file_id"),
        "relationship_id": pair.get("relationship_id"),
        "chat_type": pair.get("chat_type"),
        "scene_label": pair.get("scene_label"),
        "length_bucket": pair.get("length_bucket"),
        "score": int(pair.get("score") or 0),
        "taxonomy": taxonomy,
    }

def _sft_record(pair: Dict[str, Any], taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    context = [
        {
            "role": str(turn.get("role") or "other"),
            "text": str(turn.get("text") or ""),
            "message_count": int(turn.get("message_count") or 0),
        }
        for turn in pair.get("context") or []
        if str(turn.get("text") or "").strip()
    ]
    target = pair.get("target") or {}
    return {
        "sample_id": pair.get("pair_id"),
        "source_file_id": pair.get("source_file_id"),
        "relationship_id": pair.get("relationship_id"),
        "chat_type": pair.get("chat_type"),
        "scene_label": pair.get("scene_label"),
        "scope": taxonomy.get("scope"),
        "learning_value": taxonomy.get("learning_value"),
        "grounding_type": taxonomy.get("grounding_type"),
        "context": context[-12:],
        "target": str(target.get("text") or ""),
        "target_turn": {
            "message_count": int(target.get("message_count") or 0),
            "char_length": int(target.get("char_length") or len(str(target.get("text") or ""))),
            "length_bucket": target.get("length_bucket"),
        },
        "training_note": "Text-grounded owner turn candidate. Respect scope before any SFT or fine-tuning use.",
    }

def _percentile_int(values: Sequence[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return int(ordered[index])

def _summarize_taxonomy(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    learning = Counter()
    grounding = Counter()
    scope = Counter()
    use_for = Counter()
    scenes = Counter()
    chat_types = Counter()
    for item in records:
        taxonomy = item.get("taxonomy") or {}
        learning[str(taxonomy.get("learning_value") or "unknown")] += 1
        grounding[str(taxonomy.get("grounding_type") or "unknown")] += 1
        scope[str(taxonomy.get("scope") or "unknown")] += 1
        scenes[str(item.get("scene_label") or "unknown")] += 1
        chat_types[str(item.get("chat_type") or "unknown")] += 1
        for usage in taxonomy.get("use_for") or []:
            use_for[str(usage)] += 1
    return {
        "total": len(records),
        "learning_value_counts": dict(learning.most_common()),
        "grounding_type_counts": dict(grounding.most_common()),
        "scope_counts": dict(scope.most_common()),
        "use_for_counts": dict(use_for.most_common()),
        "scene_counts": dict(scenes.most_common()),
        "chat_type_counts": dict(chat_types.most_common()),
    }

def _build_rerank_style_rules(
    taxonomy_records: Sequence[Dict[str, Any]],
    *,
    phrase_profile: Dict[str, Any],
) -> Dict[str, Any]:
    scene_stats: Dict[str, Dict[str, Any]] = {}
    rel_stats: Dict[str, Dict[str, Any]] = {}
    global_lengths: List[int] = []
    for item in taxonomy_records:
        taxonomy = item.get("taxonomy") or {}
        scene_key = str(item.get("scene_label") or "unknown")
        rel_key = _phrase_scope_key(str(item.get("chat_type") or ""), str(item.get("relationship_id") or ""))
        target_len = int(taxonomy.get("target_char_length") or 0)
        global_lengths.append(target_len)
        for key, stats in (
            (scene_key, scene_stats.setdefault(scene_key, {
                "sample_count": 0,
                "lengths": [],
                "learning_value_counts": Counter(),
                "grounding_type_counts": Counter(),
                "scope_counts": Counter(),
                "use_for_counts": Counter(),
                "length_buckets": Counter(),
            })),
            (rel_key, rel_stats.setdefault(rel_key, {
                "sample_count": 0,
                "lengths": [],
                "scene_counts": Counter(),
                "scope_counts": Counter(),
            })),
        ):
            stats["sample_count"] += 1
            stats["lengths"].append(target_len)
            stats["scope_counts"][str(taxonomy.get("scope") or "unknown")] += 1
            if key == scene_key:
                stats["learning_value_counts"][str(taxonomy.get("learning_value") or "unknown")] += 1
                stats["grounding_type_counts"][str(taxonomy.get("grounding_type") or "unknown")] += 1
                stats["length_buckets"][str(item.get("length_bucket") or "unknown")] += 1
                for usage in taxonomy.get("use_for") or []:
                    stats["use_for_counts"][str(usage)] += 1
            else:
                stats["scene_counts"][scene_key] += 1

    scenes = {}
    for scene_key, stats in scene_stats.items():
        lengths = [int(value) for value in stats.pop("lengths")]
        scenes[scene_key] = {
            "sample_count": stats["sample_count"],
            "median_target_length": int(median(lengths)) if lengths else 0,
            "p90_target_length": _percentile_int(lengths, 0.9),
            "max_recommended_length": min(160, max(12, _percentile_int(lengths, 0.9) + 8)),
            "length_buckets": dict(stats["length_buckets"].most_common()),
            "learning_value_counts": dict(stats["learning_value_counts"].most_common()),
            "grounding_type_counts": dict(stats["grounding_type_counts"].most_common()),
            "scope_counts": dict(stats["scope_counts"].most_common()),
            "use_for_counts": dict(stats["use_for_counts"].most_common()),
        }

    relationships = {}
    for rel_key, stats in sorted(rel_stats.items(), key=lambda kv: -int(kv[1]["sample_count"]))[:100]:
        lengths = [int(value) for value in stats.pop("lengths")]
        relationships[rel_key] = {
            "sample_count": stats["sample_count"],
            "median_target_length": int(median(lengths)) if lengths else 0,
            "p90_target_length": _percentile_int(lengths, 0.9),
            "scene_counts": dict(stats["scene_counts"].most_common(6)),
            "scope_counts": dict(stats["scope_counts"].most_common()),
        }

    global_section = phrase_profile.get("global") or {}
    return {
        "schema_version": 1,
        "raw_text_policy": (
            "Contains local owner phrase snippets and statistical rules for reranking. "
            "Do not commit or expose this file."
        ),
        "global": {
            "sample_count": len(taxonomy_records),
            "median_target_length": int(median(global_lengths)) if global_lengths else 0,
            "p90_target_length": _percentile_int(global_lengths, 0.9),
            "common_short_replies": global_section.get("high_freq_short_replies") or [],
            "confirmation_templates": global_section.get("confirmation_templates") or [],
            "negative_templates": global_section.get("negative_templates") or [],
            "particles": global_section.get("particles") or [],
        },
        "hard_filters": {
            "too_ai_like_markers": ["AI", "机器人", "助手", "无法", "我不能", "根据上下文"],
            "too_formal_markers": ["您好", "请问", "非常抱歉", "感谢", "有什么可以帮"],
            "state_sensitive_markers": list(REALITY_STATE_HINTS),
        },
        "scenes": scenes,
        "per_relationship": relationships,
    }

def _build_learning_artifacts(
    raw_pairs: Sequence[Dict[str, Any]],
    *,
    phrase_profile: Dict[str, Any],
    relationship_profiles: Dict[str, Any],
) -> Dict[str, Any]:
    relationship_profiles_by_source = {
        str(item.get("source_file_id") or ""): item
        for item in relationship_profiles.get("profiles") or []
        if item.get("source_file_id")
    }
    relationship_phrase_sets = _relationship_phrase_lookup(phrase_profile)
    taxonomy_records = []
    rag_pool = []
    sft_candidates = []
    for pair in raw_pairs:
        target_text = _pair_target_text(pair)
        if not is_style_training_text_allowed(target_text):
            continue
        if any(
            turn.get("role") == "other" and is_ai_assistant_generated_text(str(turn.get("text") or ""))
            for turn in pair.get("context") or []
        ):
            continue
        taxonomy = _classify_learning_pair(
            pair,
            relationship_profiles_by_source=relationship_profiles_by_source,
            relationship_phrase_sets=relationship_phrase_sets,
        )
        base = _taxonomy_record(pair, taxonomy)
        taxonomy_records.append(base)
        use_for = set(taxonomy.get("use_for") or [])
        if "rag" in use_for:
            rag_pool.append({
                **base,
                "context": pair.get("context") or [],
                "target": pair.get("target") or {},
            })
        if "sft_candidate" in use_for:
            sft_candidates.append(_sft_record(pair, taxonomy))

    rag_pool.sort(
        key=lambda item: (
            item["taxonomy"].get("learning_value") != "high",
            item["taxonomy"].get("scope") == "same_relationship_only",
            -int(item.get("score") or 0),
            str(item.get("pair_id") or ""),
        )
    )
    sft_candidates.sort(
        key=lambda item: (
            item.get("scope") != "global_style",
            item.get("learning_value") != "high",
            str(item.get("source_file_id") or ""),
            str(item.get("sample_id") or ""),
        )
    )
    return {
        "taxonomy_records": taxonomy_records,
        "rag_pool": rag_pool,
        "sft_candidates": sft_candidates,
        "rerank_style_rules": _build_rerank_style_rules(taxonomy_records, phrase_profile=phrase_profile),
        "summary": {
            **_summarize_taxonomy(taxonomy_records),
            "rag_pool_count": len(rag_pool),
            "sft_candidate_count": len(sft_candidates),
        },
    }


__all__ = [name for name in globals() if not name.startswith("__")]
