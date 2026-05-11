"""Turn and dialogue-pair construction for QCE style distillation."""

from .qce_io import *
from .phrases import *


TURN_NGRAM_SIZE = DISTILL_SETTINGS.int_value("turns.text_features.ngram_size", 2)
TURN_KEYWORD_MIN_ALNUM_CHARS = DISTILL_SETTINGS.int_value("turns.text_features.keyword_min_alnum_chars", 2)
TURN_KEYWORD_MIN_CHINESE_NGRAM = DISTILL_SETTINGS.int_value("turns.text_features.keyword_min_chinese_ngram", 2)
TURN_KEYWORD_MAX_CHINESE_NGRAM = DISTILL_SETTINGS.int_value("turns.text_features.keyword_max_chinese_ngram", 3)
TURN_INTENT_QUESTION_BONUS = DISTILL_SETTINGS.float_value("turns.intent_similarity.question_bonus", 0.03)
TURN_INTENT_TYPE_BONUS = DISTILL_SETTINGS.float_value("turns.intent_similarity.question_type_bonus", 0.08)
TURN_INTENT_FLAG_BONUS = DISTILL_SETTINGS.float_value("turns.intent_similarity.flag_bonus", 0.05)
TURN_INTENT_MAX_SCORE = DISTILL_SETTINGS.float_value("turns.intent_similarity.max_score", 0.28)
TURN_TIMESTAMP_MS_THRESHOLD = DISTILL_SETTINGS.int_value("turns.time.timestamp_ms_threshold", 10000000000)
TURN_TIMESTAMP_MS_DIVISOR = DISTILL_SETTINGS.int_value("turns.time.timestamp_ms_divisor", 1000)
TURN_ID_HASH_CHARS = DISTILL_SETTINGS.int_value("turns.hash.id_hash_chars", 12)
TURN_TURN_ID_HASH_CHARS = DISTILL_SETTINGS.int_value("turns.hash.turn_id_hash_chars", 20)
TURN_SCENE_CONTEXT_TAIL_TURNS = DISTILL_SETTINGS.int_value("turns.scene.context_tail_turns", 4)
TURN_GROUP_DIRECT_CONTEXT_TAIL_TURNS = DISTILL_SETTINGS.int_value("turns.scene.group_direct_context_tail_turns", 3)
TURN_PRIVATE_LONG_MIN_CHARS = DISTILL_SETTINGS.int_value("turns.scene.private_long_min_chars", 80)

DIALOGUE_PAIR_MIN_SCORE = DISTILL_SETTINGS.int_value("turns.dialogue_pair.min_score", 45)
DIALOGUE_PAIR_MAX_REPLY_MULTIPLIER = DISTILL_SETTINGS.float_value("turns.dialogue_pair.max_reply_multiplier", 2.0)
DIALOGUE_PAIR_PRIVATE_MULTI_CONTEXT_MIN_TURNS = DISTILL_SETTINGS.int_value("turns.dialogue_pair.private_multi_context_min_turns", 3)
SCORE_HAS_CONTEXT = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.has_other_turn_context", 42)
SCORE_NO_CONTEXT = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.no_other_turn_context", -30)
SCORE_PRIVATE_PAIR = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.private_turn_pair", 24)
SCORE_PRIVATE_MULTI_CONTEXT = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.private_multi_turn_context", 10)
SCORE_GROUP_PAIR = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.group_turn_pair", 8)
SCORE_GROUP_DIRECTED = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.group_directed_context", 18)
SCORE_USABLE_LENGTH = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.usable_target_length", 18)
SCORE_EXPRESSIVE_LENGTH = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.expressive_target", 8)
SCORE_MERGED_OWNER_TURN = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.merged_owner_turn", 8)
SCORE_LONG_OR_FORMAL_SCENE = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.long_or_formal_scene", 6)
SCORE_QUESTION_STYLE = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.question_style", 3)
SCORE_PUNCTUATION_STYLE = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.punctuation_style", 3)
SCORE_CONTAINS_URL = DISTILL_SETTINGS.int_value("turns.dialogue_pair.score.contains_url", -18)
SCORE_USABLE_LENGTH_MIN = DISTILL_SETTINGS.int_value("turns.dialogue_pair.usable_length_min", 2)
SCORE_EXPRESSIVE_LENGTH_MIN = DISTILL_SETTINGS.int_value("turns.dialogue_pair.expressive_length_min", 8)
SCORE_EXPRESSIVE_LENGTH_MAX = DISTILL_SETTINGS.int_value("turns.dialogue_pair.expressive_length_max", 120)

CANDIDATE_QUICK_REPLY_SECONDS = DISTILL_SETTINGS.int_value("turns.candidate_score.quick_reply_seconds", 120)
CANDIDATE_GOOD_LENGTH_MIN = DISTILL_SETTINGS.int_value("turns.candidate_score.good_length_min", 3)
CANDIDATE_GOOD_LENGTH_MAX = DISTILL_SETTINGS.int_value("turns.candidate_score.good_length_max", 80)
CANDIDATE_LONG_LENGTH_MAX = DISTILL_SETTINGS.int_value("turns.candidate_score.long_length_max", MAX_REPLY_LENGTH)
CANDIDATE_MULTI_LINE_THRESHOLD = DISTILL_SETTINGS.int_value("turns.candidate_score.multi_line_threshold", 3)
CANDIDATE_MULTI_CONTEXT_MIN = DISTILL_SETTINGS.int_value("turns.candidate_score.multi_context_min", 2)
CANDIDATE_MULTI_CONTEXT_SCORE_CAP = DISTILL_SETTINGS.int_value("turns.candidate_score.multi_context_score_cap", 12)
CANDIDATE_MULTI_CONTEXT_SCORE_PER_TURN = DISTILL_SETTINGS.int_value("turns.candidate_score.multi_context_score_per_turn", 2)
CANDIDATE_SCORE_HAS_RECENT_OTHER = DISTILL_SETTINGS.int_value("turns.candidate_score.score.has_recent_other_context", 40)
CANDIDATE_SCORE_QUICK_REPLY = DISTILL_SETTINGS.int_value("turns.candidate_score.score.quick_reply", 8)
CANDIDATE_SCORE_NO_RECENT_OTHER = DISTILL_SETTINGS.int_value("turns.candidate_score.score.no_recent_other_context", -25)
CANDIDATE_SCORE_PRIVATE_CHAT = DISTILL_SETTINGS.int_value("turns.candidate_score.score.private_chat", 18)
CANDIDATE_SCORE_GROUP_CHAT = DISTILL_SETTINGS.int_value("turns.candidate_score.score.group_chat", 6)
CANDIDATE_SCORE_EXPLICIT_REPLY = DISTILL_SETTINGS.int_value("turns.candidate_score.score.explicit_reply", 18)
CANDIDATE_SCORE_MENTION = DISTILL_SETTINGS.int_value("turns.candidate_score.score.mentions", 5)
CANDIDATE_SCORE_GOOD_LENGTH = DISTILL_SETTINGS.int_value("turns.candidate_score.score.good_length", 16)
CANDIDATE_SCORE_LONG_BUT_USABLE = DISTILL_SETTINGS.int_value("turns.candidate_score.score.long_but_usable", 8)
CANDIDATE_SCORE_WEAK_LENGTH = DISTILL_SETTINGS.int_value("turns.candidate_score.score.weak_length", -15)
CANDIDATE_SCORE_URL = DISTILL_SETTINGS.int_value("turns.candidate_score.score.contains_url", -20)
CANDIDATE_SCORE_MULTI_LINE = DISTILL_SETTINGS.int_value("turns.candidate_score.score.multi_line", -8)
CANDIDATE_SCORE_EMOJI = DISTILL_SETTINGS.int_value("turns.candidate_score.score.style_marker_emoji", 3)
CANDIDATE_SCORE_QUESTION = DISTILL_SETTINGS.int_value("turns.candidate_score.score.question_style", 2)
CANDIDATE_SCORE_PUNCTUATION = DISTILL_SETTINGS.int_value("turns.candidate_score.score.punctuation_style", 2)

STYLE_SHORT_AVG_MAX = DISTILL_SETTINGS.float_value("turns.style_patch.short_avg_max", 10)
STYLE_MEDIUM_AVG_MAX = DISTILL_SETTINGS.float_value("turns.style_patch.medium_avg_max", 28)
STYLE_EMOJI_HIGH_RATIO = DISTILL_SETTINGS.float_value("turns.style_patch.emoji_high_ratio", 0.35)
STYLE_EMOJI_MEDIUM_RATIO = DISTILL_SETTINGS.float_value("turns.style_patch.emoji_medium_ratio", 0.08)
STYLE_SHORT_RATIO_THRESHOLD = DISTILL_SETTINGS.float_value("turns.style_patch.short_ratio_threshold", 0.45)
STYLE_GROUP_DOMINANCE_MULTIPLIER = DISTILL_SETTINGS.float_value("turns.style_patch.group_dominance_multiplier", 2.0)
STYLE_QUESTION_RATIO_THRESHOLD = DISTILL_SETTINGS.float_value("turns.style_patch.question_ratio_threshold", 0.08)
STYLE_ELLIPSIS_RATIO_THRESHOLD = DISTILL_SETTINGS.float_value("turns.style_patch.ellipsis_ratio_threshold", 0.03)
STYLE_P90_PERCENTILE = DISTILL_SETTINGS.float_value("turns.style_patch.p90_percentile", 0.9)

REL_LOW_EVIDENCE_MAX = DISTILL_SETTINGS.int_value("turns.relationship_labels.low_evidence_max", 20)
REL_HIGH_EVIDENCE_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.high_evidence_min", 1000)
REL_TERSE_AVG_MAX = DISTILL_SETTINGS.float_value("turns.relationship_labels.terse_avg_max", 8)
REL_BRIEF_AVG_MAX = DISTILL_SETTINGS.float_value("turns.relationship_labels.brief_avg_max", 24)
REL_HIGH_FAMILIARITY_OWNER_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.high_familiarity_owner_min", 800)
REL_HIGH_FAMILIARITY_RATIO_MIN = DISTILL_SETTINGS.float_value("turns.relationship_labels.high_familiarity_ratio_min", 0.25)
REL_ACTIVE_PRIVATE_OWNER_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.active_private_owner_min", 100)
REL_ACTIVE_GROUP_RATIO_MIN = DISTILL_SETTINGS.float_value("turns.relationship_labels.active_group_ratio_min", 0.05)
REL_ACTIVE_GROUP_OWNER_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.active_group_owner_min", 300)
REL_OCCASIONAL_GROUP_OWNER_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.occasional_group_owner_min", 100)
REL_STRONG_CONTEXT_CANDIDATES_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.strong_context_candidates_min", 50)
REL_USABLE_CONTEXT_CANDIDATES_MIN = DISTILL_SETTINGS.int_value("turns.relationship_labels.usable_context_candidates_min", 10)
REL_ELEMENT_TYPE_LIMIT = DISTILL_SETTINGS.int_value("turns.relationship_labels.element_type_limit", 10)

EVALUATION_STRONG_INDEXED_MIN = DISTILL_SETTINGS.int_value("turns.evaluation.strong_indexed_min", 1000)
EVALUATION_STRONG_SOURCE_MIN = DISTILL_SETTINGS.int_value("turns.evaluation.strong_source_min", 5)
EVALUATION_USABLE_INDEXED_MIN = DISTILL_SETTINGS.int_value("turns.evaluation.usable_indexed_min", 300)
EVALUATION_USABLE_SOURCE_MIN = DISTILL_SETTINGS.int_value("turns.evaluation.usable_source_min", 3)
EVALUATION_LORA_MIN_PAIRS = DISTILL_SETTINGS.int_value("turns.evaluation.lora_min_pairs", 3000)

SCENE_CONTEXT_MULTI_THRESHOLD = DISTILL_SETTINGS.int_value("turns.scene_profiles.multi_context_threshold", 3)
SCENE_QUICK_REPLY_SECONDS = DISTILL_SETTINGS.int_value("turns.scene_profiles.quick_reply_seconds", 120)
SCENE_SAMPLE_REF_LIMIT = DISTILL_SETTINGS.int_value("turns.scene_profiles.sample_ref_limit", 12)
SCENE_RECOMMEND_GROUP_BRIEF_MAX = DISTILL_SETTINGS.float_value("turns.scene_profiles.group_brief_avg_max", 12)
SCENE_RECOMMEND_PRIVATE_CONTEXTUAL_MIN = DISTILL_SETTINGS.float_value("turns.scene_profiles.private_contextual_avg_min", 20)
SCENE_ELEMENT_TYPE_LIMIT = DISTILL_SETTINGS.int_value("turns.scene_profiles.element_type_limit", 10)
SCENE_MISSING_DELAY_SENTINEL = DISTILL_SETTINGS.int_value("turns.scene_profiles.missing_delay_sentinel", 999999)


def _intent_rule_matches(rule: Dict[str, Any], flags: Dict[str, bool]) -> bool:
    if rule.get("default"):
        return True
    any_flags = [str(item) for item in (rule.get("any") or [])]
    all_flags = [str(item) for item in (rule.get("all") or [])]
    none_flags = [str(item) for item in (rule.get("none") or [])]
    if any_flags and not any(flags.get(item) for item in any_flags):
        return False
    if all_flags and not all(flags.get(item) for item in all_flags):
        return False
    if none_flags and any(flags.get(item) for item in none_flags):
        return False
    return bool(any_flags or all_flags or none_flags)


def _question_type_from_rules(flags: Dict[str, bool]) -> str:
    for rule in QUESTION_TYPE_RULES:
        if _intent_rule_matches(rule, flags):
            return str(rule.get("type") or "statement")
    return "statement"


def _commitment_risk_from_rules(flags: Dict[str, bool]) -> tuple[int, str, List[str]]:
    for rule in COMMITMENT_RISK_RULES:
        if not _intent_rule_matches(rule, flags):
            continue
        try:
            level = int(rule.get("level", 0))
        except (TypeError, ValueError):
            level = 0
        label = str(rule.get("label") or "phatic_social")
        reasons = [str(item) for item in (rule.get("reasons") or []) if str(item)]
        return max(0, min(3, level)), label, reasons
    return 0, "phatic_social", ["low_risk_social_reply"]


def detect_message_intent(text: str) -> Dict[str, Any]:
    """Detect coarse Chinese chat intent without requiring explicit question marks."""
    normalized = re.sub(r"\s+", "", str(text or "").lower())
    has_mark = "?" in normalized or "？" in normalized
    has_question_hint = _contains_any(normalized, QUESTION_HINTS)
    has_game_term = _contains_any(normalized, GAME_HINTS)
    has_play_invite_pattern = bool(PLAY_INVITE_RE.search(normalized))
    game_invitation = has_game_term and has_play_invite_pattern
    availability_query = _contains_any(normalized, AVAILABILITY_HINTS)
    help_request = _contains_any(normalized, HELP_HINTS)
    invitation = game_invitation or _contains_any(normalized, INVITATION_HINTS)
    task_request = _contains_any(normalized, TASK_HINTS)
    image_reference = _contains_any(normalized, IMAGE_HINTS)
    reality_state_query = _contains_any(normalized, REALITY_STATE_HINTS)
    high_risk_request = _contains_any(normalized, HIGH_RISK_COMMITMENT_HINTS)
    is_question = bool(
        has_mark
        or has_question_hint
        or availability_query
        or help_request
        or invitation
    )
    flags = {
        "is_question": is_question,
        "game_invitation": game_invitation,
        "availability_query": availability_query,
        "help_request": help_request,
        "invitation": invitation,
        "task_request": task_request,
        "image_reference": image_reference,
        "reality_state_query": reality_state_query,
        "high_risk_request": high_risk_request,
        "question_mark": has_mark,
        "question_hint": has_question_hint,
        "game_term": has_game_term,
    }
    question_type = _question_type_from_rules(flags)
    commitment_risk_level, commitment_risk_label, commitment_reasons = _commitment_risk_from_rules(flags)

    return {
        "is_question": is_question,
        "question_type": question_type,
        "availability_query": availability_query,
        "help_request": help_request,
        "invitation": invitation,
        "game_invitation": game_invitation,
        "task_request": task_request,
        "image_reference": image_reference,
        "reality_state_query": reality_state_query,
        "game_term": has_game_term,
        "question_mark": has_mark,
        "question_hint": has_question_hint,
        "high_risk_request": high_risk_request,
        "commitment_risk_level": commitment_risk_level,
        "commitment_risk_label": commitment_risk_label,
        "commitment_risk_reasons": commitment_reasons,
    }

def _features_for_text(text: str) -> Dict[str, Any]:
    intent = detect_message_intent(text)
    return {
        "char_length": len(text),
        "length_bucket": _bucket_length(len(text)),
        "emoji_count": len(EMOJI_RE.findall(text)),
        "has_question": intent["is_question"],
        "has_exclamation": "!" in text or "！" in text,
        "has_ellipsis": "..." in text or "…" in text,
        "has_url": bool(URL_RE.search(text)),
        "line_count": max(1, text.count("\n") + 1),
        "intent": intent,
    }

def _text_ngrams(text: str, n: int = TURN_NGRAM_SIZE) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i:i + n] for i in range(0, len(compact) - n + 1)}

def _keyword_tokens(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text or "").lower())
    tokens = set(re.findall(rf"[a-z0-9_]{{{TURN_KEYWORD_MIN_ALNUM_CHARS},}}", compact))
    chinese = re.findall(r"[\u4e00-\u9fff]+", compact)
    for chunk in chinese:
        if len(chunk) <= TURN_KEYWORD_MIN_CHINESE_NGRAM:
            tokens.add(chunk)
            continue
        for size in range(TURN_KEYWORD_MIN_CHINESE_NGRAM, TURN_KEYWORD_MAX_CHINESE_NGRAM + 1):
            if len(chunk) < size:
                continue
            tokens.update(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    intent = detect_message_intent(compact)
    for key in (
        "question_type",
        "availability_query",
        "help_request",
        "invitation",
        "game_invitation",
        "task_request",
        "image_reference",
        "reality_state_query",
    ):
        value = intent.get(key)
        if value:
            if key == "question_type" and value == "statement":
                continue
            tokens.add(f"intent:{key}:{value}" if isinstance(value, str) else f"intent:{key}")
    return {token for token in tokens if token}

def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))

def _intent_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    score = 0.0
    if left.get("is_question") and right.get("is_question"):
        score += TURN_INTENT_QUESTION_BONUS
    if left.get("question_type") == right.get("question_type") and left.get("question_type") != "statement":
        score += TURN_INTENT_TYPE_BONUS
    for key in (
        "availability_query",
        "help_request",
        "invitation",
        "game_invitation",
        "task_request",
        "image_reference",
        "reality_state_query",
    ):
        if left.get(key) and right.get(key):
            score += TURN_INTENT_FLAG_BONUS
    return min(score, TURN_INTENT_MAX_SCORE)

def _time_bucket(timestamp: int) -> str:
    if timestamp <= 0:
        return ""
    if timestamp > TURN_TIMESTAMP_MS_THRESHOLD:
        timestamp = timestamp // max(1, TURN_TIMESTAMP_MS_DIVISOR)
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:00")
    except (OSError, OverflowError, ValueError):
        return ""

def _message_ref(message: Dict[str, Any], record_index: int) -> Dict[str, Any]:
    timestamp = _timestamp(message)
    return {
        "record_index": record_index,
        "id_hash": _sha1_short(str(message.get("id") or ""), TURN_ID_HASH_CHARS),
        "seq_hash": _sha1_short(str(message.get("seq") or ""), TURN_ID_HASH_CHARS),
        "time_bucket": _time_bucket(timestamp),
    }

def _turn_ref(turn: Dict[str, Any], *, include_text: bool = False) -> Dict[str, Any]:
    item = {
        "turn_id": turn["turn_id"],
        "source_file_id": turn.get("source_file_id"),
        "relationship_id": turn.get("relationship_id"),
        "chat_type": turn.get("chat_type"),
        "chat_id": turn.get("chat_id"),
        "role": turn["role"],
        "sender_hash": _sha1_short(str(turn.get("sender_id") or ""), TURN_ID_HASH_CHARS),
        "start_time_bucket": _time_bucket(int(turn.get("start_ts") or 0)),
        "end_time_bucket": _time_bucket(int(turn.get("end_ts") or 0)),
        "message_count": len(turn.get("raw_texts") or []),
        "char_length": len(turn.get("text") or ""),
        "length_bucket": _bucket_length(len(turn.get("text") or "")),
        "element_types": list(turn.get("element_types") or []),
        "messages": list(turn.get("message_refs") or []),
    }
    if include_text:
        item.update({
            "sender_id": str(turn.get("sender_id") or ""),
            "raw_texts": list(turn.get("raw_texts") or []),
            "text": str(turn.get("text") or ""),
        })
    return item

def _turns_text(turns: Sequence[Dict[str, Any]], *, roles: set[str] | None = None) -> str:
    texts = []
    for turn in turns:
        if roles and str(turn.get("role") or "") not in roles:
            continue
        text = str(turn.get("text") or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts)

def _build_turns(
    messages: Sequence[Dict[str, Any]],
    *,
    source_file_id: str,
    relationship_id: str,
    chat_type: str,
    self_uin: str,
) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        current["text"] = "\n".join(current["raw_texts"]).strip()
        current["element_types"] = sorted(set(current["element_types"]))
        current["turn_id"] = _sha1_short(
            f"{source_file_id}:{current['role']}:{current['sender_id']}:"
            f"{current['start_index']}:{current['end_index']}:{current['start_ts']}",
            TURN_TURN_ID_HASH_CHARS,
        )
        turns.append(current)
        current = None

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or _is_system_or_recalled(message):
            continue
        text = _message_text(message)
        if not text or contains_sensitive_content(text):
            continue
        sender_id = _sender_uin(message)
        role = "self" if sender_id == self_uin else "other"
        if role == "self" and not is_style_training_text_allowed(text):
            continue
        timestamp = _timestamp(message)
        element_types = _element_types(message)
        same_turn = (
            current is not None
            and current["role"] == role
            and current["sender_id"] == sender_id
            and timestamp > 0
            and int(current.get("end_ts") or 0) > 0
            and timestamp - int(current["end_ts"]) <= TURN_MERGE_SECONDS
        )
        if not same_turn:
            flush()
            current = {
                "source_file_id": source_file_id,
                "relationship_id": relationship_id,
                "chat_type": chat_type,
                "chat_id": relationship_id,
                "role": role,
                "sender_id": sender_id,
                "start_index": index,
                "end_index": index,
                "start_ts": timestamp,
                "end_ts": timestamp,
                "raw_texts": [text],
                "text": text,
                "element_types": list(element_types),
                "message_refs": [_message_ref(message, index)],
            }
            continue
        current["end_index"] = index
        current["end_ts"] = timestamp
        current["raw_texts"].append(text)
        current["element_types"].extend(element_types)
        current["message_refs"].append(_message_ref(message, index))

    flush()
    return turns

def _turn_has_direct_reference(turn: Dict[str, Any]) -> bool:
    element_types = set(turn.get("element_types") or [])
    text = str(turn.get("text") or "")
    return bool({"reply", "at"} & element_types) or "@" in text

def _private_context_turns(turns: Sequence[Dict[str, Any]], target_index: int) -> List[Dict[str, Any]]:
    return list(turns[max(0, target_index - PRIVATE_CONTEXT_TURNS):target_index])

def _group_context_turns(turns: Sequence[Dict[str, Any]], target_index: int) -> List[Dict[str, Any]]:
    target = turns[target_index]
    target_ts = int(target.get("start_ts") or 0)
    context = []
    for turn in reversed(turns[:target_index]):
        turn_ts = int(turn.get("end_ts") or 0)
        if target_ts and turn_ts and target_ts - turn_ts > GROUP_CONTEXT_WINDOW_SECONDS:
            break
        context.append(turn)
        if len(context) >= GROUP_CONTEXT_TURNS:
            break
    context.reverse()
    return context

def _scene_label_for_pair(
    *,
    chat_type: str,
    target_turn: Dict[str, Any],
    context_turns: Sequence[Dict[str, Any]],
) -> str:
    target_text = str(target_turn.get("text") or "")
    context_text = "\n".join(str(item.get("text") or "") for item in context_turns[-TURN_SCENE_CONTEXT_TAIL_TURNS:])
    combined = f"{context_text}\n{target_text}"
    target_features = _features_for_text(target_text)
    if _contains_any(combined, FORMAL_WORK_HINTS) or target_features["has_url"]:
        return "formal_or_worklike"
    if chat_type == "group":
        if _turn_has_direct_reference(target_turn) or any(
            _turn_has_direct_reference(item) for item in context_turns[-TURN_GROUP_DIRECT_CONTEXT_TAIL_TURNS:]
        ):
            return "group_mentioned_or_reply"
        return "group_interjection"
    if _contains_any(combined, EMOTIONAL_HINTS) or target_features["has_exclamation"] or target_features["emoji_count"] > 0:
        return "private_emotional"
    if len(target_text) >= TURN_PRIVATE_LONG_MIN_CHARS or target_features["line_count"] > 1:
        return "private_long_explain"
    return "private_short_casual"

def _score_dialogue_pair(
    *,
    chat_type: str,
    target_turn: Dict[str, Any],
    context_turns: Sequence[Dict[str, Any]],
    scene_label: str,
) -> tuple[int, List[str]]:
    score = 0
    reasons = []
    text = str(target_turn.get("text") or "")
    length = len(text)
    other_context = [item for item in context_turns if item.get("role") == "other"]
    if other_context:
        score += SCORE_HAS_CONTEXT
        reasons.append("has_other_turn_context")
    else:
        score += SCORE_NO_CONTEXT
        reasons.append("no_other_turn_context")
    if chat_type == "private":
        score += SCORE_PRIVATE_PAIR
        reasons.append("private_turn_pair")
        if len(context_turns) >= DIALOGUE_PAIR_PRIVATE_MULTI_CONTEXT_MIN_TURNS:
            score += SCORE_PRIVATE_MULTI_CONTEXT
            reasons.append("private_multi_turn_context")
    else:
        score += SCORE_GROUP_PAIR
        reasons.append("group_turn_pair")
        if scene_label == "group_mentioned_or_reply":
            score += SCORE_GROUP_DIRECTED
            reasons.append("group_directed_context")
    if SCORE_USABLE_LENGTH_MIN <= length <= MAX_REPLY_LENGTH:
        score += SCORE_USABLE_LENGTH
        reasons.append("usable_target_length")
    if SCORE_EXPRESSIVE_LENGTH_MIN <= length <= SCORE_EXPRESSIVE_LENGTH_MAX:
        score += SCORE_EXPRESSIVE_LENGTH
        reasons.append("expressive_target")
    if len(target_turn.get("raw_texts") or []) > 1:
        score += SCORE_MERGED_OWNER_TURN
        reasons.append("merged_owner_turn")
    if scene_label in {"private_long_explain", "formal_or_worklike"}:
        score += SCORE_LONG_OR_FORMAL_SCENE
        reasons.append(scene_label)
    target_features = _features_for_text(text)
    if target_features["has_question"]:
        score += SCORE_QUESTION_STYLE
        reasons.append("question_style")
    if target_features["has_exclamation"] or target_features["has_ellipsis"]:
        score += SCORE_PUNCTUATION_STYLE
        reasons.append("punctuation_style")
    if target_features["has_url"]:
        score += SCORE_CONTAINS_URL
        reasons.append("contains_url")
    return score, reasons

def _build_dialogue_pairs_for_source(
    turns: Sequence[Dict[str, Any]],
    *,
    source_file_id: str,
    relationship_id: str,
    chat_type: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_pairs = []
    index_pairs = []
    for target_index, target_turn in enumerate(turns):
        if target_turn.get("role") != "self":
            continue
        target_text = str(target_turn.get("text") or "")
        if not is_style_training_text_allowed(target_text) or len(target_text) > MAX_REPLY_LENGTH * DIALOGUE_PAIR_MAX_REPLY_MULTIPLIER:
            continue
        if chat_type == "private":
            context_turns = _private_context_turns(turns, target_index)
        else:
            context_turns = _group_context_turns(turns, target_index)
        if not any(item.get("role") == "other" for item in context_turns):
            continue
        if any(
            item.get("role") == "other" and is_ai_assistant_generated_text(str(item.get("text") or ""))
            for item in context_turns
        ):
            continue
        scene_label = _scene_label_for_pair(
            chat_type=chat_type,
            target_turn=target_turn,
            context_turns=context_turns,
        )
        score, reasons = _score_dialogue_pair(
            chat_type=chat_type,
            target_turn=target_turn,
            context_turns=context_turns,
            scene_label=scene_label,
        )
        if score < DIALOGUE_PAIR_MIN_SCORE:
            continue
        pair_id = _sha1_short(f"{source_file_id}:{target_turn['turn_id']}:{target_index}", TURN_TURN_ID_HASH_CHARS)
        context_refs = [_turn_ref(item, include_text=False) for item in context_turns]
        context_raw = [_turn_ref(item, include_text=True) for item in context_turns]
        target_ref = _turn_ref(target_turn, include_text=False)
        target_raw = _turn_ref(target_turn, include_text=True)
        target_features = _features_for_text(target_text)
        index_pair = {
            "sample_id": pair_id,
            "pair_id": pair_id,
            "source_file_id": source_file_id,
            "relationship_id": relationship_id,
            "chat_type": chat_type,
            "scene_label": scene_label,
            "reply": {
                **target_ref,
                "record_index": target_turn["start_index"],
                "char_length": len(target_text),
                "length_bucket": _bucket_length(len(target_text)),
                "features": {
                    "emoji_count": target_features["emoji_count"],
                    "has_question": target_features["has_question"],
                    "has_exclamation": target_features["has_exclamation"],
                    "has_ellipsis": target_features["has_ellipsis"],
                    "line_count": target_features["line_count"],
                    "intent": target_features["intent"],
                },
            },
            "context": {
                "count": len(context_refs),
                "turns": context_refs,
                "messages": [
                    {**message_ref, "role": turn_ref["role"], "element_types": turn_ref.get("element_types") or []}
                    for turn_ref in context_refs
                    for message_ref in turn_ref.get("messages") or []
                ],
            },
            "score": score,
            "score_reasons": reasons,
        }
        raw_pair = {
            "pair_id": pair_id,
            "source_file_id": source_file_id,
            "relationship_id": relationship_id,
            "chat_type": chat_type,
            "scene_label": scene_label,
            "length_bucket": _bucket_length(len(target_text)),
            "score": score,
            "context": context_raw,
            "target": target_raw,
        }
        raw_pairs.append(raw_pair)
        index_pairs.append(index_pair)
    return raw_pairs, index_pairs

def _context_for(
    messages: Sequence[Dict[str, Any]],
    index: int,
    self_uin: str,
    window_seconds: int,
    max_messages: int,
) -> tuple[List[tuple[int, Dict[str, Any]]], int | None]:
    reply_time = _timestamp(messages[index])
    context: List[tuple[int, Dict[str, Any]]] = []
    first_non_self_delay: int | None = None

    for j in range(index - 1, -1, -1):
        previous = messages[j]
        if _is_system_or_recalled(previous):
            continue
        previous_time = _timestamp(previous)
        if reply_time and previous_time and reply_time - previous_time > window_seconds:
            break
        text = _message_text(previous)
        if not _is_useful_text(text):
            continue
        is_self = _sender_uin(previous) == self_uin
        if not is_self and first_non_self_delay is None and reply_time and previous_time:
            first_non_self_delay = max(0, reply_time - previous_time)
        context.append((j, previous))
        if len(context) >= max_messages:
            break

    context.reverse()
    return context, first_non_self_delay

def _score_candidate(
    *,
    chat_type: str,
    text: str,
    context: Sequence[Dict[str, Any]],
    first_non_self_delay: int | None,
    element_types: Sequence[str],
) -> tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    length = len(text)

    if first_non_self_delay is not None:
        score += CANDIDATE_SCORE_HAS_RECENT_OTHER
        reasons.append("has_recent_other_context")
        if first_non_self_delay <= CANDIDATE_QUICK_REPLY_SECONDS:
            score += CANDIDATE_SCORE_QUICK_REPLY
            reasons.append("quick_reply")
    else:
        score += CANDIDATE_SCORE_NO_RECENT_OTHER
        reasons.append("no_recent_other_context")

    if chat_type == "private":
        score += CANDIDATE_SCORE_PRIVATE_CHAT
        reasons.append("private_chat")
    elif chat_type == "group":
        score += CANDIDATE_SCORE_GROUP_CHAT
        reasons.append("group_chat")

    if "reply" in element_types:
        score += CANDIDATE_SCORE_EXPLICIT_REPLY
        reasons.append("explicit_reply")
    if "at" in element_types:
        score += CANDIDATE_SCORE_MENTION
        reasons.append("mentions")
    if len(context) >= CANDIDATE_MULTI_CONTEXT_MIN:
        score += min(CANDIDATE_MULTI_CONTEXT_SCORE_CAP, len(context) * CANDIDATE_MULTI_CONTEXT_SCORE_PER_TURN)
        reasons.append("multi_message_context")

    if CANDIDATE_GOOD_LENGTH_MIN <= length <= CANDIDATE_GOOD_LENGTH_MAX:
        score += CANDIDATE_SCORE_GOOD_LENGTH
        reasons.append("good_length")
    elif CANDIDATE_GOOD_LENGTH_MAX < length <= CANDIDATE_LONG_LENGTH_MAX:
        score += CANDIDATE_SCORE_LONG_BUT_USABLE
        reasons.append("long_but_usable")
    else:
        score += CANDIDATE_SCORE_WEAK_LENGTH
        reasons.append("weak_length")

    features = _features_for_text(text)
    if features["has_url"]:
        score += CANDIDATE_SCORE_URL
        reasons.append("contains_url")
    if features["line_count"] > CANDIDATE_MULTI_LINE_THRESHOLD:
        score += CANDIDATE_SCORE_MULTI_LINE
        reasons.append("multi_line")
    if features["emoji_count"] > 0:
        score += CANDIDATE_SCORE_EMOJI
        reasons.append("style_marker_emoji")
    if features["has_question"]:
        score += CANDIDATE_SCORE_QUESTION
        reasons.append("question_style")
    if features["has_exclamation"] or features["has_ellipsis"]:
        score += CANDIDATE_SCORE_PUNCTUATION
        reasons.append("punctuation_style")

    return score, reasons

def _percentile(values: Sequence[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))
    return int(ordered[index])

def _merge_counter(target: Counter, values: Iterable[str]) -> None:
    for value in values:
        if value:
            target[value] += 1

def _style_patch_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    owner = stats["owner"]
    lengths = owner["lengths"]
    avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0
    median_length = int(median(lengths)) if lengths else 0
    short_ratio = owner["length_buckets"].get("1-2", 0) + owner["length_buckets"].get("3-6", 0)
    short_ratio = short_ratio / max(1, owner["text_messages"])
    emoji_ratio = owner["emoji_messages"] / max(1, owner["text_messages"])
    private_count = owner["by_chat_type"].get("private", 0)
    group_count = owner["by_chat_type"].get("group", 0)

    if avg_length <= STYLE_SHORT_AVG_MAX:
        length = f"短句为主，平均约 {avg_length} 字，中位数约 {median_length} 字"
    elif avg_length <= STYLE_MEDIUM_AVG_MAX:
        length = f"中短句为主，平均约 {avg_length} 字，中位数约 {median_length} 字"
    else:
        length = f"中等长度回复较多，平均约 {avg_length} 字，中位数约 {median_length} 字"

    if emoji_ratio >= STYLE_EMOJI_HIGH_RATIO:
        emoji = "会使用表情/贴图语气，但仍以文字为主"
    elif emoji_ratio >= STYLE_EMOJI_MEDIUM_RATIO:
        emoji = "偶尔使用表情或贴图，保持克制"
    else:
        emoji = "很少显式使用 emoji；需要时可用短文本表达语气"

    tone_parts = ["口语化", "直接", "低铺垫"]
    if short_ratio >= STYLE_SHORT_RATIO_THRESHOLD:
        tone_parts.append("偏短句接话")
    if group_count > private_count * STYLE_GROUP_DOMINANCE_MULTIPLIER:
        tone_parts.append("群聊里更像插话和接梗")

    habits = [
        f"从 {owner['text_messages']} 条本人文本消息蒸馏，只保存统计特征，不保存原文",
        f"平均回复约 {avg_length} 字，中位数约 {median_length} 字",
        "优先顺着上下文直接回应，少用正式开场",
    ]
    if short_ratio >= STYLE_SHORT_RATIO_THRESHOLD:
        habits.append("大量回复是短句，适合轻量闲聊和快速确认")
    if group_count > private_count:
        habits.append("群聊发言多，常见形态是短插话、接话、反应和补充")
    if private_count:
        habits.append("私聊里比群聊更适合保留上下文和具体回应")
    if owner["question_messages"] / max(1, owner["text_messages"]) >= STYLE_QUESTION_RATIO_THRESHOLD:
        habits.append("会用反问或追问推进对话")
    if owner["ellipsis_messages"] / max(1, owner["text_messages"]) >= STYLE_ELLIPSIS_RATIO_THRESHOLD:
        habits.append("会用省略号表现停顿或无语感")

    punctuation = []
    if owner["question_messages"]:
        punctuation.append("会使用问号表达追问或反问")
    if owner["exclamation_messages"]:
        punctuation.append("偶尔用感叹号加强情绪")
    if owner["ellipsis_messages"]:
        punctuation.append("会使用省略号表达停顿")

    return {
        "tone": "、".join(tone_parts),
        "length": length,
        "emoji": emoji,
        "habits": habits,
        "common_phrases": [],
        "punctuation": punctuation,
        "stats": {
            "source": "qce_offline_stage5b",
            "sample_count": owner["text_messages"],
            "candidate_reply_pairs": stats["samples"]["candidate_count"],
            "indexed_samples": stats["samples"]["indexed_count"],
            "avg_length": avg_length,
            "median_length": median_length,
            "p90_length": _percentile(lengths, STYLE_P90_PERCENTILE),
            "private_self_messages": private_count,
            "group_self_messages": group_count,
        },
    }

def _new_source_stats(source_file_id: str, chat_type: str, message_count: int) -> Dict[str, Any]:
    return {
        "source_file_id": source_file_id,
        "chat_type": chat_type,
        "message_count": message_count,
        "owner_text_messages": 0,
        "owner_lengths": [],
        "candidate_samples": 0,
        "score_total": 0,
        "element_types": Counter(),
        "length_buckets": Counter(),
        "quick_replies": 0,
        "question_messages": 0,
        "exclamation_messages": 0,
        "ellipsis_messages": 0,
    }

def _relationship_labels(item: Dict[str, Any]) -> List[str]:
    labels = []
    chat_type = item["chat_type"]
    message_count = int(item.get("message_count") or 0)
    owner_count = int(item.get("owner_text_messages") or 0)
    candidates = int(item.get("candidate_samples") or 0)
    avg_length = float(item.get("avg_length") or 0)
    owner_ratio = owner_count / max(1, message_count)

    labels.append("private_dialogue" if chat_type == "private" else "group_chat")

    if owner_count < REL_LOW_EVIDENCE_MAX:
        labels.append("low_evidence")
    elif owner_count >= REL_HIGH_EVIDENCE_MIN:
        labels.append("high_evidence")
    else:
        labels.append("medium_evidence")

    if avg_length <= REL_TERSE_AVG_MAX:
        labels.append("terse_replies")
    elif avg_length <= REL_BRIEF_AVG_MAX:
        labels.append("brief_replies")
    else:
        labels.append("detailed_replies")

    if chat_type == "private":
        if owner_count >= REL_HIGH_FAMILIARITY_OWNER_MIN and owner_ratio >= REL_HIGH_FAMILIARITY_RATIO_MIN:
            labels.append("high_familiarity_private")
        elif owner_count >= REL_ACTIVE_PRIVATE_OWNER_MIN:
            labels.append("active_private")
        else:
            labels.append("light_private")
    else:
        if owner_ratio >= REL_ACTIVE_GROUP_RATIO_MIN and owner_count >= REL_ACTIVE_GROUP_OWNER_MIN:
            labels.append("active_group_participant")
        elif owner_count >= REL_OCCASIONAL_GROUP_OWNER_MIN:
            labels.append("occasional_group_participant")
        else:
            labels.append("low_frequency_group")

    if candidates >= REL_STRONG_CONTEXT_CANDIDATES_MIN:
        labels.append("strong_context_reply_source")
    elif candidates >= REL_USABLE_CONTEXT_CANDIDATES_MIN:
        labels.append("usable_context_reply_source")
    return labels

def _build_relationship_profiles(source_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    profiles = []
    for item in source_stats.values():
        lengths = item["owner_lengths"]
        avg_length = round(sum(lengths) / len(lengths), 1) if lengths else 0
        median_length = int(median(lengths)) if lengths else 0
        public_item = {
            "source_file_id": item["source_file_id"],
            "chat_type": item["chat_type"],
            "message_count": item["message_count"],
            "owner_text_messages": item["owner_text_messages"],
            "candidate_samples": item["candidate_samples"],
            "avg_length": avg_length,
            "median_length": median_length,
            "length_buckets": dict(item["length_buckets"]),
            "element_types": dict(item["element_types"].most_common(REL_ELEMENT_TYPE_LIMIT)),
            "quick_replies": item["quick_replies"],
            "question_messages": item["question_messages"],
            "exclamation_messages": item["exclamation_messages"],
            "ellipsis_messages": item["ellipsis_messages"],
        }
        public_item["labels"] = _relationship_labels(public_item)
        if item["candidate_samples"]:
            public_item["avg_sample_score"] = round(item["score_total"] / item["candidate_samples"], 1)
        else:
            public_item["avg_sample_score"] = 0
        profiles.append(public_item)

    profiles.sort(
        key=lambda p: (
            -int(p["candidate_samples"]),
            -int(p["owner_text_messages"]),
            str(p["source_file_id"]),
        )
    )
    label_counts = Counter(label for profile in profiles for label in profile["labels"])
    return {
        "schema_version": 1,
        "raw_text_policy": "No raw chat text is stored in relationship profiles.",
        "profile_count": len(profiles),
        "label_counts": dict(label_counts.most_common()),
        "profiles": profiles[:MAX_RELATIONSHIP_PROFILES],
    }

def _build_evaluation_report(
    *,
    summary: Dict[str, Any],
    relationship_profiles: Dict[str, Any],
    indexed: Sequence[Dict[str, Any]],
    phrase_profile: Dict[str, Any],
    taxonomy_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    score_buckets = Counter(str((int(item["score"]) // 10) * 10) for item in indexed)
    relation_profiles = relationship_profiles.get("profiles") or []
    strong_sources = sum(1 for item in relation_profiles if "strong_context_reply_source" in (item.get("labels") or []))
    usable_sources = sum(1 for item in relation_profiles if "usable_context_reply_source" in (item.get("labels") or []))
    low_evidence = sum(1 for item in relation_profiles if "low_evidence" in (item.get("labels") or []))
    samples = (summary.get("stats") or {}).get("samples") or {}
    owner = (summary.get("stats") or {}).get("owner") or {}
    turns = (summary.get("stats") or {}).get("turns") or {}
    scene_counts = samples.get("scene_counts") or {}
    sample_chat_counts = samples.get("chat_type_counts") or {}
    pair_total = int(samples.get("dialogue_pair_count") or samples.get("candidate_count") or 0)
    phrase_global = phrase_profile.get("global") or {}
    common_phrases_empty = not any(
        phrase_global.get(name)
        for name in (
            "high_freq_short_replies",
            "confirmation_templates",
            "negative_templates",
            "rhetorical_templates",
        )
    )

    readiness = "weak"
    indexed_count = int(samples.get("indexed_count") or 0)
    owner_text = int(owner.get("text_messages") or 0)
    if indexed_count >= EVALUATION_STRONG_INDEXED_MIN and strong_sources >= EVALUATION_STRONG_SOURCE_MIN:
        readiness = "strong"
    elif indexed_count >= EVALUATION_USABLE_INDEXED_MIN and (strong_sources + usable_sources) >= EVALUATION_USABLE_SOURCE_MIN:
        readiness = "usable"

    return {
        "schema_version": 1,
        "run_id": summary.get("run_id"),
        "raw_text_policy": "No raw chat text is stored in this evaluation report.",
        "readiness": readiness,
        "owner_text_messages": owner_text,
        "turn_count": int(turns.get("count") or 0),
        "dialogue_pair_count": pair_total,
        "indexed_samples": indexed_count,
        "candidate_samples": int(samples.get("candidate_count") or 0),
        "high_quality_samples": int(samples.get("high_quality_count") or 0),
        "scene_counts": dict(scene_counts),
        "taxonomy": taxonomy_summary or {},
        "chat_type_ratio": {
            "private": round(int(sample_chat_counts.get("private") or 0) / max(1, pair_total), 4),
            "group": round(int(sample_chat_counts.get("group") or 0) / max(1, pair_total), 4),
        },
        "common_phrases_empty": common_phrases_empty,
        "score_buckets": dict(score_buckets.most_common()),
        "relationship_sources": {
            "total": len(relation_profiles),
            "strong": strong_sources,
            "usable": usable_sources,
            "low_evidence": low_evidence,
        },
        "recommendations": [
            "先使用草稿模式，不开启自动发送。",
            "下一步优先补关系/场景标签，再做相似样本临时检索。",
            "对高风险现实状态、承诺、金钱和隐私问题继续保持拒绝或模糊草稿。",
        ] + (
            [f"有效 dialogue pairs 少于 {EVALUATION_LORA_MIN_PAIRS}：不要 LoRA/微调，先做 RAG 草稿模式。"]
            if int(samples.get("dialogue_pair_count") or samples.get("candidate_count") or 0) < EVALUATION_LORA_MIN_PAIRS
            else []
        ),
    }

def _scene_key_for_sample(sample: Dict[str, Any]) -> str:
    explicit = str(sample.get("scene_label") or "").strip()
    if explicit:
        return explicit
    reply = sample.get("reply") or {}
    features = reply.get("features") or {}
    context = sample.get("context") or {}
    reasons = set(sample.get("score_reasons") or [])
    parts = [str(sample.get("chat_type") or "unknown")]
    if "explicit_reply" in reasons:
        parts.append("explicit_reply")
    elif "mentions" in reasons:
        parts.append("mentioned")
    elif int(context.get("count") or 0) >= 3:
        parts.append("multi_context")
    else:
        parts.append("direct_context")
    if features.get("has_question"):
        parts.append("question_reply")
    elif features.get("has_exclamation"):
        parts.append("emotional_reply")
    elif features.get("has_ellipsis"):
        parts.append("pause_reply")
    else:
        parts.append("plain_reply")
    parts.append(str(reply.get("length_bucket") or "unknown_len"))
    return "::".join(parts)

def _scene_recommendation(scene_id: str, avg_reply_length: float) -> str:
    if "question_reply" in scene_id:
        return "contextual_follow_up"
    if "group" in scene_id and avg_reply_length <= SCENE_RECOMMEND_GROUP_BRIEF_MAX:
        return "brief_group_interjection"
    if "private" in scene_id and avg_reply_length >= SCENE_RECOMMEND_PRIVATE_CONTEXTUAL_MIN:
        return "contextual_private_reply"
    if "emotional_reply" in scene_id:
        return "short_reaction"
    return "terse_contextual_reply"

def _build_scene_profiles(indexed: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scenes: Dict[str, Dict[str, Any]] = {}
    for sample in indexed:
        scene_id = _scene_key_for_sample(sample)
        reply = sample.get("reply") or {}
        context = sample.get("context") or {}
        features = reply.get("features") or {}
        item = scenes.setdefault(scene_id, {
            "scene_id": scene_id,
            "sample_count": 0,
            "chat_types": Counter(),
            "length_buckets": Counter(),
            "element_types": Counter(),
            "context_count_buckets": Counter(),
            "score_total": 0,
            "reply_length_total": 0,
            "question_replies": 0,
            "exclamation_replies": 0,
            "ellipsis_replies": 0,
            "quick_replies": 0,
            "sample_refs": [],
        })
        item["sample_count"] += 1
        item["chat_types"][str(sample.get("chat_type") or "unknown")] += 1
        item["length_buckets"][str(reply.get("length_bucket") or "unknown")] += 1
        _merge_counter(item["element_types"], reply.get("element_types") or [])
        context_count = int(context.get("count") or 0)
        if context_count <= 1:
            item["context_count_buckets"]["1"] += 1
        elif context_count <= SCENE_CONTEXT_MULTI_THRESHOLD:
            item["context_count_buckets"]["2-3"] += 1
        else:
            item["context_count_buckets"]["4+"] += 1
        item["score_total"] += int(sample.get("score") or 0)
        item["reply_length_total"] += int(reply.get("char_length") or 0)
        if features.get("has_question"):
            item["question_replies"] += 1
        if features.get("has_exclamation"):
            item["exclamation_replies"] += 1
        if features.get("has_ellipsis"):
            item["ellipsis_replies"] += 1
        if int(context.get("first_non_self_delay_seconds") or SCENE_MISSING_DELAY_SENTINEL) <= SCENE_QUICK_REPLY_SECONDS:
            item["quick_replies"] += 1
        if len(item["sample_refs"]) < SCENE_SAMPLE_REF_LIMIT:
            item["sample_refs"].append(str(sample.get("sample_id") or ""))

    profiles = []
    for item in scenes.values():
        sample_count = max(1, item["sample_count"])
        avg_reply_length = round(item["reply_length_total"] / sample_count, 1)
        profiles.append({
            "scene_id": item["scene_id"],
            "sample_count": item["sample_count"],
            "chat_types": dict(item["chat_types"]),
            "length_buckets": dict(item["length_buckets"]),
            "element_types": dict(item["element_types"].most_common(SCENE_ELEMENT_TYPE_LIMIT)),
            "context_count_buckets": dict(item["context_count_buckets"]),
            "avg_score": round(item["score_total"] / sample_count, 1),
            "avg_reply_length": avg_reply_length,
            "question_ratio": round(item["question_replies"] / sample_count, 3),
            "exclamation_ratio": round(item["exclamation_replies"] / sample_count, 3),
            "ellipsis_ratio": round(item["ellipsis_replies"] / sample_count, 3),
            "quick_reply_ratio": round(item["quick_replies"] / sample_count, 3),
            "recommended_style": _scene_recommendation(item["scene_id"], avg_reply_length),
            "sample_refs": [ref for ref in item["sample_refs"] if ref],
        })
    profiles.sort(key=lambda item: (-int(item["sample_count"]), -float(item["avg_score"]), item["scene_id"]))
    return {
        "schema_version": 1,
        "raw_text_policy": "No raw chat text is stored in scene profiles.",
        "scene_count": len(profiles),
        "profiles": profiles,
    }


__all__ = [name for name in globals() if not name.startswith("__")]
