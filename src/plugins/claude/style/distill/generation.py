"""Retrieval-first owner-style generation helpers."""

from .qce_io import *
from .phrases import *
from .turns import *
from .taxonomy import *
from .reports import *
from .retrieval import *


GEN_DEFAULT_CANDIDATE_COUNT = DISTILL_SETTINGS.int_value("generation.retrieval_first.default_candidate_count", 8)
GEN_MIN_CANDIDATE_COUNT = DISTILL_SETTINGS.int_value("generation.retrieval_first.min_candidate_count", 3)
GEN_MAX_CANDIDATE_COUNT = DISTILL_SETTINGS.int_value("generation.retrieval_first.max_candidate_count", 8)
GEN_CONTEXT_PREVIEW_CHARS = DISTILL_SETTINGS.int_value("generation.retrieval_first.context_preview_chars", 1200)
GEN_SAMPLE_LIMIT = DISTILL_SETTINGS.int_value("generation.retrieval_first.sample_limit", 8)
GEN_SAMPLE_CONTEXT_CHARS = DISTILL_SETTINGS.int_value("generation.retrieval_first.sample_context_chars", 500)
GEN_SAMPLE_TARGET_CHARS = DISTILL_SETTINGS.int_value("generation.retrieval_first.sample_target_chars", 300)
GEN_LLM_TEMPERATURE = DISTILL_SETTINGS.float_value("generation.retrieval_first.llm_temperature", 0.8)
GEN_RERANK_INPUT_LIMIT = DISTILL_SETTINGS.int_value("generation.retrieval_first.rerank_input_limit", 12)
GEN_TARGET_ID_PREVIEW_CHARS = DISTILL_SETTINGS.int_value("generation.retrieval_first.target_id_preview_chars", 24)
GEN_PROMPT_PREVIEW_CHARS = DISTILL_SETTINGS.int_value("generation.retrieval_first.prompt_preview_chars", 2000)

PROFILE_PHRASE_MIN_CHARS = DISTILL_SETTINGS.int_value("generation.profile_phrases.min_chars", 2)
PROFILE_PHRASE_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.profile_phrases.max_chars", 16)
PROFILE_PHRASE_EXCLUDE_PREFIXES = DISTILL_SETTINGS.str_list("generation.profile_phrases.exclude_prefixes")
PUNCT_SHORT_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.punctuation.short_max_chars", 8)
STRUCTURE_LINE_MATCH_BONUS = DISTILL_SETTINGS.float_value("generation.punctuation.line_match_bonus", 0.08)
SHAPE_MICRO_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.shape.micro_max_chars", 4)
SHAPE_SHORT_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.shape.short_max_chars", 12)
SHAPE_MEDIUM_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.shape.medium_max_chars", 32)

LENGTH_EMPTY_PENALTY = DISTILL_SETTINGS.int_value("generation.length_fit.empty_penalty", -30)
LENGTH_TOLERANCE_MIN = DISTILL_SETTINGS.float_value("generation.length_fit.tolerance_min", 3.0)
LENGTH_TOLERANCE_MAX = DISTILL_SETTINGS.float_value("generation.length_fit.tolerance_max", 12.0)
LENGTH_TOLERANCE_RATIO = DISTILL_SETTINGS.float_value("generation.length_fit.tolerance_ratio", 0.55)
LENGTH_CLOSE_BONUS = DISTILL_SETTINGS.int_value("generation.length_fit.close_bonus", 18)
LENGTH_FAR_PENALTY_CAP = DISTILL_SETTINGS.int_value("generation.length_fit.far_penalty_cap", 24)
LENGTH_FAR_PENALTY_SCALE = DISTILL_SETTINGS.float_value("generation.length_fit.far_penalty_scale", 1.8)
LENGTH_OK_BONUS = DISTILL_SETTINGS.int_value("generation.length_fit.ok_bonus", 12)

HISTORY_FIT_LEXICAL_WEIGHT = DISTILL_SETTINGS.float_value("generation.history_fit.lexical_weight", 0.38)
HISTORY_FIT_KEYWORD_WEIGHT = DISTILL_SETTINGS.float_value("generation.history_fit.keyword_weight", 0.22)
HISTORY_FIT_STRUCTURE_WEIGHT = DISTILL_SETTINGS.float_value("generation.history_fit.structure_weight", 0.18)
HISTORY_FIT_LENGTH_WEIGHT = DISTILL_SETTINGS.float_value("generation.history_fit.length_weight", 0.1)
HISTORY_FIT_FREQUENCY_WEIGHT = DISTILL_SETTINGS.float_value("generation.history_fit.frequency_weight", 0.12)
HISTORY_FIT_STYLE_SCALE = DISTILL_SETTINGS.float_value("generation.history_fit.style_scale", 46)
HISTORY_FIT_SCENE_SCALE = DISTILL_SETTINGS.float_value("generation.history_fit.scene_scale", 18)
HISTORY_SHAPE_SCALE = DISTILL_SETTINGS.float_value("generation.history_fit.shape_scale", 22)

RERANK_DEFAULT_TARGET_MAX = DISTILL_SETTINGS.int_value("generation.rerank.default_target_max", 64)
RERANK_TARGET_MAX_MIN = DISTILL_SETTINGS.int_value("generation.rerank.target_max_min", 16)
RERANK_TARGET_MAX_CAP = DISTILL_SETTINGS.int_value("generation.rerank.target_max_cap", 160)
RERANK_TARGET_MAX_MULTIPLIER = DISTILL_SETTINGS.float_value("generation.rerank.target_max_multiplier", 3)
RERANK_HISTORY_TARGET_LIMIT = DISTILL_SETTINGS.int_value("generation.rerank.history_target_limit", 24)
RERANK_BASE_STYLE_SCORE = DISTILL_SETTINGS.int_value("generation.rerank.base_style_score", 48)
RERANK_BASE_SCENE_SCORE = DISTILL_SETTINGS.int_value("generation.rerank.base_scene_score", 48)
RERANK_ECHO_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.current_message_echo_penalty", 90)
RERANK_INVALID_TEXT_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.invalid_text_penalty", 90)
RERANK_TOO_LONG_PENALTY_CAP = DISTILL_SETTINGS.int_value("generation.rerank.too_long_penalty_cap", 36)
RERANK_STRUCTURED_ANSWER_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.structured_answer_penalty", 42)
RERANK_AI_LIKE_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.ai_like_penalty", 46)
RERANK_AI_MARKER_MIN_HITS = DISTILL_SETTINGS.int_value("generation.rerank.ai_marker_min_hits", 1)
RERANK_PROFILE_PHRASE_BASE_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.profile_phrase_base_bonus", 8)
RERANK_PROFILE_PHRASE_EACH_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.profile_phrase_each_bonus", 4)
RERANK_PROFILE_PHRASE_BONUS_CAP = DISTILL_SETTINGS.int_value("generation.rerank.profile_phrase_bonus_cap", 18)
RERANK_STRUCTURE_STRONG_THRESHOLD = DISTILL_SETTINGS.float_value("generation.rerank.structure_strong_threshold", 0.72)
RERANK_STRUCTURE_STRONG_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.structure_strong_bonus", 8)
RERANK_STRUCTURE_WEAK_THRESHOLD = DISTILL_SETTINGS.float_value("generation.rerank.structure_weak_threshold", 0.55)
RERANK_STRUCTURE_WEAK_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.structure_weak_bonus", 4)
RERANK_NO_PUNCT_SHORT_MAX_CHARS = DISTILL_SETTINGS.int_value("generation.rerank.no_punct_short_max_chars", 12)
RERANK_NO_PUNCT_SHORT_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.no_punct_short_bonus", 3)
RERANK_CREDENTIAL_RISK_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.credential_risk_penalty", 90)
RERANK_EXACT_LONG_COPY_MIN_CHARS = DISTILL_SETTINGS.int_value("generation.rerank.exact_long_copy_min_chars", 24)
RERANK_EXACT_LONG_COPY_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.exact_long_copy_penalty", 78)
RERANK_EXACT_SHORT_COPY_MIN_CHARS = DISTILL_SETTINGS.int_value("generation.rerank.exact_short_copy_min_chars", 2)
RERANK_EXACT_SHORT_COPY_BONUS = DISTILL_SETTINGS.int_value("generation.rerank.exact_short_copy_bonus", 10)
RERANK_NEAR_COPY_MIN_CHARS = DISTILL_SETTINGS.int_value("generation.rerank.near_copy_min_chars", 48)
RERANK_NEAR_COPY_THRESHOLD = DISTILL_SETTINGS.float_value("generation.rerank.near_copy_threshold", 0.96)
RERANK_NEAR_COPY_PENALTY = DISTILL_SETTINGS.int_value("generation.rerank.near_copy_penalty", 55)
RERANK_SCORE_STYLE_WEIGHT = DISTILL_SETTINGS.float_value("generation.rerank.score_style_weight", 0.58)
RERANK_SCORE_SCENE_WEIGHT = DISTILL_SETTINGS.float_value("generation.rerank.score_scene_weight", 0.28)
RERANK_SCORE_BASE = DISTILL_SETTINGS.float_value("generation.rerank.score_base", 18)
RERANK_ACCEPTED_THRESHOLD = DISTILL_SETTINGS.int_value("generation.rerank.accepted_threshold", 45)
RERANK_SENSITIVE_GRANT_PATTERNS = tuple(
    re.compile(pattern) for pattern in DISTILL_SETTINGS.str_list("generation.rerank.sensitive_grant_patterns")
)
RAW_FEWSHOT_EMBEDDING_SIMILARITY_FALLBACK = DISTILL_SETTINGS.float_value("generation.raw_fewshot.embedding_similarity_fallback", 0.43)
RAW_FEWSHOT_PAIR_CONTEXT_TAIL_TURNS = DISTILL_SETTINGS.int_value("generation.raw_fewshot.pair_context_tail_turns", 6)
RAW_FEWSHOT_CONTEXT_MAX_LINES = DISTILL_SETTINGS.int_value("generation.raw_fewshot.context_max_lines", 3)


def _candidate_ai_marker_hits(text: str) -> int:
    raw = str(text or "")
    compact = _compact_reply_text(raw)
    lowered = raw.lower()
    hits = 0
    for marker in AI_ASSISTANT_MARKERS:
        marker_text = str(marker or "").strip()
        if marker_text and (marker_text.lower() in lowered or marker_text in compact):
            hits += 1
    return hits


def build_retrieval_first_prompt(
    latest_message: str,
    *,
    current_context: str | Sequence[Dict[str, Any]] | None = None,
    retrieval: Dict[str, Any] | None = None,
    style_skill_context: Dict[str, Any] | None = None,
    candidate_count: int = GEN_DEFAULT_CANDIDATE_COUNT,
    include_raw_samples: bool = False,
) -> str:
    retrieval = retrieval or {}
    count = max(GEN_MIN_CANDIDATE_COUNT, min(GEN_MAX_CANDIDATE_COUNT, int(candidate_count or GEN_DEFAULT_CANDIDATE_COUNT)))
    lines = [
        f"你在生成主人聊天草稿。只输出 {count} 条候选回复，JSON 数组字符串，每条只含回复正文。",
        "要求：像真实聊天，不要像 AI 助手；不要解释；不要自称机器人；不要编造主人现实状态或承诺。",
        "相似真实样本是主要风格依据：学习语气、节奏、句式骨架和口癖；把与当前语境无关的具体事实实体替换掉。",
        f"当前场景：{retrieval.get('scene_label') or 'unknown'}",
    ]
    skill_prompt = format_style_skill_context_for_prompt(style_skill_context)
    if skill_prompt:
        lines.append(skill_prompt)
    context_text = _normalize_current_context(current_context, latest_message)
    if context_text:
        lines.append("当前聊天上下文：")
        lines.append(context_text[:GEN_CONTEXT_PREVIEW_CHARS])
    samples = retrieval.get("results") or []
    if samples:
        if include_raw_samples:
            lines.append("相似真实样本（已授权原句 few-shot）：")
        else:
            lines.append("相似样本元数据（未授权原句 few-shot，不含历史原文）：")
        for index, item in enumerate(samples[:GEN_SAMPLE_LIMIT], start=1):
            taxonomy = item.get("taxonomy") or {}
            target = (item.get("target") or {})
            lines.append(
                f"样本{index} scene={item.get('scene_label')} same_rel={item.get('same_relationship')} "
                f"scope={taxonomy.get('scope') or 'unknown'} grounding={taxonomy.get('grounding_type') or 'unknown'} "
                f"reply_len={target.get('char_length') or len(str(target.get('text') or ''))}"
            )
            if include_raw_samples:
                sample_context = _turns_text(item.get("context") or [])
                target_text = target.get("text") or ""
                if sample_context:
                    lines.append("历史上下文：" + sample_context[:GEN_SAMPLE_CONTEXT_CHARS])
                if target_text:
                    lines.append("主人真实回复：" + str(target_text)[:GEN_SAMPLE_TARGET_CHARS])
    lines.append(f"生成 {count} 条候选，彼此要有差异，但都保持口语、自然、短。")
    return "\n".join(lines)


def _compact_reply_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip("。.!！?？~～\"'“”"))


def _commitment_risk(intent: Dict[str, Any]) -> tuple[int, str]:
    try:
        level = int(intent.get("commitment_risk_level") or 0)
    except (TypeError, ValueError):
        level = 0
    return max(0, min(3, level)), str(intent.get("commitment_risk_label") or "phatic_social")


def _profile_phrase_set(style_profile: Dict[str, Any] | None) -> set[str]:
    phrases: set[str] = set()
    if not isinstance(style_profile, dict):
        return phrases
    for item in style_profile.get("common_phrases") or []:
        text = _compact_reply_text(str(item or ""))
        if PROFILE_PHRASE_MIN_CHARS <= len(text) <= PROFILE_PHRASE_MAX_CHARS:
            phrases.add(text)
    for item in style_profile.get("habits") or []:
        for part in re.split(r"[、,，；;:\s]+", str(item or "")):
            text = _compact_reply_text(part)
            if (
                PROFILE_PHRASE_MIN_CHARS <= len(text) <= PROFILE_PHRASE_MAX_CHARS
                and not text.startswith(PROFILE_PHRASE_EXCLUDE_PREFIXES)
            ):
                phrases.add(text)
    return phrases


def _punctuation_signature(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    compact = _compact_reply_text(raw)
    return {
        "no_punct": not bool(re.search(r"[，,。.!！?？；;~～…]", raw)),
        "question": "?" in raw or "？" in raw,
        "exclaim": "!" in raw or "！" in raw,
        "ellipsis": "..." in raw or "…" in raw,
        "wave": "~" in raw or "～" in raw,
        "line_count": max(1, raw.count("\n") + 1),
        "short": len(compact) <= PUNCT_SHORT_MAX_CHARS,
    }


def _structural_similarity(left: str, right: str) -> float:
    left_sig = _punctuation_signature(left)
    right_sig = _punctuation_signature(right)
    keys = ("no_punct", "question", "exclaim", "ellipsis", "wave", "short")
    matches = sum(1 for key in keys if left_sig.get(key) == right_sig.get(key))
    score = matches / max(1, len(keys))
    if left_sig["line_count"] == right_sig["line_count"]:
        score += STRUCTURE_LINE_MATCH_BONUS
    return min(1.0, score)


def _length_fit_score(text: str, target_length: float | None, target_max: int) -> tuple[int, str]:
    compact_len = len(_compact_reply_text(text))
    if compact_len <= 0:
        return LENGTH_EMPTY_PENALTY, "empty_length"
    if target_length and target_length > 0:
        diff = abs(compact_len - float(target_length))
        tolerance = max(LENGTH_TOLERANCE_MIN, min(LENGTH_TOLERANCE_MAX, float(target_length) * LENGTH_TOLERANCE_RATIO))
        if diff <= tolerance:
            return LENGTH_CLOSE_BONUS, "length_close_to_profile"
        return -min(LENGTH_FAR_PENALTY_CAP, int(round((diff - tolerance) * LENGTH_FAR_PENALTY_SCALE))), "length_far_from_profile"
    if 2 <= compact_len <= target_max:
        return LENGTH_OK_BONUS, "length_ok"
    return -min(LENGTH_FAR_PENALTY_CAP, abs(compact_len - target_max)), "length_off"


def _shape_label(text: str) -> str:
    compact = _compact_reply_text(text)
    if not compact:
        return "empty"
    if "\n" in str(text or ""):
        line = "multi"
    else:
        line = "single"
    if "?" in str(text or "") or "？" in str(text or ""):
        mood = "question"
    elif "!" in str(text or "") or "！" in str(text or ""):
        mood = "exclaim"
    else:
        mood = "plain"
    length = len(compact)
    if length <= SHAPE_MICRO_MAX_CHARS:
        bucket = "micro"
    elif length <= SHAPE_SHORT_MAX_CHARS:
        bucket = "short"
    elif length <= SHAPE_MEDIUM_MAX_CHARS:
        bucket = "medium"
    else:
        bucket = "long"
    punct = "no_punct" if _punctuation_signature(text).get("no_punct") else "punct"
    return f"{bucket}_{mood}_{line}_{punct}"


def _grants_sensitive_request(latest_message: str, candidate: str) -> bool:
    if not str(latest_message or "").strip():
        return False
    latest_intent = detect_message_intent(latest_message)
    if not latest_intent.get("high_risk_request"):
        return False
    if contains_sensitive_content(candidate):
        return True
    compact = _compact_reply_text(candidate)
    return any(
        pattern.search(compact) for pattern in RERANK_SENSITIVE_GRANT_PATTERNS
    )


def classify_reply_behavior(text: str, *, latest_message: str = "") -> Dict[str, Any]:
    """Compatibility helper for debug/tests.

    Stage 5C no longer classifies domain-specific behavior such as game
    accept/decline/defer by hand. The label is now a generic surface-shape
    bucket; only credential-like grants are treated as unsafe.
    """
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
    risk_level, risk_label = _commitment_risk(latest_intent)
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”")
    result: Dict[str, Any] = {
        "label": _shape_label(cleaned),
        "safe_for_context": True,
        "reasons": [],
        "commitment_risk_level": risk_level,
        "commitment_risk_label": risk_label,
    }
    if not cleaned:
        return {
            "label": "empty",
            "safe_for_context": False,
            "reasons": ["empty"],
            "commitment_risk_level": risk_level,
            "commitment_risk_label": risk_label,
        }

    if _grants_sensitive_request(latest_message, cleaned):
        result.update({"label": "credential_share_risk", "safe_for_context": False})
        result["reasons"].append("credential_or_finance_commitment")
    return result


def historical_behavior_distribution(
    historical_targets: Sequence[str] | None,
    *,
    latest_message: str = "",
) -> Dict[str, Any]:
    """Summarize generic response-shape preferences from retrieved replies."""
    counter: Counter[str] = Counter()
    examples: Dict[str, List[str]] = {}
    for target in historical_targets or []:
        text = re.sub(r"\s+", " ", str(target or "")).strip()
        if not text:
            continue
        label = _shape_label(text)
        counter[label] += 1
        examples.setdefault(label, [])
        if len(examples[label]) < 3:
            examples[label].append(text[:24])
    total = sum(counter.values())
    if total <= 0:
        return {
            "sample_count": 0,
            "counts": {},
            "proportions": {},
            "dominant": "",
            "examples": {},
        }
    return {
        "sample_count": total,
        "counts": dict(counter),
        "proportions": {label: round(count / total, 4) for label, count in counter.items()},
        "dominant": counter.most_common(1)[0][0],
        "examples": examples,
    }


def _historical_behavior_delta(
    label: str,
    distribution: Dict[str, Any],
    *,
    latest_intent: Dict[str, Any],
) -> tuple[int, str]:
    sample_count = int(distribution.get("sample_count") or 0)
    if sample_count <= 0:
        return 0, "no_behavior_history"
    proportions = distribution.get("proportions") or {}
    prop = float(proportions.get(label) or 0.0)
    if prop <= 0:
        return 0, f"history_shape_unseen:{label}"
    return int(round(HISTORY_SHAPE_SCALE * prop)), f"history_shape_match:{label}:{prop:.2f}"


def _history_fit(candidate: str, historical_targets: Sequence[str]) -> Dict[str, Any]:
    candidate_text = str(candidate or "")
    if not candidate_text or not historical_targets:
        return {
            "score": 0.0,
            "lexical": 0.0,
            "keyword": 0.0,
            "structure": 0.0,
            "length": 0.0,
            "frequency": 0.0,
            "best": "",
        }
    candidate_ngrams = _text_ngrams(candidate_text)
    candidate_keywords = _keyword_tokens(candidate_text)
    candidate_len = max(1, len(_compact_reply_text(candidate_text)))
    best = {
        "score": 0.0,
        "lexical": 0.0,
        "keyword": 0.0,
        "structure": 0.0,
        "length": 0.0,
        "frequency": 0.0,
        "best": "",
    }
    candidate_compact = _compact_reply_text(candidate_text)
    target_compacts = [_compact_reply_text(target) for target in historical_targets]
    exact_count = sum(1 for target in target_compacts if target and target == candidate_compact)
    frequency = min(1.0, exact_count / max(1, len(target_compacts)))
    for target in historical_targets:
        target_text = str(target or "")
        if not target_text:
            continue
        target_len = max(1, len(_compact_reply_text(target_text)))
        lexical = _jaccard(candidate_ngrams, _text_ngrams(target_text))
        keyword = _jaccard(candidate_keywords, _keyword_tokens(target_text))
        structure = _structural_similarity(candidate_text, target_text)
        length = max(0.0, 1.0 - abs(candidate_len - target_len) / max(candidate_len, target_len, 1))
        score = round(
            lexical * HISTORY_FIT_LEXICAL_WEIGHT
            + keyword * HISTORY_FIT_KEYWORD_WEIGHT
            + structure * HISTORY_FIT_STRUCTURE_WEIGHT
            + length * HISTORY_FIT_LENGTH_WEIGHT
            + frequency * HISTORY_FIT_FREQUENCY_WEIGHT,
            4,
        )
        if score > best["score"]:
            best = {
                "score": score,
                "lexical": round(lexical, 4),
                "keyword": round(keyword, 4),
                "structure": round(structure, 4),
                "length": round(length, 4),
                "frequency": round(frequency, 4),
                "best": target_text[:32],
            }
    return best

def style_rerank_candidates(
    candidates: Sequence[str],
    *,
    scene_label: str = "",
    max_length: int | None = None,
    target_length: float | None = None,
    style_profile: Dict[str, Any] | None = None,
    corrections: Sequence[Dict[str, Any]] | None = None,
    historical_targets: Sequence[str] | None = None,
    latest_message: str = "",
) -> List[Dict[str, Any]]:
    """Rerank candidates by similarity to retrieved owner replies.

    The main path is history-driven: history text fit, shape distribution,
    profile phrases, length fit, and correction feedback. Hard rejection is
    limited to malformed output, current-message echoing, credential-like
    leakage, and long raw-history copying.
    """
    if max_length is not None:
        target_max = max(1, int(max_length))
    elif target_length and float(target_length) > 0:
        target_max = max(
            RERANK_TARGET_MAX_MIN,
            min(RERANK_TARGET_MAX_CAP, int(round(float(target_length) * RERANK_TARGET_MAX_MULTIPLIER))),
        )
    else:
        target_max = RERANK_DEFAULT_TARGET_MAX
    correction_items = list(corrections or [])
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
    commitment_level, commitment_label = _commitment_risk(latest_intent)
    history_texts = [
        re.sub(r"\s+", " ", str(item or "")).strip()
        for item in (historical_targets or [])
        if str(item or "").strip()
    ][:RERANK_HISTORY_TARGET_LIMIT]
    behavior_distribution = historical_behavior_distribution(history_texts, latest_message=latest_message)
    profile_phrases = _profile_phrase_set(style_profile)
    ranked = []
    for raw in candidates:
        text = re.sub(r"\s+", " ", str(raw or "")).strip().strip("\"'“”")
        if text.startswith("[") and text.endswith("]"):
            try:
                nested = json.loads(text)
                if isinstance(nested, list) and nested:
                    text = re.sub(r"\s+", " ", str(nested[0] or "")).strip().strip("\"'“”")
            except Exception:
                pass
        if not text:
            continue
        style_score = RERANK_BASE_STYLE_SCORE
        scene_score = RERANK_BASE_SCENE_SCORE
        risk_penalty = 0
        hygiene_penalty = 0
        reasons = []
        persona_reasons = []
        scene_reasons = []
        risk_reasons = []
        hard_reject_reasons = []
        compact_latest = _compact_reply_text(latest_message)
        compact_candidate = _compact_reply_text(text)
        if compact_latest and compact_candidate == compact_latest:
            hygiene_penalty += RERANK_ECHO_PENALTY
            reasons.append("copied_current_message")
            hard_reject_reasons.append("copied_current_message")
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
            hygiene_penalty += RERANK_INVALID_TEXT_PENALTY
            reasons.append("invalid_text")
            hard_reject_reasons.append("invalid_text")
        length_delta, length_reason = _length_fit_score(text, target_length, target_max)
        style_score += length_delta
        if length_delta >= 0:
            persona_reasons.append(length_reason)
        else:
            reasons.append(length_reason)
        if len(text) > target_max:
            excess = min(RERANK_TOO_LONG_PENALTY_CAP, len(text) - target_max)
            hygiene_penalty += excess
            reasons.append("too_long")
        if re.search(r"(^|\n)\s*(?:#{1,4}\s|[-*]\s+|\d+[.)、]\s+)", text):
            hygiene_penalty += RERANK_STRUCTURED_ANSWER_PENALTY
            reasons.append("structured_answer")
            hard_reject_reasons.append("structured_answer")
        ai_marker_hits = _candidate_ai_marker_hits(text)
        if is_ai_assistant_generated_text(text) or ai_marker_hits >= RERANK_AI_MARKER_MIN_HITS:
            hygiene_penalty += RERANK_AI_LIKE_PENALTY
            reasons.append("ai_assistant_flavor")
        phrase_hits = [
            phrase for phrase in profile_phrases
            if phrase and (phrase in compact_candidate or compact_candidate in phrase)
        ][:3]
        if phrase_hits:
            style_score += min(
                RERANK_PROFILE_PHRASE_BONUS_CAP,
                RERANK_PROFILE_PHRASE_BASE_BONUS + len(phrase_hits) * RERANK_PROFILE_PHRASE_EACH_BONUS,
            )
            persona_reasons.append("phrase_overlap:" + "/".join(phrase_hits))
        history_fit = _history_fit(text, history_texts)
        if history_fit["score"] > 0:
            history_delta = int(round(float(history_fit["score"]) * HISTORY_FIT_STYLE_SCALE))
            style_score += history_delta
            scene_score += int(round(float(history_fit["score"]) * HISTORY_FIT_SCENE_SCALE))
            persona_reasons.append(
                "history_fit:"
                f"{history_fit['score']:.2f}/lex={history_fit['lexical']:.2f}"
                f"/kw={history_fit['keyword']:.2f}/shape={history_fit['structure']:.2f}"
                f"/freq={history_fit['frequency']:.2f}"
            )
        if history_texts:
            structure_match = max((_structural_similarity(text, target) for target in history_texts), default=0.0)
            if structure_match >= RERANK_STRUCTURE_STRONG_THRESHOLD:
                style_score += RERANK_STRUCTURE_STRONG_BONUS
                persona_reasons.append("structure_like_history")
            elif structure_match >= RERANK_STRUCTURE_WEAK_THRESHOLD:
                style_score += RERANK_STRUCTURE_WEAK_BONUS
                persona_reasons.append("structure_somewhat_like_history")
        if _punctuation_signature(text).get("no_punct") and len(compact_candidate) <= RERANK_NO_PUNCT_SHORT_MAX_CHARS:
            style_score += RERANK_NO_PUNCT_SHORT_BONUS
            persona_reasons.append("chat_like_no_punct")
        if _grants_sensitive_request(latest_message, text):
            risk_penalty += RERANK_CREDENTIAL_RISK_PENALTY
            reasons.append("credential_share_risk")
            risk_reasons.append("credential_share_risk")
            hard_reject_reasons.append("credential_share_risk")
        behavior = classify_reply_behavior(text, latest_message=latest_message)
        behavior_delta, behavior_reason = _historical_behavior_delta(
            str(behavior.get("label") or "neutral"),
            behavior_distribution,
            latest_intent=latest_intent,
        )
        if behavior_delta:
            scene_score += behavior_delta
            scene_reasons.append(behavior_reason)
        compact_len = len(compact_candidate)
        if compact_len >= RERANK_EXACT_LONG_COPY_MIN_CHARS and text in history_texts:
            hygiene_penalty += RERANK_EXACT_LONG_COPY_PENALTY
            reasons.append("copied_long_history_exact")
            hard_reject_reasons.append("copied_long_history_exact")
        elif compact_len >= RERANK_EXACT_SHORT_COPY_MIN_CHARS and text in history_texts:
            style_score += RERANK_EXACT_SHORT_COPY_BONUS
            reasons.append("reused_history_phrase")
            persona_reasons.append("historical_phrase_reuse")
        elif compact_len >= RERANK_NEAR_COPY_MIN_CHARS and history_texts:
            similarity = max(
                (_jaccard(_text_ngrams(text), _text_ngrams(target)) for target in history_texts),
                default=0.0,
            )
            if similarity >= RERANK_NEAR_COPY_THRESHOLD:
                hygiene_penalty += RERANK_NEAR_COPY_PENALTY
                reasons.append("copied_long_history_near")
                hard_reject_reasons.append("copied_long_history_near")
        correction_delta, correction_reasons = candidate_correction_delta(text, correction_items)
        if correction_delta:
            style_score += correction_delta
            reasons.extend(correction_reasons)
        score = int(round(
            style_score * RERANK_SCORE_STYLE_WEIGHT
            + scene_score * RERANK_SCORE_SCENE_WEIGHT
            + RERANK_SCORE_BASE
            - risk_penalty
            - hygiene_penalty
        ))
        hard_reject = bool(hard_reject_reasons)
        accepted = score >= RERANK_ACCEPTED_THRESHOLD and not hard_reject
        ranked.append({
            "text": text,
            "score": score,
            "style_score": int(round(style_score)),
            "scene_score": int(round(scene_score)),
            "risk_penalty": int(round(risk_penalty)),
            "hygiene_penalty": int(round(hygiene_penalty)),
            "commitment_risk_level": commitment_level,
            "commitment_risk_label": commitment_label,
            "reasons": reasons,
            "persona_reasons": persona_reasons,
            "scene_reasons": scene_reasons,
            "risk_reasons": risk_reasons,
            "hard_reject_reasons": hard_reject_reasons,
            "hard_reject": hard_reject,
            "accepted": accepted,
            "behavior": behavior.get("label"),
            "behavior_safe": bool(behavior.get("safe_for_context")),
            "behavior_reasons": behavior.get("reasons") or [],
            "history_fit": history_fit,
            "behavior_distribution": {
                "sample_count": behavior_distribution.get("sample_count", 0),
                "counts": behavior_distribution.get("counts", {}),
                "dominant": behavior_distribution.get("dominant", ""),
            },
        })
    ranked.sort(key=lambda item: (-int(item["score"]), len(item["text"]), item["text"]))
    return ranked

async def generate_retrieval_first_reply_candidates(
    latest_message: str,
    *,
    current_context: str | Sequence[Dict[str, Any]] | None = None,
    run_dir: str | Path | None = None,
    chat_type: str = "private",
    target_id: str | int | None = None,
    include_raw_samples: bool = False,
) -> Dict[str, Any]:
    """Offline retrieval-first draft prototype. It does not send QQ messages."""
    retrieval = retrieve_dialogue_pair_samples(
        latest_message,
        current_context=current_context,
        run_dir=run_dir,
        chat_type=chat_type,
        target_id=target_id,
        limit=GEN_DEFAULT_CANDIDATE_COUNT,
    )
    if not retrieval.get("ok"):
        return retrieval
    style_skill_context = load_style_skill_context(
        chat_type=chat_type,
        target_id=target_id,
        scene_label=str(retrieval.get("scene_label") or ""),
        latest_message=latest_message,
    )
    prompt = build_retrieval_first_prompt(
        latest_message,
        current_context=current_context,
        retrieval=retrieval,
        style_skill_context=style_skill_context,
        candidate_count=GEN_DEFAULT_CANDIDATE_COUNT,
        include_raw_samples=include_raw_samples,
    )
    from .api import llm_client
    raw = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=f"你只输出 JSON 数组字符串，数组里 {GEN_DEFAULT_CANDIDATE_COUNT} 个中文聊天候选回复。",
        temperature=GEN_LLM_TEMPERATURE,
    )
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("candidates") or parsed.get("replies") or []
        candidates = []
        for item in parsed:
            if isinstance(item, str):
                text = item.strip()
                if text.startswith("[") and text.endswith("]"):
                    try:
                        nested = json.loads(text)
                        if isinstance(nested, list):
                            candidates.extend(str(nested_item) for nested_item in nested if isinstance(nested_item, str))
                            continue
                    except Exception:
                        pass
                candidates.append(text)
    except Exception:
        candidates = [line.strip("- 0123456789.、") for line in raw.splitlines() if line.strip()]
    historical_targets = [
        str((item.get("target") or {}).get("text") or "")
        for item in (retrieval.get("results") or [])
        if isinstance(item, dict)
    ]
    reranked = style_rerank_candidates(
        candidates[:GEN_RERANK_INPUT_LIMIT],
        scene_label=str(retrieval.get("scene_label") or ""),
        corrections=style_skill_context.get("corrections") or [],
        historical_targets=historical_targets,
        latest_message=latest_message,
    )
    return {
        "ok": True,
        "run_id": retrieval.get("run_id"),
        "scene_label": retrieval.get("scene_label"),
        "retrieval": {
            "result_count": retrieval.get("result_count"),
            "target_mapping": retrieval.get("target_mapping"),
            "raw_samples_in_prompt": bool(include_raw_samples),
        },
        "style_skill": {
            "enabled": bool(style_skill_context.get("enabled")),
            "relationship_profile_found": bool(style_skill_context.get("relationship_profile_found")),
            "correction_hit_count": int(style_skill_context.get("correction_hit_count") or 0),
            "target_id": str(target_id or "")[:GEN_TARGET_ID_PREVIEW_CHARS],
        },
        "candidates": reranked,
        "prompt_preview": prompt[:GEN_PROMPT_PREVIEW_CHARS],
    }

def _clean_raw_fewshot_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""
    cleaned = RAW_URL_RE.sub("[链接]", cleaned)
    cleaned = EMAIL_RE.sub("[邮箱]", cleaned)
    cleaned = PHONE_RE.sub("[手机号]", cleaned)
    cleaned = ID_CARD_RE.sub("[证件号]", cleaned)
    cleaned = LONG_ID_RE.sub("[数字ID]", cleaned)
    if contains_sensitive_content(cleaned):
        return ""
    return cleaned[:MAX_RAW_FEWSHOT_TEXT_CHARS]

def _build_raw_few_shot_examples(
    catalog: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    retrieval_results: Sequence[Dict[str, Any]],
    *,
    limit: int = DEFAULT_RAW_FEWSHOT_LIMIT,
    preferred_chat_type: str | None = None,
    run_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Extract real historical snippets for owner-authorized few-shot prompts.

    The returned examples are intended for immediate prompt construction only
    and must not be persisted or logged.
    """
    by_sample_id = {
        str(sample.get("sample_id") or ""): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    pair_lookup: Dict[str, Dict[str, Any]] = {}
    cache: Dict[str, List[Dict[str, Any]]] = {}
    examples = []
    for result in retrieval_results:
        embedding_similarity = _safe_float(result.get("embedding_similarity"))
        if (
            _safe_float(result.get("similarity")) < MIN_RAW_FEWSHOT_SIMILARITY
            and embedding_similarity < RAW_FEWSHOT_EMBEDDING_SIMILARITY_FALLBACK
        ):
            continue
        if (
            _safe_float(result.get("context_overlap")) < MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD
            and _safe_float(result.get("keyword_overlap")) < MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD
            and embedding_similarity < RAW_FEWSHOT_EMBEDDING_SIMILARITY_FALLBACK
        ):
            continue
        if preferred_chat_type and result.get("chat_type") != preferred_chat_type:
            continue
        sample_id = str(result.get("sample_id") or result.get("pair_id") or "")
        sample = by_sample_id.get(sample_id)
        if not sample:
            if not run_path:
                continue
            if not pair_lookup:
                pair_lookup = {
                    str(pair.get("pair_id") or ""): pair
                    for pair in _load_dialogue_pairs(run_path)
                    if isinstance(pair, dict) and str(pair.get("pair_id") or "").strip()
                }
            pair = pair_lookup.get(str(result.get("pair_id") or sample_id))
            if not pair:
                continue
            context_lines = []
            for turn in (pair.get("context") or [])[-RAW_FEWSHOT_PAIR_CONTEXT_TAIL_TURNS:]:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role") or "")
                if role not in {"other", "self"}:
                    continue
                text = _clean_raw_fewshot_text(turn.get("text") or "\n".join(turn.get("raw_texts") or []))
                if not text:
                    continue
                label = "主人" if role == "self" else "对方"
                context_lines.append({"role": label, "text": text})
            if len(context_lines) > RAW_FEWSHOT_CONTEXT_MAX_LINES:
                context_lines = context_lines[-RAW_FEWSHOT_CONTEXT_MAX_LINES:]
            target = pair.get("target") if isinstance(pair.get("target"), dict) else {}
            owner_reply = _clean_raw_fewshot_text(target.get("text") or "\n".join(target.get("raw_texts") or []))
            if not owner_reply or not context_lines:
                continue
            examples.append({
                "sample_id": sample_id,
                "pair_id": pair.get("pair_id"),
                "source_file_id": pair.get("source_file_id"),
                "chat_type": pair.get("chat_type"),
                "similarity": result.get("similarity"),
                "quality_score": result.get("quality_score"),
                "retrieval_source": result.get("retrieval_source"),
                "context": context_lines,
                "owner_reply": owner_reply,
                "char_counts": {
                    "context": sum(len(item.get("text") or "") for item in context_lines),
                    "owner_reply": len(owner_reply),
                },
            })
            if len(examples) >= limit:
                break
            continue
        source_file_id = str(sample.get("source_file_id") or "")
        messages = _load_source_messages(catalog, source_file_id, cache)
        if not messages:
            continue

        context_lines = []
        for ref in (sample.get("context") or {}).get("messages") or []:
            role = str(ref.get("role") or "")
            if role not in {"other", "self"}:
                continue
            try:
                record_index = int(ref.get("record_index"))
            except (TypeError, ValueError):
                continue
            if record_index < 0 or record_index >= len(messages):
                continue
            text = _clean_raw_fewshot_text(_message_text(messages[record_index]))
            if not text:
                continue
            label = "主人" if role == "self" else "对方"
            context_lines.append({"role": label, "text": text})
        if len(context_lines) > RAW_FEWSHOT_CONTEXT_MAX_LINES:
            context_lines = context_lines[-RAW_FEWSHOT_CONTEXT_MAX_LINES:]

        reply_ref = sample.get("reply") or {}
        reply_texts = []
        reply_message_refs = reply_ref.get("messages") or []
        if isinstance(reply_message_refs, list) and reply_message_refs:
            for ref in reply_message_refs:
                if not isinstance(ref, dict):
                    continue
                reply_index = _safe_int(ref.get("record_index"), -1)
                if 0 <= reply_index < len(messages):
                    text = _clean_raw_fewshot_text(_message_text(messages[reply_index]))
                    if text:
                        reply_texts.append(text)
        else:
            reply_index = _safe_int(reply_ref.get("record_index"), -1)
            if 0 <= reply_index < len(messages):
                text = _clean_raw_fewshot_text(_message_text(messages[reply_index]))
                if text:
                    reply_texts.append(text)
        owner_reply = "\n".join(reply_texts).strip()
        if not owner_reply:
            continue
        if not context_lines:
            continue

        examples.append({
            "sample_id": sample_id,
            "pair_id": result.get("pair_id") or sample.get("pair_id"),
            "source_file_id": source_file_id,
            "chat_type": sample.get("chat_type"),
            "similarity": result.get("similarity"),
            "quality_score": result.get("quality_score"),
            "retrieval_source": result.get("retrieval_source"),
            "context": context_lines,
            "owner_reply": owner_reply,
            "char_counts": {
                "context": sum(len(item.get("text") or "") for item in context_lines),
                "owner_reply": len(owner_reply),
            },
        })
        if len(examples) >= limit:
            break
    return examples

def build_style_generation_context(
    query: str,
    *,
    run_dir: str | Path | None = None,
    limit: int = DEFAULT_GENERATION_CONTEXT_LIMIT,
    chat_type: str | None = None,
    target_id: str | int | None = None,
    include_raw_fewshot: bool = False,
    raw_fewshot_limit: int = DEFAULT_RAW_FEWSHOT_LIMIT,
) -> Dict[str, Any]:
    """Build a Stage 5B context for style draft generation.

    Raw historical snippets are included only when include_raw_fewshot=True.
    """
    query_text = query.strip()
    if not query_text:
        return {"ok": False, "message": "用法：/用我的风格回复：<对方消息>"}

    try:
        run_path = find_latest_distill_run(run_dir)
        if run_path is None:
            raise FileNotFoundError("还没有找到 Stage 5B 离线蒸馏结果。")
        target_mapping = find_source_for_target(target_id, chat_type=chat_type, run_dir=run_path) if target_id else {"matched": False}
        preferred_source = str(target_mapping.get("source_file_id") or "") if target_mapping.get("matched") else ""
        run_path, catalog, samples = _resolve_run_paths(run_path)
        retrieval = retrieve_similar_style_samples(
            query_text,
            run_dir=run_path,
            limit=limit,
            preferred_source_file_id=preferred_source or None,
            preferred_chat_type=chat_type,
        )
        relationship_path = run_path / "relationship_profiles.json"
        scene_path = run_path / "scene_profiles.json"
        evaluation_path = run_path / "evaluation_report.json"
        relationships = _load_json(relationship_path) if relationship_path.exists() else {}
        scenes = _load_json(scene_path) if scene_path.exists() else {}
        evaluation = _load_json(evaluation_path) if evaluation_path.exists() else {}
    except Exception as e:
        return {
            "ok": False,
            "message": f"Stage 5B 生成上下文不可用：{type(e).__name__}",
            "raw_text_policy": "No raw historical text is returned.",
        }

    if not retrieval.get("ok"):
        return retrieval

    retrieval_results = list(retrieval.get("results") or [])
    query_features = retrieval.get("query_features") or _features_for_text(query_text)
    relationship_profiles = _select_relationship_profiles(relationships, retrieval_results)
    if target_mapping.get("matched"):
        mapped_source = str(target_mapping.get("source_file_id") or "")
        profiles = relationships.get("profiles") or []
        mapped_profile = next(
            (
                item for item in profiles
                if isinstance(item, dict) and str(item.get("source_file_id") or "") == mapped_source
            ),
            None,
        )
        if mapped_profile and all(item.get("source_file_id") != mapped_source for item in relationship_profiles):
            relationship_profiles = _select_relationship_profiles(
                {"profiles": [mapped_profile]},
                [{"source_file_id": mapped_source}],
                limit=1,
            ) + relationship_profiles
            relationship_profiles = relationship_profiles[:3]
    scene_profiles = _select_scene_profiles(
        scenes,
        retrieval_results,
        query_features,
        chat_type=chat_type,
    )
    guidance = _derive_generation_guidance(
        query_features,
        retrieval_results,
        scene_profiles,
        relationship_profiles,
    )
    raw_examples = []
    if include_raw_fewshot:
        raw_examples = _build_raw_few_shot_examples(
            catalog,
            samples,
            retrieval_results,
            limit=max(0, min(DEFAULT_RAW_FEWSHOT_LIMIT, int(raw_fewshot_limit))),
            preferred_chat_type=chat_type,
            run_path=run_path,
        )

    return {
        "ok": True,
        "run_id": run_path.name,
        "readiness": evaluation.get("readiness") or "unknown",
        "raw_text_policy": (
            "Historical text may be read locally for similarity scoring. Raw few-shot examples are included "
            "only when explicitly authorized for immediate prompt construction."
        ),
        "raw_fewshot_included": bool(raw_examples),
        "target_mapping": {
            "matched": bool(target_mapping.get("matched")),
            "source_file_id": target_mapping.get("source_file_id") if target_mapping.get("matched") else "",
            "chat_type": target_mapping.get("chat_type") if target_mapping.get("matched") else str(chat_type or ""),
            "identifier_policy": "Target QQ/group id is used locally but not included in prompts.",
        },
        "query_features": query_features,
        "retrieval_strategy": retrieval.get("retrieval_strategy") or "rules_only",
        "embedding_status": retrieval.get("embedding_status") or {},
        "similar_samples": retrieval_results,
        "relationship_profiles": relationship_profiles,
        "scene_profiles": scene_profiles,
        "guidance": guidance,
        "few_shot_examples": raw_examples,
    }


__all__ = [name for name in globals() if not name.startswith("__")]
