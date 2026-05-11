"""Phrase profile construction for QCE style distillation."""

from .qce_io import *


PHRASE_SHORT_MIN_CHARS = DISTILL_SETTINGS.int_value("phrases.observation.short_min_chars", 2)
PHRASE_SHORT_MAX_CHARS = DISTILL_SETTINGS.int_value("phrases.observation.short_max_chars", 14)
PHRASE_ENDING_MIN_CHARS = DISTILL_SETTINGS.int_value("phrases.observation.ending_min_chars", 2)
PHRASE_ENDING_MAX_CHARS = DISTILL_SETTINGS.int_value("phrases.observation.ending_max_chars", 4)
PHRASE_TEMPLATE_MAX_CHARS = DISTILL_SETTINGS.int_value("phrases.observation.template_max_chars", 24)
PHRASE_DEFAULT_TOP_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.default_top_limit", 30)
PHRASE_TOP_SCAN_MULTIPLIER = DISTILL_SETTINGS.int_value("phrases.output.top_scan_multiplier", 2)
PHRASE_MIN_SHORT_COUNT = DISTILL_SETTINGS.int_value("phrases.output.min_short_count", 2)
PHRASE_HIGH_FREQ_SHORT_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.high_freq_short_limit", 50)
PHRASE_ENDING_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.ending_limit", 40)
PHRASE_PARTICLE_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.particle_limit", 40)
PHRASE_TEMPLATE_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.template_limit", 30)
PHRASE_COMMON_DEFAULT_LIMIT = DISTILL_SETTINGS.int_value("phrases.output.common_default_limit", 12)
PHRASE_COMMON_MAX_CHARS = DISTILL_SETTINGS.int_value("phrases.output.common_max_chars", 40)
RHETORICAL_PATTERN_RE = re.compile(DISTILL_SETTINGS.str_value("phrases.patterns.rhetorical", r"(?!)"))
RHETORICAL_DIGIT_PLACEHOLDER = DISTILL_SETTINGS.str_value("phrases.patterns.digit_placeholder", "N")


def _new_phrase_bucket() -> Dict[str, Counter]:
    return {
        "short_replies": Counter(),
        "endings": Counter(),
        "particles": Counter(),
        "rhetorical_patterns": Counter(),
        "negative_templates": Counter(),
        "confirmation_templates": Counter(),
    }

def _phrase_scope_key(chat_type: str, relationship_id: str) -> str:
    return f"{chat_type}:{relationship_id}"

def _add_phrase_observation(bucket: Dict[str, Counter], text: str) -> None:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact or contains_sensitive_content(compact):
        return
    normalized = compact.strip()
    no_space = re.sub(r"\s+", "", normalized)
    if PHRASE_SHORT_MIN_CHARS <= len(no_space) <= PHRASE_SHORT_MAX_CHARS and not URL_RE.search(no_space):
        bucket["short_replies"][no_space] += 1
    if len(no_space) >= PHRASE_ENDING_MIN_CHARS:
        tail = no_space[-min(PHRASE_ENDING_MAX_CHARS, len(no_space)):]
        bucket["endings"][tail] += 1
    for particle in PARTICLE_RE.findall(no_space):
        bucket["particles"][particle] += 1
    if RHETORICAL_PATTERN_RE.search(no_space):
        bucket["rhetorical_patterns"][re.sub(r"\d+", RHETORICAL_DIGIT_PLACEHOLDER, no_space[:PHRASE_TEMPLATE_MAX_CHARS])] += 1
    if no_space.startswith(NEGATIVE_PREFIXES):
        bucket["negative_templates"][no_space[:PHRASE_TEMPLATE_MAX_CHARS]] += 1
    if no_space.startswith(CONFIRM_PREFIXES):
        bucket["confirmation_templates"][no_space[:PHRASE_TEMPLATE_MAX_CHARS]] += 1

def _top_counter(counter: Counter, limit: int = PHRASE_DEFAULT_TOP_LIMIT, *, min_count: int = 1) -> List[Dict[str, Any]]:
    items = []
    for text, count in counter.most_common(limit * PHRASE_TOP_SCAN_MULTIPLIER):
        if count < min_count:
            continue
        items.append({"text": text, "count": int(count)})
        if len(items) >= limit:
            break
    return items

def _finalize_phrase_bucket(bucket: Dict[str, Counter], *, min_short_count: int = PHRASE_MIN_SHORT_COUNT) -> Dict[str, Any]:
    return {
        "high_freq_short_replies": _top_counter(bucket["short_replies"], PHRASE_HIGH_FREQ_SHORT_LIMIT, min_count=min_short_count),
        "endings": _top_counter(bucket["endings"], PHRASE_ENDING_LIMIT, min_count=PHRASE_MIN_SHORT_COUNT),
        "particles": _top_counter(bucket["particles"], PHRASE_PARTICLE_LIMIT, min_count=PHRASE_MIN_SHORT_COUNT),
        "rhetorical_templates": _top_counter(bucket["rhetorical_patterns"], PHRASE_TEMPLATE_LIMIT, min_count=1),
        "negative_templates": _top_counter(bucket["negative_templates"], PHRASE_TEMPLATE_LIMIT, min_count=1),
        "confirmation_templates": _top_counter(bucket["confirmation_templates"], PHRASE_TEMPLATE_LIMIT, min_count=1),
    }

def _finalize_phrase_profile(phrase_stats: Dict[str, Any]) -> Dict[str, Any]:
    per_relationship = {}
    for key, bucket in phrase_stats["per_relationship"].items():
        finalized = _finalize_phrase_bucket(bucket, min_short_count=PHRASE_MIN_SHORT_COUNT)
        total = sum(len(finalized[name]) for name in finalized)
        if total:
            per_relationship[key] = finalized
    return {
        "schema_version": 1,
        "raw_text_policy": (
            "Contains local owner phrase snippets for style modeling. "
            "Do not commit or expose this file."
        ),
        "global": _finalize_phrase_bucket(phrase_stats["global"], min_short_count=PHRASE_MIN_SHORT_COUNT),
        "private": _finalize_phrase_bucket(phrase_stats["private"], min_short_count=PHRASE_MIN_SHORT_COUNT),
        "group": _finalize_phrase_bucket(phrase_stats["group"], min_short_count=PHRASE_MIN_SHORT_COUNT),
        "per_relationship": per_relationship,
    }

def _common_phrases_from_phrase_profile(phrase_profile: Dict[str, Any], limit: int = PHRASE_COMMON_DEFAULT_LIMIT) -> List[str]:
    common = []
    seen = set()
    for section_name in ("global", "private", "group"):
        section = phrase_profile.get(section_name) or {}
        for field in (
            "high_freq_short_replies",
            "confirmation_templates",
            "negative_templates",
            "rhetorical_templates",
        ):
            for item in section.get(field) or []:
                text = str(item.get("text") or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                common.append(text[:PHRASE_COMMON_MAX_CHARS])
                if len(common) >= limit:
                    return common
    return common


__all__ = [name for name in globals() if not name.startswith("__")]
