"""Phrase profile construction for QCE style distillation."""

from .qce_io import *


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
    if 2 <= len(no_space) <= 14 and not URL_RE.search(no_space):
        bucket["short_replies"][no_space] += 1
    if len(no_space) >= 2:
        tail = no_space[-min(4, len(no_space)):]
        bucket["endings"][tail] += 1
    for particle in PARTICLE_RE.findall(no_space):
        bucket["particles"][particle] += 1
    if re.search(r"不.+吗|不是.+吗|怎么.+了|咋.+了|啥.+啊|哪.+了", no_space):
        bucket["rhetorical_patterns"][re.sub(r"\d+", "N", no_space[:24])] += 1
    if no_space.startswith(NEGATIVE_PREFIXES):
        bucket["negative_templates"][no_space[:24]] += 1
    if no_space.startswith(CONFIRM_PREFIXES):
        bucket["confirmation_templates"][no_space[:24]] += 1

def _top_counter(counter: Counter, limit: int = 30, *, min_count: int = 1) -> List[Dict[str, Any]]:
    items = []
    for text, count in counter.most_common(limit * 2):
        if count < min_count:
            continue
        items.append({"text": text, "count": int(count)})
        if len(items) >= limit:
            break
    return items

def _finalize_phrase_bucket(bucket: Dict[str, Counter], *, min_short_count: int = 2) -> Dict[str, Any]:
    return {
        "high_freq_short_replies": _top_counter(bucket["short_replies"], 50, min_count=min_short_count),
        "endings": _top_counter(bucket["endings"], 40, min_count=2),
        "particles": _top_counter(bucket["particles"], 40, min_count=2),
        "rhetorical_templates": _top_counter(bucket["rhetorical_patterns"], 30, min_count=1),
        "negative_templates": _top_counter(bucket["negative_templates"], 30, min_count=1),
        "confirmation_templates": _top_counter(bucket["confirmation_templates"], 30, min_count=1),
    }

def _finalize_phrase_profile(phrase_stats: Dict[str, Any]) -> Dict[str, Any]:
    per_relationship = {}
    for key, bucket in phrase_stats["per_relationship"].items():
        finalized = _finalize_phrase_bucket(bucket, min_short_count=2)
        total = sum(len(finalized[name]) for name in finalized)
        if total:
            per_relationship[key] = finalized
    return {
        "schema_version": 1,
        "raw_text_policy": (
            "Contains local owner phrase snippets for style modeling. "
            "Do not commit or expose this file."
        ),
        "global": _finalize_phrase_bucket(phrase_stats["global"], min_short_count=2),
        "private": _finalize_phrase_bucket(phrase_stats["private"], min_short_count=2),
        "group": _finalize_phrase_bucket(phrase_stats["group"], min_short_count=2),
        "per_relationship": per_relationship,
    }

def _common_phrases_from_phrase_profile(phrase_profile: Dict[str, Any], limit: int = 12) -> List[str]:
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
                common.append(text[:40])
                if len(common) >= limit:
                    return common
    return common


__all__ = [name for name in globals() if not name.startswith("__")]
