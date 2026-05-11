"""Retrieval-first owner-style generation helpers."""

from .qce_io import *
from .phrases import *
from .turns import *
from .taxonomy import *
from .reports import *
from .retrieval import *


def build_retrieval_first_prompt(
    latest_message: str,
    *,
    current_context: str | Sequence[Dict[str, Any]] | None = None,
    retrieval: Dict[str, Any] | None = None,
    style_skill_context: Dict[str, Any] | None = None,
    candidate_count: int = 8,
    include_raw_samples: bool = False,
) -> str:
    retrieval = retrieval or {}
    count = max(3, min(8, int(candidate_count or 8)))
    lines = [
        f"你在生成主人聊天草稿。只输出 {count} 条候选回复，JSON 数组字符串，每条只含回复正文。",
        "要求：像真实聊天，不要像 AI 助手；不要解释；不要自称机器人；不要编造主人现实状态或承诺；不要照抄历史原句。",
        "相似真实样本只用于学习语气和节奏，不要把样本里的具体事实、人物、物品、场景搬到当前回复里。",
        f"当前场景：{retrieval.get('scene_label') or 'unknown'}",
    ]
    skill_prompt = format_style_skill_context_for_prompt(style_skill_context)
    if skill_prompt:
        lines.append(skill_prompt)
    context_text = _normalize_current_context(current_context, latest_message)
    if context_text:
        lines.append("当前聊天上下文：")
        lines.append(context_text[:1200])
    samples = retrieval.get("results") or []
    if samples:
        if include_raw_samples:
            lines.append("相似真实样本（已授权原句 few-shot）：")
        else:
            lines.append("相似样本元数据（未授权原句 few-shot，不含历史原文）：")
        for index, item in enumerate(samples[:8], start=1):
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
                    lines.append("历史上下文：" + sample_context[:500])
                if target_text:
                    lines.append("主人真实回复：" + str(target_text)[:300])
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
        if 2 <= len(text) <= 16:
            phrases.add(text)
    for item in style_profile.get("habits") or []:
        for part in re.split(r"[、,，；;:\s]+", str(item or "")):
            text = _compact_reply_text(part)
            if 2 <= len(text) <= 16 and not text.startswith(("平均回复", "中位数")):
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
        "short": len(compact) <= 8,
    }


def _structural_similarity(left: str, right: str) -> float:
    left_sig = _punctuation_signature(left)
    right_sig = _punctuation_signature(right)
    keys = ("no_punct", "question", "exclaim", "ellipsis", "wave", "short")
    matches = sum(1 for key in keys if left_sig.get(key) == right_sig.get(key))
    score = matches / max(1, len(keys))
    if left_sig["line_count"] == right_sig["line_count"]:
        score += 0.08
    return min(1.0, score)


def _length_fit_score(text: str, target_length: float | None, target_max: int) -> tuple[int, str]:
    compact_len = len(_compact_reply_text(text))
    if compact_len <= 0:
        return -30, "empty_length"
    if target_length and target_length > 0:
        diff = abs(compact_len - float(target_length))
        tolerance = max(3.0, min(12.0, float(target_length) * 0.55))
        if diff <= tolerance:
            return 18, "length_close_to_profile"
        return -min(24, int(round((diff - tolerance) * 1.8))), "length_far_from_profile"
    if 2 <= compact_len <= target_max:
        return 12, "length_ok"
    return -min(24, abs(compact_len - target_max)), "length_off"


def classify_reply_behavior(text: str, *, latest_message: str = "") -> Dict[str, Any]:
    """Classify what a candidate is doing, separate from surface wording."""
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
    risk_level, risk_label = _commitment_risk(latest_intent)
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”")
    compact = _compact_reply_text(cleaned)
    result: Dict[str, Any] = {
        "label": "neutral",
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

    if latest_intent.get("high_risk_request"):
        risky_grant = (
            "发你", "给你", "发给你", "可以登", "给你登", "你登吧", "验证码",
            "我转", "转你", "打给你", "行我给", "可以给", "直接给",
        )
        if any(marker in cleaned for marker in risky_grant):
            result.update({"label": "high_risk_grant", "safe_for_context": False})
            result["reasons"].append("credential_or_finance_commitment")
            return result

    if latest_intent.get("game_invitation"):
        assistant_invite = (
            "有的呀", "有的啊", "当然", "可以呀", "可以啊", "想一起", "一起开黑",
            "来开黑", "开黑吗", "一起玩吗", "玩吗？", "玩吗?",
        )
        accept_commit = (
            "有", "有啊", "有的", "打", "打瓦", "可瓦", "可以", "能打",
            "来", "我来", "上号", "能瓦", "在打", "正在打", "打着", "上了",
        )
        if any(marker in cleaned for marker in assistant_invite):
            result.update({"label": "assistant_invite", "safe_for_context": False})
            result["reasons"].append("assistant_style_invite")
        elif compact in accept_commit or (
            compact.startswith(("有", "打", "可瓦", "上号", "在打", "正在"))
            and not compact.startswith(("暂无", "无", "不", "等", "问", "何"))
            and "问问" not in compact
        ):
            result.update({"label": "accept_commit", "safe_for_context": False})
            result["reasons"].append("unknown_owner_game_availability")
        elif compact.startswith(("无", "不打", "不玩", "没有", "暂无", "暂不")):
            result["label"] = "decline"
            result["reasons"].append("safe_game_decline")
        elif compact.startswith(("等", "等等")):
            result["label"] = "defer"
            result["reasons"].append("safe_game_defer")
        elif "问问" in compact or compact.startswith(("叫", "喊")):
            result["label"] = "ask_third_party"
            result["reasons"].append("safe_game_third_party")
        elif compact in {"何意", "何意味", "什么意思"}:
            result["label"] = "clarify"
            result["reasons"].append("safe_game_clarify")
        elif (
            any(marker in compact for marker in ("段位", "几排", "几缺", "啥段", "什么段", "哪个服", "哪个区", "谁在"))
            or compact.startswith(("啥", "咋说", "怎么说", "谁", "几个"))
        ):
            result["label"] = "engage_probe"
            result["reasons"].append("game_engagement_probe")
        elif compact in {"咋了", "咋说", "啥事", "干嘛"}:
            result["label"] = "weak_probe"
            result["reasons"].append("weak_game_probe")
        return result

    if latest_intent.get("availability_query") or latest_intent.get("reality_state_query"):
        owner_state_claims = (
            "不忙", "有空", "在家", "在宿舍", "在学校", "刚醒", "刚起",
            "已经到了", "快到了", "做完了", "弄完了", "搞完了", "在呢",
            "在的", "我在", "人在",
        )
        if any(marker in cleaned for marker in owner_state_claims):
            result.update({"label": "owner_state_commit", "safe_for_context": False})
            result["reasons"].append("unknown_owner_reality_state")
    return result


def historical_behavior_distribution(
    historical_targets: Sequence[str] | None,
    *,
    latest_message: str = "",
) -> Dict[str, Any]:
    """Infer behavior preferences from retrieved owner replies for this query."""
    counter: Counter[str] = Counter()
    examples: Dict[str, List[str]] = {}
    for target in historical_targets or []:
        text = re.sub(r"\s+", " ", str(target or "")).strip()
        if not text:
            continue
        behavior = classify_reply_behavior(text, latest_message=latest_message)
        label = str(behavior.get("label") or "neutral")
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
    dominant = str(distribution.get("dominant") or "")
    if prop >= 0.45:
        return 24, f"history_behavior_match:{label}:{prop:.2f}"
    if prop >= 0.22:
        return 15, f"history_behavior_match:{label}:{prop:.2f}"
    if prop >= 0.10:
        return 7, f"history_behavior_minor:{label}:{prop:.2f}"
    if latest_intent.get("game_invitation") and label in {"defer", "decline", "ask_third_party", "clarify"}:
        if sample_count >= 3 and dominant and dominant != label:
            return -7, f"history_behavior_not_dominant:{label}<>{dominant}"
    return 0, f"history_behavior_unseen:{label}"

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
    """Rerank candidates by persona fit, scene fit, and commitment risk.

    Stage 5C keeps safety as a dimension instead of a blanket override. Hard
    rejection is reserved for malformed text, credential/finance leakage, and
    exact raw-history/current-message copying.
    """
    formal_markers = (
        "您好", "请问", "非常抱歉", "感谢", "作为", "以下是", "总结一下",
        "整理如下", "建议你", "需要注意的是", "你好！我是",
    )
    ai_markers = (
        "机器人", "AI", "ai", "助手", "无法", "我不能", "根据上下文",
        "我可以帮你", "有什么可以帮", "好问题", "让我帮你", "让我来",
        "随时准备协助", "科研助手", "数字伙伴", "模型", "prompt",
    )
    ai_flavor_prefixes = (
        "好的，", "好的,", "理解了", "明白了", "收到，我", "没问题，我",
        "当然可以", "可以的，我", "我来帮你", "作为一个",
    )
    target_max = max_length or (72 if scene_label in {"private_long_explain", "formal_or_worklike"} else 32)
    correction_items = list(corrections or [])
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
    commitment_level, commitment_label = _commitment_risk(latest_intent)
    state_claim_markers = (
        "不忙", "有空", "在家", "在宿舍", "在学校", "刚醒", "刚起",
        "刚坐下", "已经到了", "快到了", "做完了", "弄完了", "搞完了",
        "在呢", "在的", "我在", "人在",
    )
    over_commit_markers = (
        "马上", "这就", "立刻", "一定", "肯定", "包的", "包能",
        "今天能", "今晚能", "可以弄完", "能弄完", "没问题我来",
        "快了", "差不多了", "快弄完", "快做完", "马上好",
    )
    credential_request = _contains_any(latest_message, ("账号", "密码", "验证码", "登一下", "登录", "借号"))
    risky_share_markers = (
        "发你", "给你", "发给你", "给你登", "行，发", "可以，发",
        "喏", "直接给", "你登吧",
    )
    generic_probe_replies = {"何意", "何意味", "咋了", "咋滴", "嗯？", "啥"}
    history_texts = [
        re.sub(r"\s+", " ", str(item or "")).strip()
        for item in (historical_targets or [])
        if str(item or "").strip()
    ][:12]
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
        style_score = 50
        scene_score = 50
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
            hygiene_penalty += 90
            reasons.append("copied_current_message")
            hard_reject_reasons.append("copied_current_message")
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
            hygiene_penalty += 90
            reasons.append("invalid_text")
            hard_reject_reasons.append("invalid_text")
        length_delta, length_reason = _length_fit_score(text, target_length, target_max)
        style_score += length_delta
        if length_delta >= 0:
            persona_reasons.append(length_reason)
        else:
            reasons.append(length_reason)
        if len(text) > target_max:
            excess = min(45, len(text) - target_max)
            hygiene_penalty += excess
            reasons.append("too_long")
        if any(marker in text for marker in formal_markers):
            style_score -= 24
            reasons.append("too_formal")
        if any(marker in text for marker in ai_markers):
            style_score -= 34
            reasons.append("assistant_like")
        if text.startswith(ai_flavor_prefixes) or re.search(r"^(好的|收到|明白)[，,].{2,}", text):
            style_score -= 34
            reasons.append("ai_flavor_prefix")
        if re.search(r"(^|\n)\s*(?:#{1,4}\s|[-*]\s+|\d+[.)、]\s+)", text):
            hygiene_penalty += 42
            reasons.append("structured_answer")
            hard_reject_reasons.append("structured_answer")
        if text.endswith(("。", "！")) and len(text) <= 12:
            style_score -= 4
            reasons.append("over_punctuated_short")
        phrase_hits = [
            phrase for phrase in profile_phrases
            if phrase and (phrase in compact_candidate or compact_candidate in phrase)
        ][:3]
        if phrase_hits:
            style_score += min(16, 6 + len(phrase_hits) * 3)
            persona_reasons.append("phrase_overlap:" + "/".join(phrase_hits))
        elif len(compact_candidate) <= 4:
            style_score += 5
            persona_reasons.append("short_casual_shape")
        if history_texts:
            structure_match = max((_structural_similarity(text, target) for target in history_texts), default=0.0)
            if structure_match >= 0.72:
                style_score += 10
                persona_reasons.append("structure_like_history")
            elif structure_match >= 0.55:
                style_score += 5
                persona_reasons.append("structure_somewhat_like_history")
        if _punctuation_signature(text).get("no_punct") and len(compact_candidate) <= 12:
            style_score += 5
            persona_reasons.append("chat_like_no_punct")
        if latest_intent.get("reality_state_query") and any(marker in text for marker in state_claim_markers):
            risk_penalty += 50
            reasons.append("unsafe_owner_state")
            risk_reasons.append("owner_state_claim_without_grounding")
        if (
            latest_intent.get("task_request")
            or latest_intent.get("reality_state_query")
            or _contains_any(latest_message, TASK_HINTS)
            or _contains_any(latest_message, ("弄完", "做完", "今天能", "今晚能"))
        ) and any(marker in text for marker in over_commit_markers):
            risk_penalty += 36 + commitment_level * 8
            reasons.append("over_commit")
            risk_reasons.append("action_or_progress_commitment")
        if credential_request and any(marker in text for marker in risky_share_markers):
            risk_penalty += 90
            reasons.append("credential_share_risk")
            risk_reasons.append("credential_share_risk")
            hard_reject_reasons.append("credential_share_risk")
        if (
            latest_intent.get("help_request")
            or latest_intent.get("task_request")
            or _contains_any(latest_message, HELP_HINTS + TASK_HINTS)
        ) and text in generic_probe_replies:
            scene_score -= 18
            reasons.append("too_generic_for_request")
        if latest_intent.get("emotional") and text in {"何意", "何意味"}:
            scene_score -= 12
            reasons.append("too_generic_for_emotion")
        behavior = classify_reply_behavior(text, latest_message=latest_message)
        if behavior.get("label") == "high_risk_grant":
            risk_penalty += 90
            reasons.append("high_risk_grant")
            risk_reasons.extend(behavior.get("reasons") or [])
            hard_reject_reasons.append("high_risk_grant")
        behavior_delta, behavior_reason = _historical_behavior_delta(
            str(behavior.get("label") or "neutral"),
            behavior_distribution,
            latest_intent=latest_intent,
        )
        if behavior_delta:
            scene_score += behavior_delta
            scene_reasons.append(behavior_reason)
        if latest_intent.get("game_invitation"):
            if behavior["label"] == "assistant_invite":
                style_score -= 22
                scene_score -= 14
                risk_penalty += 24
                reasons.append("assistant_like_game_invite")
            if behavior["label"] == "accept_commit":
                scene_score -= 8
                risk_penalty += 34
                reasons.append("unknown_game_availability_commit")
                risk_reasons.append("game_availability_commitment")
            if behavior["label"] in {"decline", "defer", "ask_third_party", "clarify", "engage_probe"}:
                if behavior_distribution.get("sample_count"):
                    scene_reasons.append("game_behavior_scored_by_history")
                else:
                    scene_score += 6
                    scene_reasons.append("low_risk_game_behavior_baseline")
                style_score += 4
                reasons.append("low_risk_game_behavior")
            if behavior["label"] == "weak_probe":
                if not behavior_distribution.get("sample_count"):
                    scene_score -= 12
                reasons.append("weak_game_probe")
            if text.endswith(("！", "!")):
                style_score -= 10
                reasons.append("over_excited_game_invite")
            if len(text) > 14:
                style_score -= 14
                reasons.append("too_long_for_game_invite")
        elif commitment_level == 0:
            scene_score += 4
            scene_reasons.append("low_risk_social")
        elif commitment_level == 2 and behavior.get("label") == "owner_state_commit":
            risk_penalty += 18
            risk_reasons.append("state_declaration_level2")
        if len(text) >= 10 and text in history_texts:
            hygiene_penalty += 78
            reasons.append("copied_history_exact")
            hard_reject_reasons.append("copied_history_exact")
        elif 3 <= len(text) < 10 and text in history_texts:
            hygiene_penalty += 8
            style_score += 4
            reasons.append("reused_short_history_phrase")
            persona_reasons.append("historical_short_phrase")
        elif len(text) >= 10 and history_texts:
            similarity = max(
                (_jaccard(_text_ngrams(text), _text_ngrams(target)) for target in history_texts),
                default=0.0,
            )
            if similarity >= 0.82:
                hygiene_penalty += 35
                reasons.append("copied_history_near")
                hard_reject_reasons.append("copied_history_near")
        correction_delta, correction_reasons = candidate_correction_delta(text, correction_items)
        if correction_delta:
            style_score += correction_delta
            reasons.extend(correction_reasons)
        score = int(round(style_score * 0.52 + scene_score * 0.33 + 20 - risk_penalty - hygiene_penalty))
        hard_reject = bool(hard_reject_reasons)
        accepted = score >= 58 and not hard_reject
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
        limit=8,
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
        candidate_count=8,
        include_raw_samples=include_raw_samples,
    )
    from .api import llm_client
    raw = await llm_client.chat(
        messages=[{"role": "user", "content": prompt}],
        system_prompt="你只输出 JSON 数组字符串，数组里 8 个中文聊天候选回复。",
        temperature=0.8,
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
        candidates[:12],
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
            "target_id": str(target_id or "")[:24],
        },
        "candidates": reranked,
        "prompt_preview": prompt[:2000],
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
            and embedding_similarity < 0.43
        ):
            continue
        if (
            _safe_float(result.get("context_overlap")) < MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD
            and _safe_float(result.get("keyword_overlap")) < MIN_RAW_FEWSHOT_TEXT_OR_KEYWORD
            and embedding_similarity < 0.43
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
            for turn in (pair.get("context") or [])[-6:]:
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
            if len(context_lines) > 3:
                context_lines = context_lines[-3:]
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
        if len(context_lines) > 3:
            context_lines = context_lines[-3:]

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
