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

def classify_reply_behavior(text: str, *, latest_message: str = "") -> Dict[str, Any]:
    """Classify what a candidate is doing, separate from surface wording."""
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'“”")
    compact = re.sub(r"\s+", "", cleaned.strip("。.!！?？~～"))
    result: Dict[str, Any] = {
        "label": "neutral",
        "safe_for_context": True,
        "reasons": [],
    }
    if not cleaned:
        return {"label": "empty", "safe_for_context": False, "reasons": ["empty"]}

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

def style_rerank_candidates(
    candidates: Sequence[str],
    *,
    scene_label: str = "",
    max_length: int | None = None,
    corrections: Sequence[Dict[str, Any]] | None = None,
    historical_targets: Sequence[str] | None = None,
    latest_message: str = "",
) -> List[Dict[str, Any]]:
    """Filter candidates that are too formal, too long, or assistant-like."""
    formal_markers = (
        "您好", "请问", "非常抱歉", "感谢", "作为", "以下是", "总结一下",
        "整理如下", "建议你", "需要注意的是", "你好！我是",
    )
    ai_markers = (
        "机器人", "AI", "ai", "助手", "无法", "我不能", "根据上下文",
        "我可以帮你", "有什么可以帮", "好问题", "让我帮你", "让我来",
        "随时准备协助", "科研助手", "数字伙伴", "模型", "prompt",
    )
    target_max = max_length or (72 if scene_label in {"private_long_explain", "formal_or_worklike"} else 32)
    correction_items = list(corrections or [])
    latest_intent = detect_message_intent(latest_message) if latest_message else {}
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
        score = 100
        reasons = []
        compact_latest = re.sub(r"\s+", "", str(latest_message or "").strip("。.!！?？~～"))
        compact_candidate = re.sub(r"\s+", "", text.strip("。.!！?？~～"))
        if compact_latest and compact_candidate == compact_latest:
            score -= 70
            reasons.append("copied_current_message")
        if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
            score -= 90
            reasons.append("invalid_text")
        if len(text) > target_max:
            score -= min(60, len(text) - target_max)
            reasons.append("too_long")
        if any(marker in text for marker in formal_markers):
            score -= 35
            reasons.append("too_formal")
        if any(marker in text for marker in ai_markers):
            score -= 45
            reasons.append("assistant_like")
        if re.search(r"(^|\n)\s*(?:#{1,4}\s|[-*]\s+|\d+[.)、]\s+)", text):
            score -= 25
            reasons.append("structured_answer")
        if text.endswith(("。", "！")) and len(text) <= 12:
            score -= 6
            reasons.append("over_punctuated_short")
        if latest_intent.get("reality_state_query") and any(marker in text for marker in state_claim_markers):
            score -= 42
            reasons.append("unsafe_owner_state")
        if (
            latest_intent.get("task_request")
            or latest_intent.get("reality_state_query")
            or _contains_any(latest_message, TASK_HINTS)
            or _contains_any(latest_message, ("弄完", "做完", "今天能", "今晚能"))
        ) and any(marker in text for marker in over_commit_markers):
            score -= 34
            reasons.append("over_commit")
        if credential_request and any(marker in text for marker in risky_share_markers):
            score -= 90
            reasons.append("credential_share_risk")
        if (
            latest_intent.get("help_request")
            or latest_intent.get("task_request")
            or _contains_any(latest_message, HELP_HINTS + TASK_HINTS)
        ) and text in generic_probe_replies:
            score -= 16
            reasons.append("too_generic_for_request")
        if latest_intent.get("emotional") and text in {"何意", "何意味"}:
            score -= 12
            reasons.append("too_generic_for_emotion")
        behavior = classify_reply_behavior(text, latest_message=latest_message)
        if latest_intent.get("game_invitation"):
            if behavior["label"] == "assistant_invite":
                score -= 44
                reasons.append("assistant_like_game_invite")
            if behavior["label"] == "accept_commit":
                score -= 70
                reasons.append("unknown_game_availability_commit")
            if behavior["label"] in {"decline", "defer", "ask_third_party", "clarify"}:
                score += 16
                reasons.append("safe_game_deflection")
            if behavior["label"] == "weak_probe":
                score -= 18
                reasons.append("weak_game_probe")
            if text.endswith(("！", "!")):
                score -= 12
                reasons.append("over_excited_game_invite")
            if len(text) > 14:
                score -= 18
                reasons.append("too_long_for_game_invite")
        if len(text) >= 3 and text in history_texts:
            score -= 24
            reasons.append("copied_history_exact")
        elif len(text) >= 6 and history_texts:
            similarity = max(
                (_jaccard(_text_ngrams(text), _text_ngrams(target)) for target in history_texts),
                default=0.0,
            )
            if similarity >= 0.82:
                score -= 14
                reasons.append("copied_history_near")
        correction_delta, correction_reasons = candidate_correction_delta(text, correction_items)
        if correction_delta:
            score += correction_delta
            reasons.extend(correction_reasons)
        if 2 <= len(text) <= target_max:
            score += 8
            reasons.append("length_ok")
        hard_reject = (
            "assistant_like" in reasons
            or "too_long" in reasons
            or "structured_answer" in reasons
            or "invalid_text" in reasons
            or "unsafe_owner_state" in reasons
            or "over_commit" in reasons
            or "credential_share_risk" in reasons
            or "copied_history_exact" in reasons
            or "copied_history_near" in reasons
            or "copied_current_message" in reasons
            or "assistant_like_game_invite" in reasons
            or "unknown_game_availability_commit" in reasons
            or ("too_formal" in reasons and scene_label != "formal_or_worklike")
        )
        ranked.append({
            "text": text,
            "score": score,
            "reasons": reasons,
            "accepted": score >= 55 and not hard_reject,
            "behavior": behavior.get("label"),
            "behavior_safe": bool(behavior.get("safe_for_context")),
            "behavior_reasons": behavior.get("reasons") or [],
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
