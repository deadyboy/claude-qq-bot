"""Report and run-artifact loading helpers for QCE style distillation."""

from .qce_io import *
from .phrases import *
from .turns import *
from .taxonomy import *


def format_qce_distillation_result(result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return f"离线蒸馏失败：{result.get('error') or 'unknown'}"
    applied = result.get("applied") or {}
    lines = [
        "Stage 5B 离线蒸馏完成：",
        f"- run_id：{result.get('run_id')}",
        f"- 输入文件：{result.get('input_files')} 个",
        f"- 已排除 AI/测试对话源：{result.get('excluded_sources', 0)} 个",
        f"- 总消息：{result.get('total_messages')} 条",
        f"- 本人文本：{result.get('owner_text_messages')} 条",
        f"- Turn：{result.get('turn_count')} 个",
        f"- Dialogue pairs：{result.get('dialogue_pair_count')} 条",
        f"- 候选样本：{result.get('candidate_samples')} 条",
        f"- 索引样本：{result.get('indexed_samples')} 条",
        f"- RAG pool：{result.get('rag_pool_count', 0)} 条",
        f"- SFT candidates：{result.get('sft_candidate_count', 0)} 条",
        f"- 高频短语：{'已生成' if result.get('common_phrases') else '为空'}",
        f"- 关系/场景画像：{result.get('relationship_profiles', 0)} 个",
        f"- 场景画像：{result.get('scene_profiles', 0)} 个",
        f"- 数据就绪度：{result.get('readiness')}",
        "- 原文策略：命令报告不展示聊天正文；summary/sample_index 只存元数据；turns/dialogue_pairs/phrase_profile/rag_pool/sft_candidates/rerank_rules 是本地 local raw 训练产物",
        f"- 输出目录：{result.get('output_dir')}",
    ]
    if applied:
        lines.append(f"- 已更新画像：{applied.get('profile_path')}")
    return "\n".join(lines)

def format_style_relationship_report(run_dir: str | Path | None = None) -> str:
    """Format relationship/scene source profiles without identifiers or text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    path = run_path / "relationship_profiles.json"
    if not path.exists():
        return f"这个蒸馏结果缺少 relationship_profiles.json：{run_path.name}"
    data = _load_json(path)
    profiles = data.get("profiles") or []
    label_counts = data.get("label_counts") or {}
    strong = sum(1 for item in profiles if "strong_context_reply_source" in (item.get("labels") or []))
    usable = sum(1 for item in profiles if "usable_context_reply_source" in (item.get("labels") or []))
    low = sum(1 for item in profiles if "low_evidence" in (item.get("labels") or []))
    private_count = sum(1 for item in profiles if item.get("chat_type") == "private")
    group_count = sum(1 for item in profiles if item.get("chat_type") == "group")
    lines = [
        "关系/场景画像摘要：",
        f"- run_id：{run_path.name}",
        f"- 来源：{len(profiles)} 个，私聊 {private_count}，群聊 {group_count}",
        f"- 上下文样本源：strong={strong}，usable={usable}，low={low}",
        "- 原文策略：不显示联系人、群名、QQ 或聊天正文",
    ]
    if label_counts:
        labels = "、".join(f"{key}:{value}" for key, value in list(label_counts.items())[:10])
        lines.append(f"- 标签分布：{labels}")
    top = profiles[:5]
    if top:
        lines.append("Top 来源摘要：")
        for index, item in enumerate(top, start=1):
            lines.append(
                f"{index}. {item.get('source_file_id')} {item.get('chat_type')} "
                f"owner={item.get('owner_text_messages')} samples={item.get('candidate_samples')} "
                f"avg={item.get('avg_length')} labels={','.join((item.get('labels') or [])[:3])}"
            )
    return "\n".join(lines)

def format_style_scene_report(run_dir: str | Path | None = None) -> str:
    """Format scene profiles without raw text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    path = run_path / "scene_profiles.json"
    if not path.exists():
        return f"这个蒸馏结果缺少 scene_profiles.json：{run_path.name}"
    data = _load_json(path)
    profiles = data.get("profiles") or []
    lines = [
        "场景画像摘要：",
        f"- run_id：{run_path.name}",
        f"- 场景数：{len(profiles)}",
        "- 原文策略：不显示历史上下文或真实回复正文",
    ]
    for index, item in enumerate(profiles[:8], start=1):
        lines.append(
            f"{index}. {item.get('scene_id')} count={item.get('sample_count')} "
            f"avg_len={item.get('avg_reply_length')} quick={item.get('quick_reply_ratio')} "
            f"style={item.get('recommended_style')}"
        )
    if not profiles:
        lines.append("尚未生成可用场景画像。")
    return "\n".join(lines)

def find_latest_distill_run(root: str | Path | None = None) -> Path | None:
    """Find the latest local Stage 5B distillation run directory."""
    if root:
        candidates_root = Path(root)
        if candidates_root.is_file():
            candidates_root = candidates_root.parent
        roots = [candidates_root]
    else:
        roots = list(DEFAULT_EXPORT_ROOT.glob("*/distill-runs"))
    runs: List[Path] = []
    for item in roots:
        if not item.exists():
            continue
        if item.name.startswith("stage5b_"):
            runs.append(item)
        else:
            runs.extend(path for path in item.glob("stage5b_*") if path.is_dir())
    if not runs:
        return None
    return max(runs, key=lambda path: path.stat().st_mtime)

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def format_style_evaluation_report(run_dir: str | Path | None = None) -> str:
    """Format the latest Stage 5B readiness report without raw text."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return "还没有找到 Stage 5B 离线蒸馏结果。先运行 /风格 离线蒸馏。"
    report_path = run_path / "evaluation_report.json"
    relation_path = run_path / "relationship_profiles.json"
    if not report_path.exists():
        return f"这个蒸馏结果缺少 evaluation_report.json：{run_path.name}"

    report = _load_json(report_path)
    relationships = _load_json(relation_path) if relation_path.exists() else {}
    relation_sources = report.get("relationship_sources") or {}
    label_counts = relationships.get("label_counts") or {}
    taxonomy = report.get("taxonomy") or {}
    taxonomy_use = taxonomy.get("use_for_counts") or {}
    lines = [
        "Stage 5B 评估摘要：",
        f"- run_id：{report.get('run_id')}",
        f"- 就绪度：{report.get('readiness')}",
        f"- 本人文本：{report.get('owner_text_messages', 0)} 条",
        f"- Turn：{report.get('turn_count', 0)} 个",
        f"- Dialogue pairs：{report.get('dialogue_pair_count', 0)} 条",
        f"- 候选样本：{report.get('candidate_samples', 0)} 条",
        f"- 索引样本：{report.get('indexed_samples', 0)} 条",
        f"- 高质量样本：{report.get('high_quality_samples', 0)} 条",
        f"- RAG pool：{taxonomy.get('rag_pool_count', 0)} 条",
        f"- SFT candidates：{taxonomy.get('sft_candidate_count', 0)} 条",
        f"- common_phrases 为空：{report.get('common_phrases_empty')}",
        (
            "- 关系来源："
            f"{relation_sources.get('total', 0)} 个，"
            f"strong={relation_sources.get('strong', 0)}，"
            f"usable={relation_sources.get('usable', 0)}，"
            f"low={relation_sources.get('low_evidence', 0)}"
        ),
    ]
    if label_counts:
        top_labels = "、".join(f"{key}:{value}" for key, value in list(label_counts.items())[:8])
        lines.append(f"- 场景标签：{top_labels}")
    scene_counts = report.get("scene_counts") or {}
    if scene_counts:
        scenes = "、".join(f"{key}:{value}" for key, value in list(scene_counts.items())[:8])
        lines.append(f"- 场景样本：{scenes}")
    ratios = report.get("chat_type_ratio") or {}
    if ratios:
        lines.append(f"- 私聊/群聊样本比例：private={ratios.get('private')}，group={ratios.get('group')}")
    if taxonomy_use:
        usage = "、".join(f"{key}:{value}" for key, value in list(taxonomy_use.items())[:6])
        lines.append(f"- 学习用途：{usage}")
    recommendations = report.get("recommendations") or []
    if recommendations:
        lines.append("- 建议：" + "；".join(str(item) for item in recommendations[:3]))
    lines.append("- 原文策略：评估报告不保存聊天正文")
    return "\n".join(lines)

def _resolve_run_paths(run_dir: str | Path | None = None) -> tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        raise FileNotFoundError("还没有找到 Stage 5B 离线蒸馏结果。")
    catalog_path = run_path / "source_catalog_private.json"
    index_path = run_path / "sample_index.jsonl"
    if not catalog_path.exists() or not index_path.exists():
        raise FileNotFoundError("蒸馏结果缺少 source_catalog_private.json 或 sample_index.jsonl。")
    catalog = _load_json(catalog_path)
    samples = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return run_path, catalog, samples

def _load_source_messages(catalog: Dict[str, Any], source_file_id: str, cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if source_file_id in cache:
        return cache[source_file_id]
    input_dir = Path(str(catalog.get("input_dir") or ""))
    for source in catalog.get("sources") or []:
        if source.get("source_file_id") != source_file_id:
            continue
        rel = Path(str(source.get("relative_path") or ""))
        path = (input_dir.parent / rel).resolve()
        data = _load_json(path)
        messages = data.get("messages") or []
        cache[source_file_id] = [item for item in messages if isinstance(item, dict)]
        return cache[source_file_id]
    return []

def _load_dialogue_pairs(run_path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    path = run_path / "rag_pool.jsonl"
    if not path.exists():
        path = run_path / "dialogue_pairs.jsonl"
    if not path.exists():
        return []
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                pairs.append(item)
                if limit is not None and len(pairs) >= limit:
                    break
    return pairs

def _turns_text(turns: Sequence[Dict[str, Any]], *, roles: set[str] | None = None) -> str:
    texts = []
    for turn in turns:
        if roles and str(turn.get("role") or "") not in roles:
            continue
        text = str(turn.get("text") or "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts)

def _normalize_current_context(current_context: str | Sequence[Dict[str, Any]] | None, latest_message: str) -> str:
    if isinstance(current_context, str):
        context_text = current_context.strip()
    elif current_context:
        parts = []
        for item in current_context:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "other")
            text = str(item.get("text") or item.get("content") or "").strip()
            if text:
                parts.append(f"{role}:{text}")
        context_text = "\n".join(parts)
    else:
        context_text = ""
    latest = str(latest_message or "").strip()
    return "\n".join(part for part in (context_text, latest) if part).strip()


__all__ = [name for name in globals() if not name.startswith("__")]
