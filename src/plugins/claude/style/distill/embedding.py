"""Local embedding index for Stage 5B RAG pool.

This module builds a local Chroma index from ``rag_pool.jsonl``. The Chroma
collection intentionally stores only embeddings and scalar metadata, not raw
chat text. Raw pair text remains in the existing local distillation artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .reports import _load_dialogue_pairs, find_latest_distill_run


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_ROOT = Path(os.getenv("QQBOT_EMBEDDING_MODEL_ROOT") or r"F:\ClaudeSpace2\models")
DEFAULT_SENTENCE_TRANSFORMERS_CACHE = DEFAULT_MODEL_ROOT / "sentence-transformers"
DEFAULT_HF_HOME = DEFAULT_MODEL_ROOT / "hf-home"
DEFAULT_TORCH_HOME = DEFAULT_MODEL_ROOT / "torch"
DEFAULT_COLLECTION_NAME = "stage5b_rag_pool_bge_small_zh_v1_5"
DEFAULT_BATCH_SIZE = 64
DEFAULT_QUERY_LIMIT = 20
__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "configure_embedding_environment",
    "build_embedding_text",
    "build_embedding_metadata",
    "build_stage5b_embedding_index",
    "embedding_index_status",
    "query_stage5b_embedding_index",
    "format_embedding_query_result",
]


def configure_embedding_environment() -> None:
    """Pin model/cache locations to F: before loading embedding libraries."""
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_HOME / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_SENTENCE_TRANSFORMERS_CACHE))
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TEMP", r"F:\ClaudeSpace2\tmp")
    os.environ.setdefault("TMP", r"F:\ClaudeSpace2\tmp")


def _embedding_index_dir(run_path: Path) -> Path:
    return run_path / "embedding_chroma"


def _embedding_manifest_path(run_path: Path) -> Path:
    return run_path / "embedding_manifest.json"


def _clean_index_text(text: Any, limit: int = 1600) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:limit]


def _turns_text(turns: Sequence[Dict[str, Any]], *, roles: set[str] | None = None) -> str:
    texts: List[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if roles and str(turn.get("role") or "") not in roles:
            continue
        text = _clean_index_text(turn.get("text"), 500)
        if text:
            texts.append(text)
    return "\n".join(texts)


def build_embedding_text(pair: Dict[str, Any]) -> str:
    """Build query-side text for embedding from context, not owner reply text."""
    taxonomy = pair.get("taxonomy") if isinstance(pair.get("taxonomy"), dict) else {}
    context_turns = pair.get("context") or []
    other_context = _turns_text(context_turns, roles={"other"})
    if not other_context:
        other_context = _turns_text(context_turns)

    parts = [
        f"场景：{pair.get('scene_label') or 'unknown'}",
        f"聊天类型：{pair.get('chat_type') or 'unknown'}",
        f"学习范围：{taxonomy.get('scope') or 'unknown'}",
        f"依据类型：{taxonomy.get('grounding_type') or 'unknown'}",
        "对方上下文：",
        other_context,
    ]
    return "\n".join(part for part in parts if str(part).strip()).strip()


def _safe_metadata(pair: Dict[str, Any], *, embedding_chars: int) -> Dict[str, Any]:
    taxonomy = pair.get("taxonomy") if isinstance(pair.get("taxonomy"), dict) else {}
    target = pair.get("target") if isinstance(pair.get("target"), dict) else {}
    return {
        "pair_id": str(pair.get("pair_id") or ""),
        "source_file_id": str(pair.get("source_file_id") or ""),
        "relationship_id": str(pair.get("relationship_id") or ""),
        "chat_type": str(pair.get("chat_type") or ""),
        "scene_label": str(pair.get("scene_label") or ""),
        "score": int(pair.get("score") or 0),
        "length_bucket": str(pair.get("length_bucket") or ""),
        "learning_value": str(taxonomy.get("learning_value") or ""),
        "grounding_type": str(taxonomy.get("grounding_type") or ""),
        "scope": str(taxonomy.get("scope") or ""),
        "target_char_length": int(
            taxonomy.get("target_char_length")
            or target.get("char_length")
            or len(str(target.get("text") or ""))
        ),
        "context_turn_count": int(taxonomy.get("context_turn_count") or len(pair.get("context") or [])),
        "embedding_text_chars": int(embedding_chars),
    }


def build_embedding_metadata(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Build scalar-only Chroma metadata without raw chat text."""
    return _safe_metadata(pair, embedding_chars=len(build_embedding_text(pair)))


def _iter_index_records(pairs: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        pair_id = str(pair.get("pair_id") or "").strip()
        if not pair_id:
            continue
        text = build_embedding_text(pair)
        if not text:
            continue
        yield {
            "id": pair_id,
            "text": text,
            "metadata": _safe_metadata(pair, embedding_chars=len(text)),
        }


def _local_sentence_transformer_path(model_name: str) -> Path | None:
    if model_name != DEFAULT_EMBEDDING_MODEL:
        return None
    repo_dir = DEFAULT_SENTENCE_TRANSFORMERS_CACHE / "models--BAAI--bge-small-zh-v1.5"
    ref_path = repo_dir / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = repo_dir / "snapshots" / revision
    if (snapshot / "modules.json").exists():
        return snapshot
    return None


def _load_sentence_transformer(model_name: str = DEFAULT_EMBEDDING_MODEL):
    configure_embedding_environment()
    from sentence_transformers import SentenceTransformer

    local_path = _local_sentence_transformer_path(model_name)
    load_target = str(local_path) if local_path else model_name
    return SentenceTransformer(load_target, cache_folder=str(DEFAULT_SENTENCE_TRANSFORMERS_CACHE))


def _chroma_client(index_dir: Path):
    import chromadb

    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir))


def _batched(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def build_stage5b_embedding_index(
    run_dir: str | Path | None = None,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reset: bool = True,
) -> Dict[str, Any]:
    """Build or rebuild a local embedding index for the latest Stage 5B run."""
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        raise FileNotFoundError("还没有找到 Stage 5B 离线蒸馏结果。")
    pairs = _load_dialogue_pairs(run_path)
    records = list(_iter_index_records(pairs))
    if not records:
        raise FileNotFoundError(f"没有可索引的 RAG 样本：{run_path}")

    model = _load_sentence_transformer(model_name)
    index_dir = _embedding_index_dir(run_path)
    client = _chroma_client(index_dir)
    if reset:
        try:
            client.delete_collection(DEFAULT_COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=DEFAULT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": model_name},
    )

    started = time.time()
    indexed = 0
    for batch in _batched(records, batch_size):
        texts = [item["text"] for item in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=[item["id"] for item in batch],
            embeddings=embeddings,
            metadatas=[item["metadata"] for item in batch],
        )
        indexed += len(batch)

    manifest = {
        "schema_version": 1,
        "run_id": run_path.name,
        "model": model_name,
        "collection": DEFAULT_COLLECTION_NAME,
        "index_dir": str(index_dir),
        "sample_count": indexed,
        "embedding_dimension": len(model.encode(["维度检查"], normalize_embeddings=True)[0]),
        "source": "rag_pool.jsonl",
        "raw_text_policy": "Chroma stores embeddings and scalar metadata only; raw text remains in local distillation artifacts.",
        "model_cache": str(DEFAULT_SENTENCE_TRANSFORMERS_CACHE),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    _embedding_manifest_path(run_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def embedding_index_status(run_dir: str | Path | None = None) -> Dict[str, Any]:
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return {"ok": False, "message": "还没有找到 Stage 5B 离线蒸馏结果。"}
    manifest_path = _embedding_manifest_path(run_path)
    if not manifest_path.exists():
        return {
            "ok": False,
            "run_id": run_path.name,
            "message": "当前 Stage 5B run 还没有 embedding_manifest.json。",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "run_id": run_path.name, "message": f"manifest 读取失败：{type(e).__name__}"}
    return {"ok": True, **manifest}


def _pair_lookup(run_path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(pair.get("pair_id") or ""): pair
        for pair in _load_dialogue_pairs(run_path)
        if isinstance(pair, dict) and str(pair.get("pair_id") or "").strip()
    }


def query_stage5b_embedding_index(
    query: str,
    *,
    run_dir: str | Path | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    limit: int = DEFAULT_QUERY_LIMIT,
    include_text: bool = False,
) -> Dict[str, Any]:
    """Query the local embedding index. Raw text is included only on request."""
    query_text = str(query or "").strip()
    if not query_text:
        return {"ok": False, "message": "query 不能为空。"}
    run_path = find_latest_distill_run(run_dir)
    if run_path is None:
        return {"ok": False, "message": "还没有找到 Stage 5B 离线蒸馏结果。"}
    manifest = embedding_index_status(run_path)
    if not manifest.get("ok"):
        return manifest

    model = _load_sentence_transformer(model_name)
    client = _chroma_client(_embedding_index_dir(run_path))
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)
    query_embedding = model.encode([query_text], normalize_embeddings=True).tolist()[0]
    raw_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(100, int(limit or DEFAULT_QUERY_LIMIT))),
        include=["metadatas", "distances"],
    )
    ids = (raw_result.get("ids") or [[]])[0]
    metadatas = (raw_result.get("metadatas") or [[]])[0]
    distances = (raw_result.get("distances") or [[]])[0]
    lookup = _pair_lookup(run_path) if include_text else {}
    results = []
    for pair_id, metadata, distance in zip(ids, metadatas, distances):
        item = {
            "pair_id": pair_id,
            "distance": round(float(distance), 4),
            "embedding_similarity": round(1 - float(distance), 4),
            "metadata": metadata or {},
        }
        if include_text:
            pair = lookup.get(str(pair_id)) or {}
            item["context"] = pair.get("context") or []
            item["target"] = pair.get("target") or {}
        results.append(item)

    return {
        "ok": True,
        "run_id": run_path.name,
        "model": manifest.get("model") or model_name,
        "result_count": len(results),
        "results": results,
        "raw_text_policy": (
            "Query results return metadata only by default. include_text=True reads local rag_pool text for debugging."
        ),
    }


def format_embedding_query_result(result: Dict[str, Any], *, include_text: bool = False) -> str:
    if not result.get("ok"):
        return str(result.get("message") or "embedding 检索失败。")
    lines = [
        "Stage 5B embedding 检索：",
        f"- run_id：{result.get('run_id')}",
        f"- model：{result.get('model')}",
        f"- 命中：{result.get('result_count', 0)} 条",
    ]
    for index, item in enumerate(result.get("results") or [], start=1):
        metadata = item.get("metadata") or {}
        lines.append(
            f"{index}. sim={item.get('embedding_similarity')} dist={item.get('distance')} "
            f"{metadata.get('chat_type')} {metadata.get('scene_label')} "
            f"{metadata.get('scope')} {metadata.get('grounding_type')} "
            f"score={metadata.get('score')} {metadata.get('source_file_id')}"
        )
        if include_text:
            context_text = _turns_text(item.get("context") or [])
            target = item.get("target") or {}
            if context_text:
                lines.append("   历史上下文：" + context_text[:300])
            if target.get("text"):
                lines.append("   主人回复：" + str(target.get("text"))[:180])
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build/query local Stage 5B embedding index.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build embedding index for latest or specified Stage 5B run.")
    build.add_argument("--run-dir", default=None)
    build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build.add_argument("--no-reset", action="store_true")
    status = sub.add_parser("status", help="Show embedding index status.")
    status.add_argument("--run-dir", default=None)
    query = sub.add_parser("query", help="Query embedding index.")
    query.add_argument("text")
    query.add_argument("--run-dir", default=None)
    query.add_argument("--limit", type=int, default=DEFAULT_QUERY_LIMIT)
    query.add_argument("--include-text", action="store_true")
    args = parser.parse_args()

    if args.command == "build":
        print(json.dumps(
            build_stage5b_embedding_index(
                args.run_dir,
                batch_size=args.batch_size,
                reset=not args.no_reset,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return
    if args.command == "status":
        print(json.dumps(embedding_index_status(args.run_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "query":
        print(format_embedding_query_result(
            query_stage5b_embedding_index(
                args.text,
                run_dir=args.run_dir,
                limit=args.limit,
                include_text=args.include_text,
            ),
            include_text=args.include_text,
        ))


if __name__ == "__main__":
    _main()
