"""B6 Colab extension. Run with %run -i after the B5 setup.
Requires encoder, CFG, clean, aliases, QUESTIONS, candidate_memories,
filter_candidates_by_session, and replay from the notebook.
No generative LLM. No gold labels enter scoring. Model quality is unmeasured
until this code is executed with the real encoder/reranker in Colab.
"""
import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

CFG.setdefault("use_cross_encoder", False)  # Main on/off switch.
CFG.setdefault("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
CFG.setdefault("cross_encoder_revision", None)  # Set commit hash to pin weights.
CFG.setdefault("cross_encoder_device", "auto")  # auto / cpu / cuda
CFG.setdefault("cross_encoder_batch_size", 16)
CFG.setdefault("cross_encoder_max_length", 512)

_B6_MODEL_POOL = {}


def _b6_sync():
    # Synchronize CUDA work before/after wall-clock timing.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def initialize_b6_reranker(cfg):
    """Load once, outside query latency. False switch never loads the model."""
    if not cfg.get("use_cross_encoder", False):
        return None
    import torch
    from sentence_transformers import CrossEncoder
    device = cfg.get("cross_encoder_device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (cfg["cross_encoder_model"], cfg.get("cross_encoder_revision"),
           device, int(cfg["cross_encoder_max_length"]))
    if key not in _B6_MODEL_POOL:
        _b6_sync()
        start = time.perf_counter()
        model = CrossEncoder(
            cfg["cross_encoder_model"], revision=cfg.get("cross_encoder_revision"),
            device=device, max_length=int(cfg["cross_encoder_max_length"]),
        )
        _b6_sync()
        backbone = getattr(model, "model", model)
        model_config = getattr(backbone, "config", None)
        parameters = getattr(backbone, "parameters", None)
        dtype = str(next(parameters()).dtype) if callable(parameters) else None
        _B6_MODEL_POOL[key] = {
            "model": model, "device": device,
            "load_ms": (time.perf_counter() - start) * 1000,
            "resolved_revision": getattr(model_config, "_commit_hash", None),
            "dtype": dtype,
        }
    return _B6_MODEL_POOL[key]


def _b6_predict(state, question, texts, cfg):
    import torch
    # Raw relevance logits: never use a cosine threshold on these values.
    scores = state["model"].predict(
        [(question, text) for text in texts],
        batch_size=int(cfg["cross_encoder_batch_size"]),
        show_progress_bar=False, convert_to_numpy=True,
        activation_fn=torch.nn.Identity(),
    )
    values = np.asarray(scores).reshape(-1)
    if len(values) != len(texts) or not np.isfinite(values).all():
        raise ValueError("Reranker returned invalid relevance scores")
    return values


def retrieve_b6(question, candidates, cfg):
    """Return selected items, characters, and stage timing/ranking details."""
    state = initialize_b6_reranker(cfg)  # Initialization excluded below.
    stats = {"embedding_similarity_ms": 0.0, "rerank_ms": 0.0,
             "selection_ms": 0.0, "eligible_count": 0,
             "reranked_count": 0, "ranking": []}
    if not candidates:
        return [], 0, stats
    _b6_sync()
    start = time.perf_counter()
    qvec = encoder.encode([question])[0]
    vectors = encoder.encode([c["search_text"] for c in candidates])
    cosine_scores = vectors @ qvec
    # Preserve B5's cosine eligibility threshold in both conditions.
    eligible = np.flatnonzero(cosine_scores >= cfg["min_retrieval_score"])
    _b6_sync()
    stats["embedding_similarity_ms"] = (time.perf_counter() - start) * 1000
    stats["eligible_count"] = int(len(eligible))
    rerank_scores = {}
    if state is not None and len(eligible):
        _b6_sync()
        start = time.perf_counter()
        values = _b6_predict(state, question,
                             [candidates[i]["search_text"] for i in eligible], cfg)
        _b6_sync()
        stats["rerank_ms"] = (time.perf_counter() - start) * 1000
        stats["reranked_count"] = int(len(eligible))
        rerank_scores = {int(i): float(v) for i, v in zip(eligible, values)}
        order = eligible[np.argsort(-values, kind="stable")]
    else:
        order = eligible[np.argsort(-cosine_scores[eligible], kind="stable")]

    start = time.perf_counter()
    selected, covered, used = [], set(), 0
    for i in order:
        candidate = candidates[int(i)]
        sources = set(candidate["source_ids"])
        if not sources - covered:
            continue
        size = len(candidate["context_text"])
        if used + size > cfg["context_char_budget"]:
            continue
        selected.append({**candidate, "score": float(cosine_scores[i]),
                         "reranker_score": rerank_scores.get(int(i)),
                         "ranking_method": "cross_encoder" if state else "cosine"})
        covered.update(sources)
        used += size
        if len(selected) >= cfg["top_k"]:
            break
    stats["selection_ms"] = (time.perf_counter() - start) * 1000
    # Logging is built after selection timing, outside total retrieval timer below.
    stats["_rank_order"] = [int(i) for i in order]
    stats["_cosine_scores"] = [float(v) for v in cosine_scores]
    stats["_rerank_scores"] = rerank_scores
    return selected, used, stats


def retrieve(question, candidates, cfg):
    """Drop-in replacement preserving the notebook's two-value interface.
    For accurate timing/export use evaluate_b6 or benchmark_b6 below.
    """
    selected, chars, _ = retrieve_b6(question, candidates, cfg)
    return selected, chars


def _b6_cases(cfg):
    # Replay is offline preparation, excluded from query retrieval latency.
    trace_local, _ = replay(clean, cfg)
    positions = {r["id"]: i for i, r in enumerate(clean)}
    cases = []
    for q in QUESTIONS:
        stop = positions[aliases[q["after_id"]]]
        prefix = clean[:stop + 1]
        gold = {aliases[s] for s in q["gold_source_ids"]}
        assert gold and gold <= {r["id"] for r in prefix}
        for mode in ["A_raw", "B_manual", "C_selective"]:
            cases.append((q, prefix, set(trace_local.iloc[stop].promoted_ids), mode))
    return cases, trace_local


def _b6_run_case(case, cfg):
    q, prefix, promoted, mode = case
    initialize_b6_reranker(cfg)  # Exclude lazy loading from request timing.
    # Warm-document/cold-query policy: encode each query anew for BOTH paths.
    if not hasattr(encoder, "cache"):
        raise ValueError("Benchmark requires the notebook Encoder.cache interface")
    encoder.cache.pop(q["question"], None)
    _b6_sync()
    start = time.perf_counter()
    prep_start = start
    candidates = candidate_memories(prefix, mode, promoted)
    candidates, scope = filter_candidates_by_session(q["question"], prefix, candidates, cfg)
    prepare_ms = (time.perf_counter() - prep_start) * 1000
    selected, chars, stages = retrieve_b6(q["question"], candidates, cfg)
    _b6_sync()
    total_ms = (time.perf_counter() - start) * 1000

    # Metrics, detailed rankings and serialization are outside request timing.
    sources = {s for item in selected for s in item["source_ids"]}
    gold = {aliases[s] for s in q["gold_source_ids"]}
    hits = len(sources & gold)
    first = next((i for i, item in enumerate(selected, 1)
                  if set(item["source_ids"]) & gold), None)
    row = {"qid": q["qid"], "split": q.get("split", "dev"), "mode": mode,
           "use_cross_encoder": bool(cfg.get("use_cross_encoder", False)),
           "evidence_recall": hits / len(gold),
           "evidence_precision": hits / len(sources) if sources else 0.0,
           "complete_evidence": float(gold <= sources), "MRR": 1 / first if first else 0.0,
           "source_count": len(sources), "context_chars": chars,
           "candidate_count": len(candidates), "prepare_ms": prepare_ms,
           "total_retrieval_ms": total_ms, "retrieval_ms": total_ms,
           **{k: stages[k] for k in ["embedding_similarity_ms", "rerank_ms", "selection_ms",
                                    "eligible_count", "reranked_count"]}}
    rankings = []
    selected_ids = {m["memory_id"] for m in selected}
    for rank, i in enumerate(stages.pop("_rank_order", []), 1):
        c = candidates[i]
        rankings.append({"memory_id": c["memory_id"], "source_ids": c["source_ids"],
                         "rank": rank, "cosine_score": stages["_cosine_scores"][i],
                         "reranker_score": stages["_rerank_scores"].get(i),
                         "selected": c["memory_id"] in selected_ids,
                         "search_text": c["search_text"]})
    detail = {"qid": q["qid"], "mode": mode, "question": q["question"],
              "use_cross_encoder": row["use_cross_encoder"], "session_scope": scope,
              "gold_source_ids": sorted(gold), "reference_answer": q.get("reference_answer"),
              "retrieved": selected, "candidate_ranking": rankings}
    return row, detail


def _b6_prewarm(cases, cfg):
    # Populate document vectors before timing; never cache reranker predictions.
    texts = []
    for q, prefix, promoted, mode in cases:
        cs = candidate_memories(prefix, mode, promoted)
        cs, _ = filter_candidates_by_session(q["question"], prefix, cs, cfg)
        texts.extend(c["search_text"] for c in cs)
    if texts:
        encoder.encode(list(dict.fromkeys(texts)))
    _b6_sync()


def evaluate_b6(cfg):
    """One condition; compatible metrics/details return, warm documents."""
    initialize_b6_reranker(cfg)
    cases, _ = _b6_cases(cfg)
    _b6_prewarm(cases, cfg)
    results = [_b6_run_case(case, cfg) for case in cases]
    return pd.DataFrame([r for r, _ in results]), [d for _, d in results]


def benchmark_b6(cfg, repeats=5, warmups=1,
                 output_dir="buddy_B6_results", seed=2026):
    """Paired OFF/ON benchmark. Returns a dictionary and exports a ZIP.
    Request latency includes candidate building, session filtering, query
    encoding, similarity, optional reranking, and final context selection.
    Excludes model download/loading, offline grouping replay, metrics,
    plots, and disk writes. Documents are cached; queries are re-encoded.
    """
    if repeats < 2 or warmups < 1:
        raise ValueError("Use repeats >= 2 and warmups >= 1")
    base = copy.deepcopy(cfg)
    off, on = ({**base, "use_cross_encoder": value} for value in [False, True])
    state = initialize_b6_reranker(on)
    cases, trace_local = _b6_cases(base)
    _b6_prewarm(cases, base)
    print("Loaded reranker:", on["cross_encoder_model"], "on", state["device"])
    print("Separate initialization/download time (ms):", round(state["load_ms"], 2))
    for _ in range(warmups):
        for i, case in enumerate(cases):
            for condition in ([off, on] if i % 2 == 0 else [on, off]):
                _b6_run_case(case, condition)
    rows, last_details = [], []
    rng = random.Random(seed)
    for repeat in range(repeats):
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        for sequence, index in enumerate(indices):
            order = [off, on] if (repeat + sequence) % 2 == 0 else [on, off]
            for condition in order:
                row, detail = _b6_run_case(cases[index], condition)
                row["repeat"] = repeat + 1
                rows.append(row)
                if repeat == repeats - 1:
                    last_details.append(detail)
        print(f"Finished paired repetition {repeat + 1}/{repeats}")

    runs = pd.DataFrame(rows)
    keys = ["repeat", "qid", "mode"]
    paired = runs[runs.use_cross_encoder].merge(
        runs[~runs.use_cross_encoder], on=keys, suffixes=("_on", "_off"), validate="one_to_one")
    paired["delta_total_ms"] = paired.total_retrieval_ms_on - paired.total_retrieval_ms_off
    paired["delta_recall"] = paired.evidence_recall_on - paired.evidence_recall_off
    paired["delta_complete"] = paired.complete_evidence_on - paired.complete_evidence_off
    paired["delta_MRR"] = paired.MRR_on - paired.MRR_off

    summary = runs.groupby(["mode", "use_cross_encoder"]).agg(
        median_total_ms=("total_retrieval_ms", "median"),
        p90_total_ms=("total_retrieval_ms", lambda x: x.quantile(0.9)),
        median_rerank_ms=("rerank_ms", "median"),
        mean_recall=("evidence_recall", "mean"),
        complete_fraction=("complete_evidence", "mean"),
        mean_MRR=("MRR", "mean"), mean_context_chars=("context_chars", "mean"),
        mean_reranked_count=("reranked_count", "mean")).reset_index()
    delta_summary = paired.groupby("mode").agg(
        median_paired_delta_ms=("delta_total_ms", "median"),
        p10_delta_ms=("delta_total_ms", lambda x: x.quantile(0.1)),
        p90_delta_ms=("delta_total_ms", lambda x: x.quantile(0.9)),
        mean_delta_recall=("delta_recall", "mean"),
        mean_delta_complete=("delta_complete", "mean")).reset_index()
    print("Quality and warm-request latency:")
    display(summary)
    print("Paired delta t = ON minus OFF (negative values can reflect timing noise):")
    display(delta_summary)

    # Two small panels: speed cost and complete-evidence coverage.
    modes = ["A_raw", "B_manual", "C_selective"]
    labels = ["Raw", "Manual", "Selective"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(3)
    for enabled, color, offset, label in [(False, "#2563eb", -0.18, "OFF"),
                                           (True, "#ea580c", 0.18, "ON")]:
        part = summary[summary.use_cross_encoder == enabled].set_index("mode").loc[modes]
        axes[0].bar(x + offset, part.median_total_ms, width=0.34, color=color, label=label)
        axes[1].bar(x + offset, part.complete_fraction * 100, width=0.34, color=color, label=label)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_title("Total retrieval latency")
    axes[0].set_ylabel("Median milliseconds; lower is faster")
    axes[1].set_title("Questions with complete evidence")
    axes[1].set_ylabel("Percent; higher is better")
    axes[1].set_ylim(0, 105)
    fig.suptitle("B6: cross-encoder OFF versus ON")
    fig.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "all_runs.csv", index=False)
    paired.to_csv(out / "paired_deltas.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    delta_summary.to_csv(out / "delta_summary.csv", index=False)
    trace_local.to_csv(out / "replay_trace.csv", index=False)
    (out / "retrieval_details.json").write_text(json.dumps(last_details, indent=2))
    annotations = {"questions": QUESTIONS,
                   "manual_memories": globals().get("MANUAL_MEMORIES", []),
                   "group_gold": globals().get("GROUP_GOLD", {}),
                   "trigger_gold": globals().get("TRIGGER_GOLD", {}),
                   "important_record_ids": sorted(globals().get("IMPORTANT_RECORD_IDS", set()))}
    (out / "annotations.json").write_text(json.dumps(annotations, indent=2))
    original = globals().get("records", clean)
    snapshot = "\n".join(json.dumps({k: v for k, v in r.items() if k != "_ts"},
                                    ensure_ascii=False) for r in original) + "\n"
    (out / "input_snapshot.jsonl").write_text(snapshot)
    packages = {}
    for name in ["sentence-transformers", "transformers", "torch", "numpy", "pandas", "matplotlib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {"experiment": "B6_paired_OFF_ON", "base_config": base,
                "conditions": {"off": off, "on": on}, "repeats": repeats,
                "warmup_passes": warmups, "seed": seed,
                "python": platform.python_version(), "packages": packages,
                "reranker_device": state["device"], "model_load_ms": state["load_ms"],
                "resolved_reranker_revision": state.get("resolved_revision"),
                "reranker_dtype": state.get("dtype"),
                "input_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
                "cache_policy": "document vectors warm; query vector removed before each call; no reranker score cache",
                "timing_excludes": ["model loading/download", "offline replay/grouping", "metric computation", "file/plot output"],
                "limitations": ["same development questions; not held out",
                                "repetitions measure timing variation, not independent accuracy samples",
                                "cross-encoder max_length may truncate long input pairs",
                                "both models remain loaded during paired timing"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    fig.savefig(out / "latency_quality_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "latency_quality_comparison.pdf", bbox_inches="tight")
    plt.show()
    archive_path = Path(str(out) + ".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    print("Exported:", archive_path)
    return {"runs": runs, "paired": paired, "summary": summary,
            "delta_summary": delta_summary, "details": last_details,
            "zip_path": str(archive_path)}


print("B6 loaded. Main switch: CFG['use_cross_encoder'] = True or False")
print("For paired timing: b6 = benchmark_b6(CFG, repeats=5, warmups=1)")
