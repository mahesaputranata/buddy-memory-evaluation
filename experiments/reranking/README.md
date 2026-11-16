# Finish the decision: Llama with reranking OFF versus ON

**One command. 30 answers. No Colab rerun. No new model.**

1. Extract this ZIP.
2. Open PowerShell in the folder containing `run_comparison.py`.
3. Keep Ollama open and Buddy idle.
4. Run:

```powershell
py .\run_comparison.py
```

If `py` is unavailable, use `python` instead. The code needs only Python's standard library and your installed `llama3.2:latest` model.

This automatically runs:

- 15 questions using the actual B6 **reranking OFF** selected contexts.
- The same 15 questions using the actual B6 **reranking ON** selected contexts.

Both runs use Llama, the unchanged B7 system prompt, temperature 0, seed 2026, runtime context 4,096 tokens, output cap 256, and fresh conversational messages. There is a warmup before each condition. The data comes from the B6 `retrieval_details.json` you supplied; no reranker inference happens locally in this test. These frozen contexts already include the measured effect of reranking. Source-role labels and record IDs are preserved. Written expected answers are kept out of model inputs.

The original B7 Python runner is reused unchanged as `ollama_eval_core.py`. Only the supplied cases differ. The same answer-format checker limitation remains: schema failure can also mark citation-ID validation false. We judge actual answer content separately, so this reporting issue does not change model input or output. Llama previously followed the schema consistently.

## What to send back

At completion, the terminal prints:

```text
UPLOAD THIS RESULTS ZIP: ...Llama_reranking_results_....zip
```

Upload **that generated results ZIP**, not this setup ZIP. It includes a readable `READ_RESULTS_FIRST.md`, a paired review CSV, both raw answer logs and timing summaries, and the exact model/condition metadata.

If interrupted, preserve the partial output folders; do not assume they contain a completed paired result. The program deliberately avoids overwriting earlier runs.

## How we make the decision

Review all 15 OFF/ON answer pairs. Check whether ON answers more requested facts correctly and completely, makes fewer unsupported claims, and avoids material regressions. Do not count fluent prose or valid JSON as proof of factual correctness. Identical answer text can still differ in source support when the supplied evidence changes.

**If ON provides no meaningful answer-quality benefit, keep reranking OFF and close this branch for this pilot.** If ON improves answers, document the exact cases and tradeoffs before deciding to keep it. One run of 15 development questions cannot establish universal performance.

The original failure is still informative: B6 OFF and ON both omit the final stop-following record from their selected context. The answering model must not replace that missing event with a scenario-setting request. This test checks whether changed context ordering/content alters the answer; it does not claim retrieval completeness was fixed.

## Latency caveat

The previous full run reported several seconds of model loading on most requests. This test preserves `load_ms` separately. It measures answer-request latency with frozen contexts, **not live retrieval-plus-reranking-plus-generation**. Do not add the earlier Colab reranking overhead to local answer times as a verified local end-to-end total. Prompt caching and OFF/ON block order can also affect timing.

An optional order-effect check, only if later needed:

```powershell
py .\run_comparison.py --on-first
```

Do not run extra branches before reviewing the first 30 answers.

Optional input validation without model requests:

```powershell
py .\run_comparison.py --dry-run
```

The included case files contain real conversation excerpts and expected answers for review. Inspect them before publishing them in a public repository.
