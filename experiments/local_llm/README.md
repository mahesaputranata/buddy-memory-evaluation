# B7 — local Llama 3.2 versus Qwen 2.5 answering pilot

This is the first **answer-generation** experiment. B0–B6 evaluated memory access. B7 asks whether your installed models use the same supplied evidence accurately. It does not train, fine-tune, modify Buddy, or execute rover commands.

## Models identified on your machine

| Model | Parameters reported by Ollama | Quantization | Advertised context limit | Previously listed short ID |
| --- | --- | --- | ---: | --- |
| `llama3.2:latest` | 3.2B | Q4_K_M | 131,072 | `a80c4f17acd5` |
| `qwen2.5:3b` | 3.1B | Q4_K_M | 32,768 | `357c53fb659c` |

The pilot uses **4,096 runtime context tokens for both**, not each model's different maximum. Llama is your current Buddy model; Qwen is your teacher model. That history does not determine which will perform better. Model tags can change; the script records live full digests and `/api/show` metadata.

## What is held fixed?

- The same 15 development questions, source records, context ordering, and system instructions.
- B5 raw-exchange retrieval (`A_raw`), with cross-encoder reranking OFF.
- Generation options: temperature 0, seed 2026, context 4,096, output limit 256 tokens, top-k 40, top-p 0.9, repetition penalty 1.0.
- A fresh two-message request per case, rather than carrying conversation history between cases.
- JSON response format for both models. Native model-specific chat templates remain Ollama's responsibility.

This is a controlled pilot, not a held-out or production benchmark. Different parameter counts, tokenizers, templates, and architectures still exist. The explicit task instructions and evidence content are identical.

## Two evidence conditions

| Condition | Input | Purpose |
| --- | --- | --- |
| `retrieved` | The exact B5 selected context | Evaluate answers with actual retrieval omissions and distractors |
| `reference` | Source excerpts manually selected using the answer key | Diagnose answering ability when the necessary evidence is supplied |

The reference condition contains original evidence, not the written expected answer. It is an idealized diagnostic control and must not be reported as production retrieval performance.

Example: the stop-history query has only one of its two required requests in B5's selected context. If an answer is incomplete, retrieval may be responsible. In the reference condition both stop requests are supplied. If the answer still confuses stopping with scenario configuration, the failure lies in using the evidence.

The model never receives the condition name, gold flags, written reference answer, or retrieval metrics. The harness constructs model input using an explicit allowlist of question and quoted excerpts. `evaluation_only` fields in the cases file are used for review only.

## Run on Windows PowerShell

1. Extract this ZIP to a normal folder, for example `C:\Users\Mahesa\Buddy_B7_Local_LLM`.
2. Open Ollama as usual. Use your normal local installation; no API key is needed.
3. Open PowerShell in the extracted folder.
4. Check Python:

```powershell
py --version
```

If your installation uses `python` instead, replace `py` in the commands below. The runner uses the Python standard library: there is no pip dependency or GPU framework to install.

Optional input inspection without contacting models:

```powershell
py .\run_local_llm_eval.py --dry-run
```

First smoke test: 3 questions × actual retrieval × 2 models = 6 measured answers, plus model warmups:

```powershell
py .\run_local_llm_eval.py --limit 3 --conditions retrieved
```

Full pilot: 15 questions × 2 conditions × 2 models = 60 measured answers, plus warmups:

```powershell
py .\run_local_llm_eval.py
```

Optional repetitions, once the first pilot works:

```powershell
py .\run_local_llm_eval.py --repeats 3
```

Repeated runs measure output/runtime variation on the same questions; they do not create independent accuracy samples. The runner never pulls missing models automatically. If Ollama is not reachable, open the Ollama app or start `ollama serve` in a separate terminal if no service is running.

For local model latency, run with Buddy idle and avoid simultaneous heavy GPU tasks. The script records `/api/ps` after warmup to inspect model placement; it does not forcibly stop other models or change your Ollama configuration. Automatic GPU offload and model residency may affect results.

## Outputs

Each run creates its own result folder and ZIP:

| Output | Contents |
| --- | --- |
| `answers.jsonl` | Raw/parsed answers, timing, token counts, validation, prompt fingerprint, server completion data |
| `prompt_packets.json` | Exact model-facing questions and evidence, without the answer key |
| `manifest.json` | Options, live model digests/details, Ollama version, model warmups and residency, system prompt, limitations |
| `automatic_summary.csv` | JSON/schema checks, citation-ID checks, median latency/token statistics |
| `manual_review.csv` | Question, model answer, expected answer, actual supplied evidence, and blank scoring fields |

Progress and answers are saved after each request. An interrupted run can leave a partial folder; preserve it for debugging. The script creates a new run rather than silently overwriting a prior result.

These files contain your real conversation excerpts. Inspect them before publishing a portfolio repository.

## How to score answers

Automatic checks do **not** establish factual accuracy. Valid JSON is just format compliance. A cited source ID being present only proves that the ID exists in the supplied packet, not that it supports the answer.

In `manual_review.csv`, enter 1 or 0 for each of:

| Field | 1 means |
| --- | --- |
| `fact_correct_0_or_1` | Factual statements agree with the reference evidence, with no invented numbers/events |
| `faithful_to_context_0_or_1` | Claims are supported by the evidence actually supplied in that condition |
| `requested_parts_complete_0_or_1` | All parts requested by the question are answered |
| `attribution_correct_0_or_1` | User requests and assistant reports are distinguished from verified execution |
| `appropriate_uncertainty_0_or_1` | The answer acknowledges material missing/unverified evidence without needless refusal |

Blank means unreviewed, not failure. Explain judgments in `review_note`. Preserve the raw model output, including malformed or truncated answers. `done_reason=length` indicates the output cap may have cut the answer off; do not silently retry with different limits for one model.

An answer can be faithful but incomplete when actual retrieval omitted a needed exchange. An unsupported lucky guess may match the full-log answer but is not faithful memory use. Do not treat the top-three packet as an exhaustive event history, especially for all/count questions.

Example: “Buddy reported 59 seconds and about 2 fps” is appropriately attributed. “The camera was independently verified at exactly 2 fps” overstates what the log establishes.

## Timing interpretation

- `client_elapsed_ms`: elapsed time for the local Ollama chat request.
- `time_to_first_content_ms`: time until the first nonempty response fragment (including JSON punctuation), not necessarily the first meaningful word.
- `load_ms`: server-reported model loading time; warmup responses are kept separately.
- `prompt_eval_ms` / `generation_ms`: server-reported input processing and token generation times.
- Output token count and generation token rate help explain different answer lengths.

These are answering-stage timings with **frozen context**. The runner does not recompute retrieval, so it does not measure retrieval-plus-generation or complete voice-loop latency. Do not add historical Colab retrieval times to local generation times and label the sum a local end-to-end benchmark.

Model warmup removes much first-load overhead, but Ollama's prompt/KV cache and block-by-model order can affect latency. Every request has fresh conversational messages, which is not the same as a guaranteed cold server cache. Cached-prompt counts are retained when the installed Ollama version supplies them. Repeat with reversed model order for stronger latency evidence:

```powershell
py .\run_local_llm_eval.py --models qwen2.5:3b llama3.2:latest
```

Do not judge model quality solely by speed or schema compliance. Fresh sessions, unanswerable cases, larger histories, and live retrieval integration remain later tests.

## Source and API documentation

Frozen data: `buddy_memory_results_B5.zip`, input SHA-256 `a414a1ae7cad794f0e7f1f56068c5e4e7e0cc8f8e55d168ab196c4a50633651e`. `source_b5_manifest.json` and the original input snapshot are included for audit.

API: [Ollama chat](https://docs.ollama.com/api/chat), [runtime parameter reference](https://docs.ollama.com/modelfile). The actual model test must be run on your machine; the package was checked with input validation and a simulated API, not your real model outputs.
