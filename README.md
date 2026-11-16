# Buddy conversational-memory and LLM evaluation

An offline evaluation study for a voice-controlled rover assistant: can it retrieve the right conversations, and can a local LLM answer correctly from them?

**Author:** I Made Mahesa Putranata.

**Status:** research prototype; no live rover commands. 

This repository separates the offline study from the running [Buddy assistant](https://github.com/mahesaputranata/buddy).

## What the study found

- Real dataset: 40 recorded exchanges across four sessions; two likely logging duplicates excluded, leaving 38 processed records and 37 group labels.
- Semantic grouping improved pair recall from 51.02% to 83.67% versus lexical hashing; ARI increased from 0.5852 to 0.7623 on the manually labeled development data.
- Session filtering increased complete raw-retrieval evidence from **12/15 to 14/15 development queries**. This is evidence coverage, not answer accuracy.
- Two local quantized models, Llama 3.2 and Qwen2.5, produced 60 responses to identical retrieved/reference-evidence packets.
- A further 30 Llama responses compared cross-encoder OFF/ON contexts. No convincing task benefit was observed; the extra reranker remains OFF by default.

**Two different failures:** the final “Stop following me” exchange was missing from retrieved stop-history context, while all three backward requests were present but Llama answered with movement distances. Retrieval completeness and correct evidence use require separate evaluation.

Read the [full experiment history](docs/experiment_history.md) for B0-B7, exact measurements, examples, model configurations, and limitations. This notebook implements the latest B5 baseline with optional B5.5/B6 extensions; it does not reconstruct every historical notebook revision.

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/01_memory_retrieval.ipynb` | Clean notebook, no saved outputs; B5 plus optional diagnostics |
| `scripts/run_retrieval.py` | CLI extracted from the notebook; intentionally retains notebook-style globals |
| `scripts/buddy_B5_5_rank_diagnostic.py` | Full ranking diagnostic, run inside the notebook namespace |
| `scripts/buddy_B6_switchable_reranker.py` | Optional paired retrieval/reranker benchmark extension |
| `data/benchmark/` | Minimal real-log fields and development annotations |
| `experiments/local_llm/` | B7 runner and frozen context packets |
| `experiments/reranking/` | Llama OFF/ON answering comparison and frozen packets |
| `results/reference/` | Historical aggregate answering summaries, not fresh reruns |
| `results/local/` | Ignored output location for new runs |
| `docs/` | Full experiment history, data notes, publication instructions |

## Run retrieval locally

Python 3.11+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_retrieval.py --output results/local/b5
```

The default uses `sentence-transformers/all-MiniLM-L6-v2`; model weights download on first use. For a quick run without downloading embedding weights:

```bash
python scripts/run_retrieval.py --backend hashing --output results/local/hashing_check
```

**The hashing run is a smoke test, not reproduction of the semantic B5 result.** Choose a new output path for each CLI run. Override data with `--logs your.jsonl --annotations your_annotations.json`. The benchmark annotations reference specific records and must be replaced when using unrelated logs. Optional `--config overrides.json` accepts CFG overrides; make backend changes in a fresh process so encoder and config agree.

For the notebook, launch `jupyter lab` from the root and open `notebooks/01_memory_retrieval.ipynb`. Run top-to-bottom. Optional `RUN_RANK_DIAGNOSTIC` and `RUN_B6` are False; turning B6 on permits additional model downloads and benchmarking. Notebook exports overwrite its named local export; copy/archive outputs before another run.

### Google Colab

Clone your published repository into Colab, change the working directory into the cloned root, then run the notebook. Keep the notebook's `ROOT` pointing to the repository root. The notebook installs `requirements.txt`; it no longer requires manual uploads of the extension scripts. Runtime/device differences mean timing is not directly comparable across Colab and your local machine.

## Run the local answering experiments

Install/start Ollama on the machine where inference should run and make these exact model tags available:

```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:3b
python experiments/local_llm/run_local_llm_eval.py --dry-run --output results/local/b7_inputs
python experiments/local_llm/run_local_llm_eval.py --output results/local/b7
python experiments/reranking/run_comparison.py --dry-run --output results/local/rerank_inputs
python experiments/reranking/run_comparison.py --output results/local/rerank_answers
```

Dry runs make no model requests. The local runners use the Python standard library; the retrieval dependencies are not needed for frozen-context answering alone. If Colab runs retrieval, copy the repository to your local computer for Ollama: Colab's localhost is not your home computer.

Frozen packets deliberately isolate answering from retrieval. Changing the notebook does **not** update those packets automatically. The OFF/ON answering run does not execute a cross-encoder locally and cannot measure its runtime overhead. Model tags can change upstream; historical digests/configurations are recorded in the experiment documentation.

## Evaluation boundaries

- All 15 questions are development questions repeatedly inspected during iteration; no held-out generalization claim.
- Manual memories are authored controls. No automatic episode summarizer is evaluated.
- Grouping does not feed retrieval ranking in this implementation.
- Gold labels/answers are evaluation-only. The reference-evidence condition uses gold IDs to select excerpts, but does not send expected answers to the model.
- Assistant statements are attributed claims, not verified telemetry or command receipts.
- Complete source coverage, JSON validity, citation-ID membership, and correct answers are distinct measures.
- The historical B7 checker gates source-ID validity on schema validity. Qwen's 6.67% reported value is **not** a fabricated-citation rate. See the documented checker limitation before interpreting summaries; the frozen runner is preserved, not silently fixed.
- Model loading dominated these local timings; no steady-state voice latency or production scaling claim.
- Inspired by memory/retrieval ideas in RecMem and EverMemOS; not a reproduction of those systems.

## Data and publication

The benchmark contains the author's real utterances and assistant replies, including imperfect transcription and profanity. Only IDs, timestamps, sessions, turns, inputs, and replies are included in the minimal JSONL; routing/shadow metadata is excluded. Context packets also reproduce benchmark excerpts. See [data notes](docs/data_notes.md).

Local results/private data/model weights are ignored by Git. The author has not selected an open-source license; public visibility alone does not grant reuse rights. Select an appropriate license before advertising the project as open source.

See [publishing instructions](docs/publishing.md) for GitHub or GitLab. Do not commit credentials, raw new sessions, model weights, notebook outputs, or unreviewed inference exports.
