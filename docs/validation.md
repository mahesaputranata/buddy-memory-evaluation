# Packaging validation - 16 November 2026

Performed while preparing this repository:

- Compiled all Python scripts successfully.
- Executed CLI retrieval with `--backend hashing` on the bundled real benchmark; source-overlap assertions passed and exports were produced.
- Executed the cleaned notebook baseline with hashing: 38 retained records and 15 development questions; optional B5.5/B6 sections were not enabled.
- Executed B7 and reranking answering runners with `--dry-run`; no Ollama requests were sent.
- Verified notebook outputs/execution counts and saved widget metadata are absent.
- Verified the minimal benchmark contains 40 rows and the annotations contain 15 questions.

Not rerun here: sentence-transformer inference, cross-encoder inference, or real Ollama generation. Historical quality/timing measurements are preserved in `experiment_history.md` and reference summaries; packaging checks are not fresh semantic-quality benchmarks. Dependency requirements are supported ranges, not the original environment lockfile. Refer to historical manifests/documentation for the recorded run environments.
