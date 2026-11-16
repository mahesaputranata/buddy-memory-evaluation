# Buddy memory experiment: B0–B7 results and local LLM reranking comparison

**Author:** I Made Mahesa Putranata  
**Recorded interactions:** 13 November 2026  
**Status:** B0–B6 retrieval experiments, B7 local Llama/Qwen pilot, and final Llama reranking comparison documented. Working baseline: B5 raw retrieval with cross-encoder OFF. Retrieval and evidence-use failures remain; live Buddy improvement is not yet established.  
**Scope:** Offline memory grouping, consolidation-trigger inspection, evidence retrieval, and local generated-answer evaluation using real Buddy interaction logs. B0–B6 evaluate retrieval without a generative answering model; B7 and the final comparison add Llama/Qwen answering. No robot commands are executed by this evaluation.

## 1. Problem and purpose

Buddy is a voice-first assistant used for rover interaction and engineering workflows. During a real session, the user may connect the rover, request movement, configure an experiment, start or stop following, and request saved trial results. Rephrasing, speech-transcription errors, mixed requests, and assistant-generated unsupported statements make these exchanges difficult to organize reliably.

This experiment investigates whether a memory pipeline can preserve and retrieve the right evidence across those interactions. It is isolated from Buddy's running system so that memory decisions can be inspected before deployment.

The baseline establishes three questions:

1. Do online similarity groups match manually annotated activities within each session?
2. When does a simple recurrence policy request consolidation, and what information does it leave unconsolidated?
3. Do manually supplied memories improve evidence retrieval over raw exchanges under the same retrieval limits?

**Observed baseline:** lexical grouping splits several related activities; retrieval finds all three reference records in its top-three results, but one query ranks an irrelevant record first. A reported precision gain comes from duplicated evidence. Group assignments currently do not influence retrieval.

The experiment borrows ideas from [RecMem](https://arxiv.org/abs/2605.16045) and [EverMemOS](https://arxiv.org/abs/2601.02163). It is not a reproduction of either paper and does not measure their reported performance.

## 2. Evidence and reproducibility

The source of the measured results is `buddy_memory_results.zip`, exported from `Buddy_Memory_Experiment_Colab.ipynb`. Keep this original archive unchanged when introducing future versions.

| Export | Role |
| --- | --- |
| `manifest.json` | Settings, environment, input fingerprint |
| `input_snapshot.jsonl` | Original 40 interaction records used by this run |
| `annotations.json` | User-reviewed primary-task group labels, manual memories, development questions |
| `duplicate_candidates.json` | Candidate duplicate logging events |
| `replay_trace.csv` | Online groups, recurrence counts, triggers, promoted sources |
| `retrieval_details.json` | Ranked returned items, context, scores, and evidence references |
| `retrieval_metrics.csv` | Per-question and per-method retrieval measurements |
| `summary.csv` | Mean retrieval measurements |
| `annotation_worksheet.csv` | Blank worksheet; not the authoritative populated group labels |
| `claim_review.csv` | Claim-review template; review judgments remain blank |

The input fingerprint matches the exported input bytes:

```text
a414a1ae7cad794f0e7f1f56068c5e4e7e0cc8f8e55d168ab196c4a50633651e
```

During review, the retrieval metrics were numerically reproduced from the saved annotations and configuration using the available notebook implementation. Grouping metrics below were independently calculated from `replay_trace.csv` and `annotations.json`; they are not present in `summary.csv`. Execution times were not treated as reproducible benchmark results.

### Dataset

| Item | Count |
| --- | ---: |
| Raw records | 40 |
| Sessions | 4 |
| Likely duplicate logging events excluded | 2 |
| Processed records | 38 |
| Processed records with primary-task gold labels | 37 |
| Predicted groups | 23 |
| Manual memory entries | 3 |
| Development retrieval questions | 3 |
| Held-out questions | 0 |
| Consolidation-trigger gold labels | 0 |

Session identifiers are `2026-11-13-080507`, `2026-11-13-085042`, `2026-11-13-085916`, and `2026-11-13-091923`. The raw session counts are 20, 6, 2, and 12 records. Row numbers in this document refer to the chronological 40-record input, including the excluded duplicates.

These are real recorded interactions, not synthetic conversations. The manual memories and development questions were subsequently authored for this pilot. Assistant replies and recorded routing labels are observations, not automatically correct answer keys or verified hardware outcomes.

### Duplicate handling

| Record | Retained match | Gap | Decision |
| --- | --- | ---: | --- |
| #6 `7074068a` | #5 `f57ca9d6` | 0.000550 s | Excluded as likely duplicate logging |
| #15 `295d8a50` | #14 `fa578fc0` | 0.001031 s | Excluded as likely duplicate logging |

The heuristic requires identical normalized input and response in the same session within one second. It does not delete the original snapshot. Repeated commands outside this criterion remain separate interactions.

## 3. Baseline implementation

### Configuration

```json
{
  "backend": "hashing",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "group_similarity": 0.45,
  "recurrence_similarity": 0.45,
  "recurrence_count": 3,
  "within_session": true,
  "near_duplicate_seconds": 1.0,
  "drop_duplicate_candidates": true,
  "top_k": 3,
  "context_char_budget": 2400,
  "min_retrieval_score": 0.01,
  "include_assistant_in_embedding": false
}
```

The active encoder is **word hashing**, not the configured optional sentence-transformer model. The hashing implementation uses 4,096 features, word unigrams and bigrams, nonnegative hashing, and L2 normalization. It is a lexical baseline: it does not establish semantic equivalence or understand action meaning.

### Processing stages

1. Validate JSONL records and sort timestamps chronologically.
2. Flag and optionally exclude likely duplicate logging events.
3. Encode each user input. Assistant replies are excluded from the search embedding in B0.
4. Compare with existing group centroids in the same session; merge above similarity 0.45, otherwise create a group.
5. Separately count earlier exchanges with pairwise similarity at least 0.45. Trigger when there are at least three earlier matches; the current record is not counted.
6. Preserve a cumulative set of promoted source IDs when triggers fire. This is not a generated summary or a complete consolidation/update lifecycle.
7. At each query checkpoint, retrieve raw exchanges and eligible manual memories using the same top-three item limit and 2,400-character maximum context.

Both encoders supported by the notebook are fixed rather than fitted on future records. Manual memories become available only when all cited source records have arrived. Future source records are excluded at each checkpoint.

### Retrieval variants

| Variant | Available items |
| --- | --- |
| A: `A_raw` | Raw exchanges only |
| B: `B_manual` | Raw exchanges plus all chronologically available manual memories |
| C: `C_selective` | Raw exchanges plus manual memories whose entire source set has been promoted |

Raw history remains searchable in all variants. Each returned manual memory includes its cited raw exchanges in the context; those exchanges count against the character budget. A summary citation alone is not credited as retrieved evidence.

**Architectural boundary:** group IDs do not enter retrieval ranking or filtering. Recurrence uses pairwise exchange similarity, not predicted group membership. Consequently, grouping quality and retrieval quality are separate evaluations in B0. The experiment does not yet test whether group-aware retrieval helps.

## 4. Annotation policy

Gold groups represent the **same primary activity within the same session**. Examples include connection, basic movement, discussion, measurement configuration, following, stopping, data retrieval, an audio/aliveness check, and a surprise reaction. Session prefixes prevent unrelated occurrences on different sessions from being treated as a single episode.

The user reviewed these categories. Group labels are evaluation-only; the grouping algorithm never receives them. The initial scenario-exit record #1 is unlabeled. Excluded duplicates #6 and #15 are not scored.

The current evaluator supports only one gold group per record. Three interactions have secondary purposes:

| Record | Primary label | Secondary interpretation |
| --- | --- | --- |
| #4 | Basic movement | Possible information request; interpretation remains ambiguous |
| #18 | Fetch data | Discussion/interpretation |
| #30 | Rover connection | Measurement configuration |

Additionally, #17 and #40 are labeled as stopping, although both explicitly concern following. This exposes an overlap between activity and action categories. Comments about secondary labels are not used by the evaluator.

Manual memories preserve source IDs and distinguish user requests/reports from unverified assistant claims. Those status fields are descriptive metadata in B0; they do not automatically verify facts or enforce retrieval behavior.

## 5. Measured grouping results

Metrics use the 37 labeled processed records. Pair precision asks whether predicted same-group pairs share a gold label. Pair recall asks whether gold same-group pairs were recovered.

| Metric | Value |
| --- | ---: |
| Correct same-group pairs (TP) | 25 |
| Incorrect same-group pairs (FP) | 8 |
| Missed same-group pairs (FN) | 24 |
| Pair precision: TP / (TP + FP) | 75.76% |
| Pair recall: TP / (TP + FN) | 51.02% |
| Adjusted Rand Index | 0.5852 |

These are clustering-agreement measures, not intent-classification accuracy. Many singleton groups and the small dataset limit interpretation.

### Observed grouping failures

| Failure ID | Evidence | Observed behavior | Interpretation |
| --- | --- | --- | --- |
| G1 | Session 1 movement: #3, #4, #5, #7, #9, #10, #11, #12 | Split among `g002`, `g003`, `g005`, `g006` | Forward, backward, and turning requests were not recognized as one manually defined activity |
| G2 | Session 2 backward requests: #23, #24, #25 | Assigned `g013`, `g014`, `g015` | Rephrasing/transcription variation fragmented a related sequence |
| G3 | Trial logging: #18, #19, #20 | #18–19 in `g009`; #20 in `g010` | A longer request with frustration and repeated wording was separated |
| G4 | Mixed-purpose #30 | Joined measurement group `g019`, but gold primary label is connection | Single-label scoring penalizes a plausible secondary activity |
| G5 | Stop-following #40 | Joined following group `g020`, but gold primary label is stop | Activity/action overlap and absence of action-aware distinction |
| G6 | Stop requests #37 and #40 | Separate groups | Missed a gold stop-task pair |

All eight false-positive pairs involve #30 or #40. This is an annotation/evaluator limitation as well as a grouping mismatch. Of the 24 false-negative pairs, 18 are session 1 movement, three are session 2 movement, two are session 1 logging, and one is session 4 stopping.

Lexical similarity is a plausible explanation for these patterns, but a controlled semantic-encoder comparison is still needed to establish how much it contributes. Lowering a threshold may merge fragmented groups while also increasing incorrect merges.

## 6. Consolidation-trigger observations

| Arrival | Related prior exchanges | Trigger reason |
| --- | ---: | --- |
| #9 | 3 | Recurrence |
| #11 | 5 | Recurrence |
| #33 | 3 | Recurrence |
| #35 | 4 | Recurrence |
| #39 | 5 | Recurrence |

Five of 38 processed arrivals triggered. This is a count of trigger opportunities, not LLM calls made, time saved, or measured token savings.

Observed limitations:

- Repeated requests to set the scenario to “stop and go” trigger again; the implementation does not test novelty or maintain a summarized-through boundary.
- The one-time fragile-design statement never triggers. An importance exception exists but no important record IDs were supplied.
- Three logging exchanges do not automatically satisfy a requirement for three **earlier** matches. Their lexical similarity also varies.
- No trigger answer key exists, so trigger precision/recall cannot be claimed.
- The cumulative promoted-source list can contain earlier sessions. This is historical retention, not proof that grouping crossed session boundaries; similarity comparisons remain session-scoped.

The selective variant had no manual memory available at the GPS query. At the fragile-design and saved-run queries, only the GPS memory was available. The fragile-design and trial-summary memories were never activated by the selective gate at those checkpoints.

## 7. Measured retrieval results

All three questions are development questions and use checkpoints within session 1. Sessions 2–4 are not assessed by these retrieval questions.

| Question | Reference source | First raw result | Reference rank |
| --- | --- | --- | ---: |
| GPS claim during forward request | #4 `69bd62a8` | #4 | 1 |
| User's fragile-design statement | #8 `56162f0d` | #8 | 1 |
| Saved follow-me run identifier | #20 `c867c278` | #17, “Buddy, stop following me.” | 2 |

All variants return the reference source in their top three. The run-identifier query demonstrates that evidence recall can be perfect while the best-ranked item is irrelevant to the requested identifier.

| Variant | Evidence recall | Complete evidence | Evidence precision* | MRR | Mean context characters |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: raw | 100% | 100% | 33.33% | 0.8333 | 806.00 |
| B: manual memories | 100% | 100% | 44.44% | 0.8333 | 1,023.67 |
| C: selective | 100% | 100% | 33.33% | 0.8333 | 806.00 |

MRR is the reciprocal rank of the first item containing reference evidence, averaged over the three questions. Each question currently has one reference source. Characters are not model tokens. The budget is an upper limit, not an equal-sized context allocation.

### Precision artifact: duplicate evidence

For the GPS question, B returns the manual GPS memory citing #4, then raw #4, then raw #5. The precision function counts unique source IDs, so this becomes one relevant source among two unique sources: 50%.

For the fragile-design question, B similarly returns raw #8 and a manual memory citing #8, plus another raw record. Its precision is also 50%.

This consumes retrieval slots and repeats evidence. It does not establish a genuine improvement in relevance. The mean context is approximately 27% larger in B than A. The reported precision should be retained as an original B0 metric but marked with this limitation; future versions should add overlap-aware selection and redundancy measurements.

### Indexing limitation

Raw retrieval searches user text only, although questions ask about information in Buddy's replies. The run identifier and GPS claim occur in assistant replies, not the indexed user request. Matching is therefore indirect. A future exchange-level or separate user/assistant index should retain attribution rather than treating assistant text as verified fact.

The manual trial summary was eligible in B, but its lexical score was about 0.0546, below the three selected raw items. It was not rejected because of the character budget.

## 8. Original Buddy behavior versus memory-pipeline failures

| Source | Recorded behavior | What the log establishes |
| --- | --- | --- |
| #4 | Buddy mentions a GPS-coordinate error while responding to a movement request | An assistant claim with no supporting tool receipt in this export; not a verified diagnosis |
| #29 | Buddy explains the stop-and-go scenario as a traffic-light sequence | An assistant interpretation requiring checking against the actual scenario implementation |
| #33, #35, #39 | Scenario-configuration requests receive “stop” responses | Visible request/response mismatch; actual hardware commands are not independently established here |
| #24 | Buddy reports an approximate cumulative position | A reported state, not independently confirmed odometry |
| #20 | Buddy reports a run identifier, duration, visibility, and frame rate | Reported measurements and a claimed save; the original run artifact is absent |

B0 replays these observations; it does not fix routing, execute commands, or verify telemetry. The manual summaries correctly attribute selected assistant claims, but no claim-review judgments have been completed. A statement can be supported as “Buddy said X” while X itself remains physically unverified.

## 9. Limits on conclusions

This pilot supports documenting reproducible failure examples and implementation boundaries. It does not support claims of production reliability, hallucination reduction, improved robot safety, automatic summarization accuracy, semantic-memory superiority, or paper-equivalent performance.

Specific limitations are the one-day sample, four short sessions, only three development questions, no held-out data, single-label simplification, manual memory construction, no scored trigger policy, no generated answers, no complete summary-quality review, and no live tool receipts.

Timing is descriptive only: methods execute in a fixed order and share cached vectors, and the measured region excludes grouping/consolidation. Differences do not establish deployment speedups. No statistical significance or generalization claim is made.

## 10. Proposed improvements and evaluation plan

The following roadmap was established from B0. P1 is partially completed by B1: fully redundant evidence is skipped and redundancy was checked during archive review; automated redundancy reporting and partial-overlap handling remain future work. P3 has an initial fixed-threshold comparison in B2; broader validation and calibration remain pending. The other changes are planned. Preserve the B0, B1, and B2 archives and label changed runs separately.

| ID | Proposed change | Question to test | Evidence to retain |
| --- | --- | --- | --- |
| P1 | Avoid returning redundant source evidence; export redundancy metrics | Does context become more useful without losing evidence recall? | Unique sources, repeated sources, item count, context characters |
| P2 | Add questions spanning all sessions, paraphrases, multi-record needs, and negative cases | Does the baseline fail beyond the initial sanity checks? | Versioned questions, reference evidence, explicit negative-case scoring |
| P3 | Compare fixed lexical and semantic encoders | Does semantic representation reduce fragmentation? | Group metrics and retrieval metrics under controlled settings |
| P4 | Represent multiple activities per interaction | Are secondary-activity groups being unfairly scored? | Versioned multi-label policy and suitable evaluator |
| P5 | Introduce an explicit group-aware retrieval variant | Does organization actually improve evidence selection? | Flat versus group-aware retrieval using identical checkpoints/budgets |
| P6 | Add novelty/update tracking and define trigger gold | Can consolidation avoid repeated work while capturing important events? | Trigger decisions, new-source counts, manually reviewed trigger labels |
| P7 | Compare user-only indexing with attributed exchange indexing | Can reply-specific information be retrieved directly? | Ranking, relevance, attribution inspection |
| P8 | Evaluate automatic consolidation after manual-input experiments | How much quality is lost when a model supplies summaries/facts? | Supported claims, unsupported claims, omissions, exact values, source references |

Metric/evaluator changes must be distinguished from model improvements. If duplicate selection or gold-label policy changes, rerun a comparable control under the new evaluation rather than claiming a delta against incompatible B0 numbers. Tune on development sessions and freeze choices before testing newly collected held-out sessions.

## 11. Future result ledger

| Run | Dataset/evaluator | Change | Group pair P/R | ARI | Evidence recall | Ranking/redundancy | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | This 40-record snapshot; primary-task gold; 3 dev queries | Lexical baseline | 75.76% / 51.02% | 0.5852 | 100% on 3 dev queries | MRR 0.8333; B repeats evidence | Frozen |
| B1 | Same snapshot, annotations, and 3 dev queries as B0 | Skip fully covered source evidence; hashing retained | 75.76% / 51.02% | 0.5852 | 100% on 3 dev queries | MRR 0.8333; 0 fully redundant selected items | Evaluated |
| B2 | Same snapshot, annotations, and 3 dev queries | Sentence-transformer encoder; B1 selection retained; thresholds fixed | 73.21% / 83.67% | 0.7623 | 100% on 3 dev queries per variant | MRR A/B/C: 0.8333 / 1.0000 / 0.6667; no fully redundant items | Evaluated |
| B3 | Same logs and labels; 15 development queries | Evaluation expansion only; B2 algorithm retained | 73.21% / 83.67% | 0.7623 | A/B/C: 81.11% / 87.78% / 81.11% | Complete evidence A/B/C: 11/15, 12/15, 11/15 | Evaluated |
| B4 | Same 15 development queries as B3 | Index user + assistant reply for raw retrieval only | 73.21% / 83.67% | 0.7623 | A/B/C: 91.11% / 87.78% / 91.11% | Complete evidence: 12/15 in every variant; MRR A/B/C: 0.8444 / 0.8556 / 0.8333 | Frozen |
| B5 | Same 15 development queries as B4 | Resolve session scope and filter before ranking | 73.21% / 83.67% | 0.7623 | 96.67% in every variant | Complete evidence: 14/15 in every variant | Frozen |
| B5.5 | One failed B5 question; all 12 session-4 raw candidates | Full-exchange versus user-only ranking diagnostic; production retrieval unchanged | Unchanged | Unchanged | Stop-history recall 1/2 under both ranking views at top 3 | Actual stop requests rank 2 & 6 (full), 2 & 8 (user-only) | Completed diagnostic |
| B6 | Same 15 development queries; paired OFF/ON | Optional cross-encoder relevance reranking | No intended grouping change; B6 replay export not supplied | Not independently checked for B6 | 96.67% in every variant, OFF and ON | 14/15 complete in every variant; ranking mixed; about 9.8–9.9 ms higher median latency | Evaluated; retain OFF for working baseline |

For each new run record: date, code version/commit when available, input and annotation versions, actual encoder and thresholds, development/test split, exact change, per-case regressions, and the reason for accepting or rejecting it. No code commit identifier was recorded in B0; do not infer one from the exported artifact.

## 12. Portfolio description supported by the baseline

> Built an offline replay and evaluation workflow for Buddy using 40 real interaction records across four rover sessions. Added source-linked manual memories and task annotations, evaluated online grouping and evidence retrieval, and diagnosed activity fragmentation, mixed-intent evaluation limits, and a duplicate-evidence scoring artifact. Established a frozen baseline, then removed fully redundant selected evidence without losing reference evidence on three development questions; mean context length in the manual-memory variant fell by 15.5%. A subsequent semantic-encoder comparison increased grouping pair recall from 51.0% to 83.7% and ARI from 0.585 to 0.762 on the same labeled sample, while exposing retrieval regressions in the selective variant.

Present this as a pilot engineering investigation. Later portfolio updates should state the measured before/after change, dataset size, evaluation policy, and held-out results. Record IDs provide traceable examples; inspect and redact personal content before publishing raw conversations.

## 13. B1: remove fully redundant selected evidence

**Result status:** Verified from `buddy_memory_results_B1.zip` on 13 November 2026. This section records measured changes; sections 1–9 retain the original B0 results.

### Controlled change

Only retrieval selection was intentionally changed. A candidate is skipped if all its source IDs are already represented by previously selected items. This policy is appropriate here because returned manual memories include their cited raw exchanges. It does not establish that a summary alone can replace its source.

The original input snapshot, annotations, duplicate-candidate file, and replay trace are byte-identical between B0 and B1. Manifest configuration values match. The encoder remains hashing. Grouping and trigger outputs are unchanged. The archive does not include a versioned notebook/code snapshot; output inspection verifies the expected selection behavior, rather than proving the absence of every unexported code change.

### Concrete before/after examples

| Question | B0 selected items | B1 selected items |
| --- | --- | --- |
| GPS claim during forward movement | GPS summary including the original forward request and GPS reply (#4); the same raw GPS exchange again (#4); forward confirmation (#5) | GPS summary including the original exchange (#4); forward confirmation (#5); rover-connection request (#2) |
| Fragile-design statement | Original “my design is very fragile” exchange (#8); a manual summary citing that same exchange (#8); another forward request (#9) | Original fragile-design exchange (#8); forward request (#9); scenario-exit request (#1) |
| Saved follow-me run identifier | Stop-following request (#17); trial report containing the identifier (#20); forward request (#7) | Same selections as B0 |

The raw-only and selective variants select the same items as B0. The changed manual-memory selections contribute three distinct sources instead of repeating one source. Replacement items can still be irrelevant: a rover-connection request does not answer the GPS question. B1 fixes redundancy, not semantic ranking.

### Verified measurements

| Measurement | B0 | B1 | Interpretation |
| --- | ---: | ---: | --- |
| Fully redundant selected items across all 9 query/variant combinations | 2 | 0 | Expected selection behavior confirmed |
| Method B: mean context characters | 1,023.67 | 865.33 | 15.47% reduction, rounded to 15.5% |
| Method B: mean evidence recall | 100% | 100% | Reference evidence retained on all 3 development questions |
| Method B: mean complete evidence | 100% | 100% | Same single-source reference sets recovered |
| Method B: MRR | 0.8333 | 0.8333 | Reference ranking unchanged |
| Method B: reported unique-source precision | 44.44% | 33.33% | Earlier duplication-related advantage removed |
| Method A: mean context characters | 806.00 | 806.00 | Unchanged |
| Method C: mean context characters | 806.00 | 806.00 | Unchanged |

“Fully redundant” means a selected item's entire source set is already covered by earlier selected items for the same question and variant. A selected multi-source item may still overlap partially with prior evidence; B1 does not remove repeated passages inside it. The count above was calculated from `retrieval_details.json` during review, rather than exported as a dedicated metric column.

The precision decrease is not evidence of lost reference information: the original denominator counted distinct sources while duplicate items occupied slots. The 15.5% context reduction applies only to method B on these three development queries. It is not a token-cost, latency, generated-answer, or generalization claim.

### Remaining failures

- The saved-run question still ranks “Buddy, stop following me” above the exchange containing the run identifier.
- Group fragmentation and repeated consolidation triggers remain exactly as in B0.
- Retrieval still searches user text only for raw exchanges, although some questions concern assistant replies.
- Predicted groups still do not influence retrieval.
- Negative cases, held-out questions, and generated-answer correctness are not evaluated.

**B1 conclusion:** skipping fully covered evidence removed the observed duplicated selections while preserving reference evidence on the small development set. No improvement in semantic relevance or answer accuracy has been demonstrated.

## 14. B2 setup: semantic embeddings with the B1 selector

**Status:** Completed; actual results are recorded in section 15. The setup below documents the controlled change used for B2.

Question: does replacing lexical hashing with a fixed semantic encoder improve grouping and evidence ranking under the same initial thresholds and evaluation setup?

The existing `Encoder` class accepts the exact backend string `sentence_transformer`. It does not accept `semantic` and raises `ValueError` for that value.

Install the optional dependency in a separate Colab cell if needed:

```python
%pip install -q sentence-transformers
```

Use this configuration, retaining the B1 `retrieve()` implementation:

```python
CFG = {
    "backend": "sentence_transformer",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "group_similarity": 0.45,
    "recurrence_similarity": 0.45,
    "recurrence_count": 3,
    "within_session": True,
    "near_duplicate_seconds": 1.0,
    "drop_duplicate_candidates": True,
    "top_k": 3,
    "context_char_budget": 2400,
    "min_retrieval_score": 0.01,
    "include_assistant_in_embedding": False,
}
```

The initial model download requires internet access. No generative LLM or API key is required. Restart the runtime if package installation requests it, then run the notebook from the beginning. Merely editing `CFG` without rerunning the encoder cell leaves the old encoder and cache active. Recreate `encoder = Encoder(CFG)` before replay and evaluation; running all cells in order does this.

Keep the same input, annotations, manual memories, questions, and B1 evidence-selection rule. Do not tune thresholds during this first B2 comparison. Both grouping/recurrence and retrieval use the encoder, so both can change; their outcomes must be interpreted separately.

**Threshold interpretation:** a cosine score of 0.45 does not have an equivalent empirical meaning for hashing vectors and learned sentence embeddings. B2 is a fixed-threshold encoder swap, not a comparison of each encoder at its best calibrated setting. Later threshold calibration should be a separately labeled development experiment.

After running, inspect the manifest to confirm `backend` is `sentence_transformer`, check the duplicate-evidence assertion, and compare grouping metrics, trigger events, per-question ranks, retrieved context, and character usage. In particular inspect backward-movement paraphrases, trial-logging requests, start versus stop following, and the saved-run query. Semantic similarity may capture shared topic while still confusing opposite actions.

Save a separate `buddy_memory_results_B2.zip` and retain the corresponding notebook/code. If the notebook exports the default ZIP name, rename the downloaded copy before archiving it. Record the sentence-transformers and underlying model package versions, and the resolved model revision when available, because the current manifest lists only the original core packages. The reviewed outputs are reported below; encoder-package versions and a resolved model revision are not captured by the supplied manifest.

## 15. B2 measured results: semantic encoder at fixed thresholds

**Evidence:** `buddy_memory_results_B2.zip`, compared with B1 on 13 November 2026. Input snapshot, annotations, and duplicate-candidate exports are byte-identical. The only changed manifest configuration field is `backend`: `hashing` becomes `sentence_transformer`, using `sentence-transformers/all-MiniLM-L6-v2`. All grouping, recurrence, and retrieval thresholds remain fixed. Returned selections continue to satisfy B1's no-fully-redundant-evidence condition.

The archive documents outputs and configuration, not a complete code diff or resolved model weights. The findings below are derived from the exports; B2 was not independently rerun with downloaded model weights during review.

### Grouping comparison

| Measure | B1: hashing | B2: semantic |
| --- | ---: | ---: |
| Processed records / labeled records | 38 / 37 | 38 / 37 |
| Predicted groups | 23 | 18 |
| Correct same-group pairs | 25 | 41 |
| Incorrect same-group pairs | 8 | 15 |
| Missed same-group pairs | 24 | 8 |
| Pair precision | 75.76% | 73.21% |
| Pair recall | 51.02% | 83.67% |
| Adjusted Rand Index | 0.5852 | 0.7623 |

B2 reduces fragmentation: recall rises by 32.65 percentage points, while precision falls by 2.54 percentage points. Fewer groups alone is not evidence of better grouping; the annotation-based measures support improved overall agreement on this sample.

| Interaction context | B1 behavior | B2 behavior |
| --- | --- | --- |
| Session 2 backward movement: “move backward,” “backward a bit more,” and “just move backward” (#23–25) | Three groups | One group, `g010` |
| Session 1 logging/results requests, including the longer frustrated request (#18–20) | Two groups | One group, `g007` |
| Session 1 forward movement, move-back/forward/turn sequence, and turn-around request (#4, #5, #7, #9–12) | Split across multiple groups | One group, `g003` |
| First session's initial backward-test request (#3) | Separate from forward requests | Still separate, `g002` |
| “Buddy, stop following me” in session 1 (#17) | Separate stop group | Joins movement group `g003` |
| “Stop following me” in session 4 (#40) | Joins following group | Still joins following group, now `g015` |
| “Stop the rover” in session 4 (#37) | Separate from stop-following | Still separate, `g017` |
| Combined scenario-setting and connection request (#30) | Joins measurement group | Still joins measurement group, `g014` |

The system emits group IDs, not semantic names such as “basic command.” Movement-related stop grouping can be defensible at a broad level but disagrees with the frozen primary-task gold policy. Starting and stopping following can share an activity while requiring opposite actions. An activity/action or multi-label policy is a future evaluation change; B2 gold labels were not retroactively changed to improve scores.

### Consolidation triggers

Triggers increase from five to eight. The original triggers remain; three additional arrivals trigger:

- Another forward-movement request (#7).
- “Buddy, stop following me” in the first session (#17).
- “Stop following me” in the fourth session (#40).

These additional events reflect changed similarity, not demonstrated improvement in importance selection. No trigger gold exists. The fragile-design memory and the combined trial-report memory remain unavailable to the selective retrieval variant at their tested checkpoints. Raw history remains searchable.

### Retrieval comparison

All three development questions still recover their reference evidence within the returned context in all variants. Results are mixed rather than uniformly improved.

| Variant | Mean reference recall B1 → B2 | MRR B1 → B2 | Unique-source precision B1 → B2 | Mean context characters B1 → B2 |
| --- | --- | --- | --- | --- |
| A: raw exchanges | 100% → 100% | 0.8333 → 0.8333 | 33.33% → 33.33% | 806.00 → 922.67 |
| B: raw + manual memories | 100% → 100% | 0.8333 → 1.0000 | 33.33% → 28.89% | 865.33 → 1,441.33 |
| C: selective manual memories | 100% → 100% | 0.8333 → 0.6667 | 33.33% → 33.33% | 806.00 → 1,050.00 |

MRR describes the first item carrying reference evidence, not generated-answer accuracy. Context size increases in every variant; no context-efficiency gain is shown for B2.

| Question | A: raw | B: manual | C: selective |
| --- | --- | --- | --- |
| GPS claim during forward movement | Reference falls from rank 1 to 2; backward-test request ranks first | GPS memory remains first | Reference falls from rank 1 to 2; backward-test request ranks first |
| Fragile-design statement | Correct exchange remains first | Correct exchange remains first | Correct exchange remains first |
| Saved follow-me run identifier | Correct trial-report exchange rises from rank 2 to 1 | Manual trial report rises to rank 1 | An unrelated GPS memory ranks first; correct trial report remains second |

The selective saved-run regression is important. The manual trial summary is gated out because its sources were not promoted, while the GPS memory is available and ranks above the raw trial-report record. More semantic similarity therefore does not guarantee better selection under the existing gate.

Method B's trial-report item includes three original records: requests for logging/results and the final report. With the unrelated GPS and stop-following items also selected, this question returns five distinct sources, only one of which is the current gold reference. Its precision is 1/5 = 20%, contributing to the 28.89% average. Some additional sources provide useful surrounding context, but this question's answer key asks only for the identifier. Gold relevance, source count, and context utility must be distinguished.

No fully redundant selected items were found across the nine question/variant results. Timing remains unsuitable for speedup claims because of encoder loading/encoding differences, shared caches, fixed method order, and the small sample.

### B2 conclusion

Semantic embeddings improve grouping agreement on this annotated sample and improve first-reference ranking for the all-manual-memory variant. They do not solve selective retrieval, opposite-action distinctions, consolidation policy, or context efficiency. The primary grouping remains disconnected from retrieval. These are pilot findings at fixed thresholds, not a validated production memory architecture.

## 16. Evaluation roadmap and stopping rule

Freeze B2 as the initial semantic result. Do not repeatedly tune against the same three questions until every displayed score is perfect.

**Original B3 plan (now completed for the semantic run): expand evaluation without changing the algorithm.** Add roughly 12–20 manually reviewed development questions across all four sessions. Include paraphrases, scenario configuration versus stop commands, the requested versus assistant-reported state, similar tasks in different sessions, and multi-record evidence. Add unanswerable questions only after specifying how abstention will be scored: the current evaluator assumes nonempty reference evidence and is not ready for negative cases.

Run B1 and B2 against this same expanded set. A changed question set creates a new evaluation version; do not compare its averages directly with the old three-question averages. Grouping labels should remain frozen for this comparison. If activity/action labels are introduced, score both encoders again under the new policy.

After this diagnostic comparison, implement at most one focused correction supported by failures. The user-only search index is a plausible candidate because GPS details and saved identifiers live in assistant replies; compare attributed full-exchange indexing with the same control setup. Group-aware retrieval should be an explicitly separate variant if the goal is to test the original organization-to-retrieval idea.

Finally collect at least one later, independent session, annotate its questions, and keep it held out while choosing settings. Check reference recall, first-reference ranking, source redundancy, attribution, and context budget. A single later session is a practical minimum for a portfolio pilot, not proof of broad generalization.

A reasonable stopping point for this portfolio milestone is: reproducible B0–B2 archives; an expanded, versioned development evaluation; one diagnosed improvement if needed; and an honest held-out report including regressions. Stop before integrating into live Buddy if evidence is insufficient or the change provides no practical benefit. Automatic summarization, live execution, and full paper reproduction can remain later milestones.

## 17. B3: same system, longer exam

**Evidence:** `buddy_memory_results_B3.zip`, reviewed on 13 November 2026. This run uses B2's semantic encoder and B1's evidence-selection fix. The change is 12 additional development questions, bringing the total to 15. No new interaction records were added.

### Reminder for my future self

> B2 changed how the system compares text. B3 did not change the system again. B3 asked that same system more questions to expose what it still gets wrong.

Think of testing the rover on three driving trials, then testing the same rover on fifteen more varied trials. A lower overall score on the larger test does not mean someone made the rover worse.

Input records, grouping labels, manual memories, and the grouping/trigger trace match B2. The original three questions have the same selected items and non-timing metric values. Only the evaluation coverage increased. The uploaded B3 contains the semantic run; a hashing comparison on the same 15 questions has not been supplied.

### What was measured?

| Method | Mean evidence recall | Questions with all required evidence | MRR | Mean context characters |
| --- | ---: | ---: | ---: | ---: |
| A: raw exchanges | 81.11% | 11/15 | 0.7222 | 826.93 |
| B: raw + available manual memories | 87.78% | 12/15 | 0.8222 | 1,153.33 |
| C: selective manual memories | 81.11% | 11/15 | 0.6889 | 905.47 |

Recall means the fraction of reference records found, averaged per question. Finding two out of three required requests gives that question 2/3 recall. “Complete evidence” requires all its reference records. MRR can be high after finding just one relevant item; it does not guarantee complete history retrieval. None of these measures scores a generated answer.

Grouping metrics remain B2's 73.21% pair precision, 83.67% pair recall, and 0.7623 ARI because grouping inputs and outputs did not change. Do not compare the new 15-question averages directly with B2's old three-question averages as an improvement/regression claim.

### Example 1: “I asked backward three times. Why did you find only two?”

In the second session, I made these three requests at different moments:

| Source | Recorded request |
| --- | --- |
| #23 `f5cd2ce6` | “Now can you move backward a bit?” |
| #24 `f60db3be` | “Can ive backward a bit more?” |
| #25 `5364461d` | “Buddy, at this point, just move backward a bit.” |

Then the test asks:

> “Buddy, show me all three backward requests from this second session, including the badly transcribed backward-a-bit-more one.”

All three methods return:

1. The last backward request: “just move backward a bit” (#25).
2. The badly transcribed “backward a bit more” request (#24).
3. An unrelated earlier **forward** request: “Buddy, can you move forward a bit?” (#7).

The first backward request (#23) is missing. The question's evidence recall is therefore 2/3, and complete evidence is false.

The critical distinction:

| Stage | What happened |
| --- | --- |
| Grouping | All three backward requests were correctly placed together in `g010` |
| Retrieval | Only two were selected for the question |
| Answer generation | Not run; there is no answering LLM in this experiment |

So the issue is not “Buddy grouped only two backward requests.” It is “the right records were stored and grouped, but the search did not return the complete evidence set.” Retrieval does not use `g010` to expand the history.

A future question such as “How many times did I ask you to move backward?” needs a defined time/session scope and complete matching-event collection. Counting whatever happens to be in the top-three search results is not a reliable counting method. B3 did not test a generated answer saying “twice.”

### Example 2: “What did we do in the third session?”

The third session contains a connection request, “Buddy, I need you to connect to the rover” (#27), followed by “Kind of start following me” (#28).

The test asks for those two requests in order. Every method instead returns earlier forward-movement requests from session 1. Neither required source is retrieved, so recall is zero.

The metadata exists, but retrieval does not translate “third session” into a session filter. `within_session=True` restricts grouping and recurrence comparisons, not the raw retrieval candidate pool. Later session-selection logic is needed; simply replacing an embedding model cannot reliably resolve this requirement.

### Example 3: “Tell me the recording duration, visibility, and camera speed”

The final trial-report exchange (#20) contains Buddy's reported 59-second duration, 99% visibility, and approximately 2 fps camera processing.

The test asks for those three reported figures. Raw retrieval misses the report. The all-manual-memory method retrieves the trial summary first, which includes that source exchange. The selective method misses it because the trial summary has not been activated by the recurrence gate.

This illustrates two separate problems:

- Raw search embeds my request to log the trial, but not Buddy's reply containing the requested measurements.
- The selective policy can withhold a useful manually supplied memory.

This is evidence for the usefulness of the manual summary on this question, not proof that automatic LLM summarization works.

### Example 4: “Show both ways I asked you to stop”

In session 4, I say “Stop the rover, buddy” within a frustrated request (#37), then later say “Stop following me” (#40). The test asks for both.

Each method finds the frustrated stop request but misses the final stop-following request. Other selected records concern scenario configuration and rover connection. Recall is 1/2. A shared topic or the word “stop” does not guarantee the relevant action history is complete.

### Example 5: “I asked the same thing twice, but what did you reply just now?”

“Set the scenario to stop and go” appears as the same user wording in an earlier exchange (#29) and a later exchange (#33). The earlier reply describes traffic lights; the later reply says “Okay, stop command sent.”

Because raw search embeds only the user input, both exchanges have the same search representation. The earlier exchange wins the ranking tie; the requested later reply is second. Session/recency handling and reply-aware indexing are separate mechanisms to evaluate.

### B3 conclusion

The larger test revealed incomplete multi-record retrieval, session confusion, reply-specific indexing gaps, and a useful-summary availability problem. No fully redundant selected item was found under the B1 guard, although partially overlapping source sets can still occur. The system has not become worse on its original three questions; the new questions expose previously untested weaknesses. All questions remain development examples based on the same one-day logs.

## 18. B4 guide: let retrieval search Buddy's reply too

**Status:** Completed; verified results are recorded in section 19. This is a focused indexing experiment. It is not a complete fix for B3's backward-counting, session-selection, or consolidation-policy problems.

### Simple example of the change

Original exchange:

- My request: “Log this trial. I want to see the result.”
- Buddy's reply: “59 seconds ... 99 percent ... about 2 frames per second.”

| Version | Text searched for raw records | Potential consequence |
| --- | --- | --- |
| B3 | My request only | Search cannot directly match the measurements in the reply |
| B4 | My request plus the attributed assistant reply | Search can match reply-specific concepts without promoting them to verified facts |

The complete exchange already appears in returned context. B4 changes the text used to rank raw records, not what facts are independently verified.

### Keep grouping unchanged

Do **not** simply set the existing `include_assistant_in_embedding` flag to true: that flag also changes grouping and recurrence. For this focused experiment leave it false, and add a separate retrieval-only flag:

```python
CFG["include_assistant_in_embedding"] = False
CFG["retrieval_include_assistant"] = True
```

Keep the semantic encoder, B1 `retrieve()` function, all 15 questions, manual memories, labels, thresholds, and budgets unchanged.

### Add a retrieval-only text function

Place this function before `candidate_memories()` in the retrieval/evaluation cell:

```python
def raw_retrieval_text(record, cfg):
    if cfg.get("retrieval_include_assistant", False):
        return (
            "USER REQUEST OR REPORT:\n"
            + record["input"]
            + "\nASSISTANT RESPONSE (not independently verified):\n"
            + record["response"]
        )
    return exchange_embedding_text(record, cfg)
```

Inside `candidate_memories()`, replace only this raw-candidate field:

```python
"search_text": exchange_embedding_text(r, CFG),
```

with:

```python
"search_text": raw_retrieval_text(r, CFG),
```

Leave the manual-memory `search_text` as `m["text"]`. Do not change `exchange_embedding_text()` itself: grouping and recurrence still use it. Role markers identify whose words are being indexed; they do not verify claims or enforce trustworthy model reasoning.

### Run and inspect

Run all cells from the beginning after saving this change. The semantic model's first-party package requirements remain the same as B2; no generative LLM is involved. Export separately as `buddy_memory_results_B4.zip` and retain the exact modified notebook.

Check the new manifest flag and verify the replay trace is unchanged from B3. If grouping or trigger results change, investigate before interpreting retrieval differences. The grouping/recurrence inputs should be identical in this isolated experiment.

Compare all 15 questions, especially:

| Case | What to look for |
| --- | --- |
| Recording duration, visibility, and camera speed | Does the actual reported-measurements exchange now appear? |
| GPS claim during forward movement | Does the relevant reply rank higher? |
| Identical scenario requests with different replies | Does ranking improve, and does the requested recent reply remain identifiable? |
| Three backward requests | Check for improvement, but do not expect reply indexing to guarantee complete retrieval |
| Third-session request sequence | Session selection remains an unresolved mechanism |

Retain regressions as well as gains. Full replies can add noise and unsupported claims to the search representation; more searchable text is not guaranteed to help every question. All reported numbers remain assistant claims unless backed by the original tool/measurement artifacts.

Compare B4 with the **15-question B3** results, not the three-question B2 summary. Keep session filtering, group-aware expansion, counting logic, and trigger changes for separately labeled experiments.

## 19. B4 frozen result: search the reply, not only my request

**Evidence:** `buddy_memory_results_B4.zip`, reviewed on 13 November 2026. B4 is frozen as an observed run. Input, annotations, duplicate handling, and the entire grouping/consolidation trace are byte-identical to B3. The added manifest setting is `retrieval_include_assistant=True`; the grouping flag remains `include_assistant_in_embedding=False`. The rest of the configuration is unchanged.

### Plain-language reminder

> I said “Log this trial.” Buddy replied with the recording duration, visibility, and camera speed. B3 searched only what I said. B4 searches what I said **and what Buddy replied**, so it can directly find the measurements when I ask about them later.

The technical change is **reply-aware retrieval indexing**: raw-exchange search vectors include role-attributed user and assistant text. The returned context already contained both sides before B4. The new change affects ranking, not evidence verification or answer generation.

### Exact summary measurements

All figures below come from the supplied 15-question development evaluation. These are mean per-question scores.

| Variant | Evidence recall | Unique-source precision | Complete evidence | MRR |
| --- | ---: | ---: | ---: | ---: |
| A: raw exchanges | 91.11% | 37.78% | 80.00% (12/15) | 0.8444 |
| B: raw + manual memories | 87.78% | 34.11% | 80.00% (12/15) | 0.8556 |
| C: selective manual memories | 91.11% | 37.78% | 80.00% (12/15) | 0.8333 |

| Variant | Mean returned unique sources | Mean context characters | Recorded mean retrieval time |
| --- | ---: | ---: | ---: |
| A | 3.00 | 755.33 | 58.6360 ms |
| B | 3.20 | 972.40 | 4.1788 ms |
| C | 3.00 | 812.93 | 0.1686 ms |

The recorded times include cache/order effects and are not a fair speed comparison between methods. They are retained as observed values only. Character counts are not token counts. No generated answers or live physical outcomes were measured.

### Controlled B3-to-B4 comparison

| Variant | Recall B3 → B4 | Complete evidence B3 → B4 | MRR B3 → B4 | Mean context characters B3 → B4 |
| --- | --- | --- | --- | --- |
| A | 81.11% → 91.11% | 11/15 → 12/15 | 0.7222 → 0.8444 | 826.93 → 755.33 |
| B | 87.78% → 87.78% | 12/15 → 12/15 | 0.8222 → 0.8556 | 1,153.33 → 972.40 |
| C | 81.11% → 91.11% | 11/15 → 12/15 | 0.6889 → 0.8333 | 905.47 → 812.93 |

A and C improve by 10 percentage points in mean evidence recall. Their gain comes from finding the reported-measurements exchange and partially recovering the third-session history. The complete-evidence count increases by only one because the third-session question still lacks one required record. B's recall remains unchanged, while its rankings improve on average. Different questions still regress.

### Case A: the reported measurements are now found

Question: “How long was that follow-me recording, how often could you see me, and how fast was the camera processing?”

The trial report says “59 seconds,” “99 percent,” and “about 2 frames per second” (#20). In B3, A and C miss that exchange. In B4, both rank it first. B already found the manually supplied summary in B3; B4 also ranks the actual raw report first.

Simple meaning: the search can now see the words in Buddy's reply. Technical meaning: indexing reply content improves response-specific evidence retrieval. The figures remain assistant-reported, not independently verified telemetry.

### Case B: reported backward position moves to first place

Question: “After I asked to go backward a bit more, what position did you claim?”

The badly transcribed backward-more request receives Buddy's claim of being approximately 40 cm behind the starting point (#24). That exchange moves from rank 3 to rank 1 in every variant.

The evidence was already retrieved in B3, so recall does not improve for this question. Its ranking improves. This is why recall and ranking must be reported separately.

### Case C: all three backward requests are still not found

The second-session backward requests are “Now can you move backward a bit?” (#23), “Can ive backward a bit more?” (#24), and “just move backward a bit” (#25).

B4 returns the last request (#25), the first request (#23), and an earlier backward-test request from **session 1** (#3). The backward-more request (#24) is missing.

This is still 2/3 recall, the same fraction as B3 but with a different selected pair. All three target records remain correctly grouped together. The failure is retrieval completeness and session scope, not missing storage or a generated wrong count.

### Case D: the third-session question is still incomplete

The third session contains “connect to the rover” (#27) and “start following me” (#28). B4 A and C find the correct connection exchange but miss the following request; recall rises from zero to 1/2. B returns a similar connection request from session 2 instead and remains at zero under the exact-source answer key.

Exact or near-tied connection texts can make source selection sensitive to small ranking differences. The substantive failure is that session identity is not enforced, rather than an inability to recognize connection as a topic.

### Case E: preserve the regressions

| Question | Regression |
| --- | --- |
| Saved follow-me run identifier | A ranks “Buddy, stop following me” above the correct trial report; reference rank falls 1 → 2. C falls 2 → 3 because an unrelated GPS memory and the stop-following exchange precede it. B retains the trial summary first. |
| Scenario request combined with connecting the rover | All methods rank two connection-only exchanges above the correct combined scenario/connection exchange; reference rank falls 1 → 3. |

Indexing replies can add useful information and distracting content simultaneously. The trial-measurements question in B also selects the raw report and a larger summary that repeats that report while adding other sources. This is partial overlap, which B1's fully redundant-item guard intentionally allows.

### B4 decision

Keep B4 as the working retrieval baseline: evidence coverage and average ranking improve on the fixed development set, but not on every question. No fully redundant selected items were found across the 45 question/variant combinations. Grouping and triggers remain B2/B3's outputs. B4 does not establish reliable session selection, event counting, automatic summaries, or hallucination reduction.

Do not overwrite `buddy_memory_results_B4.zip` when creating B5. Keep the modified notebook as well; the ZIP alone does not prove the exact code revision.

## 20. B5: session-aware retrieval before similarity ranking

**Status:** Completed. Section 21 records verified B5 results. The implementation guide below is retained for reproducibility.

### Simple picture

Question: “Show me all three backward requests **from the second session**.”

- B4 searches all exchanges available so far, including session 1.
- B5 first resolves “second session” to its session ID, then keeps only candidate memories whose source records belong to that session.
- The same semantic search and evidence-selection code ranks the remaining candidates.

Technical term: **query-scoped metadata prefiltering**. The candidate set is filtered before similarity ranking. Grouping is unchanged. The target session is resolved from the actual question and chronological session metadata, never from `gold_source_ids`, group labels, or expected answers.

The resolver is intentionally small and explicit. It supports “the/this first session” through “tenth session,” numeric ordinals such as “the 2nd session,” and “this/current session.” It does not support arbitrary dates, “last Tuesday,” “previous session,” cross-session requests, or unrestricted language understanding. Unsupported wording keeps the original broad search; explicit supported session numbers that are unavailable raise an error rather than silently searching another session. “First/second/third” refers to session order in this dataset prefix.

### Step 1: one new configuration flag

```python
CFG["session_aware_retrieval"] = True
```

Keep B4's `retrieval_include_assistant=True`, the semantic encoder, all thresholds, top-three item limit, character budget, questions, labels, and manual memories unchanged.

### Step 2: add this helper before evaluate()

```python
import re

def filter_candidates_by_session(question, prefix, candidates, cfg):
    """Resolve explicit session scope from query text and past metadata only."""
    if not cfg.get("session_aware_retrieval", False):
        return candidates, None

    session_order = list(dict.fromkeys(r["session_id"] for r in prefix))
    lower = question.casefold()
    ordinal = re.search(
        r"\b(?:the|this)\s+"
        r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))\s+session\b",
        lower,
    )
    if ordinal:
        word = ordinal.group(1)
        words = {word: i for i, word in enumerate(
            ["first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth", "ninth", "tenth"], 1
        )}
        number = words[word] if word in words else int(re.match(r"\d+", word).group())
        if number < 1 or number > len(session_order):
            raise ValueError("Requested session is not available at this checkpoint")
        target = session_order[number - 1]
    elif re.search(r"\b(?:this|current)\s+session\b", lower):
        if not prefix:
            raise ValueError("No current session is available")
        target = prefix[-1]["session_id"]
    else:
        return candidates, None

    source_session = {r["id"]: r["session_id"] for r in prefix}
    filtered = []
    for candidate in candidates:
        sources = candidate["source_ids"]
        if not sources:
            raise ValueError("Candidate must cite at least one source")
        if any(source not in source_session for source in sources):
            raise ValueError("Candidate references unavailable source evidence")
        if all(source_session[source] == target for source in sources):
            filtered.append(candidate)
    return filtered, target
```

Manual memories spanning several sessions are excluded from a single-session query; their source content is not silently trimmed. Questions without an explicit recognized session scope keep all chronologically available candidates.

### Step 3: filter candidates inside evaluate()

Find this existing pair of lines:

```python
candidates = candidate_memories(prefix, mode, promoted)
found, chars = retrieve(q["question"], candidates, cfg)
```

Insert the filter between them:

```python
candidates = candidate_memories(prefix, mode, promoted)
candidates, session_scope = filter_candidates_by_session(
    q["question"], prefix, candidates, cfg
)
found, chars = retrieve(q["question"], candidates, cfg)
```

The rest of the loop stays unchanged. `candidate_count` will now describe the filtered set.

In the existing `details.append({...})` dictionary, add:

```python
"session_scope": session_scope,
```

This preserves the resolved session in `retrieval_details.json` for inspection. Keep the source-reference and chronological validation assertions.

### Step 4: run and verify

Run the notebook from the beginning and export `buddy_memory_results_B5.zip`. Confirm that the manifest includes the new flag and that the replay trace is identical to B4. Examine all 15 questions for regressions.

Seven questions in this set have recognized session scope. The helper was checked against their question text and metadata, without using gold evidence to resolve sessions. Its candidate-set behavior includes:

| Question context | Available raw candidates before → after filter |
| --- | --- |
| Three backward requests from session 2 | 23 → 5 |
| Two requests in session 3 | 26 → 2 |
| Combined connection/scenario request in session 4 | 28 → 2 |
| Both stop requests in session 4 | 38 → 12 |

These are candidate counts at the causal checkpoints, not retrieval success results. The first-session start/stop question, second-session hearing check, and second-session reported-position question are also scoped. Questions such as the generic saved-run query have no recognized explicit session reference and retain the original candidate pool.

For the backward-history question, session filtering removes the wrong-session backward request, but the second session still contains connection and following exchanges. Similarity can still omit one backward request. Do not claim that B5 solves complete counting until the result is inspected. For the stop-history question, 12 candidates remain; action relevance is still a challenge.

The third-session query leaves only its two actual exchanges. This is a deliberately controlled test of metadata filtering, not proof that the resolver handles arbitrary conversational time references. The top-three limit is a maximum, so returning two candidates is valid.

B5 is an offline experiment in enforcing session scope. It does not establish trustworthy hardware state, verify assistant claims, generate answers, or connect predicted groups to retrieval. Changes for action matching, counting, group expansion, or dynamic context budgets should remain separate experiments.

## 21. B5 frozen result: search the session I actually asked about

**Evidence:** `buddy_memory_results_B5.zip`, reviewed on 13 November 2026. B5 adds `session_aware_retrieval=True` to B4's configuration. Input records, question annotations, manual memories, duplicate candidates, and grouping/trigger trace are byte-identical to B4.

### Plain-language reminder for my future self

> I asked Buddy to find three backward requests **from the second session**. Previously the search got distracted by a backward request in session 1. B5 first limits the search to session 2. Then the existing semantic search finds all three correct requests.

This is **query-scoped metadata prefiltering**. It changes which candidates may enter ranking, not the semantic model or the group assignments. Scope comes from the question text and chronological session metadata, never from the gold answer IDs.

### Exact summary numbers

All scores are means over the same 15 development questions. There are 45 query/variant combinations.

| Variant | Evidence recall | Unique-source precision | Complete evidence | MRR |
| --- | ---: | ---: | ---: | ---: |
| A: raw exchanges | 96.67% | 50.00% | 93.33% (14/15) | 0.9333 |
| B: raw + manual memories | 96.67% | 48.56% | 93.33% (14/15) | 0.9667 |
| C: selective memories | 96.67% | 50.00% | 93.33% (14/15) | 0.9222 |

| Variant | Mean unique sources returned | Mean context characters | Recorded mean retrieval time |
| --- | ---: | ---: | ---: |
| A | 2.7333 | 701.33 | 53.3324 ms |
| B | 2.9333 | 918.40 | 3.7556 ms |
| C | 2.7333 | 758.93 | 0.1492 ms |

Timing is retained as an observation, not a valid speed comparison because caches and evaluation order differ. Character counts are not tokens. Recall is the fraction of required source records recovered, not generated-answer accuracy. Complete evidence means every required reference record was returned; it does not mean an LLM produced a correct answer. MRR measures only the first relevant returned item.

### B4 → B5

| Variant | Recall | Complete evidence | MRR | Mean context characters |
| --- | --- | --- | --- | --- |
| A | 91.11% → 96.67% | 12/15 → 14/15 | 0.8444 → 0.9333 | 755.33 → 701.33 |
| B | 87.78% → 96.67% | 12/15 → 14/15 | 0.8556 → 0.9667 | 972.40 → 918.40 |
| C | 91.11% → 96.67% | 12/15 → 14/15 | 0.8333 → 0.9222 | 812.93 → 758.93 |

Seven questions resolved an explicit session. Every selected source for those questions belongs to its resolved session. Questions without recognized session scope return the same selected items as B4. No fully redundant selected items were found. Grouping remains at 73.21% pair precision, 83.67% pair recall, and 0.7623 ARI because the replay trace did not change.

### Case 1: all three backward requests now recovered

Question: “Show me all three backward requests from this second session.”

| B4 | B5 |
| --- | --- |
| Last backward request from session 2; first backward request from session 2; earlier backward-test request from session 1 | All three backward requests from session 2 |

B5 returns:

1. “Buddy, at this point, just move backward a bit.” (#25 `5364461d`)
2. “Now can you move backward a bit?” (#23 `f5cd2ce6`)
3. “Can ive backward a bit more?” (#24 `f60db3be`)

The raw candidate set shrinks from 23 to 5 at this checkpoint. Evidence recall rises from 2/3 to 3/3. All methods recover the full reference set. The order shown is relevance ranking, not chronological order.

The three requests were already grouped correctly before B5. The fix is selecting evidence from the correct session, not repairing the group. Recovering three requests on this question is not proof of a general “count every matching event” capability.

### Case 2: third-session history recovered

Question: “In the third session, remind me of the two requests I made, in order.”

Only two raw records remain after filtering: connect to the rover (#27 `102607d9`) and start following (#28 `b771d768`). Both are returned by every method. A and C improve from 1/2 recall; B improves from zero recall. All reach full evidence coverage.

The current metric verifies record access. It does not independently evaluate sequence reconstruction by an answering model. In this run the selected order also matches the interaction order.

### Case 3: combined scenario/connection request moves to first

Question: “In the fourth session, when I asked you to connect the rover, what scenario request did I include in the same message?”

The combined “set the scenario to stop and go” plus “connect the rover” exchange (#30 `270e8fa3`) moves from rank 3 to rank 1. Earlier connection-only records are excluded because they belong to other sessions. The correct record was already present in B4, so this is a ranking improvement rather than a recall gain.

### Case 4: the one remaining incomplete question

Question: “Show me both explicit stop requests in this fourth session: the frustrated stop-the-rover request and my final stop-following request.”

The required records are:

- “What the hell is that? Stop the rover, buddy. Buddy, stop the rover.” (#37 `7a9d5d9c`)
- “Stop following me.” (#40 `6defac73`)

All methods instead select these three items:

| Rank | User request and assistant reply | Similarity | Gold evidence for this question? |
| --- | --- | ---: | --- |
| 1 | User asks to set scenario to stop and go and connect rover; Buddy reports online readiness (#30) | 0.6973 | No |
| 2 | User explicitly asks to stop the rover; Buddy replies “Stopping now” (#37) | 0.6841 | Yes |
| 3 | User asks to set scenario to stop and go; Buddy replies “Okay, stop command sent” (#33) | 0.6041 | No |

The final “Stop following me” record is missing. The correct session has 12 raw candidates, so the scope is right but the top-three relevance selection is incomplete. Recall is 1/2, the only incomplete question in every variant.

**Important correction to the initial diagnosis:** this is not solely an assistant-reply contamination problem. The highest-ranked false match (#30) has no stop-action claim in its reply; the user text itself contains the scenario name “stop and go.” The failure combines action-versus-scenario meaning, a long compound retrieval question, and potentially user-versus-assistant attribution. The export contains only the selected items, so the missing final request's complete rank/score must be inspected before choosing a correction.

### B5 conclusion and scope

B5 resolves session contamination on these explicit-session development queries and improves full evidence coverage to 14/15 per variant. It does not solve arbitrary time references, source truth, automatic consolidation, exhaustive counting, or stop-action semantics. The remaining saved-run ranking weakness also persists for unscoped questions, even though its reference evidence is retrieved.

Freeze B5 before the next change. This result is useful engineering evidence, not broad generalization evidence: the same small dataset has informed every iteration and no held-out evaluation has been run.

## 22. B5.5 setup: inspect the missing stop request before changing another rule

This diagnostic was designated B5.5, separating investigation from the B6 correction. Its results appear in section 23. The agreed sequence is: document B5, diagnose and address the stop-history failure with a general mechanism, rerun all 15 development questions, then stress-test new interactions and wording. Do not add a rule that names a particular record ID or equates any occurrence of “stop” with a command.

### First diagnostic: rank all 12 session-4 raw candidates

The current export shows the top three, not the full list. Compare the original B5 full-exchange score with the same question scored against user text alone. This diagnoses role influence; it is not yet a deployed fix. The same long question is used in both columns.

Paste this after the main evaluation cell:

```python
q = next(q for q in QUESTIONS if q["qid"] == "b3_stop_requests")
position = {r["id"]: i for i, r in enumerate(clean)}
cutoff = position[aliases[q["after_id"]]]
prefix = clean[:cutoff + 1]

candidates = candidate_memories(prefix, "A_raw", set())
candidates, session_scope = filter_candidates_by_session(
    q["question"], prefix, candidates, CFG
)

qvec = encoder.encode([q["question"]])[0]
full_vectors = encoder.encode([c["search_text"] for c in candidates])
user_vectors = encoder.encode([
    by_id[c["source_ids"][0]]["input"] for c in candidates
])
full_scores = full_vectors @ qvec
user_scores = user_vectors @ qvec

# Gold IDs annotate the diagnostic only; they do not affect either score.
gold_sources = {aliases[s] for s in q["gold_source_ids"]}
rows = []
for index, candidate in enumerate(candidates):
    source_id = candidate["source_ids"][0]
    record = by_id[source_id]
    rows.append({
        "source_id": source_id,
        "user_request": record["input"],
        "buddy_reply": record["response"],
        "exchange_score": float(full_scores[index]),
        "user_only_score": float(user_scores[index]),
        "reference_evidence": source_id in gold_sources,
    })

diagnostic = pd.DataFrame(rows)
diagnostic["exchange_rank"] = diagnostic["exchange_score"].rank(
    method="first", ascending=False
).astype(int)
diagnostic["user_only_rank"] = diagnostic["user_only_score"].rank(
    method="first", ascending=False
).astype(int)

print("Resolved session:", session_scope)
with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
    display(diagnostic.sort_values("exchange_rank"))
```

The record identifier is used only to select this failed test for diagnosis; no production ranking rule depends on it. The diagnostic examines all raw candidates so an item missed by the top-three export is visible. It leaves CFG, production retrieval, grouping, questions, and gold labels unchanged.

Interpret the output as follows:

- If the missing stop-following request rises substantially under user-only scoring, request-specific indexing may help, but the B3 raw failure already shows user-only scoring is not automatically sufficient.
- If scenario-setting records still dominate user-only scoring, action-versus-scenario semantics or query decomposition needs attention.
- If the two reference requests compete with different parts of the compound question, a general multi-part retrieval strategy may be appropriate. Any manually supplied query facets must be disclosed as manual inputs, not silently derived from gold sources.
- Do not globally remove assistant replies; that would risk regressing the reported-measurements improvement from B4.

Choose the smallest mechanism supported by the complete ranking, then rerun all 15 questions. A fix passing this known question is a development result. Later stress tests should include actual stop requests, “do not stop” negation, scenario names containing “stop,” discussion of stopping without a command, and contradictory assistant replies. Keep real and synthetic interactions separately identified and freeze code before held-out testing.

## 23. B5.5 completed diagnostic: measure the rank, not a cluster

**Evidence source:** the 12-row CSV-format table pasted by the user after running `buddy_B5_5_rank_diagnostic.py`. Values below are preserved as supplied. No diagnostic ZIP or original plot has been provided for independent file-level validation. This is an observed diagnostic result, not a new production retrieval version.

### The exact question and task

> “Buddy, show me both explicit stop requests in this fourth session: the frustrated stop-the-rover request and my final stop-following request.”

There are 12 raw exchanges after session filtering. For each one, the diagnostic compares this same question with two representations:

- **Full exchange:** user request plus role-attributed Buddy reply.
- **User-only:** the user request with the matching USER prefix, excluding the assistant section.

Scores are cosine similarities from the existing sentence transformer. Rank is the sorted position: 1 is the highest score. The plot shows two ranking positions per exchange; it is not a clustering projection. Gold reference flags are applied after scoring for interpretation, not used to determine ranks.

### Actual table reported by the user

| User request | Full rank | User-only rank | Full score | User-only score | Reference evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Set the scenario to stop and go. Buddy, Buddy, can you connect the rover? | 1 | 1 | 0.6972639560699463 | 0.6733715534210205 | No |
| What the hell is that? Stop the rover, buddy. Buddy, stop the rover. | 2 | 2 | 0.6841177344322205 | 0.6255631446838379 | Yes |
| Set the scenario to stop and go. | 3 | 4 | 0.6040643453598022 | 0.5816056728363037 | No |
| Set the scenario to stop and go. | 4 | 5 | 0.5952551364898682 | 0.5816056728363037 | No |
| Set the scenario to stop and go. | 5 | 6 | 0.5952551364898682 | 0.5816056728363037 | No |
| Stop following me. | 6 | 8 | 0.5332654714584351 | 0.4623620808124542 | Yes |
| Set the scenario to end. | 7 | 7 | 0.47355198860168457 | 0.5121083855628967 | No |
| Buddy, I need you to start following me again. | 8 | 9 | 0.45726630091667175 | 0.4300895035266876 | No |
| Set the scenario to stop and go. | 9 | 3 | 0.45164889097213745 | 0.5816056728363037 | No |
| OK, buddy. Start following me again. | 10 | 11 | 0.4378220736980438 | 0.403400182723999 | No |
| Now can you start following me? | 11 | 10 | 0.43273839354515076 | 0.41588783264160156 | No |
| Whoa. | 12 | 12 | 0.36569899320602417 | 0.3355122208595276 | No |

Repeated identical scenario-request texts are separate recorded events. Their replies can differ, explaining different full-exchange scores. The four identical user-only scenario scores are ties; stable candidate order assigns their different rank positions. Those ordinal differences do not imply different meanings. Source IDs and timestamps are absent from the pasted table, so repeated rows should not be assigned to specific source records solely by their text.

### Plain-language reminder

> I asked for actual instructions to stop. The search gave the highest score to a message about setting a “stop and go” scenario. It placed “Stop the rover” second and “Stop following me” sixth. Because retrieval takes at most three items, it missed the second real stop request.

The desired result is to have both actual stop requests near the top, ideally ranks 1 and 2 in either order. The final stop-following request does not necessarily need to be first; it needs to be selected before irrelevant scenario exchanges occupy the available slots.

These results do **not** mean “ranks 1–5 form a measurement cluster, and rank 6 is an anomaly.” Consecutive rows simply have nearby scores against the same stop-history question. The algorithm did not assign those rows a measurement-group label in this diagnostic.

### Why the two colors matter

- Blue and orange at the same position means the two text representations agree on rank, not that either is correct.
- A relevant REF exchange outside the top three is a problem for this question.
- An irrelevant scenario exchange outside the top three is desirable for this question.
- Color separation shows sensitivity to indexed content. Its direction must be interpreted against evidence relevance.

For “Stop following me,” full-exchange rank is 6 and user-only rank is 8. Removing the reply makes this target rank worse. The combined scenario/connection request stays rank 1 in both views. Therefore, excluding assistant text alone does not fix the observed failure.

### Measured outcome of the diagnostic

| Measure for this failed query | Full exchange | User-only |
| --- | ---: | ---: |
| Rank of frustrated “Stop the rover” request | 2 | 2 |
| Rank of “Stop following me” request | 6 | 8 |
| Actual stop-request sources among first 3 ranks | 1 of 2 | 1 of 2 |
| Reference recall at first 3 ranks | 50% | 50% |
| Complete reference evidence within first 3 ranks | No | No |

These are ranking-position observations. The diagnostic sorts all candidates and does not simulate character-budget selection. The original B5 export independently confirms the production top-three omission for the full-exchange path.

**B5.5 objective achieved:** the complete ranking is now visible and the user-only hypothesis has been tested on this failure. **Original retrieval objective remains unresolved:** both true stop requests are not selected. No production function was changed and no new 15-question pipeline score is claimed for B5.5.

### Diagnosis and limitations

The observed model similarity does not prioritize precise action-request relevance for this compound question. Scenario naming, action intent, and the long multi-part query are plausible contributors. The comparison rules out user-only indexing as a sufficient standalone correction; it does not prove that only one linguistic mechanism caused the ranking.

This is a one-query diagnostic. It is not evidence of general user-only versus full-exchange superiority, and no significance test is needed to inspect these ranks. The broader difficulty is established in information retrieval: a bi-encoder can retrieve related but irrelevant content. [Sentence Transformers' retrieve-and-rerank documentation](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) describes adding a relevance scorer for query/candidate pairs. The portfolio contribution here is the source-grounded failure analysis, controlled experiments, and later validation, not a claim of discovering a new retrieval problem.

## 24. B6 design: relevance reranking before final selection

**Status:** Experiment completed; this section preserves the original design and hypothesis. Actual measurements and the decision are in section 25.

### What are we trying to fix?

Simple version:

> The search recognizes related rover/stop/scenario language. We need a second check of whether each exchange actually helps answer **this particular question**.

Technical version: improve query-specific relevance ordering beyond independently encoded bi-encoder cosine similarity, while retaining B5's causal boundaries, explicit session filtering, evidence attribution, redundancy guard, and final context budget.

### Proposed mechanism: a cross-encoder reranker

Keep the sentence transformer and add a pretrained cross-encoder relevance model. The current bi-encoder encodes the question and a candidate separately, then compares their vectors. A cross-encoder processes the question and candidate text together to predict a relevance score. It is a scoring model, not the Qwen/Llama generative answering stage planned later.

| Stage | Role in B6 |
| --- | --- |
| Chronological cutoff | Prevent future evidence from entering |
| Session metadata filter | Keep the session requested by the user |
| Existing semantic scores | Preserve the baseline scores for comparison and eligibility |
| Cross-encoder pair scoring | Score the question jointly with each candidate's attributed search text |
| Final selection | Sort by reranker score; apply the existing duplicate-source and character-budget rules; return at most three items |

**Do not rerank only the three items B5 already selected.** The missing stop-following exchange is rank 6, so that would exclude it before the new model has a chance to examine it. For this small pilot, rerank all eligible scoped candidates: all 12 raw exchanges for the failed question. Larger-scale shortlisting can be evaluated separately later.

Use the same query wording and role-attributed candidate text as B5. Keep manual memories eligible under the same A/B/C rules, and score them as candidates when available. Do not use gold IDs, true/false reference labels, or a hand-written “stop” boost in the model input or ranking rule. Both reference requests being in the reranked top three is a hypothesis to test, not a promised result.

The baseline minimum cosine threshold applies only to its original similarity scale. Keep it as the pre-reranking eligibility rule for a controlled first experiment; do not reuse 0.01 as a threshold on cross-encoder scores. Those scores can have a different range and are not automatically calibrated probabilities. Use the new scores to order eligible items initially.

### Concrete before/after target, not a predicted result

| Exchange | B5.5 observed ordering | B6 desired behavior |
| --- | --- | --- |
| Scenario configuration and connection request | Rank 1 | Lower than actual stop instructions for this question |
| “Stop the rover” request | Rank 2 | Retained near the top |
| “Stop following me” request | Rank 6 | Promoted into the returned evidence set |

“Topic similarity” and “usefulness for this question” can diverge. Reranking is intended to improve that distinction, but a generic relevance model may still fail on action/scenario ambiguity or negation. If it fails, preserve the result before considering structured action extraction or multi-part query retrieval as another experiment.

### Controlled evaluation and decision rule

1. Fix the reranker checkpoint before checking its performance; record its name, resolved revision, package versions, device, and precision.
2. First inspect the 12-candidate failed query: preserve cosine score/rank and reranker score/rank, along with source IDs and actual request descriptions.
3. Run the unchanged 15-question development set across A/B/C. Grouping and triggers should remain identical to B5.
4. Compare complete evidence, per-question recall, first-reference rank, source precision/redundancy, context characters, candidate counts, and added scoring time. Inspect the reported-measurements and saved-run questions for regressions.
5. Report the original development score even if a reranker does not help. Do not silently switch among checkpoints until this one failure passes.
6. After accepting or rejecting the correction, freeze it and run separately labeled stress tests and held-out real sessions. Include negation, scenario names containing action words, several requests in one exchange, and replies contradicting the user's request.

Success for the known failure means both reference stop-request records are selected without a larger final item budget. Success for the broader experiment also requires acceptable behavior on the other questions and a justified latency/context tradeoff. A perfect development score is not sufficient evidence of robustness or a requirement to continue the project.

## 25. B6 measured result: optional reranking changed order but not completeness

**Sources reviewed:** the user-supplied B6 `summary.csv` and `retrieval_details.json`. The latter contains 90 final-repetition detail entries: 15 questions × 3 memory variants × OFF/ON. The full B6 benchmark ZIP, hardware manifest, replay trace, and paired-delta CSV were not supplied during this review. Do not infer whether these timings came from Colab CPU, A100, or the intended RTX 3070 deployment. The planned script specifies the `cross-encoder/ms-marco-MiniLM-L6-v2` checkpoint, but its actual loaded revision and device require the run manifest.

### Purpose of ranking: reminder for my future self

> When I ask a question, Buddy cannot simply hand every old conversation to the answering model. Ranking decides which available memories should enter the limited context. If irrelevant scenario records take those slots, the LLM may lack the evidence it needs even though the correct conversation is stored.

B5 already ranks memories by semantic cosine similarity. Turning the cross-encoder OFF does **not** mean turning ranking OFF. The B6 experiment asks whether a second relevance scorer improves the selected evidence enough to justify extra processing.

| Term | Meaning in this experiment |
| --- | --- |
| Grouping | Organize related historical exchanges into groups |
| Semantic ranking | Sort available candidate memories by query–memory vector similarity |
| Cross-encoder reranking | Reorder those candidates using a model that scores query and candidate jointly |
| Final selection | Apply source redundancy and character-budget rules; return at most three items |
| Answer generation | Later Qwen/Llama stage, not measured in B6 |

The reranker is inside retrieval, not a separate preprocessing system feeding a second B5 search. The actual flow is: chronological cutoff → session filter → cosine eligibility/scoring → optional cross-encoder ordering → final selection → context for a future answering model. When the reranker is ON, its order replaces cosine order at the selection stage. No later B5 process automatically repairs missing evidence.

An irrelevant scenario-setting record ranking below three is generally desirable for the stop-history question. A reference stop request ranking below three is a failure. Blue/orange plot overlap indicates agreement, not correctness; rank always depends on the question.

### Main measured quality results

| Variant | Recall OFF → ON | Complete evidence OFF → ON | MRR OFF → ON | Mean context characters OFF → ON |
| --- | --- | --- | --- | --- |
| A: raw exchanges | 96.67% → 96.67% | 93.33% → 93.33% | 0.9333 → 0.9556 | 701.33 → 696.27 |
| B: manual memories | 96.67% → 96.67% | 93.33% → 93.33% | 0.9667 → 0.9556 | 918.40 → 893.53 |
| C: selective memories | 96.67% → 96.67% | 93.33% → 93.33% | 0.9222 → 0.9556 | 758.93 → 725.53 |

In the supplied final-repetition details, exactly the same 14 questions have complete evidence under both conditions in each variant. The stop-history question remains the single incomplete case. This was checked per question, rather than inferred from equal aggregate averages.

MRR changes because it rewards placing the **first** reference item earlier. It does not require all reference items to be selected. The scores are retrieval-quality measures, not a comparison of Qwen/Llama answers or hallucination rates.

### Measured latency

| Variant | Median total ms OFF → ON | Difference of median totals, ms | Median reranker-stage ms ON | P90 total ms OFF → ON |
| --- | --- | ---: | ---: | --- |
| A_raw | 6.9628 → 16.7396 | 9.7768 | 9.7407 | 7.3416 → 26.4827 |
| B_manual | 6.9234 → 16.6875 | 9.7640 | 9.6759 | 7.2339 → 31.2258 |
| C_selective | 6.9432 → 16.8534 | 9.9102 | 9.8606 | 7.2104 → 26.1909 |

The approximate 9.8–9.9 ms overhead above is a **difference between condition medians**. It is not the median of matched per-request deltas; `paired_deltas.csv` or `delta_summary.csv` is required for that statistic. The script is designed to measure warm-document/cold-query retrieval latency, exclude model loading/offline grouping/output, synchronize CUDA, alternate ON/OFF order, and avoid caching reranker predictions. Only the supplied summary/detail outputs were reviewed, so runtime execution settings and repetition count still need confirmation from the manifest.

Total retrieval latency is not complete Buddy voice latency. It excludes speech recognition, a later answering LLM, speech synthesis, and hardware tool execution. These measurements cannot predict local RTX 3070 latency without a local rerun. Timing repetitions are not independent new accuracy examples.

### Actual stop-history outcome

Question: “Show me both explicit stop requests in this fourth session…”

| Exchange | OFF rank | ON rank | ON relevance logit | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Explicit frustrated “Stop the rover, buddy” request | 2 | 1 | 1.522369 | Correct evidence improves |
| “Set the scenario to stop and go” plus connection request | 1 | 2 | 0.456825 | Irrelevant scenario content remains selected |
| “Start following me again” request | 8 | 3 | -0.823358 | Opposite-action content enters selection |
| Another “Start following me again” request | 10 | 4 | -0.972654 | Still above the missing stop request |
| “Stop following me” request | 6 | 5 | -1.074727 | Correct evidence remains outside top three |

These rankings are the same across the three variants for this query. Every one of the 12 session-4 raw candidates is present in the scored ranking, including the missing stop-following request. Its omission is not a missing-candidate or character-budget problem. Reranking produces a different order, but the selected context still contains only one of the two required requests: 50% recall for this case.

Negative cross-encoder logits are valid relative relevance scores. They are not cosine values or probabilities, and the implementation does not apply the cosine threshold to them.

### Other observed gains and regressions

- The saved-run-identifier exchange moves to rank 1 in raw and selective retrieval (from rank 2 and rank 3 respectively).
- For the question asking what Buddy replied to the scenario request, the correct exchange moves from rank 1 to rank 3 in every method.
- The first correct stop request moves from rank 2 to rank 1, improving MRR without improving completeness.
- Context size changes slightly, but no generated-answer quality benefit has been measured.

### Simple interpretation and decision

> Both versions find the same amount of required evidence on this development set. The extra model moves some records into better positions and some into worse ones. It still misses “Stop following me” from the selected stop-history evidence, while taking about 10 ms more in the supplied timing summary.

Retain B5 as the working retrieval baseline with `CFG["use_cross_encoder"] = False`. Keep the switch and benchmark for future experiments. This is a measured rejection of the **current configuration for the current objective**, not a claim that all rerankers are useless or that ranking itself is unnecessary.

Better evidence ordering could affect small-model answers even when recall is unchanged, but that requires an actual generation comparison. More history and harder distractors could change the tradeoff; the present development sample does not settle that question. Do not add a hand-written “stop” rule merely to reach 15/15 before measuring downstream behavior.

### Original supplied summary, preserved without rounding

```csv
mode,use_cross_encoder,median_total_ms,p90_total_ms,median_rerank_ms,mean_recall,complete_fraction,mean_MRR,mean_context_chars,mean_reranked_count
A_raw,False,6.962794999935795,7.341602200085617,0.0,0.9666666666666667,0.9333333333333333,0.9333333333333333,701.3333333333334,0.0
A_raw,True,16.739643999926557,26.482693799971457,9.740677999843683,0.9666666666666667,0.9333333333333333,0.9555555555555556,696.2666666666667,11.866666666666667
B_manual,False,6.9234329998835165,7.233850200054803,0.0,0.9666666666666667,0.9333333333333333,0.9666666666666667,918.4,0.0
B_manual,True,16.68746099994678,31.22576040004788,9.67586100000517,0.9666666666666667,0.9333333333333333,0.9555555555555556,893.5333333333333,13.2
C_selective,False,6.943192000107956,7.2104253999441426,0.0,0.9666666666666667,0.9333333333333333,0.9222222222222223,758.9333333333333,0.0
C_selective,True,16.85342100017806,26.190917800022362,9.860566000043036,0.9666666666666667,0.9333333333333333,0.9555555555555556,725.5333333333333,12.333333333333334
```

## 26. Original B7 plan: controlled local Qwen/Llama answering pilot

**Historical plan:** the completed pilot is documented in section 27; the final reranking comparison is in section 28. Readiness statements below describe the decision before those runs.

**Readiness decision:** ready to begin an isolated answering-model pilot. Not yet a validated live Buddy integration. The existing retrieval pipeline is sufficiently understood to measure whether an answering model uses its context correctly; it need not be perfect before this test starts.

### What must be frozen before the first comparison?

1. Choose a named retrieval condition, initially B5 raw-exchange retrieval with the cross-encoder OFF. Retain manual-memory variants as separate conditions rather than silently combining their best outputs.
2. Export identical model-facing questions and context packets, including source IDs, timestamps, role attribution, ordering, and any truncation. Feed exactly the same packet to Qwen and Llama.
3. Keep gold answers, `gold_source_ids`, relevance labels, metric results, and reference flags outside model input. They belong only to the evaluation harness. Existing retrieval detail JSON contains these fields, so it must not be passed wholesale to the answering model.
4. Define a small answer rubric before judging outputs: requested facts/values, evidence support, attribution, source references, completeness, and calibrated uncertainty. Score these separately rather than only by fluent wording.
5. Record exact Qwen/Llama model tags, weight revisions when available, parameter sizes, quantization, runtime versions, context limits, decoding settings, and seed where supported. Qwen is a family, not a sufficiently precise model identifier.

Use a shared task instruction appropriate to both models: answer from supplied evidence, distinguish user requests from assistant claims and verified tool results, cite source IDs, and state when the evidence cannot establish a fact. Model-specific chat templates may differ; preserve the same actual instruction and context content. Context character budgets are not token budgets: measure tokens for each model and prevent silent truncation.

The test harness asks models to answer historical questions. It does not execute commands from recorded user utterances and does not require live rover integration. This keeps memory/answer behavior measurable while development continues.

### Simple example of the test

Context contains Buddy saying: “The recording was 59 seconds; I could see you 99 percent of the time; processing was about 2 fps.”

Question: “How long was the recording, and what was the camera speed?”

A supported answer says: “Buddy reported 59 seconds and about 2 fps,” with its source. An answer claiming independently verified measurements overstates the evidence. An answer inventing a different number fails exact-fact accuracy.

### Separate retrieval failure from answering failure

Use two clearly labeled context conditions on a small diagnostic subset:

| Condition | Context supplied | What it helps diagnose |
| --- | --- | --- |
| Actual retrieval | The exact frozen B5 selected context | End-to-end retrieval-plus-answer behavior |
| Reference-evidence context | Manually selected raw evidence records sufficient for the question, excluding written reference answers | Whether an answering model can use adequate evidence when retrieval succeeds |

Reference-evidence context is an idealized diagnostic control, not production retrieval. Gold IDs may be used by the harness to construct that control, but the model sees ordinary source excerpts, not gold flags or the expected answer text.

For the stop-history question, actual B5 context misses the final stop-following request. The model cannot reconstruct an absent conversation reliably. The reference-evidence condition supplies both actual stop requests and tests whether it can distinguish them from scenario naming. This isolates answer use from memory access instead of blaming every bad answer on the LLM.

### Scoring and insufficient-evidence cases

Score final answers against what is established in the **supplied** context as well as the full-log reference. Keep end-to-end task success and faithfulness-to-context as separate measures: a cautious answer can be faithful but incomplete because retrieval omitted evidence. A confident correct-looking answer without source support is not evidence of reliable memory use.

Before making robustness claims, add a small versioned set with unavailable facts, ambiguous sessions, assistant-only unsupported diagnoses, and partial histories. For an exhaustive “all/how many” question, the harness should identify whether the retrieval packet is complete, partial, or unknown; do not teach the model that a top-three packet is automatically exhaustive. A metadata field should only claim completeness when the pipeline actually establishes it. The original no-LLM evaluator assumes nonempty reference evidence, so negative cases need their own answer/abstention labels instead of empty gold lists plugged into the old recall function.

Basic answer-level measures: exact fact/value correctness, requested-part coverage, unsupported claims, attribution errors, correct uncertainty/abstention, and source-citation support. Manual scoring with a documented rubric is sufficient for the first small pilot; an LLM judge is optional rather than required.

### Latency and local validation

Measure retrieval latency, model loading separately, answer generation latency, token counts/output length, and retrieval-plus-generation time. Record the actual local device configuration. If streaming is used, add time to first token separately from full completion time. Neither full voice latency nor automatic summarization latency is established by this pilot.

Start with the 15 development questions and disclose their role; do not present them as held-out tests. Add a later independent session and freeze settings before scoring it for a portfolio validation result. A larger/stress-tested corpus remains needed before asserting scaling or deploying a real-time memory change, but it is not a prerequisite to begin learning from local answering-model experiments now.

### Later stages remain separate

Testing Qwen/Llama as answering models does not validate using either model to generate episodes or extract facts automatically. The current manual memories remain idealized inputs. Automatic consolidation, action interpretation, group-aware retrieval, and live robot integration need their own measured comparisons.

## 27. B7 completed: local Llama versus Qwen answering pilot

**Status:** Completed offline pilot, reviewed 13 November 2026. This extends the retrieval experiments into actual generated answers. Earlier sections describe their historical no-generative-LLM scope; B7 adds answering, without executing rover commands or modifying live Buddy.

### Purpose in simple words

> We already measured whether the right conversations reached the model. Now we ask: when Llama or Qwen receives those conversations, does it actually answer my question correctly?

A retrieval score of 96.67% is not 96.67% answer accuracy. A model can receive the right conversation and still read the wrong part, confuse a request with a reply, or omit part of the answer.

### Frozen design and model identity

| Item | Actual pilot setting |
| --- | --- |
| Runtime | Local Ollama 0.5.4 on the user's Windows machine |
| Llama | `llama3.2:latest`, 3.2B parameters, Q4_K_M; installed ID prefix `a80c4f17acd5` |
| Qwen | `qwen2.5:3b`, 3.1B parameters, Q4_K_M; installed ID prefix `357c53fb659c` |
| Declared deployment GPU | RTX 3070, 8 GB; not a controlled CPU/GPU hardware comparison |
| Runtime context / maximum output | `num_ctx=4096`, `num_predict=256` |
| Decoding | Temperature 0, seed 2026, top_k 40, top_p 0.9, repeat_penalty 1.0 |
| Question set | Same 15 development questions; no held-out questions |
| Conditions per model | Frozen B5 raw retrieved context; reference-evidence context |
| Total | 2 models × 2 conditions × 15 questions = 60 responses |
| Request shape | Fresh system/user messages; streamed `/api/chat`, JSON output requested |
| Warmup / residency request | One warmup per model; keep_alive 10 minutes |

The larger context lengths reported by `ollama show` are model capabilities, not the context actually configured in this pilot. The manifests retain detailed model/runtime information. Llama is Buddy's current answering model; Qwen is being evaluated as an alternative here. This experiment does not train either model or evaluate a LoRA training procedure.

The same actual task instruction asks for answers supported by excerpts, source citations, separation of user requests and assistant claims, and explicit uncertainty when evidence is missing. Native model chat templates can differ. Expected output fields are `answer`, `source_ids`, `insufficient_evidence` (boolean), and `limitation` (string).

| Condition | What enters the model | Interpretation |
| --- | --- | --- |
| Retrieved | Exact B5 raw selected context, cross-encoder OFF | Current retrieval-plus-answer behavior |
| Reference | Raw excerpts manually identified as required evidence | Diagnostic control for adequate evidence; not a deployable retrieval method |

Gold answer prose and relevance flags are excluded from model input. Reference IDs select excerpts in the harness only. All 30 matching question/condition input hashes agree across models. No request failed or ended through the output length limit; maximum reported prompt length was 769 tokens and maximum output length was 120 tokens. This does not indicate truncation in these requests.

**Evidence archive:** `B7_results_20261113_213511_296400.zip`. It contains `answers.jsonl`, `automatic_summary.csv`, `manual_review.csv`, `manifest.json`, and `prompt_packets.json`. The supplied runner package is `Buddy_B7_Local_LLM.zip` (also supplied as `Buddy_B7_Local_LLM(1).zip`). Preserve the actual runner and manifest with results.

### Structural output results: do not confuse these with correctness

| Check | Llama | Qwen |
| --- | ---: | ---: |
| Responses | 30 | 30 |
| Failed requests | 0 | 0 |
| Syntactically parseable JSON | 30/30 | 30/30 |
| Required JSON schema satisfied | 30/30 | 2/30 |

Qwen often returns an empty string for `insufficient_evidence`, where a boolean is required, and sometimes omits `limitation`. This is an output-contract failure, independent of whether individual facts are correct.

**Automatic-checker limitation:** the original checker marks `source_ids_valid=False` when schema validation fails before independently checking citations. Therefore the automatic summary's Qwen source-ID validity fraction of 6.67% must not be interpreted as 93.33% invented source IDs. Inspection found that returned source-ID lists contained only IDs available in their packets; one Qwen answer had an empty list. Availability still does not prove that a citation supports the claim. No corrected global answer-accuracy percentage is claimed here; the manual review sheet is not a completed scored benchmark.

### Actual answer failures, with context

| Question / intended answer | Llama observation | Qwen observation | Diagnosis |
| --- | --- | --- | --- |
| List the three backward requests in session 2 | Both conditions answer “20 centimeters, 20 centimeters, 20 centimeters” | Retrieved condition denies backward requests; reference condition says only one is present | All three requests are supplied. This is evidence-use / instruction-following failure, not simply missing retrieval |
| Show both explicit stop requests in session 4 | Retrieved context substitutes scenario setup for the final stop-following request; reference context correctly lists both real stop requests | Retrieved answer also substitutes scenario setup; reference answer quotes both real requests but then contradicts itself by denying an explicit rover stop | Missing retrieved evidence matters; adequate evidence helps Llama on this case but does not guarantee Qwen consistency |
| Position after “backward a bit more” | Reports approximately 40 cm, omitting the full behind-start qualifier | Retrieved answer gives approximately 20 cm; reference gives 40 cm behind the starting point | Qwen confuses incremental movement with cumulative reported position in the retrieved packet |
| Corner scenario and trial count | Correctly reports corner and 0 of 3 | Reference answer claims trial count is absent, although the source says 0 of 3 | More focused reference context is not automatically sufficient |
| Scenario named “end” and trial count | Retrieved answer gives end and 0 of 3; reference answer only gives 0 of 3 | Core end/0-of-3 content appears | Llama can omit a requested part even with the exact source |
| Scenario request included in the same message as connection | Correct scenario text, but cites a separate scenario-only exchange | Requires separate citation-content review | Correct answer text alone can conceal the wrong event citation |

Both models recover several straightforward facts, including the GPS claim, fragile-design statement, and reported saved-run identifier. These are observations about what was said, not independent confirmation that GPS failed or a run file was created. Attribution and qualification still need review, even when the central value is correct.

### Latency: preserve the measurements and their limits

All values below are medians over 15 responses, in milliseconds. Column medians describe different distributions and must not be added or subtracted as though they were one representative request.

| Model / condition | Client completion | First content | Model loading | Prompt evaluation | Generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama / retrieved | 4424.29 | 4138.99 | 3593.03 | 235 | 324 |
| Llama / reference | 4519.80 | 4162.75 | 3666.06 | 198 | 329 |
| Qwen / retrieved | 4743.82 | 4453.80 | 3954.88 | 275 | 476 |
| Qwen / reference | 4806.41 | 4061.23 | 3633.34 | 214 | 524 |

Reported load duration exceeds one second in all 30 Llama requests and 28 of 30 Qwen requests, despite warmup/residency requests. The cause is not established by this export. These are not clean steady-state generation benchmarks, and they do not prove that the RTX 3070 is inadequate. First content can be JSON punctuation rather than a useful spoken answer. Retrieval is frozen outside these calls; speech recognition, speech synthesis, live tool execution, and full voice latency are not measured.

### B7 decision

Llama obeys this JSON contract more consistently and succeeds on the reference stop-request case where Qwen contradicts itself. Keep Llama as the working model for the bounded follow-up; this is not a general ranking of the model families. Neither model is validated as reliably interpreting all supplied history.

The pilot identifies two separate bottlenecks: **evidence selection** and **evidence use**. Improving one does not automatically repair the other. There is no controlled original-live-Buddy comparison, so these results do not establish an overall hallucination reduction or production improvement.

## 28. Final answer comparison: does cross-encoder reranking help Llama?

**Status:** Completed. Decision: keep the optional reranker OFF for the present working baseline; close this reranking branch on the available evidence.

### Why run this after B6?

> B6 showed that reranking did not recover more required conversations. But perhaps arranging the context differently could still help a small answering model read it correctly. This final comparison tests that possibility directly.

B5 already uses semantic ranking. OFF means the additional cross-encoder is disabled; it does not mean unranked retrieval. ON uses the B6 cross-encoder-selected context. Session filtering happens before candidate ranking and final selection, rather than after an independent reranking pipeline.

### Controlled setup and artifacts

- Model: `llama3.2:latest`, same answering instruction and decoding settings as B7.
- Inputs: actual raw-exchange OFF/ON contexts exported from B6; same 15 questions, top-three item maximum, 2400-character maximum context, and source attribution.
- Model sees ordinary excerpts, never reference answers or relevance labels.
- One response per question per condition: 15 OFF + 15 ON = 30 responses.
- Execution order: an OFF block followed by an ON block. This is not a randomized/repeated latency study.
- Cross-encoder inference is not executed during this local answering comparison: contexts were already prepared.

Runner package: `Llama_Reranking_Comparison.zip`, using `run_comparison.py`, `ollama_eval_core.py`, `cases_off.json`, and `cases_on.json`.

Results: `Llama_reranking_results_20261113_220113_314700.zip`, containing `paired_answer_review.csv`, `comparison_manifest.json`, `READ_RESULTS_FIRST.md`, and separate `off/` and `on/` copies of answers, summaries, manifests, review sheets, and prompt packets. The paired review includes reference answers and actual evidence for manual inspection. Its human scoring columns are templates, not completed scores.

### What actually changed?

Nine of 15 answer texts are exactly identical. The other six differ in wording or content; a text difference is not automatically a quality gain.

| Question | OFF answer | ON answer | Interpretation |
| --- | --- | --- | --- |
| Show all three backward requests | “20 centimeters, 20 centimeters, 20 centimeters” | Same | Failure persists even though all three request records enter both packets |
| Show both actual stop requests in session 4 | Real stop-rover request plus “Set the scenario to stop and go” | Real stop-rover request plus “Buddy, I need you to start following me again” | Both incorrect; final “Stop following me” remains absent |
| What did Buddy reply to the targeted latest stop-and-go scenario request? | “The assistant replied that the stop command was sent” | “The traffic light will now display red, orange, and green lights in sequence for your reference” | Regression against the frozen target: ON selects an earlier exchange's reply |
| Show starting and stopping following in session 1 | General summary of start then stop | More explicit start/stop wording | Presentation change; not a demonstrated rescue of a failed task |
| Recording duration, visibility, camera rate | 59 seconds, 99 percent, 2 fps | Same central figures with wording changes | No material demonstrated gain |
| Scenario named end and trial count | End, 0 of 3 trials logged | Same with a tense change | No material gain |
| GPS claim, fragile design, saved run ID, hearing check, reported position, third-session order, combined scenario request, corner/trial count | Core answers retained | Same text | No answer-text gain; detailed attribution/citation limitations still apply |

The scenario-reply regression needs precise interpretation: the ON traffic-light text really exists in an earlier supplied source. It is **wrong-event selection / temporal grounding**, rather than text fabricated from nothing. The question uses “just now” and repeated scenario wording, so explicit event anchoring would make a stronger future evaluation. Against the frozen reference and timestamps, OFF answers the intended later exchange while ON answers an earlier one. Both packets contain the target “Okay, stop command sent” record; ON places other exchanges ahead of it.

The combined scenario/connection answer is identical in both conditions but cites a separate scenario-only record (`8d44fbd4`) instead of the combined request (`270e8fa3`). This illustrates why exact answer text and valid source-ID membership are insufficient measures of event-level grounding.

### Two failures that must not be merged

**Missing evidence:** the final “Stop following me” record (`6defac73`) is absent from both selected stop-history packets. OFF substitutes scenario setup; ON substitutes restarting following. Both confidently claim adequate evidence. Neither correctly acknowledges that the requested second stop event is missing.

**Misreading available evidence:** all three session-2 backward-request records (`f5cd2ce6`, `f60db3be`, `5364461d`) are present in both packets. Llama reads distances from the assistant replies instead of listing the user's utterances. Reranking changes their order but not this behavior.

> Folder analogy: sometimes Buddy receives an incomplete folder. Sometimes it receives the complete folder and reads the wrong lines. Rearranging the folder did not fix either problem in this trial.

### Structural results and timings

| Measure | OFF | ON |
| --- | ---: | ---: |
| Responses / failed requests | 15 / 0 | 15 / 0 |
| Schema-valid responses | 15/15 | 15/15 |
| Source-ID availability check | 15/15 | 15/15 |
| Median client completion | 4298.73 ms | 4384.14 ms |
| Median first content | 3980.47 ms | 3894.11 ms |
| Median reported load | 3458.15 ms | 3339.52 ms |
| Median generation | 333 ms | 349 ms |

Source-ID availability checks membership, not whether citations entail the answer. Both bad stop answers pass these structural checks. The difference between completion medians is +85.42 ms, but this is **not measured reranker overhead**: reranking ran earlier, outside the local model calls. Loading variability and fixed block order prevent attributing this difference to the retrieval method. B6's separate retrieval timings remain in section 25; they must not be presented as measured local end-to-end cost by adding them to these medians.

### Final decision and its scope

No clear rescue of the two main failure cases was observed, and one answer regressed against the reference. More explicit wording on another answer does not justify an extra inference stage. Keep cross-encoder OFF as the current default and retain the switch/code for reproducibility. This rejects this checkpoint/configuration on this development set; it does not establish that reranking never helps larger corpora or other models.

Do not report a new overall answer-accuracy percentage from the structural checks or unchanged-text count. A fully scored, explicit answer rubric is needed for that claim. The present conclusion comes from paired answer/evidence inspection and concrete failure cases.

## 29. End-to-end framework and current handoff

### Actual architecture, from logs to an answer

1. **Read and validate JSONL:** preserve original text, replies, timestamps, sessions, and source IDs; sort causally.
2. **Handle likely duplicate logging:** exclude the two documented duplicate events from replay while preserving original input.
3. **Organize and inspect recurrence:** semantic user-text embeddings support grouping and recurrence triggers; manual group labels evaluate organization only.
4. **Construct available retrieval candidates:** raw exchanges, with optional manual memories under the A/B/C experimental conditions. Manual summaries are authored inputs, not an evaluated automatic summarizer.
5. **Apply the question checkpoint and session scope:** exclude future evidence; use supported query/session metadata rather than gold labels.
6. **Rank eligible candidates:** semantic cosine scoring; optional cross-encoder tested and currently OFF.
7. **Select context:** avoid fully redundant source sets; retain at most three items under 2400 characters, including attributed evidence.
8. **Generate a historical answer:** frozen context plus question and shared instruction enters local Llama; Qwen is a separately measured comparator.
9. **Evaluate separately:** grouping agreement, retrieved-source coverage, answer correctness/completeness, attribution, citation support, output contract, and latency.

**Important boundary:** grouping is an evaluated side path, not an input to the current retrieval ranking. Automatic LLM episode generation/fact extraction, verified robot-state updates, live control, and a complete RecMem/EverMemOS reproduction are not implemented by this experiment.

### What is established, and what remains open?

| Area | Evidence established here | Remaining limit |
| --- | --- | --- |
| Semantic grouping | Pair recall improved from 51.02% in lexical B0 to 83.67% in B2; ARI 0.5852 → 0.7623 | Small manually defined development labels; activity/action overlap |
| Retrieval engineering | Duplicate-source guard, reply indexing, and session filtering were separately tested; B5 complete reference evidence 14/15 | Top-three selection still misses the final stop request |
| Generated answers | Two local models tested on identical retrieved/reference packets | Complete context does not ensure correct answer use |
| Reranking | Retrieval and final-answer effects both inspected | No convincing benefit for the current default; OFF retained |
| Timing | Loading, prompt evaluation, generation, first content, and client time recorded | No clean steady-state local retrieval-plus-generation or full voice benchmark |
| Real-world Buddy improvement | Concrete failure mechanisms and a reviewable evaluation harness | No controlled comparison against original live Buddy, held-out validation, or live deployment |

### Bounded next work, if development resumes

The reranking branch is closed; no further checkpoint search is required to complete this experiment. Preserve the current artifacts and conclusions first. The next isolated problem should be chosen explicitly:

- **Evidence use:** test request-versus-reply extraction on the backward-history case, holding its already-complete context fixed.
- **Evidence access:** test a general method for retrieving both requested stop events, with scenario names and restart commands as distractors. Do not hard-code the known IDs or correct answers.

Change one mechanism at a time, rerun the existing questions for regressions, and then freeze it before a separately labeled held-out session/stress set. These are future proposals, not completed fixes. Expanding the corpus or adding another model is not evidence by itself that either bottleneck has been solved.

### Portfolio wording supported by the record

> Built an offline, source-traceable evaluation pipeline for a rover assistant using 40 real interaction records across four sessions. Compared lexical and semantic organization, reply-aware and session-scoped retrieval, optional cross-encoder reranking, and two local quantized answering models. Session filtering increased complete evidence coverage from 12/15 to 14/15 development questions for raw retrieval. Paired answer analysis separated missing-evidence failures from failures to interpret available evidence, and rejected reranking as the default when it provided no convincing task benefit.

This is a measured engineering investigation with documented negative results. It is not evidence of novel memory research, production reliability, autonomous robot safety, or a guaranteed reduction in hallucinations. The main deliverable is a reproducible framework that makes those future claims testable.


## Appendix A. Full interaction inventory

The text below preserves the recorded input, including transcription artifacts. `Gold` is the user-reviewed primary activity; `Predicted` is the baseline group. Original Buddy intents are recorded routing outputs, not validated intent labels. Excluded duplicates inherit their retained record's group for display only and are not scored.

| # | Session | Record ID | Recorded input | Buddy intent | Gold | Predicted | Trigger |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | 080507 | `2f5ff21d` | Set the scenario to exit. | chat | Unlabeled | g000 | No |
| 2 | 080507 | `f3bdd747` | Buddy, can you connect to the rover? | device_control | s1_rover_connection | g001 | No |
| 3 | 080507 | `193c6318` | Now let's have a bit of test. Can you move backward a bit? | device_control | s1_basic_movement | g002 | No |
| 4 | 080507 | `69bd62a8` | Buddy, I need you to move forward a bit. I want to see the problem. | device_control | s1_basic_movement | g003 | No |
| 5 | 080507 | `f57ca9d6` | Yes, buddy. I just need you to move forward a bit. | device_control | s1_basic_movement | g003 | No |
| 6 | 080507 | `7074068a` | Yes, buddy. I just need you to move forward a bit. | unknown | Excluded duplicate | g003 | Excluded duplicate |
| 7 | 080507 | `59d581b1` | Buddy, can you move forward a bit? | device_control | s1_basic_movement | g003 | No |
| 8 | 080507 | `56162f0d` | First of all, my design is very very fragile. | device_control | s1_rover_discussion | g004 | No |
| 9 | 080507 | `82434da1` | Oh my god. How about nobody can move forward a bit? Buddy, I need you to move forward a bit. | device_control | s1_basic_movement | g003 | Yes |
| 10 | 080507 | `947bc324` | OK, it fixed. Now move back. Now I need you to move forward and then turn around. | device_control | s1_basic_movement | g005 | No |
| 11 | 080507 | `dfa2c97d` | Buddy, I need you to move forward a bit and then turn around. | device_control | s1_basic_movement | g003 | Yes |
| 12 | 080507 | `038db672` | For now, just turn around basically. | device_control | s1_basic_movement | g006 | No |
| 13 | 080507 | `a54ec139` | OK. Now I need you to set the scenario to corner. Corner movement. | device_control | s1_measurement_task | g000 | No |
| 14 | 080507 | `fa578fc0` | Set the scenario to corner. | device_control | s1_measurement_task | g000 | No |
| 15 | 080507 | `295d8a50` | Set the scenario to corner. | unknown | Excluded duplicate | g000 | Excluded duplicate |
| 16 | 080507 | `06334ede` | OK, now you can start following me. Let's take the data. | device_control | s1_following_me | g007 | No |
| 17 | 080507 | `a8f6991b` | Buddy, stop following me. | device_control | s1_stop_task | g008 | No |
| 18 | 080507 | `ac3ea973` | Now tell me the result. Log the data. | chat | s1_fetch_data | g009 | No |
| 19 | 080507 | `2e5639de` | Now I need to log the data. | chat | s1_fetch_data | g009 | No |
| 20 | 080507 | `c867c278` | Oh my shit. What happened to you, buddy? Buddy, log this trial. Okay, buddy, you can now log this trial. I want to see the result. | chat | s1_fetch_data | g010 | No |
| 21 | 085042 | `ecf1e4bb` | Buddy, can you hear me? Buddy, can you hear me? | chat | s2_alive_check | g011 | No |
| 22 | 085042 | `4b89ce2f` | Buddy, I need you to connect to the rover. | device_control | s2_rover_connection | g012 | No |
| 23 | 085042 | `f5cd2ce6` | Now can you move backward a bit? | device_control | s2_basic_movement | g013 | No |
| 24 | 085042 | `f60db3be` | Can ive backward a bit more? | chat | s2_basic_movement | g014 | No |
| 25 | 085042 | `5364461d` | Buddy, at this point, just move backward a bit. | device_control | s2_basic_movement | g015 | No |
| 26 | 085042 | `d5fa3231` | OK, now can you start following me? | device_control | s2_following_me | g016 | No |
| 27 | 085916 | `102607d9` | Buddy, I need you to connect to the rover. | device_control | s3_rover_connection | g017 | No |
| 28 | 085916 | `b771d768` | Kind of start following me. | device_control | s3_following_me | g018 | No |
| 29 | 091923 | `8d44fbd4` | Set the scenario to stop and go. | chat | s4_measurement_task | g019 | No |
| 30 | 091923 | `270e8fa3` | Set the scenario to stop and go. Buddy, Buddy, can you connect the rover? | device_control | s4_rover_connection | g019 | No |
| 31 | 091923 | `260715f8` | Now can you start following me? | device_control | s4_following_me | g020 | No |
| 32 | 091923 | `410ecc97` | Set the scenario to end. | device_control | s4_measurement_task | g019 | No |
| 33 | 091923 | `532bd172` | Set the scenario to stop and go. | device_control | s4_measurement_task | g019 | Yes |
| 34 | 091923 | `07da70e7` | Whoa. | chat | s4_surprise_reaction | g021 | No |
| 35 | 091923 | `0523dafe` | Set the scenario to stop and go. | device_control | s4_measurement_task | g019 | Yes |
| 36 | 091923 | `7a89e964` | Buddy, I need you to start following me again. | device_control | s4_following_me | g020 | No |
| 37 | 091923 | `7a9d5d9c` | What the hell is that? Stop the rover, buddy. Buddy, stop the rover. | device_control | s4_stop_task | g022 | No |
| 38 | 091923 | `bb84045b` | OK, buddy. Start following me again. | device_control | s4_following_me | g020 | No |
| 39 | 091923 | `660774a9` | Set the scenario to stop and go. | device_control | s4_measurement_task | g019 | Yes |
| 40 | 091923 | `6defac73` | Stop following me. | device_control | s4_stop_task | g020 | No |

## Appendix B. Recalculate grouping metrics from the frozen archive

Run in an environment with pandas and scikit-learn installed. This only reads the archive; it does not alter labels or baseline files.

```python
import io
import itertools
import json
import zipfile

import pandas as pd
from sklearn.metrics import adjusted_rand_score

with zipfile.ZipFile("buddy_memory_results.zip") as archive:
    gold = json.loads(archive.read("annotations.json"))["group_gold"]
    trace = pd.read_csv(io.BytesIO(archive.read("replay_trace.csv")))

rows = [
    (gold[row.id], row.group_id)
    for row in trace.itertuples()
    if row.id in gold
]
tp = fp = fn = 0
for (gold_a, pred_a), (gold_b, pred_b) in itertools.combinations(rows, 2):
    same_gold = gold_a == gold_b
    same_pred = pred_a == pred_b
    tp += int(same_gold and same_pred)
    fp += int(not same_gold and same_pred)
    fn += int(same_gold and not same_pred)

print("Labeled records:", len(rows))
print("TP, FP, FN:", tp, fp, fn)
print("Pair precision:", tp / (tp + fp) if tp + fp else None)
print("Pair recall:", tp / (tp + fn) if tp + fn else None)
print("ARI:", adjusted_rand_score(
    [gold_label for gold_label, _ in rows],
    [predicted for _, predicted in rows],
))
```

Recorded environment: Python 3.13.16; NumPy 2.4.6; pandas 2.2.3; scikit-learn 1.6.1; matplotlib 3.10.0. These are the supplied run's recorded versions, not a guarantee that a fresh hosted runtime will have identical packages. Archive the notebook/code alongside outputs for future reproducibility.
