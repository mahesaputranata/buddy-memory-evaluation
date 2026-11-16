# B5.5 diagnostic: paste into a new Colab cell AFTER the B5 evaluation.
# Requires the existing encoder, CFG, QUESTIONS, clean, aliases, by_id,
# candidate_memories(), and filter_candidates_by_session().
# This block does not modify production retrieval, questions, or configuration.
import json
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

QUESTION_ID = "b3_stop_requests"
DIAG_DIR = Path("buddy_B5_5_diagnostic")
DIAG_DIR.mkdir(exist_ok=True)

question = next(q for q in QUESTIONS if q["qid"] == QUESTION_ID)
positions = {r["id"]: i for i, r in enumerate(clean)}
cutoff = positions[aliases[question["after_id"]]]
past_records = clean[:cutoff + 1]

# Use every raw candidate in the resolved session, not only the top three.
candidates = candidate_memories(past_records, "A_raw", set())
candidates, scope = filter_candidates_by_session(
    question["question"], past_records, candidates, CFG
)
assert candidates, "No candidates available. Check B5 setup."
assert scope is not None, "Enable B5 session-aware retrieval first."

query_vector = encoder.encode([question["question"]])[0]
full_texts = [c["search_text"] for c in candidates]
user_texts = [
    "USER REQUEST OR REPORT:\n" + by_id[c["source_ids"][0]]["input"]
    for c in candidates
]
full_scores = encoder.encode(full_texts) @ query_vector
user_scores = encoder.encode(user_texts) @ query_vector

# Rank 1 is highest. Stable sorting resolves exact ties by candidate order,
# matching retrieve(). A rank is not a probability or truth assessment.
def stable_ranks(scores):
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return ranks

full_ranks = stable_ranks(full_scores)
user_ranks = stable_ranks(user_scores)
reference_ids = {aliases[s] for s in question["gold_source_ids"]}
rows = []
for i, candidate in enumerate(candidates):
    assert candidate["kind"] == "raw" and len(candidate["source_ids"]) == 1
    source_id = candidate["source_ids"][0]
    record = by_id[source_id]
    assert record["session_id"] == scope
    rows.append({
        "source_id": source_id,
        "timestamp": record["ts"],
        "user_request": record["input"],
        "buddy_reply": record["response"],
        "full_score": float(full_scores[i]),
        "user_only_score": float(user_scores[i]),
        "full_rank": int(full_ranks[i]),
        "user_only_rank": int(user_ranks[i]),
        # Gold labels identify reference evidence AFTER ranking only.
        "reference_evidence": source_id in reference_ids,
        "full_search_text": full_texts[i],
        "user_only_search_text": user_texts[i],
    })

rank_table = pd.DataFrame(rows).sort_values("full_rank").reset_index(drop=True)
print("B5.5 — diagnostic only; production retrieval is unchanged")
print("Question:", question["question"])
print("Session:", scope, "| Raw candidates:", len(rank_table))
with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
    display(rank_table[["user_request", "full_rank", "user_only_rank",
                        "full_score", "user_only_score", "reference_evidence"]])

# Plot: each row is one candidate; two marks show its two ranking positions.
# Left = higher rank. REF labels do not influence scores or ordering.
y = np.arange(len(rank_table))
labels = [
    textwrap.fill(
        ("[REF] " if row.reference_evidence else "")
        + row.timestamp[11:19] + " | " + row.user_request,
        width=48,
    )
    for row in rank_table.itertuples()
]
fig, ax = plt.subplots(figsize=(12, max(6, len(rank_table) * 0.7)))
top_k = min(int(CFG["top_k"]), len(rank_table))
ax.axvspan(0.5, top_k + 0.5, color="#16a34a", alpha=0.09,
           label=f"Top {top_k} ranking positions")
for i, row in rank_table.iterrows():
    ax.plot([row.full_rank, row.user_only_rank], [i, i],
            color="#9ca3af", linewidth=1.4, zorder=1)
ax.scatter(rank_table["full_rank"], y - 0.09, s=55, marker="o",
           color="#2563eb", label="User + Buddy reply", zorder=3)
ax.scatter(rank_table["user_only_rank"], y + 0.09, s=55, marker="s",
           color="#ea580c", label="User request only", zorder=3)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xticks(range(1, len(rank_table) + 1))
ax.set_xlim(0.5, len(rank_table) + 0.5)
ax.set_xlabel("Similarity rank — 1 is highest; farther left is better")
ax.set_title("B5.5: Where do the explicit stop requests rank?", pad=18)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
for tick, is_reference in zip(ax.get_yticklabels(), rank_table.reference_evidence):
    tick.set_fontweight("bold" if is_reference else "normal")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
          ncol=2, frameon=False)
fig.text(0.01, 0.01,
         "REF = manually annotated evidence. Shading shows rank positions, not final budget-aware selection.",
         fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(DIAG_DIR / "stop_request_rank_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig(DIAG_DIR / "stop_request_rank_comparison.pdf", bbox_inches="tight")
plt.show()

rank_table.to_csv(DIAG_DIR / "all_candidate_ranks.csv", index=False)
metadata = {
    "experiment": "B5.5", "kind": "diagnostic_ablation_not_production_change",
    "qid": question["qid"], "question": question["question"],
    "after_id": question["after_id"], "session_scope": scope,
    "candidate_count": len(rank_table), "config": dict(CFG),
    "comparison": "full exchange versus user-only text with matching USER prefix",
    "limitations": ["one development query", "raw candidates only",
                    "ranking before context-budget selection", "no generated answers"],
}
(DIAG_DIR / "diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2))
zip_path = Path("buddy_memory_results_B5_5_diagnostic.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(DIAG_DIR.iterdir()):
        if path.is_file():
            archive.write(path, arcname=path.name)
print("Saved diagnostic plot, table, and metadata:", zip_path)
# Optional Colab download:
# from google.colab import files
# files.download(str(zip_path))
