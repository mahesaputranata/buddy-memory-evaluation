"""B5 retrieval replay extracted from the supplied Colab notebook.
Notebook-style globals are retained deliberately to minimize algorithm changes.
No answering model or rover command is invoked.
"""
import argparse
ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--logs', nargs='+', default=[str(ROOT/'data/benchmark/interactions.jsonl')])
parser.add_argument('--annotations', default=str(ROOT/'data/benchmark/annotations.json'))
parser.add_argument('--backend', choices=['hashing','sentence_transformer'])
parser.add_argument('--config', help='Optional JSON config overrides')
parser.add_argument('--output', default=str(ROOT/'results/local/retrieval'))
args = parser.parse_args()
ANNOTATION_PATH = args.annotations


import json, math, time, re, itertools, platform, zipfile
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import adjusted_rand_score
from IPython.display import display

CFG = {
    "backend": "sentence_transformer",  # or "sentence_transformer"
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "group_similarity": 0.45,
    "recurrence_similarity": 0.45,
    "recurrence_count": 3,  # PRIOR related exchanges required; current record excluded
    "within_session": True,
    "near_duplicate_seconds": 1.0,
    "drop_duplicate_candidates": True, # pilot choice: inspect candidates below
    "top_k": 3,
    "context_char_budget": 2400, # equal maximum context characters across methods
    "min_retrieval_score": 0.01,
    "include_assistant_in_embedding": False, # user-only reduces contamination
}

# B4
CFG["retrieval_include_assistant"] = True

# B5
CFG["session_aware_retrieval"] = True
CFG.update(json.loads(Path(args.config).read_text()) if args.config else {})
if args.backend: CFG["backend"] = args.backend


# Optional upload: uncomment and run, then set LOG_PATHS to the returned names.
# from google.colab import files
# uploaded = files.upload()
# LOG_PATHS = [name for name in uploaded if name.endswith(".jsonl")]

LOG_PATHS = args.logs  # e.g. ["/content/session_2026-11-13.jsonl"]

def load_records(paths):
    items = []
    if not paths:
        raise ValueError("Supply at least one JSONL path")
    else:
        for path in paths:
            for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from e
                items.append(row)
    if not items:
        raise ValueError("No records loaded")
    ids = set()
    for row in items:
        for field in ["id", "ts", "session_id", "input", "response"]:
            if field not in row or not isinstance(row[field], str):
                raise ValueError(f"Missing/non-string {field}: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate record ID: {row['id']}; remove overlapping files")
        ids.add(row["id"])
        ts = pd.Timestamp(row["ts"])
        if ts.tzinfo is None:
            raise ValueError(f"Timezone required: {row['id']}")
        row["_ts"] = ts.tz_convert("UTC")
    return sorted(items, key=lambda r: (r["_ts"], r["session_id"], r.get("turn", 0), r["id"]))

records = load_records(LOG_PATHS)
display(pd.DataFrame([{k:r.get(k) for k in ["id","session_id","turn","input","response"]} for r in records]))
print(f"{len(records)} records; {len(set(r['session_id'] for r in records))} sessions")


def normalize_text(s):
    return " ".join(s.casefold().split())

def deduplicate(items, cfg):
    kept, candidates, aliases, last = [], [], {}, {}
    for r in items:
        key = (r["session_id"], normalize_text(r["input"]), normalize_text(r["response"]))
        prev = last.get(key)
        is_dup = prev is not None and 0 <= (r["_ts"] - prev["_ts"]).total_seconds() <= cfg["near_duplicate_seconds"]
        if is_dup:
            canonical = aliases.get(prev["id"], prev["id"])
            candidates.append({"id":r["id"], "candidate_duplicate_of":canonical,
                               "gap_seconds":(r["_ts"]-prev["_ts"]).total_seconds()})
        if is_dup and cfg["drop_duplicate_candidates"]:
            aliases[r["id"]] = canonical
        else:
            kept.append(r)
            aliases[r["id"]] = r["id"]
        last[key] = r
    return kept, candidates, aliases

clean, duplicate_candidates, aliases = deduplicate(records, CFG)
display(pd.DataFrame(duplicate_candidates))
print(f"Retained {len(clean)} / {len(records)} records")
by_id = {r["id"]:r for r in clean}


annotation = json.loads(Path(ANNOTATION_PATH).read_text(encoding="utf-8"))
GROUP_GOLD = annotation.get("group_gold", {})
TRIGGER_GOLD = annotation.get("trigger_gold", {})
MANUAL_MEMORIES = annotation.get("manual_memories", [])
QUESTIONS = annotation.get("questions", [])
IMPORTANT_RECORD_IDS = set(annotation.get("important_record_ids", []))
# If using entirely different logs, remove/replace the starter annotations first.
known_ids = set(aliases)
for m in MANUAL_MEMORIES:
    assert m["source_ids"] and set(m["source_ids"]) <= known_ids, f"Unknown source in {m['memory_id']}"
    assert m["kind"] in {"episode", "fact"}
    for claim in m.get("claims", []):
        assert set(claim["source_ids"]) <= set(m["source_ids"])
for q in QUESTIONS:
    assert q["after_id"] in known_ids, q["qid"]
    assert q["gold_source_ids"] and set(q["gold_source_ids"]) <= known_ids
assert set(GROUP_GOLD) <= known_ids
assert set(TRIGGER_GOLD) <= known_ids
assert IMPORTANT_RECORD_IDS <= known_ids
assert len({m['memory_id'] for m in MANUAL_MEMORIES}) == len(MANUAL_MEMORIES)
assert len({q['qid'] for q in QUESTIONS}) == len(QUESTIONS)
print("Annotation references validated. Content fidelity still requires human review.")


class Encoder:
    def __init__(self, cfg):
        self.backend = cfg["backend"]
        if self.backend == "hashing":
            self.model = HashingVectorizer(n_features=4096, alternate_sign=False,
                                           ngram_range=(1,2), norm="l2")
        elif self.backend == "sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg["embedding_model"])
        else:
            raise ValueError(self.backend)
        self.cache = {}
    def encode(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            if self.backend == "hashing":
                vectors = self.model.transform(missing).toarray()
            else:
                vectors = self.model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self.cache.update(zip(missing, vectors))
        return np.asarray([self.cache[t] for t in texts])

encoder = Encoder(CFG)
def exchange_embedding_text(r, cfg):
    return r["input"] + ("\n" + r["response"] if cfg["include_assistant_in_embedding"] else "")


def replay(items, cfg):
    groups, history, rows, promoted = [], [], [], set()
    for r in items:
        vec = encoder.encode([exchange_embedding_text(r, cfg)])[0]
        eligible = [g for g in groups if not cfg["within_session"] or g["session_id"] == r["session_id"]]
        def cosine(a,b):
            return float(np.dot(a,b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
        scores = [cosine(vec, g["sum_vector"]) for g in eligible]
        if scores and max(scores) >= cfg["group_similarity"]:
            group = eligible[int(np.argmax(scores))]
            group["sum_vector"] += vec
            group["source_ids"].append(r["id"])
        else:
            group = {"group_id":f"g{len(groups):03d}", "session_id":r["session_id"],
                     "sum_vector":vec.copy(), "source_ids":[r["id"]]}
            groups.append(group)
        related = [old["id"] for old, ov in history
                   if (not cfg["within_session"] or old["session_id"] == r["session_id"])
                   and cosine(vec,ov) >= cfg["recurrence_similarity"]]
        reason = "important" if r["id"] in {aliases[x] for x in IMPORTANT_RECORD_IDS} else (
                 "recurrence" if len(related) >= cfg["recurrence_count"] else "none")
        if reason != "none":
            promoted.update(related + [r["id"]])
        rows.append({"id":r["id"], "session_id":r["session_id"], "group_id":group["group_id"],
                     "related_prior_count":len(related), "triggered":reason != "none",
                     "trigger_reason":reason, "promoted_ids":sorted(promoted)})
        history.append((r,vec))
    return pd.DataFrame(rows), groups

trace, groups = replay(clean, CFG)
display(trace.drop(columns="promoted_ids"))
print("Trigger opportunities:", int(trace.triggered.sum()), "/", len(trace))


def organization_scores(trace):
    selected = [(GROUP_GOLD[r.id], r.group_id) for r in trace.itertuples() if r.id in GROUP_GOLD]
    if len(selected) < 2:
        return {"labeled_records":len(selected), "ARI":np.nan,"pair_precision":np.nan,"pair_recall":np.nan}
    gold,pred = zip(*selected)
    tp=fp=fn=0
    for i,j in itertools.combinations(range(len(gold)),2):
        actual, guessed = gold[i]==gold[j], pred[i]==pred[j]
        tp += actual and guessed; fp += not actual and guessed; fn += actual and not guessed
    return {"labeled_records":len(selected), "ARI":adjusted_rand_score(gold,pred),
            "pair_precision":tp/(tp+fp) if tp+fp else np.nan,
            "pair_recall":tp/(tp+fn) if tp+fn else np.nan}

def trigger_scores(trace):
    pairs=[(bool(TRIGGER_GOLD[r.id]),r.triggered) for r in trace.itertuples() if r.id in TRIGGER_GOLD]
    tp=sum(g and p for g,p in pairs); fp=sum(not g and p for g,p in pairs); fn=sum(g and not p for g,p in pairs)
    return {"labeled_records":len(pairs),"precision":tp/(tp+fp) if tp+fp else np.nan,
            "recall":tp/(tp+fn) if tp+fn else np.nan}
print("Grouping:", organization_scores(trace))
print("Triggers:", trigger_scores(trace))
# Inspect a group without consulting gold labels during grouping.
for g in groups[:8]:
    print("\n",g["group_id"])
    for rid in g["source_ids"]:
        print(" ",rid,by_id[rid]["input"])


def raw_text(r):
    return f"[{r['id']}] session={r['session_id']} time={r['ts']}\nUSER: {r['input']}\nASSISTANT CLAIM (not a tool receipt): {r['response']}"

def raw_retrieval_text(record, cfg):
    if cfg.get("retrieval_include_assistant", False):
        return (
            "USER REQUEST OR REPORT:\n"
            + record["input"]
            + "\nASSISTANT RESPONSE (not independently verified):\n"
            + record["response"]
        )

    return exchange_embedding_text(record, cfg)

def candidate_memories(prefix, mode, promoted):
    seen = {r["id"] for r in prefix}
    candidates=[{"memory_id":"raw_"+r["id"],"kind":"raw", "source_ids":[r["id"]],
                 "search_text":raw_retrieval_text(r, CFG),"context_text":raw_text(r)} for r in prefix]
    if mode != "A_raw":
        for m in MANUAL_MEMORIES:
            sources = sorted({aliases[s] for s in m["source_ids"]})
            if not set(sources) <= seen:
                continue
            if mode == "C_selective" and not set(sources) <= promoted:
                continue
            evidence = "\n".join(raw_text(by_id[s]) for s in sources)
            candidates.append({**m,"source_ids":sources,"search_text":m["text"],
                               "context_text":m["text"]+"\nSOURCE RECORDS:\n"+evidence})
    return candidates

def retrieve(question, candidates, cfg):
    if not candidates:
        return [], 0

    # Encode the question and candidate texts.
    qvec = encoder.encode([question])[0]
    vectors = encoder.encode([
        candidate["search_text"]
        for candidate in candidates
    ])

    # Both supported encoder backends produce normalized vectors.
    scores = vectors @ qvec
    order = np.argsort(-scores, kind="stable")

    selected = []
    covered_sources = set()
    used_chars = 0

    for index in order:
        score = float(scores[index])

        if score < cfg["min_retrieval_score"]:
            continue

        candidate = candidates[index]
        source_ids = set(candidate["source_ids"])

        # Skip evidence already fully covered by selected items.
        new_sources = source_ids - covered_sources

        if not new_sources:
            continue

        context_size = len(candidate["context_text"])

        if used_chars + context_size > cfg["context_char_budget"]:
            continue

        selected.append({
            **candidate,
            "score": score,
        })

        covered_sources.update(source_ids)
        used_chars += context_size

        if len(selected) >= cfg["top_k"]:
            break

    return selected, used_chars

def filter_candidates_by_session(question, prefix, candidates, cfg):
    import re

    if not cfg.get("session_aware_retrieval", False):
        return candidates, None

    # Session order uses only records available at this checkpoint.
    session_order = list(dict.fromkeys(
        record["session_id"] for record in prefix
    ))

    text = question.casefold()

    match = re.search(
        r"\b(?:the|this)\s+"
        r"(first|second|third|fourth|fifth|sixth|seventh|"
        r"eighth|ninth|tenth|\d+(?:st|nd|rd|th))\s+session\b",
        text,
    )

    if match:
        ordinal = match.group(1)
        word_numbers = {
            word: number
            for number, word in enumerate(
                [
                    "first", "second", "third", "fourth", "fifth",
                    "sixth", "seventh", "eighth", "ninth", "tenth",
                ],
                start=1,
            )
        }

        if ordinal in word_numbers:
            number = word_numbers[ordinal]
        else:
            number = int(re.match(r"\d+", ordinal).group())

        if not 1 <= number <= len(session_order):
            raise ValueError(
                "Requested session is unavailable at this checkpoint."
            )

        target_session = session_order[number - 1]

    elif re.search(r"\b(?:this|current)\s+session\b", text):
        if not prefix:
            raise ValueError("No current session is available.")

        target_session = prefix[-1]["session_id"]

    else:
        # No recognized session request: preserve the original search.
        return candidates, None

    source_sessions = {
        record["id"]: record["session_id"]
        for record in prefix
    }

    filtered = []

    for candidate in candidates:
        sources = candidate["source_ids"]

        if not sources:
            raise ValueError("Candidate has no source records.")

        if any(source not in source_sessions for source in sources):
            raise ValueError("Candidate references unavailable evidence.")

        if all(
            source_sessions[source] == target_session
            for source in sources
        ):
            filtered.append(candidate)

    return filtered, target_session

def evaluate(cfg):
    trace_local,_ = replay(clean,cfg)
    position={r["id"]:i for i,r in enumerate(clean)}
    details, metrics = [], []
    for q in QUESTIONS:
        cutoff=position[aliases[q["after_id"]]]
        prefix=clean[:cutoff+1]
        seen={r["id"] for r in prefix}
        gold={aliases[s] for s in q["gold_source_ids"]}
        assert gold <= seen, f"Future gold evidence in {q['qid']}"
        promoted=set(trace_local.iloc[cutoff].promoted_ids)
        for mode in ["A_raw","B_manual","C_selective"]:
            start=time.perf_counter()
            candidates = candidate_memories(prefix, mode, promoted)

            candidates, session_scope = filter_candidates_by_session(
                q["question"], prefix, candidates, cfg
            )

            found, chars = retrieve(q["question"], candidates, cfg)
            ms=(time.perf_counter()-start)*1000
            sources={s for m in found for s in m["source_ids"]}
            hits=len(gold & sources)
            rank=next((i for i,m in enumerate(found,1) if gold & set(m["source_ids"])),None)
            metrics.append({"qid":q["qid"],"split":q.get("split","dev"),"mode":mode,
                            "evidence_recall":hits/len(gold),
                            "evidence_precision":hits/len(sources) if sources else 0.,
                            "complete_evidence":float(gold <= sources),"MRR":1/rank if rank else 0.,
                            "source_count":len(sources),"context_chars":chars,"retrieval_ms":ms,
                            "candidate_count":len(candidates)})
            details.append({"qid":q["qid"],"mode":mode,"question":q["question"],
                            "reference_answer":q["reference_answer"],"gold_source_ids":sorted(gold), "session_scope": session_scope,
                            "retrieved":found})
    return pd.DataFrame(metrics),details

metrics, details = evaluate(CFG)
display(metrics)
summary = metrics.groupby(["split","mode"])[["evidence_recall","evidence_precision","complete_evidence","MRR","source_count","context_chars","retrieval_ms"]].mean()
display(summary)
print("Latency includes candidate encoding/cache effects; not an end-to-end deployment benchmark.")


for result in details:
    covered = set()

    for item in result["retrieved"]:
        sources = set(item["source_ids"])
        new_sources = sources - covered

        assert new_sources, (
            f"Fully redundant item returned: "
            f"{result['qid']} / {result['mode']} / "
            f"{item['memory_id']}"
        )

        covered.update(sources)

print("Passed: every selected item contributes new source evidence.")

import hashlib, importlib.metadata
OUT = Path(args.output)
if OUT.exists(): raise FileExistsError(f"Output exists: {OUT}; use a fresh directory")
OUT.mkdir(parents=True)
serializable_records=[{k:v for k,v in r.items() if k!="_ts"} for r in records]
snapshot="\n".join(json.dumps(r,ensure_ascii=False) for r in serializable_records)+"\n"
manifest={"config":CFG,"important_record_ids":sorted(IMPORTANT_RECORD_IDS),
          "input_sha256":hashlib.sha256(snapshot.encode()).hexdigest(),
          "python":platform.python_version(),
          "packages":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scikit-learn","matplotlib"]},
          "limitations":["manual consolidation", "development pilot", "no generated answers", "no verified robot execution", "latency affected by caching"]}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(OUT/"input_snapshot.jsonl").write_text(snapshot)
(OUT/"annotations.json").write_text(json.dumps({"group_gold":GROUP_GOLD,"trigger_gold":TRIGGER_GOLD,
    "manual_memories":MANUAL_MEMORIES,"questions":QUESTIONS,"important_record_ids":sorted(IMPORTANT_RECORD_IDS)},indent=2))
(OUT/"retrieval_details.json").write_text(json.dumps(details,indent=2))
(OUT/"duplicate_candidates.json").write_text(json.dumps(duplicate_candidates,indent=2))
trace.to_csv(OUT/"replay_trace.csv",index=False)
metrics.to_csv(OUT/"retrieval_metrics.csv",index=False)
summary.to_csv(OUT/"summary.csv")
# Fill blank topic/trigger/claim-review fields manually; blanks are not negative labels.
pd.DataFrame([{ "id":r["id"],"input":r["input"],"gold_group":"","should_trigger":"","note":""} for r in clean]).to_csv(OUT/"annotation_worksheet.csv",index=False)
pd.DataFrame([{ "memory_id":m["memory_id"],"claim":c["text"],"source_ids":"|".join(c["source_ids"]),
               "support_verdict":"","attribution_error":"","important_omission":"","review_note":""}
              for m in MANUAL_MEMORIES for c in m.get("claims",[])]).to_csv(OUT/"claim_review.csv",index=False)
zip_path=OUT.parent / (OUT.name + ".zip")
with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir(): z.write(p,arcname=p.name)
print("Exported:",zip_path.resolve())
# from google.colab import files
# files.download(str(zip_path))
