# Data and provenance

`data/benchmark/interactions.jsonl` is a minimal projection of the 40-record 13 November 2026 source log. Original record IDs and timestamps remain so development annotations can resolve their sources. The text is real, not synthetic. Routing labels, confidence fields, context state, review/shadow metadata are excluded; this projection therefore has different bytes/fingerprint from the original full snapshot.

`annotations.json` contains primary group labels, empty trigger labels, three manual memories, and 15 development questions. Group categories were manually reviewed; the mixed activity/action taxonomy has known ambiguities. Empty trigger labels mean unjudged, not False. Manual memories preserve attribution rather than verifying assistant claims.

No third-party chat transcripts or hardware recordings are added. Local model packets include the same benchmark excerpts and expected-answer metadata for the harness; the runner's allowlist controls which fields reach the model.

Historical raw answering outputs and machine-specific manifests are not included by default. Aggregate summaries and the detailed historical report remain. Original source snapshots should be retained privately, outside the public repository.

Before adding future logs, review names, contacts, addresses, GPS coordinates, credentials, private paths, third-party content, and file artifacts. Keep unreviewed files in `data/private/` and `results/local/`.
