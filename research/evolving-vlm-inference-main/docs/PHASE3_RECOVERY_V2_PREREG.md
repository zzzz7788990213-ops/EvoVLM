# Phase 3 Recovery-v2 preregistration

Status: frozen before any Recovery-v2 GPU inference.

## Authorization and separation

Recovery-v2 is an explicit user-authorized execution-resilience repair of the
fully retained v1 campaign `6beb857c-ef2c-4e0a-a6e6-34abee7f4a13`. The v1 run
08 baseline-Thinking Dev128 process reached 7,200 seconds without committing a
result, partial record, raw prediction, or per-query journal. Its status is
`TIMEOUT_CENSORED (>7200 seconds; accuracy unavailable)`. The recovery decision
was made without observing run08 accuracy or any partial prediction, so it is
not metric-conditioned.

v1 artifacts remain byte-for-byte historical evidence. No v1 result is mixed
into Recovery-v2 selection. Candidate programs, model weights, Dev128 queries,
Speed32 queries, prompts, greedy decoding, and case-insensitive exact match are
unchanged.

Machine-readable authorization:
`evaluation/recovery_v2/recovery_authorization.json`.

## Frozen scope

The v2 schedule has 26 sequential GPU0-only entries:

- six single-query interface checks;
- Dev128 for baseline-Instruct and all four optimized programs;
- Speed32 three repeats for those same five programs, in frozen rotations.

The censored baseline-Thinking is single-smoked but is not rerun on Dev128 or
Speed32 and is ineligible for best selection. Every program has a cumulative
14,400-second formal process budget. A timed-out program is marked `TIMEOUT`,
becomes best-ineligible, and does not prevent unrelated programs from running.

Machine-readable protocol:
`evaluation/recovery_v2/protocol.json`.

## Atomic query journals

Every completed measured query is durably committed to one no-replace JSON
file after the CUDA post-synchronization and outside the measured interval.
Each row binds campaign/run/attempt IDs, source commit and tree, program source,
model/revision/processor/weight hashes, manifest/order, query identity and image
hash, prediction hash, rederived correctness, latency components, token count,
termination reason, peak memory, and GPU0 scope.

Resume accepts only a contiguous prefix of canonical journal filenames,
revalidates each row against current ground truth and frozen inputs, and skips
the validated prefix. Gaps, extras, duplicates, partial files, symlinks,
nonfinite numbers, stale source/model/manifest bindings, or correctness tamper
fail closed. Existing committed results are verified CPU-only and never replay
the GPU.

## Selection

Only `manual_instruct`, `manual_thinking`, `evolved_instruct`, and
`evolved_thinking` may win. Accuracy uses integer Dev128 correct count. Speed
uses the per-query median across all three repeats before the 32-query
aggregate. The exact tie-breaks, speed floor, and author-defined harmonic
overall score are frozen in
`evaluation/recovery_v2/selection_rules.json`.

## Frozen artifacts

- Candidate/runtime closure: `evaluation/recovery_v2/candidate_manifest.json`
- Data roles/order: `evaluation/recovery_v2/data_manifest.json`
- Protocol: `evaluation/recovery_v2/protocol.json`
- Selection: `evaluation/recovery_v2/selection_rules.json`
- Authorization: `evaluation/recovery_v2/recovery_authorization.json`

The Recovery-v2 campaign plan will additionally bind the clean Git commit/tree
that contains these files. GPU1 is never queried or used.
