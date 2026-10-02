# Phase 3 v1 failure closeout

## Immutable campaign identity

- Campaign: `6beb857c-ef2c-4e0a-a6e6-34abee7f4a13`
- Frozen source commit: `ea8b7dc064a218499a812eedf69e7f99c480beca`
- Frozen source tree: `61d5613b227952a177c49e9d638c974c9416681d`
- Campaign plan SHA-256: `effe1b6f06a78a3544444248667f9e5c5f941633424299475121cb329f526f567`
- Completed runs: 7 of the original 30-run plan
- Failed run: 08, `baseline_thinking`, Dev128, repeat 1
- Timeout: 7,200 seconds
- Incident SHA-256: `30d2490971fe06f386ecd707da0e10ec03a8ea104c925dd45a8fca148dfef4c7`

## What completed

All six single-query interface checks completed. Baseline-Instruct Dev128
completed with 51/128 exact matches, zero execution errors, accuracy
0.3984375, mean total latency 15.227511767 seconds, median total latency
15.542215432 seconds, Type-7 p95 latency 21.781016466 seconds, throughput
0.065670610 queries/second, 5,208 generated tokens, and peak allocated CUDA
memory of approximately 4.22 GiB.

## Failure semantics

The baseline-Thinking Dev128 process loaded the model and remained healthy
until the frozen whole-run timeout. Run 08 committed no result, raw records,
partial records, per-query journal, or completion event. Its status is
`TIMEOUT_CENSORED (>7200 seconds; accuracy unavailable)`, not zero accuracy and
not `NOT_RUN`. Runs 09--30 were not started in v1 and are `NOT_RUN`.

The incident, console, runtime event, campaign plan, and first seven result
artifacts are retained byte-for-byte. GPU1 was neither queried nor used.

## Historical closeout tools

- `BLOCKED_STATUS.json` records the v1 stop state.
- `evaluation/closeout_manifest.json` and `evaluation/verify_closeout.py`
  verify the historical Phase 2 and v1 evidence.
- `submission_checker.py` and `scripts/package_submission.py` implement the
  blocked-state package checks developed during closeout.
- `report/archive/phase3_v1_blocked/` contains the explicitly non-final draft
  report and its QA evidence.

Recovery-v2 is a separate user-authorized campaign. It must never overwrite
or combine these v1 artifacts with its formal selection results.
