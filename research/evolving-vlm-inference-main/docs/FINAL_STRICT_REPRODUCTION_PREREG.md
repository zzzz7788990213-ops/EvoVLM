# Final Strict-Compliance Fresh Reproduction Preregistration

Status: PREREGISTERED_BEFORE_IMPLEMENTATION

## Immutable anchors

- Starting commit: `c7b08b6f8481179047614bfa9f2b188c5c26a7c4`.
- Starting tree: `26882dd6a0df0403d0072439fe61ad961f459a84`.
- Formal manifest: `evaluation/recovery_v2/final_manifest.json`.
- Formal manifest SHA-256: `3ca726afb199f1e2890855838f9949dfcf69dad743484e3d7067ae4380e5a65a`.
- Damaged historical smoke document SHA-256: `ab6331d40fd83455049925e031e71c11258e0ec145495303ba463f7f45cc556d`.

The formal manifest and all formal results, prompts, preprocessing, decoding,
scoring, Speed32 measurements, selection rules, and best-program selection are
immutable in this repair.

## Fixed CPU repair scope

1. Replace the damaged active smoke-document generator and make the checker
   reject non-finite or placeholder active state.
2. Configure one visible CUDA device before any torch/program import in the
   official evaluator while preserving a caller-provided single GPU index.
3. Make no-argument `bash reproduce.sh` perform fresh inference and retain
   explicit `evidence`, `check`, `smoke`, and `full-inference` modes.
4. Port the naive baseline through the same path/model resolver without changing
   its inference semantics.
5. Add an immutable V1 plan-hash erratum binding the old declaration and the
   independently recomputed plan hash.
6. Strengthen attempt/start/end/runtime-event/console/result cross-binding,
   preserving transparent legacy caveats when immutable history lacks a field.
7. Package exactly the frozen Dev128 images as regular files with an ordered
   hash manifest.

## Single authorized fresh reproduction

Label: `FINAL_FRESH_REPRODUCTION_ONLY`.

Expected Dev128 exact-match counts, each with 128 unique queries in the frozen
order and zero errors:

- `baseline_instruct`: 51/128
- `manual_instruct`: 78/128
- `manual_thinking`: 23/128
- `evolved_instruct`: 78/128
- `evolved_thinking`: 44/128

`baseline_thinking` remains `TIMEOUT_CENSORED >7200s` with accuracy
unavailable. It may receive one Single1 entrypoint smoke only.

The three best programs must first be byte-identical to
`evolved_instruct.py`; each then receives one separate official-filename
Single1 smoke. A failed identity check requires that program's own Dev128 run.

## Frozen GPU lanes

- GPU0, serial within lane: `baseline_instruct`, `manual_instruct`,
  `evolved_instruct`, then `best_accuracy`, `best_speed`, and
  `best_overall` Single1.
- GPU1, serial within lane: `manual_thinking`, `evolved_thinking`, and
  optional `baseline_thinking` Single1.
- The two lanes may run concurrently. Every child sees one device as logical
  `cuda:0`, uses isolated Evo-local output/tmp/cache roots, and is reaped.
- The production run starts only from a clean git-archive extraction of the
  frozen pre-reproduction commit.

## Explicit exclusions and stop rule

This repair runs no evolution, mutation, search, new candidate, formal
Speed32, formal reselection, or method optimization. Fresh reproduction does
not affect formal metrics or selection. Any Dev128 mismatch, need to alter
inference semantics, formal-artifact mutation, unavailable Evo-local resource,
wrong host/GPU ownership, or need for network access is a hard stop. Once the
fresh receipt, report, unique final ZIP, clean-room checks, and clean Git state
all pass, work stops immediately.
