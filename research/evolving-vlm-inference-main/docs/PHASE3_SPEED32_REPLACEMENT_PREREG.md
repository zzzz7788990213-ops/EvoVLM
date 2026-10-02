# Phase 3 Speed32 replacement preregistration

Status: CPU-frozen before any replacement GPU inference on hopper-25.

## Reason and immutable history

Recovery-v2 campaign `1fa07427-6b42-4bbb-a9f9-af63742bdfc6` completed all
26 scheduled runs. A later read-only timing reconstruction proved that a
foreign GPU0 process overlapped run 22 partially and runs 23--26 completely.
Under the controlling conservative adjudication, the complete old 15-run
Speed32 set is `NON_QUALIFYING_EXTERNAL_GPU0_CONTENTION`. It is retained
byte-for-byte and must never be mixed with replacement speed evidence.

The old six single-query results and five Dev128 results remain valid. This is
a timing-only replacement: it does not reopen accuracy measurement, evolution,
model selection, prompts, decoding, manifests, or methods.

Immutable audit:
`results/phase3_recovery_v2/incidents/1fa07427-6b42-4bbb-a9f9-af63742bdfc6.gpu0_external_contention.audit.json`
(SHA-256 `2f97fbe9539fa1371fe17f43f6d55d9610af1f051037071967885e258daf3c27`).
It anchors 1,233 old campaign files and their ordered byte manifest.

## Frozen replacement

The replacement uses a new UUID but the exact Recovery-v2 schedule entries
12--26: five programs, the same Speed32 query order, three repeats, the same
rotation, one identical fixed warmup, runtime, evaluator, and selection rules.
All 15 replacement results are required. Partial replacement and old/new
Speed32 mixing fail closed.

Machine authorization:
`evaluation/recovery_v2/speed32_replacement_authorization.json`
(SHA-256 `70b67dbbcd3db6c6e7379fd308bb0fa314e907e3e20f8a0a05943e134ab3c841`).
It freezes these input hashes:

- runtime/evaluator `5b13f1f6c8543b48192bd0210e8bb1e4446fec531e415707f46b4a7dd5d4afd8`;
- protocol `a251e703c94167df9ff19dd4ff21f9c66297e1caa80d1519a0e0721e711b79dd`;
- candidates `7e1f0416cc327d6257e1d235a31f7b48e384b44a8e28eed9a75245d1bbe238f9`;
- data manifest `d9beee3fa956313c9c7d0691c03fdadeba031937b1fc8b25eaf5a29b4bb03cbd`;
- selection rules `fb3a634fa042f3e473053cfa03463b045aaec8a96989627ffebf9d9b55bc52d7`;
- Speed32 manifest `29c3c972089731c12c89c2df2ed5eb81723bf82feda9a77999e255ee10147f87`;
- warmup manifest `201bd90ecd2478682e67a422c061c8137db9fff81d8514dc329b7450555307ea`.

## GPU and evidence gates

Only physical GPU0 on hopper-25 is eligible, with
`CUDA_VISIBLE_DEVICES=0`, one visible logical `cuda:0`, and strictly serial
execution. GPU1 is neither queried nor used. Before starting, GPU0 must remain
free for a continuous 120 seconds sampled every 15 seconds. During every
replacement entry, GPU0 compute-process telemetry is sampled every five
seconds. Detection of a foreign PID stops only this project's process group,
preserves an incident, never signals the foreign process, and forbids a third
Speed32 campaign.

Each entry retains the Recovery-v2 per-query no-replace journal and semantic
validator. After all 15 results verify, a separate immutable measurements
completion record binds the plan, marker, audit, source identity, and every
result hash. Two read-only checks must leave hashes and mtimes invariant before
finalization. Only then may old Dev128 plus the full new Speed32 set enter the
frozen aggregator, best export, official report, checker, and package.

No replacement GPU query had occurred when this document was committed.
