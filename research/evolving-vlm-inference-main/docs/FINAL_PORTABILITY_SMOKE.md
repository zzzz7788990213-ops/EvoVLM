# Final portability-only Single1 smoke

Status: `SUPERSEDED_BY_POST_SMOKE_EXECUTABLE_CHANGE`.

The bytes and results below are retained as historical evidence, but this run
must not satisfy the final portability gate: `evaluation/portable_reproduce.py`
changed afterward to restore the checker return contract. A fresh relocated
four-program smoke is required from the post-fix commit.

This evidence is labeled `FINAL_PORTABILITY_SMOKE_ONLY`. It is not a formal
Phase 3 measurement, does not enter selection, and does not change the frozen
Recovery-v2 result set.

- Source commit: `b66053e0383caefdfc9ac159cbfec96dab50525e`
- Source tree: `1c11eecd96577077bc25b7a896de628963551950`
- Run ID: `94db2909-8774-40f8-a6d5-9062dd3f8c02`
- Result directory: `results/final_portability_smoke_b66053e`
- Receipt SHA-256: `9a1d35110f3896720dbab63ed559d7ddb3f2383d8d65577f3f1cd71cefc96213`
- Plan SHA-256: `a2aa2fa98f2b99fce2a725f28cb7d93cc6818ba4320558e30dab3c41fd59c0b0`
- Formal metrics affected: `false`

The source was extracted from the commit with `git archive` into a fresh
managed staging directory. The evaluator, all four program modules, and the
smoke orchestrator resolved from that staging directory. Only the sealed model
snapshots and image bytes were read from explicit read-only Evo paths. The
descriptive JSON came from the staged tree.

| Lane | Program | Errors | Total seconds | Result SHA-256 | Source SHA-256 |
|---|---|---:|---:|---|---|
| physical GPU0, logical cuda:0 | `manual_instruct` | 0 | 29.339748 | `b6ab5da7451f0964b0fe585353207294468c680501d38324f8bf8bfce7f64e2c` | `397f3a92d21f2dac64e5bf012ccdeca6e2c48fcfed54c11985268df48cf516f4` |
| physical GPU0, logical cuda:0 | `evolved_instruct` | 0 | 24.444256 | `a811e4e2caf3dfabfe0d253a43bd2cf0d3648787887dda0d70ba0ec4619bbe41` | `1c58d489733b7031c7bed3d7a27a9840cf364a9c916cd6eab54febd4b43bb984` |
| physical GPU1, logical cuda:0 | `manual_thinking` | 0 | 34.537043 | `dc5a9ab81480b9dd01f3aa008db5684bb7b1f5e66fbb8ac849dd3af7fc63444e` | `60db13925030b2e346f363ab4a96eb6317641e0e964e4dd7519f7b2f4073fb58` |
| physical GPU1, logical cuda:0 | `evolved_thinking` | 0 | 30.172258 | `218944b84dbcdcb98cef822b3d8af6bc6c7c22a917e9bc2e61562b103941c664` | `81bedc0e16a3954bc55aea73120ed47f661534a79f01ff9dfd66069c2af1efe6` |

Each result is 1/1 correct. For all four programs, independent comparison
confirmed equal query key, question, image SHA-256, prediction bytes/hash, and
correctness against the sealed formal Single1. The prediction is `60`, SHA-256
`39fa9ec190eee7b6f4dff1100d6343e10918d044c75eac8f9e9a2596173f80c9`.
The image SHA-256 is
`d597cdcfdf350fbaebcabcff19099ed7a56b5a400ccb0292ae5cee630dbf5db6`.

GPU0 and GPU1 were empty before the run; each lane saw exactly one device and
logical `cuda:0`. The lanes ran concurrently, while the two programs within
each lane ran strictly serially. Each lane had a distinct output directory and
distinct temporary, Hugging Face, Torch, Triton, CUDA, and Python cache roots.
Both GPUs had no compute process after completion.

The lane receipt hashes are
`63cb41d39bb8a803f6b144324197044b2db55c022855ee87a59d4dbcfe4a2303`
and
`9ec15616ee3e29b760d933efecf41de0b66e2254e38d01fc26dd3b58b90b5feb`.
Console hashes are
`4de2f5faba80f9fece3ba71a8f0aa9ebe5893b10ce1c8a0ec5cd64322f503bf1`
and
`eda065aede874d77a6e06af61a21e18923c31eb8057c5c5a8a080fca28d5bef3`.

## Superseded post-contract replacement

Status: `SUPERSEDED_BY_POST_SMOKE_REPRODUCE_PACKAGE_CHANGE`.

This was the portability gate evidence at that historical checkpoint. It was run from a fresh
`git archive` extraction after the executable-return contract fix.

- Source commit: `2f8f8127ecf80e9d1b8ed37ee2051cad0fd5f8d9`
- Source tree: `449afc2fc660d031bbeb824158c050dcd172a1b8`
- Run ID: `3d9be5e2-3e7c-47fe-885b-07176a64dc30`
- Result directory: `results/final_portability_smoke_2f8f812`
- Plan SHA-256: `4858153b88529910b3b2083369cdd75a9429caa49ebfb74f4bd6decba7cf98ff`
- Receipt SHA-256: `242585d6d85ee99420c2dd741889ea52cf7f3e2c4dad6177003d1cfe0783db28`
- Lane receipts: GPU0 `5a2de181f6bf5ea2f10b225a3c50ca071993135a10f2729d45b6dd5eea2d78aa`; GPU1 `b2e0dd90a5942851bdbc1c2bc358fa0d3795452c094b3cc4902421aafd9211bf`
- Formal metrics affected: `false`

| Lane | Program | Errors | Total seconds | Result SHA-256 |
|---|---|---:|---:|---|
| physical GPU0, logical cuda:0 | `manual_instruct` | 0 | 25.955387 | `77f3028b5a067c9adb2474d9b6b06375dc5d128921ebb4bc57122759e5779633` |
| physical GPU0, logical cuda:0 | `evolved_instruct` | 0 | 24.206094 | `837f4b61b0c459346fc9b84716eef0a79cad775157ea598c3f13ae678f1fcae3` |
| physical GPU1, logical cuda:0 | `manual_thinking` | 0 | 29.995968 | `07ac7f275672b60b29ff59435fd2e1aa403c456acb1d324e52f9bd07544746f3` |
| physical GPU1, logical cuda:0 | `evolved_thinking` | 0 | 29.766288 | `676a243d8ac25d9a80a3fe442627858545d510478315a8d2a166f3b1283fdde0` |

All four results are 1/1 correct with zero errors, and every sealed
query/question/image/output/correctness comparison is true. Each lane exposed
one CUDA device at logical `cuda:0`; the two programs per lane ran serially.
GPU0 and GPU1 both returned to 0 MiB/0 percent with no compute PID.

Each lane emitted one identical 68,376-byte Torch kernel cache artifact
(`db8b0880fa7ceec4e8ed449db3e90cd61f33425d17114cd0281b009395c4d911`).
After exact lstat and hashing, only those two cache inodes were unlinked.
All result, receipt, plan, and console bytes remain unchanged.

<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_BEGIN -->
## Active final fresh reproduction

Status: `COMPLETE`
Label: `FINAL_FRESH_REPRODUCTION_ONLY`
Run ID: `1aa9594f-47a3-4e87-b417-b3ef720adc1f`
Source commit: `0354229d892d242bcdc303ee6494bc74f576d1a1`
Source tree: `b748c76efa136323596d9cf03bd014eb58db9dd2`
Receipt: `results/final_fresh_20260826T083101Z_1aa9594f-47a3-4e87-b417-b3ef720adc1f/receipt.json`
Receipt SHA-256: `0ff86faa6bbbced5f6eff47c1443d5502cdabded2d87a7941e95fdbf1c51b32e`
Formal metrics affected: `false`
Selection affected: `false`

| Program | Role | GPU | Correct | Errors | Total seconds | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| baseline_instruct | dev128 | 0 | 51/128 | 0 | 2103.850934375 | 5208 |
| evolved_instruct | dev128 | 0 | 78/128 | 0 | 2065.330761124 | 751 |
| evolved_thinking | dev128 | 1 | 44/128 | 0 | 2717.824135764 | 33193 |
| manual_instruct | dev128 | 0 | 78/128 | 0 | 2311.063894181 | 751 |
| manual_thinking | dev128 | 1 | 23/128 | 0 | 2950.416876861 | 33541 |
| baseline_thinking | single1 | 1 | 0/1 | 0 | 29.553259263 | 128 |
| best_accuracy | single1 | 0 | 1/1 | 0 | 27.758427064 | 3 |
| best_overall | single1 | 0 | 1/1 | 0 | 24.422839248 | 3 |
| best_speed | single1 | 0 | 1/1 | 0 | 24.982773642 | 3 |
<!-- FINAL_FRESH_REPRODUCTION_ACTIVE_END -->
