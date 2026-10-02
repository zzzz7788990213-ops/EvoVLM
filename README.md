# EvoVLM

### Evolving Multimodal Inference Programs

**Evolve the inference program. Keep the model frozen.**

EvoVLM is a research project on **LLM-guided, multi-objective evolutionary
search** for vision-language inference. It searches over prompts, image
preprocessing, generation budgets, and answer extraction to improve the
accuracy–latency trade-off on scientific chart understanding, without
fine-tuning model weights.

中文：EvoVLM 在固定视觉语言模型权重的前提下，通过大模型引导的变异、实测评估和
Pareto 选择，迭代优化推理程序，在图表理解任务上探索准确率与推理效率的平衡。

[Method](#method) · [Results](#results) · [Quick start](#quick-start) ·
[Technical report](research/evolving-vlm-inference-main/report.pdf) ·
[Reproducibility](#reproducibility-and-scope)

## Method

The implementation follows the evolutionary-algorithm pattern of **mutation,
evaluation, and selection**, with a constrained program representation:

1. Start from manually designed inference programs for
   **Qwen3-VL-2B-Instruct** and **Qwen3-VL-2B-Thinking**.
2. Use a local Qwen model to rank a frozen catalog of typed, focus-specific
   mutations. A trusted renderer turns the selected configuration into Python.
3. Evaluate candidate programs on small search and selection subsets, recording
   correctness, latency, errors, and execution provenance.
4. Apply Pareto-based selection to retain useful accuracy–latency trade-offs and
   continue the next generation.
5. Export selected programs and verify their recorded evaluation evidence.

The recorded search is deliberately small: two generations, with at most six
children per model population. It is **not** unrestricted code generation,
weight training, or a claim to implement a named algorithm such as NSGA-II in
full. The implementation's distinguishing choices are constrained LLM-guided
mutations, executable inference programs, and fail-closed evidence verification.

## Results

Recorded results on **Dev128** and a separate **Speed32 timing protocol**:

| Program | Dev128 correct | Accuracy | Speed32 median latency ↓ |
| --- | ---: | ---: | ---: |
| Baseline Instruct | 51 / 128 | 39.84% | 14.54 s |
| Manual Instruct | 78 / 128 | 60.94% | 15.44 s |
| Evolved Instruct | 78 / 128 | 60.94% | 14.07 s |
| Manual Thinking | 23 / 128 | 17.97% | 20.04 s |
| Evolved Thinking | 44 / 128 | 34.38% | 17.91 s |

Relative to its manual seed, Evolved Instruct preserves the recorded accuracy
with an **8.9% lower median latency**. Evolved Thinking improves accuracy by
**16.41 percentage points** with a **10.6% lower median latency**.

Numbers come from the sealed
[metric trace](research/evolving-vlm-inference-main/report/generated/report_number_trace.json),
not from a new run made for this GitHub release. Speed32 first takes the median
across three repeats for each query, then the median across the 32 queries.
The unmodified Thinking baseline was timeout-censored at 7,200 seconds; a
completed Dev128 accuracy is not available for it.

**Interpretation matters:** Dev128 is a public validation subset, not an
untouched test set. Two exact queries overlap the evolution-used set, and eight
Dev128 queries share two figures with it. These are small-sample development
results, not a full CharXiv leaderboard result or evidence of generalization.
Latency is specific to the recorded hardware and software environment.

## Quick start

```bash
git clone https://github.com/zzzz7788990213-ops/EvoVLM.git
cd EvoVLM/research/evolving-vlm-inference-main

# CPU-only: replay recorded evidence, then run the final checks.
python3 -B -m evaluation.portable_reproduce
bash reproduce.sh check

# CPU-only: import the exported programs and verify their interface.
bash reproduce.sh smoke
```

The CPU checks do not require model weights and do not rerun inference.
The historical directory name is retained because some verification routines
bind to it. Keep `research/evolving-vlm-inference-main` intact.

**Always specify a mode:** `bash reproduce.sh` without a mode requests a fresh
GPU reproduction. It is not the CPU verification command.

### Run inference

GPU inference requires PyTorch, Transformers, the other dependencies, and the
external model snapshots. The recorded environment is documented in
[`requirements.lock.txt`](research/evolving-vlm-inference-main/requirements.lock.txt);
it includes Linux/CUDA-specific packages and is not a universal macOS lockfile.

See the [runtime guide](research/evolving-vlm-inference-main/README.md) for
pinned model downloads, local overrides, offline behavior, and example commands.
Model weights are **not** included in this repository.

## Repository layout

```text
EvoVLM/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── PUBLICATION_PROVENANCE.json
└── research/evolving-vlm-inference-main/
    ├── evolution/           # typed mutations, local provider, search, selection
    ├── evaluation/          # protocols, identity checks, semantic verification
    ├── tests/closeout/      # portability, evidence, and tamper tests
    ├── results/            # preserved experiment records and journals
    ├── data/               # fixed manifests and reproduction image assets
    ├── charxiv/            # upstream data, attribution, and selected images
    ├── best_*.py           # selected inference programs
    ├── reproduce.sh        # explicit CPU and GPU modes
    └── report.pdf          # archived technical report
```

## Reproducibility and scope

The repository includes source code, packaged data subsets, selected image
assets, and recorded evidence. **Model snapshots and the execution environment
are external dependencies.** This is not a self-contained model distribution
or the complete CharXiv image dataset.

Verification covers file hashes, critical model/processor identity, cross-bound
attempt records, and recomputation of metrics from per-query evidence. Missing
historical fields are recorded as legacy caveats rather than invented.

This public edition changes documentation and adds missing upstream license
texts; it does not change inference code, model settings, selected programs, or
recorded results. The original local archive is preserved separately. The
[publication provenance](PUBLICATION_PROVENANCE.json) identifies the parent
archive and every in-package publication change.

Historical evidence retains original run identifiers and server-path strings
for auditability. Those paths are not local setup instructions. The archived
report retains its original title and formatting; repository publication does
not imply conference submission or acceptance.

## Attribution and reuse

EvoVLM uses Qwen model families and CharXiv data. Upstream authorship and
bibliographic references are retained; their inclusion does not imply project
endorsement or collaboration. See [third-party notices](THIRD_PARTY_NOTICES.md).

No new repository-wide license is imposed by this publication. Existing
third-party licenses continue to apply to their respective materials.
