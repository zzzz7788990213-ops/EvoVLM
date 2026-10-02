# EvoVLM research package

This directory contains the recorded EvoVLM implementation, evaluation data,
evidence, and technical report. Start with the [project overview](../../README.md).
This public runtime guide replaces the original distribution instructions;
inference code and recorded experiment artifacts are unchanged.

## CPU-only verification

Run these commands from this directory, retaining its name
`evolving-vlm-inference-main`:

```bash
python3 -B -m evaluation.portable_reproduce
bash reproduce.sh evidence
bash reproduce.sh check
bash reproduce.sh smoke
```

These commands verify recorded evidence or program interfaces. They do not
generate new predictions. The standalone evidence replay uses Python's standard
library. `pytest` is required to run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/closeout
```

Always supply an explicit mode to `reproduce.sh`: its no-argument default is
`fresh`, which requests GPU inference.

## External model dependencies

**Neither the GitHub repository nor the source archive includes model weights.**
Code, packaged data subsets, and evidence are provided; authenticated model
snapshots and an appropriate execution environment must be obtained separately.

The fixed model IDs are:

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-2B-Thinking`

Their pinned revisions and critical file identities are recorded in:

- `evaluation/model_snapshots/qwen3_vl_2b_instruct.critical.v1.json`
- `evaluation/model_snapshots/qwen3_vl_2b_thinking.critical.v1.json`

Obtain the snapshots from the official Hugging Face model repositories at the
exact recorded revisions. Do not substitute another revision with the same
display name. For example, with `huggingface_hub` installed, download all files
into an external model directory:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

model_root = Path("/absolute/path/to/model-root")
for manifest_path in sorted(Path("evaluation/model_snapshots").glob("*.json")):
    manifest = json.loads(manifest_path.read_text())
    snapshot_download(
        repo_id=manifest["model_id"],
        revision=manifest["revision"],
        local_dir=model_root / manifest["model_id"].split("/")[-1],
    )
```

### Local overrides

`model_locator.py` recognizes exactly these three model-location variables:

| Variable | Meaning |
| --- | --- |
| `EVOLVING_VLM_INSTRUCT_MODEL_PATH` | Exact physical Instruct snapshot directory |
| `EVOLVING_VLM_THINKING_MODEL_PATH` | Exact physical Thinking snapshot directory |
| `EVOLVING_VLM_MODEL_ROOT` | Parent containing `Qwen3-VL-2B-Instruct/` and `Qwen3-VL-2B-Thinking/` |

Resolution order is the per-model override, then the shared model root, then
package-local `models/<snapshot-name>/`, then the official model ID at its locked
revision. An explicitly configured local path is authoritative: an invalid path
or identity fails closed instead of falling back to a different source. Model
and processor use the same resolved source. Keep snapshots as physical files;
symlinked snapshot paths and critical files are rejected by verification.

### Offline and online execution

Minimal offline inference, after installing the inference dependencies:

```bash
export EVOLVING_VLM_MODEL_ROOT=/absolute/path/to/model-root
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
python3 evaluate.py manual_instruct
```

Minimal online inference, allowing the initial pinned model download:

```bash
unset EVOLVING_VLM_INSTRUCT_MODEL_PATH
unset EVOLVING_VLM_THINKING_MODEL_PATH
unset EVOLVING_VLM_MODEL_ROOT
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE
python3 evaluate.py manual_instruct
```

Online mode needs network access and a writable Hugging Face cache. A
package-local snapshot, if present, still takes precedence. The general
dependencies are in `requirements.txt`; the recorded Linux/CUDA environment is
in `requirements.lock.txt`. The latter is not a cross-platform lockfile.

Fresh reproduction requires **both local snapshots**, including their config,
generation config, tokenizer, processor, and chat-template files. The launcher
checks the critical manifests before creating formal run outputs, starting
lanes, or initializing CUDA. An offline launch without valid snapshots exits
with an error. CPU evidence verification does not require these snapshots.

## Evidence and publication provenance

The experimental evidence, raw records, model manifests, and `best_*` programs
are preserved from the validated archive. `PACKAGE_MANIFEST.json` describes
this public edition. Its source metadata and the root
[`PUBLICATION_PROVENANCE.json`](../../PUBLICATION_PROVENANCE.json) distinguish the
public documentation/license additions from the sealed experimental records.

The [archived technical report](report.pdf) documents the method, evaluation
protocol, and limitations. The public Dev128 subset is not an untouched holdout;
do not present its development accuracy as a full CharXiv benchmark score.

Upstream attribution and licensing are described in
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
