"""CPU-only tests for versioned critical model snapshot validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation import model_snapshot as snapshot


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DIRECTORY = "Qwen3-VL-2B-Instruct"
REVISION = "a" * 40


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    root: Path,
    manifest: Path,
    *,
    indexed: bool = False,
    remote_code: bool = False,
) -> None:
    root.mkdir()
    contents = {
        "config.json": (
            '{"auto_map":{"AutoConfig":"configuration_test.Config"}}'
            if remote_code else "{}"
        ),
        "generation_config.json": "{}",
        "preprocessor_config.json": "{}",
        "video_preprocessor_config.json": "{}",
        "tokenizer_config.json": "{}",
        "tokenizer.json": "{}",
        "vocab.json": "{}",
        "merges.txt": "merge",
        "chat_template.json": "{}",
    }
    roles = {
        "config.json": "config",
        "generation_config.json": "config",
        "preprocessor_config.json": "processor",
        "video_preprocessor_config.json": "processor",
        "tokenizer_config.json": "tokenizer",
        "tokenizer.json": "tokenizer",
        "vocab.json": "tokenizer",
        "merges.txt": "tokenizer",
        "chat_template.json": "chat_template",
    }
    if indexed:
        contents["model-00001-of-00002.safetensors"] = "shard-one"
        contents["model-00002-of-00002.safetensors"] = "shard-two"
        contents["model.safetensors.index.json"] = json.dumps({
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            }
        })
        roles.update({
            "model-00001-of-00002.safetensors": "weight",
            "model-00002-of-00002.safetensors": "weight",
            "model.safetensors.index.json": "weight_index",
        })
        layout = "indexed_safetensors"
    else:
        contents["model.safetensors"] = "single-weight"
        roles["model.safetensors"] = "weight"
        layout = "single_safetensors"
    if remote_code:
        contents["configuration_test.py"] = "class Config: pass\n"
        roles["configuration_test.py"] = "remote_code"
    for name, payload in contents.items():
        (root / name).write_text(payload)
    files = [
        {
            "relative_path": name,
            "role": roles[name],
            "sha256": _digest(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in sorted(contents)
    ]
    runtime = sorted(contents)
    manifest.write_text(json.dumps({
        "schema_version": "critical-model-snapshot-v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "processor_revision": REVISION,
        "model_directory": DIRECTORY,
        "weight_layout": layout,
        "runtime_files": runtime,
        "remote_code_files": ["configuration_test.py"] if remote_code else [],
        "files": files,
    }, sort_keys=True, allow_nan=False))


def _environment(root: Path) -> dict[str, str]:
    return {"EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(root)}


@pytest.mark.parametrize("indexed,remote_code", [(False, False), (True, True)])
def test_snapshot_validates_single_and_indexed_complete_closures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    indexed: bool, remote_code: bool,
) -> None:
    monkeypatch.setattr(snapshot, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "snapshot"
    manifest = tmp_path / "manifest.json"
    _fixture(root, manifest, indexed=indexed, remote_code=remote_code)
    result = snapshot.validate_snapshot(
        MODEL_ID, environ=_environment(root), manifest_path=manifest,
    )
    assert result["weight_layout"] == (
        "indexed_safetensors" if indexed else "single_safetensors"
    )
    assert len(result["runtime_files"]) == (13 if indexed and remote_code else 10)


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("config.json", id="config-tamper"),
        pytest.param("tokenizer_config.json", id="tokenizer-tamper"),
        pytest.param("preprocessor_config.json", id="processor-tamper"),
    ],
)
def test_named_critical_snapshot_file_tamper_fails_closed(
    tmp_path: Path, relative: str,
) -> None:
    root = tmp_path / "snapshot"
    manifest = tmp_path / "manifest.json"
    _fixture(root, manifest)
    (root / relative).write_text('{"tampered":true}')
    with pytest.raises(snapshot.SnapshotError, match="hash/size mismatch"):
        snapshot.validate_snapshot(
            MODEL_ID, environ=_environment(root), manifest_path=manifest,
        )


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("config.json", id="missing-config"),
        pytest.param("tokenizer.json", id="missing-tokenizer"),
        pytest.param("preprocessor_config.json", id="missing-processor"),
    ],
)
def test_named_critical_snapshot_file_missing_fails_closed(
    tmp_path: Path, relative: str,
) -> None:
    root = tmp_path / "snapshot"
    manifest = tmp_path / "manifest.json"
    _fixture(root, manifest)
    (root / relative).unlink()
    with pytest.raises(snapshot.SnapshotError, match="absent/unsafe|cannot read"):
        snapshot.validate_snapshot(
            MODEL_ID, environ=_environment(root), manifest_path=manifest,
        )


def test_invalid_explicit_override_never_falls_back_to_valid_generic(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid" / DIRECTORY
    valid.parent.mkdir()
    manifest = tmp_path / "manifest.json"
    _fixture(valid, manifest)
    environment = {
        "EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(tmp_path / "absent"),
        "EVOLVING_VLM_MODEL_ROOT": str(valid.parent),
    }
    with pytest.raises(snapshot.SnapshotError, match="not a physical directory"):
        snapshot.validate_snapshot(
            MODEL_ID, environ=environment, manifest_path=manifest,
        )


def test_index_missing_shard_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    manifest = tmp_path / "manifest.json"
    _fixture(root, manifest, indexed=True)
    (root / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(snapshot.SnapshotError, match="absent/unsafe"):
        snapshot.validate_snapshot(
            MODEL_ID, environ=_environment(root), manifest_path=manifest,
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"x","schema_version":"y"}',
        '{"value":NaN}',
        "[]",
    ],
)
def test_snapshot_manifest_strict_json_rejects_duplicate_nonfinite_and_nonobject(
    tmp_path: Path, payload: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(payload)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.validate_snapshot(
            MODEL_ID,
            environ={"EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(tmp_path)},
            manifest_path=manifest,
        )



def test_dual_preflight_calls_both_in_fixed_order_and_propagates_second_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def validate(model_id: str, **kwargs: object) -> dict[str, object]:
        calls.append(model_id)
        if model_id.endswith("Thinking"):
            raise snapshot.SnapshotError("thinking snapshot invalid")
        return {"model_id": model_id}

    monkeypatch.setattr(snapshot, "validate_snapshot", validate)
    with pytest.raises(snapshot.SnapshotError, match="thinking snapshot invalid"):
        snapshot.validate_both_snapshots(environ={})
    assert calls == [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-2B-Thinking",
    ]
