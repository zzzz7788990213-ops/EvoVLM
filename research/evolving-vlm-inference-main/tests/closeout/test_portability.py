"""CPU-only portability and fail-closed model discovery tests."""

from __future__ import annotations

import copy
import importlib
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import baseline_common
import model_locator
import manual_common
from scripts import build_portability_amendment
from scripts import final_portability_smoke
from scripts import portable_full_inference
from evaluation import portable_reproduce
from evaluation.portability_identity import (
    PROGRAMS as OPTIMIZED_PROGRAMS,
    PortabilityIdentityError,
    SEALED_IDENTITIES,
    compare_fresh_single,
    load_old_single_reference,
)
from evaluation.portable_reproduce import PortableReplayError
from evolution.target_worker import PrivateQuery
from evolution import candidate_runtime
from evolution.dsl import ProgramSpec
from model_locator import ModelLocationError, resolve_model_source

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "1" * 40
DIRECTORY = "Qwen3-VL-2B-Instruct"


def _snapshot(root: Path, payload: bytes = b"portable-weight") -> tuple[Path, str]:
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(payload)
    return root, hashlib.sha256(payload).hexdigest()


def _resolve(expected: str, environment: dict[str, str]):
    return resolve_model_source(
        model_id=MODEL_ID, revision=REVISION, model_directory=DIRECTORY,
        expected_weight_sha256=expected, environ=environment,
    )


def test_specific_override_is_hash_authenticated(tmp_path: Path) -> None:
    snapshot, digest = _snapshot(tmp_path / "snapshot")
    source = _resolve(digest, {"EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(snapshot)})
    assert source.local_path == snapshot.resolve()
    assert source.pretrained_kwargs == {
        "local_files_only": True, "trust_remote_code": False,
    }
    with pytest.raises(ModelLocationError, match="mismatch"):
        _resolve("0" * 64, {"EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(snapshot)})
    with pytest.raises(ModelLocationError, match="physical directory"):
        _resolve(digest, {"EVOLVING_VLM_INSTRUCT_MODEL_PATH": str(tmp_path / "x")})


def test_generic_package_and_official_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generic, digest = _snapshot(tmp_path / "generic" / DIRECTORY)
    packaged, _ = _snapshot(tmp_path / "package" / "models" / DIRECTORY)
    monkeypatch.setattr(model_locator, "PROJECT_ROOT", tmp_path / "package")
    assert _resolve(digest, {"EVOLVING_VLM_MODEL_ROOT": str(generic.parent)}).local_path == generic.resolve()
    assert _resolve(digest, {}).local_path == packaged.resolve()
    monkeypatch.setattr(model_locator, "PROJECT_ROOT", tmp_path / "empty")
    official = _resolve("a" * 64, {})
    assert official.reference == MODEL_ID and official.local_path is None
    assert official.pretrained_kwargs == {
        "local_files_only": False, "trust_remote_code": False,
        "revision": REVISION,
    }


def test_external_image_and_private_query_are_not_project_contained(
    tmp_path: Path,
) -> None:
    image = tmp_path / "external.jpg"
    image.write_bytes(b"image-bytes")
    assert model_locator.resolve_readable_image(str(image)) == image.resolve()
    query = PrivateQuery(
        request_id="portable-1", query_key="q-1", figure_id="figure", qid="qid",
        question="question?", image_path=image, expected_answer="answer",
    )
    assert query.to_worker_mapping()["image_sha256"] == hashlib.sha256(b"image-bytes").hexdigest()


def test_unsafe_image_references_fail_closed(tmp_path: Path) -> None:
    for value in ("https://example.invalid/image.jpg", str(tmp_path / "missing")):
        with pytest.raises(ValueError):
            model_locator.resolve_readable_image(value)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"x")
    link = tmp_path / "link.jpg"
    link.symlink_to(image)
    with pytest.raises(ValueError):
        model_locator.resolve_readable_image(str(link))


def test_process_safety_runtime_root_can_be_relocated(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.update(
        EVOLVING_VLM_RUNTIME_ROOT=str(tmp_path), PYTHONDONTWRITEBYTECODE="1"
    )
    code = (
        "from evolution import process_safety as p; from pathlib import Path;"
        f"assert p.EVO_ROOT == Path({str(tmp_path.resolve())!r});"
        f"assert p.PROJECT_ROOT == Path({str(model_locator.PROJECT_ROOT.resolve())!r})"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=model_locator.PROJECT_ROOT,
        env=environment, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def _formal_inputs() -> tuple[dict, dict, dict, Path]:
    final = portable_reproduce.load_json(portable_reproduce.FINAL_MANIFEST)
    plan_path = portable_reproduce._record_hash(
        final["composition"]["retained_campaign_plan"], "old plan"
    )
    plan = portable_reproduce.load_json(plan_path)
    entry = plan["schedule"][0]
    ground_truth = portable_reproduce.load_json(portable_reproduce.GT_PATH)
    result_path = (
        portable_reproduce.CAMPAIGN_ROOT
        / portable_reproduce.OLD_CAMPAIGN_ID
        / entry["result"]
    )
    return plan, entry, ground_truth, result_path


def test_portable_replay_recomputes_all_formal_results() -> None:
    result = portable_reproduce.replay(verify_closure=False)
    assert result["status"] == "COMPLETE"
    assert result["formal_complete"] is True
    assert result["result_count"] == 26
    assert result["best"]["best_accuracy"] == "evolved_instruct"

def test_package_mode_uses_amendment_aware_raw_replay() -> None:
    source = (model_locator.PROJECT_ROOT / "reproduce.sh").read_text(encoding="utf-8")
    body = source.split("final_check() {", 1)[1].split("\n}", 1)[0]
    assert "evidence_replay" in body
    assert "evaluation.recovery_finalize" not in body
    package = source.split("  package)", 1)[1].split("    ;;", 1)[0]
    assert "final_check" in package



def test_portable_replay_rejects_tampered_raw_result(tmp_path: Path) -> None:
    plan, entry, ground_truth, result_path = _formal_inputs()
    document = portable_reproduce.load_json(result_path)
    document["evaluation"]["records"][0]["prediction"] = "tampered"
    tampered = tmp_path / "tampered-result.json"
    tampered.write_text(
        __import__("json").dumps(document, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(PortableReplayError):
        portable_reproduce.verify_result(
            tampered,
            portable_reproduce.sha256_file(tampered),
            plan,
            entry,
            ground_truth,
            portable_reproduce._hash_index(portable_reproduce.load_json(portable_reproduce.FINAL_MANIFEST)),
        )


def test_portable_replay_rejects_tampered_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, entry, ground_truth, result_path = _formal_inputs()
    original_safe_file = portable_reproduce.safe_file
    manifest = portable_reproduce.load_json(
        portable_reproduce.MANIFEST_ROOT / f"{entry['manifest']}.json"
    )
    manifest["query_keys"] = list(reversed(manifest["query_keys"]))
    tampered = tmp_path / "tampered-manifest.json"
    tampered.write_text(
        __import__("json").dumps(manifest, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    def substitute(relative: str) -> Path:
        if relative == f"data/phase3/manifests/{entry['manifest']}.json":
            return tampered
        return original_safe_file(relative)

    monkeypatch.setattr(portable_reproduce, "safe_file", substitute)
    with pytest.raises(PortableReplayError, match="manifest hash differs"):
        portable_reproduce.verify_result(
            result_path,
            portable_reproduce.sha256_file(result_path),
            plan,
            entry,
            ground_truth,
            portable_reproduce._hash_index(portable_reproduce.load_json(portable_reproduce.FINAL_MANIFEST)),
        )


def test_portable_replay_rejects_tampered_sealed_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load_json = portable_reproduce.load_json
    final = original_load_json(portable_reproduce.FINAL_MANIFEST)
    summary_path = portable_reproduce._record_hash(final["campaign_summary"], "summary")

    def substitute(path: Path) -> dict:
        value = original_load_json(path)
        if path == summary_path:
            value = copy.deepcopy(value)
            value["aggregation"]["best"]["best_accuracy"] = "manual_instruct"
        return value

    monkeypatch.setattr(portable_reproduce, "load_json", substitute)
    with pytest.raises(PortableReplayError, match="sealed aggregation differs"):
        portable_reproduce.replay(verify_closure=False)


def _full_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str,
) -> tuple[object, Path]:
    runtime = tmp_path / "managed"
    runtime.mkdir()
    data = tmp_path / "descriptive_val.json"
    data.write_text("{}", encoding="utf-8")
    images = tmp_path / "external-images"
    images.mkdir()
    monkeypatch.setenv("EVOLVING_VLM_RUNTIME_ROOT", str(runtime))
    args = portable_full_inference.build_parser().parse_args(
        [
            "--output", "fresh", "--data", str(data), "--images", str(images),
            *extra,
        ]
    )
    return args, runtime


def test_portable_full_inference_dry_run_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    args, runtime = _full_args(tmp_path, monkeypatch, "--dry-run")
    configuration = portable_full_inference.prepare(args)
    portable_full_inference.run(configuration)
    assert not (runtime / "fresh").exists()
    assert "preflight PASS" in capsys.readouterr().out


def test_portable_full_inference_rejects_unsafe_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, runtime = _full_args(tmp_path, monkeypatch, "--dry-run")
    args.output = str(tmp_path / "outside")
    with pytest.raises(portable_full_inference.FullInferenceError, match="managed"):
        portable_full_inference.prepare(args)
    existing = runtime / "existing"
    existing.mkdir()
    args.output = str(existing)
    with pytest.raises(portable_full_inference.FullInferenceError, match="exists"):
        portable_full_inference.prepare(args)


def test_portable_full_inference_requires_single_explicit_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _ = _full_args(tmp_path, monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(portable_full_inference.FullInferenceError, match="exactly one"):
        portable_full_inference.prepare(args)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(portable_full_inference.FullInferenceError, match="exactly one"):
        portable_full_inference.prepare(args)


def test_portable_full_inference_runs_fixed_program_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, runtime = _full_args(tmp_path, monkeypatch, "--limit", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(portable_full_inference.subprocess, "run", fake_run)
    portable_full_inference.run(portable_full_inference.prepare(args))
    assert (runtime / "fresh").is_dir()
    assert [command[3] for command in commands] == list(portable_full_inference.PROGRAMS)
    assert all("--limit" in command and command[command.index("--limit") + 1] == "1" for command in commands)


def test_amendment_builder_detects_changes_and_writes_no_replace(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal.txt"
    formal.write_text("before\n", encoding="utf-8")
    changed = tmp_path / "changed.txt"
    changed.write_text("old\n", encoding="utf-8")
    added = tmp_path / "added.py"
    added.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = tmp_path / "final.json"
    records = [
        {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        for path in (formal, changed)
    ]
    manifest.write_text(__import__("json").dumps({"artifacts": records}), encoding="utf-8")
    changed.write_text("new\n", encoding="utf-8")
    document = build_portability_amendment.build_document(
        tmp_path, manifest, ("formal.txt", "changed.txt", "added.py")
    )
    assert [item["path"] for item in document["changed_artifacts"]] == ["changed.txt"]
    assert [item["path"] for item in document["added_artifacts"]] == ["added.py"]
    output = tmp_path / "amendment.json"
    build_portability_amendment.write_no_replace(output, document, tmp_path)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        build_portability_amendment.write_no_replace(output, document, tmp_path)
    assert output.read_bytes() == before


def test_amendment_verifier_rehashes_portable_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = tmp_path / "formal.txt"
    formal.write_text("sealed\n", encoding="utf-8")
    added = tmp_path / "added.py"
    added.write_text("VALUE = 1\n", encoding="utf-8")
    final = tmp_path / "final.json"
    final.write_text(__import__("json").dumps({"artifacts": [{
        "path": "formal.txt", "sha256": hashlib.sha256(formal.read_bytes()).hexdigest(),
        "size_bytes": formal.stat().st_size,
    }]}), encoding="utf-8")
    amendment = tmp_path / "amendment.json"
    document = build_portability_amendment.build_document(
        tmp_path, final, ("formal.txt", "added.py")
    )
    amendment.write_text(__import__("json").dumps(document), encoding="utf-8")
    monkeypatch.setattr(portable_reproduce, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(portable_reproduce, "FINAL_MANIFEST", final)
    monkeypatch.setattr(portable_reproduce, "AMENDMENT_MANIFEST", amendment)
    monkeypatch.setattr(portable_reproduce, "PORTABLE_EXECUTABLE_CLOSURE", ("formal.txt", "added.py"))
    assert portable_reproduce.verify_amended_closure(portable_reproduce.load_json(final))["portable_executable_closure"] == 2
    added.write_text("TAMPERED = 1\n", encoding="utf-8")
    with pytest.raises(PortableReplayError, match="added artifact mismatch"):
        portable_reproduce.verify_amended_closure(portable_reproduce.load_json(final))


def test_portable_executable_closure_has_no_server_absolute_path() -> None:
    forbidden = b"/scratch/Projects/CFP-04/CFP04-SF-122/xin.w24/fmwc/zinuo/Evo"
    for relative in portable_reproduce.PORTABLE_EXECUTABLE_CLOSURE:
        path = portable_reproduce.PROJECT_ROOT / relative
        assert path.is_file() and not path.is_symlink(), relative
        assert forbidden not in path.read_bytes(), relative


class _ProcessorModelSpy:
    def __init__(self) -> None:
        self.chat_call = None
        self.processor_call = None
        self.generate_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_call = (copy.deepcopy(messages), copy.deepcopy(kwargs))
        return "RENDERED"

    def __call__(self, **kwargs):
        self.processor_call = copy.deepcopy(kwargs)
        return "INPUTS"

    def generate(self, **kwargs):
        self.generate_call = copy.deepcopy(kwargs)
        return "OUTPUT"


def test_fake_processor_model_spy_matches_sealed_manual_contracts() -> None:
    for program, spec in (
        ("manual_instruct", manual_common.INSTRUCT_SPEC),
        ("manual_thinking", manual_common.THINKING_SPEC),
    ):
        sealed = SEALED_IDENTITIES[program]
        spy = _ProcessorModelSpy()
        messages = manual_common._manual_messages("IMAGE", sealed["prompt"])
        spy.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = manual_common._manual_processor_kwargs(
            "RENDERED", ["IMAGE_INPUT"], ["VIDEO_INPUT"]
        )
        spy(**processor_kwargs)
        stopper = "STOPPER"
        generation_kwargs = manual_common._manual_generation_kwargs(
            spec, stopper
        )
        spy.generate(input_ids="IDS", **generation_kwargs)
        assert spy.chat_call == (
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "IMAGE"},
                    {"type": "text", "text": sealed["prompt"]},
                ],
            }],
            {"tokenize": False, "add_generation_prompt": True},
        )
        assert spy.processor_call == {
            "text": ["RENDERED"], "images": ["IMAGE_INPUT"],
            "videos": ["VIDEO_INPUT"], "padding": True,
            "return_tensors": "pt",
        }
        assert spy.generate_call == {
            "input_ids": "IDS", "max_new_tokens": sealed["max_new_tokens"],
            "do_sample": False, "num_beams": 1, "use_cache": True,
            "stopping_criteria": stopper,
        }


def test_fake_processor_model_spy_matches_sealed_evolved_contracts() -> None:
    for program in ("evolved_instruct", "evolved_thinking"):
        module = importlib.import_module(program)
        spec = ProgramSpec(**dict(module.CANDIDATE_SPEC))
        sealed = SEALED_IDENTITIES[program]
        spy = _ProcessorModelSpy()
        messages = candidate_runtime._candidate_messages(
            spec, "IMAGE", "What value is shown?"
        )
        spy.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = candidate_runtime._candidate_processor_kwargs(
            "RENDERED", ["IMAGE_INPUT"], ["VIDEO_INPUT"]
        )
        spy(**processor_kwargs)
        stopper = "STOPPER"
        generation_kwargs = candidate_runtime._candidate_generation_kwargs(
            spec, stopper
        )
        spy.generate(input_ids="IDS", **generation_kwargs)
        assert messages[-1]["content"][-1]["text"] == sealed["prompt"]
        assert spy.processor_call == {
            "text": ["RENDERED"], "images": ["IMAGE_INPUT"],
            "videos": ["VIDEO_INPUT"], "padding": True,
            "return_tensors": "pt",
        }
        assert spy.generate_call == {
            "input_ids": "IDS", "max_new_tokens": sealed["max_new_tokens"],
            "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
            "use_cache": True, "stopping_criteria": stopper,
        }


def test_four_algorithms_and_old_single_references_are_sealed() -> None:
    expected_sources = {
        "manual_instruct": "397f3a92d21f2dac64e5bf012ccdeca6e2c48fcfed54c11985268df48cf516f4",
        "manual_thinking": "60db13925030b2e346f363ab4a96eb6317641e0e964e4dd7519f7b2f4073fb58",
        "evolved_instruct": "1c58d489733b7031c7bed3d7a27a9840cf364a9c916cd6eab54febd4b43bb984",
        "evolved_thinking": "81bedc0e16a3954bc55aea73120ed47f661534a79f01ff9dfd66069c2af1efe6",
    }
    for program in OPTIMIZED_PROGRAMS:
        reference = load_old_single_reference(program)
        assert reference["source"]["sha256"] == expected_sources[program]
        assert reference["query_key"] == "0_0"
        assert reference["prediction_sha256"] == hashlib.sha256(
            b"60"
        ).hexdigest()
        assert reference["correct"] is True
        assert len(reference["algorithm_identity_sha256"]) == 64
        assert reference["current_dependencies"]


def test_old_single_reference_revalidates_explicit_image(
    tmp_path: Path,
) -> None:
    image = model_locator.PROJECT_ROOT / "charxiv/images/0.jpg"
    reference = load_old_single_reference(
        "manual_instruct", image_path=image,
    )
    assert reference["image_sha256"] == hashlib.sha256(
        image.read_bytes()
    ).hexdigest()
    altered = tmp_path / "0.jpg"
    altered.write_bytes(b"not-the-sealed-image")
    with pytest.raises(PortabilityIdentityError, match="image hash"):
        load_old_single_reference("manual_instruct", image_path=altered)


def test_fresh_single_comparison_rejects_any_output_drift(
    tmp_path: Path,
) -> None:
    reference = load_old_single_reference("manual_instruct")
    figure = str(reference["query_key"]).rsplit("_", 1)[0]
    document = {
        "program": "manual_instruct",
        "num_evaluated": 1,
        "records": [{
            "query_key": reference["query_key"],
            "question": reference["question"],
            "image_path": str(
                model_locator.PROJECT_ROOT
                / "charxiv/images"
                / f"{figure}.jpg"
            ),
            "response": reference["prediction"],
            "correct": reference["correct"],
        }],
    }
    valid = tmp_path / "valid.json"
    valid.write_text(
        __import__("json").dumps(document, allow_nan=False),
        encoding="utf-8",
    )
    comparison = compare_fresh_single("manual_instruct", valid)
    assert all(comparison["comparison"].values())
    assert comparison["old_result"]["sha256"] == (
        "fba0d247e6943bb84d06874524ed4b87e7113c1f13e2bacea9806f79a672e2b5"
    )
    document["records"][0]["response"] = "different"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        __import__("json").dumps(document, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(PortabilityIdentityError, match="output"):
        compare_fresh_single("manual_instruct", tampered)


def test_final_portability_smoke_dry_run_is_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("EVOLVING_VLM_RUNTIME_ROOT", str(runtime))
    monkeypatch.setattr(
        final_portability_smoke, "_sealed_input_records", lambda images: [],
    )
    assert final_portability_smoke.main([
        "--output", "dry-run", "--dry-run",
    ]) == 0
    assert not (runtime / "dry-run").exists()
    assert final_portability_smoke.LABEL in capsys.readouterr().out


def test_final_portability_lane_is_fixed_and_sequential_without_real_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "lane"
    output.mkdir()
    data = tmp_path / "data.json"
    data.write_text("{}", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setattr(
        final_portability_smoke, "compare_fresh_single",
        lambda program, path: {"program": program, "path": path.name},
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls: list[str] = []

    def fake_runner(command, **kwargs):
        program = command[3]
        calls.append(program)
        result_path = Path(command[command.index("--output") + 1])
        result_path.write_text(__import__("json").dumps({
            "program": program,
            "num_evaluated": 1,
            "num_errors": 0,
            "total_time": 1.0,
            "gpu_scope": {
                "physical_gpu_index": 0,
                "cuda_visible_devices": "0",
                "visible_device_count": 1,
                "logical_device": "cuda:0",
            },
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    receipt = final_portability_smoke.run_lane(
        output=output, lane=0, data=data, images=images, runner=fake_runner,
    )
    assert calls == ["manual_instruct", "evolved_instruct"]
    assert receipt["label"] == final_portability_smoke.LABEL
    assert receipt["formal_metrics_affected"] is False
    assert (output / "lane0" / "receipt.json").is_file()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(final_portability_smoke.PortabilitySmokeError, match="lane"):
        final_portability_smoke.run_lane(
            output=output, lane=0, data=data, images=images, runner=fake_runner,
        )


def test_final_portability_smoke_module_entrypoint_is_importable_cpu_only() -> None:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable, "-B", "-m", "scripts.final_portability_smoke",
            "--help",
        ],
        cwd=model_locator.PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_final_portability_lane_environments_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    lane0 = final_portability_smoke._lane_environment(output, 0)
    lane1 = final_portability_smoke._lane_environment(output, 1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert lane0["CUDA_VISIBLE_DEVICES"] == "0"
    assert lane1["CUDA_VISIBLE_DEVICES"] == "1"
    for key in (
        "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE",
        "TORCH_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "PYTHONPYCACHEPREFIX",
    ):
        assert lane0[key] != lane1[key]
        assert Path(lane0[key]).is_relative_to(output)
        assert Path(lane1[key]).is_relative_to(output)



def test_baseline_structured_identity_and_request_spy() -> None:
    """Freeze baseline semantics while allowing only path plumbing changes."""
    assert "torch" not in sys.modules
    assert baseline_common.DTYPE == "bfloat16"
    assert baseline_common.QUANTIZATION == "none"
    assert baseline_common.ATTENTION_IMPLEMENTATION == "sdpa"
    assert baseline_common.MAX_NEW_TOKENS == 128
    assert baseline_common.BATCH_SIZE == 1
    assert baseline_common.EFFECTIVE_GENERATION_CONFIG == {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "typical_p": None,
        "max_new_tokens": 128,
        "use_cache": True,
        "use_model_defaults": False,
    }
    marker = object()
    assert baseline_common._baseline_messages(marker, "question?") == [{
        "role": "user",
        "content": [
            {"type": "image", "image": marker},
            {"type": "text", "text": "question?"},
        ],
    }]
    assert baseline_common._baseline_processor_kwargs("chat", [marker], None) == {
        "text": ["chat"],
        "images": [marker],
        "videos": None,
        "padding": True,
        "return_tensors": "pt",
    }
    assert "torch" not in sys.modules


def _first_attempt_inputs() -> tuple[dict, dict, dict, dict, str]:
    final = portable_reproduce.load_json(portable_reproduce.FINAL_MANIFEST)
    plan_path = portable_reproduce._record_hash(
        final["composition"]["retained_campaign_plan"], "old plan"
    )
    plan = portable_reproduce.load_json(plan_path)
    entry = plan["schedule"][0]
    result_path = (
        portable_reproduce.CAMPAIGN_ROOT
        / portable_reproduce.OLD_CAMPAIGN_ID
        / entry["result"]
    )
    result = portable_reproduce.load_json(result_path)
    return final, plan, entry, result, portable_reproduce.OLD_CAMPAIGN_ID


@pytest.mark.parametrize(
    ("artifact", "field", "replacement", "error"),
    [
        ("start", "attempt_id", "wrong-attempt", "attempt ID cross-binding"),
        ("start", "program", "wrong-program", "attempt program differs"),
        ("start", "repeat", 999, "attempt repeat differs"),
        ("end", "program", "wrong-program", "attempt program differs"),
        ("end", "returncode", 1, "did not exit successfully"),
        ("end", "outcome", "FAILED", "retained attempt outcome"),
        ("end", "console_sha256", "0" * 64, "console hash differs"),
    ],
)
def test_attempt_cross_binding_rejects_semantic_tamper(
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    replacement: object,
    error: str,
) -> None:
    final, _plan, entry, result, campaign_id = _first_attempt_inputs()
    index = portable_reproduce._hash_index(final)
    relative = portable_reproduce._selected_attempt_paths(index, campaign_id, entry)[artifact]
    target = portable_reproduce.safe_file(relative)
    original_load_json = portable_reproduce.load_json

    def substitute(path: Path) -> dict:
        document = original_load_json(path)
        if path == target:
            document = copy.deepcopy(document)
            document[field] = replacement
        return document

    monkeypatch.setattr(portable_reproduce, "load_json", substitute)
    with pytest.raises(PortableReplayError, match=error):
        portable_reproduce.verify_attempt_closure(
            final, entry, result, campaign_id
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("run_id", "wrong-run", "identity cross-binding"),
        ("model_id", "wrong-model", "model binding differs"),
        ("physical_gpu_index", 1, "GPU binding differs"),
    ],
)
def test_runtime_event_cross_binding_rejects_semantic_tamper(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    error: str,
) -> None:
    final, _plan, entry, result, campaign_id = _first_attempt_inputs()
    target = portable_reproduce.safe_file(str(result["runtime_event_path"]))
    original_load_json = portable_reproduce.load_json

    def substitute(path: Path) -> dict:
        document = original_load_json(path)
        if path == target:
            document = copy.deepcopy(document)
            document[field] = replacement
        return document

    monkeypatch.setattr(portable_reproduce, "load_json", substitute)
    with pytest.raises(PortableReplayError, match=error):
        portable_reproduce.verify_attempt_closure(
            final, entry, result, campaign_id
        )




@pytest.mark.parametrize(
    "mutation",
    ["delete_start", "delete_end", "orphan_end", "extra_success", "swap_end"],
)
def test_formal_attempt_manifest_enumeration_fails_closed(mutation: str) -> None:
    entry = {"index": 1, "attempt_prefix": "run01.attempt-abc"}
    prefix = (
        "results/phase3_recovery_v2/campaigns/"
        + portable_reproduce.OLD_CAMPAIGN_ID
        + "/run01.attempt-abc."
    )
    index = {
        prefix + "start.json": {},
        prefix + "end.json": {},
        prefix + "console.log": {},
    }
    if mutation == "delete_start":
        del index[prefix + "start.json"]
    elif mutation == "delete_end":
        del index[prefix + "end.json"]
    elif mutation == "orphan_end":
        index[prefix + "orphan.end.json"] = {}
    elif mutation == "extra_success":
        index[prefix + "retry.end.json"] = {}
    else:
        del index[prefix + "end.json"]
        index[prefix + "wrong.end.json"] = {}
    with pytest.raises(PortableReplayError, match="artifact set differs|share an identity"):
        portable_reproduce._selected_attempt_paths(
            index, portable_reproduce.OLD_CAMPAIGN_ID, entry
        )


def test_formal_journal_attempt_id_cross_binding_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, entry, ground_truth, result_path = _formal_inputs()
    result = portable_reproduce.load_json(result_path)
    journal = (
        portable_reproduce.PROJECT_ROOT
        / str(result["journal_directory"])
        / "000.json"
    )
    original_load_json = portable_reproduce.load_json

    def substitute(path: Path) -> dict:
        document = original_load_json(path)
        if path == journal:
            document = copy.deepcopy(document)
            document["attempt_id"] = str(__import__("uuid").uuid4())
        return document

    monkeypatch.setattr(portable_reproduce, "load_json", substitute)
    final = original_load_json(portable_reproduce.FINAL_MANIFEST)
    with pytest.raises(PortableReplayError, match="journal/result attempt ID"):
        portable_reproduce.verify_result(
            result_path,
            portable_reproduce.sha256_file(result_path),
            plan,
            entry,
            ground_truth,
            portable_reproduce._hash_index(final),
        )
