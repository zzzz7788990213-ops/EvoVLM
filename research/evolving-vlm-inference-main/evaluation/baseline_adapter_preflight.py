"""No-weight adapter preflight for the real baseline inference entry."""

from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

import torch

from baseline_common import BaselineBackendAdapter, make_inference
from evaluation.active_contract import canonical_run_paths
from evaluation.active_paths import ACTIVE_MINOR, CAMPAIGN_ROOT
from evaluation.atomic_artifacts import write_json_noreplace
from evaluation.provenance import EVO_ROOT, PROJECT_ROOT, load_json_strict, sha256_file


class GenerateBoundaryReached(RuntimeError):
    """Expected sentinel proving the adapter reached the generate boundary."""


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=[[]])
        self.logical_device = None

    def to(self, device: str):
        if device != "cuda:0":
            raise AssertionError("adapter input moved to unexpected logical device")
        self.logical_device = device
        return self


class _FakeModel:
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state
        self.generation_config = SimpleNamespace(
            do_sample=False,
            num_beams=1,
            eos_token_id=0,
        )
        self._parameter = SimpleNamespace(device=torch.device("cuda:0"))

    def eval(self):
        self.state["eval_calls"] += 1
        return self

    def parameters(self):
        yield self._parameter

    def generate(self, **kwargs):
        self.state["generate_calls"] += 1
        if kwargs["generation_config"].max_new_tokens != 128:
            raise AssertionError("adapter saw changed max_new_tokens")
        if kwargs["generation_config"].do_sample:
            raise AssertionError("adapter saw non-greedy generation")
        raise GenerateBoundaryReached("generate boundary reached without model weights")


class _FakeProcessor:
    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    def apply_chat_template(self, messages, **kwargs):
        self.state["chat_template_calls"] += 1
        if kwargs != {"tokenize": False, "add_generation_prompt": True}:
            raise AssertionError("adapter saw changed chat-template arguments")
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("adapter saw changed baseline message shape")
        return "adapter-preflight-prompt"

    def __call__(self, **kwargs):
        self.state["processor_calls"] += 1
        if kwargs["return_tensors"] != "pt" or kwargs["padding"] is not True:
            raise AssertionError("adapter saw changed processor arguments")
        return _Inputs()


def _state() -> dict[str, int]:
    return {
        "model_loader_calls": 0,
        "processor_loader_calls": 0,
        "vision_calls": 0,
        "eval_calls": 0,
        "chat_template_calls": 0,
        "processor_calls": 0,
        "generate_calls": 0,
        "model_weight_file_reads": 0,
    }


def _backend(state: dict[str, int]) -> BaselineBackendAdapter:
    model = _FakeModel(state)
    processor = _FakeProcessor(state)

    def load_model(model_dir: Path):
        if not model_dir.is_dir():
            raise AssertionError("model directory missing")
        state["model_loader_calls"] += 1
        return model

    def load_processor(model_dir: Path):
        if not model_dir.is_dir():
            raise AssertionError("processor directory missing")
        state["processor_loader_calls"] += 1
        return processor

    def process(messages):
        state["vision_calls"] += 1
        return [messages[0]["content"][0]["image"]], None

    return BaselineBackendAdapter(
        load_model=load_model,
        load_processor=load_processor,
        process_vision_info=process,
        visible_device_count=lambda: 1,
        device_name=lambda index: "adapter-no-weight-device" if index == 0 else "",
    )


def _set_runtime_environment(run_id: str, runtime_id: str, event_path: Path):
    keys = ("EVO_RUN_ID", "EVO_RUNTIME_EVENT_ID", "EVO_RUNTIME_EVENT_PATH")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["EVO_RUN_ID"] = run_id
    os.environ["EVO_RUNTIME_EVENT_ID"] = runtime_id
    os.environ["EVO_RUNTIME_EVENT_PATH"] = str(event_path)
    return previous


def _restore_runtime_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _inference(program, state: dict[str, int]):
    return make_inference(
        program.MODEL_ID,
        program.MODEL_REVISION,
        program.MODEL_ID.rsplit("/", 1)[1],
        program.MODEL_WEIGHT_SHA256,
        backend_adapter=_backend(state),
    )


def _run_program_preflight(program_name: str) -> dict[str, object]:
    order_by_program = {"baseline_instruct": 1, "baseline_thinking": 4}
    try:
        order = order_by_program[program_name]
    except KeyError as exc:
        raise ValueError("adapter preflight program is not canonical") from exc
    active_campaign_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    )
    run_id = str(uuid4())
    runtime_id = str(uuid4())
    paths = canonical_run_paths(
        active_campaign_id,
        program_name,
        "single_1",
        1,
        run_id,
        runtime_id,
        order,
    )
    campaign = PROJECT_ROOT / paths["campaign_path"]
    campaign.parent.mkdir(parents=True, exist_ok=True)
    campaign.mkdir(mode=0o700, exist_ok=False)
    event_path = PROJECT_ROOT / paths["runtime_event_path"]
    image_path = PROJECT_ROOT / "charxiv/images/75.jpg"

    state = _state()
    previous = _set_runtime_environment(run_id, runtime_id, event_path)
    original_count = torch.cuda.device_count
    original_name = torch.cuda.get_device_name

    def forbidden_cuda_call(*args, **kwargs):
        raise AssertionError("adapter preflight invoked a real CUDA query")

    torch.cuda.device_count = forbidden_cuda_call
    torch.cuda.get_device_name = forbidden_cuda_call
    boundary_reached = False
    event_document = None
    event_hash = None
    try:
        program = importlib.import_module(program_name)
        inference = _inference(program, state)
        try:
            inference(str(image_path), "Adapter preflight question.")
        except GenerateBoundaryReached:
            boundary_reached = True
        if not boundary_reached:
            raise AssertionError("valid adapter did not reach generate boundary")
        event_document = load_json_strict(event_path)
        event_hash = sha256_file(event_path)
        if inference.runtime_event_path != str(event_path):
            raise AssertionError("baseline did not expose canonical runtime event")
    finally:
        torch.cuda.device_count = original_count
        torch.cuda.get_device_name = original_name
        _restore_runtime_environment(previous)

    legacy_state = _state()
    legacy_tag = f"v4_{ACTIVE_MINOR - 1}"
    legacy_rejected = False
    legacy_generate_calls = None
    with TemporaryDirectory(
        prefix="baseline-adapter-preflight-",
        dir=EVO_ROOT / ".tmp",
    ) as temporary:
        legacy_campaign = (
            Path(temporary)
            / "results/evaluations"
            / legacy_tag
            / "campaigns"
            / active_campaign_id
        )
        legacy_campaign.mkdir(parents=True)
        legacy_runtime_id = str(uuid4())
        legacy_run_id = str(uuid4())
        legacy_event = legacy_campaign / f"runtime_event.{legacy_runtime_id}.json"
        previous = _set_runtime_environment(
            legacy_run_id,
            legacy_runtime_id,
            legacy_event,
        )
        try:
            rejected = _inference(program, legacy_state)
            try:
                rejected(str(image_path), "Adapter legacy-path rejection question.")
            except RuntimeError as exc:
                legacy_rejected = "predeclared active run artifact" in str(exc)
            if legacy_event.exists() or legacy_event.is_symlink():
                raise AssertionError("legacy runtime event was published")
            legacy_generate_calls = legacy_state["generate_calls"]
        finally:
            _restore_runtime_environment(previous)

    if not legacy_rejected or legacy_generate_calls != 0:
        raise AssertionError("legacy runtime path did not fail before generate")
    if state["model_weight_file_reads"] != 0:
        raise AssertionError("adapter preflight read model weights")
    if state["generate_calls"] != 1:
        raise AssertionError("valid adapter generate boundary count mismatch")
    if event_document["runtime_event_id"] != runtime_id:
        raise AssertionError("runtime event ID mismatch")

    event_path.unlink()
    campaign.rmdir()
    document = {
        "schema_version": 1,
        "status": "pass",
        "mode": "real_baseline_common_no_weight_adapter_preflight",
        "active_campaign_root": CAMPAIGN_ROOT,
        "baseline_module": program_name,
        "entrypoint": "baseline_common.make_inference.vlm_inference",
        "canonical_runtime_event_path": paths["runtime_event_path"],
        "canonical_runtime_event_sha256": event_hash,
        "runtime_event_id": runtime_id,
        "run_id": run_id,
        "generate_boundary_reached": boundary_reached,
        "legacy_path_tag": legacy_tag,
        "legacy_path_rejected_before_generate": legacy_rejected,
        "real_cuda_query_calls": 0,
        "model_weight_file_reads": state["model_weight_file_reads"],
        "adapter_calls": state,
        "greedy_unchanged": True,
        "max_new_tokens": 128,
    }
    return document


def run_preflight(output: Path | None = None) -> dict[str, object]:
    programs = [
        _run_program_preflight("baseline_instruct"),
        _run_program_preflight("baseline_thinking"),
    ]
    if any(item["status"] != "pass" for item in programs):
        raise AssertionError("one or more baseline adapter preflights failed")
    if any(item["model_weight_file_reads"] != 0 for item in programs):
        raise AssertionError("baseline adapter preflight read model weights")
    document = {
        "schema_version": 1,
        "status": "pass",
        "mode": "real_baseline_common_no_weight_adapter_preflight",
        "active_campaign_root": CAMPAIGN_ROOT,
        "program_count": len(programs),
        "programs": programs,
        "production_backend_selection": "code_only_no_cli_or_environment_selector",
        "real_cuda_query_calls": 0,
        "model_weight_file_reads": 0,
        "greedy_unchanged": True,
        "max_new_tokens": 128,
    }
    if output is not None:
        resolved_parent = output.parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(EVO_ROOT.resolve(strict=True)):
            raise ValueError("adapter preflight output escaped Evo")
        write_json_noreplace(output, document)
    return document


def main() -> int:
    output = EVO_ROOT / "logs" / f"baseline_adapter_preflight.{uuid4()}.json"
    document = run_preflight(output)
    print(
        json.dumps(
            {
                **document,
                "evidence_path": output.relative_to(EVO_ROOT).as_posix(),
                "evidence_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
