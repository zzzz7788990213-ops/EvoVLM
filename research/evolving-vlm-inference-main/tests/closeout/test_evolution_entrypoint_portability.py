"""CPU-only tests for sealed completed-campaign entrypoint replay."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evolution import local_campaign
from evolution.dsl import INSTRUCT_MODEL_ID, THINKING_MODEL_ID


MODELS = (INSTRUCT_MODEL_ID, THINKING_MODEL_ID)


def _missing_generation_two(path: str | Path) -> dict[str, object]:
    raise FileNotFoundError(Path(path))


@pytest.mark.parametrize("model_id", MODELS)
def test_packaged_completed_state_recovers_from_sealed_plan_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    monkeypatch.setattr(local_campaign, "load_checkpoint", _missing_generation_two)
    campaign = local_campaign.LocalEvolutionCampaign(model_id=model_id)
    export_path = (
        local_campaign.PROJECT_ROOT
        / f"evolved_{'instruct' if model_id == INSTRUCT_MODEL_ID else 'thinking'}.py"
    )
    before = export_path.read_bytes()

    state = campaign._load_completed()

    assert state is not None
    assert state.generation == 2
    assert state.model_id == model_id
    assert Path(state.campaign_directory) == campaign.campaign_directory
    assert Path(state.run_manifest_path) == (
        campaign.campaign_directory / "run_manifest.json"
    )
    assert export_path.read_bytes() == before


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("argv", ([], ["--resume"]), ids=("default", "resume"))
def test_completed_entrypoints_are_idempotent_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model_id: str,
    argv: list[str],
) -> None:
    monkeypatch.setattr(local_campaign, "load_checkpoint", _missing_generation_two)

    assert local_campaign.cli_main(model_id, argv) == 0

    output = capsys.readouterr().out
    assert "'generation': 2" in output
    assert "'campaign_id': 'campaign-" in output


@pytest.mark.parametrize("model_id", MODELS)
def test_recovery_rejects_plan_state_not_bound_by_sealed_summary(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    real_loader = local_campaign._load_strict_object

    def tampered_loader(path: Path) -> dict[str, object]:
        document = real_loader(path)
        if path.name != "generation2_finalization_plan.json":
            return document
        tampered = copy.deepcopy(document)
        state = tampered["final_state"]
        assert isinstance(state, dict)
        state["gpu_elapsed_seconds"] = float(state["gpu_elapsed_seconds"]) + 1.0
        return tampered

    monkeypatch.setattr(local_campaign, "_load_strict_object", tampered_loader)
    monkeypatch.setattr(local_campaign, "load_checkpoint", _missing_generation_two)
    campaign = local_campaign.LocalEvolutionCampaign(model_id=model_id)

    with pytest.raises(
        local_campaign.LocalCampaignError,
        match="not bound by the sealed summary",
    ):
        campaign._load_completed()


@pytest.mark.parametrize("model_id", MODELS)
def test_dry_run_uses_package_managed_tmp_not_user_tmpdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
) -> None:
    external_tmp = tmp_path / "ordinary-user-tmp"
    external_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(external_tmp))
    real_import_check = local_campaign.import_check_rendered_program
    observed: list[Path] = []

    def checked_import(source: str, *, temporary_directory: Path) -> dict[str, object]:
        observed.append(temporary_directory)
        return real_import_check(
            source,
            temporary_directory=temporary_directory,
        )

    monkeypatch.setattr(
        local_campaign,
        "import_check_rendered_program",
        checked_import,
    )
    result = local_campaign.LocalEvolutionCampaign(model_id=model_id).dry_run()

    assert result["gpu_called"] is False
    assert len(observed) == 1
    assert observed[0] == local_campaign.MANAGED_IMPORT_CHECK_ROOT.resolve(
        strict=True
    )
    assert observed[0].is_relative_to(local_campaign.EVO_ROOT)
    assert not observed[0].is_relative_to(external_tmp)
