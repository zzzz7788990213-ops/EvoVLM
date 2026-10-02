#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly PROJECT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly MODE="${1:-fresh}"
readonly PYTHON="${PYTHON:-python3}"
readonly RUNTIME_ROOT="${EVOLVING_VLM_RUNTIME_ROOT:-$PROJECT/.reproduce_runtime}"
readonly REQUESTED_CAMPAIGN_ID="${2:-${RECOVERY_CAMPAIGN_ID:-}}"

[[ "$RUNTIME_ROOT" == "$PROJECT" || "$RUNTIME_ROOT" == "$PROJECT/"* ]] \
  || { echo "reproduction failed: runtime root must remain inside this package" >&2; exit 2; }
mkdir -p "$RUNTIME_ROOT/.tmp" "$RUNTIME_ROOT/.cache"
cd "$PROJECT"
[[ "$(pwd -P)" == "$PROJECT" ]]

export EVOLVING_VLM_RUNTIME_ROOT="$RUNTIME_ROOT"
export TMPDIR="$RUNTIME_ROOT/.tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/.cache"
export HF_HOME="$RUNTIME_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/.cache/huggingface/transformers"
export TORCH_HOME="$RUNTIME_ROOT/.cache/torch"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
cd "$PROJECT"

fail() {
  echo "reproduction failed: $*" >&2
  exit 2
}

offline_evidence_mode() {
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export EVO_NETWORK_DISABLED=1
  export HF_DATASETS_OFFLINE=1
}

evidence_replay() {
  "$PYTHON" -B -m evaluation.portable_reproduce
}

portable_smoke() {
  "$PYTHON" -B - <<'PY'
import importlib, inspect, pathlib
root = pathlib.Path.cwd().resolve()
names = ("manual_instruct", "manual_thinking", "evolved_instruct", "evolved_thinking", "best_accuracy", "best_speed", "best_overall")
for name in names:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().parent == root
    assert list(inspect.signature(module.vlm_inference).parameters) == ["image_path", "question"]
print("portable CPU import/signature smoke PASS")
PY
}

require_campaign_id() {
  [[ -n "$REQUESTED_CAMPAIGN_ID" ]] \
    || fail "campaign ID required as argument 2 or RECOVERY_CAMPAIGN_ID"
}

phase2_verify() {
  "$PYTHON" "$PROJECT/evaluation/verify_closeout.py" \
    --root "$PROJECT" --scope phase2
}

recovery_check() {
  require_campaign_id
  local campaign_dir
  campaign_dir="$PROJECT/results/phase3_recovery_v2/campaigns/$REQUESTED_CAMPAIGN_ID"
  if [[ -f "$campaign_dir/speed32_replacement_completed.json" ]]; then
    "$PYTHON" -m evaluation.phase3_speed_replacement check \
      --campaign-id "$REQUESTED_CAMPAIGN_ID"
  else
    "$PYTHON" -m evaluation.phase3_recovery check \
      --campaign-id "$REQUESTED_CAMPAIGN_ID"
  fi
}

final_check() {
  evidence_replay
  "$PYTHON" "$PROJECT/submission_checker.py" \
    --stage final --root "$PROJECT" --anchor "$PROJECT"
}

build_report() {
  evidence_replay
  [[ ! -e "$PROJECT/report.pdf" ]] || fail "report.pdf already exists; refusing overwrite"
  [[ ! -e "$PROJECT/report/generated/pdf_qa.json" ]] \
    || fail "PDF QA receipt already exists; refusing overwrite"
  local tectonic_bin
  if [[ -n "${TECTONIC_BIN:-}" ]]; then
    tectonic_bin="$TECTONIC_BIN"
  elif command -v tectonic >/dev/null; then
    tectonic_bin="$(command -v tectonic)"
  elif [[ -x "$RUNTIME_ROOT/.runtime/tectonic-0.16.9/tectonic" ]]; then
    tectonic_bin="$RUNTIME_ROOT/.runtime/tectonic-0.16.9/tectonic"
  else
    fail "Tectonic unavailable; official NeurIPS PDF not built"
  fi
  [[ -x "$tectonic_bin" && ! -L "$tectonic_bin" ]] \
    || fail "Tectonic path is not a physical executable"
  local compiler_version build_dir
  compiler_version="$($tectonic_bin --version)"
  build_dir="$(mktemp -d "$RUNTIME_ROOT/.tmp/report-build.XXXXXXXX")"
  (
    cd "$PROJECT/report"
    "$tectonic_bin" --untrusted --keep-intermediates --keep-logs \
      --outdir "$build_dir" tectonic_entry.tex
  )
  [[ -s "$build_dir/tectonic_entry.pdf" && -s "$build_dir/tectonic_entry.log" ]] \
    || fail "official TeX compile produced no PDF or log"
  "$PYTHON" "$PROJECT/scripts/qa_report_pdf.py" \
    --pdf "$build_dir/tectonic_entry.pdf" \
    --tex "$PROJECT/report/main.tex" \
    --entry "$PROJECT/report/tectonic_entry.tex" \
    --style "$PROJECT/report/neurips_2025.sty" \
    --log "$build_dir/tectonic_entry.log" \
    --render-dir "$build_dir/rendered" \
    --receipt "$build_dir/pdf_qa.json" \
    --compiler "$compiler_version"
  ln "$build_dir/pdf_qa.json" "$PROJECT/report/generated/pdf_qa.json"
  ln "$build_dir/tectonic_entry.pdf" "$PROJECT/report.pdf"
  echo "official NeurIPS PDF built: report.pdf; QA: report/generated/pdf_qa.json; rendered pages: $build_dir/rendered"
}

case "$MODE" in
  fresh)
    offline_evidence_mode
    "$PYTHON" -B "$PROJECT/scripts/final_fresh_reproduction.py"
    ;;
  evidence|status)
    offline_evidence_mode
    evidence_replay
    ;;
  phase2-verify)
    phase2_verify
    ;;
  cpu-tests)
    offline_evidence_mode
    "$PYTHON" -m pytest -q -p no:cacheprovider
    "$PYTHON" -m pip check
    ;;
  smoke)
    offline_evidence_mode
    portable_smoke
    ;;
  check)
    offline_evidence_mode
    evidence_replay
    phase2_verify
    "$PYTHON" "$PROJECT/submission_checker.py" \
      --stage final --root "$PROJECT" --anchor "$PROJECT"
    ;;
  package)
    offline_evidence_mode
    final_check
    "$PYTHON" "$PROJECT/scripts/package_submission.py" --project "$PROJECT"
    ;;
  full-inference)
    shift
    "$PYTHON" -B "$PROJECT/scripts/portable_full_inference.py" "$@"
    ;;
  -h|--help|help)
    cat <<'EOF'
Usage: bash reproduce.sh [MODE]
  (no MODE), fresh     one fresh dual-GPU reproduction through evaluate.py
  evidence             CPU-only replay of sealed per-query evidence
  check                evidence replay plus closeout/package checks
  smoke                CPU-only imports and official function signatures
  full-inference --output DIR [--data JSON --images DIR]
                       optional fresh GPU inference into a unique directory
  phase2-verify | cpu-tests | package
EOF
    ;;
  *)
    echo "unknown reproduction mode: $MODE (use --help)" >&2
    exit 64
    ;;
esac
