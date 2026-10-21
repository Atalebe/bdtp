#!/usr/bin/env bash
# Invoke with bash, never source. Each run requires a new --output directory.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'HOLD: run with bash; do not source.'; return 2
fi
set +e
set +u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$ROOT/.local_runs" || exit 2
LOG="$(mktemp "$ROOT/.local_runs/campaign_XXXXXXXX.log")" || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" -u "$ROOT/scripts/paper_campaign.py" "$@" >"$LOG" 2>&1
STATUS=$?
cat "$LOG"
if [[ $STATUS -ne 0 ]]; then printf '%s\n' 'HOLD: logs retained; no completion, logbook or Git action.'; fi
printf 'diagnostics=%s\ncaller_terminal_preserved=true\n' "$LOG"
exit "$STATUS"
