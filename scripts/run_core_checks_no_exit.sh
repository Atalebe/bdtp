#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'HOLD: run with bash; do not source.'; return 2
fi
set +e
set +u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$ROOT/.local_runs" || exit 2
OUT="$(mktemp -d "$ROOT/.local_runs/core_XXXXXXXX")" || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
(
 set -e
 cd "$ROOT"
 "$PYTHON" scripts/verify_bootstrap.py
 "$PYTHON" -m unittest discover -s tests -v
 "$PYTHON" scripts/check_numerical_core.py
 if [[ "${1:-}" == "--inject-failure" ]]; then
   printf '%s\n' 'Intentional failure before completion.'; false
 elif [[ $# -gt 0 ]]; then
   printf '%s\n' 'Unexpected argument.'; false
 fi
) >"$OUT/checks.log" 2>&1
STATUS=$?
cat "$OUT/checks.log"
if [[ $STATUS -eq 0 ]]; then printf '%s\n' 'CORE_CHECKS_PASSED' >"$OUT/PASS.txt"; else printf '%s\n' 'HOLD: no completion or logbook written.'; fi
printf 'diagnostics=%s\ncaller_terminal_preserved=true\n' "$OUT"
exit "$STATUS"
