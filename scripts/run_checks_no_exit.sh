#!/usr/bin/env bash
# Invoke with bash; never source into an interactive shell.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'HOLD: invoke this launcher with bash, do not source it.'
  return 2
fi
set +e
set +u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$ROOT" ]]; then exit 2; fi
mkdir -p "$ROOT/.local_runs"
OUT="$(mktemp -d "$ROOT/.local_runs/bootstrap_XXXXXXXX")"
if [[ -z "$OUT" ]]; then exit 2; fi
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
STATUS=0
(
  set -e
  cd "$ROOT"
  "$PYTHON" scripts/verify_bootstrap.py
  "$PYTHON" -m unittest discover -s tests -v
  if [[ "${1:-}" == "--inject-failure" ]]; then
    printf '%s\n' 'Deliberate infrastructure failure; downstream completion must stop.'
    false
  elif [[ $# -gt 0 ]]; then
    printf '%s\n' 'Unknown argument.'
    false
  fi
) >"$OUT/checks.log" 2>&1 || STATUS=$?
cat "$OUT/checks.log"
if [[ $STATUS -eq 0 ]]; then
  printf '%s\n' 'Repository infrastructure checks passed.' >"$OUT/PASS.txt"
else
  printf '%s\n' 'HOLD: diagnostics retained; no completion or milestone written.'
fi
printf 'diagnostics=%s\ncaller_terminal_preserved=true\n' "$OUT"
exit "$STATUS"
