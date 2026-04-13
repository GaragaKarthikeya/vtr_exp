#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing .env at: $ENV_FILE" >&2
  exit 1
fi

# Load repo-local configuration.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [ -z "${VTR_FLOW_SCRIPT:-}" ] || [ ! -f "$VTR_FLOW_SCRIPT" ]; then
  echo "VTR_FLOW_SCRIPT is missing or invalid in .env" >&2
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -n "${VTR_VENV_PATH:-}" ] && [ -f "$VTR_VENV_PATH/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VTR_VENV_PATH/bin/activate"
  elif [ -n "${VTR_ROOT:-}" ] && [ -f "$VTR_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$VTR_ROOT/.venv/bin/activate"
  fi
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "VTR virtualenv is not active. Run: source ./activate-vtr.sh" >&2
  exit 1
fi

if [ ! -f "$VIRTUAL_ENV/bin/activate" ]; then
  echo "VIRTUAL_ENV does not point to a valid virtualenv: $VIRTUAL_ENV" >&2
  exit 1
fi

if [ $# -lt 2 ]; then
  cat >&2 <<'EOF'
Usage:
  ./run-vtr.sh <arch.xml> <circuit.v|circuit.eblif> [extra run_vtr_flow.py args...]

Example:
  ./run-vtr.sh vtr_arch/timing/<arch-file>.xml vtr_tests/<design>.v
EOF
  exit 1
fi

ARCH_FILE="$1"
CIRCUIT_FILE="$2"
shift 2

case "$ARCH_FILE" in
  *.xml) ;;
  *)
    echo "Expected an architecture XML as the first argument, got: $ARCH_FILE" >&2
    exit 1
    ;;
esac

case "$CIRCUIT_FILE" in
  *.v|*.sv|*.eblif) ;;
  *)
    echo "Expected a circuit file (.v, .sv, or .eblif) as the second argument, got: $CIRCUIT_FILE" >&2
    exit 1
    ;;
esac

TEMP_DIR="$SCRIPT_DIR/temp"
if [ -e "$TEMP_DIR" ]; then
  rm -rf -- "$TEMP_DIR"
fi

exec python3 "$VTR_FLOW_SCRIPT" "$CIRCUIT_FILE" "$ARCH_FILE" "$@"
