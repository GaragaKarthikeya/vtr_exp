# Source this file from your shell to activate the VTR virtual environment.
# Example: source /home/karthikeya/vtr_exp/activate-vtr.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
  # Load repo-local configuration first so the helper follows the .env file.
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

VTR_ROOT="${VTR_ROOT:-/home/karthikeya/vtr-verilog-to-routing}"
VTR_VENV_PATH="${VTR_VENV_PATH:-$VTR_ROOT/.venv}"
ACTIVATE_FILE="$VTR_VENV_PATH/bin/activate"

if [ ! -f "$ACTIVATE_FILE" ]; then
  echo "Virtual environment not found at: $VTR_VENV_PATH" >&2
  echo "Create it first, or set VTR_VENV_PATH to the correct location." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE_FILE"
