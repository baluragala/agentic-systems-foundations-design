#!/usr/bin/env bash
# ===========================================================================
# Agentic Systems Foundations — one-shot environment setup for macOS / Linux
# ===========================================================================
# Creates a Python virtual environment in .venv, installs dependencies, and
# registers a Jupyter kernel so the notebooks pick up this environment.
#
# Usage:   bash scripts/setup.sh
# Then:    source .venv/bin/activate
# ===========================================================================
set -euo pipefail

# Move to the project root (this script lives in scripts/).
cd "$(dirname "$0")/.."

# Pick a python: prefer python3, fall back to python.
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "ERROR: Python 3 was not found. Install Python 3.9+ and re-run." >&2
  exit 1
fi
echo "Using: $("$PY" --version)"

echo "==> Creating virtual environment in .venv"
"$PY" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo "==> Installing project dependencies (requirements.txt)"
pip install -r requirements.txt

echo "==> Installing JupyterLab + kernel"
pip install jupyterlab ipykernel
python -m ipykernel install --user --name agentic-systems \
  --display-name "Python (Agentic Systems)"

echo ""
echo "============================================================"
echo " Setup complete."
echo " Activate the environment:   source .venv/bin/activate"
echo " Launch the notebooks:       jupyter lab"
echo "   (in a notebook, choose the 'Python (Agentic Systems)' kernel)"
echo " Deactivate when done:       deactivate"
echo "============================================================"
