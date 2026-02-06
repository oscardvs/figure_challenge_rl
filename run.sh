#!/bin/bash
# One-command execution script for the RL browser agent submission.
#
# Usage:
#   ./run.sh              # Run all 30 challenges (API mode)
#   ./run.sh --mode local # Run with fine-tuned local model
#   ./run.sh --parallel 4 # Run 4 challenges in parallel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment.
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default to API mode with 4 parallel workers.
MODE="${1:---mode}"
shift 2>/dev/null || true

echo "=========================================="
echo " RL Browser Agent — 30 Challenge Gauntlet"
echo "=========================================="
echo ""

# Run the solver.
python -m src.runner.solve_all --parallel 4 "$@"
