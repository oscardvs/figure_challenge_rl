#!/bin/bash
# One-command execution script for the RL browser agent submission.
#
# Usage:
#   ./run.sh                        # Run the 30-step gauntlet (API mode)
#   ./run.sh --mode local           # Run with fine-tuned local model
#   ./run.sh --provider google      # Use Gemini instead of Claude

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment.
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo " RL Browser Agent — 30-Step Gauntlet"
echo "=========================================="
echo ""

# Run the solver.
python -m src.runner.solve_all "$@"
