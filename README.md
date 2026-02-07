# Figure RL — Browser Navigation Gauntlet Agent

RL-trained browser agent that solves a 30-step sequential web navigation gauntlet. Each step presents a unique puzzle (scroll, click, hover, drag, shadow DOM, canvas, audio, websockets, etc.) that reveals a 6-character code. The agent must find the code, enter it, and submit to advance.

## Quick Start

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 2. API key (at least one)
echo 'GOOGLE_API_KEY=your-key' >> .env        # https://aistudio.google.com/apikey
echo 'ANTHROPIC_API_KEY=your-key' >> .env      # optional
echo 'OPENAI_API_KEY=your-key' >> .env         # optional

# 3. Run
./run.sh --provider google
```

## Running the Agent

Four modes are available:

```bash
# API mode — LLM agent solves challenges via API calls
python -m src.runner.solve_all --provider google           # Gemini Flash
python -m src.runner.solve_all --provider anthropic        # Claude Sonnet
python -m src.runner.solve_all --provider openai           # GPT-4o

# Deterministic mode — hardcoded solver, no LLM cost
python -m src.runner.solve_all --mode deterministic

# Hybrid mode — deterministic solver first, LLM fallback for unsolved steps
python -m src.runner.solve_all --mode hybrid --provider google

# Local mode — fine-tuned Qwen2.5-3B (requires training first)
python -m src.runner.solve_all --mode local
```

Results are saved to `results/metrics.json`.

## Data Collection

Two methods for generating training data:

### Method 1: Expert Trajectories (deterministic solver)

The deterministic solver acts as a teacher. It runs Playwright to solve challenges while a `RecordingPage` proxy intercepts every call and translates it to BrowserGym action format. This produces clean, labeled trajectories with no API cost.

```bash
# Collect trajectories for all 30 steps, 15 runs each
# Challenges are randomized per run, so more runs = more challenge type diversity
python -m src.training.collect_expert_trajectories --runs-per-step 15

# Collect specific steps only
python -m src.training.collect_expert_trajectories --steps 1 2 3 4 5 --runs-per-step 10

# Large overnight run for maximum coverage
python -m src.training.collect_expert_trajectories --runs-per-step 20
```

Output:
- `data/expert_trajectories/step_NN.json` — trajectories in SFT-ready format
- `data/expert_preference_pairs/step_NN.json` — success/failure contrast pairs for DPO

Multiple runs **merge** with existing data (append, not overwrite), so you can run collection incrementally.

**Why multiple runs?** Challenges are randomized per run. Step 2 might be a scroll challenge one time and a shadow DOM challenge the next. The solver handles ~21/30 challenge types, so 15+ runs per step gives good probability of hitting solvable variants at each position.

**Check coverage after collection:**

```bash
python -c "
import json
from pathlib import Path
from collections import Counter
types = Counter()
for f in Path('data/expert_trajectories').glob('step_*.json'):
    for t in json.loads(f.read_text()):
        if t['success']: types[t['challenge_type']] += 1
for typ, cnt in types.most_common(): print(f'  {typ}: {cnt}')
print(f'Total: {sum(types.values())} successful across {len(types)} types')
"
```

### Method 2: MCTS Trajectories (API agent)

Uses an LLM agent with Monte Carlo Tree Search to explore challenges. More expensive but covers challenge types the deterministic solver can't handle.

```bash
# Collect MCTS trajectories using Gemini Flash
python -m src.training.collect_trajectories --provider google

# Specific steps
python -m src.training.collect_trajectories --steps 1 2 3 --provider google
```

Output:
- `data/trajectories/` — MCTS trajectories
- `data/preference_pairs/` — preference pairs from MCTS win/loss contrasts
- `data/known_solutions.json` — accumulated step solutions for replay

### Add Chain-of-thought Reasoning (optional)

Annotates each (observation, action) pair with 2-4 sentence reasoning via API. This produces the `<think>...</think>` reasoning the model learns to generate.

```bash
# Annotate expert trajectories with Gemini Flash (cheapest)
python -m src.training.generate_cot --provider google

# Use a different model
python -m src.training.generate_cot --provider anthropic --model claude-sonnet-4-20250514

# Custom input/output dirs
python -m src.training.generate_cot --input-dir data/expert_trajectories --output-dir data/cot_trajectories
```

API responses are cached in `data/cot_cache.json` to avoid redundant calls on re-runs.

## Training Pipeline

Three-stage pipeline: SFT -> DPO -> M-GRPO. Each stage builds on the previous one.

### Stage 1: Supervised Fine-tuning (SFT)

Trains Qwen2.5-3B with QLoRA on successful trajectories. Learns the basic observation -> action mapping.

```bash
# Train on expert trajectories only
python -m src.training.sft_train --expert-dir data/expert_trajectories

# Train on both MCTS and expert trajectories
python -m src.training.sft_train --data-dir data/trajectories --expert-dir data/expert_trajectories

# Custom epochs
python -m src.training.sft_train --expert-dir data/expert_trajectories --epochs 5

# Output: models/sft/
```

### Stage 2: Direct Preference Optimization (DPO)

Refines the SFT model using preference pairs (chosen vs rejected actions for the same observation).

```bash
# Train on expert preference pairs
python -m src.training.dpo_train --sft-model models/sft --expert-pairs-dir data/expert_preference_pairs

# Train on both MCTS and expert pairs
python -m src.training.dpo_train --sft-model models/sft \
  --data-dir data/preference_pairs \
  --expert-pairs-dir data/expert_preference_pairs

# Output: models/dpo/
```

### Stage 3: Group Relative Policy Optimization (M-GRPO)

Online RL with live browser rollouts. The model generates multiple trajectories per challenge, receives rewards based on progress, and updates via clipped importance-weighted loss (no value network needed).

```bash
# Default: 500 episodes, group size 4
python -m src.training.grpo_train --dpo-model models/dpo

# More episodes
python -m src.training.grpo_train --dpo-model models/dpo --episodes 1000

# Output: models/grpo/
```

M-GRPO uses `GauntletEnv` with randomized challenges and curriculum learning — starts with early steps and expands as the model improves.

### Run with Trained Model

```bash
python -m src.runner.solve_all --mode local
```

## Project Structure

```
figure_rl/
├── config/
│   ├── challenge_config.yaml        # Challenge URL, step count, MCTS params
│   └── model_config.yaml            # Base model, QLoRA, training hyperparams
├── src/
│   ├── agent/
│   │   ├── policy.py                # LLM wrappers (Anthropic, OpenAI, Gemini)
│   │   ├── mcts.py                  # Monte Carlo Tree Search over browser states
│   │   ├── prompts.py               # System/task prompts, action parsing
│   │   └── trajectory_recorder.py   # RecordingPage proxy (Playwright -> BrowserGym)
│   ├── environment/
│   │   ├── browser_env.py           # BrowserGym envs (GauntletEnv, StepEnv)
│   │   ├── action_space.py          # BrowserGym HighLevelActionSet + js_eval
│   │   └── observation.py           # AXTree pruning, HTML snippet extraction
│   ├── solver/
│   │   ├── challenge_detector.py    # Classifies pages into 21 challenge types
│   │   ├── challenge_handlers.py    # Per-type solve handlers (all sync Playwright)
│   │   ├── deterministic_solver.py  # Orchestrates detect -> handle -> submit loop
│   │   └── hybrid_agent.py          # Deterministic solver + LLM fallback
│   ├── runner/
│   │   ├── solve_all.py             # Main entry point — solve full gauntlet
│   │   └── metrics.py               # Per-step and aggregate metrics
│   └── training/
│       ├── collect_expert_trajectories.py  # Expert data from deterministic solver
│       ├── collect_trajectories.py         # MCTS data from API agent
│       ├── generate_cot.py                 # Chain-of-thought annotation via API
│       ├── sft_train.py                    # Stage 1: Supervised fine-tuning
│       ├── dpo_train.py                    # Stage 2: DPO on preference pairs
│       └── grpo_train.py                   # Stage 3: M-GRPO with live rollouts
├── data/
│   ├── expert_trajectories/         # Solver-generated trajectories (SFT format)
│   ├── expert_preference_pairs/     # Success/failure pairs from solver runs
│   ├── trajectories/                # MCTS-generated trajectories
│   ├── preference_pairs/            # MCTS win/loss preference pairs
│   └── known_solutions.json         # Accumulated step solutions for replay
├── models/                          # Trained model checkpoints (sft/, dpo/, grpo/)
├── results/                         # Run metrics and logs
├── scripts/
│   ├── debug_run.py                 # Smoke test — 15 actions on step 1
│   └── explore_challenges.py        # Manual challenge exploration
├── .env                             # API keys (gitignored)
├── run.sh                           # One-command entry point
└── requirements.txt
```

## How It Works

### The Gauntlet

The target is a React SPA with 30 sequential puzzle steps. Each step randomly assigns one of ~21 challenge types:

| Category | Challenge Types |
|----------|----------------|
| Navigation | Scroll to find, click sequence, hover reveal |
| Input | Form fill, drag and drop, slider |
| Hidden content | Shadow DOM, iframe, computed style, mutation observer |
| Timing | Timer, animated button, websocket |
| Media | Canvas, audio, video |
| Advanced | Multi-step, obfuscated, encrypted, cookie/storage |

Codes are 6-character alphanumeric strings (e.g., `K7SMGA`), dynamically generated each run. Direct URL navigation only works for step 1 — all other steps require sequential traversal from the start.

### Architecture

**Deterministic Solver** (`src/solver/`): Pattern-matching approach. A `ChallengeDetector` collects DOM signals via a single `page.evaluate()` call and classifies the challenge type. Then a type-specific handler from `ChallengeHandlers` executes the solve strategy using raw Playwright. Handles 21/30 challenge types.

**LLM Agent** (`src/agent/`): Observation-based approach. Receives an AXTree + HTML snippet of the page, reasons about what to do, and emits BrowserGym actions. Uses `<think>...</think>` format for chain-of-thought reasoning.

**RecordingPage** (`src/agent/trajectory_recorder.py`): Bridge between the two. Wraps a Playwright Page to intercept solver calls and map them to BrowserGym actions:
- `page.evaluate(js)` -> `js_eval(js)` (filtered by side-effect classification)
- `page.keyboard.type(text)` -> `fill(bid, text)`
- `page.mouse.click(x, y)` -> `mouse_click(x, y)`
- `page.mouse.down()` ... `page.mouse.up()` -> `mouse_drag(x1, y1, x2, y2)`

### Training Loop

1. **Expert data**: Deterministic solver generates clean trajectories via RecordingPage
2. **SFT**: Model learns basic obs -> action mapping from expert demonstrations
3. **DPO**: Model learns to prefer successful actions over failed ones
4. **M-GRPO**: Model improves via online RL with live browser interaction

## Supported Providers

| Provider | Flag | Default Model | Env Variable |
|----------|------|---------------|--------------|
| Google | `--provider google` | `gemini-3-flash-preview` | `GOOGLE_API_KEY` |
| Anthropic | `--provider anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `--provider openai` | `gpt-4o` | `OPENAI_API_KEY` |

## Debug

```bash
# Smoke test — runs 15 actions on step 1, dumps observations
python scripts/debug_run.py

# Explore challenges interactively with visible browser
python scripts/explore_challenges.py --interactive

# Capture AXTree snapshots of each step
python scripts/explore_challenges.py --save-snapshots
```
