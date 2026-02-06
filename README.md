# Figure RL — Browser Navigation Gauntlet Agent

RL-trained browser agent that solves a 30-step sequential web navigation gauntlet. Each step presents a unique puzzle (scroll challenges, timers, hidden elements, etc.) that reveals a 6-character code. The agent must find the code, enter it, and submit to advance.

## Quick Start

### 1. Create venv and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Add your API key

```bash
# .env
GOOGLE_API_KEY=your-gemini-api-key-here
```

Get a key at https://aistudio.google.com/apikey

Other providers (optional):
```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

### 3. Run the agent

```bash
# Solve all 30 steps with Gemini 3 Flash
./run.sh --provider google

# Or explicitly:
python -m src.runner.solve_all --provider google

# Use Claude instead:
python -m src.runner.solve_all --provider anthropic

# Use a specific model:
python -m src.runner.solve_all --provider google --model gemini-3-pro-preview
```

Results are saved to `results/metrics.json`.

### 4. Explore challenges manually (optional)

```bash
# Non-interactive — capture AXTree snapshots of each step
python scripts/explore_challenges.py --save-snapshots

# Interactive — step through with a visible browser
python scripts/explore_challenges.py --interactive
```

## Training Pipeline

### Phase 1: Collect MCTS trajectories

```bash
# Collect data for step 1 (the only step directly reachable)
python -m src.training.collect_trajectories --steps 1 --provider google

# Data saved to data/trajectories/ and data/preference_pairs/
```

### Phase 2: SFT on successful trajectories

```bash
python -m src.training.sft_train --data-dir data/trajectories --epochs 3
# Model saved to models/sft/
```

### Phase 3: DPO on preference pairs

```bash
python -m src.training.dpo_train --sft-model models/sft
# Model saved to models/dpo/
```

### Phase 4: M-GRPO with live rollouts

```bash
python -m src.training.grpo_train --dpo-model models/dpo --episodes 500
# Model saved to models/grpo/
```

### Phase 5: Run with fine-tuned model

```bash
python -m src.runner.solve_all --mode local
```

## Project Structure

```
figure_rl/
├── config/
│   ├── challenge_config.yaml   # Challenge URL, step count, MCTS params
│   └── model_config.yaml       # Base model, QLoRA, training, API models
├── src/
│   ├── agent/
│   │   ├── policy.py           # LLM wrappers (Anthropic, OpenAI, Gemini)
│   │   ├── mcts.py             # Monte Carlo Tree Search over browser states
│   │   └── prompts.py          # System/task prompts, action parsing
│   ├── environment/
│   │   ├── browser_env.py      # BrowserGym envs (GauntletEnv, StepEnv)
│   │   ├── action_space.py     # BrowserGym HighLevelActionSet config
│   │   └── observation.py      # AXTree pruning + observation formatting
│   ├── runner/
│   │   ├── solve_all.py        # Main entry point — solve full gauntlet
│   │   └── metrics.py          # Per-step and aggregate metrics
│   └── training/
│       ├── collect_trajectories.py  # MCTS data collection
│       ├── sft_train.py             # Stage 1: Supervised fine-tuning
│       ├── dpo_train.py             # Stage 2: DPO on preference pairs
│       └── grpo_train.py            # Stage 3: M-GRPO with live rollouts
├── scripts/
│   └── explore_challenges.py   # Manual challenge exploration
├── data/                       # Collected trajectories + snapshots
├── .env                        # API keys (gitignored)
├── run.sh                      # One-command entry point
└── requirements.txt
```

## Supported Providers

| Provider | Flag | Default Model | Env Variable |
|----------|------|---------------|--------------|
| Google | `--provider google` | `gemini-3-flash-preview` | `GOOGLE_API_KEY` |
| Anthropic | `--provider anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `--provider openai` | `gpt-4o` | `OPENAI_API_KEY` |
