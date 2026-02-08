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

# Local mode — fine-tuned Qwen3-4B (requires training first)
python -m src.runner.solve_all --mode local
```

Results are saved to `results/metrics.json`.

## Data Collection

Two methods for generating training data:

### Method 1: Expert Trajectories (deterministic solver)

The deterministic solver acts as a teacher. It runs Playwright to solve challenges while a `RecordingPage` proxy intercepts every call and translates it to BrowserGym action format. This produces clean, labeled trajectories with no API cost.

```bash
# Collect trajectories for all 30 steps, 5 runs each
python -m src.training.collect_expert_trajectories --runs-per-step 5

# Collect specific steps only
python -m src.training.collect_expert_trajectories --steps 1 2 3 4 5 --runs-per-step 10

# Watch the solver in a visible browser window (useful for debugging)
# Set headless: false in config/challenge_config.yaml, then run as normal
```

Output:
- `data/expert_trajectories/step_NN.json` — trajectories in SFT-ready format
- `data/expert_preference_pairs/step_NN.json` — success/failure contrast pairs for DPO

Multiple runs **merge** with existing data (append, not overwrite), so you can run collection incrementally.

#### How collection works

The gauntlet is a **sequential React SPA**: you must start at step 1 and solve each step in order to reach the next. There is no way to jump directly to step 15 — you have to solve steps 1 through 14 first.

Challenges are **randomized on every page load**. Each time you open a fresh browser and click START, the 30 step positions get randomly assigned challenge types and codes. So step 5 might be a "hover reveal" challenge on one run and a "shadow DOM" challenge on the next.

To collect a trajectory for step N, the script:

1. Opens a **fresh Chromium browser** (new session, new randomization)
2. Clicks START and lands on step 1
3. Solves steps 1 through N-1 sequentially (unrecorded, just to reach the target)
4. **Records** the solver's actions on step N via the `RecordingPage` proxy
5. Closes the browser

Each "run" repeats this entire process from scratch. With `--runs-per-step 5`, five separate browser sessions are opened for the same target step, each seeing a different randomization of challenges and codes. This is how we get diverse training data: the model sees many different challenge types at the same step position.

```
Run 0: [browser 1] START → step 1 → step 2 → ... → step N-1 → RECORD step N → close
Run 1: [browser 2] START → step 1 → step 2 → ... → step N-1 → RECORD step N → close
Run 2: [browser 3] START → step 1 → step 2 → ... → step N-1 → RECORD step N → close
```

The trade-off: higher step numbers are slower to collect because each run must solve all prior steps as warm-up. Step 20 requires solving 19 steps before recording even begins. If any warm-up step hits a challenge type the solver can't handle, that entire run fails.

#### Check coverage after collection

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

Trains Qwen3-4B with QLoRA on successful trajectories. Learns the basic observation -> action mapping.

```bash
# Train on expert trajectories only
python -m src.training.sft_train --expert-dir data/expert_trajectories

# Train on both MCTS and expert trajectories
python -m src.training.sft_train --data-dir data/trajectories --expert-dir data/expert_trajectories

# Custom epochs
python -m src.training.sft_train --expert-dir data/expert_trajectories --epochs 5

# Output: models/sft/
```

#### QLoRA Rank Tuning

LoRA rank (`r`) controls the number of trainable parameters. Higher rank = more capacity but more overfitting risk, especially with small datasets.

| Rank | Alpha | Trainable params | % of 4B | Notes |
|------|-------|-----------------|---------|-------|
| r=16 | 32 | 33M | 1.3% | Default — good for <500 trajectories |
| r=32 | 64 | 66M | 2.6% | Consider at 500+ trajectories |
| r=64 | 128 | 132M | 5.0% | For 1000+ trajectories, tight on 16GB |
| r=128 | 256 | 264M | 9.5% | Not recommended on 16GB VRAM |

**Guidelines** (from [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)):
- Keep `alpha = 2 * rank` or `alpha = rank`
- Training loss below 0.2 signals overfitting — reduce epochs or increase dropout
- 1-3 epochs recommended; >3 epochs gives diminishing returns with small data
- `lora_dropout=0.05` helps regularize with <500 examples
- Target all linear layers (q/k/v/o_proj + gate/up/down_proj) for best results

See also: [QLoRA paper](https://arxiv.org/abs/2305.14314), [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

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

## Training Parameters Reference

All parameters are in `config/model_config.yaml`.

### Base Model

| Parameter | Value | Notes |
|-----------|-------|-------|
| `base_model.name` | `Qwen/Qwen3-4B` | Full-precision model name |
| `base_model.quantized_name` | `unsloth/Qwen3-4B-unsloth-bnb-4bit` | 4-bit quantized for training |

### QLoRA

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` | 16 | LoRA rank — 33M trainable params (1.3% of 4B) |
| `alpha` | 32 | Scaling factor — `alpha = 2 * r` |
| `dropout` | 0.0 | No dropout (small dataset, few epochs) |
| `target_modules` | q/k/v/o_proj + gate/up/down_proj | All linear layers |

### SFT

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_seq_length` | 6144 | Observations are ~2500 tokens + 1522 token system prompt |
| `per_device_batch_size` | 2 | |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `learning_rate` | 2e-4 | Standard for QLoRA SFT |
| `num_epochs` | 5 | Small dataset needs more passes |
| `warmup_ratio` | 0.1 | |
| `gradient_checkpointing` | `"unsloth"` | Unsloth's optimized checkpointing |

### DPO

| Parameter | Value | Notes |
|-----------|-------|-------|
| `beta` | 0.05 | KL penalty weight — low for RL agents |
| `learning_rate` | 5e-6 | ~40x lower than SFT to avoid catastrophic forgetting |
| `per_device_batch_size` | 1 | DPO needs 4 forward passes per step (chosen+rejected × model+ref) |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8 |
| `max_seq_length` | 6144 | Must match SFT — observations are large |
| `max_prompt_length` | 5632 | `max_seq_length - 512` — actions are short (~50 tokens) |
| `num_epochs` | 2 | |
| `ref_model` | dual-adapter | Loads SFT adapter twice (trainable + frozen reference) |

### M-GRPO

| Parameter | Value | Notes |
|-----------|-------|-------|
| `kl_coefficient` | 0.001 | Very low — allow policy to diverge from SFT |
| `clip_epsilon` | 0.2 | PPO-style clipping |
| `group_size` | 8 | Rollouts per challenge for advantage estimation |
| `max_new_tokens` | 512 | Generation budget per action |
| `sampling_temperature` | 0.8 | |
| `rollout_episodes` | 500 | |
| `curriculum_max_steps` | 5 | Start easy, expand as success rate improves |

### Local Inference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | 0.7 | Inherited from GRPO config |
| `max_new_tokens` | 512 | |
| `max_seq_length` | 6144 | Matches training |
| `repetition_penalty` | 1.15 | Prevents action loops |
| `top_p` | 0.9 | Nucleus sampling |

## Local Mode (`--mode local`)

Loads the fine-tuned LoRA adapter and runs inference locally using the same Unsloth stack as training.

```bash
# Default adapter (models/sft/)
python -m src.runner.solve_all --mode local

# Specific checkpoint
python -m src.runner.solve_all --mode local --adapter-dir models/sft/checkpoint-45

# DPO model
python -m src.runner.solve_all --mode local --adapter-dir models/dpo
```

### How it works

`LocalPolicy` in `src/agent/policy.py`:

1. **Model loading**: Tries Unsloth `FastLanguageModel.from_pretrained()` first (applies inference optimizations via `.for_inference()`), falls back to PEFT `AutoPeftModelForCausalLM` + 4-bit BitsAndBytes.

2. **Prompt format**: Uses `PURE_AGENT_SYSTEM_PROMPT` — the same prompt expert trajectories were trained on. No challenge-type hints.

3. **Multi-turn conversation**: Maintains conversation history within each step, matching the exact training format:
   ```
   system prompt
   user: "Complete step N.\n\n[Action 0] Current page state:\n{obs}"
   assistant: "click(bid)"
   user: "[Action 1] Current page state:\n{obs}"
   assistant: "js_eval(...)"
   ...
   ```

4. **Context management**: Monitors token count and trims oldest conversation turns (keeping the first user message with task instruction) to stay within `max_seq_length`. Prevents the right-truncation problem where the tokenizer would chop the most recent observation.

5. **Step transitions**: Detects step changes from the task prompt and resets conversation history.

6. **Generation**: `model.generate()` with `repetition_penalty=1.15` to prevent action loops, dual EOS tokens (`<|im_end|>` + `<|endoftext|>`) for robust stopping.

### Performance

On RTX 4080 SUPER (16GB VRAM):
- Model load: ~30s (first time), ~10s (cached)
- Per-action generation: ~1-3s
- VRAM usage: ~5GB (4-bit quantized)

## Pipeline Status

Current data coverage:

| Steps | Expert Trajectories | Preference Pairs | Notes |
|-------|-------------------|-----------------|-------|
| 1-12 | 137 successful | 849 pairs | Good coverage |
| 13-25 | 0 successful | 0 pairs | Solver fails on these challenge types |
| 26-30 | Not attempted | — | Need steps 13-25 first |

### Lessons Learned

- **SFT loss 0.85 → 0.05** over 5 epochs with 115 expert trajectories. Model generates valid BrowserGym actions but has JavaScript knowledge gaps (e.g., `NodeList.map()` instead of `Array.from(...).map()`).
- **CoT annotation matters**: Without reasoning in `<think>` blocks, the model pattern-matches actions without understanding observations. With CoT, it learns to attend to relevant page clues before acting.
- **DPO `max_seq_length` must match SFT**: Observations are ~2500 tokens + 1522 token system prompt. The original DPO config of 2048 truncated everything to garbage.
- **Context window management is critical for local inference**: After ~5 multi-turn exchanges, the conversation exceeds `max_seq_length`. Without active trimming, right-truncation chops the latest observation and the model degenerates into outputting observation-like text.
- **Repetition penalty (1.15) prevents action loops**: Without it, the model repeats the same `js_eval(scrollTo...)` indefinitely.
- **Never `el.remove()` React DOM nodes**: Use CSS hiding instead. Removing React-managed nodes desynchronizes the virtual DOM and crashes the SPA permanently.
- **Never reload the page at step 2+**: The SPA uses client-side routing. `page.reload()` destroys routing state.

## Debug

```bash
# Smoke test — runs 15 actions on step 1, dumps observations
python scripts/debug_run.py

# Explore challenges interactively with visible browser
python scripts/explore_challenges.py --interactive

# Capture AXTree snapshots of each step
python scripts/explore_challenges.py --save-snapshots
```
