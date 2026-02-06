# Figure RL — Browser Navigation Gauntlet Agent

## Project Overview
RL-trained browser agent that solves a 30-step sequential web navigation gauntlet. Each step is a unique puzzle (scroll, click, hover, fill, shadow DOM, canvas, audio, etc.) that reveals a 6-character code. The agent enters the code and clicks Submit to advance.

**Target site**: React SPA on Netlify with client-side routing. Direct URL navigation only works for step 1; all other steps require sequential traversal from START.

## Architecture
```
src/
  agent/       — LLM policy, MCTS search, prompts/action parsing
  environment/ — BrowserGym wrappers (GauntletEnv, StepEnv), action space, observation pruning
  training/    — SFT, DPO, GRPO training + MCTS trajectory collection
  runner/      — solve_all.py (baseline), metrics, parallel executor
config/        — challenge_config.yaml, model_config.yaml
data/          — trajectories, preference_pairs, known_solutions.json
scripts/       — debug_run.py, explore_challenges.py
```

## Key Files & What They Do
- `src/environment/browser_env.py` — GauntletEnv (full 30-step), StepEnv (single-step for training). Handles SPA content-wait (`_wait_for_content`), overlay dismissal, step transition detection, prior solution replay.
- `src/environment/action_space.py` — BrowserGym HighLevelActionSet with `js_eval` custom action. The `js_eval` function injects results into a visible DOM element for AXTree readback.
- `src/agent/prompts.py` — System prompt, action parser (`_ACTION_PATTERN` regex). The parser extracts actions from `<think>...</think>\naction()` format.
- `src/agent/policy.py` — LLMPolicy wrapping Anthropic/OpenAI/Google API clients.
- `src/agent/mcts.py` — MCTSSearch for trajectory + preference pair collection.
- `src/training/collect_trajectories.py` — Orchestrates MCTS collection across steps, accumulates `data/known_solutions.json`.
- `src/training/sft_train.py` — Supervised fine-tuning (Qwen2.5-3B + QLoRA via Unsloth).
- `src/training/dpo_train.py` — DPO training from preference pairs.
- `src/training/grpo_train.py` — Group-relative policy optimization with live rollouts.

## Important Patterns & Gotchas

### SPA Content Wait + Reload Fallback
The React SPA frequently fails to re-mount into `<div id="root">` after client-side route changes (BrowserGym's DOM instrumentation breaks React's virtual DOM reconciliation). `_wait_for_content()` polls `#root.children.length > 0` for 5s. **If React doesn't mount, it reloads the page** (`page.reload(wait_until="networkidle")`) to force fresh hydration, then waits again. This is critical — without the reload, step 2+ observations are always empty (194 chars). Both `reset()` and `step()` call this on transitions, then re-extract obs via `_env._get_obs()`. The `__bgym_js_result` div is also cleaned up on step transitions to avoid polluting the next step.

### Action Parser
`_ACTION_PATTERN` in `prompts.py` must include ALL valid BrowserGym actions. If a new action is added to the action set, it MUST be added to this regex or it will silently become `noop()`.

Current valid actions: `click`, `dblclick`, `fill`, `select_option`, `hover`, `press`, `focus`, `clear`, `scroll`, `drag_and_drop`, `upload_file`, `goto`, `go_back`, `go_forward`, `new_tab`, `tab_close`, `tab_focus`, `js_eval`, `send_msg_to_user`, `noop`.

### Custom Actions (js_eval)
BrowserGym custom actions receive `page` (Playwright Page) via `exec()` globals — NOT as a function parameter. The function body references `page` directly. Function must have a docstring with an `Examples:` section.

### Prior Solutions for Training
`StepEnv` accepts `prior_solutions: dict[int, str]` which `StepTask._replay_prior_steps()` uses to auto-navigate to the target step. Solutions accumulate in `data/known_solutions.json`.

## Commands

### Debug / Smoke Test
```bash
python scripts/debug_run.py  # 15 actions on step 1, dumps observations
```

### Baseline Run (full gauntlet)
```bash
python -m src.runner.solve_all --provider google      # Gemini Flash
python -m src.runner.solve_all --provider anthropic    # Claude Sonnet
python -m src.runner.solve_all --provider openai       # GPT-4o
```
Results written to `results/metrics.json`.

### Trajectory Collection
```bash
python -m src.training.collect_trajectories --provider google
python -m src.training.collect_trajectories --steps 1 2 3 --provider google
```

### Training Pipeline
```bash
python -m src.training.sft_train       # SFT on successful trajectories
python -m src.training.dpo_train       # DPO from preference pairs
python -m src.training.grpo_train      # GRPO with live rollouts
```

### Quick Run
```bash
./run.sh                          # default API mode
./run.sh --provider google        # use Gemini
```

## Environment Setup
- Python 3.12, virtualenv at `.venv/`
- API keys in `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- Playwright browsers: `playwright install chromium`
- Key deps: `browsergym-core`, `playwright`, `torch`, `transformers`, `trl`, `unsloth`, `vllm`

## Code Style
- Type hints everywhere (`from __future__ import annotations`)
- Logging via `logging.getLogger(__name__)`
- Config-driven via YAML files in `config/`
- No unnecessary abstractions — keep environment wrappers thin over BrowserGym
