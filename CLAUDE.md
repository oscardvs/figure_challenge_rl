# Figure RL — Browser Navigation Gauntlet Agent

## Project Overview
RL-trained browser agent that solves a 30-step sequential web navigation gauntlet. Each step is a unique puzzle (scroll, click, hover, fill, shadow DOM, canvas, audio, etc.) that reveals a 6-character code. The agent enters the code and clicks Submit to advance.

**Target site**: React SPA on Netlify with client-side routing. Direct URL navigation only works for step 1; all other steps require sequential traversal from START.

## Architecture
```
src/
  agent/       — LLM policy, MCTS search, prompts/action parsing
  environment/ — BrowserGym wrappers (GauntletEnv, StepEnv), action space, observation pruning
  solver/      — Deterministic challenge solver (21 challenge types), hybrid agent
  training/    — SFT, DPO, GRPO training + MCTS trajectory collection
  runner/      — solve_all.py (api + deterministic + hybrid + local modes), metrics, parallel executor
config/        — challenge_config.yaml, model_config.yaml
data/          — trajectories, preference_pairs, known_solutions.json
scripts/       — debug_run.py, explore_challenges.py
```

### Randomized Challenges
Challenges are **randomized per run** — challenge type X can appear at any step position, and codes are dynamically generated each time. This means `known_solutions.json` and `StepEnv` with `prior_solutions` are only useful for training on fixed challenge sets. For randomized challenges, use `GauntletEnv` (sequential from step 1) with the deterministic solver or hybrid agent.

## Key Files & What They Do
- `src/environment/browser_env.py` — GauntletEnv (full 30-step), StepEnv (single-step for training). Handles SPA content-wait (`_wait_for_content`), overlay dismissal, step transition detection, prior solution replay.
- `src/environment/action_space.py` — BrowserGym HighLevelActionSet with `js_eval` custom action. The `js_eval` function injects results into a visible DOM element for AXTree readback.
- `src/agent/prompts.py` — System prompt, action parser (`_ACTION_PATTERN` regex). The parser extracts actions from `<think>...</think>\naction()` format.
- `src/agent/policy.py` — `LLMPolicy` (API: Anthropic/OpenAI/Google) + `LocalPolicy` (local LoRA inference via Unsloth/PEFT).
- `src/agent/mcts.py` — MCTSSearch for trajectory + preference pair collection.
- `src/training/collect_trajectories.py` — Orchestrates MCTS collection across steps, accumulates `data/known_solutions.json`.
- `src/training/sft_train.py` — Supervised fine-tuning (Qwen3-4B + QLoRA via Unsloth).
- `src/training/dpo_train.py` — DPO training from preference pairs.
- `src/training/grpo_train.py` — Group-relative policy optimization with live rollouts.
- `src/solver/challenge_detector.py` — `ChallengeDetector` classifies the current page into one of 21 challenge types via a single `page.evaluate()` JS signal collection call.
- `src/solver/challenge_handlers.py` — `ChallengeHandlers` with 21 per-type handler methods + shared utilities (`clear_popups`, `fill_and_submit`, `extract_hidden_codes`, `deep_code_extraction`). All sync Playwright.
- `src/solver/deterministic_solver.py` — `DeterministicSolver` orchestrates the per-step solve loop: detect → handle → extract codes → submit → check progress.
- `src/solver/hybrid_agent.py` — `HybridAgent` wraps `DeterministicSolver` + optional `LLMPolicy`. Deterministic-first, LLM-fallback for each step.

## Important Patterns & Gotchas

### Step Transition: Context-Manager Protected Observation
BrowserGym's `set_of_marks` injects `bid`/`browsergym_visibility_ratio` attributes into ALL DOM elements during `_get_obs()`. This crashes React's virtual DOM reconciliation. The `_pre_extract_disabled()` context manager disables DOM mutation during `_env.step()` and guarantees restoration via `try/finally`, even on exceptions.

After step transitions, `_wait_for_content()` uses a multi-signal approach: first waits for `#root > *` mount, then waits for interactive elements (buttons, inputs) to confirm puzzle content rendered. `_get_reliable_obs()` retries observation extraction up to 3 times with exponential backoff.

### CRITICAL: Never Use el.remove() on React-Managed DOM Nodes
All popup/overlay dismissal MUST use CSS hiding (`display:none; visibility:hidden; pointer-events:none; zIndex:-1`), NOT `el.remove()`. Removing React-managed nodes desynchronizes the virtual DOM — when React later tries to unmount during step transitions, it throws `NotFoundError: Failed to execute 'removeChild'` and crashes permanently, leaving `#root` empty. The `clear_popups()` handler and `_DISMISS_OVERLAYS_JS` both use the `hide()` pattern. Only non-React elements (like `__bgym_js_result` appended directly to `document.body`) are safe to `.remove()`.

### CRITICAL: Never Reload the Page at Step 2+
Direct URL navigation only works for step 1. Calling `page.reload()` at step 2+ destroys the SPA's client-side routing state and leaves the page permanently broken (`#root` empty). The `_wait_for_page_ready()` and `_wait_for_content()` functions must never reload — they wait patiently and return best-effort if React is slow to render.

**Observation quality check**: `_is_obs_valid()` checks for interactive elements (button/textbox/input/etc.) AND minimum length > 500 chars, rather than raw length alone. Some legitimate steps produce short AXTrees.

The `__bgym_js_result` div is cleaned up on step transitions to prevent result accumulation ("JS Result: JS Result: ...").

Step transitions are detected via URL matching with DOM fallback: if the URL doesn't reflect a step change, the DOM is checked for "Step N of 30" text.

### Action Parser
`_ACTION_PATTERN` in `prompts.py` must include ALL valid BrowserGym actions. If a new action is added to the action set, it MUST be added to this regex or it will silently become `noop()`.

Current valid actions: `click`, `dblclick`, `fill`, `type`, `select_option`, `hover`, `press`, `focus`, `clear`, `scroll`, `drag_and_drop`, `upload_file`, `mouse_click`, `mouse_move`, `mouse_drag`, `mouse_upload_file`, `goto`, `go_back`, `go_forward`, `new_tab`, `tab_close`, `close_tab`, `tab_focus`, `js_eval`, `send_msg_to_user`, `report_infeasible`, `noop`.

### Custom Actions (js_eval)
BrowserGym custom actions receive `page` (Playwright Page) via `exec()` globals — NOT as a function parameter. The function body references `page` directly. Function must have a docstring with an `Examples:` section.

### Training Dependencies & Gotchas
- **Unsloth/vllm/trl versions must be compatible** — unsloth pins `trl<=0.24.0` and `datasets<4.4.0`. Upgrading one without checking constraints breaks imports.
- **HF Hub timeouts**: `HfFileSystem.glob()` in unsloth's loader uses `huggingface_hub.constants.HF_HUB_ETAG_TIMEOUT` (default 10s). Monkey-patch the constant directly before `from_pretrained()` — env vars may be evaluated at import time before your code sets them.
- **Unsloth `formatting_func`**: Must always return `list[str]`, even for single examples. Use `tokenizer.apply_chat_template()` per unsloth chat template docs.
- **Model downloads**: Set `HF_HUB_ENABLE_HF_TRANSFER=1` for fast HF downloads. Once cached, unsloth still hits HF API for metadata.

### Observation Token Budget (CRITICAL for DPO)
Observations have two sections: **AXTree** (interactive elements, pruned by `AXTreePruner`) and **HTML Content** (raw innerHTML). The HTML section is ~58% of the observation but is redundant with the AXTree — it's useful for SFT (where the agent learns to read HTML for shadow DOM / canvas challenges) but wastes tokens in DPO.

**Budget constraints on 16GB VRAM (RTX 4080 SUPER):**
- DPO requires dual forward passes (policy + ref) per training step
- At `max_seq_length=3072`, DPO OOMs during TRL's ref logprob precompute (the unsloth DPO trainer has a memory leak in its precompute loop that crashes at ~83% regardless of GC/cache clearing — the leak is in TRL/accelerator internals, not the model's forward pass)
- At `max_seq_length=2048`, dual forward passes use ~10GB — fits comfortably

**Solution: compact observations for training data collection:**
- `AXTreePruner(target_tokens=1200)` in `collect_expert_trajectories.py` (was 2000)
- `extract_html_snippet(max_chars=2000)` in `observation.py` (was 6000)
- `DPO_SYSTEM_PROMPT` in `prompts.py` — 73 tokens vs 1517 for `PURE_AGENT_SYSTEM_PROMPT` (action descriptions already learned in SFT)
- `dpo_train.py` strips the HTML Content section from observations (SFT still uses it at 4096 max_seq_length)
- Final DPO token budget: prompt ~1503 tokens + completion ~450 tokens = ~1953 tokens (fits in 2048)

**Do NOT try to fix the DPO OOM with `precompute_ref_log_probs=True`** — the unsloth/TRL precompute loop has a memory leak that accumulates through accelerator internals. The model's forward pass itself does NOT leak (verified: 0 growth over 200 iterations). The only reliable fix is reducing max_seq_length to 2048 via compact observations.

### Prior Solutions for Training
`StepEnv` accepts `prior_solutions: dict[int, str]` which `StepTask._replay_prior_steps()` uses to auto-navigate to the target step. Solutions accumulate in `data/known_solutions.json`.

## Commands

### Debug / Smoke Test
```bash
python scripts/debug_run.py  # 15 actions on step 1, dumps observations
```

### Baseline Run (full gauntlet)
```bash
python -m src.runner.solve_all --provider google      # Gemini Flash (API mode)
python -m src.runner.solve_all --provider anthropic    # Claude Sonnet (API mode)
python -m src.runner.solve_all --provider openai       # GPT-4o (API mode)
python -m src.runner.solve_all --mode deterministic    # Deterministic solver only (no LLM cost)
python -m src.runner.solve_all --mode hybrid --provider google  # Deterministic + LLM fallback
python -m src.runner.solve_all --mode local            # Fine-tuned Qwen3-4B (models/sft/)
python -m src.runner.solve_all --mode local --adapter-dir models/dpo  # DPO model
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
- Activate with `source .venv/bin/activate` before running any python commands
- API keys in `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- Playwright browsers: `playwright install chromium`
- Key deps: `browsergym-core`, `playwright`, `torch`, `transformers`, `trl`, `unsloth`, `vllm`

## Expert Trajectory Pipeline (Pure Learned Agent)

### Overview
The deterministic solver acts as a **teacher**: it generates expert trajectories that a Qwen3-4B model learns from. At inference time, the trained agent solves challenges from observations alone — no hardcoded detectors or handlers.

### Pipeline
1. **Collect expert trajectories**: `python -m src.training.collect_expert_trajectories`
   - Runs solver with `RecordingPage` proxy that intercepts Playwright calls and maps to BrowserGym actions.
   - Output: `data/expert_trajectories/step_NN.json` + `data/expert_preference_pairs/step_NN.json`
2. **Add CoT reasoning**: `python -m src.training.generate_cot --provider local`
   - Annotates each (obs, action) pair with 2-4 sentence reasoning.
   - `--provider local` uses Qwen3-14B-AWQ via vLLM batched inference (free, ~15-25 min for all steps).
   - `--provider google` uses Gemini Flash API (costs ~$1-2, ~2.5 hours sequential).
   - Output: Updated trajectory files with `reasoning` field. Cache: `data/cot_cache.json`.
3. **SFT**: `python -m src.training.sft_train --expert-dir data/expert_trajectories`
   - Combines MCTS + expert trajectories. Expert trajectories use `PURE_AGENT_SYSTEM_PROMPT`.
4. **DPO**: `python -m src.training.dpo_train --expert-pairs-dir data/expert_preference_pairs`
   - Combines MCTS + expert preference pairs. Uses `DPO_SYSTEM_PROMPT` (compact) and strips HTML from observations to fit in 2048 max_seq_length.
5. **M-GRPO**: `python -m src.training.grpo_train`
   - Uses `GauntletEnv` (randomized challenges) with fractional reward and curriculum learning.

### Key Files
- `src/agent/trajectory_recorder.py` — `RecordingPage` proxy (Playwright → BrowserGym action mapping)
- `src/agent/prompts.py` — `PURE_AGENT_SYSTEM_PROMPT` (SFT, no challenge-type hints) + `DPO_SYSTEM_PROMPT` (compact, no action descriptions)
- `src/training/collect_expert_trajectories.py` — Expert trajectory collection
- `src/training/generate_cot.py` — Synthetic chain-of-thought annotation
- `src/environment/observation.py` — `extract_html_snippet()` for enhanced observations

### Data Directories
- `data/expert_trajectories/` — Solver-generated trajectories in SFT format
- `data/expert_preference_pairs/` — Success/failure contrast pairs for DPO
- `data/cot_cache.json` — API response cache for CoT generation

## Code Style
- Type hints everywhere (`from __future__ import annotations`)
- Logging via `logging.getLogger(__name__)`
- Config-driven via YAML files in `config/`
- No unnecessary abstractions — keep environment wrappers thin over BrowserGym
