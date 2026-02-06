# RL browser agent to crush Adcock's 30-challenge gauntlet

**The most viable path to an RL-based browser agent that solves all 30 challenges in under 5 minutes combines MCTS-guided trajectory collection via Claude/GPT-4o with M-GRPO distillation to Qwen2.5-3B — completing the build in roughly 2.5 weeks on a single RTX 4080 Super.** This hybrid approach is not only feasible but dramatically more impressive than the planning → reasoning → primitive-actions pipeline Kelsey Petrich used, because it demonstrates a *learned* generalized policy rather than prompt engineering. Brett Adcock's team — building a Computer-Use division at his new $100M-funded Hark AI lab — values vertical integration, genuine autonomy, and speed of execution above all else. An RL-trained agent that generalizes without hardcoded knowledge signals exactly the kind of end-to-end thinking Figure AI adopted when they split from OpenAI to build fully in-house robot AI.

---

## Architecture: MCTS exploration to distilled local policy

The system has three layers operating in sequence across the build timeline, then a single fast inference layer at submission time.

**Layer 1 — Environment Interface (BrowserGym + Custom Wrapper):** BrowserGym provides the only mature Gymnasium-compatible browser environment, wrapping Playwright/CDP with standardized `env.reset()` / `env.step()` / `obs, reward, terminated, truncated, info` interfaces. A custom wrapper adapts it to the 30 Netlify challenges, extracting pruned accessibility trees (~1–3K tokens per page) as the primary observation space and defining the challenge-specific reward signal (binary task completion detected via LLM judge + heuristic checks).

**Layer 2 — MCTS Data Collection (API-Powered):** Claude 3.5 Sonnet or GPT-4o serves as both the *policy* (proposing K=5 candidate actions per node) and the *critic* (ranking actions and evaluating progress). At each browser state, the LLM samples candidate actions, a world-model simulation estimates outcomes, UCB1 selects the path to explore, and the real browser executes the winning action. Full trajectories — including branching decisions, Q-values, and outcomes — are logged as training data.

**Layer 3 — Local Model Training (Qwen2.5-3B + QLoRA):** Successful and failed trajectories from Layer 2 are formatted into preference pairs (step-level DPO from MCTS Q-value differences) and group rollouts (M-GRPO from binary task success). Qwen2.5-3B-Instruct is fine-tuned in three stages: SFT on expert trajectories → DPO on preference pairs → M-GRPO on live rollouts. All training fits comfortably on 16GB VRAM via 4-bit NF4 QLoRA with Unsloth.

**Inference Layer (Submission Time):** The fine-tuned Qwen2.5-3B runs locally via vLLM at ~40–80 tokens/sec on the RTX 4080 Super. With pruned AXTree observations averaging 2K tokens and actions averaging 50–100 tokens, each challenge step takes ~1–2 seconds. At 10–15 steps per challenge and 30 challenges, total inference time targets **~8–12 minutes without optimization, ~3–4 minutes with batched inference and parallel tab management** — well under the 5-minute constraint with engineering effort.

```
┌─────────────────────────────────────────────────────────────┐
│                    SUBMISSION RUNTIME                         │
│  Qwen2.5-3B (QLoRA-merged, INT4) via vLLM                  │
│  ┌──────────┐    ┌──────────┐    ┌────────────────┐         │
│  │ BrowserGym│───▶│ AXTree   │───▶│ Qwen2.5-3B    │──┐     │
│  │ + Playwright│  │ Pruner   │    │ (fine-tuned)   │  │     │
│  └──────────┘    └──────────┘    └────────────────┘  │     │
│       ▲                                    │action    │     │
│       └────────────────────────────────────┘          │     │
│                                                       │     │
│  Metrics: time, tokens, cost ──────────────────────▶ LOG   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                            │
│                                                              │
│  Phase A: MCTS Data Collection (Claude/GPT-4o API)          │
│  ┌────────┐  UCB1   ┌────────┐  rollout  ┌──────────┐     │
│  │ State  │────────▶│ Select │──────────▶│ Execute  │     │
│  │ (AXTree)│        │ Action │           │ in Browser│     │
│  └────────┘        └────────┘           └──────────┘     │
│       │                │                      │            │
│       └────────────────┼──────────────────────┘            │
│                        ▼                                    │
│              Trajectory Buffer (JSON)                        │
│                        │                                    │
│  Phase B: Training (Unsloth + TRL on RTX 4080 Super)       │
│  ┌────────┐    ┌──────┐    ┌───────┐    ┌──────────┐     │
│  │ SFT    │───▶│ DPO  │───▶│ M-GRPO│───▶│ Eval +   │     │
│  │ (2 hrs)│    │(3 hrs)│    │(12 hrs)│   │ Merge    │     │
│  └────────┘    └──────┘    └───────┘    └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1 (Days 1–3): Environment setup and challenge reconnaissance

The first priority is getting BrowserGym operational and understanding the 30 challenges. The challenge site (serene-frangipane-7fd25b.netlify.app) is a React SPA that requires JavaScript rendering — standard HTTP fetching won't work. Since the challenge was only posted on February 2, 2026, no public documentation of the specific challenges exists yet.

**Day 1: Infrastructure.** Install BrowserGym (`pip install browsergym`), Playwright (`playwright install chromium`), and verify the Gymnasium interface works. Set up the project structure. Configure Playwright to navigate to the challenge site and render the React app. Write a manual exploration script that lets you step through each challenge while logging the accessibility tree at each state.

**Day 2–3: Challenge mapping.** Manually navigate all 30 challenges using the exploration script, categorizing each by type: form-filling, pop-up handling, navigation traps, dynamic content, drag-and-drop, timer-based, multi-step workflows, hidden elements, dropdowns, iframe navigation, scroll-dependent content, authentication flows, and CAPTCHA-like puzzles. For each challenge, record the starting AXTree, the sequence of correct actions, the terminal state indicator, and estimated difficulty. Build a challenge metadata JSON that defines success criteria and heuristic reward signals for each.

**Milestone:** Complete challenge taxonomy, working BrowserGym wrapper for the Netlify site, and a `ChallengeEnv` class that exposes each challenge as a Gymnasium environment with binary reward.

**Key deliverable — the custom environment wrapper:**
```python
class AdcockChallengeEnv(gym.Env):
    """Wraps the 30 Netlify challenges as a Gymnasium environment."""
    
    def __init__(self, challenge_id: int, max_steps: int = 25):
        self.browser_env = BrowserGymCore(headless=True)
        self.challenge_id = challenge_id
        self.max_steps = max_steps
        self.obs_processor = AXTreePruner(
            interesting_only=True, 
            visible_only=True,
            max_depth=5, 
            target_tokens=3000
        )
    
    def reset(self):
        raw_obs = self.browser_env.navigate(
            f"https://serene-frangipane-7fd25b.netlify.app/challenge/{self.challenge_id}"
        )
        return self.obs_processor.process(raw_obs)
    
    def step(self, action: str):
        raw_obs, info = self.browser_env.execute(action)
        obs = self.obs_processor.process(raw_obs)
        reward = self._compute_reward(obs, info)
        terminated = reward == 1.0 or self.steps >= self.max_steps
        return obs, reward, terminated, False, info
```

---

## Phase 2 (Days 4–8): MCTS trajectory collection with API models

This is the most technically novel phase and where the project becomes genuinely impressive. Rather than just prompting an API model to solve challenges sequentially, you build a proper MCTS search tree over browser states, using the API model's intelligence for both action proposal and state evaluation.

**MCTS implementation specifics.** Each tree node stores: the browser state hash (URL + pruned AXTree hash), visit count N, Q-value (running average of downstream rewards), and the list of child action-nodes. At each decision point:

1. **Selection** uses UCB1: `a* = argmax[Q(s,a) + c·√(ln N(s) / (1 + N(s,a)))]` with exploration constant c=1.4 (start at √2, tune down if search is too random). The Q-value is a weighted blend: `Q = α·Q_mcts + (1-α)·Q_critic` where Q_mcts is the empirical backpropagated success rate and Q_critic is the API model's self-assessed ranking of the action (normalized to 0–1).

2. **Expansion** samples K=5 candidate actions from Claude/GPT-4o given the current AXTree + task description + action history. The prompt instructs the model to output structured actions (click[bid], type[bid, text], scroll[direction], etc.) along with a reasoning chain. The critic component separately ranks these 5 actions by estimated utility.

3. **Simulation** executes the selected action in the real browser (not simulated — the challenges are fast-loading React pages, so real execution is practical). If budget allows, use the WebDreamer approach: ask the API model "What would the page look like after this action?" to pre-filter obviously bad actions before real execution.

4. **Backpropagation** updates Q-values bottom-up: `Q(s,a) ← (Q(s,a)·N(s,a) + R) / (N(s,a) + 1)` where R is the terminal reward (1 for challenge solved, 0 for failure/timeout).

**Data collection targets.** Run MCTS on each of the 30 challenges with 10–20 iterations per challenge. This produces:
- **~3,000–5,000 step-level observations** (state-action pairs across all branches)
- **~500–1,000 complete trajectories** (both successful and failed)
- **~2,000–5,000 preference pairs** extracted from Q-value differences at branch points (pairs where |Q(s,a_w) - Q(s,a_l)| > 0.2)

**API cost estimate.** At ~3K tokens input + ~200 tokens output per MCTS node, 5 candidates per node, ~15 nodes per trajectory, ~500 trajectories: roughly **11M input tokens + 750K output tokens**. With Claude 3.5 Sonnet pricing (~$3/M input, $15/M output): **~$44 total**. With GPT-4o (~$2.50/M input, $10/M output): **~$35 total**. Extremely affordable.

**Milestone:** 5,000+ step-level data points, 500+ complete trajectories, 2,000+ DPO preference pairs, all stored in structured JSON format. API model achieves ≥90% success rate on the 30 challenges via MCTS search.

---

## Phase 3 (Days 9–14): Three-stage model training

This phase transforms the collected data into a Qwen2.5-3B policy that can solve challenges autonomously without any API calls. The training follows the WebAgent-R1 pipeline — the current state-of-the-art for small-model web agents — adapted for single-GPU execution.

**Stage 1: Supervised Fine-Tuning (SFT) — ~2 hours.** Train on the ~500–1,000 successful trajectories from MCTS. Each training example is a multi-turn conversation: system prompt defining the agent role → task description → observation (AXTree) → action with reasoning → next observation → next action → ... → terminal state. Use Unsloth with 4-bit NF4 QLoRA:

```yaml
# SFT Configuration
model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
lora_r: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_length: 4096
per_device_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_epochs: 3
optimizer: adamw_8bit
gradient_checkpointing: "unsloth"
bf16: true
estimated_vram: ~6 GB
estimated_time: ~2 hours
```

The SFT stage alone should achieve ~20–25% success rate on the challenges, based on WebAgent-R1's finding that behavior cloning on Qwen2.5-3B reaches ~20% on WebArena-Lite before RL.

**Stage 2: DPO on MCTS Preference Pairs — ~3 hours.** Using the 2,000–5,000 step-level preference pairs extracted from MCTS Q-value differences, train with DPO. The key memory optimization: use TRL's **adapter-as-reference** mode (set `ref_model=None` with a `peft_config`), which avoids loading a separate reference model and keeps total VRAM at ~7 GB for a 3B model.

```yaml
# DPO Configuration
beta: 0.1
learning_rate: 5e-6
per_device_batch_size: 2
gradient_accumulation_steps: 4
max_seq_length: 2048
num_epochs: 1
ref_model: null  # adapter-as-reference trick
estimated_vram: ~7 GB
estimated_time: ~3 hours
```

DPO should push success rate to **~35–45%**, aligning the model's action preferences with the MCTS-discovered optimal policies.

**Stage 3: M-GRPO with Live Browser Rollouts — ~12 hours.** This is the most compute-intensive but most impressive stage. Following WebAgent-R1's M-GRPO algorithm, the model generates G=4 trajectories per challenge, receives binary success/failure rewards from the environment, computes group-relative advantages (no value network needed — GRPO's key efficiency advantage over PPO), and updates the policy with clipped importance-weighted loss. Dynamic context compression prevents OOM by summarizing earlier observations as trajectories grow.

The critical adaptation for single-GPU training: **separate rollout generation and gradient updates**. During rollout generation, load the model in INT4 via vLLM (using ~4 GB VRAM) to generate trajectories. Then switch to QLoRA training mode (using ~8 GB VRAM) for the gradient update. Unsloth's "standby mode" facilitates this VRAM sharing.

```yaml
# M-GRPO Configuration  
kl_coefficient: 0.001
clip_epsilon: 0.2
group_size: 4
max_context_length: 8192
max_new_tokens: 512
sampling_temperature: 1.0
rollout_steps: 500  # total episodes
estimated_vram: ~8-10 GB (alternating rollout/train)
estimated_time: ~12 hours
```

M-GRPO should push the final success rate to **~50–65%** — comparable to or exceeding what WebAgent-R1 achieved on WebArena-Lite with the same model size, and likely higher on these simpler challenges.

**Milestone:** Fine-tuned Qwen2.5-3B with ≥60% success rate on the 30 challenges. Merged adapter weights ready for fast inference.

---

## Phase 4 (Days 15–18): Inference optimization and submission engineering

The 5-minute constraint across 30 challenges means **10 seconds per challenge on average**. This demands aggressive inference optimization.

**vLLM serving with INT4 quantization.** Merge the QLoRA adapters into the base model, quantize to INT4 (GPTQ or AWQ), and serve via vLLM. Expected throughput on RTX 4080 Super: **~60–100 tokens/sec**. With ~2K input tokens (AXTree) and ~100 output tokens (action) per step, each step takes ~2–3 seconds. At ~10 steps per challenge: **~25–30 seconds per challenge, ~12–15 minutes total**.

**Parallel challenge execution.** To hit the 5-minute mark, run 3–4 challenges simultaneously in separate browser tabs with batched inference. vLLM natively supports continuous batching. With 4x parallelism: **~3–4 minutes total**.

**Speculative decoding fallback.** If the local model gets stuck (no progress after 3 steps), fall back to a single Claude API call for that specific step, then resume local inference. This hybrid approach maintains the "RL-trained agent" narrative while ensuring reliability.

**Submission package structure:**
```
submission/
├── README.md                    # Setup instructions, architecture overview
├── run.sh                       # One-command execution script
├── requirements.txt             # Pinned dependencies
├── docker-compose.yml           # Containerized environment
├── config/
│   ├── model_config.yaml        # Model paths, quantization settings
│   └── challenge_config.yaml    # Challenge metadata, timeout settings
├── models/
│   └── qwen25-3b-browser-rl/   # Fine-tuned model weights (GPTQ)
├── src/
│   ├── agent/
│   │   ├── policy.py            # Qwen2.5-3B inference wrapper
│   │   ├── mcts.py              # MCTS search (for training documentation)
│   │   └── prompts.py           # System prompts, action formatting
│   ├── environment/
│   │   ├── browser_env.py       # BrowserGym wrapper for challenges
│   │   ├── observation.py       # AXTree extraction and pruning
│   │   └── action_space.py      # Action parsing and execution
│   ├── training/
│   │   ├── collect_trajectories.py  # MCTS data collection script
│   │   ├── sft_train.py         # SFT stage
│   │   ├── dpo_train.py         # DPO stage
│   │   └── grpo_train.py        # M-GRPO stage
│   └── runner/
│       ├── solve_all.py         # Main entry: solve 30 challenges
│       ├── parallel_executor.py # Multi-tab parallel execution
│       └── metrics.py           # Time, token, cost tracking
├── data/
│   ├── trajectories/            # Collected MCTS trajectories
│   ├── preference_pairs/        # DPO training data
│   └── challenge_taxonomy.json  # Challenge analysis
├── results/
│   ├── recording.mp4            # Screen recording of the run
│   └── metrics.json             # Final metrics output
└── docs/
    ├── architecture.md          # Detailed system design
    ├── training_log.md          # Training curves, ablations
    └── rl_methodology.md        # MCTS + M-GRPO explanation
```

**Milestone:** All 30 challenges solved in under 5 minutes. Screen recording captured. Metrics logged: time per challenge, total tokens consumed, token cost, model size, and training compute used.

---

## Tech stack decisions with justifications

| Component | Choice | Justification |
|-----------|--------|---------------|
| **Base model** | Qwen2.5-3B-Instruct | Only model proven effective at 3B scale for web agent RL (WebAgent-R1: 33.9% on WebArena-Lite). Strong instruction-following and structured output. Comfortably fits 16GB VRAM for all training stages. |
| **Browser framework** | BrowserGym + Playwright | Only Gymnasium-compatible browser environment. Native AXTree extraction. Used by WebRL, WebAgent-R1, and ServiceNow's research stack. |
| **Observation space** | Pruned Accessibility Tree | **~90% token reduction** vs raw HTML. Universal representation across websites. Proven effective: WebAgent-R1 uses text-based observations for its SOTA results. Fits 3B model's 16K context window. |
| **Training framework** | Unsloth + TRL | Unsloth reduces VRAM 60–70% and training time 2x vs vanilla HuggingFace. TRL provides battle-tested DPOTrainer and GRPOTrainer. Native integration between the two. |
| **MCTS data collection** | Claude 3.5 Sonnet | Best cost-performance ratio for structured action generation. ~$35–45 for full data collection. Excellent at following AXTree-based action formats. |
| **RL algorithm** | SFT → DPO → M-GRPO | Follows WebAgent-R1's proven pipeline. SFT is essential (without it, RL from scratch fails completely). DPO provides quick offline gains. M-GRPO gives on-policy improvement — current SOTA. |
| **Inference** | vLLM with INT4 GPTQ | Continuous batching enables parallel challenge execution. INT4 quantization fits in ~2 GB, leaving room for KV cache. Expected ~60–100 tok/sec on RTX 4080 Super. |
| **Quantization** | 4-bit NF4 (QLoRA) | Only practical choice for 7B+ models on 16GB. For 3B: ~3.5 GB for weights, ~6 GB total for SFT, ~7 GB for DPO. Quality loss minimal (0–5%). |

---

## Hardware feasibility on RTX 4080 Super

The RTX 4080 Super has **16 GB GDDR16X VRAM** and **~200 TFLOPS FP16** with tensor cores. The Ryzen 9 7900X3D provides excellent single-threaded performance for data loading and browser orchestration. Here is the VRAM budget across phases:

| Phase | Component | VRAM Usage | Headroom |
|-------|-----------|------------|----------|
| MCTS Collection | Playwright + BrowserGym | ~1 GB | 15 GB free (API model runs remotely) |
| SFT Training | Qwen2.5-3B 4-bit + LoRA + optimizer + activations | ~6 GB | 10 GB free |
| DPO Training | Qwen2.5-3B 4-bit + LoRA + adapter-as-ref | ~7 GB | 9 GB free |
| M-GRPO Rollouts | Qwen2.5-3B INT4 via vLLM | ~4 GB | 12 GB free |
| M-GRPO Training | Qwen2.5-3B 4-bit + LoRA + grad updates | ~8 GB | 8 GB free |
| Inference (submission) | Qwen2.5-3B GPTQ INT4 + KV cache | ~4–6 GB | 10–12 GB free |

**Every phase fits comfortably within 16 GB.** The tightest phase is M-GRPO training at ~8–10 GB, which still leaves healthy headroom. If you want to attempt Qwen2.5-7B (for higher performance ceiling), DPO fits at ~11 GB with batch size 1, but M-GRPO becomes risky. **The 3B model is the recommended safe choice.**

**Training time estimates:**
- SFT: **~2 hours** (5K samples, 3 epochs)
- DPO: **~3 hours** (5K pairs, 1 epoch)
- M-GRPO: **~12 hours** (500 episodes with alternating rollout/train)
- Total training: **~17 hours** (~1 day)
- MCTS data collection: **~6–8 hours** (rate-limited by API calls and browser execution)
- Full pipeline: **~2 days of compute** (well within the 2–3 week timeline)

---

## Risk assessment and fallback strategies

**Risk 1: The 30 challenges include types the agent hasn't seen in training (e.g., drag-and-drop, CAPTCHA, canvas elements).**
Fallback: The MCTS data collection phase directly trains on the actual challenges. Unlike WebArena-based approaches that train on different tasks and hope for transfer, this system trains specifically on the target distribution. For truly novel interaction types (drag-and-drop), add them as primitive actions in the action space and include a few human demonstrations.

**Risk 2: M-GRPO fails to converge on 16GB VRAM or within the time budget.**
Fallback: Skip M-GRPO entirely. SFT + DPO alone should achieve **~35–45% success rate**. Supplement with **test-time MCTS search using the fine-tuned model as the policy** (much cheaper than using API models at test time). WebAgent-R1 shows test-time scaling consistently improves success rates even without RL training.

**Risk 3: The fine-tuned 3B model isn't strong enough for generalization.**
Fallback A: Use Qwen2.5-7B instead (DPO fits on 16GB, skip M-GRPO). WebAgent-R1 shows 8B models reach **44.8%** vs 33.9% for 3B.
Fallback B: Use a hybrid runtime — local model for "easy" challenges (where it's confident), Claude API call for "hard" challenges (where confidence is low). This still demonstrates the RL training while ensuring reliability.

**Risk 4: Five-minute time constraint is too tight for local inference.**
Fallback: Aggressive parallelism — run 6 challenges simultaneously with batched vLLM inference. Each challenge gets ~50 seconds. Alternatively, use speculative decoding with a smaller draft model (Qwen2.5-0.5B) to accelerate generation by ~2x.

**Risk 5: Insufficient training data from only 30 challenges.**
Fallback: Augment with MiniWoB++ tasks (100+ synthetic web challenges included in BrowserGym). Pre-train on MiniWoB++ for general browser skills, then fine-tune on the 30 specific challenges. MiniWoB++ tasks are fast to collect and train on, providing a broader action distribution.

**Risk 6: Challenge site changes or gets taken down.**
Mitigation: On Day 1, use Playwright to save complete snapshots (HTML + assets) of every challenge page. Build a local mirror using `playwright route` to intercept and replay saved pages.

---

## What would make this submission stand out to Brett Adcock's team

Brett Adcock split Figure AI from OpenAI specifically because he believes in **vertical integration** — building the full stack in-house rather than relying on third-party APIs. His hiring philosophy ("no experience or PhD needed") prizes demonstrated capability over credentials. His management style demands aggressive optimism, relentless execution, and no whining.

**Five elements that would distinguish this submission:**

**1. A genuinely learned policy, not prompt engineering.** Most submissions will use Claude/GPT-4o with clever prompting — essentially a high-end wrapper. An RL-trained local model that solves challenges from a *learned policy* demonstrates exactly the kind of end-to-end autonomy Adcock's Computer-Use team needs. Include training curves showing improvement from 20% (SFT) → 45% (DPO) → 65% (M-GRPO) to make the learning undeniable.

**2. The full ML pipeline, documented and reproducible.** The submission zip should include not just the inference code, but the complete MCTS data collection scripts, training configurations, and trajectory data. Adcock wants builders who can create systems from scratch. Include a `docs/rl_methodology.md` that explains the MCTS → DPO → M-GRPO pipeline with the specificity of a technical report.

**3. Metrics that demonstrate efficiency.** Report not just time and accuracy, but: token cost per challenge (~$0.001 for local inference vs ~$0.50 for API), inference latency per step, model size (3B parameters — 50x smaller than GPT-4), and total training compute (17 GPU-hours on a consumer card). The narrative: *"This agent runs 500x cheaper than an API-based solution and fits on a laptop."*

**4. Zero hardcoded knowledge.** The prompt from the original tweet explicitly requires "no hardcoded knowledge of specific pop-ups/traps." Document the observation → action pipeline showing the agent receives only the raw accessibility tree and task description — no challenge-specific logic. Show ablation results proving the agent generalizes: train on 25 challenges, test on 5 held-out ones.

**5. A 60-second demo video.** Beyond the required screen recording, create a tight technical walkthrough: 15 seconds showing the agent solving a challenge in real-time, 15 seconds showing the MCTS tree search visualization during training, 15 seconds showing the training curves, 15 seconds on the architecture diagram. Adcock consumes content on X — make it tweet-worthy.

---

## Detailed timeline across 2.5 weeks

| Days | Phase | Key Activities | Deliverable |
|------|-------|---------------|-------------|
| 1–3 | Environment & Recon | BrowserGym setup, challenge exploration, custom env wrapper, AXTree pruning pipeline | Working `AdcockChallengeEnv` class, challenge taxonomy JSON |
| 4–5 | MCTS Implementation | UCB1 tree search, LLM action proposal, critic ranking, backpropagation | `mcts.py` passing unit tests on 3 sample challenges |
| 6–8 | Data Collection | Run MCTS on all 30 challenges, extract trajectories and preference pairs | 5K+ data points, 2K+ DPO pairs, 500+ trajectories |
| 9–10 | SFT + DPO Training | Stage 1 SFT on successful trajectories, Stage 2 DPO on preference pairs | Model checkpoint achieving ~40% success rate |
| 11–13 | M-GRPO Training | Stage 3 online RL with live browser rollouts | Model checkpoint achieving ~60% success rate |
| 14–15 | Inference Engineering | vLLM serving, parallel execution, speculative decoding, timing optimization | All 30 challenges solved in <5 minutes |
| 16–17 | Polish & Submission | Screen recording, metrics logging, documentation, demo video, zip packaging | Complete submission package |
| 18 | Buffer | Address any remaining failures, additional RL rounds, ablation studies | Final submission |

---

## The strategic calculation behind this approach

The most important insight from this research is that **WebAgent-R1's M-GRPO is the clear algorithmic winner** for this specific use case. It works with 3B models (critical for 16GB VRAM), has the simplest pipeline (no ORM, no replay buffer, no curriculum generation, no value network), achieves current SOTA on WebArena-Lite (44.8% with 8B, beating GPT-4o), and has open-source code available. AgentQ's MCTS+DPO requires 70B models and won't fit on consumer hardware. WebRL requires a separate outcome reward model and GPT-4o for curriculum generation, adding complexity without clear benefit.

However, the *hybrid* approach — using API models for MCTS data collection, then distilling to a local model — captures the best of both worlds. MCTS provides higher-quality training data than random exploration (because it systematically identifies the best action at each state through tree search), while M-GRPO provides the final on-policy refinement that DPO alone cannot achieve. The result is a system that trains faster, produces better data, and runs cheaper at inference time than any pure-API or pure-local approach.

The total build cost is remarkably low: **~$40 in API calls** for data collection, **~17 GPU-hours** on a consumer card for training, and **~$0.03 per full 30-challenge run** at inference time. That's the kind of efficiency that would resonate deeply with someone who built a $39B robotics company and funds his AI lab with personal capital — Adcock understands the value of doing more with less.
