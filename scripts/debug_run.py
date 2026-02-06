#!/usr/bin/env python3
"""Debug script: run 3 actions on step 1 and dump everything."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.agent.policy import LLMPolicy
from src.agent.prompts import format_gauntlet_task_prompt
from src.environment.action_space import get_action_description
from src.environment.browser_env import GauntletEnv

env = GauntletEnv(headless=True, max_actions_per_step=30)
obs_text, info = env.reset()

print("=" * 80)
print("INITIAL OBSERVATION (first 2000 chars):")
print("=" * 80)
print(obs_text[:2000])
print(f"\n[obs length: {len(obs_text)} chars]")
print(f"[info: {info}]")

task_prompt = format_gauntlet_task_prompt(1)
action_desc = get_action_description()

policy = LLMPolicy(
    provider="google",
    model="gemini-3-flash-preview",
    action_description=action_desc,
    max_tokens=2048,
)

action_history = []

for i in range(15):
    print(f"\n{'=' * 80}")
    print(f"ACTION {i + 1}")
    print("=" * 80)

    action, reasoning = policy.select_action(
        obs_text=obs_text,
        task_prompt=task_prompt,
        action_history=action_history[-10:],
        step=i,
    )

    print(f"\nPARSED ACTION: {repr(action)}")

    try:
        obs_text, reward, terminated, truncated, info = env.step(action)
        action_history.append(action)
        task_info = info.get("task_info", info)
        print(f"RESULT: reward={reward}, step={task_info.get('current_step')}")
        # Show key parts of observation.
        print(f"OBS: {len(obs_text)} chars")
        # If observation is very short, print it all (likely a page transition issue).
        if len(obs_text) < 500:
            print(f"  FULL OBS:\n{obs_text}")
        else:
            for line in obs_text.split("\n"):
                ll = line.strip().lower()
                if any(k in ll for k in ("textbox", "submit", "code", "challenge", "scroll to", "hidden", "step ", "error", "intercept", "timer", "wait", "reveal")):
                    print(f"  >> {line.strip()}")
    except Exception as e:
        print(f"\nACTION FAILED: {e}")
        break

    if terminated or truncated:
        break

env.close()
print("\nDone.")
