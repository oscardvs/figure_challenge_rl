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
)

action_history = []

for i in range(3):
    print(f"\n{'=' * 80}")
    print(f"ACTION {i + 1}")
    print("=" * 80)

    action, reasoning = policy.select_action(
        obs_text=obs_text,
        task_prompt=task_prompt,
        action_history=action_history[-10:],
        step=i,
    )

    print(f"\nLLM RESPONSE (full):\n{reasoning[:1500]}")
    print(f"\nPARSED ACTION: {repr(action)}")

    try:
        obs_text, reward, terminated, truncated, info = env.step(action)
        action_history.append(action)
        print(f"\nRESULT: reward={reward}, terminated={terminated}, truncated={truncated}")
        print(f"INFO: {info}")
        print(f"\nNEW OBSERVATION (first 1500 chars):\n{obs_text[:1500]}")
    except Exception as e:
        print(f"\nACTION FAILED: {e}")
        break

    if terminated or truncated:
        break

env.close()
print("\nDone.")
