#!/usr/bin/env python3
"""Focused test: solve step 1 quickly, check if step 2 observation is good."""
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.environment.browser_env import GauntletEnv

env = GauntletEnv(headless=True, max_actions_per_step=30)
obs_text, info = env.reset()
step = info.get("task_info", {}).get("current_step", "?")
print(f"[RESET] step={step}, obs={len(obs_text)} chars")

# Try scroll to reveal code
obs_text, reward, term, trunc, info = env.step("scroll(0, 600)")
print(f"[SCROLL] obs={len(obs_text)} chars, reward={reward}")

# Try click Reveal Code if it exists
if "Reveal Code" in obs_text:
    bid_match = re.search(r"\[(\d+)\] button 'Reveal Code'", obs_text)
    if bid_match:
        obs_text, reward, term, trunc, info = env.step(f"click('{bid_match.group(1)}')")
        print(f"[REVEAL] obs={len(obs_text)} chars, reward={reward}")
        # Click again to actually reveal
        bid_match2 = re.search(r"\[(\d+)\] button '(?:Reveal Code|Code Revealed)'", obs_text)
        if bid_match2:
            obs_text, reward, term, trunc, info = env.step(f"click('{bid_match2.group(1)}')")
            print(f"[REVEAL2] obs={len(obs_text)} chars, reward={reward}")

# Extract any 6-char codes from the obs
codes = re.findall(r"'([A-Z0-9]{6})'", obs_text)
# Filter out common non-codes
codes = [c for c in codes if c not in ("Submit",)]
print(f"Candidate codes: {codes}")

if codes:
    code = codes[0]
    # Find textbox and submit button
    tb = re.search(r"\[(\d+)\] textbox 'Enter 6-character code'", obs_text)
    if tb:
        obs_text, reward, term, trunc, info = env.step(f"fill('{tb.group(1)}', '{code}')")
        step = info.get("task_info", {}).get("current_step", "?")
        print(f"[FILL] code={code}, obs={len(obs_text)} chars, step={step}")

        sb = re.search(r"\[(\d+)\] button 'Submit Code', clickable", obs_text)
        if sb:
            obs_text, reward, term, trunc, info = env.step(f"click('{sb.group(1)}')")
            step = info.get("task_info", {}).get("current_step", "?")
            print(f"[SUBMIT] obs={len(obs_text)} chars, reward={reward}, step={step}")
            if reward > 0:
                print(f"\n=== STEP 2 OBSERVATION ({len(obs_text)} chars) ===")
                print(obs_text[:3000])
            else:
                print("Wrong code or submit failed")
                print(obs_text[:1000])
        else:
            print("Submit button not clickable")
    else:
        print("Textbox not found")
else:
    print("No code found — this puzzle variant needs different approach")
    print("Obs excerpt:", obs_text[:500])

env.close()
print("\nDone.")
