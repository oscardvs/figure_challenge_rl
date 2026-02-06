#!/usr/bin/env python3
"""Manual exploration script for the 30-step gauntlet.

Usage:
    python scripts/explore_challenges.py                    # Auto-explore from home
    python scripts/explore_challenges.py --step 5           # Start at step 5
    python scripts/explore_challenges.py --interactive       # Interactive step-through
    python scripts/explore_challenges.py --save-snapshots    # Save full HTML snapshots
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TAXONOMY_PATH = DATA_DIR / "challenge_taxonomy.json"


def explore_with_playwright(interactive: bool = False, start_step: int = 1, save_snapshots: bool = False):
    """Explore the gauntlet using raw Playwright (no BrowserGym).

    This is more reliable for initial reconnaissance since we can
    control the browser directly without the BrowserGym abstraction.
    """
    from playwright.sync_api import sync_playwright

    base_url = "https://serene-frangipane-7fd25b.netlify.app"
    taxonomy = []
    snapshot_dir = DATA_DIR / "snapshots"

    if save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not interactive)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()

        # Navigate to home page and click START.
        page.goto(base_url, wait_until="networkidle")
        logger.info("Loaded home page")

        try:
            start_btn = page.get_by_role("button", name="START")
            start_btn.click()
            page.wait_for_url("**/step1**", timeout=10000)
            logger.info("Clicked START, now on step 1")
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            browser.close()
            return

        current_step = 1

        while current_step <= 30:
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {current_step}")
            logger.info(f"{'='*60}")
            logger.info(f"URL: {page.url}")

            # Wait for page to settle.
            time.sleep(1)

            # Get accessibility tree.
            try:
                axtree = page.accessibility.snapshot()
                axtree_str = _format_axtree(axtree, indent=0)
            except Exception as e:
                axtree_str = f"[Error getting AXTree: {e}]"

            # Get page content summary.
            try:
                content = page.content()
            except Exception:
                content = ""

            step_data = {
                "step": current_step,
                "url": page.url,
                "title": page.title(),
                "axtree": axtree_str,
            }

            print(f"\n--- Step {current_step} AXTree ---")
            print(axtree_str[:3000])  # Truncate for display.
            if len(axtree_str) > 3000:
                print(f"\n[... {len(axtree_str) - 3000} more chars ...]")

            if save_snapshots:
                with open(snapshot_dir / f"step{current_step:02d}.html", "w") as f:
                    f.write(content)
                with open(snapshot_dir / f"step{current_step:02d}_axtree.txt", "w") as f:
                    f.write(axtree_str)
                logger.info(f"Saved snapshot for step {current_step}")

            taxonomy.append(step_data)

            if interactive:
                while True:
                    cmd = input(f"\n[Step {current_step}] Action (or 'next'/'quit'): ").strip()
                    if cmd.lower() == "quit":
                        browser.close()
                        _save_taxonomy(taxonomy)
                        return
                    if cmd.lower() == "next":
                        break
                    if cmd.lower() == "tree":
                        print(axtree_str)
                        continue
                    if cmd.lower().startswith("scroll"):
                        page.evaluate("window.scrollBy(0, 500)")
                        print(f"Scrolled. Y={page.evaluate('window.scrollY')}")
                        continue
                    if cmd.lower().startswith("wait"):
                        secs = int(cmd.split()[-1]) if len(cmd.split()) > 1 else 5
                        print(f"Waiting {secs}s...")
                        time.sleep(secs)
                        # Refresh AXTree after wait.
                        try:
                            axtree = page.accessibility.snapshot()
                            axtree_str = _format_axtree(axtree, indent=0)
                            print(axtree_str[:2000])
                        except Exception:
                            pass
                        continue
                    # Try executing as a Playwright action.
                    try:
                        eval(f"page.{cmd}")
                        print(f"Executed: {cmd}")
                        print(f"URL: {page.url}")
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                # Non-interactive: just log and move on.
                # We can't auto-solve steps here, so just capture initial state.
                current_step += 1
                if current_step <= 30:
                    # Try navigating directly to next step.
                    next_url = f"{base_url}/step{current_step}?version=3"
                    try:
                        page.goto(next_url, wait_until="networkidle", timeout=10000)
                    except Exception as e:
                        logger.warning(f"Could not navigate to step {current_step}: {e}")
                        break

        browser.close()

    _save_taxonomy(taxonomy)


def _format_axtree(node: dict | None, indent: int = 0) -> str:
    """Recursively format an accessibility tree node."""
    if node is None:
        return "[empty]"

    lines = []
    prefix = "  " * indent
    role = node.get("role", "unknown")
    name = node.get("name", "")
    value = node.get("value", "")

    parts = [f"{prefix}[{role}]"]
    if name:
        parts.append(f"'{name}'")
    if value:
        parts.append(f"value='{value}'")

    # Add useful properties.
    for prop in ("checked", "disabled", "expanded", "pressed", "selected"):
        if node.get(prop) is not None:
            parts.append(f"{prop}={node[prop]}")

    lines.append(" ".join(parts))

    for child in node.get("children", []):
        lines.append(_format_axtree(child, indent + 1))

    return "\n".join(lines)


def _save_taxonomy(taxonomy: list[dict]):
    """Save taxonomy to JSON."""
    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Strip large axtree from saved taxonomy (save separately).
    save_data = []
    for entry in taxonomy:
        save_entry = {k: v for k, v in entry.items() if k != "axtree"}
        save_entry["axtree_length"] = len(entry.get("axtree", ""))
        save_data.append(save_entry)

    with open(TAXONOMY_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Saved taxonomy to {TAXONOMY_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Explore the 30-step gauntlet")
    parser.add_argument("--step", type=int, default=1, help="Starting step number")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--save-snapshots", action="store_true", help="Save HTML snapshots")
    args = parser.parse_args()

    explore_with_playwright(
        interactive=args.interactive,
        start_step=args.step,
        save_snapshots=args.save_snapshots,
    )


if __name__ == "__main__":
    main()
