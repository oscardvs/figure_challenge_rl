"""AXTree observation extraction and pruning for the browser agent."""

from __future__ import annotations

import re
from typing import Optional

from browsergym.utils.obs import flatten_axtree_to_str


class AXTreePruner:
    """Prunes and formats accessibility trees to fit within token budgets.

    Uses BrowserGym's built-in flatten_axtree_to_str with additional
    token-budget enforcement via iterative truncation.
    """

    def __init__(
        self,
        visible_only: bool = True,
        with_bid_only: bool = True,
        skip_generic: bool = True,
        with_clickable: bool = True,
        with_visible: bool = True,
        target_tokens: int = 3000,
        remove_redundant_static_text: bool = True,
    ):
        self.visible_only = visible_only
        self.with_bid_only = with_bid_only
        self.skip_generic = skip_generic
        self.with_clickable = with_clickable
        self.with_visible = with_visible
        self.target_tokens = target_tokens
        self.remove_redundant_static_text = remove_redundant_static_text

    def process(
        self,
        axtree_object: dict,
        extra_properties: Optional[dict] = None,
    ) -> str:
        """Convert an AXTree object into a pruned text representation.

        Args:
            axtree_object: The raw AXTree dict from BrowserGym observation.
            extra_properties: Element properties dict (visibility, bbox, etc.)

        Returns:
            Pruned AXTree as a string, targeting self.target_tokens tokens.
        """
        text = flatten_axtree_to_str(
            axtree_object,
            extra_properties=extra_properties,
            with_visible=self.with_visible,
            with_clickable=self.with_clickable,
            skip_generic=self.skip_generic,
            filter_visible_only=self.visible_only,
            filter_with_bid_only=self.with_bid_only,
            remove_redundant_static_text=self.remove_redundant_static_text,
        )

        # Enforce token budget via line truncation.
        # Rough estimate: 1 token ≈ 4 characters.
        char_budget = self.target_tokens * 4
        if len(text) > char_budget:
            text = self._truncate_to_budget(text, char_budget)

        return text

    def _truncate_to_budget(self, text: str, char_budget: int) -> str:
        """Truncate AXTree text while preserving structure.

        Keeps the first and last portions of the tree to preserve both
        the page header and footer/action areas.
        """
        lines = text.split("\n")
        if len(lines) <= 10:
            return text[:char_budget]

        # Keep first 40% and last 20% of lines, drop the middle.
        head_count = max(5, int(len(lines) * 0.4))
        tail_count = max(3, int(len(lines) * 0.2))

        head = lines[:head_count]
        tail = lines[-tail_count:]
        truncated = "\n".join(head) + "\n[... truncated ...]\n" + "\n".join(tail)

        # If still too long, hard-cut.
        if len(truncated) > char_budget:
            truncated = truncated[:char_budget] + "\n[... truncated ...]"

        return truncated


def extract_obs_text(obs: dict, pruner: AXTreePruner) -> str:
    """Extract a text observation from a BrowserGym observation dict.

    Combines URL, page title, and pruned AXTree into a single string
    suitable for LLM consumption.
    """
    url = obs.get("url", "")
    titles = obs.get("open_pages_titles", [])
    title = titles[0] if titles else ""
    focused = obs.get("focused_element_bid", "")
    last_action = obs.get("last_action", "")
    last_error = obs.get("last_action_error", "")

    axtree = pruner.process(
        obs.get("axtree_object", {}),
        extra_properties=obs.get("extra_element_properties", {}),
    )

    parts = [
        f"URL: {url}",
        f"Title: {title}",
    ]
    if focused:
        parts.append(f"Focused element: [{focused}]")
    if last_action:
        parts.append(f"Previous action: {last_action}")
    if last_error:
        parts.append(f"Action error: {last_error}")
    parts.append(f"\nAccessibility Tree:\n{axtree}")

    return "\n".join(parts)
