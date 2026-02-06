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

        # Strip repetitive filler content first.
        text = self._strip_filler(text)

        # Enforce token budget via smart truncation.
        char_budget = self.target_tokens * 4
        if len(text) > char_budget:
            text = self._truncate_to_budget(text, char_budget)

        return text

    def _strip_filler(self, text: str) -> str:
        """Remove repetitive filler lines that add no information.

        The challenge pages have 100+ identical 'Section N / filler content'
        lines and empty paragraph stubs. Strip them aggressively.
        """
        lines = text.split("\n")
        result = []
        filler_count = 0
        filler_inserted = False

        # Patterns that indicate filler content.
        # Allow optional suffixes like ", visible", ", clickable" after the role.
        _filler_re = re.compile(
            r"^\[\d+\]\s*(heading\s+'section\s+\d+'|paragraph\s+'')(,\s*\w+)*\s*$",
            re.IGNORECASE,
        )

        for line in lines:
            stripped = line.strip().lower()
            is_filler = (
                "filler content" in stripped
                or "keep scrolling to find" in stripped
                or _filler_re.match(stripped)
            )

            if is_filler:
                filler_count += 1
                if filler_count <= 2:
                    result.append(line)
                elif not filler_inserted:
                    result.append("\t[... filler sections omitted ...]")
                    filler_inserted = True
            else:
                # Also strip lorem ipsum padding from modal dialogs.
                if any(p in stripped for p in (
                    "lorem ipsum", "sed ut perspiciatis", "nemo enim",
                    "neque porro", "at vero eos", "voluptatum deleniti",
                    "totam rem", "duis aute", "excepteur sint",
                    "ut enim ad minim",
                )):
                    continue
                result.append(line)

        return "\n".join(result)

    def _truncate_to_budget(self, text: str, char_budget: int) -> str:
        """Truncate AXTree text while preserving important elements.

        Strategy: keep interactive elements (buttons, inputs, links) and
        the first/last portions. Drop repetitive filler (like "Section N"
        headings with identical content).
        """
        lines = text.split("\n")
        if len(lines) <= 20:
            return text[:char_budget]

        # Separate lines into interactive (buttons, inputs, links, etc.)
        # and non-interactive (static text, headings, paragraphs).
        interactive_keywords = (
            "button", "textbox", "checkbox", "radio", "combobox",
            "link", "input", "select", "slider", "textarea",
        )

        important = []  # (line_index, line)
        filler = []     # (line_index, line)

        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            is_interactive = any(kw in stripped for kw in interactive_keywords)
            is_challenge_text = any(kw in stripped for kw in (
                "code", "challenge", "step ", "scroll", "hidden", "reveal",
                "click here", "submit", "enter", "timer", "wait",
            ))
            if is_interactive or is_challenge_text or i < 15 or i >= len(lines) - 10:
                important.append((i, lines[i]))
            else:
                filler.append((i, lines[i]))

        # Build output from important lines, preserving order.
        kept = [line for _, line in important]
        truncated = "\n".join(kept)

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
    if len(last_action) > 200:
        last_action = last_action[:200] + "..."
    last_error = obs.get("last_action_error", "")
    if len(last_error) > 300:
        last_error = last_error[:300] + "..."

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
