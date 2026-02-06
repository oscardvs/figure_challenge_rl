"""Action space definition and utilities for the browser agent."""

from __future__ import annotations

from browsergym.core.action.highlevel import HighLevelActionSet


def get_action_set(
    include_nav: bool = True,
    include_tab: bool = True,
    include_chat: bool = False,
    strict: bool = False,
) -> HighLevelActionSet:
    """Create the standard action set for the challenge agent.

    Actions available via the 'bid' subset:
        click(bid, button="left", modifiers=[])
        dblclick(bid, button="left", modifiers=[])
        fill(bid, value)
        select_option(bid, options)
        hover(bid)
        press(bid, key)
        focus(bid)
        clear(bid)
        scroll(delta_x=0, delta_y=0)
        drag_and_drop(bid, target_bid)
        upload_file(bid, file_path)
    """
    subsets = ["bid"]
    if include_nav:
        subsets.append("nav")
    if include_tab:
        subsets.append("tab")
    if include_chat:
        subsets.append("chat")

    return HighLevelActionSet(
        subsets=subsets,
        multiaction=False,  # One action per step for cleaner trajectories.
        strict=strict,
    )


def get_action_description(action_set: HighLevelActionSet | None = None) -> str:
    """Return a human-readable description of the available actions.

    Suitable for inclusion in LLM system prompts.
    """
    if action_set is None:
        action_set = get_action_set()
    return action_set.describe(with_long_description=True, with_examples=True)


def get_action_example(action_set: HighLevelActionSet | None = None) -> str:
    """Return an example action string."""
    if action_set is None:
        action_set = get_action_set()
    return action_set.example_action(abstract=False)
