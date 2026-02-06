"""Action space definition and utilities for the browser agent."""

from __future__ import annotations

from browsergym.core.action.highlevel import HighLevelActionSet


def js_eval(code: str):
    """Execute arbitrary JavaScript on the page and display the result.

    Use this for content not visible in the accessibility tree: shadow DOM,
    canvas readback, audio/video state, service worker messages, websocket
    data, iframe content, or DOM mutation observation.

    The result is injected into a visible DOM element so the agent can read
    it in the next observation's accessibility tree.

    Examples:
        js_eval("document.querySelector('#shadow-host').shadowRoot.textContent")
        js_eval("document.querySelector('canvas').toDataURL().slice(0,100)")
        js_eval("window.__secretCode || 'not found'")
    """
    import json as _json

    result = page.evaluate(code)  # noqa: F821 — page is injected by BrowserGym exec context

    # Serialize result to string.
    if result is None:
        result_str = "null"
    elif isinstance(result, str):
        result_str = result
    else:
        try:
            result_str = _json.dumps(result)
        except (TypeError, ValueError):
            result_str = str(result)

    # Inject result into a visible DOM element so the agent sees it in AXTree.
    page.evaluate(  # noqa: F821
        """(resultText) => {
            let el = document.getElementById('__bgym_js_result');
            if (!el) {
                el = document.createElement('div');
                el.id = '__bgym_js_result';
                el.setAttribute('role', 'status');
                el.setAttribute('aria-label', 'JavaScript result');
                el.style.cssText = 'position:fixed;bottom:0;left:0;right:0;'
                    + 'background:#ffe;color:#000;padding:8px;z-index:99999;'
                    + 'font-family:monospace;font-size:14px;border-top:2px solid #cc0;'
                    + 'max-height:200px;overflow:auto;';
                document.body.appendChild(el);
            }
            el.textContent = 'JS Result: ' + resultText;
        }""",
        result_str,
    )


def get_action_set(
    include_nav: bool = True,
    include_tab: bool = True,
    include_chat: bool = False,
    include_custom: bool = True,
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

    Custom actions:
        js_eval(code) — execute JavaScript on the page
    """
    subsets = ["bid"]
    if include_nav:
        subsets.append("nav")
    if include_tab:
        subsets.append("tab")
    if include_chat:
        subsets.append("chat")

    custom_actions = None
    if include_custom:
        subsets.append("custom")
        custom_actions = [js_eval]

    return HighLevelActionSet(
        subsets=subsets,
        custom_actions=custom_actions,
        multiaction=False,  # One action per step for cleaner trajectories.
        strict=strict,
        retry_with_force=True,  # Retry clicks with force=True on interception.
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
