"""Monte Carlo Tree Search over browser states.

Implements UCB1-guided tree search using an API LLM as both the policy
(proposing candidate actions) and the critic (evaluating state-action
utility). Actions are executed in the real browser — no simulated
world model — since the challenge pages are fast-loading React SPAs.

The key data product: trajectories and step-level preference pairs
for downstream SFT, DPO, and M-GRPO training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.agent.policy import ActionCandidate, LLMPolicy
from src.environment.browser_env import AdcockChallengeEnv

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""
    state_hash: str
    obs_text: str
    parent: Optional[MCTSNode] = None
    parent_action: Optional[str] = None
    children: dict[str, MCTSNode] = field(default_factory=dict)  # action -> child node

    visit_count: int = 0
    total_reward: float = 0.0
    q_value: float = 0.0
    critic_score: float = 0.5  # LLM critic's evaluation

    # Candidate actions proposed but not yet expanded.
    unexpanded_actions: list[ActionCandidate] = field(default_factory=list)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.unexpanded_actions) == 0


@dataclass
class TrajectoryStep:
    """A single step in a trajectory, used for training data."""
    obs_text: str
    action: str
    reasoning: str
    reward: float
    q_value: float
    critic_score: float
    step_index: int
    state_hash: str


@dataclass
class Trajectory:
    """A complete trajectory through a challenge."""
    challenge_id: int
    steps: list[TrajectoryStep]
    total_reward: float
    success: bool
    duration_seconds: float


@dataclass
class PreferencePair:
    """A DPO preference pair extracted from MCTS Q-value differences."""
    obs_text: str
    chosen_action: str
    rejected_action: str
    chosen_q: float
    rejected_q: float
    challenge_id: int
    step_index: int


def _hash_state(obs_text: str) -> str:
    """Create a compact hash of the observation for deduplication."""
    return hashlib.md5(obs_text.encode()).hexdigest()[:12]


class MCTSSearch:
    """MCTS search over browser states for a single challenge.

    Algorithm per the plan:
    1. Selection: UCB1 with Q-value blending (MCTS + critic).
    2. Expansion: LLM proposes K candidate actions.
    3. Simulation: Real browser execution (not simulated).
    4. Backpropagation: Binary reward propagated up the tree.
    """

    def __init__(
        self,
        env: AdcockChallengeEnv,
        policy: LLMPolicy,
        task_prompt: str,
        num_iterations: int = 15,
        candidates_per_node: int = 5,
        exploration_constant: float = 1.4,
        q_blend_alpha: float = 0.7,
        min_q_diff_for_dpo: float = 0.2,
    ):
        self.env = env
        self.policy = policy
        self.task_prompt = task_prompt
        self.num_iterations = num_iterations
        self.candidates_per_node = candidates_per_node
        self.c = exploration_constant
        self.alpha = q_blend_alpha
        self.min_q_diff = min_q_diff_for_dpo

        self.root: Optional[MCTSNode] = None
        self.trajectories: list[Trajectory] = []
        self.preference_pairs: list[PreferencePair] = []
        self.all_steps: list[TrajectoryStep] = []

    def search(self) -> list[Trajectory]:
        """Run MCTS search and collect trajectories.

        Returns:
            List of all trajectories (successful and failed).
        """
        start_time = time.time()

        # Initialize root from environment reset.
        obs_text, info = self.env.reset()
        self.root = MCTSNode(
            state_hash=_hash_state(obs_text),
            obs_text=obs_text,
            depth=0,
        )

        for iteration in range(self.num_iterations):
            logger.info(f"MCTS iteration {iteration + 1}/{self.num_iterations}")

            # Reset env to start state for this iteration.
            obs_text, _ = self.env.reset()

            # Run one full trajectory from root to terminal.
            trajectory = self._run_iteration(obs_text, iteration)
            self.trajectories.append(trajectory)

            logger.info(
                f"  Iteration {iteration + 1}: "
                f"{'SUCCESS' if trajectory.success else 'FAIL'} "
                f"in {len(trajectory.steps)} steps, "
                f"reward={trajectory.total_reward:.1f}"
            )

        # Extract preference pairs from the tree.
        self._extract_preference_pairs(self.root)

        elapsed = time.time() - start_time
        logger.info(
            f"MCTS complete: {len(self.trajectories)} trajectories, "
            f"{len(self.preference_pairs)} preference pairs, "
            f"{len(self.all_steps)} total steps in {elapsed:.1f}s"
        )

        return self.trajectories

    def _run_iteration(self, initial_obs: str, iteration: int) -> Trajectory:
        """Run a single MCTS iteration: select → expand → simulate → backprop."""
        steps: list[TrajectoryStep] = []
        action_history: list[str] = []
        current_obs = initial_obs
        current_node = self.root
        total_reward = 0.0
        start_time = time.time()

        for step_idx in range(self.env.max_steps):
            # Phase 1: Selection — walk existing tree with UCB1.
            while not current_node.is_leaf and not current_node.is_fully_expanded:
                action_str = self._ucb1_select(current_node)
                if action_str in current_node.children:
                    current_node = current_node.children[action_str]
                else:
                    break

            # Phase 2: Expansion — get LLM candidates if needed.
            if current_node.is_leaf or not current_node.is_fully_expanded:
                if not current_node.unexpanded_actions:
                    candidates = self.policy.propose_actions(
                        obs_text=current_obs,
                        task_prompt=self.task_prompt,
                        action_history=action_history,
                        step=step_idx,
                        k=self.candidates_per_node,
                    )
                    current_node.unexpanded_actions = candidates

                # Pick the best unexpanded action.
                if current_node.unexpanded_actions:
                    candidate = current_node.unexpanded_actions.pop(0)
                    action_str = candidate.action
                    reasoning = candidate.reasoning
                    critic_score = candidate.critic_score
                else:
                    # Fallback: direct policy query.
                    action_str, reasoning = self.policy.select_action(
                        obs_text=current_obs,
                        task_prompt=self.task_prompt,
                        action_history=action_history,
                        step=step_idx,
                    )
                    critic_score = 0.5
            else:
                # Tree is fully expanded at this node — use UCB1.
                action_str = self._ucb1_select(current_node)
                reasoning = ""
                critic_score = 0.5

            # Phase 3: Simulation — execute action in real browser.
            try:
                obs_text, reward, terminated, truncated, info = self.env.step(action_str)
            except Exception as e:
                logger.warning(f"Action execution failed: {e}")
                obs_text = current_obs
                reward = 0.0
                terminated = True
                truncated = False
                info = {"error": str(e)}

            total_reward = max(total_reward, reward)

            # Record step.
            step_data = TrajectoryStep(
                obs_text=current_obs,
                action=action_str,
                reasoning=reasoning,
                reward=reward,
                q_value=0.0,  # Will be updated during backpropagation.
                critic_score=critic_score,
                step_index=step_idx,
                state_hash=_hash_state(current_obs),
            )
            steps.append(step_data)
            self.all_steps.append(step_data)
            action_history.append(action_str)

            # Create/update child node.
            child_hash = _hash_state(obs_text)
            if action_str not in current_node.children:
                child_node = MCTSNode(
                    state_hash=child_hash,
                    obs_text=obs_text,
                    parent=current_node,
                    parent_action=action_str,
                    depth=current_node.depth + 1,
                    critic_score=critic_score,
                )
                current_node.children[action_str] = child_node
            else:
                child_node = current_node.children[action_str]

            current_node = child_node
            current_obs = obs_text

            if terminated or truncated:
                break

        # Phase 4: Backpropagation.
        self._backpropagate(current_node, total_reward)

        # Update Q-values on trajectory steps.
        node = current_node
        for step in reversed(steps):
            if node and node.parent:
                step.q_value = node.q_value
                node = node.parent

        success = total_reward > 0
        return Trajectory(
            challenge_id=self.env.challenge_id,
            steps=steps,
            total_reward=total_reward,
            success=success,
            duration_seconds=time.time() - start_time,
        )

    def _ucb1_select(self, node: MCTSNode) -> str:
        """Select the best child action using UCB1."""
        best_action = None
        best_score = -float("inf")

        for action_str, child in node.children.items():
            if child.visit_count == 0:
                return action_str  # Prioritize unvisited children.

            # Blended Q-value: MCTS empirical + critic evaluation.
            q_blended = (
                self.alpha * child.q_value
                + (1 - self.alpha) * child.critic_score
            )

            # UCB1 exploration bonus.
            exploration = self.c * math.sqrt(
                math.log(node.visit_count) / child.visit_count
            )

            score = q_blended + exploration
            if score > best_score:
                best_score = score
                best_action = action_str

        return best_action

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Update Q-values bottom-up from terminal node to root."""
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node.q_value = node.total_reward / node.visit_count
            node = node.parent

    def _extract_preference_pairs(self, node: MCTSNode):
        """Extract DPO preference pairs from branching points in the tree.

        At each node with multiple children, create preference pairs from
        children whose Q-values differ by at least min_q_diff.
        """
        if len(node.children) >= 2:
            children_list = [
                (action, child)
                for action, child in node.children.items()
                if child.visit_count > 0
            ]
            # Sort by Q-value descending.
            children_list.sort(key=lambda x: x[1].q_value, reverse=True)

            for i in range(len(children_list)):
                for j in range(i + 1, len(children_list)):
                    a_win, c_win = children_list[i]
                    a_lose, c_lose = children_list[j]
                    q_diff = c_win.q_value - c_lose.q_value

                    if q_diff >= self.min_q_diff:
                        self.preference_pairs.append(PreferencePair(
                            obs_text=node.obs_text,
                            chosen_action=a_win,
                            rejected_action=a_lose,
                            chosen_q=c_win.q_value,
                            rejected_q=c_lose.q_value,
                            challenge_id=self.env.challenge_id,
                            step_index=node.depth,
                        ))

        # Recurse into children.
        for child in node.children.values():
            self._extract_preference_pairs(child)

    def get_results(self) -> dict:
        """Return collected data in a serializable format."""
        return {
            "trajectories": [
                {
                    "challenge_id": t.challenge_id,
                    "success": t.success,
                    "total_reward": t.total_reward,
                    "num_steps": len(t.steps),
                    "duration_seconds": t.duration_seconds,
                    "steps": [
                        {
                            "obs_text": s.obs_text,
                            "action": s.action,
                            "reasoning": s.reasoning,
                            "reward": s.reward,
                            "q_value": s.q_value,
                            "critic_score": s.critic_score,
                            "step_index": s.step_index,
                        }
                        for s in t.steps
                    ],
                }
                for t in self.trajectories
            ],
            "preference_pairs": [
                {
                    "obs_text": p.obs_text,
                    "chosen_action": p.chosen_action,
                    "rejected_action": p.rejected_action,
                    "chosen_q": p.chosen_q,
                    "rejected_q": p.rejected_q,
                    "challenge_id": p.challenge_id,
                    "step_index": p.step_index,
                }
                for p in self.preference_pairs
            ],
            "stats": {
                "total_trajectories": len(self.trajectories),
                "successful_trajectories": sum(
                    1 for t in self.trajectories if t.success
                ),
                "total_steps": len(self.all_steps),
                "total_preference_pairs": len(self.preference_pairs),
                "api_tokens": self.policy.total_tokens,
            },
        }
