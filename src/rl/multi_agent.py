"""Multi-agent reinforcement learning for coordinated security analysis."""

from typing import Any, Dict, List, Optional

import numpy as np

from src.rl.environment import CyberSecurityEnv
from src.rl.q_learning import QLearningAgent
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SpecialistAgent:
    """A specialist agent that analyzes a specific security signal domain."""

    def __init__(self, name: str, focus: str, state_dim: int = 6, n_actions: int = 8):
        self.name = name
        self.focus = focus
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.q_agent = QLearningAgent()
        self.q_agent.state_dim = state_dim

    def analyze(self, state: np.ndarray) -> Dict[str, Any]:
        """Analyze a state and produce a signal.

        Args:
            state: State vector relevant to this specialist's focus.

        Returns:
            Dictionary with analysis results.
        """
        focus_weights = {
            "network_traffic": [0.3, 0.2, 0.1, 0.1, 0.2, 0.1],
            "authentication": [0.2, 0.1, 0.1, 0.1, 0.3, 0.2],
            "endpoint_behavior": [0.1, 0.3, 0.2, 0.2, 0.1, 0.1],
        }
        weights = focus_weights.get(self.focus, [1/6] * 6)
        signal = float(np.dot(state[:6], weights))

        return {
            "agent": self.name,
            "focus": self.focus,
            "signal": signal,
            "confidence": float(np.clip(signal, 0, 1)),
        }


class DecisionAgent:
    """The central decision agent that selects responses based on specialist signals."""

    def __init__(self, n_actions: int = 8):
        self.n_actions = n_actions
        self.q_agent = QLearningAgent()
        self.q_agent.n_actions = n_actions

    def decide(self, signals: List[Dict[str, Any]]) -> int:
        """Decide on an action based on specialist signals.

        Args:
            signals: List of specialist analysis results.

        Returns:
            Action index.
        """
        combined_signal = np.mean([s["signal"] for s in signals])
        state = np.array([
            combined_signal,
            combined_signal,
            np.mean([s.get("confidence", 0) for s in signals]),
            0.5,
            0.5,
            0.5,
        ])
        return self.q_agent.choose_action(state, training=False)


class MultiAgentSystem:
    """Coordinates multiple specialist agents and a decision agent."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("multi_agent_rl")
        self.enabled = self.config.get("enabled", False)
        self.coordination = self.config.get("coordination", "centralized")

        self.specialists: List[SpecialistAgent] = []
        agent_configs = self.config.get("agents", [])
        for agent_config in agent_configs:
            self.specialists.append(
                SpecialistAgent(
                    name=agent_config.get("name", "agent"),
                    focus=agent_config.get("focus", "network_traffic"),
                )
            )

        self.decision_agent = DecisionAgent()

    def analyze_event(self, state: np.ndarray) -> Dict[str, Any]:
        """Analyze a security event using all specialist agents.

        Args:
            state: Full state vector.

        Returns:
            Dictionary with analysis results and recommended action.
        """
        signals = []
        for specialist in self.specialists:
            signal = specialist.analyze(state)
            signals.append(signal)

        action_idx = self.decision_agent.decide(signals)
        action_names = ["monitor", "alert", "block_ip", "quarantine_host",
                        "reset_credentials", "patch_vulnerability", "investigate", "escalate"]

        return {
            "signals": signals,
            "recommended_action": action_names[action_idx] if action_idx < len(action_names) else "monitor",
            "action_index": action_idx,
            "coordination": self.coordination,
        }

    def train(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Train the multi-agent system.

        Args:
            n_episodes: Number of training episodes.

        Returns:
            Dictionary with training metrics.
        """
        if not self.enabled:
            logger.info("Multi-agent RL is disabled. Skipping training.")
            return {}

        logger.info(f"Training multi-agent system for {n_episodes} episodes...")
        env = CyberSecurityEnv()
        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[bool] = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            steps = 0
            success = False

            for step in range(env.max_steps):
                signals = [s.analyze(state) for s in self.specialists]

                combined_signal = np.mean([s["signal"] for s in signals])
                decision_state = np.array([
                    combined_signal,
                    combined_signal,
                    np.mean([s.get("confidence", 0) for s in signals]),
                    0.5, 0.5, 0.5,
                ])
                action = self.decision_agent.q_agent.choose_action(decision_state, training=True)

                next_state, reward, terminated, truncated, info = env.step(action)

                self.decision_agent.q_agent.update(decision_state, action, reward, next_state, terminated)

                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = True
                    break
                if truncated:
                    break

            rewards.append(total_reward)
            lengths.append(steps)
            successes.append(success)

            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(rewards[-20:])
                logger.info(f"Episode {episode + 1}/{n_episodes} - Avg Reward: {avg_reward:.2f}")

        from src.utils.metrics import rl_metrics

        metrics = rl_metrics(rewards, lengths, successes)
        logger.info(f"Multi-agent training metrics: {metrics}")
        return metrics