"""Q-Learning agent for learning optimal cyber security response strategies."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.rl.environment import CyberSecurityEnv
from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.metrics import rl_metrics

logger = get_logger(__name__)


class QLearningAgent:
    """Tabular Q-Learning agent for cyber security response selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("reinforcement_learning.q_learning")
        self.alpha = self.config.get("alpha", 0.1)
        self.gamma = self.config.get("gamma", 0.95)
        self.epsilon = self.config.get("epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.episodes = self.config.get("episodes", 1000)
        self.max_steps = self.config.get("max_steps", 100)

        self.env = CyberSecurityEnv()
        self.n_actions = self.env.get_action_space_size()
        self.state_dim = self.env.get_state_dim()

        # Discretize state space for tabular Q-learning
        self.n_bins = 10
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_success: List[bool] = []

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize a continuous state into bins.

        Args:
            state: Continuous state vector.

        Returns:
            Tuple of bin indices.
        """
        bins = tuple(
            min(int(s * self.n_bins), self.n_bins - 1) for s in state
        )
        return bins

    def _get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Get Q-values for a state.

        Args:
            state: Discretized state.

        Returns:
            Q-value array for all actions.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current state.
            training: Whether to use exploration.

        Returns:
            Action index.
        """
        discrete_state = self._discretize_state(state)
        q_values = self._get_q_values(discrete_state)

        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Update Q-values using the Q-Learning update rule.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            terminated: Whether the episode terminated.
        """
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        q_values = self._get_q_values(discrete_state)
        next_q_values = self._get_q_values(discrete_next_state)

        # Q-Learning update
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(next_q_values)

        q_values[action] += self.alpha * (target - q_values[action])

    def _decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the Q-Learning agent.

        Args:
            n_episodes: Number of episodes to train. Defaults to config value.

        Returns:
            Dictionary with training metrics.
        """
        n = n_episodes or self.episodes
        logger.info(f"Training Q-Learning agent for {n} episodes...")

        for episode in range(n):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            success = False

            for step in range(self.max_steps):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.update(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = True
                    break
                if truncated:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.episode_success.append(success)
            self._decay_epsilon()

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}/{n} - Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        metrics = rl_metrics(
            self.episode_rewards,
            self.episode_lengths,
            self.episode_success,
        )
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    def evaluate(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate the trained agent without exploration.

        Args:
            n_episodes: Number of evaluation episodes.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info(f"Evaluating Q-Learning agent for {n_episodes} episodes...")
        eval_rewards: List[float] = []
        eval_lengths: List[int] = []
        eval_success: List[bool] = []

        original_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        for _ in range(n_episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            success = False

            for step in range(self.max_steps):
                action = self.choose_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = True
                    break
                if truncated:
                    break

            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
            eval_success.append(success)

        self.epsilon = original_epsilon

        metrics = rl_metrics(eval_rewards, eval_lengths, eval_success)
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def get_policy(self) -> Dict[str, Any]:
        """Extract the learned policy.

        Returns:
            Dictionary mapping state descriptions to best actions.
        """
        policy: Dict[str, Any] = {}
        action_names = self.env.get_action_names()

        for state, q_values in self.q_table.items():
            best_action = int(np.argmax(q_values))
            policy[str(state)] = {
                "best_action": action_names[best_action],
                "q_values": q_values.tolist(),
            }

        return policy

    def save(self, path: str) -> None:
        """Save the Q-table to disk.

        Args:
            path: Path to save the Q-table.
        """
        import json

        data = {
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "config": self.config,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_success": self.episode_success,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved Q-table to {path}")

    def load(self, path: str) -> None:
        """Load a Q-table from disk.

        Args:
            path: Path to load the Q-table from.
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)

        self.q_table = {
            tuple(int(x) for x in k.strip("()").split(",")): np.array(v)
            for k, v in data["q_table"].items()
        }
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lengths = data.get("episode_lengths", [])
        self.episode_success = data.get("episode_success", [])
        logger.info(f"Loaded Q-table from {path}")