"""Gymnasium environment for cyber security response decision-making."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False


class CyberSecurityEnv:
    """A cyber security response environment for reinforcement learning.

    The environment simulates security events and allows an agent to
    choose response actions. Rewards are based on the correctness of
    the response and the cost of the action.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("reinforcement_learning")
        self.env_config = self.config.get("environment", {})
        self.action_space = self.env_config.get("action_space", [
            "monitor", "alert", "block_ip", "quarantine_host",
            "reset_credentials", "patch_vulnerability", "investigate", "escalate",
        ])
        self.rewards = self.env_config.get("rewards", {})
        self.action_costs = self.rewards.get("action_cost", {})
        self.max_steps = self.config.get("max_steps", 100)

        self.n_actions = len(self.action_space)
        self.action_to_idx = {a: i for i, a in enumerate(self.action_space)}

        # State: [risk_score, anomaly_score, vulnerability_severity, asset_criticality, threat_level, time_sensitivity]
        self.state_dim = 6
        self.current_step = 0
        self.current_state = None
        self.true_risk = None

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state.

        Returns:
            Initial state vector.
        """
        self.current_step = 0
        # Generate a random security event
        self.true_risk = np.random.uniform(0, 1)
        self.current_state = self._generate_state(self.true_risk)
        return self.current_state

    def _generate_state(self, true_risk: float) -> np.ndarray:
        """Generate a state vector from a true risk level.

        Args:
            true_risk: The actual risk level of the event (0-1).

        Returns:
            State vector.
        """
        # Add noise to the true risk to simulate imperfect detection
        noise = np.random.normal(0, 0.1)
        observed_risk = np.clip(true_risk + noise, 0, 1)

        state = np.array([
            observed_risk,  # risk score
            np.clip(true_risk * np.random.uniform(0.7, 1.3), 0, 1),  # anomaly score
            np.random.uniform(0, 1),  # vulnerability severity
            np.random.uniform(0, 1),  # asset criticality
            np.random.uniform(0, 1),  # threat level
            np.random.uniform(0, 1),  # time sensitivity
        ])
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Index of the action to take.

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info).
        """
        self.current_step += 1
        action_name = self.action_space[action]

        # Determine if the action was appropriate for the true risk level
        risk = self.true_risk
        action_appropriate = self._is_action_appropriate(action_name, risk)

        # Compute reward
        reward = self._compute_reward(action_name, action_appropriate, risk)

        # Generate next state (slightly changed)
        next_state = self._generate_state(risk)

        # Check termination
        terminated = action_appropriate and action_name in ["block_ip", "quarantine_host", "reset_credentials", "patch_vulnerability"]
        truncated = self.current_step >= self.max_steps

        info = {
            "action": action_name,
            "action_appropriate": action_appropriate,
            "true_risk": risk,
            "step": self.current_step,
        }

        self.current_state = next_state
        return next_state, reward, terminated, truncated, info

    def _is_action_appropriate(self, action: str, risk: float) -> bool:
        """Determine if an action is appropriate for the given risk level.

        Args:
            action: Action name.
            risk: True risk level (0-1).

        Returns:
            True if the action is appropriate.
        """
        if risk < 0.3:  # Minimal/Low risk
            return action in ["monitor", "alert"]
        elif risk < 0.6:  # Medium risk
            return action in ["alert", "investigate", "reset_credentials"]
        elif risk < 0.8:  # High risk
            return action in ["block_ip", "quarantine_host", "reset_credentials", "patch_vulnerability", "investigate"]
        else:  # Critical risk
            return action in ["block_ip", "quarantine_host", "escalate"]

    def _compute_reward(self, action: str, appropriate: bool, risk: float) -> float:
        """Compute the reward for an action.

        Args:
            action: Action name.
            appropriate: Whether the action was appropriate.
            risk: True risk level.

        Returns:
            Reward value.
        """
        action_cost = self.action_costs.get(action, -0.5)

        if appropriate:
            if risk >= 0.8:
                reward = self.rewards.get("true_positive", 10)
            else:
                reward = self.rewards.get("true_negative", 2)
        else:
            if risk >= 0.8:
                reward = self.rewards.get("false_negative", -20)
            else:
                reward = self.rewards.get("false_positive", -5)

        return reward + action_cost

    def get_action_names(self) -> List[str]:
        """Get the list of action names.

        Returns:
            List of action names.
        """
        return self.action_space

    def get_state_dim(self) -> int:
        """Get the state dimension.

        Returns:
            State dimension.
        """
        return self.state_dim

    def get_action_space_size(self) -> int:
        """Get the number of actions.

        Returns:
            Number of actions.
        """
        return self.n_actions


class GymnasiumCyberSecurityEnv(CyberSecurityEnv):
    """Gymnasium-compatible wrapper for the cyber security environment."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not _HAS_GYM:
            raise ImportError("Gymnasium is required for the Gymnasium wrapper. Install with: pip install gymnasium")

        super().__init__(config)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space_gym = spaces.Discrete(self.n_actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment (Gymnasium API).

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            np.random.seed(seed)
        obs = super().reset()
        return obs.astype(np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step (Gymnasium API).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = super().step(action)
        return obs.astype(np.float32), float(reward), terminated, truncated, info