"""Tests for the reinforcement learning modules."""

import numpy as np
import pytest

from src.rl.environment import CyberSecurityEnv
from src.rl.multi_agent import DecisionAgent, MultiAgentSystem, SpecialistAgent
from src.rl.q_learning import QLearningAgent


@pytest.fixture
def env():
    return CyberSecurityEnv()


@pytest.fixture
def agent():
    return QLearningAgent()


def test_env_reset(env):
    """Test environment reset returns expected state shape."""
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)
    assert env.current_step == 0


def test_env_step(env):
    """Test taking steps in the environment."""
    env.reset()
    next_state, reward, terminated, truncated, info = env.step(0)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (6,)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action" in info
    assert "true_risk" in info


def test_env_action_space(env):
    """Test action space queries."""
    actions = env.get_action_names()
    assert len(actions) == env.get_action_space_size()
    assert "monitor" in actions
    assert "block_ip" in actions
    assert env.get_state_dim() == 6


def test_q_learning_choose_action(agent):
    """Test action selection for QLearningAgent."""
    state = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    action = agent.choose_action(state, training=False)
    assert 0 <= action < agent.n_actions


def test_q_learning_update(agent):
    """Test Q-table update rule."""
    state = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    next_state = np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    action = 1
    reward = 5.0
    discrete_state = agent._discretize_state(state)
    initial_q = agent._get_q_values(discrete_state)[action]

    agent.update(state, action, reward, next_state, terminated=False)
    updated_q = agent._get_q_values(discrete_state)[action]
    assert updated_q != initial_q


def test_q_learning_train_and_evaluate(agent):
    """Test lightweight training and evaluation runs."""
    train_metrics = agent.train(n_episodes=5)
    assert "avg_reward" in train_metrics
    assert "cumulative_reward" in train_metrics

    eval_metrics = agent.evaluate(n_episodes=2)
    assert "avg_reward" in eval_metrics


def test_q_learning_save_load(agent, tmp_path):
    """Test saving and loading Q-table."""
    state = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    agent.choose_action(state, training=True)
    agent.update(state, 0, 1.0, state, False)

    save_file = str(tmp_path / "q_table.json")
    agent.save(save_file)

    new_agent = QLearningAgent()
    new_agent.load(save_file)
    discrete_state = new_agent._discretize_state(state)
    assert discrete_state in new_agent.q_table


def test_specialist_and_decision_agents():
    """Test specialist analysis and decision agent response."""
    specialist = SpecialistAgent(name="network_specialist", focus="network_traffic")
    state = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    signal = specialist.analyze(state)

    assert signal["agent"] == "network_specialist"
    assert signal["focus"] == "network_traffic"
    assert "signal" in signal
    assert "confidence" in signal

    decision = DecisionAgent()
    action = decision.decide([signal])
    assert 0 <= action < decision.n_actions


def test_multi_agent_system():
    """Test multi-agent coordination."""
    mas = MultiAgentSystem()
    state = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    result = mas.analyze_event(state)

    assert "signals" in result
    assert "recommended_action" in result
    assert "coordination" in result
