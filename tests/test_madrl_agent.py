#!/usr/bin/env python3
"""
Unit tests for the MADRL Agent Module

This module contains comprehensive unit tests for the Multi-Agent Deep
Reinforcement Learning implementation, including agent behavior, environment
dynamics, reward calculation, and DQN functionality.

Author: Research Team
Date: 2025
"""

import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.madrl_agent import (
    MADRLConfig, RewardCalculator, DQNNetwork, MADRLAgent, 
    MultiAgentEnvironment
)
from core.network_topology import NetworkTopology, NetworkConfig
from core.channel_model import ChannelModel
from core.irs_optimization import IRSOptimizer


class TestMADRLConfig(unittest.TestCase):
    """Test cases for MADRL configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MADRLConfig()
        
        self.assertEqual(config.hidden_layers, [128, 64, 32])
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.discount_factor, 0.95)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.epsilon_start, 1.0)
        self.assertEqual(config.epsilon_end, 0.01)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MADRLConfig(
            hidden_layers=[64, 32],
            learning_rate=0.01,
            batch_size=64
        )
        
        self.assertEqual(config.hidden_layers, [64, 32])
        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.batch_size, 64)


class TestRewardCalculator(unittest.TestCase):
    """Test cases for reward calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MADRLConfig()
        self.reward_calculator = RewardCalculator(self.config)
    
    def test_initialization(self):
        """Test reward calculator initialization."""
        self.assertEqual(len(self.reward_calculator.performance_history), 0)
        self.assertEqual(len(self.reward_calculator.energy_history), 0)
        self.assertEqual(self.reward_calculator.baseline_performance, 1e6)
        self.assertEqual(self.reward_calculator.baseline_energy, 1.0)
    
    def test_performance_reward_calculation(self):
        """Test performance reward component calculation."""
        state = {
            'sum_rate': 2e6,  # 2 Mbps
            'min_rate': 1e5,  # 100 kbps
            'fairness_index': 0.8
        }
        
        reward = self.reward_calculator._calculate_performance_reward(state)
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)
    
    def test_energy_reward_calculation(self):
        """Test energy reward component calculation."""
        state = {
            'total_power': 0.8,
            'energy_efficiency': 1e6,  # 1 Mbps/W
            'renewable_energy_ratio': 0.6
        }
        
        reward = self.reward_calculator._calculate_energy_reward(state)
        self.assertIsInstance(reward, float)
    
    def test_penalty_calculation(self):
        """Test penalty calculation."""
        state = {'previous_action': np.array([0.1, 0.2, 0.3])}
        action = np.array([0.2, 0.3, 0.4])
        next_state = {'constraint_violated': False, 'min_rate': 2e5}
        
        penalty = self.reward_calculator._calculate_penalties(state, action, next_state)
        self.assertIsInstance(penalty, float)
        self.assertGreaterEqual(penalty, 0)
    
    def test_full_reward_calculation(self):
        """Test complete reward calculation."""
        state = {'previous_action': np.array([0.1, 0.2])}
        action = np.array([0.2, 0.3])
        next_state = {
            'sum_rate': 1.5e6,
            'min_rate': 1e5,
            'fairness_index': 0.7,
            'total_power': 0.6,
            'energy_efficiency': 8e5,
            'renewable_energy_ratio': 0.4,
            'constraint_violated': False
        }
        
        reward = self.reward_calculator.calculate_reward(state, action, next_state)
        self.assertIsInstance(reward, float)
    
    def test_reward_statistics(self):
        """Test reward statistics calculation."""
        # Add some history
        self.reward_calculator.performance_history.extend([1.0, 2.0, 3.0])
        self.reward_calculator.energy_history.extend([0.5, 1.0, 1.5])
        
        stats = self.reward_calculator.get_reward_statistics()
        
        self.assertIn('avg_performance_reward', stats)
        self.assertIn('avg_energy_reward', stats)
        self.assertEqual(stats['avg_performance_reward'], 2.0)
        self.assertEqual(stats['avg_energy_reward'], 1.0)
    
    def test_reset_history(self):
        """Test history reset functionality."""
        self.reward_calculator.performance_history.extend([1.0, 2.0])
        self.reward_calculator.energy_history.extend([0.5, 1.0])
        
        self.reward_calculator.reset_history()
        
        self.assertEqual(len(self.reward_calculator.performance_history), 0)
        self.assertEqual(len(self.reward_calculator.energy_history), 0)


class TestDQNNetwork(unittest.TestCase):
    """Test cases for DQN network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 5
        self.hidden_layers = [32, 16]
        self.network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_layers)
    
    def test_network_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.state_dim, self.state_dim)
        self.assertEqual(self.network.action_dim, self.action_dim)
        self.assertIsInstance(self.network.layers_list, list)
    
    def test_forward_pass(self):
        """Test forward pass through the network."""
        batch_size = 4
        inputs = tf.random.normal((batch_size, self.state_dim))
        
        outputs = self.network(inputs)
        
        self.assertEqual(outputs.shape, (batch_size, self.action_dim))
        self.assertEqual(outputs.dtype, tf.float32)
    
    def test_training_mode(self):
        """Test network in training mode."""
        inputs = tf.random.normal((1, self.state_dim))
        
        # Should work in both training and inference modes
        output_train = self.network(inputs, training=True)
        output_infer = self.network(inputs, training=False)
        
        self.assertEqual(output_train.shape, output_infer.shape)


class TestMADRLAgent(unittest.TestCase):
    """Test cases for MADRL agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent_id = 0
        self.state_dim = 10
        self.action_dim = 5
        self.config = MADRLConfig(memory_size=100, batch_size=4)
        self.agent = MADRLAgent(self.agent_id, self.state_dim, self.action_dim, self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, self.agent_id)
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.epsilon, self.config.epsilon_start)
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_action_selection_exploration(self):
        """Test action selection during exploration."""
        state = np.random.random(self.state_dim)
        
        # Force exploration
        self.agent.epsilon = 1.0
        action = self.agent.get_action(state, training=True)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))
    
    def test_action_selection_exploitation(self):
        """Test action selection during exploitation."""
        state = np.random.random(self.state_dim)
        
        # Force exploitation
        self.agent.epsilon = 0.0
        action = self.agent.get_action(state, training=True)
        
        self.assertEqual(len(action), self.action_dim)
        self.assertIsInstance(action, np.ndarray)
    
    def test_experience_storage(self):
        """Test experience storage in replay buffer."""
        state = np.random.random(self.state_dim)
        action = np.random.random(self.action_dim)
        reward = 1.0
        next_state = np.random.random(self.state_dim)
        done = False
        
        self.agent.store_experience(state, action, reward, next_state, done)
        
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_training_insufficient_memory(self):
        """Test training with insufficient memory."""
        loss = self.agent.train()
        self.assertEqual(loss, 0.0)
    
    def test_training_with_sufficient_memory(self):
        """Test training with sufficient memory."""
        # Fill memory with experiences
        for _ in range(self.config.batch_size):
            state = np.random.random(self.state_dim)
            action = np.random.random(self.action_dim)
            reward = np.random.random()
            next_state = np.random.random(self.state_dim)
            done = False
            
            self.agent.store_experience(state, action, reward, next_state, done)
        
        loss = self.agent.train()
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        initial_epsilon = self.agent.epsilon
        
        # Fill memory and train
        for _ in range(self.config.batch_size):
            state = np.random.random(self.state_dim)
            action = np.random.random(self.action_dim)
            reward = 1.0
            next_state = np.random.random(self.state_dim)
            done = False
            self.agent.store_experience(state, action, reward, next_state, done)
        
        self.agent.train()
        
        self.assertLessEqual(self.agent.epsilon, initial_epsilon)
    
    def test_training_statistics(self):
        """Test training statistics collection."""
        stats = self.agent.get_training_statistics()
        
        self.assertIn('agent_id', stats)
        self.assertIn('episodes_trained', stats)
        self.assertIn('current_epsilon', stats)
        self.assertEqual(stats['agent_id'], self.agent_id)
    
    def test_episode_reset(self):
        """Test episode reset functionality."""
        initial_episodes = self.agent.episode_count
        self.agent.reset_episode()
        self.assertEqual(self.agent.episode_count, initial_episodes + 1)


class TestMultiAgentEnvironment(unittest.TestCase):
    """Test cases for multi-agent environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create network topology
        network_config = NetworkConfig(num_iot_devices=5, num_irs_elements=20)
        self.network = NetworkTopology(network_config)
        
        # Create channel model
        self.channel_model = ChannelModel(self.network)
        
        # Create IRS optimizer
        self.irs_optimizer = IRSOptimizer(self.network, self.channel_model)
        
        # Create MADRL config
        self.config = MADRLConfig(max_steps_per_episode=10)
        
        # Create environment
        self.env = MultiAgentEnvironment(
            self.network, self.channel_model, self.irs_optimizer, self.config
        )
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env.network)
        self.assertIsNotNone(self.env.channel_model)
        self.assertIsNotNone(self.env.irs_optimizer)
        self.assertIsNotNone(self.env.reward_calculator)
        self.assertGreater(self.env.state_dim, 0)
        self.assertGreater(self.env.action_dim, 0)
    
    def test_state_dimension_calculation(self):
        """Test state dimension calculation."""
        state_dim = self.env._calculate_state_dimension()
        self.assertIsInstance(state_dim, int)
        self.assertGreater(state_dim, 0)
    
    def test_action_dimension_calculation(self):
        """Test action dimension calculation."""
        action_dim = self.env._calculate_action_dimension()
        self.assertIsInstance(action_dim, int)
        self.assertGreater(action_dim, 0)
    
    def test_agent_addition(self):
        """Test adding agents to the environment."""
        agent_id = 0
        agent = self.env.add_agent(agent_id)
        
        self.assertIn(agent_id, self.env.agents)
        self.assertEqual(agent.agent_id, agent_id)
        self.assertEqual(agent.state_dim, self.env.state_dim)
        self.assertEqual(agent.action_dim, self.env.action_dim)
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        # Add an agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        observations = self.env.reset()
        
        self.assertIn(agent_id, observations)
        self.assertEqual(len(observations[agent_id]), self.env.state_dim)
        self.assertEqual(self.env.episode_step, 0)
        self.assertEqual(self.env.episode_reward, 0.0)
    
    def test_environment_step(self):
        """Test environment step functionality."""
        # Add an agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        # Reset environment
        observations = self.env.reset()
        
        # Take a step
        actions = {agent_id: np.random.uniform(-1, 1, self.env.action_dim)}
        next_observations, rewards, dones, info = self.env.step(actions)
        
        self.assertIn(agent_id, next_observations)
        self.assertIn(agent_id, rewards)
        self.assertIn(agent_id, dones)
        self.assertIsInstance(rewards[agent_id], float)
        self.assertIsInstance(dones[agent_id], bool)
        self.assertIn('episode_step', info)
        self.assertIn('performance_metrics', info)
    
    def test_current_state_calculation(self):
        """Test current state calculation."""
        state = self.env._get_current_state()
        
        self.assertIn('channel_gains', state)
        self.assertIn('phase_shifts', state)
        self.assertIn('sum_rate', state)
        self.assertIn('energy_efficiency', state)
        self.assertIsInstance(state['sum_rate'], (int, float))
        self.assertIsInstance(state['energy_efficiency'], (int, float))
    
    def test_agent_observation(self):
        """Test agent observation generation."""
        # Add an agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        # Reset to initialize state
        self.env.reset()
        
        observation = self.env._get_agent_observation(agent_id)
        
        self.assertEqual(len(observation), self.env.state_dim)
        self.assertIsInstance(observation, np.ndarray)
    
    def test_action_application(self):
        """Test action application to environment."""
        # Add an agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        # Reset environment
        self.env.reset()
        
        # Store initial IRS configuration
        initial_phase_shifts = self.env.irs_optimizer.phase_controller.get_phase_shifts().copy()
        
        # Apply actions
        actions = {agent_id: np.random.uniform(-1, 1, self.env.action_dim)}
        self.env._apply_actions(actions)
        
        # Check that IRS configuration changed
        new_phase_shifts = self.env.irs_optimizer.phase_controller.get_phase_shifts()
        self.assertFalse(np.array_equal(initial_phase_shifts, new_phase_shifts))
    
    def test_environment_statistics(self):
        """Test environment statistics collection."""
        # Add agents
        for i in range(3):
            self.env.add_agent(i)
        
        stats = self.env.get_environment_statistics()
        
        self.assertIn('num_agents', stats)
        self.assertIn('state_dimension', stats)
        self.assertIn('action_dimension', stats)
        self.assertEqual(stats['num_agents'], 3)
        self.assertEqual(stats['state_dimension'], self.env.state_dim)
        self.assertEqual(stats['action_dimension'], self.env.action_dim)
    
    @patch('core.madrl_agent.logging')
    def test_training_agents(self, mock_logging):
        """Test agent training functionality."""
        # Add agents
        for i in range(2):
            self.env.add_agent(i)
        
        # Run short training
        num_episodes = 2
        results = self.env.train_agents(num_episodes)
        
        self.assertIn('episode_rewards', results)
        self.assertIn('episode_lengths', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('agent_statistics', results)
        self.assertEqual(len(results['episode_rewards']), num_episodes)
    
    @patch('core.madrl_agent.logging')
    def test_agent_evaluation(self, mock_logging):
        """Test agent evaluation functionality."""
        # Add agents
        for i in range(2):
            self.env.add_agent(i)
        
        # Run evaluation
        num_episodes = 2
        results = self.env.evaluate_agents(num_episodes)
        
        self.assertIn('episode_rewards', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('average_reward', results)
        self.assertEqual(len(results['episode_rewards']), num_episodes)


class TestIntegration(unittest.TestCase):
    """Integration tests for MADRL components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create complete system
        network_config = NetworkConfig(num_iot_devices=3, num_irs_elements=10)
        self.network = NetworkTopology(network_config)
        self.channel_model = ChannelModel(self.network)
        self.irs_optimizer = IRSOptimizer(self.network, self.channel_model)
        
        config = MADRLConfig(
            max_episodes=2,
            max_steps_per_episode=5,
            memory_size=50,
            batch_size=4
        )
        
        self.env = MultiAgentEnvironment(
            self.network, self.channel_model, self.irs_optimizer, config
        )
    
    def test_complete_training_cycle(self):
        """Test complete training cycle integration."""
        # Add agents
        num_agents = 2
        for i in range(num_agents):
            self.env.add_agent(i)
        
        # Run training
        results = self.env.train_agents(num_episodes=2)
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('episode_rewards', results)
        self.assertIn('agent_statistics', results)
        
        # Verify agents were trained
        for agent_id in range(num_agents):
            self.assertIn(agent_id, results['agent_statistics'])
            agent_stats = results['agent_statistics'][agent_id]
            self.assertGreater(agent_stats['episodes_trained'], 0)
    
    def test_reward_calculation_integration(self):
        """Test reward calculation with real environment state."""
        # Add agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        # Reset and take step
        observations = self.env.reset()
        actions = {agent_id: np.random.uniform(-0.1, 0.1, self.env.action_dim)}
        next_observations, rewards, dones, info = self.env.step(actions)
        
        # Verify reward is calculated
        self.assertIn(agent_id, rewards)
        self.assertIsInstance(rewards[agent_id], float)
        
        # Verify performance metrics are updated
        self.assertIn('performance_metrics', info)
        self.assertIn('sum_rate_bps', info['performance_metrics'])
    
    def test_irs_optimization_integration(self):
        """Test IRS optimization integration with MADRL."""
        # Add agent
        agent_id = 0
        self.env.add_agent(agent_id)
        
        # Reset environment
        self.env.reset()
        
        # Store initial performance
        initial_metrics = self.env.irs_optimizer.evaluate_performance()
        initial_sum_rate = initial_metrics['sum_rate_bps']
        
        # Take several steps with actions that should improve performance
        for _ in range(3):
            # Small positive adjustments
            actions = {agent_id: np.random.uniform(0, 0.1, self.env.action_dim)}
            observations, rewards, dones, info = self.env.step(actions)
        
        # Check that system state has changed
        final_metrics = self.env.irs_optimizer.evaluate_performance()
        final_sum_rate = final_metrics['sum_rate_bps']
        
        # Performance should have changed (not necessarily improved due to random actions)
        self.assertNotEqual(initial_sum_rate, final_sum_rate)


if __name__ == '__main__':
    # Configure TensorFlow to reduce verbosity
    tf.get_logger().setLevel('ERROR')
    
    # Run tests
    unittest.main(verbosity=2)