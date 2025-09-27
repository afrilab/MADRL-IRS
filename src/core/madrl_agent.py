#!/usr/bin/env python3
"""
Multi-Agent Deep Reinforcement Learning (MADRL) Module for 6G IoT Networks

This module implements the Multi-Agent Deep Reinforcement Learning framework
described in the research paper for green energy optimization in 6G IoT networks
with Intelligent Reflecting Surfaces. It includes agent coordination, reward
calculation, and Deep Q-Network implementation.

Author: Research Team
Date: 2025
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from collections import deque
import random
from .network_topology import NetworkTopology
from .channel_model import ChannelModel
from .irs_optimization import IRSOptimizer
from .exceptions import MADRLError, ValidationError, ComputationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_float_range,
    validate_array_shape, validate_string_choice
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Set up logger
logger = get_logger(__name__)


@dataclass
class MADRLConfig:
    """Configuration parameters for MADRL agents."""
    # Network architecture
    hidden_layers: List[int] = None
    learning_rate: float = 0.001
    discount_factor: float = 0.95

    # Training parameters
    batch_size: int = 32
    memory_size: int = 10000
    target_update_frequency: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Environment parameters
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    reward_scaling: float = 1.0

    # Green energy parameters
    energy_weight: float = 0.3
    performance_weight: float = 0.7
    energy_threshold: float = 0.5  # watts

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        try:
            # Set default hidden layers if not provided
            if self.hidden_layers is None:
                self.hidden_layers = [128, 64, 32]

            # Validate hidden layers
            if not isinstance(self.hidden_layers, list) or not self.hidden_layers:
                raise ValidationError(
                    "hidden_layers must be a non-empty list",
                    parameter='hidden_layers'
                )

            for i, layer_size in enumerate(self.hidden_layers):
                validate_integer_range(
                    layer_size, f'hidden_layers[{i}]', min_value=1, max_value=1024
                )

            # Validate learning parameters
            self.learning_rate = validate_float_range(
                self.learning_rate, 'learning_rate',
                min_value=1e-6, max_value=1.0, inclusive_min=False
            )

            self.discount_factor = validate_float_range(
                self.discount_factor, 'discount_factor',
                min_value=0.0, max_value=1.0
            )

            # Validate training parameters
            self.batch_size = validate_integer_range(
                self.batch_size, 'batch_size', min_value=1, max_value=512
            )

            self.memory_size = validate_integer_range(
                self.memory_size, 'memory_size', min_value=100, max_value=1000000
            )

            self.target_update_frequency = validate_integer_range(
                self.target_update_frequency, 'target_update_frequency',
                min_value=1, max_value=10000
            )

            # Validate epsilon parameters
            self.epsilon_start = validate_float_range(
                self.epsilon_start, 'epsilon_start', min_value=0.0, max_value=1.0
            )

            self.epsilon_end = validate_float_range(
                self.epsilon_end, 'epsilon_end', min_value=0.0, max_value=1.0
            )

            self.epsilon_decay = validate_float_range(
                self.epsilon_decay, 'epsilon_decay', min_value=0.0, max_value=1.0
            )

            # Validate environment parameters
            self.max_episodes = validate_integer_range(
                self.max_episodes, 'max_episodes', min_value=1, max_value=100000
            )

            self.max_steps_per_episode = validate_integer_range(
                self.max_steps_per_episode, 'max_steps_per_episode',
                min_value=1, max_value=10000
            )

            self.reward_scaling = validate_positive_number(
                self.reward_scaling, 'reward_scaling'
            )

            # Validate green energy parameters
            self.energy_weight = validate_float_range(
                self.energy_weight, 'energy_weight', min_value=0.0, max_value=1.0
            )

            self.performance_weight = validate_float_range(
                self.performance_weight, 'performance_weight', min_value=0.0, max_value=1.0
            )

            # Check that weights sum to 1
            if abs(self.energy_weight + self.performance_weight - 1.0) > 1e-6:
                raise ValidationError(
                    "energy_weight and performance_weight must sum to 1.0",
                    parameter='weights'
                )

            self.energy_threshold = validate_positive_number(
                self.energy_threshold, 'energy_threshold'
            )

            logger.debug(f"MADRLConfig validated: {self}")

        except ValidationError as e:
            logger.error(f"MADRLConfig validation failed: {e}")
            raise MADRLError(
                f"Invalid MADRL configuration: {e.message}"
            ) from e


class RewardCalculator:
    """
    Reward Calculator for green energy optimization.

    This class implements the reward function for MADRL agents focusing on
    green energy optimization while maintaining communication performance.
    """

    def __init__(self, config: MADRLConfig):
        """
        Initialize the reward calculator.

        Args:
            config: MADRL configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Historical data for reward normalization
        self.performance_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)

        # Baseline values for normalization
        self.baseline_performance = 1e6  # 1 Mbps baseline
        self.baseline_energy = 1.0  # 1 watt baseline

    def calculate_reward(self, state: Dict[str, Any], action: np.ndarray,
                        next_state: Dict[str, Any]) -> float:
        """
        Calculate reward for green energy optimization.

        The reward function balances communication performance and energy efficiency:
        R = α * R_performance + β * R_energy - γ * R_penalty

        Args:
            state: Current environment state
            action: Action taken by the agent
            next_state: Resulting environment state

        Returns:
            Calculated reward value
        """
        # Performance reward component
        performance_reward = self._calculate_performance_reward(next_state)

        # Energy efficiency reward component
        energy_reward = self._calculate_energy_reward(next_state)

        # Penalty components
        penalty = self._calculate_penalties(state, action, next_state)

        # Combined reward
        total_reward = (self.config.performance_weight * performance_reward +
                       self.config.energy_weight * energy_reward - penalty)

        # Scale reward
        total_reward *= self.config.reward_scaling

        # Update history for normalization
        self.performance_history.append(performance_reward)
        self.energy_history.append(energy_reward)

        return total_reward

    def _calculate_performance_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate performance-based reward component.

        Args:
            state: Environment state containing performance metrics

        Returns:
            Performance reward value
        """
        # Extract performance metrics
        sum_rate = state.get('sum_rate', 0)
        min_rate = state.get('min_rate', 0)
        fairness_index = state.get('fairness_index', 0)

        # Normalize performance metrics
        normalized_sum_rate = sum_rate / self.baseline_performance
        normalized_min_rate = min_rate / (self.baseline_performance / 10)  # Lower baseline for min rate

        # Combined performance reward
        performance_reward = (0.5 * normalized_sum_rate +
                            0.3 * normalized_min_rate +
                            0.2 * fairness_index)

        return performance_reward

    def _calculate_energy_reward(self, state: Dict[str, Any]) -> float:
        """
        Calculate energy efficiency reward component.

        Args:
            state: Environment state containing energy metrics

        Returns:
            Energy efficiency reward value
        """
        # Extract energy metrics
        total_power = state.get('total_power', self.baseline_energy)
        energy_efficiency = state.get('energy_efficiency', 0)
        renewable_energy_ratio = state.get('renewable_energy_ratio', 0)

        # Energy efficiency reward (higher is better)
        efficiency_reward = energy_efficiency / (self.baseline_performance / self.baseline_energy)

        # Power consumption penalty (lower is better)
        power_penalty = max(0, (total_power - self.config.energy_threshold) / self.baseline_energy)

        # Renewable energy bonus
        renewable_bonus = renewable_energy_ratio * 0.5

        # Combined energy reward
        energy_reward = efficiency_reward - power_penalty + renewable_bonus

        return energy_reward

    def _calculate_penalties(self, state: Dict[str, Any], action: np.ndarray,
                           next_state: Dict[str, Any]) -> float:
        """
        Calculate penalty components for the reward function.

        Args:
            state: Current environment state
            action: Action taken by the agent
            next_state: Resulting environment state

        Returns:
            Total penalty value
        """
        penalty = 0.0

        # Action smoothness penalty (discourage rapid changes)
        if 'previous_action' in state:
            action_diff = np.linalg.norm(action - state['previous_action'])
            penalty += 0.1 * action_diff

        # Constraint violation penalties
        if next_state.get('constraint_violated', False):
            penalty += 1.0

        # QoS violation penalty
        min_rate_threshold = 1e5  # 100 kbps minimum
        if next_state.get('min_rate', 0) < min_rate_threshold:
            penalty += 0.5

        return penalty

    def get_reward_statistics(self) -> Dict[str, float]:
        """
        Get statistics about reward components.

        Returns:
            Dictionary with reward statistics
        """
        if not self.performance_history or not self.energy_history:
            return {}

        return {
            'avg_performance_reward': np.mean(self.performance_history),
            'std_performance_reward': np.std(self.performance_history),
            'avg_energy_reward': np.mean(self.energy_history),
            'std_energy_reward': np.std(self.energy_history),
            'performance_weight': self.config.performance_weight,
            'energy_weight': self.config.energy_weight
        }

    def reset_history(self) -> None:
        """Reset reward history."""
        self.performance_history.clear()
        self.energy_history.clear()


class DQNNetwork(tf.keras.Model):
    """
    Deep Q-Network implementation for MADRL agents.

    This class implements the neural network architecture for the DQN agent,
    including the main network and target network for stable training.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int]):
        """
        Initialize the DQN network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_layers: List of hidden layer sizes
        """
        super(DQNNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network layers
        self.layers_list = []

        # Input layer
        self.layers_list.append(tf.keras.layers.Dense(
            hidden_layers[0], activation='relu', input_shape=(state_dim,)
        ))

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers_list.append(tf.keras.layers.Dense(
                hidden_layers[i], activation='relu'
            ))
            self.layers_list.append(tf.keras.layers.Dropout(0.2))

        # Output layer
        self.layers_list.append(tf.keras.layers.Dense(action_dim, activation='linear'))

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the network.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Q-values for all actions
        """
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x


class MADRLAgent:
    """
    Multi-Agent Deep Reinforcement Learning Agent.

    This class implements a single MADRL agent that learns to optimize
    IRS configuration and power allocation for green energy efficiency
    in 6G IoT networks.
    """

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 config: MADRLConfig):
        """
        Initialize the MADRL agent.

        Args:
            agent_id: Unique identifier for the agent
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: MADRL configuration parameters

        Raises:
            MADRLError: If initialization fails
        """
        try:
            # Validate inputs
            self.agent_id = validate_integer_range(
                agent_id, 'agent_id', min_value=0, max_value=1000
            )

            self.state_dim = validate_integer_range(
                state_dim, 'state_dim', min_value=1, max_value=10000
            )

            self.action_dim = validate_integer_range(
                action_dim, 'action_dim', min_value=1, max_value=1000
            )

            if not isinstance(config, MADRLConfig):
                raise ValidationError(
                    "config must be a MADRLConfig instance",
                    parameter='config'
                )

            self.config = config

            # Initialize networks
            self.q_network = DQNNetwork(state_dim, action_dim, config.hidden_layers)
            self.target_network = DQNNetwork(state_dim, action_dim, config.hidden_layers)

            # Initialize optimizer
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

            # Experience replay buffer
            self.memory = deque(maxlen=config.memory_size)

            # Training parameters
            self.epsilon = config.epsilon_start
            self.step_count = 0

            # Performance tracking
            self.episode_rewards = []
            self.episode_losses = []

            logger.info(f"MADRLAgent {self.agent_id} initialized with state_dim={state_dim}, action_dim={action_dim}")

        except ValidationError as e:
            logger.error(f"MADRLAgent initialization failed: {e}")
            raise MADRLError(
                f"Invalid agent parameters: {e.message}",
                agent_id=agent_id
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in MADRLAgent initialization: {e}")
            log_exception(logger, e, f"MADRLAgent {agent_id} initialization")
            raise MADRLError(
                f"Failed to initialize MADRL agent: {str(e)}",
                agent_id=agent_id
            ) from e
        self.episode_count = 0

        # Performance tracking
        self.training_history = []
        self.loss_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize target network
        self._update_target_network()

    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # Greedy action (exploitation)
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            q_values = self.q_network(state_tensor, training=False)
            action_index = tf.argmax(q_values[0]).numpy()

            # Convert discrete action to continuous action
            action = self._discrete_to_continuous_action(action_index)

        return action

    def _discrete_to_continuous_action(self, action_index: int) -> np.ndarray:
        """
        Convert discrete action index to continuous action vector.

        Args:
            action_index: Discrete action index

        Returns:
            Continuous action vector
        """
        # Simple mapping from discrete to continuous actions
        # This can be customized based on the specific action space
        action = np.zeros(self.action_dim)

        # Map action index to phase shift adjustments
        if action_index < self.action_dim:
            action[action_index] = 1.0
        elif action_index < 2 * self.action_dim:
            action[action_index - self.action_dim] = -1.0
        else:
            # Random small adjustment
            action = np.random.uniform(-0.1, 0.1, self.action_dim)

        return action

    def store_experience(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Convert continuous action to discrete for storage
        action_index = self._continuous_to_discrete_action(action)

        experience = (state, action_index, reward, next_state, done)
        self.memory.append(experience)

    def _continuous_to_discrete_action(self, action: np.ndarray) -> int:
        """
        Convert continuous action to discrete action index.

        Args:
            action: Continuous action vector

        Returns:
            Discrete action index
        """
        # Find the element with maximum absolute value
        max_index = np.argmax(np.abs(action))

        # Determine if positive or negative
        if action[max_index] >= 0:
            return max_index
        else:
            return max_index + self.action_dim

    def train(self) -> float:
        """
        Train the agent using experience replay.

        Returns:
            Training loss

        Raises:
            MADRLError: If training fails
        """
        try:
            if len(self.memory) < self.config.batch_size:
                return 0.0

            # Sample batch from memory
            batch = random.sample(self.memory, self.config.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors with validation
            try:
                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.bool)
            except Exception as e:
                raise ComputationError(
                    f"Failed to convert batch to tensors: {str(e)}",
                    operation="tensor_conversion"
                ) from e

            # Validate tensor shapes
            if states.shape[1] != self.state_dim:
                raise ComputationError(
                    f"Invalid state dimension: expected {self.state_dim}, got {states.shape[1]}",
                    operation="shape_validation"
                )

            # Calculate target Q-values
            next_q_values = self.target_network(next_states, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - tf.cast(dones, tf.float32)) * self.config.discount_factor * max_next_q_values

            # Check for invalid values
            if tf.reduce_any(tf.math.is_nan(target_q_values)) or tf.reduce_any(tf.math.is_inf(target_q_values)):
                raise ComputationError(
                    "Invalid target Q-values (NaN or Inf)",
                    operation="target_calculation"
                )

            # Train the network
            with tf.GradientTape() as tape:
                current_q_values = self.q_network(states, training=True)
                action_masks = tf.one_hot(actions, self.action_dim)
                masked_q_values = tf.reduce_sum(current_q_values * action_masks, axis=1)

                loss = tf.keras.losses.mse(target_q_values, masked_q_values)

                # Check for invalid loss
                if tf.math.is_nan(loss) or tf.math.is_inf(loss):
                    raise ComputationError(
                        "Invalid loss value (NaN or Inf)",
                        operation="loss_calculation"
                    )

            # Apply gradients
            gradients = tape.gradient(loss, self.q_network.trainable_variables)

            # Check for invalid gradients
            if any(tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad))
                   for grad in gradients if grad is not None):
                raise ComputationError(
                    "Invalid gradients (NaN or Inf)",
                    operation="gradient_calculation"
                )

            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

            # Update epsilon
            if self.epsilon > self.config.epsilon_end:
                self.epsilon *= self.config.epsilon_decay

            # Update target network
            self.step_count += 1
            if self.step_count % self.config.target_update_frequency == 0:
                self._update_target_network()

            # Store loss
            loss_value = float(loss.numpy())
            if hasattr(self, 'loss_history'):
                self.loss_history.append(loss_value)
            else:
                self.loss_history = [loss_value]

            return loss_value

        except (ComputationError, MADRLError):
            raise
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Training failed: {e}")
            log_exception(logger, e, f"MADRL Agent {self.agent_id} training")
            raise MADRLError(
                f"Agent {self.agent_id}: Training failed: {str(e)}",
                agent_id=self.agent_id
            ) from e

        return loss_value

    def _update_target_network(self) -> None:
        """Update target network weights."""
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        self.q_network.save_weights(filepath)
        self.logger.info(f"Agent {self.agent_id} model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to load the model from
        """
        self.q_network.load_weights(filepath)
        self._update_target_network()
        self.logger.info(f"Agent {self.agent_id} model loaded from {filepath}")

    def get_training_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        if not self.loss_history:
            return {}

        return {
            'agent_id': self.agent_id,
            'episodes_trained': self.episode_count,
            'steps_trained': self.step_count,
            'current_epsilon': self.epsilon,
            'avg_loss': np.mean(self.loss_history[-100:]),  # Last 100 losses
            'memory_size': len(self.memory),
            'total_experiences': len(self.memory)
        }

    def reset_episode(self) -> None:
        """Reset agent for new episode."""
        self.episode_count += 1


class MultiAgentEnvironment:
    """
    Multi-Agent Environment for agent coordination.

    This class manages the environment for multiple MADRL agents,
    handling state transitions, reward calculation, and agent coordination
    for green energy optimization in 6G IoT networks.
    """

    def __init__(self, network: NetworkTopology, channel_model: ChannelModel,
                 irs_optimizer: IRSOptimizer, config: MADRLConfig):
        """
        Initialize the multi-agent environment.

        Args:
            network: Network topology
            channel_model: Channel model
            irs_optimizer: IRS optimizer
            config: MADRL configuration
        """
        self.network = network
        self.channel_model = channel_model
        self.irs_optimizer = irs_optimizer
        self.config = config

        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(config)

        # Environment state
        self.current_state = None
        self.episode_step = 0
        self.episode_reward = 0.0

        # State and action dimensions
        self.state_dim = self._calculate_state_dimension()
        self.action_dim = self._calculate_action_dimension()

        # Initialize agents
        self.agents = {}
        self.num_agents = len(network.iot_devices)

        # Performance tracking
        self.episode_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _calculate_state_dimension(self) -> int:
        """Calculate the dimension of the state space."""
        # State includes:
        # - Channel gains for each device
        # - Current IRS configuration
        # - Power levels
        # - Energy metrics
        # - Network topology information

        num_devices = len(self.network.iot_devices)
        num_irs_elements = self.network.irs.num_elements

        state_dim = (
            num_devices +           # Channel gains
            num_irs_elements +      # IRS phase shifts
            num_devices +           # Power levels
            5 +                     # Energy metrics (total power, efficiency, etc.)
            3                       # Network statistics (distances, etc.)
        )

        return state_dim

    def _calculate_action_dimension(self) -> int:
        """Calculate the dimension of the action space."""
        # Actions include:
        # - IRS phase shift adjustments
        # - Power allocation adjustments

        num_irs_elements = self.network.irs.num_elements
        num_devices = len(self.network.iot_devices)

        # Simplified action space (subset of IRS elements + power control)
        action_dim = min(10, num_irs_elements) + num_devices

        return action_dim

    def add_agent(self, agent_id: int) -> MADRLAgent:
        """
        Add a new agent to the environment.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            Created MADRL agent
        """
        agent = MADRLAgent(agent_id, self.state_dim, self.action_dim, self.config)
        self.agents[agent_id] = agent
        return agent

    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state observations for all agents
        """
        # Reset episode counters
        self.episode_step = 0
        self.episode_reward = 0.0

        # Randomize network conditions
        self.channel_model.regenerate_channels()
        self.irs_optimizer.phase_controller.randomize_phase_shifts()

        # Calculate initial state
        self.current_state = self._get_current_state()

        # Return initial observations for all agents
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_agent_observation(agent_id)

        return observations

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray],
                                                           Dict[int, float],
                                                           Dict[int, bool],
                                                           Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        # Store previous state
        previous_state = self.current_state.copy()

        # Apply actions to the environment
        self._apply_actions(actions)

        # Update environment state
        self.current_state = self._get_current_state()

        # Calculate rewards for each agent
        rewards = {}
        for agent_id in self.agents:
            agent_reward = self.reward_calculator.calculate_reward(
                previous_state, actions[agent_id], self.current_state
            )
            rewards[agent_id] = agent_reward
            self.episode_reward += agent_reward

        # Get new observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_agent_observation(agent_id)

        # Check if episode is done
        self.episode_step += 1
        done = self.episode_step >= self.config.max_steps_per_episode
        dones = {agent_id: done for agent_id in self.agents}

        # Additional info
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'performance_metrics': self.irs_optimizer.evaluate_performance()
        }

        return observations, rewards, dones, info

    def _apply_actions(self, actions: Dict[int, np.ndarray]) -> None:
        """
        Apply agent actions to the environment.

        Args:
            actions: Dictionary of actions for each agent
        """
        # Combine actions from all agents
        combined_action = np.zeros(self.action_dim)
        for agent_id, action in actions.items():
            combined_action += action / len(actions)  # Average actions

        # Apply IRS phase shift adjustments
        num_irs_elements = self.network.irs.num_elements
        irs_action_dim = min(10, num_irs_elements)

        current_phase_shifts = self.irs_optimizer.phase_controller.get_phase_shifts()
        phase_adjustments = combined_action[:irs_action_dim]

        # Apply adjustments to a subset of IRS elements
        indices = np.linspace(0, num_irs_elements-1, irs_action_dim, dtype=int)
        for i, idx in enumerate(indices):
            current_phase_shifts[idx] += 0.1 * phase_adjustments[i]  # Small adjustments

        # Update IRS configuration
        self.irs_optimizer.phase_controller.set_phase_shifts(current_phase_shifts)
        self.network.irs.set_phase_shifts(current_phase_shifts)

        # Update channel model with new IRS configuration
        self.channel_model.update_with_irs()

    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.

        Returns:
            Dictionary containing current state information
        """
        # Update performance metrics
        performance_metrics = self.irs_optimizer.evaluate_performance()

        # Get channel information
        channel_gains = self.channel_model.get_channel_gain().flatten()
        snr_values = self.channel_model.calculate_snr(self.config.energy_threshold)

        # Get IRS configuration
        phase_shifts = self.irs_optimizer.phase_controller.get_phase_shifts()

        # Calculate energy metrics
        total_power = self.config.energy_threshold  # Simplified
        energy_efficiency = performance_metrics.get('energy_efficiency_bps_per_watt', 0)

        state = {
            'channel_gains': channel_gains,
            'snr_values': snr_values,
            'phase_shifts': phase_shifts,
            'sum_rate': performance_metrics.get('sum_rate_bps', 0),
            'min_rate': performance_metrics.get('min_rate_bps', 0),
            'fairness_index': performance_metrics.get('jains_fairness_index', 0),
            'total_power': total_power,
            'energy_efficiency': energy_efficiency,
            'renewable_energy_ratio': 0.5,  # Simplified assumption
            'constraint_violated': False,
            'episode_step': self.episode_step
        }

        return state

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            State observation array
        """
        state = self.current_state

        # Construct observation vector
        observation = []

        # Channel gains
        observation.extend(state['channel_gains'])

        # IRS phase shifts (subset)
        phase_shifts = state['phase_shifts']
        observation.extend(phase_shifts[:min(10, len(phase_shifts))])

        # Power levels (simplified)
        num_devices = len(self.network.iot_devices)
        power_levels = np.full(num_devices, self.config.energy_threshold / num_devices)
        observation.extend(power_levels)

        # Energy metrics
        observation.extend([
            state['total_power'],
            state['energy_efficiency'],
            state['renewable_energy_ratio'],
            state['sum_rate'] / 1e6,  # Normalize to Mbps
            state['fairness_index']
        ])

        # Network statistics
        distances = self.network.get_bs_iot_distances()
        observation.extend([
            np.mean(distances),
            np.std(distances),
            state['episode_step'] / self.config.max_steps_per_episode
        ])

        # Pad or truncate to match state dimension
        observation = np.array(observation)
        if len(observation) > self.state_dim:
            observation = observation[:self.state_dim]
        elif len(observation) < self.state_dim:
            padding = np.zeros(self.state_dim - len(observation))
            observation = np.concatenate([observation, padding])

        return observation

    def train_agents(self, num_episodes: int) -> Dict[str, Any]:
        """
        Train all agents in the environment.

        Args:
            num_episodes: Number of training episodes

        Returns:
            Training results
        """
        self.logger.info(f"Starting MADRL training for {num_episodes} episodes")

        training_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'performance_metrics': [],
            'agent_statistics': {}
        }

        for episode in range(num_episodes):
            # Reset environment
            observations = self.reset()
            episode_reward = 0.0
            episode_length = 0

            while True:
                # Get actions from all agents
                actions = {}
                for agent_id, agent in self.agents.items():
                    action = agent.get_action(observations[agent_id], training=True)
                    actions[agent_id] = action

                # Execute step
                next_observations, rewards, dones, info = self.step(actions)

                # Store experiences and train agents
                for agent_id, agent in self.agents.items():
                    agent.store_experience(
                        observations[agent_id],
                        actions[agent_id],
                        rewards[agent_id],
                        next_observations[agent_id],
                        dones[agent_id]
                    )

                    # Train agent
                    loss = agent.train()

                # Update for next iteration
                observations = next_observations
                episode_reward += sum(rewards.values())
                episode_length += 1

                # Check if episode is done
                if all(dones.values()):
                    break

            # Reset agents for new episode
            for agent in self.agents.values():
                agent.reset_episode()

            # Store episode results
            training_results['episode_rewards'].append(episode_reward)
            training_results['episode_lengths'].append(episode_length)
            training_results['performance_metrics'].append(info['performance_metrics'])

            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(training_results['episode_rewards'][-100:])
                self.logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

        # Get final agent statistics
        for agent_id, agent in self.agents.items():
            training_results['agent_statistics'][agent_id] = agent.get_training_statistics()

        self.logger.info("MADRL training completed")
        return training_results

    def evaluate_agents(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained agents.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating agents for {num_episodes} episodes")

        evaluation_results = {
            'episode_rewards': [],
            'performance_metrics': [],
            'final_configurations': []
        }

        for episode in range(num_episodes):
            # Reset environment
            observations = self.reset()
            episode_reward = 0.0

            while True:
                # Get actions from all agents (no exploration)
                actions = {}
                for agent_id, agent in self.agents.items():
                    action = agent.get_action(observations[agent_id], training=False)
                    actions[agent_id] = action

                # Execute step
                next_observations, rewards, dones, info = self.step(actions)

                # Update for next iteration
                observations = next_observations
                episode_reward += sum(rewards.values())

                # Check if episode is done
                if all(dones.values()):
                    break

            # Store results
            evaluation_results['episode_rewards'].append(episode_reward)
            evaluation_results['performance_metrics'].append(info['performance_metrics'])
            evaluation_results['final_configurations'].append(
                self.irs_optimizer.phase_controller.get_phase_shifts().copy()
            )

        # Calculate statistics
        evaluation_results['average_reward'] = np.mean(evaluation_results['episode_rewards'])
        evaluation_results['std_reward'] = np.std(evaluation_results['episode_rewards'])

        self.logger.info(f"Evaluation completed. Average reward: {evaluation_results['average_reward']:.2f}")
        return evaluation_results

    def get_environment_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics.

        Returns:
            Dictionary with environment statistics
        """
        return {
            'num_agents': len(self.agents),
            'state_dimension': self.state_dim,
            'action_dimension': self.action_dim,
            'num_iot_devices': len(self.network.iot_devices),
            'num_irs_elements': self.network.irs.num_elements,
            'reward_statistics': self.reward_calculator.get_reward_statistics(),
            'current_episode_step': self.episode_step,
            'current_episode_reward': self.episode_reward
        }

    def __repr__(self) -> str:
        return (f"MultiAgentEnvironment(agents={len(self.agents)}, "
                f"state_dim={self.state_dim}, action_dim={self.action_dim})")
