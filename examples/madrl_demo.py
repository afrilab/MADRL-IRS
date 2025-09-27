#!/usr/bin/env python3
"""
MADRL Agent Demonstration Script

This script demonstrates the Multi-Agent Deep Reinforcement Learning (MADRL)
implementation for green energy optimization in 6G IoT networks with
Intelligent Reflecting Surfaces.

Author: Research Team
Date: 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import NetworkTopology, NetworkConfig
from core.channel_model import ChannelModel, ChannelParams
from core.irs_optimization import IRSOptimizer, IRSOptimizationConfig
from core.madrl_agent import (
    MultiAgentEnvironment, MADRLConfig, MADRLAgent, RewardCalculator
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('madrl_demo.log'),
            logging.StreamHandler()
        ]
    )


def create_network_system():
    """
    Create the complete network system for MADRL demonstration.
    
    Returns:
        Tuple of (network, channel_model, irs_optimizer)
    """
    print("Creating 6G IoT network system...")
    
    # Network configuration
    network_config = NetworkConfig(
        num_iot_devices=10,
        num_irs_elements=50,
        area_size=100.0,
        bs_position=(50.0, 50.0, 10.0),
        irs_position=(80.0, 50.0, 5.0)
    )
    
    # Create network topology
    network = NetworkTopology(network_config)
    
    # Channel parameters for medium conditions
    channel_params = ChannelParams(
        path_loss_exponent_direct=3.0,
        path_loss_exponent_irs=2.5,
        rician_k_direct=5.0,
        rician_k_irs=10.0,
        carrier_frequency=28.0,
        bandwidth=100.0
    )
    
    # Create channel model
    channel_model = ChannelModel(network, channel_params, condition='medium')
    
    # IRS optimization configuration
    irs_config = IRSOptimizationConfig(
        max_iterations=100,
        learning_rate=0.01,
        optimization_objective='sum_rate',
        transmit_power=0.1
    )
    
    # Create IRS optimizer
    irs_optimizer = IRSOptimizer(network, channel_model, irs_config)
    
    print(f"Network created with {len(network.iot_devices)} IoT devices and {network.irs.num_elements} IRS elements")
    
    return network, channel_model, irs_optimizer


def demonstrate_reward_calculator():
    """Demonstrate the reward calculator functionality."""
    print("\n" + "="*60)
    print("REWARD CALCULATOR DEMONSTRATION")
    print("="*60)
    
    config = MADRLConfig()
    reward_calculator = RewardCalculator(config)
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'High Performance, Low Energy',
            'state': {},
            'action': np.array([0.1, 0.2]),
            'next_state': {
                'sum_rate': 3e6,  # 3 Mbps
                'min_rate': 2e5,  # 200 kbps
                'fairness_index': 0.9,
                'total_power': 0.3,  # Low power
                'energy_efficiency': 1.5e6,  # High efficiency
                'renewable_energy_ratio': 0.8,  # High renewable
                'constraint_violated': False
            }
        },
        {
            'name': 'Medium Performance, Medium Energy',
            'state': {},
            'action': np.array([0.05, 0.1]),
            'next_state': {
                'sum_rate': 1.5e6,  # 1.5 Mbps
                'min_rate': 1e5,   # 100 kbps
                'fairness_index': 0.7,
                'total_power': 0.6,  # Medium power
                'energy_efficiency': 8e5,   # Medium efficiency
                'renewable_energy_ratio': 0.5,  # Medium renewable
                'constraint_violated': False
            }
        },
        {
            'name': 'Low Performance, High Energy',
            'state': {},
            'action': np.array([0.2, 0.3]),
            'next_state': {
                'sum_rate': 0.8e6,  # 0.8 Mbps
                'min_rate': 5e4,    # 50 kbps
                'fairness_index': 0.5,
                'total_power': 1.2,  # High power
                'energy_efficiency': 4e5,    # Low efficiency
                'renewable_energy_ratio': 0.2,  # Low renewable
                'constraint_violated': True
            }
        }
    ]
    
    print("Testing different reward scenarios:")
    for scenario in scenarios:
        reward = reward_calculator.calculate_reward(
            scenario['state'], scenario['action'], scenario['next_state']
        )
        print(f"  {scenario['name']}: Reward = {reward:.4f}")
    
    # Show reward statistics
    stats = reward_calculator.get_reward_statistics()
    if stats:
        print(f"\nReward Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")


def demonstrate_single_agent():
    """Demonstrate single MADRL agent functionality."""
    print("\n" + "="*60)
    print("SINGLE AGENT DEMONSTRATION")
    print("="*60)
    
    # Create agent
    config = MADRLConfig(
        hidden_layers=[64, 32],
        memory_size=1000,
        batch_size=16,
        epsilon_decay=0.99
    )
    
    agent = MADRLAgent(
        agent_id=0,
        state_dim=20,
        action_dim=10,
        config=config
    )
    
    print(f"Created agent {agent.agent_id} with state_dim={agent.state_dim}, action_dim={agent.action_dim}")
    
    # Demonstrate action selection
    print("\nTesting action selection:")
    state = np.random.random(20)
    
    # Exploration
    agent.epsilon = 1.0
    action_explore = agent.get_action(state, training=True)
    print(f"  Exploration action: {action_explore[:5]}... (first 5 elements)")
    
    # Exploitation
    agent.epsilon = 0.0
    action_exploit = agent.get_action(state, training=True)
    print(f"  Exploitation action: {action_exploit[:5]}... (first 5 elements)")
    
    # Store some experiences
    print("\nStoring experiences and training:")
    for i in range(50):
        state = np.random.random(20)
        action = np.random.uniform(-1, 1, 10)
        reward = np.random.random()
        next_state = np.random.random(20)
        done = i == 49
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"  Stored {len(agent.memory)} experiences")
    
    # Train agent
    loss = agent.train()
    print(f"  Training loss: {loss:.6f}")
    
    # Show statistics
    stats = agent.get_training_statistics()
    print(f"\nAgent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demonstrate_multi_agent_environment():
    """Demonstrate multi-agent environment functionality."""
    print("\n" + "="*60)
    print("MULTI-AGENT ENVIRONMENT DEMONSTRATION")
    print("="*60)
    
    # Create network system
    network, channel_model, irs_optimizer = create_network_system()
    
    # Create MADRL configuration
    config = MADRLConfig(
        max_episodes=5,
        max_steps_per_episode=20,
        hidden_layers=[32, 16],
        memory_size=500,
        batch_size=8,
        learning_rate=0.01
    )
    
    # Create multi-agent environment
    env = MultiAgentEnvironment(network, channel_model, irs_optimizer, config)
    
    print(f"Created environment with state_dim={env.state_dim}, action_dim={env.action_dim}")
    
    # Add agents
    num_agents = 3
    for i in range(num_agents):
        agent = env.add_agent(i)
        print(f"  Added agent {i}")
    
    print(f"\nEnvironment has {len(env.agents)} agents")
    
    # Demonstrate environment reset and step
    print("\nTesting environment dynamics:")
    observations = env.reset()
    print(f"  Reset complete. Observation shapes: {[obs.shape for obs in observations.values()]}")
    
    # Take a few steps
    for step in range(3):
        # Generate random actions for all agents
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = np.random.uniform(-0.1, 0.1, env.action_dim)
        
        # Execute step
        next_observations, rewards, dones, info = env.step(actions)
        
        print(f"  Step {step + 1}:")
        print(f"    Rewards: {[f'{r:.4f}' for r in rewards.values()]}")
        print(f"    Sum rate: {info['performance_metrics']['sum_rate_mbps']:.2f} Mbps")
        print(f"    Energy efficiency: {info['performance_metrics']['energy_efficiency_bps_per_watt']:.0f} bps/W")
        
        observations = next_observations
        
        if all(dones.values()):
            print("    Episode completed")
            break
    
    # Show environment statistics
    stats = env.get_environment_statistics()
    print(f"\nEnvironment Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


def run_training_demonstration():
    """Run a complete training demonstration."""
    print("\n" + "="*60)
    print("TRAINING DEMONSTRATION")
    print("="*60)
    
    # Create network system
    network, channel_model, irs_optimizer = create_network_system()
    
    # Create MADRL configuration for training
    config = MADRLConfig(
        max_episodes=20,
        max_steps_per_episode=30,
        hidden_layers=[64, 32],
        memory_size=2000,
        batch_size=32,
        learning_rate=0.001,
        epsilon_decay=0.995
    )
    
    # Create environment
    env = MultiAgentEnvironment(network, channel_model, irs_optimizer, config)
    
    # Add agents
    num_agents = 2
    for i in range(num_agents):
        env.add_agent(i)
    
    print(f"Starting training with {num_agents} agents for {config.max_episodes} episodes")
    
    # Run training
    training_results = env.train_agents(config.max_episodes)
    
    # Display results
    print(f"\nTraining completed!")
    print(f"  Episodes: {len(training_results['episode_rewards'])}")
    print(f"  Average reward (last 5): {np.mean(training_results['episode_rewards'][-5:]):.4f}")
    print(f"  Average episode length: {np.mean(training_results['episode_lengths']):.1f}")
    
    # Show agent statistics
    print(f"\nAgent Training Statistics:")
    for agent_id, stats in training_results['agent_statistics'].items():
        print(f"  Agent {agent_id}:")
        for key, value in stats.items():
            if key != 'agent_id':
                print(f"    {key}: {value}")
    
    return training_results


def run_evaluation_demonstration(training_results=None):
    """Run evaluation demonstration."""
    print("\n" + "="*60)
    print("EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create network system
    network, channel_model, irs_optimizer = create_network_system()
    
    # Create configuration
    config = MADRLConfig(
        max_steps_per_episode=50,
        hidden_layers=[64, 32]
    )
    
    # Create environment
    env = MultiAgentEnvironment(network, channel_model, irs_optimizer, config)
    
    # Add agents
    num_agents = 2
    for i in range(num_agents):
        env.add_agent(i)
    
    print(f"Evaluating {num_agents} agents")
    
    # Run evaluation
    evaluation_results = env.evaluate_agents(num_episodes=5)
    
    # Display results
    print(f"\nEvaluation Results:")
    print(f"  Episodes: {len(evaluation_results['episode_rewards'])}")
    print(f"  Average reward: {evaluation_results['average_reward']:.4f}")
    print(f"  Reward std: {evaluation_results['std_reward']:.4f}")
    
    # Show performance metrics from final episode
    final_metrics = evaluation_results['performance_metrics'][-1]
    print(f"\nFinal Episode Performance:")
    print(f"  Sum rate: {final_metrics['sum_rate_mbps']:.2f} Mbps")
    print(f"  Min rate: {final_metrics['min_rate_bps']/1e3:.1f} kbps")
    print(f"  Energy efficiency: {final_metrics['energy_efficiency_bps_per_watt']:.0f} bps/W")
    print(f"  Fairness index: {final_metrics['jains_fairness_index']:.3f}")
    
    return evaluation_results


def create_performance_plots(training_results, evaluation_results):
    """Create performance visualization plots."""
    print("\n" + "="*60)
    print("CREATING PERFORMANCE PLOTS")
    print("="*60)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MADRL Agent Performance Analysis', fontsize=16)
        
        # Training rewards
        axes[0, 0].plot(training_results['episode_rewards'])
        axes[0, 0].set_title('Training Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(training_results['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Sum rate evolution
        sum_rates = [metrics['sum_rate_mbps'] for metrics in training_results['performance_metrics']]
        axes[1, 0].plot(sum_rates)
        axes[1, 0].set_title('Sum Rate Evolution')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sum Rate (Mbps)')
        axes[1, 0].grid(True)
        
        # Energy efficiency evolution
        energy_eff = [metrics['energy_efficiency_bps_per_watt'] for metrics in training_results['performance_metrics']]
        axes[1, 1].plot(energy_eff)
        axes[1, 1].set_title('Energy Efficiency Evolution')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy Efficiency (bps/W)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('madrl_performance.png', dpi=300, bbox_inches='tight')
        print("Performance plots saved as 'madrl_performance.png'")
        
        # Evaluation results histogram
        plt.figure(figsize=(8, 6))
        plt.hist(evaluation_results['episode_rewards'], bins=10, alpha=0.7, edgecolor='black')
        plt.title('Evaluation Episode Rewards Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('madrl_evaluation_histogram.png', dpi=300, bbox_inches='tight')
        print("Evaluation histogram saved as 'madrl_evaluation_histogram.png'")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Matplotlib may not be available or configured properly")


def compare_with_baselines():
    """Compare MADRL performance with baseline methods."""
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    # Create network system
    network, channel_model, irs_optimizer = create_network_system()
    
    # Test different IRS configurations
    baselines = ['random', 'uniform', 'no_irs']
    results = {}
    
    for baseline in baselines:
        print(f"\nTesting {baseline} baseline:")
        comparison = irs_optimizer.compare_with_baseline(baseline)
        results[baseline] = comparison
        
        current_metrics = comparison['current_metrics']
        baseline_metrics = comparison['baseline_metrics']
        improvements = comparison['improvements']
        
        print(f"  Sum rate improvement: {improvements['sum_rate_bps_improvement_percent']:.1f}%")
        print(f"  Energy efficiency improvement: {improvements['energy_efficiency_bps_per_watt_improvement_percent']:.1f}%")
        print(f"  Fairness improvement: {improvements['jains_fairness_index_improvement_percent']:.1f}%")
    
    return results


def main():
    """Main demonstration function."""
    print("="*60)
    print("MADRL AGENT DEMONSTRATION FOR 6G IoT NETWORKS")
    print("="*60)
    
    # Set up logging
    setup_logging()
    
    try:
        # Run demonstrations
        demonstrate_reward_calculator()
        demonstrate_single_agent()
        demonstrate_multi_agent_environment()
        
        # Run training
        training_results = run_training_demonstration()
        
        # Run evaluation
        evaluation_results = run_evaluation_demonstration(training_results)
        
        # Create performance plots
        create_performance_plots(training_results, evaluation_results)
        
        # Compare with baselines
        baseline_results = compare_with_baselines()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Results:")
        print(f"  Training episodes: {len(training_results['episode_rewards'])}")
        print(f"  Final average reward: {np.mean(training_results['episode_rewards'][-5:]):.4f}")
        print(f"  Evaluation average reward: {evaluation_results['average_reward']:.4f}")
        print(f"  Final sum rate: {training_results['performance_metrics'][-1]['sum_rate_mbps']:.2f} Mbps")
        print(f"  Final energy efficiency: {training_results['performance_metrics'][-1]['energy_efficiency_bps_per_watt']:.0f} bps/W")
        
        print("\nFiles generated:")
        print("  - madrl_demo.log (execution log)")
        print("  - madrl_performance.png (training plots)")
        print("  - madrl_evaluation_histogram.png (evaluation results)")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)