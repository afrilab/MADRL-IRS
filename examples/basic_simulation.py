#!/usr/bin/env python3
"""
Basic Simulation Example for 6G Green IoT Networks with FL and IRS

This script demonstrates a basic simulation of the federated learning system
with intelligent reflecting surface optimization as described in the research paper.

Usage:
    python examples/basic_simulation.py [--devices 20] [--rounds 10] [--dataset mnist]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import core modules
from core.network_topology import NetworkTopology
from core.channel_model import ChannelModel
from core.federated_learning import FederatedLearning
from core.irs_optimization import IRSOptimizer
from core.madrl_agent import MADRLAgent

# Import utilities
from utils.visualization import NetworkVisualizer, PerformanceVisualizer
from utils.metrics import (
    FederatedLearningMetrics, WirelessMetrics, EnergyMetrics, 
    generate_performance_report
)
from utils.logging_config import setup_default_logging, log_exception, PerformanceLogger

# Import exceptions
from core.exceptions import (
    ResearchFrameworkError, NetworkConfigurationError, 
    FederatedLearningError, IRSOptimizationError
)


def run_basic_simulation(args):
    """
    Run a basic FL-IRS simulation with comprehensive error handling.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with simulation results
        
    Raises:
        ResearchFrameworkError: If simulation fails
    """
    # Set up logging
    logger = setup_default_logging(level="INFO", log_dir="logs")
    
    try:
        logger.info("=" * 60)
        logger.info("6G Green IoT Networks - Basic Simulation")
        logger.info("=" * 60)
        logger.info(f"Configuration:")
        logger.info(f"  - Number of IoT devices: {args.devices}")
        logger.info(f"  - FL rounds: {args.rounds}")
        logger.info(f"  - Dataset: {args.dataset}")
        logger.info(f"  - Data distribution: {args.distribution}")
        logger.info(f"  - Channel condition: {args.channel}")
        
        results = {}
        
        with PerformanceLogger(logger, "Complete simulation"):
            # Step 1: Initialize Network Topology
            logger.info("Step 1: Initializing network topology...")
            try:
                network = NetworkTopology(
                    num_iot_devices=args.devices,
                    num_irs_elements=100,
                    area_size=100
                )
                logger.info(f"Network topology initialized with {len(network.iot_devices)} IoT devices")
                results['network_initialized'] = True
                
            except NetworkConfigurationError as e:
                logger.error(f"Network topology initialization failed: {e}")
                raise ResearchFrameworkError(
                    f"Failed to initialize network topology: {e.message}"
                ) from e
            
            # Step 2: Initialize Channel Model
            logger.info("Step 2: Initializing channel model...")
            try:
                channel_model = ChannelModel(network)
                logger.info("Channel model initialized successfully")
                results['channel_initialized'] = True
                
            except Exception as e:
                logger.error(f"Channel model initialization failed: {e}")
                log_exception(logger, e, "Channel model initialization")
                raise ResearchFrameworkError(
                    f"Failed to initialize channel model: {str(e)}"
                ) from e
            
            # Step 3: Initialize Federated Learning
            logger.info("Step 3: Initializing federated learning...")
            try:
                fl_system = FederatedLearning(
                    num_devices=args.devices,
                    dataset=args.dataset,
                    data_distribution=args.distribution
                )
                logger.info("Federated learning system initialized successfully")
                results['fl_initialized'] = True
                
            except FederatedLearningError as e:
                logger.error(f"Federated learning initialization failed: {e}")
                raise ResearchFrameworkError(
                    f"Failed to initialize federated learning: {e.message}"
                ) from e
            
            # Step 4: Initialize IRS Optimizer
            logger.info("Step 4: Initializing IRS optimizer...")
            try:
                irs_optimizer = IRSOptimizer(network, channel_model)
                logger.info("IRS optimizer initialized successfully")
                results['irs_initialized'] = True
                
            except IRSOptimizationError as e:
                logger.error(f"IRS optimizer initialization failed: {e}")
                raise ResearchFrameworkError(
                    f"Failed to initialize IRS optimizer: {e.message}"
                ) from e
            
            # Step 5: Run Simulation
            logger.info("Step 5: Running simulation...")
            try:
                simulation_results = _run_simulation_loop(
                    network, channel_model, fl_system, irs_optimizer, args, logger
                )
                results.update(simulation_results)
                
            except Exception as e:
                logger.error(f"Simulation execution failed: {e}")
                log_exception(logger, e, "Simulation execution")
                raise ResearchFrameworkError(
                    f"Simulation execution failed: {str(e)}"
                ) from e
            
            logger.info("Simulation completed successfully")
            return results
            
    except ResearchFrameworkError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in simulation: {e}")
        log_exception(logger, e, "Basic simulation")
        raise ResearchFrameworkError(
            f"Unexpected simulation error: {str(e)}"
        ) from e


def _run_simulation_loop(network, channel_model, fl_system, irs_optimizer, args, logger):
    """
    Run the main simulation loop with error handling.
    
    Args:
        network: Network topology
        channel_model: Channel model
        fl_system: Federated learning system
        irs_optimizer: IRS optimizer
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Dictionary with simulation results
    """
    results = {
        'fl_accuracies': [],
        'irs_gains': [],
        'energy_consumption': [],
        'communication_rounds': args.rounds
    }
    
    try:
        for round_num in range(args.rounds):
            logger.info(f"Round {round_num + 1}/{args.rounds}")
            
            # Optimize IRS configuration
            try:
                irs_gain = irs_optimizer.optimize()
                results['irs_gains'].append(irs_gain)
                logger.debug(f"IRS gain for round {round_num + 1}: {irs_gain:.4f}")
                
            except IRSOptimizationError as e:
                logger.warning(f"IRS optimization failed in round {round_num + 1}: {e}")
                results['irs_gains'].append(0.0)  # Use default gain
            
            # Run federated learning round
            try:
                fl_accuracy = fl_system.run_round()
                results['fl_accuracies'].append(fl_accuracy)
                logger.debug(f"FL accuracy for round {round_num + 1}: {fl_accuracy:.4f}")
                
            except FederatedLearningError as e:
                logger.error(f"Federated learning failed in round {round_num + 1}: {e}")
                raise ResearchFrameworkError(
                    f"Federated learning failed in round {round_num + 1}: {e.message}"
                ) from e
            
            # Calculate energy consumption
            try:
                energy = _calculate_round_energy(network, channel_model, irs_gain)
                results['energy_consumption'].append(energy)
                
            except Exception as e:
                logger.warning(f"Energy calculation failed in round {round_num + 1}: {e}")
                results['energy_consumption'].append(1.0)  # Use default value
        
        # Calculate final metrics
        results['final_accuracy'] = results['fl_accuracies'][-1] if results['fl_accuracies'] else 0.0
        results['average_irs_gain'] = np.mean(results['irs_gains']) if results['irs_gains'] else 0.0
        results['total_energy'] = sum(results['energy_consumption'])
        
        logger.info(f"Final accuracy: {results['final_accuracy']:.4f}")
        logger.info(f"Average IRS gain: {results['average_irs_gain']:.4f}")
        logger.info(f"Total energy consumption: {results['total_energy']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation loop failed: {e}")
        raise


def _calculate_round_energy(network, channel_model, irs_gain):
    """
    Calculate energy consumption for a single round.
    
    Args:
        network: Network topology
        channel_model: Channel model
        irs_gain: IRS gain for this round
        
    Returns:
        Energy consumption value
    """
    # Simplified energy calculation
    base_energy = 1.0  # Base energy consumption
    irs_energy_reduction = irs_gain * 0.1  # IRS reduces energy consumption
    return max(0.1, base_energy - irs_energy_reduction)
    print(f"  ✓ Network created with {args.devices} IoT devices")
    
    # Step 2: Initialize Channel Model
    print("Step 2: Initializing channel model...")
    channel = ChannelModel(network, condition=args.channel)
    print(f"  ✓ Channel model initialized ({args.channel} conditions)")
    
    # Step 3: Initialize IRS Optimizer
    print("Step 3: Initializing IRS optimizer...")
    irs_optimizer = IRSOptimizer(network, channel)
    print("  ✓ IRS optimizer initialized")
    
    # Step 4: Initialize Federated Learning
    print("Step 4: Initializing federated learning...")
    fl_system = FederatedLearning(
        num_devices=args.devices,
        dataset=args.dataset,
        data_distribution=args.distribution,
        local_epochs=5,
        batch_size=32
    )
    print(f"  ✓ FL system initialized with {args.dataset} dataset")
    
    # Step 5: Initialize MADRL Agent (optional)
    if args.use_madrl:
        print("Step 5: Initializing MADRL agent...")
        madrl_agent = MADRLAgent(network, channel)
        print("  ✓ MADRL agent initialized")
    else:
        madrl_agent = None
    
    # Step 6: Run Simulation
    print("\nStep 6: Running simulation...")
    print("-" * 40)
    
    # Storage for metrics
    accuracy_history = []
    loss_history = []
    energy_efficiency_history = []
    spectral_efficiency_history = []
    
    # Simulation loop
    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        
        # Optimize IRS configuration
        if round_num % 5 == 0:  # Optimize every 5 rounds
            print("  Optimizing IRS configuration...")
            irs_optimizer.optimize()
            channel.update_with_irs(irs_optimizer)
        
        # Run FL round
        print("  Running FL round...")
        metrics = fl_system.run_round(channel, network)
        
        # Store metrics
        accuracy_history.append(metrics.get('accuracy', 0))
        loss_history.append(metrics.get('loss', 0))
        
        # Calculate energy and spectral efficiency
        data_rates = channel.get_data_rate(transmit_power=0.1)  # 100mW
        energy_eff = calculate_energy_efficiency(data_rates, power_consumption=0.5)
        spectral_eff = calculate_spectral_efficiency(data_rates, channel.bandwidth)
        
        energy_efficiency_history.append(energy_eff)
        spectral_efficiency_history.append(spectral_eff)
        
        # MADRL optimization (if enabled)
        if madrl_agent and round_num % 3 == 0:
            print("  Running MADRL optimization...")
            madrl_agent.train_step(metrics)
        
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}, "
              f"Loss: {metrics.get('loss', 0):.4f}, "
              f"Energy Eff: {energy_eff:.2f} Mbps/W")
    
    # Step 7: Results Analysis
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    final_accuracy = accuracy_history[-1] if accuracy_history else 0
    final_loss = loss_history[-1] if loss_history else 0
    avg_energy_eff = np.mean(energy_efficiency_history)
    avg_spectral_eff = np.mean(spectral_efficiency_history)
    
    print(f"Final Model Accuracy: {final_accuracy:.4f}")
    print(f"Final Training Loss: {final_loss:.4f}")
    print(f"Average Energy Efficiency: {avg_energy_eff:.2f} Mbps/W")
    print(f"Average Spectral Efficiency: {avg_spectral_eff:.2f} bps/Hz")
    
    # Calculate improvement with IRS
    channel_no_irs = ChannelModel(network, condition=args.channel)
    data_rates_no_irs = channel_no_irs.get_data_rate(transmit_power=0.1)
    data_rates_with_irs = channel.get_data_rate(transmit_power=0.1)
    
    improvement = np.mean(data_rates_with_irs) / np.mean(data_rates_no_irs)
    print(f"IRS Performance Improvement: {improvement:.2f}x")
    
    # Step 8: Generate Visualizations
    if args.plot:
        print("\nGenerating visualizations...")
        
        # Create results directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        
        # Plot network topology
        fig1 = plot_network_topology(network, save_path='results/figures/network_topology.png')
        
        # Plot convergence
        fig2 = plot_convergence(
            accuracy_history, loss_history,
            save_path='results/figures/fl_convergence.png'
        )
        
        # Plot efficiency metrics
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(energy_efficiency_history, 'b-', linewidth=2)
        plt.title('Energy Efficiency Over Time')
        plt.xlabel('FL Round')
        plt.ylabel('Energy Efficiency (Mbps/W)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(spectral_efficiency_history, 'r-', linewidth=2)
        plt.title('Spectral Efficiency Over Time')
        plt.xlabel('FL Round')
        plt.ylabel('Spectral Efficiency (bps/Hz)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/efficiency_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Visualizations saved to results/figures/")
    
    # Step 9: Save Results
    if args.save_results:
        print("\nSaving results...")
        
        # Create results directory
        os.makedirs('results/logs', exist_ok=True)
        
        # Save simulation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results/logs/simulation_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            f.write("6G Green IoT Networks - Simulation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Devices: {args.devices}\n")
            f.write(f"  Rounds: {args.rounds}\n")
            f.write(f"  Dataset: {args.dataset}\n")
            f.write(f"  Distribution: {args.distribution}\n")
            f.write(f"  Channel: {args.channel}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Final Accuracy: {final_accuracy:.4f}\n")
            f.write(f"  Final Loss: {final_loss:.4f}\n")
            f.write(f"  Avg Energy Efficiency: {avg_energy_eff:.2f} Mbps/W\n")
            f.write(f"  Avg Spectral Efficiency: {avg_spectral_eff:.2f} bps/Hz\n")
            f.write(f"  IRS Improvement: {improvement:.2f}x\n")
        
        print(f"  ✓ Results saved to {results_file}")
    
    print("\nSimulation completed successfully!")
    return {
        'accuracy': final_accuracy,
        'loss': final_loss,
        'energy_efficiency': avg_energy_eff,
        'spectral_efficiency': avg_spectral_eff,
        'irs_improvement': improvement
    }


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Basic simulation for 6G Green IoT Networks with FL and IRS'
    )
    
    # Simulation parameters
    parser.add_argument('--devices', type=int, default=20,
                       help='Number of IoT devices (default: 20)')
    parser.add_argument('--rounds', type=int, default=10,
                       help='Number of FL rounds (default: 10)')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non_iid'],
                       help='Data distribution (default: iid)')
    parser.add_argument('--channel', type=str, default='medium',
                       choices=['good', 'medium', 'bad'],
                       help='Channel condition (default: medium)')
    
    # Optional features
    parser.add_argument('--use-madrl', action='store_true',
                       help='Enable MADRL optimization')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Run simulation
    try:
        results = run_basic_simulation(args)
        print(f"\nSimulation Summary:")
        print(f"  Final Accuracy: {results['accuracy']:.4f}")
        print(f"  IRS Improvement: {results['irs_improvement']:.2f}x")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())