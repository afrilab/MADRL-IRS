#!/usr/bin/env python3
"""
IRS Optimization Demo

This script demonstrates the usage of the IRS optimization module
for 6G IoT networks with intelligent reflecting surfaces.

Author: Research Team
Date: 2025
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import NetworkTopology, NetworkConfig
from core.channel_model import ChannelModel, ChannelParams
from core.irs_optimization import IRSOptimizer, IRSOptimizationConfig


def main():
    """Main demonstration function."""
    print("IRS Optimization Demo for 6G IoT Networks")
    print("=" * 50)
    
    # Create network topology
    print("\n1. Creating Network Topology...")
    network_config = NetworkConfig(
        num_iot_devices=10,
        num_irs_elements=64,
        area_size=100.0,
        bs_position=(50.0, 50.0, 10.0),
        irs_position=(75.0, 25.0, 5.0)
    )
    network = NetworkTopology(network_config)
    print(f"   Network created: {network}")
    
    # Create channel model
    print("\n2. Initializing Channel Model...")
    channel_params = ChannelParams(
        path_loss_exponent_direct=3.0,
        path_loss_exponent_irs=2.5,
        rician_k_direct=5.0,
        rician_k_irs=10.0
    )
    channel_model = ChannelModel(network, channel_params, condition='medium')
    print(f"   Channel model created: {channel_model}")
    
    # Create IRS optimizer
    print("\n3. Setting up IRS Optimizer...")
    opt_config = IRSOptimizationConfig(
        max_iterations=100,
        learning_rate=0.01,
        convergence_threshold=1e-5,
        optimization_objective='sum_rate',
        transmit_power=0.1,
        use_momentum=True,
        momentum_factor=0.9
    )
    optimizer = IRSOptimizer(network, channel_model, opt_config)
    print(f"   Optimizer created: {optimizer}")
    
    # Evaluate initial performance
    print("\n4. Evaluating Initial Performance...")
    initial_metrics = optimizer.evaluate_performance()
    print(f"   Initial Sum Rate: {initial_metrics['sum_rate_mbps']:.2f} Mbps")
    print(f"   Initial Average SNR: {initial_metrics['average_snr_db']:.2f} dB")
    print(f"   Initial Energy Efficiency: {initial_metrics['energy_efficiency_bps_per_watt']:.0f} bps/W")
    
    # Compare with random baseline
    print("\n5. Comparing with Random Baseline...")
    random_comparison = optimizer.compare_with_baseline('random')
    print(f"   Random baseline sum rate: {random_comparison['baseline_metrics']['sum_rate_mbps']:.2f} Mbps")
    
    # Run optimization
    print("\n6. Running IRS Optimization...")
    print("   Optimizing phase shifts for maximum sum rate...")
    results = optimizer.optimize()
    
    print(f"   Optimization completed in {results['iterations']} iterations")
    print(f"   Converged: {results['converged']}")
    print(f"   Best objective value: {results['best_objective_value']:.2f}")
    
    # Evaluate final performance
    print("\n7. Final Performance Metrics...")
    final_metrics = results['final_metrics']
    print(f"   Final Sum Rate: {final_metrics['sum_rate_mbps']:.2f} Mbps")
    print(f"   Final Average SNR: {final_metrics['average_snr_db']:.2f} dB")
    print(f"   Final Energy Efficiency: {final_metrics['energy_efficiency_bps_per_watt']:.0f} bps/W")
    print(f"   Jain's Fairness Index: {final_metrics['jains_fairness_index']:.3f}")
    
    # Calculate improvements
    sum_rate_improvement = (final_metrics['sum_rate_mbps'] / initial_metrics['sum_rate_mbps'] - 1) * 100
    snr_improvement = final_metrics['average_snr_db'] - initial_metrics['average_snr_db']
    
    print(f"\n8. Performance Improvements...")
    print(f"   Sum Rate Improvement: {sum_rate_improvement:.1f}%")
    print(f"   SNR Improvement: {snr_improvement:.2f} dB")
    
    # Compare with different baselines
    print("\n9. Baseline Comparisons...")
    baselines = ['random', 'uniform', 'no_irs']
    
    for baseline in baselines:
        comparison = optimizer.compare_with_baseline(baseline)
        improvement = comparison['improvements']['sum_rate_mbps_improvement_percent']
        print(f"   vs {baseline.capitalize()}: {improvement:.1f}% improvement")
    
    # Test different optimization objectives
    print("\n10. Testing Different Objectives...")
    objectives = ['sum_rate', 'min_rate', 'energy_efficiency']
    
    for objective in objectives:
        obj_config = IRSOptimizationConfig(
            max_iterations=50,
            optimization_objective=objective
        )
        obj_optimizer = IRSOptimizer(network, channel_model, obj_config)
        obj_results = obj_optimizer.optimize()
        obj_metrics = obj_results['final_metrics']
        
        print(f"   {objective.replace('_', ' ').title()}:")
        print(f"     Sum Rate: {obj_metrics['sum_rate_mbps']:.2f} Mbps")
        print(f"     Min Rate: {obj_metrics['min_rate_bps']/1e6:.2f} Mbps")
        print(f"     Energy Efficiency: {obj_metrics['energy_efficiency_bps_per_watt']:.0f} bps/W")
    
    print("\n" + "=" * 50)
    print("IRS Optimization Demo Completed Successfully!")


if __name__ == "__main__":
    main()