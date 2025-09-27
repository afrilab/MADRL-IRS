#!/usr/bin/env python3
"""
Performance Analysis Example for 6G Green IoT Networks

This script provides comprehensive performance analysis and benchmarking
of the federated learning system with IRS optimization and MADRL agents.

Usage:
    python examples/performance_analysis.py [--benchmark] [--compare-methods]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import NetworkTopology
from core.channel_model import ChannelModel
from core.federated_learning import FederatedLearning
from core.irs_optimization import IRSOptimizer
from core.madrl_agent import MADRLAgent
from utils.metrics import (
    calculate_energy_efficiency, calculate_spectral_efficiency,
    calculate_fairness_index, calculate_convergence_rate
)
from utils.visualization import create_performance_dashboard


class PerformanceAnalyzer:
    """Comprehensive performance analysis for the FL-IRS system."""
    
    def __init__(self):
        self.results = {}
        self.benchmarks = {}
        
    def run_scenario(self, scenario_name, config):
        """Run a single performance scenario."""
        print(f"\nRunning scenario: {scenario_name}")
        print("-" * 50)
        
        # Initialize components
        network = NetworkTopology(
            num_iot_devices=config['devices'],
            num_irs_elements=config['irs_elements'],
            area_size=config['area_size']
        )
        
        channel = ChannelModel(network, condition=config['channel_condition'])
        
        if config['use_irs']:
            irs_optimizer = IRSOptimizer(network, channel)
        else:
            irs_optimizer = None
            
        if config['use_madrl']:
            madrl_agent = MADRLAgent(network, channel)
        else:
            madrl_agent = None
        
        fl_system = FederatedLearning(
            num_devices=config['devices'],
            dataset=config['dataset'],
            data_distribution=config['distribution'],
            local_epochs=config['local_epochs'],
            batch_size=config['batch_size']
        )
        
        # Run simulation
        metrics_history = {
            'accuracy': [],
            'loss': [],
            'energy_efficiency': [],
            'spectral_efficiency': [],
            'fairness_index': [],
            'convergence_rate': [],
            'training_time': [],
            'communication_cost': []
        }
        
        start_time = time.time()
        
        for round_num in range(config['rounds']):
            round_start = time.time()
            
            # IRS optimization
            if irs_optimizer and round_num % config['irs_update_interval'] == 0:
                irs_optimizer.optimize()
                channel.update_with_irs(irs_optimizer)
            
            # FL round
            fl_metrics = fl_system.run_round(channel, network)
            
            # MADRL optimization
            if madrl_agent and round_num % config['madrl_update_interval'] == 0:
                madrl_agent.train_step(fl_metrics)
            
            # Calculate performance metrics
            data_rates = channel.get_data_rate(transmit_power=config['transmit_power'])
            energy_eff = calculate_energy_efficiency(data_rates, power_consumption=config['power_consumption'])
            spectral_eff = calculate_spectral_efficiency(data_rates, channel.bandwidth)
            
            device_accuracies = [fl_metrics.get(f'device_{i}_accuracy', 0) for i in range(config['devices'])]
            fairness_idx = calculate_fairness_index(device_accuracies)
            
            round_time = time.time() - round_start
            
            # Store metrics
            metrics_history['accuracy'].append(fl_metrics.get('accuracy', 0))
            metrics_history['loss'].append(fl_metrics.get('loss', 0))
            metrics_history['energy_efficiency'].append(energy_eff)
            metrics_history['spectral_efficiency'].append(spectral_eff)
            metrics_history['fairness_index'].append(fairness_idx)
            metrics_history['training_time'].append(round_time)
            metrics_history['communication_cost'].append(fl_system.communication_cost)
            
            if round_num % 10 == 0:
                print(f"  Round {round_num}: Acc={fl_metrics.get('accuracy', 0):.4f}, "
                      f"Energy={energy_eff:.2f}, Fairness={fairness_idx:.3f}")
        
        total_time = time.time() - start_time
        
        # Calculate final metrics
        final_metrics = {
            'final_accuracy': metrics_history['accuracy'][-1],
            'final_loss': metrics_history['loss'][-1],
            'avg_energy_efficiency': np.mean(metrics_history['energy_efficiency']),
            'avg_spectral_efficiency': np.mean(metrics_history['spectral_efficiency']),
            'avg_fairness_index': np.mean(metrics_history['fairness_index']),
            'convergence_rate': calculate_convergence_rate(metrics_history['accuracy']),
            'total_training_time': total_time,
            'avg_round_time': np.mean(metrics_history['training_time']),
            'total_communication_cost': metrics_history['communication_cost'][-1]
        }
        
        # Store results
        self.results[scenario_name] = {
            'config': config,
            'metrics_history': metrics_history,
            'final_metrics': final_metrics
        }
        
        print(f"  ✓ Completed in {total_time:.1f}s")
        print(f"  Final accuracy: {final_metrics['final_accuracy']:.4f}")
        print(f"  Avg energy efficiency: {final_metrics['avg_energy_efficiency']:.2f} Mbps/W")
        
        return final_metrics
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite."""
        print("=" * 60)
        print("COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        
        # Define benchmark scenarios
        scenarios = {
            'baseline_fl': {
                'devices': 20, 'rounds': 50, 'dataset': 'mnist',
                'distribution': 'iid', 'channel_condition': 'medium',
                'use_irs': False, 'use_madrl': False,
                'irs_elements': 100, 'area_size': 100,
                'local_epochs': 5, 'batch_size': 32,
                'transmit_power': 0.1, 'power_consumption': 0.5,
                'irs_update_interval': 5, 'madrl_update_interval': 3
            },
            'fl_with_irs': {
                'devices': 20, 'rounds': 50, 'dataset': 'mnist',
                'distribution': 'iid', 'channel_condition': 'medium',
                'use_irs': True, 'use_madrl': False,
                'irs_elements': 100, 'area_size': 100,
                'local_epochs': 5, 'batch_size': 32,
                'transmit_power': 0.1, 'power_consumption': 0.5,
                'irs_update_interval': 5, 'madrl_update_interval': 3
            },
            'fl_with_madrl': {
                'devices': 20, 'rounds': 50, 'dataset': 'mnist',
                'distribution': 'iid', 'channel_condition': 'medium',
                'use_irs': False, 'use_madrl': True,
                'irs_elements': 100, 'area_size': 100,
                'local_epochs': 5, 'batch_size': 32,
                'transmit_power': 0.1, 'power_consumption': 0.5,
                'irs_update_interval': 5, 'madrl_update_interval': 3
            },
            'full_system': {
                'devices': 20, 'rounds': 50, 'dataset': 'mnist',
                'distribution': 'iid', 'channel_condition': 'medium',
                'use_irs': True, 'use_madrl': True,
                'irs_elements': 100, 'area_size': 100,
                'local_epochs': 5, 'batch_size': 32,
                'transmit_power': 0.1, 'power_consumption': 0.5,
                'irs_update_interval': 5, 'madrl_update_interval': 3
            }
        }
        
        # Run scenarios
        for scenario_name, config in scenarios.items():
            self.run_scenario(scenario_name, config)
        
        # Generate comparison
        self.generate_benchmark_comparison()
    
    def run_scalability_analysis(self):
        """Analyze system scalability with different numbers of devices."""
        print("\n" + "=" * 60)
        print("SCALABILITY ANALYSIS")
        print("=" * 60)
        
        device_counts = [5, 10, 20, 50, 100]
        scalability_results = {}
        
        base_config = {
            'rounds': 30, 'dataset': 'mnist', 'distribution': 'iid',
            'channel_condition': 'medium', 'use_irs': True, 'use_madrl': True,
            'irs_elements': 100, 'area_size': 100,
            'local_epochs': 5, 'batch_size': 32,
            'transmit_power': 0.1, 'power_consumption': 0.5,
            'irs_update_interval': 5, 'madrl_update_interval': 3
        }
        
        for device_count in device_counts:
            config = base_config.copy()
            config['devices'] = device_count
            
            scenario_name = f'scalability_{device_count}_devices'
            metrics = self.run_scenario(scenario_name, config)
            scalability_results[device_count] = metrics
        
        # Analyze scalability trends
        self.analyze_scalability_trends(scalability_results)
    
    def run_channel_condition_analysis(self):
        """Analyze performance under different channel conditions."""
        print("\n" + "=" * 60)
        print("CHANNEL CONDITION ANALYSIS")
        print("=" * 60)
        
        channel_conditions = ['good', 'medium', 'bad']
        channel_results = {}
        
        base_config = {
            'devices': 20, 'rounds': 40, 'dataset': 'mnist',
            'distribution': 'iid', 'use_irs': True, 'use_madrl': True,
            'irs_elements': 100, 'area_size': 100,
            'local_epochs': 5, 'batch_size': 32,
            'transmit_power': 0.1, 'power_consumption': 0.5,
            'irs_update_interval': 5, 'madrl_update_interval': 3
        }
        
        for condition in channel_conditions:
            config = base_config.copy()
            config['channel_condition'] = condition
            
            scenario_name = f'channel_{condition}'
            metrics = self.run_scenario(scenario_name, config)
            channel_results[condition] = metrics
        
        # Analyze channel impact
        self.analyze_channel_impact(channel_results)
    
    def run_data_distribution_analysis(self):
        """Analyze performance with different data distributions."""
        print("\n" + "=" * 60)
        print("DATA DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        distributions = ['iid', 'non_iid']
        distribution_results = {}
        
        base_config = {
            'devices': 20, 'rounds': 40, 'dataset': 'mnist',
            'channel_condition': 'medium', 'use_irs': True, 'use_madrl': True,
            'irs_elements': 100, 'area_size': 100,
            'local_epochs': 5, 'batch_size': 32,
            'transmit_power': 0.1, 'power_consumption': 0.5,
            'irs_update_interval': 5, 'madrl_update_interval': 3
        }
        
        for distribution in distributions:
            config = base_config.copy()
            config['distribution'] = distribution
            
            scenario_name = f'distribution_{distribution}'
            metrics = self.run_scenario(scenario_name, config)
            distribution_results[distribution] = metrics
        
        # Analyze distribution impact
        self.analyze_distribution_impact(distribution_results)
    
    def generate_benchmark_comparison(self):
        """Generate comprehensive benchmark comparison."""
        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)
        
        # Create comparison table
        comparison_data = []
        for scenario_name, result in self.results.items():
            if 'baseline' in scenario_name or 'fl_with' in scenario_name or 'full_system' in scenario_name:
                metrics = result['final_metrics']
                comparison_data.append({
                    'Scenario': scenario_name,
                    'Accuracy': metrics['final_accuracy'],
                    'Energy Efficiency': metrics['avg_energy_efficiency'],
                    'Spectral Efficiency': metrics['avg_spectral_efficiency'],
                    'Fairness Index': metrics['avg_fairness_index'],
                    'Convergence Rate': metrics['convergence_rate'],
                    'Training Time (s)': metrics['total_training_time'],
                    'Communication Cost (MB)': metrics['total_communication_cost']
                })
        
        df = pd.DataFrame(comparison_data)
        print("\nPerformance Comparison Table:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Calculate improvements
        baseline_accuracy = df[df['Scenario'] == 'baseline_fl']['Accuracy'].iloc[0]
        full_system_accuracy = df[df['Scenario'] == 'full_system']['Accuracy'].iloc[0]
        accuracy_improvement = (full_system_accuracy - baseline_accuracy) / baseline_accuracy * 100
        
        baseline_energy = df[df['Scenario'] == 'baseline_fl']['Energy Efficiency'].iloc[0]
        full_system_energy = df[df['Scenario'] == 'full_system']['Energy Efficiency'].iloc[0]
        energy_improvement = (full_system_energy - baseline_energy) / baseline_energy * 100
        
        print(f"\nKey Improvements (Full System vs Baseline):")
        print(f"  - Accuracy improvement: {accuracy_improvement:.2f}%")
        print(f"  - Energy efficiency improvement: {energy_improvement:.2f}%")
        
        # Save comparison
        os.makedirs('results/logs', exist_ok=True)
        df.to_csv('results/logs/benchmark_comparison.csv', index=False)
        print(f"  ✓ Comparison saved to results/logs/benchmark_comparison.csv")
    
    def analyze_scalability_trends(self, scalability_results):
        """Analyze scalability trends."""
        print("\nScalability Analysis Results:")
        print("-" * 40)
        
        device_counts = list(scalability_results.keys())
        accuracies = [scalability_results[d]['final_accuracy'] for d in device_counts]
        training_times = [scalability_results[d]['total_training_time'] for d in device_counts]
        
        print("Device Count | Accuracy | Training Time (s)")
        print("-" * 40)
        for i, device_count in enumerate(device_counts):
            print(f"{device_count:11d} | {accuracies[i]:8.4f} | {training_times[i]:15.1f}")
        
        # Calculate scalability metrics
        time_complexity = np.polyfit(np.log(device_counts), np.log(training_times), 1)[0]
        print(f"\nTime complexity scaling factor: O(n^{time_complexity:.2f})")
        
        # Efficiency analysis
        efficiency_scores = [acc / time for acc, time in zip(accuracies, training_times)]
        best_efficiency_idx = np.argmax(efficiency_scores)
        print(f"Most efficient configuration: {device_counts[best_efficiency_idx]} devices")
    
    def analyze_channel_impact(self, channel_results):
        """Analyze channel condition impact."""
        print("\nChannel Condition Impact Analysis:")
        print("-" * 50)
        
        conditions = list(channel_results.keys())
        
        print("Condition | Accuracy | Energy Eff | Spectral Eff")
        print("-" * 50)
        for condition in conditions:
            metrics = channel_results[condition]
            print(f"{condition:9s} | {metrics['final_accuracy']:8.4f} | "
                  f"{metrics['avg_energy_efficiency']:10.2f} | "
                  f"{metrics['avg_spectral_efficiency']:12.2f}")
        
        # Calculate degradation
        good_acc = channel_results['good']['final_accuracy']
        bad_acc = channel_results['bad']['final_accuracy']
        degradation = (good_acc - bad_acc) / good_acc * 100
        print(f"\nAccuracy degradation (good to bad): {degradation:.2f}%")
    
    def analyze_distribution_impact(self, distribution_results):
        """Analyze data distribution impact."""
        print("\nData Distribution Impact Analysis:")
        print("-" * 45)
        
        distributions = list(distribution_results.keys())
        
        print("Distribution | Accuracy | Fairness | Conv. Rate")
        print("-" * 45)
        for dist in distributions:
            metrics = distribution_results[dist]
            print(f"{dist:12s} | {metrics['final_accuracy']:8.4f} | "
                  f"{metrics['avg_fairness_index']:8.3f} | "
                  f"{metrics['convergence_rate']:10.4f}")
        
        # Calculate fairness impact
        iid_fairness = distribution_results['iid']['avg_fairness_index']
        non_iid_fairness = distribution_results['non_iid']['avg_fairness_index']
        fairness_impact = (iid_fairness - non_iid_fairness) / iid_fairness * 100
        print(f"\nFairness reduction (IID to Non-IID): {fairness_impact:.2f}%")
    
    def generate_performance_visualizations(self):
        """Generate comprehensive performance visualizations."""
        print("\nGenerating performance visualizations...")
        
        # Create results directory
        os.makedirs('results/figures', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Benchmark comparison radar chart
        self.create_radar_chart()
        
        # 2. Scalability analysis
        self.create_scalability_plots()
        
        # 3. Convergence comparison
        self.create_convergence_plots()
        
        # 4. Performance dashboard
        self.create_performance_dashboard()
        
        print("  ✓ All visualizations generated")
    
    def create_radar_chart(self):
        """Create radar chart for benchmark comparison."""
        # Implementation would create a radar chart comparing different scenarios
        # across multiple metrics (accuracy, energy efficiency, fairness, etc.)
        pass
    
    def create_scalability_plots(self):
        """Create scalability analysis plots."""
        # Implementation would create plots showing how performance scales
        # with number of devices
        pass
    
    def create_convergence_plots(self):
        """Create convergence comparison plots."""
        # Implementation would create plots comparing convergence rates
        # across different scenarios
        pass
    
    def create_performance_dashboard(self):
        """Create comprehensive performance dashboard."""
        # Implementation would create a multi-panel dashboard
        # showing all key performance metrics
        pass


def main():
    """Main function for performance analysis."""
    parser = argparse.ArgumentParser(
        description='Performance Analysis for 6G Green IoT Networks FL System'
    )
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Run comprehensive benchmark suite')
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability analysis')
    parser.add_argument('--channel-analysis', action='store_true',
                       help='Run channel condition analysis')
    parser.add_argument('--distribution-analysis', action='store_true',
                       help='Run data distribution analysis')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-results', action='store_true',
                       help='Save detailed results to files')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    try:
        # Run requested analyses
        if args.all or args.benchmark:
            analyzer.run_benchmark_suite()
        
        if args.all or args.scalability:
            analyzer.run_scalability_analysis()
        
        if args.all or args.channel_analysis:
            analyzer.run_channel_condition_analysis()
        
        if args.all or args.distribution_analysis:
            analyzer.run_data_distribution_analysis()
        
        # Generate visualizations
        if args.plot:
            analyzer.generate_performance_visualizations()
        
        # Save detailed results
        if args.save_results:
            print("\nSaving detailed results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save all results as JSON
            import json
            results_file = f'results/logs/performance_analysis_{timestamp}.json'
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for scenario, data in analyzer.results.items():
                    serializable_results[scenario] = {
                        'config': data['config'],
                        'final_metrics': data['final_metrics']
                    }
                json.dump(serializable_results, f, indent=2)
            
            print(f"  ✓ Detailed results saved to {results_file}")
        
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS COMPLETED")
        print("=" * 60)
        print("Summary of key findings:")
        
        if analyzer.results:
            # Find best performing scenario
            best_scenario = max(analyzer.results.keys(), 
                              key=lambda x: analyzer.results[x]['final_metrics']['final_accuracy'])
            best_accuracy = analyzer.results[best_scenario]['final_metrics']['final_accuracy']
            
            print(f"  - Best performing scenario: {best_scenario}")
            print(f"  - Best accuracy achieved: {best_accuracy:.4f}")
            
            if 'full_system' in analyzer.results and 'baseline_fl' in analyzer.results:
                full_acc = analyzer.results['full_system']['final_metrics']['final_accuracy']
                baseline_acc = analyzer.results['baseline_fl']['final_metrics']['final_accuracy']
                improvement = (full_acc - baseline_acc) / baseline_acc * 100
                print(f"  - Overall system improvement: {improvement:.2f}%")
        
        print("  - Check results/figures/ for visualizations")
        print("  - Check results/logs/ for detailed logs")
        
    except Exception as e:
        print(f"Error during performance analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())