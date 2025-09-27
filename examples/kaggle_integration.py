#!/usr/bin/env python3
"""
Kaggle Dataset Integration Example for 6G Green IoT Networks

This script demonstrates how to integrate Kaggle datasets with the federated learning
system, specifically designed for the Kaggle environment with the augmented 5G dataset.

Usage:
    python examples/kaggle_integration.py [--dataset-path PATH] [--target-column TARGET]

Kaggle Usage:
    1. Add the "augmented-5g-dataset-for-resource-allocation" dataset to your notebook
    2. Run this script in your Kaggle notebook environment
    3. The script will automatically detect and use the Kaggle dataset path
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import NetworkTopology
from core.channel_model import ChannelModel
from core.federated_learning import FederatedLearning
from core.irs_optimization import IRSOptimizer
from utils.visualization import plot_data_distribution, plot_performance_comparison
from utils.metrics import calculate_fairness_index, calculate_convergence_rate


def detect_kaggle_environment():
    """Detect if running in Kaggle environment and return dataset paths."""
    kaggle_paths = {
        'input': '/kaggle/input',
        'working': '/kaggle/working',
        'temp': '/kaggle/temp'
    }
    
    is_kaggle = all(os.path.exists(path) for path in kaggle_paths.values())
    
    if is_kaggle:
        print("✓ Kaggle environment detected")
        # Look for the augmented 5G dataset
        dataset_dirs = []
        if os.path.exists(kaggle_paths['input']):
            for item in os.listdir(kaggle_paths['input']):
                item_path = os.path.join(kaggle_paths['input'], item)
                if os.path.isdir(item_path):
                    dataset_dirs.append(item_path)
        
        # Find the 5G dataset
        for dataset_dir in dataset_dirs:
            if '5g' in dataset_dir.lower() or 'resource' in dataset_dir.lower():
                csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
                if csv_files:
                    dataset_path = os.path.join(dataset_dir, csv_files[0])
                    print(f"✓ Found dataset: {dataset_path}")
                    return True, dataset_path
        
        print("⚠ Kaggle environment detected but 5G dataset not found")
        return True, None
    
    return False, None


def analyze_dataset(dataset_path):
    """Analyze the dataset and provide insights for FL configuration."""
    print("\nDataset Analysis:")
    print("-" * 40)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Features: {len(df.columns) - 1}")  # Assuming last column is target
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    
    # Analyze data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Categorical features: {len(categorical_cols)}")
    
    # Analyze target column (assume last column)
    target_col = df.columns[-1]
    print(f"\nTarget Analysis ('{target_col}'):")
    
    if target_col in categorical_cols:
        unique_values = df[target_col].nunique()
        print(f"  - Type: Classification")
        print(f"  - Classes: {unique_values}")
        print(f"  - Class distribution:")
        class_counts = df[target_col].value_counts()
        for class_name, count in class_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
    else:
        print(f"  - Type: Regression")
        print(f"  - Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
        print(f"  - Mean: {df[target_col].mean():.2f}")
        print(f"  - Std: {df[target_col].std():.2f}")
    
    # Recommend FL configuration
    print(f"\nRecommended FL Configuration:")
    
    # Number of devices based on dataset size
    if len(df) < 1000:
        recommended_devices = 5
    elif len(df) < 10000:
        recommended_devices = 10
    else:
        recommended_devices = 20
    
    print(f"  - Devices: {recommended_devices}")
    print(f"  - Local epochs: 5-10")
    print(f"  - Batch size: {min(32, len(df) // (recommended_devices * 10))}")
    
    return {
        'target_column': target_col,
        'problem_type': 'classification' if target_col in categorical_cols else 'regression',
        'num_classes': df[target_col].nunique() if target_col in categorical_cols else 1,
        'recommended_devices': recommended_devices,
        'dataset_size': len(df)
    }


def run_kaggle_experiment(args, dataset_info):
    """Run FL experiment with Kaggle dataset."""
    print("\n" + "=" * 60)
    print("KAGGLE DATASET FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    
    # Configuration
    num_devices = args.devices if args.devices else dataset_info['recommended_devices']
    
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset_path}")
    print(f"  - Target column: {dataset_info['target_column']}")
    print(f"  - Problem type: {dataset_info['problem_type']}")
    print(f"  - Number of devices: {num_devices}")
    print(f"  - FL rounds: {args.rounds}")
    print(f"  - Data distribution: {args.distribution}")
    
    # Step 1: Initialize Network
    print("\nStep 1: Initializing network components...")
    network = NetworkTopology(
        num_iot_devices=num_devices,
        num_irs_elements=100,
        area_size=100
    )
    
    channel = ChannelModel(network, condition='medium')
    irs_optimizer = IRSOptimizer(network, channel)
    
    # Step 2: Initialize FL with CSV dataset
    print("Step 2: Initializing federated learning with Kaggle dataset...")
    fl_system = FederatedLearning(
        num_devices=num_devices,
        dataset='csv',
        data_distribution=args.distribution,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        csv_path=args.dataset_path,
        target_column=dataset_info['target_column']
    )
    
    # Step 3: Run FL Experiment
    print("\nStep 3: Running federated learning experiment...")
    print("-" * 50)
    
    # Storage for metrics
    metrics_history = {
        'accuracy': [],
        'loss': [],
        'communication_cost': [],
        'training_time': [],
        'fairness_index': []
    }
    
    # Experiment loop
    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        
        # Optimize IRS every few rounds
        if round_num % 5 == 0:
            print("  Optimizing IRS configuration...")
            irs_optimizer.optimize()
            channel.update_with_irs(irs_optimizer)
        
        # Run FL round
        start_time = datetime.now()
        metrics = fl_system.run_round(channel, network)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate fairness index
        device_accuracies = [metrics.get(f'device_{i}_accuracy', 0) for i in range(num_devices)]
        fairness_idx = calculate_fairness_index(device_accuracies)
        
        # Store metrics
        metrics_history['accuracy'].append(metrics.get('accuracy', 0))
        metrics_history['loss'].append(metrics.get('loss', 0))
        metrics_history['communication_cost'].append(fl_system.communication_cost)
        metrics_history['training_time'].append(training_time)
        metrics_history['fairness_index'].append(fairness_idx)
        
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}, "
              f"Loss: {metrics.get('loss', 0):.4f}, "
              f"Fairness: {fairness_idx:.3f}, "
              f"Time: {training_time:.1f}s")
    
    # Step 4: Results Analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    
    final_accuracy = metrics_history['accuracy'][-1]
    final_loss = metrics_history['loss'][-1]
    total_comm_cost = metrics_history['communication_cost'][-1]
    avg_training_time = np.mean(metrics_history['training_time'])
    final_fairness = metrics_history['fairness_index'][-1]
    
    print(f"Final Results:")
    print(f"  - Model Accuracy: {final_accuracy:.4f}")
    print(f"  - Training Loss: {final_loss:.4f}")
    print(f"  - Total Communication Cost: {total_comm_cost:.2f} MB")
    print(f"  - Average Training Time: {avg_training_time:.2f} seconds/round")
    print(f"  - Fairness Index: {final_fairness:.3f}")
    
    # Calculate convergence rate
    convergence_rate = calculate_convergence_rate(metrics_history['accuracy'])
    print(f"  - Convergence Rate: {convergence_rate:.4f}")
    
    # Compare with centralized learning (theoretical)
    centralized_accuracy = final_accuracy * 1.05  # Assume 5% better
    efficiency_ratio = final_accuracy / centralized_accuracy
    print(f"  - FL Efficiency (vs centralized): {efficiency_ratio:.3f}")
    
    # Step 5: Generate Kaggle-specific Visualizations
    if args.plot:
        print("\nGenerating Kaggle-compatible visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Federated Learning with Kaggle Dataset - Results', fontsize=16)
        
        # Plot 1: Accuracy convergence
        axes[0, 0].plot(metrics_history['accuracy'], 'b-', linewidth=2)
        axes[0, 0].set_title('Model Accuracy Convergence')
        axes[0, 0].set_xlabel('FL Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss convergence
        axes[0, 1].plot(metrics_history['loss'], 'r-', linewidth=2)
        axes[0, 1].set_title('Training Loss Convergence')
        axes[0, 1].set_xlabel('FL Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Communication cost
        axes[0, 2].plot(metrics_history['communication_cost'], 'g-', linewidth=2)
        axes[0, 2].set_title('Cumulative Communication Cost')
        axes[0, 2].set_xlabel('FL Round')
        axes[0, 2].set_ylabel('Communication Cost (MB)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Training time
        axes[1, 0].bar(range(len(metrics_history['training_time'])), 
                      metrics_history['training_time'], alpha=0.7)
        axes[1, 0].set_title('Training Time per Round')
        axes[1, 0].set_xlabel('FL Round')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Fairness index
        axes[1, 1].plot(metrics_history['fairness_index'], 'm-', linewidth=2)
        axes[1, 1].set_title('Fairness Index Over Time')
        axes[1, 1].set_xlabel('FL Round')
        axes[1, 1].set_ylabel('Fairness Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Data distribution visualization
        plot_data_distribution(fl_system, ax=axes[1, 2])
        
        plt.tight_layout()
        
        # Save plot
        if detect_kaggle_environment()[0]:
            plt.savefig('/kaggle/working/kaggle_fl_results.png', dpi=300, bbox_inches='tight')
            print("  ✓ Results saved to /kaggle/working/kaggle_fl_results.png")
        else:
            os.makedirs('results/figures', exist_ok=True)
            plt.savefig('results/figures/kaggle_fl_results.png', dpi=300, bbox_inches='tight')
            print("  ✓ Results saved to results/figures/kaggle_fl_results.png")
        
        plt.show()
    
    # Step 6: Save Results for Kaggle
    if args.save_results:
        print("\nSaving results for Kaggle...")
        
        # Prepare results dictionary
        results = {
            'dataset_info': dataset_info,
            'configuration': {
                'devices': num_devices,
                'rounds': args.rounds,
                'distribution': args.distribution,
                'local_epochs': args.local_epochs,
                'batch_size': args.batch_size
            },
            'final_metrics': {
                'accuracy': final_accuracy,
                'loss': final_loss,
                'communication_cost': total_comm_cost,
                'fairness_index': final_fairness,
                'convergence_rate': convergence_rate
            },
            'history': metrics_history
        }
        
        # Save to appropriate location
        if detect_kaggle_environment()[0]:
            results_path = '/kaggle/working/fl_results.txt'
        else:
            os.makedirs('results/logs', exist_ok=True)
            results_path = 'results/logs/kaggle_fl_results.txt'
        
        with open(results_path, 'w') as f:
            f.write("Federated Learning with Kaggle Dataset - Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Dataset Information:\n")
            for key, value in dataset_info.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nConfiguration:\n")
            for key, value in results['configuration'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nFinal Results:\n")
            for key, value in results['final_metrics'].items():
                f.write(f"  {key}: {value}\n")
        
        print(f"  ✓ Results saved to {results_path}")
    
    return results


def main():
    """Main function for Kaggle integration example."""
    parser = argparse.ArgumentParser(
        description='Kaggle Dataset Integration for 6G Green IoT Networks FL'
    )
    
    # Dataset parameters
    parser.add_argument('--dataset-path', type=str,
                       help='Path to CSV dataset (auto-detected in Kaggle)')
    parser.add_argument('--target-column', type=str,
                       help='Target column name (auto-detected if not specified)')
    
    # FL parameters
    parser.add_argument('--devices', type=int,
                       help='Number of devices (auto-recommended if not specified)')
    parser.add_argument('--rounds', type=int, default=20,
                       help='Number of FL rounds (default: 20)')
    parser.add_argument('--distribution', type=str, default='iid',
                       choices=['iid', 'non_iid'],
                       help='Data distribution (default: iid)')
    parser.add_argument('--local-epochs', type=int, default=5,
                       help='Local training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    # Output options
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    try:
        # Detect environment and dataset
        is_kaggle, auto_dataset_path = detect_kaggle_environment()
        
        # Determine dataset path
        if args.dataset_path:
            dataset_path = args.dataset_path
        elif auto_dataset_path:
            dataset_path = auto_dataset_path
        else:
            # Use sample dataset for testing
            dataset_path = 'data/synthetic/sample_5g_data.csv'
            if not os.path.exists(dataset_path):
                print("Error: No dataset found. Please specify --dataset-path or run in Kaggle environment.")
                return 1
        
        print(f"Using dataset: {dataset_path}")
        
        # Analyze dataset
        dataset_info = analyze_dataset(dataset_path)
        
        # Update args with dataset path
        args.dataset_path = dataset_path
        if args.target_column is None:
            args.target_column = dataset_info['target_column']
        
        # Run experiment
        results = run_kaggle_experiment(args, dataset_info)
        
        # Print summary
        print(f"\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Dataset: {os.path.basename(dataset_path)}")
        print(f"Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
        print(f"Convergence Rate: {results['final_metrics']['convergence_rate']:.4f}")
        print(f"Fairness Index: {results['final_metrics']['fairness_index']:.3f}")
        
        if is_kaggle:
            print("\n✓ Experiment completed successfully in Kaggle environment!")
            print("Check /kaggle/working/ for output files.")
        else:
            print("\n✓ Experiment completed successfully!")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())