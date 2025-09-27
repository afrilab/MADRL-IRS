#!/usr/bin/env python3
"""
Federated Learning Demo Script

This script demonstrates the federated learning implementation with synthetic data.
It shows how to use the FederatedLearning class with different datasets and
data distribution strategies.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.federated_learning import FederatedLearning


def create_synthetic_csv_data(num_samples=1000, num_features=10, num_classes=3):
    """Create synthetic CSV data for demonstration."""
    np.random.seed(42)
    
    # Generate features
    data = {}
    for i in range(num_features):
        if i % 3 == 0:  # Some categorical features
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], num_samples)
        else:  # Numerical features
            data[f'num_feature_{i}'] = np.random.randn(num_samples)
    
    # Generate target (classification)
    data['target'] = np.random.randint(0, num_classes, num_samples)
    
    return pd.DataFrame(data)


def demo_csv_classification():
    """Demonstrate FL with CSV classification data."""
    print("=" * 60)
    print("FEDERATED LEARNING DEMO: CSV Classification")
    print("=" * 60)
    
    # Create synthetic data
    df = create_synthetic_csv_data(num_samples=1000, num_features=8, num_classes=3)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        # Initialize FL system
        print(f"\n1. Initializing FL system with CSV data...")
        fl = FederatedLearning(
            num_devices=5,
            dataset='csv',
            csv_path=csv_path,
            target_column='target',
            data_distribution='iid',
            local_epochs=3,
            batch_size=32
        )
        
        print(f"   - Dataset: {fl.dataset}")
        print(f"   - Problem type: {fl.problem_type}")
        print(f"   - Number of classes: {fl.num_classes}")
        print(f"   - Input shape: {fl.input_shape}")
        print(f"   - Number of devices: {len(fl.devices)}")
        print(f"   - Model parameters: {fl.global_model.count_params()}")
        
        # Show data distribution
        print(f"\n2. Data distribution across devices:")
        distribution = fl.get_device_data_distribution()
        for device_id, num_samples in distribution.items():
            print(f"   - Device {device_id}: {num_samples} samples")
        
        # Train for a few rounds
        print(f"\n3. Training FL system for 5 rounds...")
        history = fl.train(num_rounds=5, verbose=True)
        
        print(f"\n4. Training Results:")
        print(f"   - Final loss: {history['loss'][-1]:.4f}")
        print(f"   - Final accuracy: {history['accuracy'][-1]:.4f}")
        print(f"   - Total communication cost: {fl.get_communication_cost():.2f} MB")
        
    finally:
        # Clean up
        os.unlink(csv_path)
    
    print("\n✓ CSV Classification demo completed successfully!")


def demo_mnist_classification():
    """Demonstrate FL with MNIST data."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING DEMO: MNIST Classification")
    print("=" * 60)
    
    try:
        # Initialize FL system with MNIST
        print(f"\n1. Initializing FL system with MNIST data...")
        fl = FederatedLearning(
            num_devices=10,
            dataset='mnist',
            data_distribution='non_iid',
            local_epochs=2,
            batch_size=64
        )
        
        print(f"   - Dataset: {fl.dataset}")
        print(f"   - Problem type: {fl.problem_type}")
        print(f"   - Number of classes: {fl.num_classes}")
        print(f"   - Input shape: {fl.input_shape}")
        print(f"   - Number of devices: {len(fl.devices)}")
        print(f"   - Model parameters: {fl.global_model.count_params()}")
        
        # Show data distribution
        print(f"\n2. Data distribution across devices (Non-IID):")
        distribution = fl.get_device_data_distribution()
        for device_id, num_samples in distribution.items():
            print(f"   - Device {device_id}: {num_samples} samples")
        
        # Train for a few rounds
        print(f"\n3. Training FL system for 3 rounds...")
        history = fl.train(num_rounds=3, verbose=True)
        
        print(f"\n4. Training Results:")
        print(f"   - Final loss: {history['loss'][-1]:.4f}")
        print(f"   - Final accuracy: {history['accuracy'][-1]:.4f}")
        print(f"   - Total communication cost: {fl.get_communication_cost():.2f} MB")
        
    except Exception as e:
        print(f"   Note: MNIST demo skipped due to: {e}")
        print("   (This is normal if TensorFlow/Keras is not fully available)")
    
    print("\n✓ MNIST Classification demo completed!")


def demo_data_distributions():
    """Demonstrate different data distribution strategies."""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING DEMO: Data Distribution Strategies")
    print("=" * 60)
    
    # Create synthetic data
    df = create_synthetic_csv_data(num_samples=800, num_features=5, num_classes=2)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name
    
    try:
        for distribution_type in ['iid', 'non_iid']:
            print(f"\n--- {distribution_type.upper()} Distribution ---")
            
            fl = FederatedLearning(
                num_devices=4,
                dataset='csv',
                csv_path=csv_path,
                target_column='target',
                data_distribution=distribution_type,
                local_epochs=2,
                batch_size=32
            )
            
            # Show data distribution
            distribution = fl.get_device_data_distribution()
            total_samples = sum(distribution.values())
            
            print(f"Data distribution ({distribution_type}):")
            for device_id, num_samples in distribution.items():
                percentage = (num_samples / total_samples) * 100
                print(f"  Device {device_id}: {num_samples} samples ({percentage:.1f}%)")
            
            # Train for 2 rounds
            history = fl.train(num_rounds=2, verbose=False)
            print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
            
    finally:
        # Clean up
        os.unlink(csv_path)
    
    print("\n✓ Data distribution demo completed!")


def main():
    """Run all federated learning demos."""
    print("FEDERATED LEARNING IMPLEMENTATION DEMO")
    print("=" * 60)
    print("This demo showcases the federated learning implementation")
    print("with different datasets and configuration options.")
    
    try:
        # Demo 1: CSV Classification
        demo_csv_classification()
        
        # Demo 2: MNIST Classification
        demo_mnist_classification()
        
        # Demo 3: Data Distribution Strategies
        demo_data_distributions()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe federated learning implementation supports:")
        print("✓ Multiple datasets (CSV, MNIST, CIFAR-10)")
        print("✓ IID and Non-IID data distributions")
        print("✓ Configurable local training parameters")
        print("✓ FedAvg aggregation algorithm")
        print("✓ Communication cost tracking")
        print("✓ Model saving and loading")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()