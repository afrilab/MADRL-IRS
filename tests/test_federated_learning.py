#!/usr/bin/env python3
"""
Unit tests for federated learning implementation.

Tests the core functionality of the federated learning system including
data distribution, local training, and global aggregation.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.federated_learning import FederatedLearning, LocalDevice, GlobalAggregator
from core.federated_learning import FederatedLearningError, DataDistributionError


class TestLocalDevice(unittest.TestCase):
    """Test LocalDevice class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = LocalDevice(device_id=0, local_epochs=2, batch_size=16)
        
    def test_device_initialization(self):
        """Test device initialization."""
        self.assertEqual(self.device.device_id, 0)
        self.assertEqual(self.device.local_epochs, 2)
        self.assertEqual(self.device.batch_size, 16)
        self.assertIsNone(self.device.local_data)
        self.assertEqual(self.device.num_samples, 0)
        
    def test_set_data(self):
        """Test setting local data."""
        x_data = np.random.rand(100, 10)
        y_data = np.random.rand(100, 1)
        
        self.device.set_data(x_data, y_data)
        
        self.assertEqual(self.device.num_samples, 100)
        np.testing.assert_array_equal(self.device.local_data['x'], x_data)
        np.testing.assert_array_equal(self.device.local_data['y'], y_data)


class TestGlobalAggregator(unittest.TestCase):
    """Test GlobalAggregator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = GlobalAggregator()
        
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        self.assertIsNone(self.aggregator.global_model)
        self.assertEqual(self.aggregator.round_number, 0)
        
    def test_federated_averaging(self):
        """Test federated averaging algorithm."""
        # Create mock weights
        weights1 = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0])]
        weights2 = [np.array([[2.0, 3.0], [4.0, 5.0]]), np.array([2.0, 3.0])]
        weights3 = [np.array([[3.0, 4.0], [5.0, 6.0]]), np.array([3.0, 4.0])]
        
        local_weights_list = [weights1, weights2, weights3]
        sample_counts = [100, 200, 100]  # Total: 400
        
        # Expected weighted average
        # Device 1: 100/400 = 0.25, Device 2: 200/400 = 0.5, Device 3: 100/400 = 0.25
        expected_layer1 = 0.25 * weights1[0] + 0.5 * weights2[0] + 0.25 * weights3[0]
        expected_layer2 = 0.25 * weights1[1] + 0.5 * weights2[1] + 0.25 * weights3[1]
        
        aggregated = self.aggregator.federated_averaging(local_weights_list, sample_counts)
        
        np.testing.assert_array_almost_equal(aggregated[0], expected_layer1)
        np.testing.assert_array_almost_equal(aggregated[1], expected_layer2)
        self.assertEqual(self.aggregator.round_number, 1)
        
    def test_empty_weights_error(self):
        """Test error handling for empty weights."""
        with self.assertRaises(ValueError):
            self.aggregator.federated_averaging([], [])


class TestFederatedLearning(unittest.TestCase):
    """Test FederatedLearning class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary CSV file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create sample CSV data
        np.random.seed(42)
        data = {
            'feature1': np.random.rand(200),
            'feature2': np.random.rand(200),
            'feature3': np.random.randint(0, 3, 200),
            'target': np.random.randint(0, 2, 200)
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        os.rmdir(self.temp_dir)
        
    def test_csv_dataset_loading(self):
        """Test CSV dataset loading and preprocessing."""
        fl = FederatedLearning(
            num_devices=4,
            dataset='csv',
            csv_path=self.csv_path,
            target_column='target',
            local_epochs=1
        )
        
        self.assertEqual(fl.dataset, 'csv')
        self.assertEqual(fl.problem_type, 'classification')
        self.assertEqual(fl.num_classes, 2)
        self.assertIsNotNone(fl.x_train)
        self.assertIsNotNone(fl.y_train)
        self.assertEqual(len(fl.devices), 4)
        
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_mnist_dataset_loading(self, mock_load_data):
        """Test MNIST dataset loading."""
        # Mock MNIST data
        x_train = np.random.rand(1000, 28, 28)
        y_train = np.random.randint(0, 10, 1000)
        x_test = np.random.rand(200, 28, 28)
        y_test = np.random.randint(0, 10, 200)
        
        mock_load_data.return_value = ((x_train, y_train), (x_test, y_test))
        
        fl = FederatedLearning(
            num_devices=5,
            dataset='mnist',
            local_epochs=1
        )
        
        self.assertEqual(fl.dataset, 'mnist')
        self.assertEqual(fl.problem_type, 'classification')
        self.assertEqual(fl.num_classes, 10)
        self.assertEqual(fl.input_shape, (28, 28, 1))
        
    def test_iid_data_distribution(self):
        """Test IID data distribution."""
        fl = FederatedLearning(
            num_devices=4,
            dataset='csv',
            csv_path=self.csv_path,
            data_distribution='iid',
            local_epochs=1
        )
        
        # Check that all devices have data
        total_samples = 0
        for device in fl.devices:
            self.assertGreater(device.num_samples, 0)
            total_samples += device.num_samples
            
        # Total samples should match training set size
        self.assertEqual(total_samples, len(fl.x_train))
        
    def test_non_iid_data_distribution(self):
        """Test Non-IID data distribution."""
        fl = FederatedLearning(
            num_devices=4,
            dataset='csv',
            csv_path=self.csv_path,
            data_distribution='non_iid',
            local_epochs=1
        )
        
        # Check that all devices have data
        total_samples = 0
        for device in fl.devices:
            self.assertGreater(device.num_samples, 0)
            total_samples += device.num_samples
            
        # Total samples should match training set size
        self.assertEqual(total_samples, len(fl.x_train))
        
    def test_model_creation(self):
        """Test model creation for different datasets."""
        fl = FederatedLearning(
            num_devices=2,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        
        self.assertIsNotNone(fl.global_model)
        self.assertGreater(fl.global_model.count_params(), 0)
        self.assertGreater(fl.model_size, 0)
        
    def test_single_fl_round(self):
        """Test running a single FL round."""
        fl = FederatedLearning(
            num_devices=2,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        
        # Run one round
        metrics = fl.run_round()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertGreater(fl.communication_cost, 0)
        
    def test_training_multiple_rounds(self):
        """Test training for multiple rounds."""
        fl = FederatedLearning(
            num_devices=2,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        
        # Train for 3 rounds
        history = fl.train(num_rounds=3, verbose=False)
        
        self.assertIsInstance(history, dict)
        self.assertEqual(len(history['loss']), 3)
        self.assertEqual(len(history['communication_cost']), 3)
        
    def test_invalid_dataset_error(self):
        """Test error handling for invalid dataset."""
        with self.assertRaises(ValueError):
            FederatedLearning(
                num_devices=2,
                dataset='invalid_dataset',
                local_epochs=1
            )
            
    def test_missing_csv_path_error(self):
        """Test error handling for missing CSV path."""
        with self.assertRaises(ValueError):
            FederatedLearning(
                num_devices=2,
                dataset='csv',
                local_epochs=1
            )
            
    def test_nonexistent_csv_file_error(self):
        """Test error handling for nonexistent CSV file."""
        with self.assertRaises(FileNotFoundError):
            FederatedLearning(
                num_devices=2,
                dataset='csv',
                csv_path='/nonexistent/path/file.csv',
                local_epochs=1
            )
            
    def test_get_device_data_distribution(self):
        """Test getting device data distribution."""
        fl = FederatedLearning(
            num_devices=3,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        
        distribution = fl.get_device_data_distribution()
        
        self.assertEqual(len(distribution), 3)
        self.assertTrue(all(count > 0 for count in distribution.values()))
        
    def test_model_save_load(self):
        """Test model saving and loading."""
        fl = FederatedLearning(
            num_devices=2,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        
        # Train for one round to get some weights
        fl.run_round()
        original_weights = fl.get_model_weights()
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.h5')
        fl.save_model(model_path)
        
        # Create new FL instance and load model
        fl2 = FederatedLearning(
            num_devices=2,
            dataset='csv',
            csv_path=self.csv_path,
            local_epochs=1
        )
        fl2.load_model(model_path)
        
        loaded_weights = fl2.get_model_weights()
        
        # Compare weights
        for orig, loaded in zip(original_weights, loaded_weights):
            np.testing.assert_array_almost_equal(orig, loaded)
            
        # Clean up
        os.remove(model_path)


if __name__ == '__main__':
    unittest.main()