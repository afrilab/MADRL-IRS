#!/usr/bin/env python3
"""
Federated Learning Implementation for 6G Green IoT Networks

This module implements the federated learning algorithm with support for multiple datasets
and data distribution strategies as described in the research paper.

Author: Research Team
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import logging

# Import custom exceptions and validation
from .exceptions import FederatedLearningError, DatasetError, ValidationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_string_choice,
    validate_file_path, validate_dataframe, validate_fl_config
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


class LocalDevice:
    """
    Local device implementation for federated learning.

    Represents an individual IoT device participating in federated learning
    with local training capabilities.
    """

    def __init__(self, device_id: int, local_epochs: int = 5, batch_size: int = 32):
        """
        Initialize local device.

        Args:
            device_id: Unique device identifier
            local_epochs: Number of local training epochs
            batch_size: Mini-batch size for local training

        Raises:
            FederatedLearningError: If parameters are invalid
        """
        try:
            # Validate parameters
            self.device_id = validate_integer_range(
                device_id, 'device_id', min_value=0
            )
            self.local_epochs = validate_integer_range(
                local_epochs, 'local_epochs', min_value=1, max_value=100
            )
            self.batch_size = validate_integer_range(
                batch_size, 'batch_size', min_value=1, max_value=1024
            )

            self.local_data = None
            self.local_model = None
            self.num_samples = 0

            logger.debug(f"LocalDevice {self.device_id} initialized with "
                        f"{self.local_epochs} epochs, batch size {self.batch_size}")

        except ValidationError as e:
            logger.error(f"LocalDevice initialization failed: {e}")
            raise FederatedLearningError(
                f"Invalid local device parameters: {e.message}",
                device_id=device_id
            ) from e

    def set_data(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Set local training data for the device.

        Args:
            x_data: Input features
            y_data: Target labels
        """
        self.local_data = {'x': x_data, 'y': y_data}
        self.num_samples = len(x_data)
        logger.info(f"Device {self.device_id}: Set {self.num_samples} samples")

    def train_local_model(self, global_weights: List[np.ndarray],
                         model_architecture: tf.keras.Model) -> Tuple[List[np.ndarray], int]:
        """
        Perform local training on device data.

        Implements local SGD training as described in the paper.

        Args:
            global_weights: Current global model weights
            model_architecture: Model architecture to clone

        Returns:
            Tuple of (updated_weights, num_samples)

        Raises:
            FederatedLearningError: If training fails
        """
        try:
            with PerformanceLogger(logger, f"Local training device {self.device_id}"):
                # Validate inputs
                if self.local_data is None:
                    raise FederatedLearningError(
                        f"Device {self.device_id}: No local data set",
                        device_id=self.device_id
                    )

                if not global_weights:
                    raise FederatedLearningError(
                        f"Device {self.device_id}: No global weights provided",
                        device_id=self.device_id
                    )

                logger.debug(f"Device {self.device_id}: Starting local training with "
                           f"{self.num_samples} samples")

                # Create local model copy
                self.local_model = tf.keras.models.clone_model(model_architecture)
                self.local_model.compile(
                    optimizer='adam',
                    loss=model_architecture.loss,
                    metrics=model_architecture.metrics
                )
                self.local_model.set_weights(global_weights)

                # Local training
                history = self.local_model.fit(
                    self.local_data['x'],
                    self.local_data['y'],
                    epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    verbose=0
                )

                final_loss = history.history['loss'][-1]
                logger.debug(f"Device {self.device_id}: Local training completed, "
                           f"final loss: {final_loss:.4f}")

                # Validate training results
                if np.isnan(final_loss) or np.isinf(final_loss):
                    raise FederatedLearningError(
                        f"Device {self.device_id}: Training resulted in invalid loss",
                        device_id=self.device_id
                    )

                return self.local_model.get_weights(), self.num_samples

        except FederatedLearningError:
            raise
        except Exception as e:
            logger.error(f"Device {self.device_id}: Local training failed: {e}")
            log_exception(logger, e, f"Local training device {self.device_id}")
            raise FederatedLearningError(
                f"Device {self.device_id}: Local training failed: {str(e)}",
                device_id=self.device_id
            ) from e


class GlobalAggregator:
    """
    Global aggregator for federated learning.

    Implements FedAvg algorithm for aggregating local model updates.
    """

    def __init__(self):
        """Initialize global aggregator."""
        self.global_model = None
        self.round_number = 0

    def set_global_model(self, model: tf.keras.Model) -> None:
        """
        Set the global model architecture.

        Args:
            model: Global model architecture
        """
        self.global_model = model

    def federated_averaging(self, local_weights_list: List[List[np.ndarray]],
                          sample_counts: List[int]) -> List[np.ndarray]:
        """
        Perform federated averaging of local model weights.

        Implements the FedAvg algorithm: w_t+1 = sum(n_k/n * w_k)

        Args:
            local_weights_list: List of local model weights from devices
            sample_counts: Number of samples per device

        Returns:
            Aggregated global weights
        """
        if not local_weights_list:
            raise ValueError("No local weights provided for aggregation")

        total_samples = sum(sample_counts)

        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in local_weights_list[0]]

        # Weighted aggregation
        for device_weights, num_samples in zip(local_weights_list, sample_counts):
            weight_factor = num_samples / total_samples

            for i, layer_weights in enumerate(device_weights):
                aggregated_weights[i] += weight_factor * layer_weights

        self.round_number += 1
        logger.info(f"Round {self.round_number}: Aggregated weights from "
                   f"{len(local_weights_list)} devices ({total_samples} total samples)")

        return aggregated_weights


class FederatedLearning:
    """
    Main Federated Learning implementation with multi-dataset support.

    Implements the FL algorithm with local training and global aggregation
    as described in the paper, with support for multiple datasets and
    data distribution strategies.
    """

    def __init__(self, num_devices: int = 20, dataset: str = 'csv',
                 data_distribution: str = 'iid', local_epochs: int = 5,
                 batch_size: int = 32, csv_path: Optional[str] = None,
                 target_column: Optional[str] = None):
        """
        Initialize FL system.

        Args:
            num_devices: Number of participating devices (K)
            dataset: Dataset to use ('mnist', 'cifar10', 'csv')
            data_distribution: Data distribution type ('iid', 'non_iid')
            local_epochs: Local training epochs (E)
            batch_size: Mini-batch size (B)
            csv_path: Path to CSV file (required if dataset='csv')
            target_column: Target column name for CSV dataset
        """
        self.num_devices = num_devices
        self.dataset = dataset
        self.data_distribution = data_distribution
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.target_column = target_column

        # Initialize components
        self.devices = []
        self.aggregator = GlobalAggregator()

        # Dataset properties
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.input_shape = None
        self.num_classes = None
        self.problem_type = None

        # Model and training
        self.global_model = None
        self.model_size = 0

        # Metrics
        self.accuracy_history = []
        self.loss_history = []
        self.communication_cost = 0

        # Initialize system
        self._initialize_system()

    def _initialize_system(self) -> None:
        """Initialize the federated learning system."""
        logger.info(f"Initializing FL system with {self.num_devices} devices")

        # Load and prepare dataset
        self._load_dataset()

        # Create global model
        self._create_model()

        # Initialize devices
        self._initialize_devices()

        # Distribute data among devices
        self._distribute_data()

        logger.info("FL system initialization completed")

    def _load_dataset(self) -> None:
        """Load and preprocess dataset based on type."""
        if self.dataset == 'csv':
            self._load_csv_dataset()
        elif self.dataset == 'mnist':
            self._load_mnist_dataset()
        elif self.dataset == 'cifar10':
            self._load_cifar10_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

    def _load_csv_dataset(self) -> None:
        """Load and preprocess CSV dataset."""
        if self.csv_path is None:
            raise ValueError("CSV path must be provided for CSV dataset")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        logger.info(f"Loading CSV dataset from {self.csv_path}")

        # Load CSV data
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

        # Determine target column
        if self.target_column is None:
            self.target_column = df.columns[-1]

        logger.info(f"Using '{self.target_column}' as target column")

        # Separate features and target
        feature_columns = [col for col in df.columns if col != self.target_column]
        X = df[feature_columns].values
        y = df[self.target_column].values

        # Process features
        X_processed = self._process_features(df, feature_columns)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_processed)

        # Process target
        y_processed = self._process_target(y)

        # Split into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_processed, test_size=0.2, random_state=42
        )

        # Set input shape
        self.input_shape = (X_scaled.shape[1],)

        logger.info(f"Train set: {self.x_train.shape}, Test set: {self.x_test.shape}")

    def _process_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Process and encode features."""
        self.label_encoders = {}
        X_processed = np.zeros((len(df), len(feature_columns)), dtype=float)

        for i, col in enumerate(feature_columns):
            if df[col].dtype == 'object':
                # Categorical feature
                le = LabelEncoder()
                X_processed[:, i] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                # Numerical feature
                X_processed[:, i] = df[col].values

        return X_processed

    def _process_target(self, y: np.ndarray) -> np.ndarray:
        """Process and encode target variable."""
        # Determine problem type
        unique_values = np.unique(y)

        if len(unique_values) <= 20 and y.dtype != float:
            # Classification problem
            self.problem_type = 'classification'
            self.num_classes = len(unique_values)

            # Encode labels for classification
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y)
            else:
                y_encoded = y.astype(int)
                self.target_encoder = None

            # One-hot encode for multi-class
            if self.num_classes > 2:
                y_processed = tf.keras.utils.to_categorical(y_encoded, self.num_classes)
            else:
                y_processed = y_encoded.reshape(-1, 1)

            logger.info(f"Detected classification problem with {self.num_classes} classes")
        else:
            # Regression problem
            self.problem_type = 'regression'
            self.num_classes = 1

            # Scale target for regression
            self.target_scaler = StandardScaler()
            y_processed = self.target_scaler.fit_transform(y.reshape(-1, 1))

            logger.info("Detected regression problem")

        return y_processed

    def _load_mnist_dataset(self) -> None:
        """Load MNIST dataset."""
        logger.info("Loading MNIST dataset")

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # Reshape for CNN
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)

        # One-hot encode
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.problem_type = 'classification'

        logger.info(f"MNIST loaded: Train {self.x_train.shape}, Test {self.x_test.shape}")

    def _load_cifar10_dataset(self) -> None:
        """Load CIFAR-10 dataset."""
        logger.info("Loading CIFAR-10 dataset")

        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # One-hot encode
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, 10)

        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.problem_type = 'classification'

        logger.info(f"CIFAR-10 loaded: Train {self.x_train.shape}, Test {self.x_test.shape}")

    def _create_model(self) -> None:
        """Create neural network model architecture based on dataset type."""
        if self.dataset == 'csv':
            # Dense network for tabular data
            self.global_model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=self.input_shape),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2)
            ])

            # Output layer based on problem type
            if self.problem_type == 'classification':
                if self.num_classes > 2:
                    self.global_model.add(layers.Dense(self.num_classes, activation='softmax'))
                    loss = 'categorical_crossentropy'
                    metrics = ['accuracy']
                else:
                    self.global_model.add(layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
            else:  # regression
                self.global_model.add(layers.Dense(1, activation='linear'))
                loss = 'mse'
                metrics = ['mae']

        elif self.dataset == 'mnist':
            # CNN for MNIST
            self.global_model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        elif self.dataset == 'cifar10':
            # CNN for CIFAR-10
            self.global_model = models.Sequential([
                layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.input_shape),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        # Compile model
        self.global_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
        )

        # Set up aggregator
        self.aggregator.set_global_model(self.global_model)

        # Calculate model size for communication cost
        self.model_size = self._get_model_size()

        logger.info(f"Model created with {self.global_model.count_params()} parameters")
        logger.info(f"Model size: {self.model_size:.2f} MB")

    def _get_model_size(self) -> float:
        """Calculate model size in MB."""
        weights = self.global_model.get_weights()
        size_bytes = sum(weight.nbytes for weight in weights)
        return size_bytes / (1024 * 1024)

    def _initialize_devices(self) -> None:
        """Initialize local devices."""
        self.devices = []
        for i in range(self.num_devices):
            device = LocalDevice(
                device_id=i,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size
            )
            self.devices.append(device)

        logger.info(f"Initialized {len(self.devices)} local devices")

    def _distribute_data(self) -> None:
        """Distribute data among devices according to specified distribution."""
        num_samples = len(self.x_train)

        if self.data_distribution == 'iid':
            self._distribute_iid_data(num_samples)
        elif self.data_distribution == 'non_iid':
            self._distribute_non_iid_data(num_samples)
        else:
            raise ValueError(f"Unsupported data distribution: {self.data_distribution}")

        logger.info(f"Data distributed among {self.num_devices} devices ({self.data_distribution})")

    def _distribute_iid_data(self, num_samples: int) -> None:
        """Distribute data in IID manner."""
        indices = np.random.permutation(num_samples)
        samples_per_device = num_samples // self.num_devices

        for i, device in enumerate(self.devices):
            start_idx = i * samples_per_device
            end_idx = (i + 1) * samples_per_device if i < self.num_devices - 1 else num_samples
            device_indices = indices[start_idx:end_idx]

            device.set_data(
                self.x_train[device_indices],
                self.y_train[device_indices]
            )

    def _distribute_non_iid_data(self, num_samples: int) -> None:
        """Distribute data in Non-IID manner."""
        if self.problem_type == 'classification' and self.dataset in ['mnist', 'cifar10']:
            self._distribute_non_iid_by_class(num_samples)
        else:
            self._distribute_non_iid_sorted(num_samples)

    def _distribute_non_iid_by_class(self, num_samples: int) -> None:
        """Distribute data by class for image datasets."""
        if self.num_classes > 2:
            sorted_indices = np.argsort(np.argmax(self.y_train, axis=1))
        else:
            sorted_indices = np.argsort(self.y_train.flatten())

        samples_per_class = num_samples // self.num_classes

        for i, device in enumerate(self.devices):
            # Each device gets 2 classes
            class1 = (2 * i) % self.num_classes
            class2 = (2 * i + 1) % self.num_classes

            class1_start = class1 * samples_per_class
            class1_end = min((class1 + 1) * samples_per_class, num_samples)
            class2_start = class2 * samples_per_class
            class2_end = min((class2 + 1) * samples_per_class, num_samples)

            class1_indices = sorted_indices[class1_start:class1_end]
            class2_indices = sorted_indices[class2_start:class2_end]

            device_indices = np.concatenate([class1_indices, class2_indices])
            np.random.shuffle(device_indices)

            device.set_data(
                self.x_train[device_indices],
                self.y_train[device_indices]
            )

    def _distribute_non_iid_sorted(self, num_samples: int) -> None:
        """Distribute data in sorted manner for CSV datasets."""
        if self.problem_type == 'regression':
            sorted_indices = np.argsort(self.y_train.flatten())
        else:
            sorted_indices = np.argsort(self.y_train.flatten())

        samples_per_device = num_samples // self.num_devices

        for i, device in enumerate(self.devices):
            start_idx = i * samples_per_device
            end_idx = (i + 1) * samples_per_device if i < self.num_devices - 1 else num_samples
            device_indices = sorted_indices[start_idx:end_idx]

            device.set_data(
                self.x_train[device_indices],
                self.y_train[device_indices]
            )

    def run_round(self) -> Dict[str, float]:
        """
        Run one FL round implementing FedAvg algorithm.

        Implements the federated averaging algorithm described in the paper.

        Returns:
            Dictionary containing round metrics
        """
        # Get current global model weights
        global_weights = self.global_model.get_weights()

        # Collect local updates
        local_weights_list = []
        sample_counts = []

        # Local training on each device
        for device in self.devices:
            device_weights, num_samples = device.train_local_model(
                global_weights, self.global_model
            )
            local_weights_list.append(device_weights)
            sample_counts.append(num_samples)

        # Aggregate local updates
        aggregated_weights = self.aggregator.federated_averaging(
            local_weights_list, sample_counts
        )

        # Update global model
        self.global_model.set_weights(aggregated_weights)

        # Calculate communication cost
        self.communication_cost += self.model_size * self.num_devices * 2  # Upload + download

        # Evaluate global model
        metrics = self._evaluate_global_model()

        return metrics

    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on test set."""
        if self.x_test is None or self.y_test is None:
            return {}

        # Evaluate on test set
        test_results = self.global_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )

        # Extract metrics
        metrics = {}
        metric_names = self.global_model.metrics_names

        for i, name in enumerate(metric_names):
            metrics[name] = test_results[i]

        # Store history
        self.loss_history.append(metrics.get('loss', 0))
        if 'accuracy' in metrics:
            self.accuracy_history.append(metrics['accuracy'])

        return metrics

    def train(self, num_rounds: int = 100, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the federated learning system.

        Args:
            num_rounds: Number of FL rounds
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting FL training for {num_rounds} rounds")

        history = {
            'loss': [],
            'accuracy': [],
            'communication_cost': []
        }

        for round_num in range(num_rounds):
            # Run FL round
            round_metrics = self.run_round()

            # Store metrics
            history['loss'].append(round_metrics.get('loss', 0))
            history['accuracy'].append(round_metrics.get('accuracy', 0))
            history['communication_cost'].append(self.communication_cost)

            if verbose and (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}/{num_rounds}: "
                           f"Loss: {round_metrics.get('loss', 0):.4f}, "
                           f"Accuracy: {round_metrics.get('accuracy', 0):.4f}")

        logger.info("FL training completed")
        return history

    def get_model_weights(self) -> List[np.ndarray]:
        """Get current global model weights."""
        return self.global_model.get_weights()

    def set_model_weights(self, weights: List[np.ndarray]) -> None:
        """Set global model weights."""
        self.global_model.set_weights(weights)

    def get_communication_cost(self) -> float:
        """Get total communication cost in MB."""
        return self.communication_cost

    def get_device_data_distribution(self) -> Dict[int, int]:
        """Get data distribution across devices."""
        distribution = {}
        for device in self.devices:
            distribution[device.device_id] = device.num_samples
        return distribution

    def save_model(self, filepath: str) -> None:
        """Save the global model."""
        self.global_model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model."""
        self.global_model = tf.keras.models.load_model(filepath)
        self.aggregator.set_global_model(self.global_model)
        logger.info(f"Model loaded from {filepath}")


