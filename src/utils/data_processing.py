"""
Data processing utilities for the 6G IoT research framework.

This module provides utilities for handling datasets, data distribution,
and preprocessing for federated learning scenarios.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# Import custom exceptions and logging
from ..core.exceptions import DatasetError, ValidationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_file_path,
    validate_dataframe, validate_array_shape
)
from .logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)


class DataDistributor:
    """Handles data distribution for federated learning scenarios."""

    def __init__(self, random_seed: int = 42):
        """
        Initialize the data distributor.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    def create_iid_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create IID (Independent and Identically Distributed) data distribution.

        Args:
            X: Input features
            y: Target labels
            num_clients: Number of federated learning clients

        Returns:
            List of (X_client, y_client) tuples for each client

        Raises:
            DatasetError: If data distribution fails
        """
        try:
            # Validate inputs
            X = validate_array_shape(X, 'X', min_dimensions=2)
            y = validate_array_shape(y, 'y', min_dimensions=1)
            num_clients = validate_integer_range(
                num_clients, 'num_clients', min_value=1, max_value=1000
            )

            if len(X) != len(y):
                raise ValidationError(
                    "X and y must have the same number of samples",
                    parameter='data_length'
                )

            logger.info(f"Creating IID distribution for {num_clients} clients")

            with PerformanceLogger(logger, "IID data distribution"):
                # Shuffle data
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                # Split data evenly among clients
                client_data = []
                samples_per_client = len(X) // num_clients

                for i in range(num_clients):
                    start_idx = i * samples_per_client
                    if i == num_clients - 1:  # Last client gets remaining samples
                        end_idx = len(X)
                    else:
                        end_idx = (i + 1) * samples_per_client

                    X_client = X_shuffled[start_idx:end_idx]
                    y_client = y_shuffled[start_idx:end_idx]
                    client_data.append((X_client, y_client))

                logger.info(f"Created IID distribution with {len(client_data)} clients")
                return client_data

        except ValidationError as e:
            logger.error(f"Data validation failed for IID distribution: {e}")
            raise DatasetError(
                f"Invalid data for IID distribution: {e.message}",
                dataset_type='iid_distribution'
            ) from e
        except Exception as e:
            logger.error(f"Failed to create IID distribution: {e}")
            log_exception(logger, e, "IID data distribution")
            raise DatasetError(
                f"Failed to create IID distribution: {str(e)}",
                dataset_type='iid_distribution'
            ) from e

    def create_non_iid_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_clients: int,
        alpha: float = 0.5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create Non-IID data distribution using Dirichlet distribution.

        Args:
            X: Input features
            y: Target labels
            num_clients: Number of federated learning clients
            alpha: Dirichlet distribution parameter (lower = more non-IID)

        Returns:
            List of (X_client, y_client) tuples for each client
        """
        logger.info(
            f"Creating Non-IID distribution for {num_clients} clients with alpha={alpha}"
        )

        num_classes = len(np.unique(y))
        client_data = [[] for _ in range(num_clients)]

        # Group data by class
        class_indices = {}
        for class_id in range(num_classes):
            class_indices[class_id] = np.where(y == class_id)[0]
            np.random.shuffle(class_indices[class_id])

        # Distribute each class using Dirichlet distribution
        for class_id in range(num_classes):
            class_data_indices = class_indices[class_id]
            proportions = np.random.dirichlet([alpha] * num_clients)

            # Calculate number of samples for each client
            class_splits = (proportions * len(class_data_indices)).astype(int)
            # Adjust last split
            class_splits[-1] = len(class_data_indices) - np.sum(class_splits[:-1])

            # Distribute class data to clients
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = start_idx + class_splits[client_id]
                client_indices = class_data_indices[start_idx:end_idx]
                client_data[client_id].extend(client_indices)
                start_idx = end_idx

        # Convert to final format
        result = []
        for client_id in range(num_clients):
            client_indices = np.array(client_data[client_id])
            if len(client_indices) > 0:
                X_client = X[client_indices]
                y_client = y[client_indices]
            else:
                # Handle empty clients
                X_client = np.empty((0,) + X.shape[1:])
                y_client = np.empty((0,))
            result.append((X_client, y_client))

        logger.info(f"Created Non-IID distribution with {len(result)} clients")
        return result


class DatasetLoader:
    """Handles loading and preprocessing of various datasets."""

    @staticmethod
    def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess MNIST dataset.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Loading MNIST dataset")

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Reshape for CNN if needed
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        logger.info(f"MNIST loaded: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CIFAR-10 dataset.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info("Loading CIFAR-10 dataset")

        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Normalize pixel values
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        logger.info(f"CIFAR-10 loaded: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, y_train, X_test, y_test

    @staticmethod
    def load_csv_dataset(
        filepath: str,
        target_column: str,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess CSV dataset.

        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            test_size: Proportion of data for testing
            normalize: Whether to normalize features

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)

        Raises:
            DatasetError: If dataset loading or processing fails
        """
        try:
            # Validate inputs
            filepath = validate_file_path(
                filepath, 'filepath', must_exist=True, allowed_extensions=['.csv']
            )
            test_size = validate_float_range(
                test_size, 'test_size', min_value=0.0, max_value=1.0,
                inclusive_min=False, inclusive_max=False
            )

            logger.info(f"Loading CSV dataset from {filepath}")

            with PerformanceLogger(logger, "CSV dataset loading"):
                # Load data
                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    raise DatasetError(
                        f"Failed to read CSV file: {str(e)}",
                        dataset_type='csv',
                        file_path=filepath
                    ) from e

                # Validate DataFrame
                df = validate_dataframe(df, 'dataset', min_rows=10)

                # Check if target column exists
                if target_column not in df.columns:
                    raise DatasetError(
                        f"Target column '{target_column}' not found in dataset. "
                        f"Available columns: {list(df.columns)}",
                        dataset_type='csv',
                        file_path=filepath
                    )

                # Separate features and target
                X = df.drop(columns=[target_column]).values
                y = df[target_column].values

                # Check for missing values
                if np.any(pd.isna(X)) or np.any(pd.isna(y)):
                    logger.warning("Dataset contains missing values, removing rows with NaN")
                    mask = ~(np.any(pd.isna(X), axis=1) | pd.isna(y))
                    X = X[mask]
                    y = y[mask]

                # Encode categorical target if needed
                if y.dtype == 'object':
                    try:
                        label_encoder = LabelEncoder()
                        y = label_encoder.fit_transform(y)
                        logger.info(f"Encoded categorical target with {len(label_encoder.classes_)} classes")
                    except Exception as e:
                        raise DatasetError(
                            f"Failed to encode categorical target: {str(e)}",
                            dataset_type='csv',
                            file_path=filepath
                        ) from e

                # Split data
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                except Exception as e:
                    # Try without stratification if it fails
                    logger.warning("Stratified split failed, using random split")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                # Normalize features if requested
                if normalize:
                    try:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        logger.info("Features normalized using StandardScaler")
                    except Exception as e:
                        logger.warning(f"Feature normalization failed: {e}")

                logger.info(f"CSV dataset loaded: Train {X_train.shape}, Test {X_test.shape}")
                return X_train, y_train, X_test, y_test

        except DatasetError:
            raise
        except ValidationError as e:
            logger.error(f"Data validation failed for CSV dataset: {e}")
            raise DatasetError(
                f"Invalid parameters for CSV dataset loading: {e.message}",
                dataset_type='csv',
                file_path=filepath
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error loading CSV dataset: {e}")
            log_exception(logger, e, "CSV dataset loading")
            raise DatasetError(
                f"Failed to load CSV dataset: {str(e)}",
                dataset_type='csv',
                file_path=filepath
            ) from e


class DataAugmentation:
    """Data augmentation utilities for improving model robustness."""

    @staticmethod
    def add_noise(X: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to data.

        Args:
            X: Input data
            noise_factor: Standard deviation of noise

        Returns:
            Noisy data
        """
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise

    @staticmethod
    def simulate_channel_effects(
        X: np.ndarray,
        snr_db: float = 20.0
    ) -> np.ndarray:
        """
        Simulate wireless channel effects on data transmission.

        Args:
            X: Input data
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Data with simulated channel effects
        """
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)

        # Calculate noise power
        signal_power = np.mean(X ** 2)
        noise_power = signal_power / snr_linear

        # Add noise
        noise = np.random.normal(0, np.sqrt(noise_power), X.shape)
        return X + noise


def validate_data_distribution(
    client_data: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Any]:
    """
    Validate and analyze data distribution across clients.

    Args:
        client_data: List of (X, y) tuples for each client

    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'num_clients': len(client_data),
        'total_samples': 0,
        'samples_per_client': [],
        'class_distribution': {},
        'balance_score': 0.0
    }

    all_labels = []
    for X_client, y_client in client_data:
        stats['total_samples'] += len(X_client)
        stats['samples_per_client'].append(len(X_client))
        all_labels.extend(y_client.tolist())

    # Calculate class distribution
    unique_classes = np.unique(all_labels)
    for class_id in unique_classes:
        stats['class_distribution'][int(class_id)] = int(np.sum(np.array(all_labels) == class_id))

    # Calculate balance score (coefficient of variation)
    if len(stats['samples_per_client']) > 1:
        mean_samples = np.mean(stats['samples_per_client'])
        std_samples = np.std(stats['samples_per_client'])
        stats['balance_score'] = std_samples / mean_samples if mean_samples > 0 else 0.0

    return stats


def preprocess_network_data(network_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Preprocess network topology and channel data for ML models.

    Args:
        network_data: Dictionary containing network parameters

    Returns:
        Dictionary of preprocessed numpy arrays
    """
    processed_data = {}

    try:
        # Process device positions
        if 'device_positions' in network_data:
            positions = np.array(network_data['device_positions'])
            processed_data['device_positions'] = positions

        # Process channel coefficients
        if 'channel_coefficients' in network_data:
            channels = np.array(network_data['channel_coefficients'])
            # Normalize channel coefficients
            processed_data['channel_coefficients'] = channels / np.max(np.abs(channels))

        # Process SNR values
        if 'snr_values' in network_data:
            snr = np.array(network_data['snr_values'])
            # Convert to linear scale if in dB
            if np.any(snr < 0):  # Likely in dB
                snr_linear = 10 ** (snr / 10)
                processed_data['snr_linear'] = snr_linear
            processed_data['snr_db'] = snr

        # Process data rates
        if 'data_rates' in network_data:
            rates = np.array(network_data['data_rates'])
            processed_data['data_rates'] = rates

        logger.info("Network data preprocessing completed")
        return processed_data

    except Exception as e:
        logger.error(f"Failed to preprocess network data: {str(e)}")
        raise ValueError(f"Failed to preprocess network data: {str(e)}")


def create_synthetic_dataset(
    num_samples: int,
    num_features: int,
    num_classes: int,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing purposes.

    Args:
        num_samples: Number of samples to generate
        num_features: Number of features per sample
        num_classes: Number of classes
        noise_level: Amount of noise to add
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(random_state)

    # Generate random features
    X = np.random.randn(num_samples, num_features)

    # Generate labels based on linear combination of features with noise
    weights = np.random.randn(num_features)
    linear_combination = X @ weights

    # Add noise
    linear_combination += noise_level * np.random.randn(num_samples)

    # Convert to class labels
    thresholds = np.linspace(linear_combination.min(), linear_combination.max(), num_classes + 1)
    y = np.digitize(linear_combination, thresholds[1:-1])

    logger.info(f"Created synthetic dataset: {num_samples} samples, {num_features} features, {num_classes} classes")
    return X, y
