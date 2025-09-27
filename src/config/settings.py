"""
Configuration management for the 6G IoT research framework.

This module provides centralized configuration management with support for
different environments, parameter validation, and easy configuration loading.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path

# Import custom exceptions and logging
from ..core.exceptions import ConfigurationError, ValidationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_float_range,
    validate_string_choice, validate_file_path
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class NetworkConfig:
    """Network topology configuration parameters."""

    # Basic network parameters
    num_iot_devices: int = 20
    area_size: float = 100.0  # meters

    # Base station configuration
    bs_position: List[float] = None
    bs_height: float = 10.0
    bs_transmit_power_dbm: float = 30.0

    # IRS configuration
    irs_position: List[float] = None
    irs_height: float = 5.0
    num_irs_elements: int = 100
    irs_element_spacing: float = 0.5  # wavelengths

    # IoT device configuration
    iot_height_range: List[float] = None
    iot_transmit_power_dbm: float = 20.0

    def __post_init__(self):
        """Set default values for list fields."""
        if self.bs_position is None:
            self.bs_position = [50.0, 50.0, self.bs_height]
        if self.irs_position is None:
            self.irs_position = [75.0, 25.0, self.irs_height]
        if self.iot_height_range is None:
            self.iot_height_range = [1.0, 3.0]


@dataclass
class ChannelConfig:
    """Wireless channel configuration parameters."""

    # Frequency and bandwidth
    carrier_frequency_ghz: float = 28.0
    bandwidth_mhz: float = 100.0

    # Path loss parameters
    path_loss_exponent_direct: float = 3.0
    path_loss_exponent_irs: float = 2.5
    reference_distance: float = 1.0  # meters

    # Fading parameters
    rician_k_direct: float = 5.0  # dB
    rician_k_irs: float = 10.0  # dB
    shadowing_std: float = 6.0  # dB

    # Noise parameters
    noise_figure_db: float = 9.0
    thermal_noise_density_dbm_hz: float = -174.0


@dataclass
class FederatedLearningConfig:
    """Federated learning configuration parameters."""

    # Basic FL parameters
    num_clients: int = 20
    num_rounds: int = 100
    clients_per_round: int = 10

    # Local training parameters
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001

    # Data distribution
    data_distribution: str = 'iid'  # 'iid' or 'non_iid'
    non_iid_alpha: float = 0.5  # Dirichlet parameter for non-IID

    # Model parameters
    model_type: str = 'cnn'  # 'cnn', 'mlp', 'resnet'
    num_classes: int = 10

    # Aggregation parameters
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox', 'scaffold'
    aggregation_weights: str = 'uniform'  # 'uniform', 'data_size', 'accuracy'


@dataclass
class IRSConfig:
    """IRS optimization configuration parameters."""

    # Optimization algorithm
    optimization_method: str = 'gradient_descent'  # 'gradient_descent', 'genetic', 'pso'
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

    # Gradient descent parameters
    learning_rate: float = 0.01
    momentum: float = 0.9

    # Genetic algorithm parameters
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # PSO parameters
    num_particles: int = 30
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0

    # Phase shift constraints
    phase_shift_resolution: int = 64  # Number of discrete phase levels
    continuous_phase: bool = False  # True for continuous, False for discrete


@dataclass
class MADRLConfig:
    """Multi-Agent Deep Reinforcement Learning configuration parameters."""

    # Environment parameters
    num_agents: int = 20
    state_dim: int = 10
    action_dim: int = 4
    max_episode_length: int = 1000

    # Training parameters
    num_episodes: int = 5000
    learning_rate: float = 0.0003
    batch_size: int = 64
    memory_size: int = 100000

    # Network architecture
    hidden_layers: List[int] = None
    activation_function: str = 'relu'

    # RL algorithm parameters
    algorithm: str = 'dqn'  # 'dqn', 'ddpg', 'ppo', 'sac'
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update parameter
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Reward function parameters
    energy_weight: float = 0.3
    throughput_weight: float = 0.4
    fairness_weight: float = 0.3

    def __post_init__(self):
        """Set default values for list fields."""
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256]


@dataclass
class SimulationConfig:
    """Simulation configuration parameters."""

    # General simulation parameters
    random_seed: int = 42
    num_monte_carlo_runs: int = 100
    save_results: bool = True
    results_dir: str = 'results'

    # Logging configuration
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_file: str = 'simulation.log'

    # Performance parameters
    use_gpu: bool = True
    num_parallel_workers: int = 4
    memory_limit_gb: float = 8.0

    # Visualization parameters
    generate_plots: bool = True
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'
    plot_dpi: int = 300


@dataclass
class SystemConfig:
    """Complete system configuration combining all components."""

    network: NetworkConfig = None
    channel: ChannelConfig = None
    federated_learning: FederatedLearningConfig = None
    irs: IRSConfig = None
    madrl: MADRLConfig = None
    simulation: SimulationConfig = None

    def __post_init__(self):
        """Initialize sub-configurations with defaults if not provided."""
        if self.network is None:
            self.network = NetworkConfig()
        if self.channel is None:
            self.channel = ChannelConfig()
        if self.federated_learning is None:
            self.federated_learning = FederatedLearningConfig()
        if self.irs is None:
            self.irs = IRSConfig()
        if self.madrl is None:
            self.madrl = MADRLConfig()
        if self.simulation is None:
            self.simulation = SimulationConfig()


class ConfigManager:
    """Configuration manager for loading, saving, and validating configurations."""

    def __init__(self, config_dir: str = 'config'):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def load_config(self, config_path: Union[str, Path]) -> SystemConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            SystemConfig object

        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Validate file path
            config_path = Path(config_path)

            if not config_path.exists():
                logger.warning(f"Configuration file {config_path} not found. Using defaults.")
                return SystemConfig()

            # Validate file extension
            allowed_extensions = ['.yaml', '.yml', '.json']
            if config_path.suffix.lower() not in allowed_extensions:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}. "
                    f"Supported formats: {allowed_extensions}",
                    config_file=str(config_path)
                )

            logger.info(f"Loading configuration from {config_path}")

            with PerformanceLogger(logger, "Configuration loading"):
                with open(config_path, 'r') as f:
                    try:
                        if config_path.suffix.lower() in ['.yaml', '.yml']:
                            config_dict = yaml.safe_load(f)
                        elif config_path.suffix.lower() == '.json':
                            config_dict = json.load(f)
                    except yaml.YAMLError as e:
                        raise ConfigurationError(
                            f"Invalid YAML format in configuration file: {str(e)}",
                            config_file=str(config_path)
                        ) from e
                    except json.JSONDecodeError as e:
                        raise ConfigurationError(
                            f"Invalid JSON format in configuration file: {str(e)}",
                            config_file=str(config_path)
                        ) from e

                if config_dict is None:
                    logger.warning("Configuration file is empty, using defaults")
                    return SystemConfig()

                # Convert dictionary to configuration object
                config = self._dict_to_config(config_dict)

                # Validate the loaded configuration
                validation_errors = self.validate_config(config)
                if validation_errors:
                    error_msg = "Configuration validation failed:\n"
                    for component, errors in validation_errors.items():
                        error_msg += f"  {component}: {', '.join(errors)}\n"

                    logger.error(error_msg)
                    raise ConfigurationError(
                        f"Configuration validation failed: {validation_errors}",
                        config_file=str(config_path)
                    )

                logger.info("Configuration loaded and validated successfully")
                return config

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration from {config_path}: {e}")
            log_exception(logger, e, "Configuration loading")
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                config_file=str(config_path)
            ) from e

    def save_config(self, config: SystemConfig, config_path: Union[str, Path]) -> None:
        """
        Save configuration to file.

        Args:
            config: SystemConfig object to save
            config_path: Path where to save the configuration

        Raises:
            ConfigurationError: If configuration saving fails
        """
        try:
            config_path = Path(config_path)

            # Validate file extension
            allowed_extensions = ['.yaml', '.yml', '.json']
            if config_path.suffix.lower() not in allowed_extensions:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {config_path.suffix}. "
                    f"Supported formats: {allowed_extensions}",
                    config_file=str(config_path)
                )

            # Create parent directories
            config_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving configuration to {config_path}")

            with PerformanceLogger(logger, "Configuration saving"):
                # Validate configuration before saving
                validation_errors = self.validate_config(config)
                if validation_errors:
                    error_msg = "Cannot save invalid configuration:\n"
                    for component, errors in validation_errors.items():
                        error_msg += f"  {component}: {', '.join(errors)}\n"

                    raise ConfigurationError(
                        f"Configuration validation failed: {validation_errors}",
                        config_file=str(config_path)
                    )

                config_dict = self._config_to_dict(config)

                with open(config_path, 'w') as f:
                    try:
                        if config_path.suffix.lower() in ['.yaml', '.yml']:
                            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                        elif config_path.suffix.lower() == '.json':
                            json.dump(config_dict, f, indent=2)
                    except Exception as e:
                        raise ConfigurationError(
                            f"Failed to write configuration file: {str(e)}",
                            config_file=str(config_path)
                        ) from e

                logger.info(f"Configuration saved successfully to {config_path}")

        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving configuration to {config_path}: {e}")
            log_exception(logger, e, "Configuration saving")
            raise ConfigurationError(
                f"Failed to save configuration: {str(e)}",
                config_file=str(config_path)
            ) from e

    def validate_config(self, config: SystemConfig) -> Dict[str, List[str]]:
        """
        Validate configuration parameters.

        Args:
            config: SystemConfig object to validate

        Returns:
            Dictionary with validation errors by component
        """
        errors = {}

        # Validate network configuration
        network_errors = []
        if config.network.num_iot_devices <= 0:
            network_errors.append("num_iot_devices must be positive")
        if config.network.area_size <= 0:
            network_errors.append("area_size must be positive")
        if config.network.num_irs_elements <= 0:
            network_errors.append("num_irs_elements must be positive")
        if len(config.network.bs_position) != 3:
            network_errors.append("bs_position must have 3 coordinates")
        if len(config.network.irs_position) != 3:
            network_errors.append("irs_position must have 3 coordinates")

        if network_errors:
            errors['network'] = network_errors

        # Validate channel configuration
        channel_errors = []
        if config.channel.carrier_frequency_ghz <= 0:
            channel_errors.append("carrier_frequency_ghz must be positive")
        if config.channel.bandwidth_mhz <= 0:
            channel_errors.append("bandwidth_mhz must be positive")
        if config.channel.path_loss_exponent_direct < 2.0:
            channel_errors.append("path_loss_exponent_direct should be >= 2.0")

        if channel_errors:
            errors['channel'] = channel_errors

        # Validate federated learning configuration
        fl_errors = []
        if config.federated_learning.num_clients <= 0:
            fl_errors.append("num_clients must be positive")
        if config.federated_learning.num_rounds <= 0:
            fl_errors.append("num_rounds must be positive")
        if config.federated_learning.clients_per_round > config.federated_learning.num_clients:
            fl_errors.append("clients_per_round cannot exceed num_clients")
        if config.federated_learning.learning_rate <= 0:
            fl_errors.append("learning_rate must be positive")
        if config.federated_learning.data_distribution not in ['iid', 'non_iid']:
            fl_errors.append("data_distribution must be 'iid' or 'non_iid'")

        if fl_errors:
            errors['federated_learning'] = fl_errors

        # Validate IRS configuration
        irs_errors = []
        if config.irs.max_iterations <= 0:
            irs_errors.append("max_iterations must be positive")
        if config.irs.learning_rate <= 0:
            irs_errors.append("learning_rate must be positive")
        if config.irs.phase_shift_resolution <= 0:
            irs_errors.append("phase_shift_resolution must be positive")

        if irs_errors:
            errors['irs'] = irs_errors

        # Validate MADRL configuration
        madrl_errors = []
        if config.madrl.num_agents <= 0:
            madrl_errors.append("num_agents must be positive")
        if config.madrl.learning_rate <= 0:
            madrl_errors.append("learning_rate must be positive")
        if config.madrl.gamma < 0 or config.madrl.gamma > 1:
            madrl_errors.append("gamma must be between 0 and 1")
        if config.madrl.epsilon_start < 0 or config.madrl.epsilon_start > 1:
            madrl_errors.append("epsilon_start must be between 0 and 1")

        if madrl_errors:
            errors['madrl'] = madrl_errors

        return errors

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object."""
        config = SystemConfig()

        if 'network' in config_dict:
            config.network = NetworkConfig(**config_dict['network'])
        if 'channel' in config_dict:
            config.channel = ChannelConfig(**config_dict['channel'])
        if 'federated_learning' in config_dict:
            config.federated_learning = FederatedLearningConfig(**config_dict['federated_learning'])
        if 'irs' in config_dict:
            config.irs = IRSConfig(**config_dict['irs'])
        if 'madrl' in config_dict:
            config.madrl = MADRLConfig(**config_dict['madrl'])
        if 'simulation' in config_dict:
            config.simulation = SimulationConfig(**config_dict['simulation'])

        return config

    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig object to dictionary."""
        return {
            'network': asdict(config.network),
            'channel': asdict(config.channel),
            'federated_learning': asdict(config.federated_learning),
            'irs': asdict(config.irs),
            'madrl': asdict(config.madrl),
            'simulation': asdict(config.simulation)
        }

    def create_default_config(self, config_path: Union[str, Path]) -> None:
        """
        Create a default configuration file.

        Args:
            config_path: Path where to save the default configuration
        """
        default_config = SystemConfig()
        self.save_config(default_config, config_path)
        logger.info(f"Default configuration created at {config_path}")

    def update_config(self, config: SystemConfig, updates: Dict[str, Any]) -> SystemConfig:
        """
        Update configuration with new values.

        Args:
            config: Original configuration
            updates: Dictionary with updates (nested keys supported with dots)

        Returns:
            Updated configuration
        """
        config_dict = self._config_to_dict(config)

        for key, value in updates.items():
            keys = key.split('.')
            current = config_dict

            # Navigate to the nested dictionary
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the value
            current[keys[-1]] = value

        return self._dict_to_config(config_dict)


def load_config_from_env() -> SystemConfig:
    """
    Load configuration from environment variables.

    Returns:
        SystemConfig object with values from environment variables
    """
    config = SystemConfig()

    # Network configuration from environment
    if 'NUM_IOT_DEVICES' in os.environ:
        config.network.num_iot_devices = int(os.environ['NUM_IOT_DEVICES'])
    if 'AREA_SIZE' in os.environ:
        config.network.area_size = float(os.environ['AREA_SIZE'])

    # Federated learning configuration from environment
    if 'FL_NUM_CLIENTS' in os.environ:
        config.federated_learning.num_clients = int(os.environ['FL_NUM_CLIENTS'])
    if 'FL_NUM_ROUNDS' in os.environ:
        config.federated_learning.num_rounds = int(os.environ['FL_NUM_ROUNDS'])
    if 'FL_LEARNING_RATE' in os.environ:
        config.federated_learning.learning_rate = float(os.environ['FL_LEARNING_RATE'])

    # Simulation configuration from environment
    if 'RANDOM_SEED' in os.environ:
        config.simulation.random_seed = int(os.environ['RANDOM_SEED'])
    if 'USE_GPU' in os.environ:
        config.simulation.use_gpu = os.environ['USE_GPU'].lower() in ['true', '1', 'yes']

    return config


def get_default_config() -> SystemConfig:
    """
    Get default system configuration.

    Returns:
        SystemConfig object with default values
    """
    return SystemConfig()


# Global configuration instance
_global_config: Optional[SystemConfig] = None


def set_global_config(config: SystemConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def get_global_config() -> SystemConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = SystemConfig()
    return _global_config


def reset_global_config() -> None:
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = SystemConfig()
