#!/usr/bin/env python3
"""
Input Validation Module for 6G IoT Research Framework

This module provides comprehensive input validation functions for all
configuration parameters and data inputs throughout the framework.

Author: Research Team
Date: 2025
"""

import os
import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Dict, Optional, Union, Callable
from ..core.exceptions import ValidationError, ConfigurationError


def validate_positive_number(value: Union[int, float],
                           parameter_name: str,
                           allow_zero: bool = False) -> Union[int, float]:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        allow_zero: Whether to allow zero values

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{parameter_name} must be a number",
            parameter=parameter_name,
            expected_type="int or float",
            actual_value=value
        )

    if np.isnan(value) or np.isinf(value):
        raise ValidationError(
            f"{parameter_name} must be a finite number",
            parameter=parameter_name,
            actual_value=value
        )

    if allow_zero and value < 0:
        raise ValidationError(
            f"{parameter_name} must be non-negative",
            parameter=parameter_name,
            actual_value=value
        )
    elif not allow_zero and value <= 0:
        raise ValidationError(
            f"{parameter_name} must be positive",
            parameter=parameter_name,
            actual_value=value
        )

    return value


def validate_integer_range(value: int,
                          parameter_name: str,
                          min_value: Optional[int] = None,
                          max_value: Optional[int] = None) -> int:
    """
    Validate that a value is an integer within a specified range.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        raise ValidationError(
            f"{parameter_name} must be an integer",
            parameter=parameter_name,
            expected_type="int",
            actual_value=value
        )

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{parameter_name} must be >= {min_value}",
            parameter=parameter_name,
            actual_value=value
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{parameter_name} must be <= {max_value}",
            parameter=parameter_name,
            actual_value=value
        )

    return value


def validate_float_range(value: float,
                        parameter_name: str,
                        min_value: Optional[float] = None,
                        max_value: Optional[float] = None,
                        inclusive_min: bool = True,
                        inclusive_max: bool = True) -> float:
    """
    Validate that a value is a float within a specified range.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        inclusive_min: Whether minimum is inclusive
        inclusive_max: Whether maximum is inclusive

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{parameter_name} must be a number",
            parameter=parameter_name,
            expected_type="float",
            actual_value=value
        )

    value = float(value)

    if np.isnan(value) or np.isinf(value):
        raise ValidationError(
            f"{parameter_name} must be a finite number",
            parameter=parameter_name,
            actual_value=value
        )

    if min_value is not None:
        if inclusive_min and value < min_value:
            raise ValidationError(
                f"{parameter_name} must be >= {min_value}",
                parameter=parameter_name,
                actual_value=value
            )
        elif not inclusive_min and value <= min_value:
            raise ValidationError(
                f"{parameter_name} must be > {min_value}",
                parameter=parameter_name,
                actual_value=value
            )

    if max_value is not None:
        if inclusive_max and value > max_value:
            raise ValidationError(
                f"{parameter_name} must be <= {max_value}",
                parameter=parameter_name,
                actual_value=value
            )
        elif not inclusive_max and value >= max_value:
            raise ValidationError(
                f"{parameter_name} must be < {max_value}",
                parameter=parameter_name,
                actual_value=value
            )

    return value


def validate_string_choice(value: str,
                          parameter_name: str,
                          valid_choices: List[str],
                          case_sensitive: bool = True) -> str:
    """
    Validate that a string value is one of the allowed choices.

    Args:
        value: Value to validate
        parameter_name: Name of the parameter for error messages
        valid_choices: List of valid string choices
        case_sensitive: Whether comparison is case sensitive

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{parameter_name} must be a string",
            parameter=parameter_name,
            expected_type="str",
            actual_value=value
        )

    comparison_value = value if case_sensitive else value.lower()
    comparison_choices = valid_choices if case_sensitive else [c.lower() for c in valid_choices]

    if comparison_value not in comparison_choices:
        raise ValidationError(
            f"{parameter_name} must be one of {valid_choices}",
            parameter=parameter_name,
            actual_value=value
        )

    return value


def validate_array_shape(array: np.ndarray,
                        parameter_name: str,
                        expected_shape: Optional[Tuple[int, ...]] = None,
                        min_dimensions: Optional[int] = None,
                        max_dimensions: Optional[int] = None) -> np.ndarray:
    """
    Validate numpy array shape and dimensions.

    Args:
        array: Array to validate
        parameter_name: Name of the parameter for error messages
        expected_shape: Expected exact shape (None for any)
        min_dimensions: Minimum number of dimensions
        max_dimensions: Maximum number of dimensions

    Returns:
        Validated array

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(
            f"{parameter_name} must be a numpy array",
            parameter=parameter_name,
            expected_type="numpy.ndarray",
            actual_value=type(array).__name__
        )

    if expected_shape is not None and array.shape != expected_shape:
        raise ValidationError(
            f"{parameter_name} must have shape {expected_shape}, got {array.shape}",
            parameter=parameter_name,
            actual_value=array.shape
        )

    if min_dimensions is not None and array.ndim < min_dimensions:
        raise ValidationError(
            f"{parameter_name} must have at least {min_dimensions} dimensions, got {array.ndim}",
            parameter=parameter_name,
            actual_value=array.ndim
        )

    if max_dimensions is not None and array.ndim > max_dimensions:
        raise ValidationError(
            f"{parameter_name} must have at most {max_dimensions} dimensions, got {array.ndim}",
            parameter=parameter_name,
            actual_value=array.ndim
        )

    return array


def validate_position_3d(position: Tuple[float, float, float],
                        parameter_name: str,
                        bounds: Optional[Tuple[Tuple[float, float], ...]] = None) -> Tuple[float, float, float]:
    """
    Validate 3D position coordinates.

    Args:
        position: 3D position tuple (x, y, z)
        parameter_name: Name of the parameter for error messages
        bounds: Optional bounds for each dimension ((x_min, x_max), (y_min, y_max), (z_min, z_max))

    Returns:
        Validated position tuple

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(position, (tuple, list)) or len(position) != 3:
        raise ValidationError(
            f"{parameter_name} must be a 3-element tuple or list (x, y, z)",
            parameter=parameter_name,
            expected_type="tuple[float, float, float]",
            actual_value=position
        )

    # Validate each coordinate
    validated_coords = []
    coord_names = ['x', 'y', 'z']

    for i, (coord, name) in enumerate(zip(position, coord_names)):
        if not isinstance(coord, (int, float)):
            raise ValidationError(
                f"{parameter_name}.{name} must be a number",
                parameter=f"{parameter_name}.{name}",
                expected_type="float",
                actual_value=coord
            )

        coord = float(coord)

        if np.isnan(coord) or np.isinf(coord):
            raise ValidationError(
                f"{parameter_name}.{name} must be a finite number",
                parameter=f"{parameter_name}.{name}",
                actual_value=coord
            )

        # Check bounds if provided
        if bounds is not None and i < len(bounds):
            min_bound, max_bound = bounds[i]
            if coord < min_bound or coord > max_bound:
                raise ValidationError(
                    f"{parameter_name}.{name} must be between {min_bound} and {max_bound}",
                    parameter=f"{parameter_name}.{name}",
                    actual_value=coord
                )

        validated_coords.append(coord)

    return tuple(validated_coords)


def validate_file_path(file_path: str,
                      parameter_name: str,
                      must_exist: bool = True,
                      allowed_extensions: Optional[List[str]] = None) -> str:
    """
    Validate file path.

    Args:
        file_path: File path to validate
        parameter_name: Name of the parameter for error messages
        must_exist: Whether the file must exist
        allowed_extensions: List of allowed file extensions (with dots)

    Returns:
        Validated file path

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(file_path, str):
        raise ValidationError(
            f"{parameter_name} must be a string",
            parameter=parameter_name,
            expected_type="str",
            actual_value=file_path
        )

    if not file_path.strip():
        raise ValidationError(
            f"{parameter_name} cannot be empty",
            parameter=parameter_name,
            actual_value=file_path
        )

    if must_exist and not os.path.exists(file_path):
        raise ValidationError(
            f"File not found: {file_path}",
            parameter=parameter_name,
            actual_value=file_path
        )

    if allowed_extensions is not None:
        file_ext = os.path.splitext(file_path)[1].lower()
        allowed_extensions_lower = [ext.lower() for ext in allowed_extensions]

        if file_ext not in allowed_extensions_lower:
            raise ValidationError(
                f"{parameter_name} must have one of these extensions: {allowed_extensions}",
                parameter=parameter_name,
                actual_value=file_ext
            )

    return file_path


def validate_dataframe(df: pd.DataFrame,
                      parameter_name: str,
                      required_columns: Optional[List[str]] = None,
                      min_rows: Optional[int] = None,
                      max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Validate pandas DataFrame.

    Args:
        df: DataFrame to validate
        parameter_name: Name of the parameter for error messages
        required_columns: List of required column names
        min_rows: Minimum number of rows
        max_rows: Maximum number of rows

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"{parameter_name} must be a pandas DataFrame",
            parameter=parameter_name,
            expected_type="pandas.DataFrame",
            actual_value=type(df).__name__
        )

    if df.empty:
        raise ValidationError(
            f"{parameter_name} cannot be empty",
            parameter=parameter_name,
            actual_value="empty DataFrame"
        )

    if required_columns is not None:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(
                f"{parameter_name} is missing required columns: {list(missing_columns)}",
                parameter=parameter_name,
                actual_value=list(df.columns)
            )

    if min_rows is not None and len(df) < min_rows:
        raise ValidationError(
            f"{parameter_name} must have at least {min_rows} rows, got {len(df)}",
            parameter=parameter_name,
            actual_value=len(df)
        )

    if max_rows is not None and len(df) > max_rows:
        raise ValidationError(
            f"{parameter_name} must have at most {max_rows} rows, got {len(df)}",
            parameter=parameter_name,
            actual_value=len(df)
        )

    return df


def validate_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate network configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If validation fails
    """
    validated_config = {}

    # Number of IoT devices
    if 'num_iot_devices' in config:
        validated_config['num_iot_devices'] = validate_integer_range(
            config['num_iot_devices'], 'num_iot_devices', min_value=1, max_value=1000
        )

    # Number of IRS elements
    if 'num_irs_elements' in config:
        validated_config['num_irs_elements'] = validate_integer_range(
            config['num_irs_elements'], 'num_irs_elements', min_value=1, max_value=10000
        )

    # Area size
    if 'area_size' in config:
        validated_config['area_size'] = validate_positive_number(
            config['area_size'], 'area_size'
        )

    # Base station position
    if 'bs_position' in config:
        validated_config['bs_position'] = validate_position_3d(
            config['bs_position'], 'bs_position'
        )

    # IRS position
    if 'irs_position' in config:
        validated_config['irs_position'] = validate_position_3d(
            config['irs_position'], 'irs_position'
        )

    # IoT height
    if 'iot_height' in config:
        validated_config['iot_height'] = validate_positive_number(
            config['iot_height'], 'iot_height'
        )

    return validated_config


def validate_channel_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate channel model parameters.

    Args:
        params: Channel parameters dictionary

    Returns:
        Validated parameters dictionary

    Raises:
        ValidationError: If validation fails
    """
    validated_params = {}

    # Path loss exponents
    if 'path_loss_exponent_direct' in params:
        validated_params['path_loss_exponent_direct'] = validate_float_range(
            params['path_loss_exponent_direct'], 'path_loss_exponent_direct',
            min_value=1.0, max_value=6.0
        )

    if 'path_loss_exponent_irs' in params:
        validated_params['path_loss_exponent_irs'] = validate_float_range(
            params['path_loss_exponent_irs'], 'path_loss_exponent_irs',
            min_value=1.0, max_value=6.0
        )

    # Rician K-factors
    if 'rician_k_direct' in params:
        validated_params['rician_k_direct'] = validate_positive_number(
            params['rician_k_direct'], 'rician_k_direct', allow_zero=True
        )

    if 'rician_k_irs' in params:
        validated_params['rician_k_irs'] = validate_positive_number(
            params['rician_k_irs'], 'rician_k_irs', allow_zero=True
        )

    # Shadowing standard deviations
    if 'shadowing_std_direct' in params:
        validated_params['shadowing_std_direct'] = validate_positive_number(
            params['shadowing_std_direct'], 'shadowing_std_direct'
        )

    if 'shadowing_std_irs' in params:
        validated_params['shadowing_std_irs'] = validate_positive_number(
            params['shadowing_std_irs'], 'shadowing_std_irs'
        )

    # System parameters
    if 'carrier_frequency' in params:
        validated_params['carrier_frequency'] = validate_float_range(
            params['carrier_frequency'], 'carrier_frequency',
            min_value=0.1, max_value=300.0  # GHz
        )

    if 'bandwidth' in params:
        validated_params['bandwidth'] = validate_positive_number(
            params['bandwidth'], 'bandwidth'
        )

    return validated_params


def validate_fl_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate federated learning configuration parameters.

    Args:
        config: FL configuration dictionary

    Returns:
        Validated configuration dictionary

    Raises:
        ValidationError: If validation fails
    """
    validated_config = {}

    # Number of devices
    if 'num_devices' in config:
        validated_config['num_devices'] = validate_integer_range(
            config['num_devices'], 'num_devices', min_value=2, max_value=1000
        )

    # Local epochs
    if 'local_epochs' in config:
        validated_config['local_epochs'] = validate_integer_range(
            config['local_epochs'], 'local_epochs', min_value=1, max_value=100
        )

    # Batch size
    if 'batch_size' in config:
        validated_config['batch_size'] = validate_integer_range(
            config['batch_size'], 'batch_size', min_value=1, max_value=1024
        )

    # Learning rate
    if 'learning_rate' in config:
        validated_config['learning_rate'] = validate_float_range(
            config['learning_rate'], 'learning_rate',
            min_value=1e-6, max_value=1.0, inclusive_min=False
        )

    # Number of rounds
    if 'num_rounds' in config:
        validated_config['num_rounds'] = validate_integer_range(
            config['num_rounds'], 'num_rounds', min_value=1, max_value=10000
        )

    # Data distribution
    if 'data_distribution' in config:
        validated_config['data_distribution'] = validate_string_choice(
            config['data_distribution'], 'data_distribution',
            ['iid', 'non_iid'], case_sensitive=False
        )

    # Dataset type
    if 'dataset' in config:
        validated_config['dataset'] = validate_string_choice(
            config['dataset'], 'dataset',
            ['mnist', 'cifar10', 'csv'], case_sensitive=False
        )

    # CSV file path (if dataset is CSV)
    if config.get('dataset', '').lower() == 'csv' and 'csv_path' in config:
        validated_config['csv_path'] = validate_file_path(
            config['csv_path'], 'csv_path', must_exist=True,
            allowed_extensions=['.csv']
        )

    return validated_config


class ConfigValidator:
    """
    Configuration validator class with custom validation rules.
    """

    def __init__(self):
        """Initialize the validator."""
        self.validators: Dict[str, Callable] = {
            'network': validate_network_config,
            'channel': validate_channel_params,
            'federated_learning': validate_fl_config
        }

    def add_validator(self, config_type: str, validator_func: Callable) -> None:
        """
        Add a custom validator function.

        Args:
            config_type: Type of configuration
            validator_func: Validation function
        """
        self.validators[config_type] = validator_func

    def validate(self, config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
        """
        Validate configuration using the appropriate validator.

        Args:
            config: Configuration dictionary
            config_type: Type of configuration

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigurationError: If validator not found
            ValidationError: If validation fails
        """
        if config_type not in self.validators:
            raise ConfigurationError(
                f"No validator found for configuration type: {config_type}",
                section=config_type
            )

        try:
            return self.validators[config_type](config)
        except ValidationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Validation failed for {config_type}: {str(e)}",
                section=config_type
            ) from e


# Global validator instance
_config_validator = ConfigValidator()


def validate_config(config: Dict[str, Any], config_type: str) -> Dict[str, Any]:
    """
    Validate configuration using the global validator.

    Args:
        config: Configuration dictionary
        config_type: Type of configuration

    Returns:
        Validated configuration dictionary
    """
    return _config_validator.validate(config, config_type)
