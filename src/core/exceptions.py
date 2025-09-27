#!/usr/bin/env python3
"""
Custom Exception Classes for 6G IoT Research Framework

This module defines custom exception classes for different error types
that can occur throughout the research framework.

Author: Research Team
Date: 2025
"""


class ResearchFrameworkError(Exception):
    """
    Base exception for the research framework.

    All custom exceptions in the framework should inherit from this class.
    """

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize the base exception.

        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class NetworkConfigurationError(ResearchFrameworkError):
    """
    Raised when network configuration is invalid.

    This exception is raised when there are issues with network topology
    configuration, such as invalid positions, device counts, or parameters.
    """

    def __init__(self, message: str, parameter: str = None, value=None):
        """
        Initialize network configuration error.

        Args:
            message: Error message
            parameter: Name of the invalid parameter
            value: Invalid value that caused the error
        """
        details = {}
        if parameter:
            details['parameter'] = parameter
        if value is not None:
            details['value'] = value

        super().__init__(message, "NET_CONFIG", details)
        self.parameter = parameter
        self.value = value


class ChannelModelError(ResearchFrameworkError):
    """
    Raised when channel modeling operations fail.

    This exception is raised when there are issues with channel coefficient
    generation, path loss calculations, or fading model operations.
    """

    def __init__(self, message: str, channel_type: str = None, operation: str = None):
        """
        Initialize channel model error.

        Args:
            message: Error message
            channel_type: Type of channel ('direct', 'irs', etc.)
            operation: Operation that failed
        """
        details = {}
        if channel_type:
            details['channel_type'] = channel_type
        if operation:
            details['operation'] = operation

        super().__init__(message, "CHANNEL_MODEL", details)
        self.channel_type = channel_type
        self.operation = operation


class FederatedLearningError(ResearchFrameworkError):
    """
    Raised when federated learning operations fail.

    This exception is raised when there are issues with FL training,
    aggregation, or device management.
    """

    def __init__(self, message: str, device_id: int = None, round_number: int = None):
        """
        Initialize federated learning error.

        Args:
            message: Error message
            device_id: ID of the device that caused the error
            round_number: FL round number when error occurred
        """
        details = {}
        if device_id is not None:
            details['device_id'] = device_id
        if round_number is not None:
            details['round_number'] = round_number

        super().__init__(message, "FL_ERROR", details)
        self.device_id = device_id
        self.round_number = round_number


class DatasetError(ResearchFrameworkError):
    """
    Raised when dataset operations fail.

    This exception is raised when there are issues with dataset loading,
    preprocessing, or data distribution.
    """

    def __init__(self, message: str, dataset_type: str = None, file_path: str = None):
        """
        Initialize dataset error.

        Args:
            message: Error message
            dataset_type: Type of dataset ('csv', 'mnist', 'cifar10')
            file_path: Path to the dataset file (if applicable)
        """
        details = {}
        if dataset_type:
            details['dataset_type'] = dataset_type
        if file_path:
            details['file_path'] = file_path

        super().__init__(message, "DATASET_ERROR", details)
        self.dataset_type = dataset_type
        self.file_path = file_path


class IRSOptimizationError(ResearchFrameworkError):
    """
    Raised when IRS optimization operations fail.

    This exception is raised when there are issues with IRS phase shift
    optimization, configuration, or performance evaluation.
    """

    def __init__(self, message: str, optimization_method: str = None, iteration: int = None):
        """
        Initialize IRS optimization error.

        Args:
            message: Error message
            optimization_method: Optimization method that failed
            iteration: Iteration number when error occurred
        """
        details = {}
        if optimization_method:
            details['optimization_method'] = optimization_method
        if iteration is not None:
            details['iteration'] = iteration

        super().__init__(message, "IRS_OPT", details)
        self.optimization_method = optimization_method
        self.iteration = iteration


class MADRLError(ResearchFrameworkError):
    """
    Raised when Multi-Agent Deep Reinforcement Learning operations fail.

    This exception is raised when there are issues with MADRL agent training,
    environment setup, or reward calculation.
    """

    def __init__(self, message: str, agent_id: int = None, episode: int = None):
        """
        Initialize MADRL error.

        Args:
            message: Error message
            agent_id: ID of the agent that caused the error
            episode: Episode number when error occurred
        """
        details = {}
        if agent_id is not None:
            details['agent_id'] = agent_id
        if episode is not None:
            details['episode'] = episode

        super().__init__(message, "MADRL_ERROR", details)
        self.agent_id = agent_id
        self.episode = episode


class ValidationError(ResearchFrameworkError):
    """
    Raised when input validation fails.

    This exception is raised when input parameters fail validation checks.
    """

    def __init__(self, message: str, parameter: str = None, expected_type: str = None,
                 actual_value=None):
        """
        Initialize validation error.

        Args:
            message: Error message
            parameter: Name of the parameter that failed validation
            expected_type: Expected type or format
            actual_value: Actual value that failed validation
        """
        details = {}
        if parameter:
            details['parameter'] = parameter
        if expected_type:
            details['expected_type'] = expected_type
        if actual_value is not None:
            details['actual_value'] = actual_value

        super().__init__(message, "VALIDATION", details)
        self.parameter = parameter
        self.expected_type = expected_type
        self.actual_value = actual_value


class ConfigurationError(ResearchFrameworkError):
    """
    Raised when configuration loading or parsing fails.

    This exception is raised when there are issues with configuration files
    or parameter settings.
    """

    def __init__(self, message: str, config_file: str = None, section: str = None):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_file: Configuration file that caused the error
            section: Configuration section with the error
        """
        details = {}
        if config_file:
            details['config_file'] = config_file
        if section:
            details['section'] = section

        super().__init__(message, "CONFIG_ERROR", details)
        self.config_file = config_file
        self.section = section


class ComputationError(ResearchFrameworkError):
    """
    Raised when computational operations fail.

    This exception is raised when there are issues with mathematical
    computations, matrix operations, or numerical instabilities.
    """

    def __init__(self, message: str, operation: str = None, matrix_shape: tuple = None):
        """
        Initialize computation error.

        Args:
            message: Error message
            operation: Mathematical operation that failed
            matrix_shape: Shape of matrices involved (if applicable)
        """
        details = {}
        if operation:
            details['operation'] = operation
        if matrix_shape:
            details['matrix_shape'] = matrix_shape

        super().__init__(message, "COMPUTATION", details)
        self.operation = operation
        self.matrix_shape = matrix_shape


class ResourceError(ResearchFrameworkError):
    """
    Raised when resource-related operations fail.

    This exception is raised when there are issues with memory allocation,
    file I/O, or other resource management operations.
    """

    def __init__(self, message: str, resource_type: str = None, resource_limit=None):
        """
        Initialize resource error.

        Args:
            message: Error message
            resource_type: Type of resource ('memory', 'disk', 'cpu')
            resource_limit: Resource limit that was exceeded
        """
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_limit is not None:
            details['resource_limit'] = resource_limit

        super().__init__(message, "RESOURCE", details)
        self.resource_type = resource_type
        self.resource_limit = resource_limit
