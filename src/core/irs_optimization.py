#!/usr/bin/env python3
"""
IRS Optimization Module for 6G IoT Networks

This module implements the Intelligent Reflecting Surface (IRS) optimization
algorithms described in the research paper. It includes gradient-based optimization,
phase shift control, and performance evaluation metrics for IRS-assisted
communication systems.

Author: Research Team
Date: 2025
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass
import logging
from .network_topology import NetworkTopology
from .channel_model import ChannelModel
from .exceptions import IRSOptimizationError, ValidationError, ComputationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_float_range,
    validate_array_shape, validate_string_choice
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Set up logger
logger = get_logger(__name__)


@dataclass
class IRSOptimizationConfig:
    """Configuration parameters for IRS optimization."""
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    learning_rate: float = 0.01
    optimization_objective: str = 'sum_rate'  # 'sum_rate', 'min_rate', 'energy_efficiency'
    transmit_power: float = 0.1  # watts
    regularization_factor: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        try:
            # Validate max_iterations
            self.max_iterations = validate_integer_range(
                self.max_iterations, 'max_iterations', min_value=1, max_value=100000
            )

            # Validate convergence_threshold
            self.convergence_threshold = validate_float_range(
                self.convergence_threshold, 'convergence_threshold',
                min_value=1e-12, max_value=1e-2, inclusive_min=False
            )

            # Validate learning_rate
            self.learning_rate = validate_float_range(
                self.learning_rate, 'learning_rate',
                min_value=1e-6, max_value=1.0, inclusive_min=False
            )

            # Validate optimization_objective
            self.optimization_objective = validate_string_choice(
                self.optimization_objective, 'optimization_objective',
                ['sum_rate', 'min_rate', 'energy_efficiency'], case_sensitive=False
            )

            # Validate transmit_power
            self.transmit_power = validate_positive_number(
                self.transmit_power, 'transmit_power'
            )

            # Validate regularization_factor
            self.regularization_factor = validate_float_range(
                self.regularization_factor, 'regularization_factor',
                min_value=0.0, max_value=1.0
            )

            logger.debug(f"IRSOptimizationConfig validated: {self}")

        except ValidationError as e:
            logger.error(f"IRSOptimizationConfig validation failed: {e}")
            raise IRSOptimizationError(
                f"Invalid IRS optimization configuration: {e.message}",
                optimization_method="config_validation"
            ) from e
    use_momentum: bool = True
    momentum_factor: float = 0.9


class PhaseShiftController:
    """
    Phase Shift Controller for IRS configuration management.

    This class manages the phase shift configuration of the IRS elements,
    providing methods for setting, updating, and constraining phase shifts.
    """

    def __init__(self, num_elements: int):
        """
        Initialize the phase shift controller.

        Args:
            num_elements: Number of IRS elements

        Raises:
            IRSOptimizationError: If initialization fails
        """
        try:
            # Validate number of elements
            self.num_elements = validate_integer_range(
                num_elements, 'num_elements', min_value=1, max_value=10000
            )

            # Initialize phase shifts randomly
            self.phase_shifts = np.random.uniform(0, 2*np.pi, self.num_elements)
            self.phase_history = []

            logger.debug(f"PhaseShiftController initialized with {self.num_elements} elements")

        except ValidationError as e:
            logger.error(f"PhaseShiftController initialization failed: {e}")
            raise IRSOptimizationError(
                f"Invalid number of IRS elements: {e.message}",
                optimization_method="controller_init"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in PhaseShiftController initialization: {e}")
            raise IRSOptimizationError(
                f"Failed to initialize phase shift controller: {str(e)}",
                optimization_method="controller_init"
            ) from e

    def set_phase_shifts(self, phase_shifts: np.ndarray) -> None:
        """
        Set the phase shifts for all elements.

        Args:
            phase_shifts: Array of phase shifts in radians [0, 2π]

        Raises:
            IRSOptimizationError: If phase shifts are invalid
        """
        try:
            # Validate input array
            phase_shifts = validate_array_shape(
                phase_shifts, 'phase_shifts',
                expected_shape=(self.num_elements,)
            )

            # Validate phase shift values (should be between 0 and 2π)
            if np.any(phase_shifts < 0) or np.any(phase_shifts > 2 * np.pi):
                raise ValidationError(
                    "Phase shifts must be between 0 and 2π radians",
                    parameter='phase_shifts'
                )

            # Check for NaN or infinite values
            if np.any(~np.isfinite(phase_shifts)):
                raise ValidationError(
                    "Phase shifts must be finite values",
                    parameter='phase_shifts'
                )

            # Constrain phase shifts to [0, 2π]
            self.phase_shifts = np.mod(phase_shifts, 2*np.pi)
            self.phase_history.append(self.phase_shifts.copy())

            logger.debug(f"Phase shifts updated: {len(phase_shifts)} elements")

        except ValidationError as e:
            logger.error(f"Invalid phase shifts: {e}")
            raise IRSOptimizationError(
                f"Invalid phase shifts: {e.message}",
                optimization_method="phase_shift_update"
            ) from e
        except Exception as e:
            logger.error(f"Failed to set phase shifts: {e}")
            raise IRSOptimizationError(
                f"Failed to set phase shifts: {str(e)}",
                optimization_method="phase_shift_update"
            ) from e

    def get_phase_shifts(self) -> np.ndarray:
        """Get the current phase shifts."""
        return self.phase_shifts.copy()

    def get_reflection_coefficients(self) -> np.ndarray:
        """
        Get the complex reflection coefficients.

        Returns:
            Complex array of reflection coefficients e^(j*θ)
        """
        return np.exp(1j * self.phase_shifts)

    def update_phase_shifts(self, gradient: np.ndarray, learning_rate: float) -> None:
        """
        Update phase shifts using gradient information.

        Args:
            gradient: Gradient of the objective function w.r.t. phase shifts
            learning_rate: Learning rate for the update
        """
        # Gradient ascent update (for maximization problems)
        new_phase_shifts = self.phase_shifts + learning_rate * gradient
        self.set_phase_shifts(new_phase_shifts)

    def randomize_phase_shifts(self) -> None:
        """Randomize all phase shifts."""
        self.phase_shifts = np.random.uniform(0, 2*np.pi, self.num_elements)
        self.phase_history.append(self.phase_shifts.copy())

    def set_uniform_phase_shifts(self, phase: float) -> None:
        """
        Set all elements to the same phase shift.

        Args:
            phase: Phase shift value in radians
        """
        self.phase_shifts = np.full(self.num_elements, np.mod(phase, 2*np.pi))
        self.phase_history.append(self.phase_shifts.copy())

    def get_phase_history(self) -> List[np.ndarray]:
        """Get the history of phase shift configurations."""
        return self.phase_history.copy()

    def clear_history(self) -> None:
        """Clear the phase shift history."""
        self.phase_history = []

    def __repr__(self) -> str:
        return f"PhaseShiftController(elements={self.num_elements})"


class IRSOptimizer:
    """
    IRS Optimizer implementing gradient-based optimization algorithms.

    This class implements the IRS optimization algorithms described in the paper,
    focusing on maximizing system performance metrics such as sum rate,
    minimum rate, or energy efficiency.
    """

    def __init__(self, network: NetworkTopology, channel_model: ChannelModel,
                 config: Optional[IRSOptimizationConfig] = None):
        """
        Initialize the IRS optimizer.

        Args:
            network: Network topology
            channel_model: Channel model
            config: Optimization configuration

        Raises:
            IRSOptimizationError: If initialization fails
        """
        try:
            # Validate inputs
            if not isinstance(network, NetworkTopology):
                raise ValidationError(
                    "network must be a NetworkTopology instance",
                    parameter='network'
                )

            if not isinstance(channel_model, ChannelModel):
                raise ValidationError(
                    "channel_model must be a ChannelModel instance",
                    parameter='channel_model'
                )

            self.network = network
            self.channel_model = channel_model
            self.config = config or IRSOptimizationConfig()

            # Initialize phase shift controller
            self.phase_controller = PhaseShiftController(network.irs.num_elements)

            # Optimization state
            self.optimization_history = []
            self.convergence_history = []
            self.momentum_velocity = np.zeros(network.irs.num_elements)

            # Performance metrics
            self.performance_metrics = {}

            logger.info(f"IRSOptimizer initialized with {network.irs.num_elements} IRS elements")

        except ValidationError as e:
            logger.error(f"IRSOptimizer initialization failed: {e}")
            raise IRSOptimizationError(
                f"Invalid optimizer parameters: {e.message}",
                optimization_method="optimizer_init"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in IRSOptimizer initialization: {e}")
            log_exception(logger, e, "IRSOptimizer initialization")
            raise IRSOptimizationError(
                f"Failed to initialize IRS optimizer: {str(e)}",
                optimization_method="optimizer_init"
            ) from e

        # Initialize objective function
        self.objective_functions = {
            'sum_rate': self._sum_rate_objective,
            'min_rate': self._min_rate_objective,
            'energy_efficiency': self._energy_efficiency_objective
        }

    def _sum_rate_objective(self, phase_shifts: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Sum rate maximization objective function.

        Args:
            phase_shifts: Current phase shift configuration

        Returns:
            Tuple of (objective_value, gradient)
        """
        # Update IRS configuration
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)

        # Calculate data rates
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        sum_rate = np.sum(data_rates)

        # Calculate gradient using finite differences
        gradient = self._calculate_gradient_finite_diff(phase_shifts, self._sum_rate_value)

        return sum_rate, gradient

    def _sum_rate_value(self, phase_shifts: np.ndarray) -> float:
        """Helper function to calculate sum rate value only."""
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        return np.sum(data_rates)

    def _min_rate_objective(self, phase_shifts: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Minimum rate maximization objective function.

        Args:
            phase_shifts: Current phase shift configuration

        Returns:
            Tuple of (objective_value, gradient)
        """
        # Update IRS configuration
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)

        # Calculate data rates
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        min_rate = np.min(data_rates)

        # Calculate gradient
        gradient = self._calculate_gradient_finite_diff(phase_shifts, self._min_rate_value)

        return min_rate, gradient

    def _min_rate_value(self, phase_shifts: np.ndarray) -> float:
        """Helper function to calculate min rate value only."""
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        return np.min(data_rates)

    def _energy_efficiency_objective(self, phase_shifts: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Energy efficiency maximization objective function.

        Args:
            phase_shifts: Current phase shift configuration

        Returns:
            Tuple of (objective_value, gradient)
        """
        # Update IRS configuration
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)

        # Calculate energy efficiency (bits/joule)
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        sum_rate = np.sum(data_rates)

        # Total power consumption (transmit power + circuit power)
        circuit_power = 0.01  # watts (assumed circuit power)
        total_power = self.config.transmit_power + circuit_power

        energy_efficiency = sum_rate / total_power

        # Calculate gradient
        gradient = self._calculate_gradient_finite_diff(phase_shifts, self._energy_efficiency_value)

        return energy_efficiency, gradient

    def _energy_efficiency_value(self, phase_shifts: np.ndarray) -> float:
        """Helper function to calculate energy efficiency value only."""
        irs_config = np.exp(1j * phase_shifts)
        self.channel_model.update_with_irs(irs_config)
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        sum_rate = np.sum(data_rates)
        circuit_power = 0.01
        total_power = self.config.transmit_power + circuit_power
        return sum_rate / total_power

    def _calculate_gradient_finite_diff(self, phase_shifts: np.ndarray,
                                      objective_func: Callable) -> np.ndarray:
        """
        Calculate gradient using finite differences.

        Args:
            phase_shifts: Current phase shift configuration
            objective_func: Objective function to differentiate

        Returns:
            Gradient array
        """
        gradient = np.zeros_like(phase_shifts)
        epsilon = 1e-6

        base_value = objective_func(phase_shifts)

        for i in range(len(phase_shifts)):
            # Forward difference
            phase_shifts_plus = phase_shifts.copy()
            phase_shifts_plus[i] += epsilon
            value_plus = objective_func(phase_shifts_plus)

            gradient[i] = (value_plus - base_value) / epsilon

        return gradient

    def optimize(self, initial_phase_shifts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform IRS optimization using gradient-based methods.

        Args:
            initial_phase_shifts: Initial phase shift configuration

        Returns:
            Optimization results dictionary

        Raises:
            IRSOptimizationError: If optimization fails
        """
        try:
            with PerformanceLogger(logger, f"IRS optimization ({self.config.optimization_objective})"):
                logger.info(f"Starting IRS optimization with {self.config.optimization_objective} objective")

                # Validate initial phase shifts if provided
                if initial_phase_shifts is not None:
                    initial_phase_shifts = validate_array_shape(
                        initial_phase_shifts, 'initial_phase_shifts',
                        expected_shape=(self.network.irs.num_elements,)
                    )

                # Initialize phase shifts
                if initial_phase_shifts is not None:
                    self.phase_controller.set_phase_shifts(initial_phase_shifts)
                else:
                    self.phase_controller.randomize_phase_shifts()

                # Get objective function
                if self.config.optimization_objective not in self.objective_functions:
                    raise IRSOptimizationError(
                        f"Unknown optimization objective: {self.config.optimization_objective}",
                        optimization_method=self.config.optimization_objective
                    )

                objective_func = self.objective_functions[self.config.optimization_objective]

                # Optimization loop
                best_objective = -np.inf
                best_phase_shifts = None
                convergence_count = 0

                for iteration in range(self.config.max_iterations):
                    try:
                        current_phase_shifts = self.phase_controller.get_phase_shifts()

                        # Calculate objective and gradient
                        objective_value, gradient = objective_func(current_phase_shifts)

                        # Validate objective value and gradient
                        if np.isnan(objective_value) or np.isinf(objective_value):
                            raise ComputationError(
                                f"Invalid objective value at iteration {iteration}: {objective_value}",
                                operation="objective_calculation"
                            )

                        if np.any(~np.isfinite(gradient)):
                            raise ComputationError(
                                f"Invalid gradient at iteration {iteration}",
                                operation="gradient_calculation"
                            )

                        # Add regularization if specified
            if self.config.regularization_factor > 0:
                objective_value -= self.config.regularization_factor * np.sum(current_phase_shifts**2)
                gradient -= 2 * self.config.regularization_factor * current_phase_shifts

            # Update best solution
            if objective_value > best_objective:
                best_objective = objective_value
                best_phase_shifts = current_phase_shifts.copy()
                convergence_count = 0
            else:
                convergence_count += 1

            # Store history
            self.optimization_history.append({
                'iteration': iteration,
                'objective_value': objective_value,
                'phase_shifts': current_phase_shifts.copy(),
                'gradient_norm': np.linalg.norm(gradient)
            })

                        # Check convergence
                        if np.linalg.norm(gradient) < self.config.convergence_threshold:
                            logger.info(f"Converged at iteration {iteration}")
                            break

                        if convergence_count > 50:  # Early stopping
                            logger.info(f"Early stopping at iteration {iteration}")
                            break

                        # Update phase shifts using gradient ascent with momentum
                        if hasattr(self.config, 'use_momentum') and self.config.use_momentum:
                            momentum_factor = getattr(self.config, 'momentum_factor', 0.9)
                            self.momentum_velocity = (momentum_factor * self.momentum_velocity +
                                                    self.config.learning_rate * gradient)
                            self.phase_controller.update_phase_shifts(self.momentum_velocity, 1.0)
                        else:
                            self.phase_controller.update_phase_shifts(gradient, self.config.learning_rate)

                    except (ComputationError, IRSOptimizationError):
                        raise
                    except Exception as e:
                        logger.error(f"Error in optimization iteration {iteration}: {e}")
                        raise IRSOptimizationError(
                            f"Optimization failed at iteration {iteration}: {str(e)}",
                            optimization_method=self.config.optimization_objective,
                            iteration=iteration
                        ) from e

                # Set best configuration
                if best_phase_shifts is not None:
                    self.phase_controller.set_phase_shifts(best_phase_shifts)
                    self.network.irs.set_phase_shifts(best_phase_shifts)
                else:
                    raise IRSOptimizationError(
                        "Optimization failed to find valid solution",
                        optimization_method=self.config.optimization_objective
                    )

                # Calculate final performance metrics
                final_metrics = self.evaluate_performance()

                results = {
                    'best_objective_value': best_objective,
                    'best_phase_shifts': best_phase_shifts,
                    'final_metrics': final_metrics,
                    'iterations': len(self.optimization_history),
                    'converged': np.linalg.norm(gradient) < self.config.convergence_threshold,
                    'optimization_history': self.optimization_history
                }

                logger.info(f"Optimization completed. Best objective: {best_objective:.6f}")
                return results

        except IRSOptimizationError:
            raise
        except ValidationError as e:
            logger.error(f"Validation error in IRS optimization: {e}")
            raise IRSOptimizationError(
                f"Invalid optimization parameters: {e.message}",
                optimization_method=self.config.optimization_objective
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in IRS optimization: {e}")
            log_exception(logger, e, "IRS optimization")
            raise IRSOptimizationError(
                f"IRS optimization failed: {str(e)}",
                optimization_method=self.config.optimization_objective
            ) from e

    def evaluate_performance(self) -> Dict[str, Any]:
        """
        Evaluate comprehensive performance metrics for the current IRS configuration.

        Returns:
            Dictionary containing performance metrics
        """
        # Update channel model with current IRS configuration
        current_config = self.phase_controller.get_reflection_coefficients()
        self.channel_model.update_with_irs(current_config)

        # Calculate basic metrics
        data_rates = self.channel_model.calculate_data_rate(self.config.transmit_power)
        snr_values = self.channel_model.calculate_snr(self.config.transmit_power)
        spectral_efficiency = self.channel_model.calculate_spectral_efficiency(self.config.transmit_power)

        # Performance metrics
        metrics = {
            'sum_rate_bps': np.sum(data_rates),
            'sum_rate_mbps': np.sum(data_rates) / 1e6,
            'average_rate_bps': np.mean(data_rates),
            'min_rate_bps': np.min(data_rates),
            'max_rate_bps': np.max(data_rates),
            'rate_std_bps': np.std(data_rates),
            'average_snr_db': 10 * np.log10(np.mean(snr_values)),
            'min_snr_db': 10 * np.log10(np.min(snr_values)),
            'max_snr_db': 10 * np.log10(np.max(snr_values)),
            'average_spectral_efficiency': np.mean(spectral_efficiency),
            'sum_spectral_efficiency': np.sum(spectral_efficiency)
        }

        # Energy efficiency
        circuit_power = 0.01  # watts
        total_power = self.config.transmit_power + circuit_power
        metrics['energy_efficiency_bps_per_watt'] = metrics['sum_rate_bps'] / total_power

        # Fairness metrics (Jain's fairness index)
        if len(data_rates) > 1:
            fairness_numerator = (np.sum(data_rates))**2
            fairness_denominator = len(data_rates) * np.sum(data_rates**2)
            metrics['jains_fairness_index'] = fairness_numerator / fairness_denominator
        else:
            metrics['jains_fairness_index'] = 1.0

        # IRS-specific metrics
        phase_shifts = self.phase_controller.get_phase_shifts()
        metrics['phase_shift_variance'] = np.var(phase_shifts)
        metrics['phase_shift_mean'] = np.mean(phase_shifts)
        metrics['phase_shift_range'] = np.max(phase_shifts) - np.min(phase_shifts)

        # Store metrics
        self.performance_metrics = metrics

        return metrics

    def compare_with_baseline(self, baseline_config: str = 'random') -> Dict[str, Any]:
        """
        Compare current IRS configuration with baseline configurations.

        Args:
            baseline_config: Type of baseline ('random', 'uniform', 'no_irs')

        Returns:
            Comparison results
        """
        # Store current configuration
        current_config = self.phase_controller.get_phase_shifts()
        current_metrics = self.evaluate_performance()

        # Evaluate baseline
        if baseline_config == 'random':
            self.phase_controller.randomize_phase_shifts()
        elif baseline_config == 'uniform':
            self.phase_controller.set_uniform_phase_shifts(0.0)
        elif baseline_config == 'no_irs':
            # Set all phase shifts to create destructive interference
            self.phase_controller.set_uniform_phase_shifts(np.pi)

        baseline_metrics = self.evaluate_performance()

        # Restore current configuration
        self.phase_controller.set_phase_shifts(current_config)

        # Calculate improvements
        improvements = {}
        for key in current_metrics:
            if isinstance(current_metrics[key], (int, float)):
                if baseline_metrics[key] != 0:
                    improvement_ratio = current_metrics[key] / baseline_metrics[key]
                    improvement_percent = (improvement_ratio - 1) * 100
                else:
                    improvement_ratio = float('inf') if current_metrics[key] > 0 else 1.0
                    improvement_percent = float('inf') if current_metrics[key] > 0 else 0.0

                improvements[f'{key}_improvement_ratio'] = improvement_ratio
                improvements[f'{key}_improvement_percent'] = improvement_percent

        return {
            'current_metrics': current_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements,
            'baseline_type': baseline_config
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history."""
        return self.optimization_history.copy()

    def reset_optimization(self) -> None:
        """Reset optimization state."""
        self.optimization_history = []
        self.convergence_history = []
        self.momentum_velocity = np.zeros(self.network.irs.num_elements)
        self.performance_metrics = {}
        self.phase_controller.clear_history()

    def save_configuration(self, filename: str) -> None:
        """
        Save current IRS configuration to file.

        Args:
            filename: Output filename
        """
        config_data = {
            'phase_shifts': self.phase_controller.get_phase_shifts(),
            'performance_metrics': self.performance_metrics,
            'optimization_config': self.config.__dict__,
            'network_config': self.network.config.__dict__
        }

        np.savez(filename, **config_data)
        self.logger.info(f"Configuration saved to {filename}")

    def load_configuration(self, filename: str) -> None:
        """
        Load IRS configuration from file.

        Args:
            filename: Input filename
        """
        data = np.load(filename, allow_pickle=True)

        phase_shifts = data['phase_shifts']
        self.phase_controller.set_phase_shifts(phase_shifts)
        self.network.irs.set_phase_shifts(phase_shifts)

        if 'performance_metrics' in data:
            self.performance_metrics = data['performance_metrics'].item()

        self.logger.info(f"Configuration loaded from {filename}")

    def __repr__(self) -> str:
        return (f"IRSOptimizer(objective='{self.config.optimization_objective}', "
                f"elements={self.network.irs.num_elements}, "
                f"devices={len(self.network.iot_devices)})")
