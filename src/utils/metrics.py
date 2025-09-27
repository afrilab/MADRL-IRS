"""
Performance metrics calculations for the 6G IoT research framework.

This module provides utilities for calculating and analyzing performance metrics
for federated learning, wireless communications, and energy efficiency.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Import custom exceptions and logging
from ..core.exceptions import ValidationError, ComputationError
from ..utils.validation import (
    validate_positive_number, validate_float_range, validate_array_shape,
    validate_integer_range
)
from .logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)


class FederatedLearningMetrics:
    """Metrics for evaluating federated learning performance."""

    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate classification accuracy.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score

        Raises:
            ValidationError: If input arrays are invalid
            ComputationError: If accuracy calculation fails
        """
        try:
            # Validate inputs
            y_true = validate_array_shape(y_true, 'y_true', min_dimensions=1, max_dimensions=1)
            y_pred = validate_array_shape(y_pred, 'y_pred', min_dimensions=1, max_dimensions=1)

            if len(y_true) != len(y_pred):
                raise ValidationError(
                    "y_true and y_pred must have the same length",
                    parameter='array_lengths'
                )

            if len(y_true) == 0:
                raise ValidationError(
                    "Input arrays cannot be empty",
                    parameter='array_length'
                )

            accuracy = accuracy_score(y_true, y_pred)
            return float(accuracy)

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            raise ComputationError(
                f"Accuracy calculation failed: {str(e)}",
                operation='accuracy_calculation'
            ) from e

    @staticmethod
    def calculate_precision_recall_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method ('weighted', 'macro', 'micro')

        Returns:
            Dictionary with precision, recall, and F1-score
        """
        return {
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
        }

    @staticmethod
    def calculate_communication_efficiency(
        num_rounds: int,
        num_clients: int,
        model_size_mb: float,
        final_accuracy: float
    ) -> Dict[str, float]:
        """
        Calculate communication efficiency metrics.

        Args:
            num_rounds: Number of communication rounds
            num_clients: Number of participating clients
            model_size_mb: Model size in MB
            final_accuracy: Final achieved accuracy

        Returns:
            Dictionary with communication efficiency metrics
        """
        total_data_transmitted = num_rounds * num_clients * model_size_mb * 2  # Upload + download

        return {
            'total_communication_mb': total_data_transmitted,
            'communication_per_accuracy': total_data_transmitted / final_accuracy if final_accuracy > 0 else float('inf'),
            'rounds_to_convergence': num_rounds,
            'efficiency_score': final_accuracy / (num_rounds * model_size_mb) if num_rounds > 0 and model_size_mb > 0 else 0.0
        }

    @staticmethod
    def calculate_convergence_metrics(
        accuracy_history: List[float],
        convergence_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate convergence-related metrics.

        Args:
            accuracy_history: List of accuracy values over rounds
            convergence_threshold: Threshold for determining convergence

        Returns:
            Dictionary with convergence metrics
        """
        if len(accuracy_history) < 2:
            return {
                'converged': False,
                'convergence_round': None,
                'final_accuracy': accuracy_history[0] if accuracy_history else 0.0,
                'improvement_rate': 0.0
            }

        # Check for convergence
        converged = False
        convergence_round = None

        for i in range(1, len(accuracy_history)):
            if abs(accuracy_history[i] - accuracy_history[i-1]) < convergence_threshold:
                converged = True
                convergence_round = i
                break

        # Calculate improvement rate
        improvement_rate = (accuracy_history[-1] - accuracy_history[0]) / len(accuracy_history)

        return {
            'converged': converged,
            'convergence_round': convergence_round,
            'final_accuracy': accuracy_history[-1],
            'improvement_rate': improvement_rate,
            'max_accuracy': max(accuracy_history),
            'min_accuracy': min(accuracy_history)
        }


class WirelessMetrics:
    """Metrics for evaluating wireless communication performance."""

    @staticmethod
    def calculate_snr_db(signal_power: float, noise_power: float) -> float:
        """
        Calculate Signal-to-Noise Ratio in dB.

        Args:
            signal_power: Signal power in linear scale
            noise_power: Noise power in linear scale

        Returns:
            SNR in dB

        Raises:
            ValidationError: If input powers are invalid
            ComputationError: If SNR calculation fails
        """
        try:
            # Validate inputs
            signal_power = validate_positive_number(signal_power, 'signal_power', allow_zero=True)
            noise_power = validate_positive_number(noise_power, 'noise_power', allow_zero=False)

            if np.isnan(signal_power) or np.isinf(signal_power):
                raise ValidationError(
                    "Signal power must be a finite number",
                    parameter='signal_power'
                )

            if np.isnan(noise_power) or np.isinf(noise_power):
                raise ValidationError(
                    "Noise power must be a finite number",
                    parameter='noise_power'
                )

            if signal_power == 0:
                return float('-inf')

            snr_db = 10 * np.log10(signal_power / noise_power)

            if np.isnan(snr_db) or np.isinf(snr_db):
                raise ComputationError(
                    "SNR calculation resulted in invalid value",
                    operation='snr_calculation'
                )

            return float(snr_db)

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to calculate SNR: {e}")
            raise ComputationError(
                f"SNR calculation failed: {str(e)}",
                operation='snr_calculation'
            ) from e

    @staticmethod
    def calculate_data_rate_mbps(
        bandwidth_mhz: float,
        snr_db: float
    ) -> float:
        """
        Calculate data rate using Shannon capacity formula.

        Args:
            bandwidth_mhz: Bandwidth in MHz
            snr_db: SNR in dB

        Returns:
            Data rate in Mbps
        """
        snr_linear = 10 ** (snr_db / 10)
        capacity_bps = bandwidth_mhz * 1e6 * np.log2(1 + snr_linear)
        return capacity_bps / 1e6  # Convert to Mbps

    @staticmethod
    def calculate_path_loss_db(
        distance_m: float,
        frequency_ghz: float,
        path_loss_exponent: float = 2.0
    ) -> float:
        """
        Calculate path loss in dB.

        Args:
            distance_m: Distance in meters
            frequency_ghz: Frequency in GHz
            path_loss_exponent: Path loss exponent

        Returns:
            Path loss in dB
        """
        if distance_m <= 0:
            return 0.0

        # Free space path loss at 1m
        fspl_1m = 20 * np.log10(frequency_ghz) + 92.45

        # Path loss with distance
        path_loss = fspl_1m + 10 * path_loss_exponent * np.log10(distance_m)

        return path_loss

    @staticmethod
    def calculate_channel_capacity_improvement(
        snr_direct_db: float,
        snr_irs_db: float,
        bandwidth_mhz: float
    ) -> Dict[str, float]:
        """
        Calculate channel capacity improvement with IRS.

        Args:
            snr_direct_db: SNR for direct communication in dB
            snr_irs_db: SNR for IRS-assisted communication in dB
            bandwidth_mhz: Bandwidth in MHz

        Returns:
            Dictionary with capacity metrics
        """
        capacity_direct = WirelessMetrics.calculate_data_rate_mbps(bandwidth_mhz, snr_direct_db)
        capacity_irs = WirelessMetrics.calculate_data_rate_mbps(bandwidth_mhz, snr_irs_db)

        improvement_ratio = capacity_irs / capacity_direct if capacity_direct > 0 else float('inf')
        improvement_percent = (improvement_ratio - 1) * 100

        return {
            'capacity_direct_mbps': capacity_direct,
            'capacity_irs_mbps': capacity_irs,
            'improvement_ratio': improvement_ratio,
            'improvement_percent': improvement_percent
        }


class EnergyMetrics:
    """Metrics for evaluating energy efficiency and green communications."""

    @staticmethod
    def calculate_energy_efficiency(
        data_rate_mbps: float,
        power_consumption_w: float
    ) -> float:
        """
        Calculate energy efficiency in bits per joule.

        Args:
            data_rate_mbps: Data rate in Mbps
            power_consumption_w: Power consumption in Watts

        Returns:
            Energy efficiency in bits per joule
        """
        if power_consumption_w <= 0:
            return float('inf')

        data_rate_bps = data_rate_mbps * 1e6
        return data_rate_bps / power_consumption_w

    @staticmethod
    def calculate_power_consumption(
        transmit_power_w: float,
        circuit_power_w: float,
        processing_power_w: float = 0.0
    ) -> float:
        """
        Calculate total power consumption.

        Args:
            transmit_power_w: Transmission power in Watts
            circuit_power_w: Circuit power consumption in Watts
            processing_power_w: Processing power consumption in Watts

        Returns:
            Total power consumption in Watts
        """
        return transmit_power_w + circuit_power_w + processing_power_w

    @staticmethod
    def calculate_green_efficiency_score(
        energy_efficiency: float,
        renewable_energy_ratio: float = 1.0,
        carbon_intensity: float = 0.5  # kg CO2/kWh
    ) -> Dict[str, float]:
        """
        Calculate green efficiency score considering renewable energy.

        Args:
            energy_efficiency: Energy efficiency in bits per joule
            renewable_energy_ratio: Ratio of renewable energy (0-1)
            carbon_intensity: Carbon intensity in kg CO2/kWh

        Returns:
            Dictionary with green efficiency metrics
        """
        # Calculate carbon footprint per bit
        power_per_bit = 1 / energy_efficiency if energy_efficiency > 0 else float('inf')
        carbon_per_bit = power_per_bit * carbon_intensity * (1 - renewable_energy_ratio) / 3.6e6  # Convert to kg CO2/bit

        # Green efficiency score (higher is better)
        green_score = energy_efficiency * renewable_energy_ratio / (1 + carbon_per_bit * 1e9)

        return {
            'energy_efficiency_bits_per_joule': energy_efficiency,
            'carbon_footprint_kg_per_gbit': carbon_per_bit * 1e9,
            'green_efficiency_score': green_score,
            'renewable_ratio': renewable_energy_ratio
        }


class SystemMetrics:
    """System-level performance metrics."""

    @staticmethod
    def calculate_network_throughput(
        individual_rates: List[float],
        aggregation_method: str = 'sum'
    ) -> float:
        """
        Calculate network-wide throughput.

        Args:
            individual_rates: List of individual device data rates
            aggregation_method: Method for aggregation ('sum', 'mean', 'min')

        Returns:
            Network throughput
        """
        if not individual_rates:
            return 0.0

        if aggregation_method == 'sum':
            return sum(individual_rates)
        elif aggregation_method == 'mean':
            return np.mean(individual_rates)
        elif aggregation_method == 'min':
            return min(individual_rates)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    @staticmethod
    def calculate_fairness_index(values: List[float]) -> float:
        """
        Calculate Jain's fairness index.

        Args:
            values: List of values (e.g., data rates, accuracies)

        Returns:
            Fairness index (0-1, where 1 is perfectly fair)
        """
        if not values or len(values) == 1:
            return 1.0

        values = np.array(values)
        numerator = (np.sum(values)) ** 2
        denominator = len(values) * np.sum(values ** 2)

        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def calculate_system_reliability(
        success_rates: List[float],
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate system reliability metrics.

        Args:
            success_rates: List of success rates for different components
            weights: Optional weights for different components

        Returns:
            Dictionary with reliability metrics
        """
        if not success_rates:
            return {'overall_reliability': 0.0, 'min_reliability': 0.0, 'mean_reliability': 0.0}

        success_rates = np.array(success_rates)

        if weights is None:
            weights = np.ones(len(success_rates)) / len(success_rates)
        else:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize

        overall_reliability = np.sum(success_rates * weights)

        return {
            'overall_reliability': float(overall_reliability),
            'min_reliability': float(np.min(success_rates)),
            'mean_reliability': float(np.mean(success_rates)),
            'std_reliability': float(np.std(success_rates))
        }


def generate_performance_report(
    fl_metrics: Dict[str, Any],
    wireless_metrics: Dict[str, Any],
    energy_metrics: Dict[str, Any],
    system_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive performance report.

    Args:
        fl_metrics: Federated learning metrics
        wireless_metrics: Wireless communication metrics
        energy_metrics: Energy efficiency metrics
        system_metrics: System-level metrics

    Returns:
        Comprehensive performance report
    """
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'federated_learning': fl_metrics,
        'wireless_communication': wireless_metrics,
        'energy_efficiency': energy_metrics,
        'system_performance': system_metrics,
        'summary': {}
    }

    # Generate summary metrics
    try:
        report['summary'] = {
            'overall_accuracy': fl_metrics.get('final_accuracy', 0.0),
            'communication_efficiency': fl_metrics.get('efficiency_score', 0.0),
            'energy_efficiency': energy_metrics.get('energy_efficiency_bits_per_joule', 0.0),
            'green_score': energy_metrics.get('green_efficiency_score', 0.0),
            'system_reliability': system_metrics.get('overall_reliability', 0.0),
            'fairness_index': system_metrics.get('fairness_index', 0.0)
        }
    except Exception as e:
        logger.warning(f"Error generating summary metrics: {e}")
        report['summary'] = {'error': str(e)}

    return report


def compare_methods(
    method_results: Dict[str, Dict[str, Any]],
    metrics_to_compare: List[str]
) -> pd.DataFrame:
    """
    Compare performance of different methods.

    Args:
        method_results: Dictionary with method names as keys and results as values
        metrics_to_compare: List of metric names to compare

    Returns:
        DataFrame with comparison results
    """
    comparison_data = {}

    for method_name, results in method_results.items():
        comparison_data[method_name] = {}
        for metric in metrics_to_compare:
            # Navigate nested dictionaries to find the metric
            value = results
            for key in metric.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            comparison_data[method_name][metric] = value

    return pd.DataFrame(comparison_data).T


def calculate_statistical_significance(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two sets of results.

    Args:
        results_a: Results from method A
        results_b: Results from method B
        alpha: Significance level

    Returns:
        Dictionary with statistical test results
    """
    from scipy import stats

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(results_a, results_b)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(results_a) - 1) * np.var(results_a, ddof=1) +
                         (len(results_b) - 1) * np.var(results_b, ddof=1)) /
                        (len(results_a) + len(results_b) - 2))

    cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std if pooled_std > 0 else 0

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'cohens_d': float(cohens_d),
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large',
        'mean_a': float(np.mean(results_a)),
        'mean_b': float(np.mean(results_b)),
        'std_a': float(np.std(results_a)),
        'std_b': float(np.std(results_b))
    }


class IRSMetrics:
    """Metrics specific to IRS optimization and performance."""

    @staticmethod
    def calculate_beamforming_gain(
        irs_channel: np.ndarray,
        phase_shifts: np.ndarray
    ) -> float:
        """
        Calculate beamforming gain from IRS configuration.

        Args:
            irs_channel: IRS channel coefficients
            phase_shifts: Phase shift values for IRS elements

        Returns:
            Beamforming gain
        """
        # Apply phase shifts to channel
        configured_channel = irs_channel * np.exp(1j * phase_shifts)

        # Calculate beamforming gain
        gain = np.abs(np.sum(configured_channel)) ** 2

        return float(gain)

    @staticmethod
    def calculate_irs_efficiency(
        snr_without_irs: float,
        snr_with_irs: float
    ) -> Dict[str, float]:
        """
        Calculate IRS efficiency metrics.

        Args:
            snr_without_irs: SNR without IRS assistance in dB
            snr_with_irs: SNR with IRS assistance in dB

        Returns:
            Dictionary with IRS efficiency metrics
        """
        if snr_without_irs <= -100:  # Very low SNR case
            return {
                'snr_improvement_db': float('inf'),
                'snr_improvement_ratio': float('inf'),
                'snr_without_irs_db': snr_without_irs,
                'snr_with_irs_db': snr_with_irs
            }

        snr_improvement_db = snr_with_irs - snr_without_irs
        snr_improvement_ratio = (10 ** (snr_with_irs / 10)) / (10 ** (snr_without_irs / 10))

        return {
            'snr_improvement_db': float(snr_improvement_db),
            'snr_improvement_ratio': float(snr_improvement_ratio),
            'snr_without_irs_db': float(snr_without_irs),
            'snr_with_irs_db': float(snr_with_irs)
        }

    @staticmethod
    def calculate_phase_shift_stability(
        phase_shifts_history: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate stability of phase shift optimization.

        Args:
            phase_shifts_history: List of phase shift arrays over iterations

        Returns:
            Dictionary with stability metrics
        """
        if len(phase_shifts_history) < 2:
            return {'mean_variation': 0.0, 'max_variation': 0.0, 'convergence_rate': 0.0}

        variations = []
        for i in range(1, len(phase_shifts_history)):
            variation = np.mean(np.abs(phase_shifts_history[i] - phase_shifts_history[i-1]))
            variations.append(variation)

        # Calculate convergence rate
        if len(variations) > 1:
            convergence_rate = (variations[0] - variations[-1]) / len(variations)
        else:
            convergence_rate = 0.0

        return {
            'mean_variation': float(np.mean(variations)),
            'max_variation': float(np.max(variations)),
            'min_variation': float(np.min(variations)),
            'convergence_rate': float(convergence_rate)
        }


class MADRLMetrics:
    """Metrics for Multi-Agent Deep Reinforcement Learning evaluation."""

    @staticmethod
    def calculate_reward_metrics(
        rewards_per_agent: Dict[int, List[float]]
    ) -> Dict[str, Any]:
        """
        Calculate reward-based metrics for MADRL.

        Args:
            rewards_per_agent: Dictionary with agent IDs and their reward histories

        Returns:
            Dictionary with reward metrics
        """
        if not rewards_per_agent:
            return {'total_reward': 0.0, 'mean_reward': 0.0, 'reward_fairness': 0.0}

        all_final_rewards = [rewards[-1] for rewards in rewards_per_agent.values() if rewards]
        all_cumulative_rewards = [sum(rewards) for rewards in rewards_per_agent.values()]

        # Calculate fairness using Jain's index
        fairness = SystemMetrics.calculate_fairness_index(all_final_rewards)

        return {
            'total_reward': float(sum(all_cumulative_rewards)),
            'mean_final_reward': float(np.mean(all_final_rewards)) if all_final_rewards else 0.0,
            'std_final_reward': float(np.std(all_final_rewards)) if all_final_rewards else 0.0,
            'reward_fairness': float(fairness),
            'num_agents': len(rewards_per_agent)
        }

    @staticmethod
    def calculate_convergence_stability(
        convergence_histories: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate convergence stability across multiple training runs.

        Args:
            convergence_histories: List of convergence histories from multiple runs

        Returns:
            Dictionary with stability metrics
        """
        if not convergence_histories:
            return {'mean_final_value': 0.0, 'std_final_value': 0.0, 'stability_score': 0.0}

        final_values = [history[-1] for history in convergence_histories if history]

        if not final_values:
            return {'mean_final_value': 0.0, 'std_final_value': 0.0, 'stability_score': 0.0}

        mean_final = np.mean(final_values)
        std_final = np.std(final_values)

        # Stability score (higher is better, lower coefficient of variation)
        stability_score = 1 / (1 + std_final / mean_final) if mean_final != 0 else 0.0

        return {
            'mean_final_value': float(mean_final),
            'std_final_value': float(std_final),
            'stability_score': float(stability_score),
            'coefficient_of_variation': float(std_final / mean_final) if mean_final != 0 else float('inf')
        }

    @staticmethod
    def calculate_learning_efficiency(
        rewards_history: List[float],
        computation_times: List[float]
    ) -> Dict[str, float]:
        """
        Calculate learning efficiency metrics.

        Args:
            rewards_history: History of rewards over episodes
            computation_times: Computation times for each episode

        Returns:
            Dictionary with efficiency metrics
        """
        if not rewards_history or not computation_times:
            return {'learning_rate': 0.0, 'time_to_convergence': 0.0, 'efficiency_score': 0.0}

        # Calculate learning rate as improvement per episode
        if len(rewards_history) > 1:
            learning_rate = (rewards_history[-1] - rewards_history[0]) / len(rewards_history)
        else:
            learning_rate = 0.0

        # Find time to convergence (when reward stabilizes)
        convergence_threshold = 0.01 * abs(rewards_history[-1]) if rewards_history[-1] != 0 else 0.01
        time_to_convergence = sum(computation_times)  # Default to total time

        for i in range(1, len(rewards_history)):
            if abs(rewards_history[i] - rewards_history[i-1]) < convergence_threshold:
                time_to_convergence = sum(computation_times[:i+1])
                break

        # Efficiency score: reward improvement per unit time
        total_time = sum(computation_times)
        efficiency_score = abs(learning_rate) / (total_time / len(computation_times)) if total_time > 0 else 0.0

        return {
            'learning_rate': float(learning_rate),
            'time_to_convergence': float(time_to_convergence),
            'total_training_time': float(total_time),
            'efficiency_score': float(efficiency_score)
        }


def calculate_comprehensive_system_metrics(
    fl_results: Dict[str, Any],
    wireless_results: Dict[str, Any],
    energy_results: Dict[str, Any],
    irs_results: Optional[Dict[str, Any]] = None,
    madrl_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive system-wide metrics.

    Args:
        fl_results: Federated learning results
        wireless_results: Wireless communication results
        energy_results: Energy efficiency results
        irs_results: IRS optimization results (optional)
        madrl_results: MADRL results (optional)

    Returns:
        Dictionary with comprehensive system metrics
    """
    comprehensive_metrics = {
        'federated_learning': fl_results,
        'wireless_communication': wireless_results,
        'energy_efficiency': energy_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    # Add IRS metrics if available
    if irs_results:
        comprehensive_metrics['irs_optimization'] = irs_results

    # Add MADRL metrics if available
    if madrl_results:
        comprehensive_metrics['madrl_performance'] = madrl_results

    # Calculate overall system score
    try:
        # Normalize and weight different aspects
        fl_score = fl_results.get('final_accuracy', 0.0)
        wireless_score = min(wireless_results.get('improvement_ratio', 1.0) / 2.0, 1.0)  # Cap at 1.0
        energy_score = min(energy_results.get('green_efficiency_score', 0.0) / 1000.0, 1.0)  # Normalize

        # Overall system performance score
        weights = {'fl': 0.4, 'wireless': 0.3, 'energy': 0.3}
        overall_score = (weights['fl'] * fl_score +
                        weights['wireless'] * wireless_score +
                        weights['energy'] * energy_score)

        comprehensive_metrics['overall_performance'] = {
            'system_score': float(overall_score),
            'fl_contribution': float(weights['fl'] * fl_score),
            'wireless_contribution': float(weights['wireless'] * wireless_score),
            'energy_contribution': float(weights['energy'] * energy_score)
        }

    except Exception as e:
        logger.warning(f"Error calculating overall performance score: {e}")
        comprehensive_metrics['overall_performance'] = {'error': str(e)}

    return comprehensive_metrics
