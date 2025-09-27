#!/usr/bin/env python3
"""
Wireless Channel Model Module for 6G IoT Networks

This module implements the wireless channel characteristics described in the paper,
including path loss modeling, small-scale fading (Rician), large-scale fading
(shadowing), and both direct and IRS-assisted channels.

Author: Research Team
Date: 2025
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from .network_topology import NetworkTopology
from .exceptions import ChannelModelError, ValidationError, ComputationError
from ..utils.validation import (
    validate_positive_number, validate_float_range, validate_string_choice,
    validate_array_shape, validate_channel_params
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class ChannelParams:
    """Channel parameters for 6G mmWave communications."""
    # Path loss exponents
    path_loss_exponent_direct: float = 3.0
    path_loss_exponent_irs: float = 2.5

    # Rician K-factors (dB)
    rician_k_direct: float = 5.0
    rician_k_irs: float = 10.0

    # Shadowing standard deviations (dB)
    shadowing_std_direct: float = 6.0
    shadowing_std_irs: float = 4.0

    # System parameters
    reference_distance: float = 1.0  # meters
    path_loss_at_reference: float = 30.0  # dB
    carrier_frequency: float = 28.0  # GHz
    bandwidth: float = 100.0  # MHz
    noise_power_dbm: float = -174.0  # dBm/Hz

    def __post_init__(self):
        """Validate channel parameters after initialization."""
        try:
            # Validate path loss exponents
            self.path_loss_exponent_direct = validate_float_range(
                self.path_loss_exponent_direct, 'path_loss_exponent_direct',
                min_value=1.0, max_value=6.0
            )
            self.path_loss_exponent_irs = validate_float_range(
                self.path_loss_exponent_irs, 'path_loss_exponent_irs',
                min_value=1.0, max_value=6.0
            )

            # Validate Rician K-factors
            self.rician_k_direct = validate_positive_number(
                self.rician_k_direct, 'rician_k_direct', allow_zero=True
            )
            self.rician_k_irs = validate_positive_number(
                self.rician_k_irs, 'rician_k_irs', allow_zero=True
            )

            # Validate shadowing standard deviations
            self.shadowing_std_direct = validate_positive_number(
                self.shadowing_std_direct, 'shadowing_std_direct'
            )
            self.shadowing_std_irs = validate_positive_number(
                self.shadowing_std_irs, 'shadowing_std_irs'
            )

            # Validate system parameters
            self.reference_distance = validate_positive_number(
                self.reference_distance, 'reference_distance'
            )
            self.path_loss_at_reference = validate_positive_number(
                self.path_loss_at_reference, 'path_loss_at_reference'
            )
            self.carrier_frequency = validate_float_range(
                self.carrier_frequency, 'carrier_frequency',
                min_value=0.1, max_value=300.0
            )
            self.bandwidth = validate_positive_number(
                self.bandwidth, 'bandwidth'
            )

            logger.debug(f"ChannelParams validated: {self}")

        except ValidationError as e:
            logger.error(f"ChannelParams validation failed: {e}")
            raise ChannelModelError(
                f"Invalid channel parameters: {e.message}",
                operation="parameter_validation"
            ) from e


class PathLossModel:
    """
    Path loss model for 6G mmWave communications.

    Implements the path loss calculation according to the system model
    described in the paper.
    """

    def __init__(self, params: ChannelParams):
        """
        Initialize the path loss model.

        Args:
            params: Channel parameters
        """
        self.params = params

    def calculate_path_loss_db(self, distance: float, channel_type: str = 'direct') -> float:
        """
        Calculate path loss in dB.

        Args:
            distance: Distance in meters
            channel_type: 'direct' or 'irs'

        Returns:
            Path loss in dB
        """
        if channel_type == 'direct':
            exponent = self.params.path_loss_exponent_direct
        else:
            exponent = self.params.path_loss_exponent_irs

        path_loss_db = (self.params.path_loss_at_reference +
                       10 * exponent * np.log10(distance / self.params.reference_distance))

        return path_loss_db

    def calculate_path_loss_linear(self, distance: float, channel_type: str = 'direct') -> float:
        """
        Calculate path loss in linear scale.

        Args:
            distance: Distance in meters
            channel_type: 'direct' or 'irs'

        Returns:
            Path loss in linear scale
        """
        path_loss_db = self.calculate_path_loss_db(distance, channel_type)
        return 10 ** (-path_loss_db / 10)


class FadingModel:
    """
    Fading model implementing both large-scale and small-scale fading.

    Implements Rician fading for small-scale effects and log-normal
    shadowing for large-scale effects.
    """

    def __init__(self, params: ChannelParams):
        """
        Initialize the fading model.

        Args:
            params: Channel parameters
        """
        self.params = params

    def generate_shadowing_db(self, channel_type: str = 'direct') -> float:
        """
        Generate log-normal shadowing in dB.

        Args:
            channel_type: 'direct' or 'irs'

        Returns:
            Shadowing value in dB
        """
        if channel_type == 'direct':
            std = self.params.shadowing_std_direct
        else:
            std = self.params.shadowing_std_irs

        return np.random.normal(0, std)

    def generate_rician_fading(self, k_factor_db: float) -> complex:
        """
        Generate Rician fading coefficient.

        Args:
            k_factor_db: Rician K-factor in dB

        Returns:
            Complex fading coefficient
        """
        k_linear = 10 ** (k_factor_db / 10)

        # Line-of-sight component
        los_component = np.sqrt(k_linear / (k_linear + 1))

        # Non-line-of-sight component
        nlos_real = np.random.normal(0, 1)
        nlos_imag = np.random.normal(0, 1)
        nlos_component = (np.sqrt(1 / (k_linear + 1)) *
                         (nlos_real + 1j * nlos_imag) / np.sqrt(2))

        return los_component + nlos_component

    def generate_small_scale_fading(self, channel_type: str = 'direct') -> complex:
        """
        Generate small-scale fading coefficient.

        Args:
            channel_type: 'direct' or 'irs'

        Returns:
            Complex small-scale fading coefficient
        """
        if channel_type == 'direct':
            k_factor = self.params.rician_k_direct
        else:
            k_factor = self.params.rician_k_irs

        return self.generate_rician_fading(k_factor)


class ChannelModel:
    """
    Complete channel model implementing the wireless channel characteristics.

    This class implements the channel model from Section III-B of the paper,
    including path loss modeling, small-scale fading (Rician), large-scale
    fading (shadowing), and both direct and IRS-assisted channels.
    """

    def __init__(self, network: NetworkTopology, params: Optional[ChannelParams] = None,
                 condition: str = 'medium'):
        """
        Initialize the channel model.

        Args:
            network: Network topology
            params: Channel parameters (if None, will be set based on condition)
            condition: Channel condition ('good', 'medium', 'bad')

        Raises:
            ChannelModelError: If initialization fails
        """
        try:
            with PerformanceLogger(logger, "ChannelModel initialization"):
                # Validate inputs
                if not isinstance(network, NetworkTopology):
                    raise ValidationError(
                        "network must be a NetworkTopology instance",
                        parameter='network',
                        expected_type='NetworkTopology',
                        actual_value=type(network).__name__
                    )

                condition = validate_string_choice(
                    condition, 'condition', ['good', 'medium', 'bad'], case_sensitive=False
                )

                self.network = network
                self.condition = condition.lower()

                logger.info(f"Initializing channel model with condition: {self.condition}")

                # Set channel parameters based on condition if not provided
                if params is None:
                    self.params = self._get_params_for_condition(self.condition)
                else:
                    self.params = params

                # Initialize models
                self.path_loss_model = PathLossModel(self.params)
                self.fading_model = FadingModel(self.params)

                # Calculate noise power
                self.noise_power = self._calculate_noise_power()

                # Initialize channel coefficients
                self._initialize_channels()

                logger.info(f"Channel model initialized successfully for "
                           f"{len(self.network.iot_devices)} devices")

        except ValidationError as e:
            logger.error(f"Channel model validation failed: {e}")
            raise ChannelModelError(
                f"Invalid channel model parameters: {e.message}",
                operation="initialization"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize channel model: {e}")
            log_exception(logger, e, "ChannelModel initialization")
            raise ChannelModelError(
                f"Failed to initialize channel model: {str(e)}",
                operation="initialization"
            ) from e

    def _get_params_for_condition(self, condition: str) -> ChannelParams:
        """Get channel parameters based on channel condition."""
        if condition == 'good':
            return ChannelParams(
                path_loss_exponent_direct=2.0,
                path_loss_exponent_irs=2.2,
                rician_k_direct=10.0,
                rician_k_irs=15.0,
                shadowing_std_direct=4.0,
                shadowing_std_irs=3.0
            )
        elif condition == 'medium':
            return ChannelParams(
                path_loss_exponent_direct=3.0,
                path_loss_exponent_irs=2.5,
                rician_k_direct=5.0,
                rician_k_irs=10.0,
                shadowing_std_direct=6.0,
                shadowing_std_irs=4.0
            )
        else:  # bad
            return ChannelParams(
                path_loss_exponent_direct=3.5,
                path_loss_exponent_irs=2.8,
                rician_k_direct=2.0,
                rician_k_irs=5.0,
                shadowing_std_direct=8.0,
                shadowing_std_irs=5.0
            )

    def _calculate_noise_power(self) -> float:
        """Calculate noise power in watts."""
        noise_power_dbm = (self.params.noise_power_dbm +
                          10 * np.log10(self.params.bandwidth * 1e6))
        return 10 ** (noise_power_dbm / 10) * 1e-3

    def _initialize_channels(self) -> None:
        """Initialize all channel coefficients."""
        num_iot_devices = len(self.network.iot_devices)
        num_irs_elements = self.network.irs.num_elements

        # Get distances
        bs_iot_distances = self.network.get_bs_iot_distances()
        irs_iot_distances = self.network.get_irs_iot_distances()
        bs_irs_distance = self.network.get_bs_irs_distance()

        # Initialize channel matrices
        self.h_direct = np.zeros((num_iot_devices, 1), dtype=complex)
        self.h_irs_iot = np.zeros((num_iot_devices, num_irs_elements), dtype=complex)
        self.h_bs_irs = np.zeros((1, num_irs_elements), dtype=complex)

        # Generate direct channels (BS to IoT devices)
        for i in range(num_iot_devices):
            self.h_direct[i, 0] = self._generate_channel_coefficient(
                bs_iot_distances[i], 'direct'
            )

        # Generate IRS-IoT channels
        for i in range(num_iot_devices):
            for j in range(num_irs_elements):
                self.h_irs_iot[i, j] = self._generate_channel_coefficient(
                    irs_iot_distances[i], 'irs'
                )

        # Generate BS-IRS channels
        for j in range(num_irs_elements):
            self.h_bs_irs[0, j] = self._generate_channel_coefficient(
                bs_irs_distance, 'irs'
            )

    def _generate_channel_coefficient(self, distance: float, channel_type: str) -> complex:
        """
        Generate a single channel coefficient.

        Args:
            distance: Distance in meters
            channel_type: 'direct' or 'irs'

        Returns:
            Complex channel coefficient
        """
        # Large-scale fading (path loss + shadowing)
        path_loss_linear = self.path_loss_model.calculate_path_loss_linear(
            distance, channel_type
        )
        shadowing_db = self.fading_model.generate_shadowing_db(channel_type)
        shadowing_linear = 10 ** (shadowing_db / 10)
        large_scale_fading = path_loss_linear * shadowing_linear

        # Small-scale fading
        small_scale_fading = self.fading_model.generate_small_scale_fading(channel_type)

        # Complete channel coefficient
        return np.sqrt(large_scale_fading) * small_scale_fading

    def update_with_irs(self, irs_config: Optional[np.ndarray] = None) -> None:
        """
        Update effective channel with IRS configuration.

        Implements equation (1) from the paper:
        h_eff_k = h_k + sum(f_n * theta_n * g_k,n)

        Args:
            irs_config: IRS phase shift configuration (if None, uses current IRS config)
        """
        if irs_config is None:
            irs_config = self.network.irs.get_configuration()

        num_iot_devices = len(self.network.iot_devices)
        self.h_effective = np.zeros_like(self.h_direct)

        for i in range(num_iot_devices):
            # Direct channel component
            self.h_effective[i, 0] = self.h_direct[i, 0]

            # IRS-assisted channel component
            irs_contribution = 0
            for j in range(self.network.irs.num_elements):
                irs_contribution += (self.h_bs_irs[0, j] *
                                   irs_config[j] *
                                   self.h_irs_iot[i, j])

            self.h_effective[i, 0] += irs_contribution

    def get_direct_channel(self) -> np.ndarray:
        """Get direct channel coefficients."""
        return self.h_direct.copy()

    def get_irs_iot_channel(self) -> np.ndarray:
        """Get IRS-IoT channel coefficients."""
        return self.h_irs_iot.copy()

    def get_bs_irs_channel(self) -> np.ndarray:
        """Get BS-IRS channel coefficients."""
        return self.h_bs_irs.copy()

    def get_effective_channel(self) -> np.ndarray:
        """Get the effective channel (direct + IRS-assisted)."""
        if hasattr(self, 'h_effective'):
            return self.h_effective.copy()
        else:
            return self.h_direct.copy()

    def get_channel_gain(self) -> np.ndarray:
        """Get channel power gain |h_eff|^2."""
        h_effective = self.get_effective_channel()
        return np.abs(h_effective) ** 2

    def calculate_snr(self, transmit_power: float) -> np.ndarray:
        """
        Calculate SNR for each IoT device.

        Args:
            transmit_power: Transmit power in watts

        Returns:
            SNR array for all IoT devices
        """
        channel_gain = self.get_channel_gain()
        snr = transmit_power * channel_gain.flatten() / self.noise_power
        return snr

    def calculate_data_rate(self, transmit_power: float) -> np.ndarray:
        """
        Calculate achievable data rate using Shannon capacity.

        Args:
            transmit_power: Transmit power in watts

        Returns:
            Data rate array for all IoT devices (bps)
        """
        snr = self.calculate_snr(transmit_power)
        data_rate = self.params.bandwidth * 1e6 * np.log2(1 + snr)
        return data_rate

    def calculate_spectral_efficiency(self, transmit_power: float) -> np.ndarray:
        """
        Calculate spectral efficiency.

        Args:
            transmit_power: Transmit power in watts

        Returns:
            Spectral efficiency array (bps/Hz)
        """
        snr = self.calculate_snr(transmit_power)
        return np.log2(1 + snr)

    def regenerate_channels(self) -> None:
        """Regenerate all channel coefficients (for time-varying channels)."""
        self._initialize_channels()

    def get_channel_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information about the channels.

        Returns:
            Dictionary with channel statistics
        """
        direct_gains = np.abs(self.h_direct.flatten()) ** 2

        stats = {
            'direct_channel_stats': {
                'mean_gain_db': 10 * np.log10(np.mean(direct_gains)),
                'std_gain_db': 10 * np.log10(np.std(direct_gains)),
                'min_gain_db': 10 * np.log10(np.min(direct_gains)),
                'max_gain_db': 10 * np.log10(np.max(direct_gains))
            },
            'noise_power_dbm': 10 * np.log10(self.noise_power * 1000),
            'channel_condition': self.condition,
            'num_iot_devices': len(self.network.iot_devices),
            'num_irs_elements': self.network.irs.num_elements
        }

        if hasattr(self, 'h_effective'):
            effective_gains = np.abs(self.h_effective.flatten()) ** 2
            stats['effective_channel_stats'] = {
                'mean_gain_db': 10 * np.log10(np.mean(effective_gains)),
                'std_gain_db': 10 * np.log10(np.std(effective_gains)),
                'min_gain_db': 10 * np.log10(np.min(effective_gains)),
                'max_gain_db': 10 * np.log10(np.max(effective_gains))
            }

        return stats

    def __repr__(self) -> str:
        return (f"ChannelModel(condition='{self.condition}', "
                f"devices={len(self.network.iot_devices)}, "
                f"irs_elements={self.network.irs.num_elements})")
