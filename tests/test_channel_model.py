#!/usr/bin/env python3
"""
Unit tests for Channel Model Module

This module contains comprehensive tests for the wireless channel modeling
functionality, including path loss calculations, fading models, SNR
computations, and both direct and IRS-assisted channel characteristics.

Author: Research Team
Date: 2025
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.channel_model import (
    ChannelParams, ChannelModel, PathLossModel, FadingModel
)
from core.network_topology import NetworkTopology, NetworkConfig
from core.exceptions import ChannelModelError, ValidationError, ComputationError


class TestChannelParams(unittest.TestCase):
    """Test cases for ChannelParams class."""
    
    def test_default_parameters(self):
        """Test default channel parameters."""
        params = ChannelParams()
        
        self.assertEqual(params.path_loss_exponent_direct, 3.0)
        self.assertEqual(params.path_loss_exponent_irs, 2.5)
        self.assertEqual(params.rician_k_direct, 5.0)
        self.assertEqual(params.rician_k_irs, 10.0)
        self.assertEqual(params.shadowing_std_direct, 6.0)
        self.assertEqual(params.shadowing_std_irs, 4.0)
        self.assertEqual(params.reference_distance, 1.0)
        self.assertEqual(params.path_loss_at_reference, 30.0)
        self.assertEqual(params.carrier_frequency, 28.0)
        self.assertEqual(params.bandwidth, 100.0)
        self.assertEqual(params.noise_power_dbm, -174.0)
    
    def test_custom_parameters(self):
        """Test custom channel parameters."""
        params = ChannelParams(
            path_loss_exponent_direct=3.5,
            path_loss_exponent_irs=2.8,
            rician_k_direct=8.0,
            rician_k_irs=12.0,
            carrier_frequency=60.0,
            bandwidth=200.0
        )
        
        self.assertEqual(params.path_loss_exponent_direct, 3.5)
        self.assertEqual(params.path_loss_exponent_irs, 2.8)
        self.assertEqual(params.rician_k_direct, 8.0)
        self.assertEqual(params.rician_k_irs, 12.0)
        self.assertEqual(params.carrier_frequency, 60.0)
        self.assertEqual(params.bandwidth, 200.0)
    
    def test_invalid_parameters(self):
        """Test validation of invalid channel parameters."""
        with self.assertRaises(ChannelModelError):
            ChannelParams(path_loss_exponent_direct=-1.0)
        
        with self.assertRaises(ChannelModelError):
            ChannelParams(rician_k_direct=-5.0)
        
        with self.assertRaises(ChannelModelError):
            ChannelParams(carrier_frequency=0.0)
        
        with self.assertRaises(ChannelModelError):
            ChannelParams(bandwidth=-10.0)
        
        with self.assertRaises(ChannelModelError):
            ChannelParams(reference_distance=0.0)


class TestPathLossModel(unittest.TestCase):
    """Test cases for PathLossModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = ChannelParams()
        self.path_loss_model = PathLossModel(self.params)
    
    def test_initialization(self):
        """Test path loss model initialization."""
        self.assertEqual(self.path_loss_model.params, self.params)
        self.assertEqual(self.path_loss_model.reference_distance, 1.0)
        self.assertEqual(self.path_loss_model.path_loss_at_reference, 30.0)
    
    def test_direct_path_loss_calculation(self):
        """Test direct path loss calculation."""
        # Test at reference distance
        path_loss_db = self.path_loss_model.calculate_direct_path_loss(1.0)
        self.assertAlmostEqual(path_loss_db, 30.0, places=5)
        
        # Test at 10 meters
        path_loss_db = self.path_loss_model.calculate_direct_path_loss(10.0)
        expected = 30.0 + 10 * 3.0 * np.log10(10.0)  # 30 + 30*log10(10) = 60 dB
        self.assertAlmostEqual(path_loss_db, expected, places=5)
        
        # Test at 100 meters
        path_loss_db = self.path_loss_model.calculate_direct_path_loss(100.0)
        expected = 30.0 + 10 * 3.0 * np.log10(100.0)  # 30 + 60 = 90 dB
        self.assertAlmostEqual(path_loss_db, expected, places=5)
    
    def test_irs_path_loss_calculation(self):
        """Test IRS-assisted path loss calculation."""
        # Test single hop (BS to IRS or IRS to device)
        path_loss_db = self.path_loss_model.calculate_irs_path_loss(10.0)
        expected = 30.0 + 10 * 2.5 * np.log10(10.0)  # 30 + 25 = 55 dB
        self.assertAlmostEqual(path_loss_db, expected, places=5)
        
        # Test two-hop path loss (BS -> IRS -> Device)
        bs_irs_distance = 50.0
        irs_device_distance = 20.0
        total_path_loss = self.path_loss_model.calculate_two_hop_path_loss(
            bs_irs_distance, irs_device_distance
        )
        
        expected_bs_irs = 30.0 + 10 * 2.5 * np.log10(50.0)
        expected_irs_device = 30.0 + 10 * 2.5 * np.log10(20.0)
        expected_total = expected_bs_irs + expected_irs_device
        
        self.assertAlmostEqual(total_path_loss, expected_total, places=5)
    
    def test_free_space_path_loss(self):
        """Test free space path loss calculation."""
        # Test at 1 km with 28 GHz
        distance_m = 1000.0
        frequency_ghz = 28.0
        
        path_loss_db = self.path_loss_model.calculate_free_space_path_loss(
            distance_m, frequency_ghz
        )
        
        # Free space path loss formula: 20*log10(4*pi*d*f/c)
        c = 3e8  # Speed of light
        expected = 20 * np.log10(4 * np.pi * distance_m * frequency_ghz * 1e9 / c)
        
        self.assertAlmostEqual(path_loss_db, expected, places=3)
    
    def test_invalid_distances(self):
        """Test error handling for invalid distances."""
        with self.assertRaises(ChannelModelError):
            self.path_loss_model.calculate_direct_path_loss(0.0)
        
        with self.assertRaises(ChannelModelError):
            self.path_loss_model.calculate_direct_path_loss(-5.0)
        
        with self.assertRaises(ChannelModelError):
            self.path_loss_model.calculate_irs_path_loss(np.nan)
        
        with self.assertRaises(ChannelModelError):
            self.path_loss_model.calculate_two_hop_path_loss(10.0, 0.0)


class TestFadingModel(unittest.TestCase):
    """Test cases for FadingModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = ChannelParams()
        self.fading_model = FadingModel(self.params)
    
    def test_initialization(self):
        """Test fading model initialization."""
        self.assertEqual(self.fading_model.params, self.params)
        self.assertEqual(self.fading_model.rician_k_direct_linear, 10**(5.0/10))
        self.assertEqual(self.fading_model.rician_k_irs_linear, 10**(10.0/10))
    
    def test_rician_fading_generation(self):
        """Test Rician fading coefficient generation."""
        # Test direct channel
        num_samples = 1000
        fading_coeffs = self.fading_model.generate_rician_fading(
            num_samples, channel_type='direct'
        )
        
        self.assertEqual(len(fading_coeffs), num_samples)
        self.assertTrue(np.all(np.isfinite(fading_coeffs)))
        
        # Test IRS channel
        fading_coeffs = self.fading_model.generate_rician_fading(
            num_samples, channel_type='irs'
        )
        
        self.assertEqual(len(fading_coeffs), num_samples)
        self.assertTrue(np.all(np.isfinite(fading_coeffs)))
    
    def test_rician_fading_statistics(self):
        """Test Rician fading statistical properties."""
        num_samples = 10000
        k_factor_db = 10.0
        k_factor_linear = 10**(k_factor_db/10)
        
        # Generate large number of samples
        fading_coeffs = self.fading_model.generate_rician_fading(
            num_samples, k_factor_db=k_factor_db
        )
        
        # Calculate power (magnitude squared)
        powers = np.abs(fading_coeffs)**2
        
        # For Rician fading, mean power should be approximately 1
        mean_power = np.mean(powers)
        self.assertAlmostEqual(mean_power, 1.0, delta=0.1)
        
        # Standard deviation should be related to K-factor
        std_power = np.std(powers)
        self.assertGreater(std_power, 0)
    
    def test_shadowing_generation(self):
        """Test log-normal shadowing generation."""
        num_samples = 1000
        
        # Test direct channel shadowing
        shadowing_db = self.fading_model.generate_shadowing(
            num_samples, channel_type='direct'
        )
        
        self.assertEqual(len(shadowing_db), num_samples)
        self.assertTrue(np.all(np.isfinite(shadowing_db)))
        
        # Mean should be approximately 0
        self.assertAlmostEqual(np.mean(shadowing_db), 0.0, delta=0.5)
        
        # Standard deviation should match parameter
        self.assertAlmostEqual(np.std(shadowing_db), 6.0, delta=0.5)
    
    def test_composite_fading_generation(self):
        """Test composite fading (Rician + shadowing) generation."""
        num_samples = 1000
        
        composite_coeffs = self.fading_model.generate_composite_fading(
            num_samples, channel_type='direct'
        )
        
        self.assertEqual(len(composite_coeffs), num_samples)
        self.assertTrue(np.all(np.isfinite(composite_coeffs)))
        self.assertTrue(np.all(composite_coeffs > 0))  # Should be positive (magnitude)
    
    def test_invalid_channel_type(self):
        """Test error handling for invalid channel type."""
        with self.assertRaises(ChannelModelError):
            self.fading_model.generate_rician_fading(100, channel_type='invalid')
        
        with self.assertRaises(ChannelModelError):
            self.fading_model.generate_shadowing(100, channel_type='unknown')
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with self.assertRaises(ChannelModelError):
            self.fading_model.generate_rician_fading(0)  # Zero samples
        
        with self.assertRaises(ChannelModelError):
            self.fading_model.generate_rician_fading(-10)  # Negative samples


class TestChannelModel(unittest.TestCase):
    """Test cases for ChannelModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create network topology
        self.network_config = NetworkConfig(
            num_iot_devices=5,
            num_irs_elements=32,
            area_size=50.0
        )
        self.network = NetworkTopology(self.network_config)
        
        # Create channel parameters
        self.channel_params = ChannelParams()
        
        # Create channel model
        self.channel_model = ChannelModel(self.network, self.channel_params)
    
    def test_initialization(self):
        """Test channel model initialization."""
        self.assertEqual(self.channel_model.network, self.network)
        self.assertEqual(self.channel_model.params, self.channel_params)
        self.assertIsInstance(self.channel_model.path_loss_model, PathLossModel)
        self.assertIsInstance(self.channel_model.fading_model, FadingModel)
    
    def test_initialization_with_condition(self):
        """Test channel model initialization with channel condition."""
        # Test different channel conditions
        conditions = ['good', 'medium', 'bad']
        
        for condition in conditions:
            channel_model = ChannelModel(self.network, condition=condition)
            self.assertIsNotNone(channel_model.params)
            self.assertIsInstance(channel_model.path_loss_model, PathLossModel)
    
    def test_direct_channel_coefficients(self):
        """Test direct channel coefficient calculation."""
        coefficients = self.channel_model.get_direct_channel_coefficients()
        
        # Should have one coefficient per IoT device
        self.assertEqual(len(coefficients), len(self.network.iot_devices))
        
        # All coefficients should be complex and finite
        self.assertTrue(np.all(np.isfinite(coefficients)))
        self.assertTrue(np.iscomplexobj(coefficients))
        
        # Magnitudes should be positive
        magnitudes = np.abs(coefficients)
        self.assertTrue(np.all(magnitudes > 0))
    
    def test_irs_channel_coefficients(self):
        """Test IRS channel coefficient calculation."""
        # BS to IRS coefficients
        bs_irs_coeffs = self.channel_model.get_bs_to_irs_coefficients()
        self.assertEqual(len(bs_irs_coeffs), self.network.irs.num_elements)
        self.assertTrue(np.all(np.isfinite(bs_irs_coeffs)))
        
        # IRS to devices coefficients
        irs_device_coeffs = self.channel_model.get_irs_to_devices_coefficients()
        expected_shape = (len(self.network.iot_devices), self.network.irs.num_elements)
        self.assertEqual(irs_device_coeffs.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(irs_device_coeffs)))
    
    def test_effective_channel_coefficients(self):
        """Test effective channel coefficient calculation with IRS."""
        effective_coeffs = self.channel_model.get_effective_channel_coefficients()
        
        # Should have one coefficient per IoT device
        self.assertEqual(len(effective_coeffs), len(self.network.iot_devices))
        self.assertTrue(np.all(np.isfinite(effective_coeffs)))
        self.assertTrue(np.iscomplexobj(effective_coeffs))
    
    def test_snr_calculations(self):
        """Test SNR calculations."""
        # Direct channel SNR
        direct_snr = self.channel_model.calculate_direct_snr()
        self.assertEqual(len(direct_snr), len(self.network.iot_devices))
        self.assertTrue(np.all(direct_snr > 0))
        self.assertTrue(np.all(np.isfinite(direct_snr)))
        
        # IRS-assisted SNR
        irs_snr = self.channel_model.calculate_irs_assisted_snr()
        self.assertEqual(len(irs_snr), len(self.network.iot_devices))
        self.assertTrue(np.all(irs_snr > 0))
        self.assertTrue(np.all(np.isfinite(irs_snr)))
    
    def test_data_rate_calculations(self):
        """Test data rate calculations."""
        # Direct channel data rates
        direct_rates = self.channel_model.calculate_direct_data_rates()
        self.assertEqual(len(direct_rates), len(self.network.iot_devices))
        self.assertTrue(np.all(direct_rates > 0))
        self.assertTrue(np.all(np.isfinite(direct_rates)))
        
        # IRS-assisted data rates
        irs_rates = self.channel_model.calculate_irs_assisted_data_rates()
        self.assertEqual(len(irs_rates), len(self.network.iot_devices))
        self.assertTrue(np.all(irs_rates > 0))
        self.assertTrue(np.all(np.isfinite(irs_rates)))
    
    def test_channel_gain_calculations(self):
        """Test channel gain calculations."""
        # Direct channel gains
        direct_gains = self.channel_model.calculate_direct_channel_gains()
        self.assertEqual(len(direct_gains), len(self.network.iot_devices))
        self.assertTrue(np.all(direct_gains > 0))
        
        # IRS-assisted channel gains
        irs_gains = self.channel_model.calculate_irs_assisted_channel_gains()
        self.assertEqual(len(irs_gains), len(self.network.iot_devices))
        self.assertTrue(np.all(irs_gains > 0))
    
    def test_channel_capacity_calculations(self):
        """Test channel capacity calculations using Shannon formula."""
        # Direct channel capacity
        direct_capacity = self.channel_model.calculate_direct_channel_capacity()
        self.assertEqual(len(direct_capacity), len(self.network.iot_devices))
        self.assertTrue(np.all(direct_capacity > 0))
        
        # IRS-assisted channel capacity
        irs_capacity = self.channel_model.calculate_irs_assisted_channel_capacity()
        self.assertEqual(len(irs_capacity), len(self.network.iot_devices))
        self.assertTrue(np.all(irs_capacity > 0))
    
    def test_channel_update(self):
        """Test channel coefficient updates."""
        # Get initial coefficients
        initial_direct = self.channel_model.get_direct_channel_coefficients()
        initial_effective = self.channel_model.get_effective_channel_coefficients()
        
        # Update channels
        self.channel_model.update_channel_coefficients()
        
        # Get new coefficients
        new_direct = self.channel_model.get_direct_channel_coefficients()
        new_effective = self.channel_model.get_effective_channel_coefficients()
        
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(initial_direct, new_direct))
        self.assertFalse(np.array_equal(initial_effective, new_effective))
    
    def test_irs_configuration_impact(self):
        """Test impact of IRS configuration on channel performance."""
        # Calculate initial performance
        initial_snr = self.channel_model.calculate_irs_assisted_snr()
        initial_rates = self.channel_model.calculate_irs_assisted_data_rates()
        
        # Change IRS configuration
        new_phases = np.random.uniform(0, 2*np.pi, self.network.irs.num_elements)
        self.network.update_irs_configuration(new_phases)
        
        # Calculate new performance
        new_snr = self.channel_model.calculate_irs_assisted_snr()
        new_rates = self.channel_model.calculate_irs_assisted_data_rates()
        
        # Performance should change
        self.assertFalse(np.array_equal(initial_snr, new_snr))
        self.assertFalse(np.array_equal(initial_rates, new_rates))
    
    def test_channel_statistics(self):
        """Test channel statistics calculation."""
        stats = self.channel_model.get_channel_statistics()
        
        # Check required statistics
        required_stats = [
            'avg_direct_snr_db', 'avg_irs_snr_db',
            'avg_direct_rate_bps', 'avg_irs_rate_bps',
            'min_direct_snr_db', 'max_direct_snr_db',
            'min_irs_snr_db', 'max_irs_snr_db',
            'sum_direct_rate_bps', 'sum_irs_rate_bps'
        ]
        
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsInstance(stats[stat], (int, float))
            self.assertTrue(np.isfinite(stats[stat]))
        
        # Check logical relationships
        self.assertGreaterEqual(stats['max_direct_snr_db'], stats['min_direct_snr_db'])
        self.assertGreaterEqual(stats['max_irs_snr_db'], stats['min_irs_snr_db'])
        self.assertGreater(stats['sum_direct_rate_bps'], 0)
        self.assertGreater(stats['sum_irs_rate_bps'], 0)
    
    def test_different_channel_conditions(self):
        """Test channel model with different channel conditions."""
        conditions = ['good', 'medium', 'bad']
        results = {}
        
        for condition in conditions:
            channel_model = ChannelModel(self.network, condition=condition)
            snr = channel_model.calculate_direct_snr()
            results[condition] = np.mean(snr)
        
        # Good conditions should generally have higher SNR than bad conditions
        self.assertGreaterEqual(results['good'], results['bad'])
    
    def test_noise_power_calculation(self):
        """Test noise power calculation."""
        noise_power_linear = self.channel_model.get_noise_power_linear()
        
        # Convert back to dBm and compare
        noise_power_dbm = 10 * np.log10(noise_power_linear * 1000)  # Convert to dBm
        expected_dbm = self.channel_params.noise_power_dbm + 10 * np.log10(self.channel_params.bandwidth * 1e6)
        
        self.assertAlmostEqual(noise_power_dbm, expected_dbm, places=3)
    
    def test_invalid_network(self):
        """Test error handling for invalid network."""
        with self.assertRaises(ChannelModelError):
            ChannelModel(None, self.channel_params)
    
    def test_invalid_channel_condition(self):
        """Test error handling for invalid channel condition."""
        with self.assertRaises(ChannelModelError):
            ChannelModel(self.network, condition='invalid_condition')


class TestChannelModelIntegration(unittest.TestCase):
    """Integration tests for channel model with network topology."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create larger network for more comprehensive testing
        self.network_config = NetworkConfig(
            num_iot_devices=20,
            num_irs_elements=100,
            area_size=100.0
        )
        self.network = NetworkTopology(self.network_config)
        self.channel_model = ChannelModel(self.network)
    
    def test_end_to_end_channel_modeling(self):
        """Test complete end-to-end channel modeling process."""
        # Calculate all channel metrics
        direct_snr = self.channel_model.calculate_direct_snr()
        irs_snr = self.channel_model.calculate_irs_assisted_snr()
        direct_rates = self.channel_model.calculate_direct_data_rates()
        irs_rates = self.channel_model.calculate_irs_assisted_data_rates()
        
        # Verify all calculations completed successfully
        self.assertEqual(len(direct_snr), 20)
        self.assertEqual(len(irs_snr), 20)
        self.assertEqual(len(direct_rates), 20)
        self.assertEqual(len(irs_rates), 20)
        
        # All values should be positive and finite
        self.assertTrue(np.all(direct_snr > 0))
        self.assertTrue(np.all(irs_snr > 0))
        self.assertTrue(np.all(direct_rates > 0))
        self.assertTrue(np.all(irs_rates > 0))
        self.assertTrue(np.all(np.isfinite(direct_snr)))
        self.assertTrue(np.all(np.isfinite(irs_snr)))
        self.assertTrue(np.all(np.isfinite(direct_rates)))
        self.assertTrue(np.all(np.isfinite(irs_rates)))
    
    def test_irs_optimization_impact(self):
        """Test impact of IRS optimization on channel performance."""
        # Calculate baseline performance with random IRS configuration
        self.network.reset_irs_configuration()
        baseline_rates = self.channel_model.calculate_irs_assisted_data_rates()
        baseline_sum_rate = np.sum(baseline_rates)
        
        # Try multiple IRS configurations and find the best
        best_sum_rate = baseline_sum_rate
        best_phases = self.network.irs.get_phase_shifts().copy()
        
        for _ in range(10):
            # Try random configuration
            test_phases = np.random.uniform(0, 2*np.pi, self.network.irs.num_elements)
            self.network.update_irs_configuration(test_phases)
            
            # Calculate performance
            test_rates = self.channel_model.calculate_irs_assisted_data_rates()
            test_sum_rate = np.sum(test_rates)
            
            # Keep track of best configuration
            if test_sum_rate > best_sum_rate:
                best_sum_rate = test_sum_rate
                best_phases = test_phases.copy()
        
        # Apply best configuration and verify improvement
        self.network.update_irs_configuration(best_phases)
        final_rates = self.channel_model.calculate_irs_assisted_data_rates()
        final_sum_rate = np.sum(final_rates)
        
        self.assertGreaterEqual(final_sum_rate, baseline_sum_rate)
    
    def test_channel_correlation_with_distance(self):
        """Test correlation between channel quality and distance."""
        # Get distances and SNR values
        distances_to_bs = self.network.get_distances_to_bs()
        direct_snr = self.channel_model.calculate_direct_snr()
        
        # Convert SNR to dB for analysis
        direct_snr_db = 10 * np.log10(direct_snr)
        
        # Generally, closer devices should have higher SNR
        # Find closest and farthest devices
        closest_idx = np.argmin(distances_to_bs)
        farthest_idx = np.argmax(distances_to_bs)
        
        # This is a statistical test, so we'll check if the trend holds on average
        # by comparing the mean SNR of the closest half vs farthest half
        sorted_indices = np.argsort(distances_to_bs)
        half_point = len(sorted_indices) // 2
        
        close_devices = sorted_indices[:half_point]
        far_devices = sorted_indices[half_point:]
        
        avg_snr_close = np.mean(direct_snr_db[close_devices])
        avg_snr_far = np.mean(direct_snr_db[far_devices])
        
        # Closer devices should generally have higher SNR
        self.assertGreater(avg_snr_close, avg_snr_far)
    
    def test_performance_comparison_direct_vs_irs(self):
        """Test performance comparison between direct and IRS-assisted channels."""
        # Calculate performance for both channel types
        direct_rates = self.channel_model.calculate_direct_data_rates()
        irs_rates = self.channel_model.calculate_irs_assisted_data_rates()
        
        direct_sum_rate = np.sum(direct_rates)
        irs_sum_rate = np.sum(irs_rates)
        
        # IRS should generally improve performance (though not guaranteed for random config)
        # At minimum, both should be positive and finite
        self.assertGreater(direct_sum_rate, 0)
        self.assertGreater(irs_sum_rate, 0)
        self.assertTrue(np.isfinite(direct_sum_rate))
        self.assertTrue(np.isfinite(irs_sum_rate))
        
        # Calculate improvement ratio
        improvement_ratio = irs_sum_rate / direct_sum_rate
        self.assertGreater(improvement_ratio, 0)  # Should be positive
    
    def test_channel_model_scalability(self):
        """Test channel model scalability with different network sizes."""
        network_sizes = [5, 10, 20, 50]
        
        for num_devices in network_sizes:
            # Create network of different size
            config = NetworkConfig(
                num_iot_devices=num_devices,
                num_irs_elements=64
            )
            network = NetworkTopology(config)
            channel_model = ChannelModel(network)
            
            # Calculate performance metrics
            direct_snr = channel_model.calculate_direct_snr()
            irs_snr = channel_model.calculate_irs_assisted_snr()
            
            # Verify correct dimensions
            self.assertEqual(len(direct_snr), num_devices)
            self.assertEqual(len(irs_snr), num_devices)
            
            # Verify all values are valid
            self.assertTrue(np.all(direct_snr > 0))
            self.assertTrue(np.all(irs_snr > 0))
            self.assertTrue(np.all(np.isfinite(direct_snr)))
            self.assertTrue(np.all(np.isfinite(irs_snr)))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)