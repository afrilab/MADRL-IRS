#!/usr/bin/env python3
"""
Unit tests for Network Topology Module

This module contains comprehensive tests for the network topology functionality,
including network configuration, base station modeling, IoT device management,
and IRS positioning with 3D distance calculations.

Author: Research Team
Date: 2025
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import (
    NetworkConfig, NetworkTopology, BaseStation, IoTDevice, 
    IntelligentReflectingSurface
)
from core.exceptions import NetworkConfigurationError, ValidationError, ComputationError


class TestNetworkConfig(unittest.TestCase):
    """Test cases for NetworkConfig class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = NetworkConfig()
        
        self.assertEqual(config.num_iot_devices, 20)
        self.assertEqual(config.num_irs_elements, 100)
        self.assertEqual(config.area_size, 100.0)
        self.assertEqual(config.bs_position, (50.0, 50.0, 10.0))
        self.assertEqual(config.irs_position, (100.0, 50.0, 5.0))
        self.assertEqual(config.iot_height, 1.5)
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = NetworkConfig(
            num_iot_devices=50,
            num_irs_elements=200,
            area_size=200.0,
            bs_position=(25.0, 25.0, 15.0),
            irs_position=(150.0, 75.0, 8.0),
            iot_height=2.0
        )
        
        self.assertEqual(config.num_iot_devices, 50)
        self.assertEqual(config.num_irs_elements, 200)
        self.assertEqual(config.area_size, 200.0)
        self.assertEqual(config.bs_position, (25.0, 25.0, 15.0))
        self.assertEqual(config.irs_position, (150.0, 75.0, 8.0))
        self.assertEqual(config.iot_height, 2.0)
    
    def test_invalid_num_iot_devices(self):
        """Test validation of invalid number of IoT devices."""
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_iot_devices=0)
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_iot_devices=-5)
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_iot_devices=2000)  # Too many
    
    def test_invalid_num_irs_elements(self):
        """Test validation of invalid number of IRS elements."""
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_irs_elements=0)
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_irs_elements=-10)
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(num_irs_elements=20000)  # Too many
    
    def test_invalid_area_size(self):
        """Test validation of invalid area size."""
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(area_size=0.0)
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(area_size=-50.0)
    
    def test_invalid_positions(self):
        """Test validation of invalid positions."""
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(bs_position=(50.0, 50.0))  # Wrong dimensions
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(irs_position=(100.0, np.nan, 5.0))  # NaN value
        
        with self.assertRaises(NetworkConfigurationError):
            NetworkConfig(bs_position=(50.0, 50.0, -5.0))  # Negative height


class TestBaseStation(unittest.TestCase):
    """Test cases for BaseStation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bs_position = (50.0, 50.0, 10.0)
        self.bs = BaseStation(self.bs_position)
    
    def test_initialization(self):
        """Test base station initialization."""
        self.assertEqual(self.bs.x, 50.0)
        self.assertEqual(self.bs.y, 50.0)
        self.assertEqual(self.bs.z, 10.0)
        self.assertEqual(self.bs.position, self.bs_position)
        self.assertEqual(self.bs.transmit_power, 30.0)  # Default power
        self.assertEqual(self.bs.antenna_gain, 15.0)  # Default gain
    
    def test_custom_parameters(self):
        """Test base station with custom parameters."""
        bs = BaseStation(
            position=(25.0, 75.0, 15.0),
            transmit_power=40.0,
            antenna_gain=20.0
        )
        
        self.assertEqual(bs.position, (25.0, 75.0, 15.0))
        self.assertEqual(bs.transmit_power, 40.0)
        self.assertEqual(bs.antenna_gain, 20.0)
    
    def test_distance_calculation(self):
        """Test distance calculation to other points."""
        # Distance to origin
        distance = self.bs.distance_to(np.array([0.0, 0.0, 0.0]))
        expected = np.sqrt(50**2 + 50**2 + 10**2)
        self.assertAlmostEqual(distance, expected, places=10)
        
        # Distance to same point
        distance = self.bs.distance_to(np.array([50.0, 50.0, 10.0]))
        self.assertAlmostEqual(distance, 0.0, places=10)
        
        # Distance to point on same horizontal plane
        distance = self.bs.distance_to(np.array([53.0, 54.0, 10.0]))
        expected = np.sqrt(3**2 + 4**2)  # 3-4-5 triangle
        self.assertAlmostEqual(distance, 5.0, places=10)
    
    def test_invalid_distance_calculation(self):
        """Test error handling in distance calculation."""
        with self.assertRaises(ComputationError):
            self.bs.distance_to(np.array([1.0, 2.0]))  # Wrong dimensions
        
        with self.assertRaises(ComputationError):
            self.bs.distance_to(np.array([np.nan, 2.0, 3.0]))  # NaN value
        
        with self.assertRaises(ComputationError):
            self.bs.distance_to(np.array([np.inf, 2.0, 3.0]))  # Infinite value
    
    def test_invalid_initialization(self):
        """Test error handling in base station initialization."""
        with self.assertRaises(NetworkConfigurationError):
            BaseStation((50.0, 50.0))  # Wrong dimensions
        
        with self.assertRaises(NetworkConfigurationError):
            BaseStation((50.0, 50.0, np.nan))  # NaN value
        
        with self.assertRaises(NetworkConfigurationError):
            BaseStation((50.0, 50.0, 10.0), transmit_power=-5.0)  # Negative power


class TestIoTDevice(unittest.TestCase):
    """Test cases for IoTDevice class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device_id = 5
        self.position = (25.0, 75.0, 1.5)
        self.device = IoTDevice(self.device_id, self.position)
    
    def test_initialization(self):
        """Test IoT device initialization."""
        self.assertEqual(self.device.device_id, self.device_id)
        self.assertEqual(self.device.position, self.position)
        self.assertEqual(self.device.x, 25.0)
        self.assertEqual(self.device.y, 75.0)
        self.assertEqual(self.device.z, 1.5)
        self.assertEqual(self.device.transmit_power, 20.0)  # Default power
        self.assertEqual(self.device.antenna_gain, 0.0)  # Default gain
    
    def test_custom_parameters(self):
        """Test IoT device with custom parameters."""
        device = IoTDevice(
            device_id=10,
            position=(10.0, 20.0, 2.0),
            transmit_power=15.0,
            antenna_gain=3.0
        )
        
        self.assertEqual(device.device_id, 10)
        self.assertEqual(device.position, (10.0, 20.0, 2.0))
        self.assertEqual(device.transmit_power, 15.0)
        self.assertEqual(device.antenna_gain, 3.0)
    
    def test_distance_calculation(self):
        """Test distance calculation from IoT device."""
        # Distance to origin
        distance = self.device.distance_to(np.array([0.0, 0.0, 0.0]))
        expected = np.sqrt(25**2 + 75**2 + 1.5**2)
        self.assertAlmostEqual(distance, expected, places=10)
        
        # Distance to base station at (50, 50, 10)
        distance = self.device.distance_to(np.array([50.0, 50.0, 10.0]))
        expected = np.sqrt((50-25)**2 + (50-75)**2 + (10-1.5)**2)
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_invalid_initialization(self):
        """Test error handling in IoT device initialization."""
        with self.assertRaises(NetworkConfigurationError):
            IoTDevice(-1, (10.0, 20.0, 1.5))  # Invalid device ID
        
        with self.assertRaises(NetworkConfigurationError):
            IoTDevice(5, (10.0, 20.0))  # Wrong position dimensions
        
        with self.assertRaises(NetworkConfigurationError):
            IoTDevice(5, (10.0, 20.0, 1.5), transmit_power=0.0)  # Zero power


class TestIntelligentReflectingSurface(unittest.TestCase):
    """Test cases for IntelligentReflectingSurface class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position = (100.0, 50.0, 5.0)
        self.num_elements = 64
        self.irs = IntelligentReflectingSurface(self.position, self.num_elements)
    
    def test_initialization(self):
        """Test IRS initialization."""
        self.assertEqual(self.irs.position, self.position)
        self.assertEqual(self.irs.num_elements, self.num_elements)
        self.assertEqual(self.irs.x, 100.0)
        self.assertEqual(self.irs.y, 50.0)
        self.assertEqual(self.irs.z, 5.0)
        self.assertEqual(len(self.irs.phase_shifts), self.num_elements)
        self.assertTrue(np.all(self.irs.phase_shifts >= 0))
        self.assertTrue(np.all(self.irs.phase_shifts < 2*np.pi))
    
    def test_custom_parameters(self):
        """Test IRS with custom parameters."""
        irs = IntelligentReflectingSurface(
            position=(80.0, 60.0, 8.0),
            num_elements=128,
            element_spacing=0.5,
            surface_area=2.0
        )
        
        self.assertEqual(irs.position, (80.0, 60.0, 8.0))
        self.assertEqual(irs.num_elements, 128)
        self.assertEqual(irs.element_spacing, 0.5)
        self.assertEqual(irs.surface_area, 2.0)
    
    def test_phase_shift_operations(self):
        """Test phase shift operations."""
        # Set specific phase shifts
        test_phases = np.linspace(0, 2*np.pi, self.num_elements, endpoint=False)
        self.irs.set_phase_shifts(test_phases)
        
        np.testing.assert_array_almost_equal(
            self.irs.get_phase_shifts(), test_phases
        )
        
        # Set uniform phase shifts
        uniform_phase = np.pi/4
        self.irs.set_uniform_phase_shifts(uniform_phase)
        expected = np.full(self.num_elements, uniform_phase)
        
        np.testing.assert_array_almost_equal(
            self.irs.get_phase_shifts(), expected
        )
    
    def test_reflection_coefficients(self):
        """Test reflection coefficient calculation."""
        test_phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        irs = IntelligentReflectingSurface((0, 0, 0), 4)
        irs.set_phase_shifts(test_phases)
        
        coefficients = irs.get_reflection_coefficients()
        expected = np.exp(1j * test_phases)
        
        np.testing.assert_array_almost_equal(coefficients, expected)
    
    def test_randomize_phase_shifts(self):
        """Test phase shift randomization."""
        original_phases = self.irs.get_phase_shifts().copy()
        self.irs.randomize_phase_shifts()
        new_phases = self.irs.get_phase_shifts()
        
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(original_phases, new_phases))
        
        # Should still be valid
        self.assertTrue(np.all(new_phases >= 0))
        self.assertTrue(np.all(new_phases < 2*np.pi))
    
    def test_distance_calculation(self):
        """Test distance calculation from IRS."""
        # Distance to origin
        distance = self.irs.distance_to(np.array([0.0, 0.0, 0.0]))
        expected = np.sqrt(100**2 + 50**2 + 5**2)
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_invalid_initialization(self):
        """Test error handling in IRS initialization."""
        with self.assertRaises(NetworkConfigurationError):
            IntelligentReflectingSurface((100.0, 50.0), 64)  # Wrong position dimensions
        
        with self.assertRaises(NetworkConfigurationError):
            IntelligentReflectingSurface((100.0, 50.0, 5.0), 0)  # Zero elements
        
        with self.assertRaises(NetworkConfigurationError):
            IntelligentReflectingSurface((100.0, 50.0, 5.0), 64, element_spacing=-0.1)  # Negative spacing
    
    def test_invalid_phase_shift_operations(self):
        """Test error handling in phase shift operations."""
        with self.assertRaises(NetworkConfigurationError):
            self.irs.set_phase_shifts(np.array([1, 2, 3]))  # Wrong size
        
        with self.assertRaises(NetworkConfigurationError):
            self.irs.set_phase_shifts(np.array([np.nan] * self.num_elements))  # NaN values


class TestNetworkTopology(unittest.TestCase):
    """Test cases for NetworkTopology class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NetworkConfig(
            num_iot_devices=10,
            num_irs_elements=50,
            area_size=50.0
        )
        self.network = NetworkTopology(self.config)
    
    def test_initialization_with_config(self):
        """Test network topology initialization with configuration."""
        self.assertEqual(len(self.network.iot_devices), 10)
        self.assertEqual(self.network.irs.num_elements, 50)
        self.assertIsInstance(self.network.base_station, BaseStation)
        self.assertIsInstance(self.network.irs, IntelligentReflectingSurface)
    
    def test_default_initialization(self):
        """Test network topology initialization with default configuration."""
        network = NetworkTopology()
        
        self.assertEqual(len(network.iot_devices), 20)  # Default value
        self.assertEqual(network.irs.num_elements, 100)  # Default value
        self.assertIsInstance(network.base_station, BaseStation)
        self.assertIsInstance(network.irs, IntelligentReflectingSurface)
    
    def test_iot_device_positioning(self):
        """Test IoT device positioning within area."""
        for device in self.network.iot_devices:
            self.assertGreaterEqual(device.x, 0)
            self.assertLessEqual(device.x, self.config.area_size)
            self.assertGreaterEqual(device.y, 0)
            self.assertLessEqual(device.y, self.config.area_size)
            self.assertEqual(device.z, self.config.iot_height)
    
    def test_device_id_uniqueness(self):
        """Test that all IoT devices have unique IDs."""
        device_ids = [device.device_id for device in self.network.iot_devices]
        self.assertEqual(len(device_ids), len(set(device_ids)))
        self.assertEqual(min(device_ids), 0)
        self.assertEqual(max(device_ids), len(self.network.iot_devices) - 1)
    
    def test_get_device_positions(self):
        """Test getting all device positions."""
        positions = self.network.get_device_positions()
        
        self.assertEqual(positions.shape, (len(self.network.iot_devices), 3))
        
        # Check that positions match individual devices
        for i, device in enumerate(self.network.iot_devices):
            np.testing.assert_array_equal(positions[i], [device.x, device.y, device.z])
    
    def test_get_distances_to_bs(self):
        """Test distance calculations to base station."""
        distances = self.network.get_distances_to_bs()
        
        self.assertEqual(len(distances), len(self.network.iot_devices))
        self.assertTrue(np.all(distances > 0))
        
        # Verify distances manually for first device
        device = self.network.iot_devices[0]
        expected_distance = device.distance_to(np.array(self.network.base_station.position))
        self.assertAlmostEqual(distances[0], expected_distance, places=10)
    
    def test_get_distances_to_irs(self):
        """Test distance calculations to IRS."""
        distances = self.network.get_distances_to_irs()
        
        self.assertEqual(len(distances), len(self.network.iot_devices))
        self.assertTrue(np.all(distances > 0))
        
        # Verify distances manually for first device
        device = self.network.iot_devices[0]
        expected_distance = device.distance_to(np.array(self.network.irs.position))
        self.assertAlmostEqual(distances[0], expected_distance, places=10)
    
    def test_get_bs_to_irs_distance(self):
        """Test base station to IRS distance calculation."""
        distance = self.network.get_bs_to_irs_distance()
        
        expected = self.network.base_station.distance_to(np.array(self.network.irs.position))
        self.assertAlmostEqual(distance, expected, places=10)
    
    def test_network_statistics(self):
        """Test network statistics calculation."""
        stats = self.network.get_network_statistics()
        
        self.assertIn('num_iot_devices', stats)
        self.assertIn('num_irs_elements', stats)
        self.assertIn('area_size', stats)
        self.assertIn('avg_distance_to_bs', stats)
        self.assertIn('avg_distance_to_irs', stats)
        self.assertIn('bs_to_irs_distance', stats)
        
        self.assertEqual(stats['num_iot_devices'], len(self.network.iot_devices))
        self.assertEqual(stats['num_irs_elements'], self.network.irs.num_elements)
        self.assertGreater(stats['avg_distance_to_bs'], 0)
        self.assertGreater(stats['avg_distance_to_irs'], 0)
    
    def test_update_irs_configuration(self):
        """Test updating IRS configuration."""
        new_phases = np.random.uniform(0, 2*np.pi, self.network.irs.num_elements)
        self.network.update_irs_configuration(new_phases)
        
        np.testing.assert_array_almost_equal(
            self.network.irs.get_phase_shifts(), new_phases
        )
    
    def test_reset_irs_configuration(self):
        """Test resetting IRS configuration."""
        # Set specific configuration
        test_phases = np.ones(self.network.irs.num_elements) * np.pi/2
        self.network.update_irs_configuration(test_phases)
        
        # Reset to random
        self.network.reset_irs_configuration()
        new_phases = self.network.irs.get_phase_shifts()
        
        # Should be different from test phases
        self.assertFalse(np.array_equal(new_phases, test_phases))
        
        # Should be valid
        self.assertTrue(np.all(new_phases >= 0))
        self.assertTrue(np.all(new_phases < 2*np.pi))
    
    def test_invalid_irs_update(self):
        """Test error handling in IRS configuration update."""
        with self.assertRaises(NetworkConfigurationError):
            # Wrong number of phase shifts
            self.network.update_irs_configuration(np.array([1, 2, 3]))
        
        with self.assertRaises(NetworkConfigurationError):
            # NaN values
            bad_phases = np.full(self.network.irs.num_elements, np.nan)
            self.network.update_irs_configuration(bad_phases)


class TestNetworkTopologyIntegration(unittest.TestCase):
    """Integration tests for network topology components."""
    
    def test_realistic_network_setup(self):
        """Test realistic network setup with proper scaling."""
        config = NetworkConfig(
            num_iot_devices=100,
            num_irs_elements=256,
            area_size=500.0,
            bs_position=(250.0, 250.0, 25.0),
            irs_position=(400.0, 250.0, 15.0)
        )
        
        network = NetworkTopology(config)
        
        # Verify network was created successfully
        self.assertEqual(len(network.iot_devices), 100)
        self.assertEqual(network.irs.num_elements, 256)
        
        # Check that devices are properly distributed
        positions = network.get_device_positions()
        self.assertTrue(np.all(positions[:, 0] >= 0))
        self.assertTrue(np.all(positions[:, 0] <= 500.0))
        self.assertTrue(np.all(positions[:, 1] >= 0))
        self.assertTrue(np.all(positions[:, 1] <= 500.0))
    
    def test_distance_calculations_consistency(self):
        """Test consistency of distance calculations across methods."""
        network = NetworkTopology()
        
        # Get distances using different methods
        bs_distances = network.get_distances_to_bs()
        irs_distances = network.get_distances_to_irs()
        
        # Manually calculate distances for verification
        for i, device in enumerate(network.iot_devices):
            manual_bs_distance = device.distance_to(np.array(network.base_station.position))
            manual_irs_distance = device.distance_to(np.array(network.irs.position))
            
            self.assertAlmostEqual(bs_distances[i], manual_bs_distance, places=10)
            self.assertAlmostEqual(irs_distances[i], manual_irs_distance, places=10)
    
    def test_network_reconfiguration(self):
        """Test network reconfiguration capabilities."""
        network = NetworkTopology()
        
        # Store original configuration
        original_phases = network.irs.get_phase_shifts().copy()
        
        # Reconfigure multiple times
        for _ in range(5):
            new_phases = np.random.uniform(0, 2*np.pi, network.irs.num_elements)
            network.update_irs_configuration(new_phases)
            
            # Verify configuration was applied
            current_phases = network.irs.get_phase_shifts()
            np.testing.assert_array_almost_equal(current_phases, new_phases)
        
        # Reset and verify it's different
        network.reset_irs_configuration()
        final_phases = network.irs.get_phase_shifts()
        self.assertFalse(np.array_equal(final_phases, original_phases))


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)