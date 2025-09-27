#!/usr/bin/env python3
"""
Unit tests for IRS Optimization Module

This module contains comprehensive tests for the IRS optimization functionality,
including phase shift control, gradient-based optimization, and performance
evaluation metrics.

Author: Research Team
Date: 2025
"""

import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.network_topology import NetworkTopology, NetworkConfig
from core.channel_model import ChannelModel, ChannelParams
from core.irs_optimization import IRSOptimizer, PhaseShiftController, IRSOptimizationConfig


class TestPhaseShiftController(unittest.TestCase):
    """Test cases for PhaseShiftController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_elements = 64
        self.controller = PhaseShiftController(self.num_elements)
    
    def test_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.num_elements, self.num_elements)
        self.assertEqual(len(self.controller.phase_shifts), self.num_elements)
        self.assertTrue(np.all(self.controller.phase_shifts >= 0))
        self.assertTrue(np.all(self.controller.phase_shifts < 2*np.pi))
    
    def test_set_phase_shifts(self):
        """Test setting phase shifts."""
        test_phases = np.linspace(0, 2*np.pi, self.num_elements)
        self.controller.set_phase_shifts(test_phases)
        
        np.testing.assert_array_almost_equal(
            self.controller.get_phase_shifts(), test_phases
        )
    
    def test_phase_shift_constraints(self):
        """Test phase shift constraints."""
        # Test with phases outside [0, 2π]
        test_phases = np.array([3*np.pi, -np.pi, 4*np.pi, -2*np.pi])
        controller = PhaseShiftController(4)
        controller.set_phase_shifts(test_phases)
        
        constrained_phases = controller.get_phase_shifts()
        self.assertTrue(np.all(constrained_phases >= 0))
        self.assertTrue(np.all(constrained_phases < 2*np.pi))
    
    def test_reflection_coefficients(self):
        """Test reflection coefficient calculation."""
        test_phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        controller = PhaseShiftController(4)
        controller.set_phase_shifts(test_phases)
        
        coefficients = controller.get_reflection_coefficients()
        expected = np.exp(1j * test_phases)
        
        np.testing.assert_array_almost_equal(coefficients, expected)
    
    def test_uniform_phase_shifts(self):
        """Test setting uniform phase shifts."""
        test_phase = np.pi/4
        self.controller.set_uniform_phase_shifts(test_phase)
        
        expected = np.full(self.num_elements, test_phase)
        np.testing.assert_array_almost_equal(
            self.controller.get_phase_shifts(), expected
        )
    
    def test_phase_history(self):
        """Test phase shift history tracking."""
        initial_length = len(self.controller.get_phase_history())
        
        # Make some changes
        self.controller.randomize_phase_shifts()
        self.controller.set_uniform_phase_shifts(0)
        
        history = self.controller.get_phase_history()
        self.assertEqual(len(history), initial_length + 2)
        
        # Clear history
        self.controller.clear_history()
        self.assertEqual(len(self.controller.get_phase_history()), 0)


class TestIRSOptimizer(unittest.TestCase):
    """Test cases for IRSOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create network topology
        self.network_config = NetworkConfig(
            num_iot_devices=5,
            num_irs_elements=32,
            area_size=50.0
        )
        self.network = NetworkTopology(self.network_config)
        
        # Create channel model
        self.channel_params = ChannelParams()
        self.channel_model = ChannelModel(self.network, self.channel_params)
        
        # Create optimizer
        self.opt_config = IRSOptimizationConfig(
            max_iterations=50,
            learning_rate=0.1,
            convergence_threshold=1e-4
        )
        self.optimizer = IRSOptimizer(self.network, self.channel_model, self.opt_config)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.network, self.network)
        self.assertEqual(self.optimizer.channel_model, self.channel_model)
        self.assertEqual(self.optimizer.config, self.opt_config)
        self.assertIsInstance(self.optimizer.phase_controller, PhaseShiftController)
    
    def test_sum_rate_objective(self):
        """Test sum rate objective function."""
        test_phases = np.random.uniform(0, 2*np.pi, self.network_config.num_irs_elements)
        
        objective_value, gradient = self.optimizer._sum_rate_objective(test_phases)
        
        self.assertIsInstance(objective_value, float)
        self.assertGreater(objective_value, 0)
        self.assertEqual(len(gradient), len(test_phases))
        self.assertTrue(np.all(np.isfinite(gradient)))
    
    def test_min_rate_objective(self):
        """Test minimum rate objective function."""
        test_phases = np.random.uniform(0, 2*np.pi, self.network_config.num_irs_elements)
        
        objective_value, gradient = self.optimizer._min_rate_objective(test_phases)
        
        self.assertIsInstance(objective_value, float)
        self.assertGreater(objective_value, 0)
        self.assertEqual(len(gradient), len(test_phases))
        self.assertTrue(np.all(np.isfinite(gradient)))
    
    def test_energy_efficiency_objective(self):
        """Test energy efficiency objective function."""
        test_phases = np.random.uniform(0, 2*np.pi, self.network_config.num_irs_elements)
        
        objective_value, gradient = self.optimizer._energy_efficiency_objective(test_phases)
        
        self.assertIsInstance(objective_value, float)
        self.assertGreater(objective_value, 0)
        self.assertEqual(len(gradient), len(test_phases))
        self.assertTrue(np.all(np.isfinite(gradient)))
    
    def test_optimization_convergence(self):
        """Test optimization convergence."""
        # Use a simple configuration for faster convergence
        simple_config = IRSOptimizationConfig(
            max_iterations=20,
            learning_rate=0.05,
            convergence_threshold=1e-3,
            optimization_objective='sum_rate'
        )
        
        optimizer = IRSOptimizer(self.network, self.channel_model, simple_config)
        results = optimizer.optimize()
        
        self.assertIn('best_objective_value', results)
        self.assertIn('best_phase_shifts', results)
        self.assertIn('final_metrics', results)
        self.assertIn('iterations', results)
        self.assertIn('converged', results)
        
        self.assertGreater(results['best_objective_value'], 0)
        self.assertEqual(len(results['best_phase_shifts']), self.network_config.num_irs_elements)
        self.assertLessEqual(results['iterations'], simple_config.max_iterations)
    
    def test_performance_evaluation(self):
        """Test performance evaluation metrics."""
        metrics = self.optimizer.evaluate_performance()
        
        # Check required metrics
        required_metrics = [
            'sum_rate_bps', 'sum_rate_mbps', 'average_rate_bps',
            'min_rate_bps', 'max_rate_bps', 'rate_std_bps',
            'average_snr_db', 'min_snr_db', 'max_snr_db',
            'average_spectral_efficiency', 'sum_spectral_efficiency',
            'energy_efficiency_bps_per_watt', 'jains_fairness_index'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(np.isfinite(metrics[metric]))
        
        # Check logical relationships
        self.assertGreaterEqual(metrics['max_rate_bps'], metrics['min_rate_bps'])
        self.assertGreaterEqual(metrics['max_snr_db'], metrics['min_snr_db'])
        self.assertGreaterEqual(metrics['jains_fairness_index'], 0)
        self.assertLessEqual(metrics['jains_fairness_index'], 1)
    
    def test_baseline_comparison(self):
        """Test baseline comparison functionality."""
        # Test different baseline types
        baseline_types = ['random', 'uniform', 'no_irs']
        
        for baseline_type in baseline_types:
            comparison = self.optimizer.compare_with_baseline(baseline_type)
            
            self.assertIn('current_metrics', comparison)
            self.assertIn('baseline_metrics', comparison)
            self.assertIn('improvements', comparison)
            self.assertIn('baseline_type', comparison)
            
            self.assertEqual(comparison['baseline_type'], baseline_type)
            
            # Check that improvements are calculated
            improvements = comparison['improvements']
            self.assertGreater(len(improvements), 0)
    
    def test_optimization_with_different_objectives(self):
        """Test optimization with different objective functions."""
        objectives = ['sum_rate', 'min_rate', 'energy_efficiency']
        
        for objective in objectives:
            config = IRSOptimizationConfig(
                max_iterations=10,
                optimization_objective=objective
            )
            
            optimizer = IRSOptimizer(self.network, self.channel_model, config)
            results = optimizer.optimize()
            
            self.assertGreater(results['best_objective_value'], 0)
            self.assertIsNotNone(results['best_phase_shifts'])
    
    def test_momentum_optimization(self):
        """Test optimization with momentum."""
        config_with_momentum = IRSOptimizationConfig(
            max_iterations=20,
            use_momentum=True,
            momentum_factor=0.9
        )
        
        config_without_momentum = IRSOptimizationConfig(
            max_iterations=20,
            use_momentum=False
        )
        
        # Test both configurations
        for config in [config_with_momentum, config_without_momentum]:
            optimizer = IRSOptimizer(self.network, self.channel_model, config)
            results = optimizer.optimize()
            
            self.assertGreater(results['best_objective_value'], 0)
    
    def test_gradient_calculation(self):
        """Test gradient calculation using finite differences."""
        test_phases = np.random.uniform(0, 2*np.pi, self.network_config.num_irs_elements)
        
        # Test gradient calculation for sum rate
        gradient = self.optimizer._calculate_gradient_finite_diff(
            test_phases, self.optimizer._sum_rate_value
        )
        
        self.assertEqual(len(gradient), len(test_phases))
        self.assertTrue(np.all(np.isfinite(gradient)))
    
    def test_reset_optimization(self):
        """Test optimization reset functionality."""
        # Run optimization to populate history
        self.optimizer.optimize()
        
        # Check that history exists
        self.assertGreater(len(self.optimizer.get_optimization_history()), 0)
        
        # Reset and check
        self.optimizer.reset_optimization()
        self.assertEqual(len(self.optimizer.get_optimization_history()), 0)
        self.assertEqual(len(self.optimizer.phase_controller.get_phase_history()), 0)


class TestIRSOptimizationIntegration(unittest.TestCase):
    """Integration tests for IRS optimization with network and channel models."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a more realistic network
        self.network_config = NetworkConfig(
            num_iot_devices=10,
            num_irs_elements=64,
            area_size=100.0
        )
        self.network = NetworkTopology(self.network_config)
        
        # Create channel model with different conditions
        self.channel_params = ChannelParams()
        self.channel_model = ChannelModel(self.network, self.channel_params, 'medium')
        
        # Create optimizer
        self.opt_config = IRSOptimizationConfig(
            max_iterations=100,
            learning_rate=0.01,
            convergence_threshold=1e-5
        )
        self.optimizer = IRSOptimizer(self.network, self.channel_model, self.opt_config)
    
    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization process."""
        # Initial performance
        initial_metrics = self.optimizer.evaluate_performance()
        initial_sum_rate = initial_metrics['sum_rate_bps']
        
        # Run optimization
        results = self.optimizer.optimize()
        
        # Final performance
        final_metrics = self.optimizer.evaluate_performance()
        final_sum_rate = final_metrics['sum_rate_bps']
        
        # Check improvement
        self.assertGreaterEqual(final_sum_rate, initial_sum_rate)
        self.assertGreater(results['best_objective_value'], 0)
    
    def test_optimization_with_different_channel_conditions(self):
        """Test optimization under different channel conditions."""
        conditions = ['good', 'medium', 'bad']
        results = {}
        
        for condition in conditions:
            # Create channel model for this condition
            channel_model = ChannelModel(self.network, condition=condition)
            optimizer = IRSOptimizer(self.network, channel_model, self.opt_config)
            
            # Run optimization
            opt_results = optimizer.optimize()
            results[condition] = opt_results['best_objective_value']
        
        # Good conditions should generally perform better
        self.assertGreaterEqual(results['good'], results['bad'])
    
    def test_scalability_with_network_size(self):
        """Test optimization scalability with different network sizes."""
        network_sizes = [5, 10, 15]
        
        for num_devices in network_sizes:
            # Create network of different sizes
            config = NetworkConfig(
                num_iot_devices=num_devices,
                num_irs_elements=32
            )
            network = NetworkTopology(config)
            channel_model = ChannelModel(network)
            
            # Create optimizer with reduced iterations for speed
            opt_config = IRSOptimizationConfig(max_iterations=20)
            optimizer = IRSOptimizer(network, channel_model, opt_config)
            
            # Run optimization
            results = optimizer.optimize()
            
            # Should complete successfully
            self.assertGreater(results['best_objective_value'], 0)
            self.assertIsNotNone(results['best_phase_shifts'])


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)