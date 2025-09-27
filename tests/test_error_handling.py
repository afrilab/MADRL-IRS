#!/usr/bin/env python3
"""
Test Error Handling and Logging Implementation

This module tests the comprehensive error handling and logging system
implemented throughout the research framework.

Author: Research Team
Date: 2025
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import modules to test
from src.core.exceptions import (
    ResearchFrameworkError, NetworkConfigurationError, ChannelModelError,
    FederatedLearningError, DatasetError, ValidationError, ConfigurationError,
    ComputationError, ResourceError
)
from src.utils.validation import (
    validate_positive_number, validate_integer_range, validate_float_range,
    validate_string_choice, validate_position_3d, validate_file_path,
    validate_network_config, validate_channel_params, validate_fl_config
)
from src.utils.logging_config import (
    get_logger, setup_default_logging, log_exception, log_performance_metrics,
    PerformanceLogger
)
from src.core.network_topology import NetworkConfig, NetworkTopology, BaseStation, IoTDevice
from src.core.channel_model import ChannelParams


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_base_exception(self):
        """Test base ResearchFrameworkError."""
        error = ResearchFrameworkError(
            "Test error", 
            error_code="TEST_001", 
            details={"param": "value"}
        )
        
        assert str(error) == "[TEST_001] Test error"
        assert error.error_code == "TEST_001"
        assert error.details == {"param": "value"}
    
    def test_network_configuration_error(self):
        """Test NetworkConfigurationError."""
        error = NetworkConfigurationError(
            "Invalid position", 
            parameter="position", 
            value=(1, 2, 3)
        )
        
        assert "Invalid position" in str(error)
        assert error.parameter == "position"
        assert error.value == (1, 2, 3)
    
    def test_channel_model_error(self):
        """Test ChannelModelError."""
        error = ChannelModelError(
            "Channel calculation failed",
            channel_type="direct",
            operation="path_loss"
        )
        
        assert error.channel_type == "direct"
        assert error.operation == "path_loss"
    
    def test_federated_learning_error(self):
        """Test FederatedLearningError."""
        error = FederatedLearningError(
            "Training failed",
            device_id=5,
            round_number=10
        )
        
        assert error.device_id == 5
        assert error.round_number == 10


class TestValidationFunctions:
    """Test input validation functions."""
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid cases
        assert validate_positive_number(5.0, "test") == 5.0
        assert validate_positive_number(1, "test") == 1
        assert validate_positive_number(0, "test", allow_zero=True) == 0
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_positive_number(-1, "test")
        
        with pytest.raises(ValidationError):
            validate_positive_number(0, "test", allow_zero=False)
        
        with pytest.raises(ValidationError):
            validate_positive_number("not_a_number", "test")
        
        with pytest.raises(ValidationError):
            validate_positive_number(np.nan, "test")
    
    def test_validate_integer_range(self):
        """Test integer range validation."""
        # Valid cases
        assert validate_integer_range(5, "test", min_value=1, max_value=10) == 5
        assert validate_integer_range(1, "test", min_value=1) == 1
        assert validate_integer_range(10, "test", max_value=10) == 10
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_integer_range(0, "test", min_value=1)
        
        with pytest.raises(ValidationError):
            validate_integer_range(11, "test", max_value=10)
        
        with pytest.raises(ValidationError):
            validate_integer_range(5.5, "test")
    
    def test_validate_float_range(self):
        """Test float range validation."""
        # Valid cases
        assert validate_float_range(5.5, "test", min_value=1.0, max_value=10.0) == 5.5
        assert validate_float_range(1.0, "test", min_value=1.0, inclusive_min=True) == 1.0
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_float_range(0.5, "test", min_value=1.0)
        
        with pytest.raises(ValidationError):
            validate_float_range(1.0, "test", min_value=1.0, inclusive_min=False)
        
        with pytest.raises(ValidationError):
            validate_float_range(np.inf, "test")
    
    def test_validate_string_choice(self):
        """Test string choice validation."""
        # Valid cases
        assert validate_string_choice("option1", "test", ["option1", "option2"]) == "option1"
        assert validate_string_choice("OPTION1", "test", ["option1", "option2"], case_sensitive=False) == "OPTION1"
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_string_choice("option3", "test", ["option1", "option2"])
        
        with pytest.raises(ValidationError):
            validate_string_choice(123, "test", ["option1", "option2"])
    
    def test_validate_position_3d(self):
        """Test 3D position validation."""
        # Valid cases
        assert validate_position_3d((1.0, 2.0, 3.0), "test") == (1.0, 2.0, 3.0)
        assert validate_position_3d([1, 2, 3], "test") == (1.0, 2.0, 3.0)
        
        # Invalid cases
        with pytest.raises(ValidationError):
            validate_position_3d((1, 2), "test")  # Wrong length
        
        with pytest.raises(ValidationError):
            validate_position_3d((1, 2, "three"), "test")  # Non-numeric
        
        with pytest.raises(ValidationError):
            validate_position_3d((1, 2, np.nan), "test")  # NaN value
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Valid cases
            assert validate_file_path(tmp_path, "test", must_exist=True) == tmp_path
            assert validate_file_path(tmp_path, "test", allowed_extensions=['.csv']) == tmp_path
            
            # Invalid cases
            with pytest.raises(ValidationError):
                validate_file_path("nonexistent.csv", "test", must_exist=True)
            
            with pytest.raises(ValidationError):
                validate_file_path(tmp_path, "test", allowed_extensions=['.txt'])
            
            with pytest.raises(ValidationError):
                validate_file_path("", "test")
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_network_config(self):
        """Test network configuration validation."""
        # Valid config
        config = {
            'num_iot_devices': 20,
            'num_irs_elements': 100,
            'area_size': 100.0,
            'bs_position': (50.0, 50.0, 10.0),
            'irs_position': (100.0, 50.0, 5.0),
            'iot_height': 1.5
        }
        
        validated = validate_network_config(config)
        assert validated['num_iot_devices'] == 20
        assert validated['area_size'] == 100.0
        
        # Invalid config
        invalid_config = {'num_iot_devices': -1}
        with pytest.raises(ValidationError):
            validate_network_config(invalid_config)


class TestLoggingConfiguration:
    """Test logging configuration and functionality."""
    
    def test_get_logger(self):
        """Test logger creation and configuration."""
        logger = get_logger("test_logger")
        assert logger.name == "test_logger"
        
        # Test logging
        logger.info("Test message")
        logger.debug("Debug message")
        logger.error("Error message")
    
    def test_setup_default_logging(self):
        """Test default logging setup."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_default_logging(level="DEBUG", log_dir=tmp_dir)
            
            logger.info("Test message")
            
            # Check if log file was created
            log_files = os.listdir(tmp_dir)
            assert any(f.endswith('.log') for f in log_files)
    
    def test_log_exception(self):
        """Test exception logging."""
        logger = get_logger("test_exception")
        
        try:
            raise ValueError("Test exception")
        except Exception as e:
            log_exception(logger, e, "Test context", {"extra": "info"})
    
    def test_performance_logger(self):
        """Test performance logging context manager."""
        logger = get_logger("test_performance")
        
        with PerformanceLogger(logger, "test_operation", {"metric": "value"}):
            # Simulate some work
            import time
            time.sleep(0.01)
    
    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        logger = get_logger("test_metrics")
        
        log_performance_metrics(
            logger, 
            "test_operation", 
            0.123, 
            {"accuracy": 0.95, "loss": 0.05}
        )


class TestNetworkTopologyErrorHandling:
    """Test error handling in network topology module."""
    
    def test_network_config_validation(self):
        """Test NetworkConfig validation."""
        # Valid config
        config = NetworkConfig()
        assert config.num_iot_devices == 20
        
        # Invalid config
        with pytest.raises(NetworkConfigurationError):
            NetworkConfig(num_iot_devices=-1)
        
        with pytest.raises(NetworkConfigurationError):
            NetworkConfig(area_size=-10.0)
    
    def test_base_station_validation(self):
        """Test BaseStation validation."""
        # Valid position
        bs = BaseStation((50.0, 50.0, 10.0))
        assert bs.x == 50.0
        
        # Invalid position
        with pytest.raises(NetworkConfigurationError):
            BaseStation((50.0, 50.0, np.nan))
    
    def test_iot_device_validation(self):
        """Test IoTDevice validation."""
        # Valid device
        device = IoTDevice(0, (10.0, 20.0, 1.5))
        assert device.device_id == 0
        
        # Invalid device ID
        with pytest.raises(NetworkConfigurationError):
            IoTDevice(-1, (10.0, 20.0, 1.5))
        
        # Invalid position
        with pytest.raises(NetworkConfigurationError):
            IoTDevice(0, (10.0, np.inf, 1.5))
    
    def test_distance_calculation_error_handling(self):
        """Test distance calculation error handling."""
        bs = BaseStation((0.0, 0.0, 0.0))
        
        # Valid distance calculation
        distance = bs.distance_to(np.array([3.0, 4.0, 0.0]))
        assert abs(distance - 5.0) < 1e-10
        
        # Invalid target position
        with pytest.raises(ComputationError):
            bs.distance_to(np.array([1.0, 2.0]))  # Wrong shape
        
        with pytest.raises(ComputationError):
            bs.distance_to(np.array([np.nan, 2.0, 3.0]))  # NaN value
    
    def test_network_topology_initialization_error_handling(self):
        """Test NetworkTopology initialization error handling."""
        # Valid initialization
        topology = NetworkTopology()
        assert len(topology.iot_devices) == 20
        
        # Invalid configuration
        with pytest.raises(NetworkConfigurationError):
            invalid_config = NetworkConfig(num_iot_devices=-1)


class TestChannelModelErrorHandling:
    """Test error handling in channel model module."""
    
    def test_channel_params_validation(self):
        """Test ChannelParams validation."""
        # Valid params
        params = ChannelParams()
        assert params.path_loss_exponent_direct == 3.0
        
        # Invalid params
        with pytest.raises(ChannelModelError):
            ChannelParams(path_loss_exponent_direct=-1.0)
        
        with pytest.raises(ChannelModelError):
            ChannelParams(rician_k_direct=-5.0)


class TestIRSOptimizationErrorHandling:
    """Test error handling in IRS optimization module."""
    
    def test_irs_config_validation(self):
        """Test IRSOptimizationConfig validation."""
        from src.core.irs_optimization import IRSOptimizationConfig
        
        # Valid config
        config = IRSOptimizationConfig()
        assert config.max_iterations == 1000
        
        # Invalid config
        with pytest.raises(IRSOptimizationError):
            IRSOptimizationConfig(max_iterations=-1)
        
        with pytest.raises(IRSOptimizationError):
            IRSOptimizationConfig(learning_rate=-0.1)
    
    def test_phase_shift_controller_validation(self):
        """Test PhaseShiftController validation."""
        from src.core.irs_optimization import PhaseShiftController
        
        # Valid controller
        controller = PhaseShiftController(100)
        assert controller.num_elements == 100
        
        # Invalid number of elements
        with pytest.raises(IRSOptimizationError):
            PhaseShiftController(-1)
        
        # Invalid phase shifts
        controller = PhaseShiftController(10)
        with pytest.raises(IRSOptimizationError):
            controller.set_phase_shifts(np.array([1, 2, 3]))  # Wrong size
        
        with pytest.raises(IRSOptimizationError):
            controller.set_phase_shifts(np.array([np.nan] * 10))  # NaN values


class TestMADRLErrorHandling:
    """Test error handling in MADRL module."""
    
    def test_madrl_config_validation(self):
        """Test MADRLConfig validation."""
        from src.core.madrl_agent import MADRLConfig
        
        # Valid config
        config = MADRLConfig()
        assert config.learning_rate == 0.001
        
        # Invalid config
        with pytest.raises(MADRLError):
            MADRLConfig(learning_rate=-0.1)
        
        with pytest.raises(MADRLError):
            MADRLConfig(energy_weight=0.3, performance_weight=0.8)  # Don't sum to 1
    
    def test_madrl_agent_validation(self):
        """Test MADRLAgent validation."""
        from src.core.madrl_agent import MADRLAgent, MADRLConfig
        
        config = MADRLConfig()
        
        # Valid agent
        agent = MADRLAgent(0, 10, 5, config)
        assert agent.agent_id == 0
        
        # Invalid agent parameters
        with pytest.raises(MADRLError):
            MADRLAgent(-1, 10, 5, config)  # Invalid agent_id
        
        with pytest.raises(MADRLError):
            MADRLAgent(0, 0, 5, config)  # Invalid state_dim


class TestIntegrationErrorHandling:
    """Test integrated error handling across modules."""
    
    def test_end_to_end_error_propagation(self):
        """Test error propagation through the system."""
        # Test that errors propagate correctly through the system
        with pytest.raises(NetworkConfigurationError):
            # This should fail at NetworkConfig validation
            config = NetworkConfig(num_iot_devices=0)
            topology = NetworkTopology(config)
    
    def test_logging_integration(self):
        """Test that logging works across all modules."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set up logging
            logger = setup_default_logging(level="DEBUG", log_dir=tmp_dir)
            
            # Create network topology (should generate logs)
            topology = NetworkTopology()
            
            # Check that logs were created
            log_files = os.listdir(tmp_dir)
            assert any(f.endswith('.log') for f in log_files)
    
    def test_comprehensive_error_handling(self):
        """Test comprehensive error handling across all modules."""
        # Test that all modules properly handle and log errors
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_default_logging(level="DEBUG", log_dir=tmp_dir)
            
            # Test network topology errors
            with pytest.raises(NetworkConfigurationError):
                NetworkConfig(num_iot_devices=-1)
            
            # Test channel model errors
            with pytest.raises(ChannelModelError):
                ChannelParams(path_loss_exponent_direct=-1.0)
            
            # Test IRS optimization errors
            from src.core.irs_optimization import IRSOptimizationConfig
            with pytest.raises(IRSOptimizationError):
                IRSOptimizationConfig(max_iterations=0)
            
            # Test MADRL errors
            from src.core.madrl_agent import MADRLConfig
            with pytest.raises(MADRLError):
                MADRLConfig(learning_rate=0.0)
            
            # Verify all errors were logged
            log_files = os.listdir(tmp_dir)
            assert any(f.endswith('_errors.log') for f in log_files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])