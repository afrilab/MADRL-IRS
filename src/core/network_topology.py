#!/usr/bin/env python3
"""
Network Topology Module for 6G IoT Networks with IRS

This module implements the network topology modeling for the 6G IoT system
described in the research paper. It includes classes for base stations,
IoT devices, and intelligent reflecting surfaces with 3D positioning
and distance calculation capabilities.

Author: Research Team
Date: 2025
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

# Import custom exceptions and validation
from .exceptions import NetworkConfigurationError, ValidationError, ComputationError
from ..utils.validation import (
    validate_positive_number, validate_integer_range, validate_position_3d,
    validate_array_shape, validate_network_config
)
from ..utils.logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class NetworkConfig:
    """Configuration parameters for the network topology."""
    num_iot_devices: int = 20
    num_irs_elements: int = 100
    area_size: float = 100.0
    bs_position: Tuple[float, float, float] = (50.0, 50.0, 10.0)
    irs_position: Tuple[float, float, float] = (100.0, 50.0, 5.0)
    iot_height: float = 1.5

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        try:
            # Validate all parameters
            self.num_iot_devices = validate_integer_range(
                self.num_iot_devices, 'num_iot_devices', min_value=1, max_value=1000
            )
            self.num_irs_elements = validate_integer_range(
                self.num_irs_elements, 'num_irs_elements', min_value=1, max_value=10000
            )
            self.area_size = validate_positive_number(self.area_size, 'area_size')
            self.bs_position = validate_position_3d(self.bs_position, 'bs_position')
            self.irs_position = validate_position_3d(self.irs_position, 'irs_position')
            self.iot_height = validate_positive_number(self.iot_height, 'iot_height')

            logger.debug(f"NetworkConfig validated: {self}")

        except ValidationError as e:
            logger.error(f"NetworkConfig validation failed: {e}")
            raise NetworkConfigurationError(
                f"Invalid network configuration: {e.message}",
                parameter=e.parameter,
                value=e.actual_value
            ) from e


class BaseStation:
    """
    Base Station class for 6G IoT networks.

    Represents the base station in the network topology with 3D positioning
    and communication capabilities.
    """

    def __init__(self, position: Tuple[float, float, float] = (50.0, 50.0, 10.0)):
        """
        Initialize the base station.

        Args:
            position: 3D position (x, y, z) in meters

        Raises:
            NetworkConfigurationError: If position is invalid
        """
        try:
            # Validate position
            validated_position = validate_position_3d(position, 'bs_position')
            self.position = np.array(validated_position, dtype=float)
            self.x, self.y, self.z = self.position

            logger.debug(f"BaseStation initialized at position {self.position}")

        except ValidationError as e:
            logger.error(f"BaseStation initialization failed: {e}")
            raise NetworkConfigurationError(
                f"Invalid base station position: {e.message}",
                parameter='position',
                value=position
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in BaseStation initialization: {e}")
            raise NetworkConfigurationError(
                f"Failed to initialize base station: {str(e)}",
                parameter='position',
                value=position
            ) from e

    def get_position(self) -> np.ndarray:
        """Get the 3D position of the base station."""
        return self.position.copy()

    def distance_to(self, target_position: np.ndarray) -> float:
        """
        Calculate Euclidean distance to a target position.

        Args:
            target_position: 3D position array

        Returns:
            Distance in meters

        Raises:
            ComputationError: If distance calculation fails
        """
        try:
            # Validate input
            target_position = validate_array_shape(
                target_position, 'target_position', expected_shape=(3,)
            )

            # Calculate distance
            distance = np.linalg.norm(self.position - target_position)

            if np.isnan(distance) or np.isinf(distance):
                raise ComputationError(
                    "Distance calculation resulted in invalid value",
                    operation="distance_calculation"
                )

            return float(distance)

        except ValidationError as e:
            logger.error(f"Invalid target position for distance calculation: {e}")
            raise ComputationError(
                f"Invalid target position: {e.message}",
                operation="distance_calculation"
            ) from e
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            raise ComputationError(
                f"Failed to calculate distance: {str(e)}",
                operation="distance_calculation"
            ) from e

    def __repr__(self) -> str:
        return f"BaseStation(position=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}))"


class IoTDevice:
    """
    IoT Device class for 6G networks.

    Represents individual IoT devices in the network with positioning
    and identification capabilities.
    """

    def __init__(self, device_id: int, position: Tuple[float, float, float]):
        """
        Initialize an IoT device.

        Args:
            device_id: Unique identifier for the device
            position: 3D position (x, y, z) in meters

        Raises:
            NetworkConfigurationError: If parameters are invalid
        """
        try:
            # Validate device ID
            self.device_id = validate_integer_range(
                device_id, 'device_id', min_value=0
            )

            # Validate position
            validated_position = validate_position_3d(position, 'iot_position')
            self.position = np.array(validated_position, dtype=float)
            self.x, self.y, self.z = self.position

            logger.debug(f"IoTDevice {self.device_id} initialized at position {self.position}")

        except ValidationError as e:
            logger.error(f"IoTDevice initialization failed: {e}")
            raise NetworkConfigurationError(
                f"Invalid IoT device parameters: {e.message}",
                parameter=e.parameter,
                value=e.actual_value
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error in IoTDevice initialization: {e}")
            raise NetworkConfigurationError(
                f"Failed to initialize IoT device: {str(e)}"
            ) from e

    def get_position(self) -> np.ndarray:
        """Get the 3D position of the IoT device."""
        return self.position.copy()

    def distance_to(self, target_position: np.ndarray) -> float:
        """
        Calculate Euclidean distance to a target position.

        Args:
            target_position: 3D position array

        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.position - target_position)

    def set_position(self, position: Tuple[float, float, float]) -> None:
        """
        Update the device position.

        Args:
            position: New 3D position (x, y, z) in meters
        """
        self.position = np.array(position, dtype=float)
        self.x, self.y, self.z = self.position

    def __repr__(self) -> str:
        return f"IoTDevice(id={self.device_id}, position=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}))"


class IntelligentReflectingSurface:
    """
    Intelligent Reflecting Surface (IRS) class.

    Represents the IRS in the network topology with configurable reflecting
    elements and 3D positioning capabilities.
    """

    def __init__(self, position: Tuple[float, float, float] = (100.0, 50.0, 5.0),
                 num_elements: int = 100):
        """
        Initialize the IRS.

        Args:
            position: 3D position (x, y, z) in meters
            num_elements: Number of reflecting elements
        """
        self.position = np.array(position, dtype=float)
        self.x, self.y, self.z = self.position
        self.num_elements = num_elements

        # Initialize phase shifts (random initial configuration)
        self.phase_shifts = np.random.uniform(0, 2*np.pi, num_elements)

    def get_position(self) -> np.ndarray:
        """Get the 3D position of the IRS."""
        return self.position.copy()

    def distance_to(self, target_position: np.ndarray) -> float:
        """
        Calculate Euclidean distance to a target position.

        Args:
            target_position: 3D position array

        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.position - target_position)

    def set_phase_shifts(self, phase_shifts: np.ndarray) -> None:
        """
        Set the phase shifts for all reflecting elements.

        Args:
            phase_shifts: Array of phase shifts in radians

        Raises:
            NetworkConfigurationError: If phase shifts are invalid
        """
        try:
            # Validate input array
            phase_shifts = validate_array_shape(
                phase_shifts, 'phase_shifts', expected_shape=(self.num_elements,)
            )

            # Validate phase shift values (should be between 0 and 2π)
            if np.any(phase_shifts < 0) or np.any(phase_shifts > 2 * np.pi):
                raise ValidationError(
                    "Phase shifts must be between 0 and 2π radians",
                    parameter='phase_shifts',
                    actual_value=f"min={np.min(phase_shifts):.3f}, max={np.max(phase_shifts):.3f}"
                )

            # Check for NaN or infinite values
            if np.any(~np.isfinite(phase_shifts)):
                raise ValidationError(
                    "Phase shifts must be finite values",
                    parameter='phase_shifts'
                )

            self.phase_shifts = phase_shifts.copy()
            logger.debug(f"IRS phase shifts updated: {len(phase_shifts)} elements")

        except ValidationError as e:
            logger.error(f"Invalid phase shifts: {e}")
            raise NetworkConfigurationError(
                f"Invalid phase shifts: {e.message}",
                parameter='phase_shifts',
                value=phase_shifts if isinstance(phase_shifts, np.ndarray) else str(phase_shifts)
            ) from e
        except Exception as e:
            logger.error(f"Failed to set phase shifts: {e}")
            raise NetworkConfigurationError(
                f"Failed to set phase shifts: {str(e)}"
            ) from e

    def get_phase_shifts(self) -> np.ndarray:
        """Get the current phase shifts configuration."""
        return self.phase_shifts.copy()

    def get_configuration(self) -> np.ndarray:
        """
        Get the complex reflection coefficients.

        Returns:
            Complex array of reflection coefficients e^(j*theta)
        """
        return np.exp(1j * self.phase_shifts)

    def randomize_configuration(self) -> None:
        """Randomize the phase shifts configuration."""
        self.phase_shifts = np.random.uniform(0, 2*np.pi, self.num_elements)

    def __repr__(self) -> str:
        return f"IRS(position=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), elements={self.num_elements})"


class NetworkTopology:
    """
    Network topology class implementing the system model described in the paper.

    This class models the 6G IoT network with base station, IoT devices, and IRS
    as described in Section III-A of the paper. It handles 3D positioning,
    distance calculations, and network configuration.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize the network topology.

        Args:
            config: Network configuration parameters

        Raises:
            NetworkConfigurationError: If initialization fails
        """
        try:
            with PerformanceLogger(logger, "NetworkTopology initialization"):
                # Validate and set configuration
                self.config = config or NetworkConfig()
                logger.info(f"Initializing network topology with config: {self.config}")

                # Initialize network components
                self.base_station = BaseStation(self.config.bs_position)
                self.irs = IntelligentReflectingSurface(
                    self.config.irs_position,
                    self.config.num_irs_elements
                )
                self.iot_devices: List[IoTDevice] = []

                # Generate IoT device positions
                self._generate_iot_positions()

                # Calculate and cache distances
                self._calculate_distances()

                logger.info(f"Network topology initialized successfully with "
                           f"{len(self.iot_devices)} IoT devices")

        except NetworkConfigurationError:
            # Re-raise network configuration errors
            raise
        except Exception as e:
            logger.error(f"Failed to initialize network topology: {e}")
            log_exception(logger, e, "NetworkTopology initialization")
            raise NetworkConfigurationError(
                f"Failed to initialize network topology: {str(e)}"
            ) from e

    def _generate_iot_positions(self) -> None:
        """
        Generate random positions for IoT devices within the coverage area.

        Raises:
            NetworkConfigurationError: If device generation fails
        """
        try:
            self.iot_devices = []
            logger.debug(f"Generating {self.config.num_iot_devices} IoT device positions")

            for i in range(self.config.num_iot_devices):
                try:
                    # Random position within the area
                    x = np.random.uniform(0, self.config.area_size)
                    y = np.random.uniform(0, self.config.area_size)
                    z = self.config.iot_height

                    device = IoTDevice(device_id=i, position=(x, y, z))
                    self.iot_devices.append(device)

                except Exception as e:
                    logger.error(f"Failed to create IoT device {i}: {e}")
                    raise NetworkConfigurationError(
                        f"Failed to create IoT device {i}: {str(e)}"
                    ) from e

            logger.debug(f"Successfully generated {len(self.iot_devices)} IoT devices")

        except NetworkConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate IoT device positions: {e}")
            raise NetworkConfigurationError(
                f"Failed to generate IoT device positions: {str(e)}"
            ) from e

    def _calculate_distances(self) -> None:
        """
        Calculate and cache distances between network components.

        Raises:
            ComputationError: If distance calculations fail
        """
        try:
            with PerformanceLogger(logger, "Distance calculations"):
                logger.debug("Calculating distances between network components")

                # Distance from BS to each IoT device
                self.bs_iot_distances = np.zeros(self.config.num_iot_devices)
                for i, device in enumerate(self.iot_devices):
                    try:
                        self.bs_iot_distances[i] = self.base_station.distance_to(device.get_position())
                    except Exception as e:
                        logger.error(f"Failed to calculate BS-IoT distance for device {i}: {e}")
                        raise ComputationError(
                            f"Failed to calculate BS-IoT distance for device {i}: {str(e)}",
                            operation="bs_iot_distance"
                        ) from e

                # Distance from IRS to each IoT device
                self.irs_iot_distances = np.zeros(self.config.num_iot_devices)
                for i, device in enumerate(self.iot_devices):
                    try:
                        self.irs_iot_distances[i] = self.irs.distance_to(device.get_position())
                    except Exception as e:
                        logger.error(f"Failed to calculate IRS-IoT distance for device {i}: {e}")
                        raise ComputationError(
                            f"Failed to calculate IRS-IoT distance for device {i}: {str(e)}",
                            operation="irs_iot_distance"
                        ) from e

                # Distance from BS to IRS
                try:
                    self.bs_irs_distance = self.base_station.distance_to(self.irs.get_position())
                except Exception as e:
                    logger.error(f"Failed to calculate BS-IRS distance: {e}")
                    raise ComputationError(
                        f"Failed to calculate BS-IRS distance: {str(e)}",
                        operation="bs_irs_distance"
                    ) from e

                logger.debug(f"Distance calculations completed: "
                           f"BS-IoT: {np.mean(self.bs_iot_distances):.2f}m avg, "
                           f"IRS-IoT: {np.mean(self.irs_iot_distances):.2f}m avg, "
                           f"BS-IRS: {self.bs_irs_distance:.2f}m")

        except ComputationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in distance calculations: {e}")
            raise ComputationError(
                f"Distance calculation failed: {str(e)}",
                operation="distance_calculation"
            ) from e

    def add_iot_device(self, position: Tuple[float, float, float]) -> int:
        """
        Add a new IoT device to the network.

        Args:
            position: 3D position (x, y, z) in meters

        Returns:
            Device ID of the newly added device
        """
        device_id = len(self.iot_devices)
        device = IoTDevice(device_id, position)
        self.iot_devices.append(device)

        # Update configuration and recalculate distances
        self.config.num_iot_devices += 1
        self._calculate_distances()

        return device_id

    def remove_iot_device(self, device_id: int) -> bool:
        """
        Remove an IoT device from the network.

        Args:
            device_id: ID of the device to remove

        Returns:
            True if device was removed, False if not found
        """
        for i, device in enumerate(self.iot_devices):
            if device.device_id == device_id:
                self.iot_devices.pop(i)
                self.config.num_iot_devices -= 1
                self._calculate_distances()
                return True
        return False

    def move_iot_device(self, device_id: int, new_position: Tuple[float, float, float]) -> bool:
        """
        Move an IoT device to a new position.

        Args:
            device_id: ID of the device to move
            new_position: New 3D position (x, y, z) in meters

        Returns:
            True if device was moved, False if not found
        """
        for device in self.iot_devices:
            if device.device_id == device_id:
                device.set_position(new_position)
                self._calculate_distances()
                return True
        return False

    def get_bs_position(self) -> np.ndarray:
        """Get the base station position."""
        return self.base_station.get_position()

    def get_irs_position(self) -> np.ndarray:
        """Get the IRS position."""
        return self.irs.get_position()

    def get_iot_positions(self) -> np.ndarray:
        """Get all IoT device positions as a 2D array."""
        positions = np.zeros((len(self.iot_devices), 3))
        for i, device in enumerate(self.iot_devices):
            positions[i] = device.get_position()
        return positions

    def get_bs_iot_distances(self) -> np.ndarray:
        """Get distances from base station to all IoT devices."""
        return self.bs_iot_distances.copy()

    def get_irs_iot_distances(self) -> np.ndarray:
        """Get distances from IRS to all IoT devices."""
        return self.irs_iot_distances.copy()

    def get_bs_irs_distance(self) -> float:
        """Get distance from base station to IRS."""
        return self.bs_irs_distance

    def get_device_by_id(self, device_id: int) -> Optional[IoTDevice]:
        """
        Get an IoT device by its ID.

        Args:
            device_id: Device ID to search for

        Returns:
            IoTDevice if found, None otherwise
        """
        for device in self.iot_devices:
            if device.device_id == device_id:
                return device
        return None

    def get_all_distances(self) -> dict:
        """
        Get all calculated distances in the network.

        Returns:
            Dictionary containing all distance measurements
        """
        return {
            'bs_iot_distances': self.get_bs_iot_distances(),
            'irs_iot_distances': self.get_irs_iot_distances(),
            'bs_irs_distance': self.get_bs_irs_distance()
        }

    def calculate_coverage_area(self, max_distance: float) -> List[int]:
        """
        Calculate which IoT devices are within coverage range of the base station.

        Args:
            max_distance: Maximum coverage distance in meters

        Returns:
            List of device IDs within coverage
        """
        covered_devices = []
        for i, distance in enumerate(self.bs_iot_distances):
            if distance <= max_distance:
                covered_devices.append(i)
        return covered_devices

    def get_network_statistics(self) -> dict:
        """
        Get statistical information about the network topology.

        Returns:
            Dictionary with network statistics
        """
        bs_distances = self.get_bs_iot_distances()
        irs_distances = self.get_irs_iot_distances()

        return {
            'num_devices': len(self.iot_devices),
            'num_irs_elements': self.irs.num_elements,
            'area_size': self.config.area_size,
            'bs_iot_distance_stats': {
                'mean': np.mean(bs_distances),
                'std': np.std(bs_distances),
                'min': np.min(bs_distances),
                'max': np.max(bs_distances)
            },
            'irs_iot_distance_stats': {
                'mean': np.mean(irs_distances),
                'std': np.std(irs_distances),
                'min': np.min(irs_distances),
                'max': np.max(irs_distances)
            },
            'bs_irs_distance': self.bs_irs_distance
        }

    def __repr__(self) -> str:
        return (f"NetworkTopology(devices={len(self.iot_devices)}, "
                f"irs_elements={self.irs.num_elements}, "
                f"area={self.config.area_size}x{self.config.area_size})")
