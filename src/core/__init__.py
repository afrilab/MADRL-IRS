"""
Core modules for 6G IoT Networks with IRS and Federated Learning.

This package contains the core implementations for the research paper:
"Federated Learning with Intelligent Reflecting Surface and Multi-Agent
Deep Reinforcement Learning for 6G Green IoT Networks"

Main components:
- Network topology modeling
- Wireless channel modeling
- Federated learning implementation
- IRS optimization
- Multi-agent deep reinforcement learning
"""

from .network_topology import (
    NetworkTopology,
    BaseStation,
    IoTDevice,
    IntelligentReflectingSurface,
    NetworkConfig
)

__all__ = [
    'NetworkTopology',
    'BaseStation',
    'IoTDevice',
    'IntelligentReflectingSurface',
    'NetworkConfig'
]
