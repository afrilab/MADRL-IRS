"""
Visualization utilities for the 6G IoT research framework.

This module provides plotting and visualization functions for network topology,
performance metrics, and experimental results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import custom exceptions and logging
from ..core.exceptions import ValidationError, ResourceError
from ..utils.validation import validate_positive_number, validate_position_3d, validate_array_shape
from .logging_config import get_logger, log_exception, PerformanceLogger

# Get logger for this module
logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class NetworkVisualizer:
    """Visualization utilities for network topology and wireless communications."""
    @staticmethod
    def plot_network_topology_2d(
        bs_position: Tuple[float, float],
        irs_position: Tuple[float, float],
        iot_positions: List[Tuple[float, float]],
        coverage_radius: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2D network topology.
        Args:
            bs_position: Base station (x, y) position
            irs_position: IRS (x, y) position
            iot_positions: List of IoT device (x, y) positions
            coverage_radius: Optional coverage radius to display
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object

        Raises:
            ValidationError: If input parameters are invalid
            ResourceError: If plotting fails
        """
        try:
            # Validate inputs
            if len(bs_position) != 2:
                raise ValidationError(
                    "Base station position must be a 2D tuple (x, y)",
                    parameter='bs_position'
                )
            if len(irs_position) != 2:
                raise ValidationError(
                    "IRS position must be a 2D tuple (x, y)",
                    parameter='irs_position'
                )

            # Validate IoT positions
            for i, pos in enumerate(iot_positions):
                if len(pos) != 2:
                    raise ValidationError(
                        f"IoT device position {i} must be a 2D tuple (x, y)",
                        parameter=f'iot_positions[{i}]'
                    )

            if coverage_radius is not None:
                coverage_radius = validate_positive_number(coverage_radius, 'coverage_radius')

            logger.debug(f"Plotting 2D network topology with {len(iot_positions)} IoT devices")

            with PerformanceLogger(logger, "2D network topology plotting"):
                fig, ax = plt.subplots(figsize=(10, 8))

                # Plot base station
                ax.scatter(*bs_position, s=200, c='red', marker='^',
                          label='Base Station', edgecolors='black', linewidth=2)

                # Plot IRS
                ax.scatter(*irs_position, s=150, c='blue', marker='s',
                          label='IRS', edgecolors='black', linewidth=2)

                # Plot IoT devices
                if iot_positions:
                    iot_x, iot_y = zip(*iot_positions)
                    ax.scatter(iot_x, iot_y, s=50, c='green', marker='o',
                              label='IoT Devices', alpha=0.7)

                # Add coverage radius if specified
                if coverage_radius:
                    circle = plt.Circle(bs_position, coverage_radius,
                                      fill=False, linestyle='--', alpha=0.5)
                    ax.add_patch(circle)

                # Draw connections
                for iot_pos in iot_positions:
                    # Direct link to BS
                    ax.plot([bs_position[0], iot_pos[0]],
                           [bs_position[1], iot_pos[1]],
                           'r--', alpha=0.3, linewidth=1)
                    # IRS-assisted link
                    ax.plot([iot_pos[0], irs_position[0]],
                           [iot_pos[1], irs_position[1]],
                           'b:', alpha=0.3, linewidth=1)
                    ax.plot([irs_position[0], bs_position[0]],
                           [irs_position[1], bs_position[1]],
                           'b:', alpha=0.3, linewidth=1)

                ax.set_xlabel('X Position (m)')
                ax.set_ylabel('Y Position (m)')
                ax.set_title('6G IoT Network Topology')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')

                if save_path:
                    try:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        logger.info(f"Network topology saved to {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save plot to {save_path}: {e}")

                return fig

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to plot 2D network topology: {e}")
            log_exception(logger, e, "2D network topology plotting")
            raise ResourceError(
                f"Failed to create 2D network topology plot: {str(e)}",
                resource_type='visualization'
            ) from e

    @staticmethod
    def plot_network_topology_3d(
        bs_position: Tuple[float, float, float],
        irs_position: Tuple[float, float, float],
        iot_positions: List[Tuple[float, float, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 3D network topology.

        Args:
            bs_position: Base station (x, y, z) position
            irs_position: IRS (x, y, z) position
            iot_positions: List of IoT device (x, y, z) positions
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot base station
        ax.scatter(*bs_position, s=200, c='red', marker='^',
                  label='Base Station', edgecolors='black', linewidth=2)

        # Plot IRS
        ax.scatter(*irs_position, s=150, c='blue', marker='s',
                  label='IRS', edgecolors='black', linewidth=2)

        # Plot IoT devices
        if iot_positions:
            iot_x, iot_y, iot_z = zip(*iot_positions)
            ax.scatter(iot_x, iot_y, iot_z, s=50, c='green', marker='o',
                      label='IoT Devices', alpha=0.7)

        # Draw connections
        for iot_pos in iot_positions:
            # Direct link to BS
            ax.plot([bs_position[0], iot_pos[0]],
                   [bs_position[1], iot_pos[1]],
                   [bs_position[2], iot_pos[2]],
                   'r--', alpha=0.3, linewidth=1)
            # IRS-assisted link
            ax.plot([iot_pos[0], irs_position[0]],
                   [iot_pos[1], irs_position[1]],
                   [iot_pos[2], irs_position[2]],
                   'b:', alpha=0.3, linewidth=1)
            ax.plot([irs_position[0], bs_position[0]],
                   [irs_position[1], bs_position[1]],
                   [irs_position[2], bs_position[2]],
                   'b:', alpha=0.3, linewidth=1)

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D 6G IoT Network Topology')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D network topology saved to {save_path}")

        return fig


class PerformanceVisualizer:
    """Visualization utilities for performance metrics and results."""

    @staticmethod
    def plot_training_curves(
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_accuracies: Optional[List[float]] = None,
        val_accuracies: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training curves for federated learning.

        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            train_accuracies: Training accuracy values
            val_accuracies: Validation accuracy values
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Communication Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracies
        if train_accuracies:
            axes[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        if val_accuracies:
            axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Communication Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")

        return fig

    @staticmethod
    def plot_convergence_comparison(
        results_dict: Dict[str, List[float]],
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence comparison between different methods.

        Args:
            results_dict: Dictionary with method names as keys and metric values as values
            metric_name: Name of the metric being plotted
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for method_name, values in results_dict.items():
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, label=method_name, linewidth=2, marker='o', markersize=4)

        ax.set_xlabel('Communication Round')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence comparison saved to {save_path}")

        return fig

    @staticmethod
    def plot_data_distribution(
        client_data_stats: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot data distribution across federated learning clients.

        Args:
            client_data_stats: Statistics from validate_data_distribution function
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot samples per client
        client_ids = range(1, client_data_stats['num_clients'] + 1)
        axes[0].bar(client_ids, client_data_stats['samples_per_client'])
        axes[0].set_xlabel('Client ID')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_title('Data Distribution Across Clients')
        axes[0].grid(True, alpha=0.3)

        # Plot class distribution
        classes = list(client_data_stats['class_distribution'].keys())
        counts = list(client_data_stats['class_distribution'].values())
        axes[1].bar(classes, counts)
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title('Overall Class Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {save_path}")

        return fig


class ChannelVisualizer:
    """Visualization utilities for wireless channel characteristics."""

    @staticmethod
    def plot_channel_gain_heatmap(
        positions: List[Tuple[float, float]],
        channel_gains: np.ndarray,
        title: str = "Channel Gain Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot channel gain as a heatmap.

        Args:
            positions: List of (x, y) positions
            channel_gains: Channel gain values
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # Create scatter plot with color mapping
        scatter = ax.scatter(x_coords, y_coords, c=channel_gains,
                           cmap='viridis', s=100, alpha=0.8)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Channel Gain (dB)')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Channel gain heatmap saved to {save_path}")

        return fig

    @staticmethod
    def plot_snr_vs_distance(
        distances: np.ndarray,
        snr_values: np.ndarray,
        method_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot SNR vs distance for different communication methods.

        Args:
            distances: Distance values
            snr_values: SNR values (can be 2D for multiple methods)
            method_labels: Labels for different methods
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if snr_values.ndim == 1:
            ax.plot(distances, snr_values, 'b-', linewidth=2, marker='o')
        else:
            for i, snr_method in enumerate(snr_values.T):
                label = method_labels[i] if method_labels else f'Method {i+1}'
                ax.plot(distances, snr_method, linewidth=2, marker='o', label=label)
            ax.legend()

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('SNR (dB)')
        ax.set_title('SNR vs Distance')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SNR vs distance plot saved to {save_path}")

        return fig


def create_interactive_network_plot(
    bs_position: Tuple[float, float, float],
    irs_position: Tuple[float, float, float],
    iot_positions: List[Tuple[float, float, float]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive 3D network topology plot using Plotly.

    Args:
        bs_position: Base station (x, y, z) position
        irs_position: IRS (x, y, z) position
        iot_positions: List of IoT device (x, y, z) positions
        save_path: Optional path to save the figure

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add base station
    fig.add_trace(go.Scatter3d(
        x=[bs_position[0]], y=[bs_position[1]], z=[bs_position[2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='Base Station'
    ))

    # Add IRS
    fig.add_trace(go.Scatter3d(
        x=[irs_position[0]], y=[irs_position[1]], z=[irs_position[2]],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='square'),
        name='IRS'
    ))

    # Add IoT devices
    if iot_positions:
        iot_x, iot_y, iot_z = zip(*iot_positions)
        fig.add_trace(go.Scatter3d(
            x=iot_x, y=iot_y, z=iot_z,
            mode='markers',
            marker=dict(size=6, color='green'),
            name='IoT Devices'
        ))

    # Add connections
    for iot_pos in iot_positions:
        # Direct link to BS
        fig.add_trace(go.Scatter3d(
            x=[bs_position[0], iot_pos[0]],
            y=[bs_position[1], iot_pos[1]],
            z=[bs_position[2], iot_pos[2]],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False,
            opacity=0.3
        ))

        # IRS-assisted link
        fig.add_trace(go.Scatter3d(
            x=[iot_pos[0], irs_position[0], bs_position[0]],
            y=[iot_pos[1], irs_position[1], bs_position[1]],
            z=[iot_pos[2], irs_position[2], bs_position[2]],
            mode='lines',
            line=dict(color='blue', width=2, dash='dot'),
            showlegend=False,
            opacity=0.3
        ))

    fig.update_layout(
        title='Interactive 3D 6G IoT Network Topology',
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)'
        )
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive network plot saved to {save_path}")

    return fig


def save_all_plots(figures: List[plt.Figure], base_path: str, format: str = 'png'):
    """
    Save multiple matplotlib figures to files.

    Args:
        figures: List of matplotlib figures
        base_path: Base path for saving files
        format: File format (png, pdf, svg)
    """
    for i, fig in enumerate(figures):
        save_path = f"{base_path}_{i+1}.{format}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure {i+1} saved to {save_path}")


class IRSVisualizer:
    """Visualization utilities for IRS optimization results."""

    @staticmethod
    def plot_phase_shift_optimization(
        phase_shifts: np.ndarray,
        objective_values: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IRS phase shift optimization progress.

        Args:
            phase_shifts: Phase shift values over iterations
            objective_values: Objective function values over iterations
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot phase shifts evolution
        iterations = range(len(objective_values))
        for i in range(min(10, phase_shifts.shape[1])):  # Show first 10 elements
            ax1.plot(iterations, phase_shifts[:, i], alpha=0.7, label=f'Element {i+1}')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Phase Shift (radians)')
        ax1.set_title('IRS Phase Shift Evolution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot objective function
        ax2.plot(iterations, objective_values, 'b-', linewidth=2, marker='o')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Optimization Convergence')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"IRS optimization plot saved to {save_path}")

        return fig

    @staticmethod
    def plot_beamforming_pattern(
        angles: np.ndarray,
        beam_pattern: np.ndarray,
        title: str = "IRS Beamforming Pattern",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot IRS beamforming pattern.

        Args:
            angles: Angle values in degrees
            beam_pattern: Beam pattern values
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection='polar'))

        # Convert angles to radians
        angles_rad = np.deg2rad(angles)

        ax.plot(angles_rad, beam_pattern, 'b-', linewidth=2)
        ax.fill_between(angles_rad, beam_pattern, alpha=0.3)
        ax.set_title(title, pad=20)
        ax.set_ylim(0, np.max(beam_pattern) * 1.1)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Beamforming pattern saved to {save_path}")

        return fig


class MADRLVisualizer:
    """Visualization utilities for MADRL results."""

    @staticmethod
    def plot_reward_evolution(
        rewards_per_agent: Dict[int, List[float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot reward evolution for multiple agents.

        Args:
            rewards_per_agent: Dictionary with agent IDs and their reward histories
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent_id, rewards in rewards_per_agent.items():
            episodes = range(1, len(rewards) + 1)
            ax.plot(episodes, rewards, label=f'Agent {agent_id}', linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('MADRL Agent Reward Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"MADRL reward evolution saved to {save_path}")

        return fig

    @staticmethod
    def plot_energy_efficiency(
        energy_consumption: List[float],
        data_rates: List[float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot energy efficiency over time.

        Args:
            energy_consumption: Energy consumption values
            data_rates: Data rate values
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        time_steps = range(1, len(energy_consumption) + 1)

        # Energy consumption
        ax1.plot(time_steps, energy_consumption, 'r-', linewidth=2)
        ax1.set_ylabel('Energy Consumption (J)')
        ax1.set_title('Energy Consumption Over Time')
        ax1.grid(True, alpha=0.3)

        # Data rates
        ax2.plot(time_steps, data_rates, 'b-', linewidth=2)
        ax2.set_ylabel('Data Rate (Mbps)')
        ax2.set_title('Data Rate Over Time')
        ax2.grid(True, alpha=0.3)

        # Energy efficiency
        efficiency = np.array(data_rates) / np.array(energy_consumption)
        ax3.plot(time_steps, efficiency, 'g-', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Energy Efficiency (Mbps/J)')
        ax3.set_title('Energy Efficiency Over Time')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy efficiency plot saved to {save_path}")

        return fig


def create_publication_ready_plot(
    plot_function,
    *args,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    style: str = 'seaborn-v0_8',
    **kwargs
) -> plt.Figure:
    """
    Create publication-ready plots with consistent styling.

    Args:
        plot_function: Function to create the plot
        *args: Arguments for the plot function
        figsize: Figure size
        dpi: Resolution for saving
        style: Matplotlib style
        **kwargs: Keyword arguments for the plot function

    Returns:
        Matplotlib figure object
    """
    # Set publication style
    with plt.style.context(style):
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })

        # Create plot
        fig = plot_function(*args, **kwargs)

        # Ensure tight layout
        plt.tight_layout()

        return fig
