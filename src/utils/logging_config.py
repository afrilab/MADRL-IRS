#!/usr/bin/env python3
"""
Logging Configuration Module for 6G IoT Research Framework

This module provides comprehensive logging configuration with different
levels, formatters, and handlers for debugging and monitoring.

Author: Research Team
Date: 2025
"""

import os
import sys
import logging
import logging.handlers
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (f"{self.COLORS[record.levelname]}"
                              f"{record.levelname}"
                              f"{self.COLORS['RESET']}")

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value

        return json.dumps(log_entry)


class ResearchFrameworkLogger:
    """
    Main logger class for the research framework.

    Provides centralized logging configuration with support for different
    output formats, log levels, and destinations.
    """

    def __init__(self, name: str = "research_framework"):
        """
        Initialize the logger.

        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.handlers = {}
        self.configured = False

    def configure(self,
                 level: str = "INFO",
                 console_output: bool = True,
                 file_output: bool = True,
                 json_output: bool = False,
                 log_dir: str = "logs",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 format_string: Optional[str] = None) -> None:
        """
        Configure the logger with specified settings.

        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            console_output: Enable console output
            file_output: Enable file output
            json_output: Enable JSON formatted output
            log_dir: Directory for log files
            max_file_size: Maximum size for log files before rotation
            backup_count: Number of backup files to keep
            format_string: Custom format string
        """
        # Clear existing handlers
        self.logger.handlers.clear()
        self.handlers.clear()

        # Set logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)

        # Default format string
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)

            # Use colored formatter for console
            console_formatter = ColoredFormatter(format_string)
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(console_handler)
            self.handlers['console'] = console_handler

        # File handler
        if file_output:
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Regular log file
            log_file = os.path.join(log_dir, f"{self.name}.log")
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setLevel(numeric_level)

            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)

            self.logger.addHandler(file_handler)
            self.handlers['file'] = file_handler

            # Error log file (only errors and critical)
            error_log_file = os.path.join(log_dir, f"{self.name}_errors.log")
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)

            error_formatter = logging.Formatter(format_string)
            error_handler.setFormatter(error_formatter)

            self.logger.addHandler(error_handler)
            self.handlers['error_file'] = error_handler

        # JSON handler
        if json_output:
            os.makedirs(log_dir, exist_ok=True)

            json_log_file = os.path.join(log_dir, f"{self.name}.json")
            json_handler = logging.handlers.RotatingFileHandler(
                json_log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            json_handler.setLevel(numeric_level)

            json_formatter = JSONFormatter()
            json_handler.setFormatter(json_formatter)

            self.logger.addHandler(json_handler)
            self.handlers['json'] = json_handler

        self.configured = True
        self.logger.info(f"Logger '{self.name}' configured with level {level}")

    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        if not self.configured:
            self.configure()
        return self.logger

    def add_handler(self, name: str, handler: logging.Handler) -> None:
        """Add a custom handler to the logger."""
        self.logger.addHandler(handler)
        self.handlers[name] = handler

    def remove_handler(self, name: str) -> None:
        """Remove a handler from the logger."""
        if name in self.handlers:
            self.logger.removeHandler(self.handlers[name])
            del self.handlers[name]

    def set_level(self, level: str) -> None:
        """Change the logging level."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)

        # Update all handlers
        for handler in self.logger.handlers:
            handler.setLevel(numeric_level)

    def log_system_info(self) -> None:
        """Log system information for debugging."""
        import platform
        import psutil

        self.logger.info("=== System Information ===")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"CPU Count: {psutil.cpu_count()}")
        self.logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        self.logger.info("=== End System Information ===")


# Global logger instances
_loggers: Dict[str, ResearchFrameworkLogger] = {}


def get_logger(name: str = "research_framework",
               auto_configure: bool = True,
               **config_kwargs) -> logging.Logger:
    """
    Get a logger instance with optional auto-configuration.

    Args:
        name: Logger name
        auto_configure: Whether to auto-configure if not already configured
        **config_kwargs: Configuration arguments for auto-configuration

    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        _loggers[name] = ResearchFrameworkLogger(name)

    framework_logger = _loggers[name]

    if auto_configure and not framework_logger.configured:
        framework_logger.configure(**config_kwargs)

    return framework_logger.get_logger()


def configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging from a configuration dictionary.

    Args:
        config: Configuration dictionary with logging settings
    """
    logger_name = config.get('name', 'research_framework')

    if logger_name not in _loggers:
        _loggers[logger_name] = ResearchFrameworkLogger(logger_name)

    framework_logger = _loggers[logger_name]
    framework_logger.configure(**config)


def setup_default_logging(level: str = "INFO",
                         log_dir: str = "logs",
                         enable_json: bool = False) -> logging.Logger:
    """
    Set up default logging configuration for the research framework.

    Args:
        level: Logging level
        log_dir: Directory for log files
        enable_json: Enable JSON formatted logging

    Returns:
        Configured logger instance
    """
    logger = get_logger(
        name="research_framework",
        level=level,
        console_output=True,
        file_output=True,
        json_output=enable_json,
        log_dir=log_dir
    )

    # Log initial setup information
    logger.info("Default logging configuration initialized")

    return logger


def log_exception(logger: logging.Logger,
                 exception: Exception,
                 context: str = None,
                 extra_info: Dict[str, Any] = None) -> None:
    """
    Log an exception with additional context information.

    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context information
        extra_info: Extra information dictionary
    """
    error_msg = f"Exception occurred: {type(exception).__name__}: {str(exception)}"

    if context:
        error_msg = f"{context} - {error_msg}"

    # Create extra fields for structured logging
    extra = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception)
    }

    if extra_info:
        extra.update(extra_info)

    # Log with exception traceback
    logger.error(error_msg, exc_info=True, extra=extra)


def log_performance_metrics(logger: logging.Logger,
                          operation: str,
                          duration: float,
                          metrics: Dict[str, Any] = None) -> None:
    """
    Log performance metrics for operations.

    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        metrics: Additional performance metrics
    """
    perf_msg = f"Performance - {operation}: {duration:.4f}s"

    extra = {
        'operation': operation,
        'duration_seconds': duration,
        'performance_log': True
    }

    if metrics:
        extra.update(metrics)

    logger.info(perf_msg, extra=extra)


# Context manager for performance logging
class PerformanceLogger:
    """Context manager for automatic performance logging."""

    def __init__(self, logger: logging.Logger, operation: str,
                 metrics: Dict[str, Any] = None):
        """
        Initialize performance logger.

        Args:
            logger: Logger instance
            operation: Operation name
            metrics: Additional metrics to log
        """
        self.logger = logger
        self.operation = operation
        self.metrics = metrics or {}
        self.start_time = None

    def __enter__(self):
        """Start timing the operation."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        import time
        duration = time.time() - self.start_time

        if exc_type is None:
            log_performance_metrics(
                self.logger, self.operation, duration, self.metrics
            )
        else:
            self.logger.error(
                f"Operation failed: {self.operation} after {duration:.4f}s",
                exc_info=True
            )
