# Contributing to FL-IRS-MADRL-6G

Thank you for your interest in contributing to our research implementation! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Documentation](#documentation)
8. [Testing](#testing)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of federated learning, wireless communications, or reinforcement learning
- Familiarity with PyTorch/TensorFlow

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/fl-irs-madrl-6g.git
   cd fl-irs-madrl-6g
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Verify setup**
   ```bash
   # Run tests
   python -m pytest tests/ -v
   
   # Run linting
   flake8 src/ tests/ examples/
   black --check src/ tests/ examples/
   ```

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **Feature additions**: Implement new functionality
- **Documentation**: Improve or add documentation
- **Examples**: Add new example scripts or notebooks
- **Tests**: Improve test coverage
- **Performance**: Optimize existing algorithms

### Coding Standards

#### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 88 characters (Black default)

#### Code Quality

- Write clear, readable code with meaningful variable names
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Follow existing code patterns and architecture

#### Example Code Style

```python
def calculate_energy_efficiency(
    data_rates: List[float], 
    power_consumption: List[float]
) -> float:
    """
    Calculate energy efficiency for IoT devices.
    
    Args:
        data_rates: List of data rates in bps
        power_consumption: List of power consumption in watts
        
    Returns:
        Energy efficiency in bits/joule
        
    Raises:
        ValueError: If input lists have different lengths
    """
    if len(data_rates) != len(power_consumption):
        raise ValueError("Data rates and power consumption lists must have same length")
    
    total_rate = sum(data_rates)
    total_power = sum(power_consumption)
    
    return total_rate / total_power if total_power > 0 else 0.0
```

### Commit Message Guidelines

Use clear, descriptive commit messages following this format:

```
type(scope): brief description

Detailed explanation if necessary

Fixes #issue_number
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements

**Examples:**
```
feat(irs): add gradient-based phase optimization

Implement gradient descent optimization for IRS phase shifts
using Riemannian manifold optimization techniques.

Fixes #42
```

```
fix(fl): resolve convergence issue in non-IID scenarios

Fixed numerical instability in federated averaging when
dealing with highly skewed data distributions.

Fixes #38
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   python -m pytest tests/ -v
   
   # Run specific test file
   python -m pytest tests/test_your_module.py -v
   
   # Check code style
   black src/ tests/ examples/
   flake8 src/ tests/ examples/
   ```

4. **Update documentation**
   - Update docstrings for new/modified functions
   - Update README.md if adding new features
   - Add examples if appropriate

### Submitting the Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request**
   - Use a clear, descriptive title
   - Provide detailed description of changes
   - Reference related issues
   - Include screenshots for UI changes

3. **Pull request template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] Added tests for new functionality
   - [ ] Updated documentation
   
   ## Related Issues
   Fixes #issue_number
   ```

### Review Process

1. **Automated checks**: All PRs must pass automated tests and linting
2. **Code review**: At least one maintainer will review your code
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: PR will be merged after approval

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - Operating system
   - Package versions (`pip freeze`)

2. **Steps to reproduce**
   - Minimal code example
   - Expected vs actual behavior
   - Error messages/stack traces

3. **Additional context**
   - Screenshots if applicable
   - Related issues or PRs

### Feature Requests

For feature requests, please provide:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives considered**: Other approaches you've considered
4. **Additional context**: Use cases, examples, etc.

### Issue Templates

Use our issue templates when creating new issues:

- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Documentation**: For documentation improvements
- **Question**: For asking questions about usage

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and inline comments
2. **API documentation**: Automatically generated from docstrings
3. **User guides**: Tutorials and examples
4. **Developer documentation**: Architecture and design decisions

### Documentation Standards

- Use clear, concise language
- Include code examples where appropriate
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Testing

### Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_network.py
│   ├── test_channel.py
│   └── test_federated.py
├── integration/          # Integration tests
│   └── test_simulation.py
└── fixtures/            # Test data and fixtures
    └── sample_data.py
```

### Writing Tests

- Use `pytest` for testing framework
- Write tests for all new functionality
- Aim for high test coverage (>80%)
- Use descriptive test names

### Example Test

```python
import pytest
from src.core.network_topology import NetworkTopology

class TestNetworkTopology:
    def test_initialization(self):
        """Test network topology initialization."""
        network = NetworkTopology(num_iot_devices=10)
        
        assert len(network.iot_devices) == 10
        assert network.bs is not None
        assert network.irs is not None
    
    def test_distance_calculation(self):
        """Test distance calculation between devices."""
        network = NetworkTopology()
        
        distance = network.calculate_distance(
            network.bs.position,
            network.iot_devices[0].position
        )
        
        assert distance > 0
        assert isinstance(distance, float)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_network.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run performance tests
python -m pytest tests/ -m performance
```

## Getting Help

If you need help with contributing:

1. **Check existing issues**: Your question might already be answered
2. **GitHub Discussions**: Ask questions in our discussions forum
3. **Email**: Contact maintainers at research@example.com
4. **Documentation**: Check our documentation and examples

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release notes**: Major contributions mentioned in releases
- **Academic papers**: Significant contributions may be acknowledged in future publications

Thank you for contributing to our research project!