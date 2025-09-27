"""
Setup script for the Federated Learning with IRS and MADRL research implementation.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    return "Federated Learning with Intelligent Reflecting Surface and Multi-Agent Deep Reinforcement Learning for 6G Green IoT Networks"

# Read requirements from requirements.txt
def read_requirements():
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="fl-irs-madrl-6g",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Federated Learning with IRS and MADRL for 6G Green IoT Networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/fl-irs-madrl-6g",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fl-irs-simulation=examples.basic_simulation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="federated-learning, intelligent-reflecting-surface, multi-agent, reinforcement-learning, 6g, iot, wireless-communication",
    project_urls={
        "Bug Reports": "https://github.com/research-team/fl-irs-madrl-6g/issues",
        "Source": "https://github.com/research-team/fl-irs-madrl-6g",
        "Documentation": "https://fl-irs-madrl-6g.readthedocs.io/",
    },
)