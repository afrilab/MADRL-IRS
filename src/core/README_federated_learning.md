# Federated Learning Implementation

This module implements a comprehensive federated learning system for 6G Green IoT Networks, supporting multiple datasets and data distribution strategies.

## Overview

The federated learning implementation consists of three main components:

1. **LocalDevice**: Represents individual IoT devices that perform local training
2. **GlobalAggregator**: Implements the FedAvg algorithm for model aggregation
3. **FederatedLearning**: Main coordinator that manages the entire FL process

## Features

### Dataset Support
- **CSV datasets**: Automatic preprocessing for tabular data with classification/regression detection
- **MNIST**: Handwritten digit recognition with CNN architecture
- **CIFAR-10**: Object recognition with deep CNN architecture

### Data Distribution Strategies
- **IID (Independent and Identically Distributed)**: Random data distribution across devices
- **Non-IID**: Realistic data distribution where devices have different data characteristics
  - For image datasets: Each device gets samples from 2 classes
  - For tabular data: Sorted distribution based on target values

### Model Architectures
- **Tabular data**: Dense neural networks with dropout regularization
- **Image data**: Convolutional neural networks optimized for each dataset
- **Automatic architecture selection**: Based on dataset type and problem (classification/regression)

### Training Features
- **FedAvg algorithm**: Weighted averaging based on local dataset sizes
- **Configurable local training**: Adjustable epochs and batch sizes
- **Communication cost tracking**: Monitor data transfer overhead
- **Model persistence**: Save and load trained models

## Usage Examples

### Basic CSV Classification

```python
from core.federated_learning import FederatedLearning

# Initialize FL system
fl = FederatedLearning(
    num_devices=10,
    dataset='csv',
    csv_path='data/my_dataset.csv',
    target_column='label',
    data_distribution='iid',
    local_epochs=5,
    batch_size=32
)

# Train the model
history = fl.train(num_rounds=50, verbose=True)

# Get results
print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
print(f"Communication cost: {fl.get_communication_cost():.2f} MB")
```

### MNIST with Non-IID Distribution

```python
# Initialize with MNIST dataset
fl = FederatedLearning(
    num_devices=20,
    dataset='mnist',
    data_distribution='non_iid',
    local_epochs=3,
    batch_size=64
)

# Train and evaluate
history = fl.train(num_rounds=100)

# Save the trained model
fl.save_model('models/mnist_federated.h5')
```

### Custom CSV Regression

```python
# For regression problems
fl = FederatedLearning(
    num_devices=5,
    dataset='csv',
    csv_path='data/regression_data.csv',
    target_column='price',
    data_distribution='iid'
)

# The system automatically detects regression and configures accordingly
history = fl.train(num_rounds=30)
```

## API Reference

### FederatedLearning Class

#### Constructor Parameters
- `num_devices` (int): Number of participating devices
- `dataset` (str): Dataset type ('csv', 'mnist', 'cifar10')
- `data_distribution` (str): Distribution strategy ('iid', 'non_iid')
- `local_epochs` (int): Number of local training epochs
- `batch_size` (int): Mini-batch size for local training
- `csv_path` (str): Path to CSV file (required for CSV datasets)
- `target_column` (str): Target column name (optional for CSV)

#### Key Methods
- `train(num_rounds, verbose)`: Train the federated model
- `run_round()`: Execute a single FL round
- `get_communication_cost()`: Get total communication overhead
- `get_device_data_distribution()`: Get data distribution across devices
- `save_model(filepath)`: Save the global model
- `load_model(filepath)`: Load a saved model

### LocalDevice Class

#### Constructor Parameters
- `device_id` (int): Unique device identifier
- `local_epochs` (int): Local training epochs
- `batch_size` (int): Local batch size

#### Key Methods
- `set_data(x_data, y_data)`: Set local training data
- `train_local_model(global_weights, model_architecture)`: Perform local training

### GlobalAggregator Class

#### Key Methods
- `set_global_model(model)`: Set the global model architecture
- `federated_averaging(local_weights_list, sample_counts)`: Aggregate local updates

## Data Preprocessing

### CSV Datasets
The system automatically handles:
- **Categorical encoding**: Label encoding for categorical features
- **Feature scaling**: StandardScaler for numerical features
- **Target processing**: Automatic classification/regression detection
- **Train/test splitting**: 80/20 split with stratification

### Image Datasets
- **Normalization**: Pixel values scaled to [0, 1]
- **Reshaping**: Proper tensor shapes for CNN input
- **One-hot encoding**: Multi-class labels for classification

## Error Handling

The module includes custom exceptions:
- `FederatedLearningError`: Base exception for FL operations
- `DataDistributionError`: Data distribution failures
- `ModelTrainingError`: Training-related errors
- `DatasetError`: Dataset loading/processing errors

## Performance Considerations

### Memory Management
- **Lazy loading**: Data loaded only when needed
- **Model cloning**: Efficient device model creation
- **Gradient cleanup**: Automatic memory cleanup after training

### Communication Efficiency
- **Model compression**: Efficient weight serialization
- **Selective updates**: Only necessary parameters transmitted
- **Cost tracking**: Monitor communication overhead

### Scalability
- **Configurable devices**: Support for varying numbers of devices
- **Batch processing**: Efficient mini-batch training
- **Parallel training**: Device training can be parallelized

## Integration with Research Framework

This federated learning implementation is designed to integrate with:
- **Network topology models**: Device positioning and connectivity
- **Channel models**: Communication quality assessment
- **IRS optimization**: Intelligent reflecting surface coordination
- **MADRL agents**: Multi-agent reinforcement learning

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_federated_learning.py -v
```

The tests cover:
- Data loading and preprocessing
- Model architecture creation
- Local device training
- Global aggregation
- Different data distributions
- Error handling scenarios

## Example Scripts

See the `examples/` directory for complete usage examples:
- `federated_learning_demo.py`: Comprehensive demonstration
- `csv_classification_example.py`: CSV dataset usage
- `mnist_non_iid_example.py`: Non-IID distribution example

## Requirements

- TensorFlow >= 2.0
- NumPy >= 1.19
- Pandas >= 1.0
- Scikit-learn >= 0.24

## Citation

If you use this federated learning implementation in your research, please cite:

```bibtex
@article{federated_learning_irs_2025,
  title={Federated Learning with Intelligent Reflecting Surface and Multi-Agent Deep Reinforcement Learning for 6G Green IoT Networks},
  author={Research Team},
  journal={IEEE Transactions on Communications},
  year={2025}
}
```