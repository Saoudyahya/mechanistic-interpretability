# Switch Transformer Visualizer with Mechanistic Interpretability

A Python tool for visualizing and analyzing Switch Transformer activations using sparse autoencoders for mechanistic interpretability.

## Overview

This tool provides visualization and analysis capabilities for Switch Transformer models, focusing on:
- Neuron activation patterns
- Expert gate behavior
- Feature extraction through sparse autoencoders
- Mechanistic interpretability analysis

## Installation

### Prerequisites
```bash
pip install torch
pip install transformers
pip install matplotlib
pip install seaborn
pip install numpy
```

### Dependencies
- Python 3.7+
- PyTorch
- Transformers
- Matplotlib
- Seaborn
- NumPy

## Quick Start

```python
from switch_transformer_viz import analyze_texts

# Configure model parameters
model_config = {
    'input_dim': 256,
    'num_experts': 32,
    'hidden_dim': 512,
    'num_layers': 4
}

# Analyze texts
texts = [
    "The strategic merger between the two companies.",
    "Climate change poses significant challenges."
]

# Generate visualizations
analyze_texts(texts, model_config)
```

## Core Components

### SwitchTransformerVisualizer

Main class for visualization and analysis:
- Initializes model and tokenizer
- Manages activation collection
- Handles visualization generation
- Coordinates autoencoder training and analysis

### SparseAutoencoder

Neural network for feature extraction:
- Reduces dimensionality of activation data
- Enforces sparsity through L1 regularization
- Provides interpretable features
- Uses custom loss function combining reconstruction and sparsity

## Visualization Features

1. **Neuron Activations**
   - Heatmap of original neuron activations
   - Token-wise activation patterns
   - Top neuron analysis

2. **Expert Gate Analysis**
   - Gate score distributions
   - Expert routing patterns
   - Top expert visualization

3. **Feature Analysis**
   - Learned feature representations
   - Feature importance metrics
   - Feature correlation matrix
   - Activation patterns of top features

## Advanced Usage

### Custom Autoencoder Training

```python
visualizer = SwitchTransformerVisualizer(model_config)
activations, tokens = visualizer.get_neuron_activations(text)

# Train autoencoder with custom parameters
losses = visualizer.train_autoencoder(
    activation_data=activations['expert_output'],
    hidden_dim=50,
    epochs=100,
    batch_size=32
)
```

### Feature Analysis

```python
# Analyze learned features
encoded_features, importance, correlations = visualizer.analyze_features(activation_matrix)

# Access specific feature patterns
top_features = torch.topk(importance, k=5).indices
feature_patterns = encoded_features[:, top_features]
```

## Visualization Parameters

Key parameters that can be adjusted:
- `top_k`: Number of top neurons/features to display (default: 5)
- `layer_idx`: Which transformer layer to analyze (default: 0)
- `expert_idx`: Which expert to focus on (default: 0)
- `hidden_dim`: Dimensionality of autoencoder features (default: 50)

## Model Configuration

The model configuration dictionary requires:
- `input_dim`: Input dimension size
- `num_experts`: Number of experts in the model
- `hidden_dim`: Hidden layer dimension
- `num_layers`: Number of transformer layers

## Output Examples

The tool generates several visualizations:
1. Original neuron activation heatmaps
2. Expert gate score distributions
3. Learned feature representations
4. Feature correlation matrices
5. Feature importance plots
6. Top feature activation patterns

## Best Practices

1. **Data Preparation**
   - Use representative text samples
   - Keep sequences within reasonable length
   - Clean and preprocess input text

2. **Autoencoder Training**
   - Adjust sparsity weights based on needs
   - Monitor training loss
   - Use appropriate batch sizes

3. **Visualization**
   - Focus on relevant layers/experts
   - Adjust colormap ranges for clarity
   - Save important visualizations

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Citation

If you use this tool in your research, please cite:
```
@software{switch_transformer_viz,
  title={Switch Transformer Visualizer with Mechanistic Interpretability},
  author={[Saoud Yahya]},
  year={2025},
  url={[[Repository URL](https://github.com/Saoudyahya/mechanistic-interpretability/edit/main/README.md)]}
}
```

## Support

For issues and questions:
1. Check existing issues in the repository
2. Create a new issue with:
   - Clear description
   - Minimal example
   - Error messages if applicable
   - System information
   
