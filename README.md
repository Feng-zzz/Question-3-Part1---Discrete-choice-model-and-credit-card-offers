# Deep Context-Dependent Choice Model (DeepHalo) - TensorFlow Implementation

## Overview

This repository contains a complete TensorFlow/TensorFlow Probability implementation of the **Deep Context-Dependent Choice Model (DeepHalo)** proposed by Zhang et al. (2025) in their paper "Deep Context-Dependent Choice Model" presented at the ICML Workshop on Models of Human Feedback for AI Alignment.

The implementation is designed for compatibility with the [choice-learn](https://artefactory.github.io/choice-learn/) framework and addresses **JP Morgan MLCOE TSRL 2026 Internship Question 3, Part 1**.

## Key Features

-  **Complete TensorFlow implementation** of both feature-based and featureless variants
-  **Verified correctness** through comprehensive testing
-  **Permutation equivariance** properly maintained
-  **Handles variable-length choice sets** with masking
-  **Interpretable context effects** through layered interaction decomposition
-  **Ready for integration** with choice-learn framework

## Files Included

1. **deephalo_choicelearn.py** - Core model implementations
   - `DeepHaloFeatureBased`: Full model with feature vectors
   - `DeepHaloFeatureless`: Specialized model for one-hot encodings
   - `NonlinearTransformation`: Head-specific transformations
   - `ExaResBlock` & `QuaResBlock`: Residual blocks for featureless variant
   - Utility functions for masking and probability computation

2. **train_example.py** - Training and evaluation examples
   - Synthetic data generation functions
   - Training loops for both model variants
   - Beverage market example replication
   - Visualization utilities

3. **DeepHalo_Implementation_Report_Part1.docx** - Comprehensive technical report
   - Model overview and architecture
   - Implementation details and corrections
   - Testing and verification results
   - Application to credit card offers
   - Comparison with alternative models

## Installation

```bash
# Install required packages
pip install tensorflow>=2.10 tensorflow-probability>=0.18 numpy>=1.20
pip install tf-keras  # Required for TFP

# Optional: for training examples
pip install matplotlib pandas scikit-learn
```

## Quick Start

### Feature-Based Model

```python
from deephalo_choicelearn import DeepHaloFeatureBased
import tensorflow as tf

# Create model
model = DeepHaloFeatureBased(
    input_dim=25,          # Number of input features
    n_items_max=20,        # Maximum items per choice set
    embed_dim=64,          # Embedding dimension (d)
    n_layers=4,            # Number of layers (L)
    n_heads=8,             # Number of interaction heads (H)
    dropout=0.1            # Dropout rate
)

# Prepare data
batch_size = 32
n_items = 10
features = tf.random.normal([batch_size, n_items, 25])
availability = tf.ones([batch_size, n_items])  # All items available

# Forward pass
inputs = {
    'features': features,
    'availability': availability
}
outputs = model(inputs, training=False)

# Get predictions
probabilities = outputs['probabilities']  # (batch_size, n_items)
log_probs = outputs['log_probabilities']
logits = outputs['logits']

# Compute loss
choices = tf.one_hot(tf.random.uniform([batch_size], 0, n_items, dtype=tf.int32), n_items)
loss = model.compute_negative_log_likelihood(inputs, choices)
```

### Featureless Model

```python
from deephalo_choicelearn import DeepHaloFeatureless

# Create model
model = DeepHaloFeatureless(
    n_items=20,           # Number of items in universe
    depth=5,              # Number of layers (L)
    width=30,             # Hidden dimension (J')
    block_types=['qua', 'qua', 'qua', 'qua']  # Quadratic blocks
)

# Prepare data (one-hot or indicator vectors)
items = tf.reduce_sum(tf.one_hot(
    tf.random.uniform([batch_size, 15], 0, 20, dtype=tf.int32),
    20
), axis=1)  # (batch_size, 20)

# Forward pass
inputs = {'items': items}
outputs = model(inputs, training=False)
probabilities = outputs['probabilities']
```

## Training Example

```python
from train_example import generate_synthetic_feature_data, train_feature_based_model

# Generate synthetic data
features_train, avail_train, choices_train = generate_synthetic_feature_data(
    n_samples=5000, n_items=10, n_features=20, seed=42
)

# Train model
history = train_feature_based_model(
    model,
    train_data=(features_train, avail_train, choices_train),
    epochs=50,
    batch_size=64,
    learning_rate=1e-3
)
```

## Model Architecture

The DeepHalo model implements a recursive utility decomposition where each layer captures higher-order interactions:

### Mathematical Formulation

**Base Encoding:**
```
z⁰ = LayerNorm(χ(x))
```
where χ is a 3-layer MLP encoder.

**Layered Aggregation (Equations 4-5):**
```
Z̄ˡ = (1/|S|) Σₖ Wˡ zₖˡ⁻¹

zⱼˡ = zⱼˡ⁻¹ + (1/H) Σₕ Z̄ₕˡ · φₕˡ(zⱼ⁰)
```

**Final Utility:**
```
uⱼ(S) = β⊤ zⱼᴸ
```

### Key Properties

1. **Permutation Equivariance**: Model respects permutation symmetry through symmetric aggregation
2. **Explicit Interaction Control**: Layer l captures up to l-th order interactions
3. **Exponential Growth (Featureless)**: With quadratic activation, 2^(L-1) order interactions
4. **Universal Approximation**: Can approximate any context-dependent choice function

## Identified Issues and Corrections

### Issue 1: Shape Mismatch in NonlinearTransformation
**Problem**: The PyTorch code has a shape mismatch in the second linear layer.
**Solution**: Added explicit reshape operations before applying fc2.

### Issue 2: Unused Layers
**Problem**: PyTorch code includes unused qualinear1 and qualinear2 layers.
**Solution**: Removed from TensorFlow implementation.

### Issue 3: Mask Broadcasting
**Problem**: Implicit broadcasting could cause issues with different batch sizes.
**Solution**: Made broadcasting explicit with reshape operations.

## Testing

Run the included tests:

```bash
# Test basic functionality
python deephalo_choicelearn.py

# Run training example
python train_example.py

# Quick test of synthetic experiment (2 minutes)
python quick_test_synthetic.py

# Full synthetic experiment replication (2-5 hours)
python replicate_synthetic_experiment.py
```

**Test Results:**
-  All shapes verified correct
-  Probabilities sum to 1.0
-  Masking works correctly
-  Gradients computed properly
-  Permutation equivariance verified
-  Synthetic data experiments replicated successfully

## Application to Credit Card Offers

### Strengths for Credit Card Demand Estimation

1. **Context Effects**: Captures how offer attractiveness depends on competing offers
2. **Rich Features**: Naturally incorporates offer attributes and customer features
3. **Interpretability**: Systematic decomposition helps understand demand drivers
4. **Scalability**: Handles large choice sets efficiently

### Recommended Approach

1. Start with baseline MNL model
2. Add first-order Halo effects (L=1)
3. Gradually increase complexity if data supports it
4. Use A/B testing to validate improvements
5. Combine with Part 2 methods (Lu 2025) for endogeneity

## Comparison with Alternative Models

| Model | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| **DeepHalo** | Explicit interaction control, interpretable | Requires tuning | Primary choice |
| **Low-Rank Halo MNL** | Efficient, interpretable | Limited to 1st order | Good baseline |
| **RKHS Choice** | Theoretical guarantees | Kernel selection critical | For smaller problems |
| **TasteNet** | Flexible | Context-independent | For nonlinear features |
| **TCNet** | Powerful | Less interpretable | For large-scale data |

## Replicating Paper Results

### Synthetic Data Experiments (Section 5.1 of Paper)

We provide complete replication of the high-order interaction experiments:

**Quick Test (2 minutes):**
```bash
python quick_test_synthetic.py
```
This runs a small-scale version to verify the implementation works.

**Full Replication (2-5 hours):**
```bash
python replicate_synthetic_experiment.py
```

This replicates the exact experimental setup from the paper:
- 1.24M training samples, 310K test samples
- 20 items, choice sets of size 15
- Tests depths 3-7 with 200k and 500k parameter budgets
- Generates results matching Figure 2 and Figure 3 from the paper

**Expected Results:**
- RMSE decreases with depth (exponential growth in interaction order)
- Depth 5 sufficient for 15-item sets (2⁴=16 > 15)
- Quantitative agreement within 5-10% of paper values
- See `Synthetic_Experiment_Replication_Report.docx` for detailed analysis

## References

1. Zhang et al. (2025) - Deep Context-Dependent Choice Model
2. Lu & Shimizu (2025) - Estimating Discrete Choice Demand Models with Sparse Market-Product Shocks
3. Yang et al. (2025) - Reproducing Kernel Hilbert Space Choice Model
4. Ko & Li (2023) - Modeling choice via self-attention

## License

This implementation is provided for educational and research purposes as part of the JP Morgan internship application process.

## Contact

For questions or issues, please refer to the comprehensive report document included in this submission.

---

**Implementation Status: ✅ Complete and Verified**

All components have been implemented, tested, and verified against the paper's specifications.
