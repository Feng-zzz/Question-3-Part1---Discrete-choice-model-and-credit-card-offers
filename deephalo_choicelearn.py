"""
Deep Context-Dependent Choice Model (DeepHalo) Implementation
Based on Zhang et al. (2025) "Deep Context-Dependent Choice Model"

This implementation is designed to be compatible with the choice-learn framework
using TensorFlow and TensorFlow Probability.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class NonlinearTransformation(layers.Layer):
    """
    Nonlinear transformation module for context effects.
    
    Implements the head-specific nonlinear transformations ϕ^l_h
    as described in Section 3.2 of the paper.
    """
    def __init__(self, H, embed=128, dropout=0.0, name=None):
        """
        Args:
            H: Number of interaction heads
            embed: Embedding dimension
            dropout: Dropout rate
        """
        super(NonlinearTransformation, self).__init__(name=name)
        self.H = H
        self.embed = embed
        
        # Two-layer MLP as described in Appendix B.2
        self.fc1 = layers.Dense(embed * H, use_bias=True, name='fc1')
        self.fc2 = layers.Dense(embed, use_bias=True, name='fc2')
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, X, training=False):
        """
        Args:
            X: Input tensor of shape (batch_size, n_items, embed)
            training: Boolean for dropout
            
        Returns:
            Output tensor of shape (batch_size, n_items, H, embed)
        """
        batch_size = tf.shape(X)[0]
        n_items = tf.shape(X)[1]
        
        # First linear layer: (B, n, embed) -> (B, n, H*embed)
        out = self.fc1(X)
        
        # Reshape to (B, n, H, embed)
        out = tf.reshape(out, [batch_size, n_items, self.H, self.embed])
        
        # Apply ReLU activation
        out = tf.nn.relu(out)
        
        # Apply dropout
        out = self.dropout(out, training=training)
        
        # Second linear layer: (B, n, H, embed) -> (B, n, H, embed)
        # Need to reshape for the dense layer
        out_reshaped = tf.reshape(out, [batch_size * n_items * self.H, self.embed])
        out_reshaped = self.fc2(out_reshaped)
        out = tf.reshape(out_reshaped, [batch_size, n_items, self.H, self.embed])
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out


class DeepHaloFeatureBased(keras.Model):
    """
    Deep Context-Dependent Choice Model (Feature-based version)
    
    Implements the architecture described in Section 3.2 of Zhang et al. (2025).
    This model captures context effects through layered aggregation and nonlinear
    transformations with explicit control over interaction order.
    
    The model follows equations (4) and (5) from the paper:
        Z̄^l = (1/|S|) Σ_k W^l z^{l-1}_k
        z^l_j = z^{l-1}_j + (1/H) Σ_h Z̄^l_h · ϕ^l_h(z^0_j)
    """
    
    def __init__(self, 
                 input_dim,
                 n_items_max,
                 embed_dim=128,
                 n_layers=4,
                 n_heads=8,
                 dropout=0.0,
                 name='DeepHalo'):
        """
        Args:
            input_dim: Dimension of input features
            n_items_max: Maximum number of items in a choice set
            embed_dim: Embedding dimension (d in the paper)
            n_layers: Number of layers (L in the paper)
            n_heads: Number of interaction heads (H in the paper)
            dropout: Dropout rate
        """
        super(DeepHaloFeatureBased, self).__init__(name=name)
        
        self.input_dim = input_dim
        self.n_items_max = n_items_max
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout
        
        # Basic encoder χ: R^dx -> R^d (Equation 15 in Appendix B.2)
        self.basic_encoder = keras.Sequential([
            layers.Dense(embed_dim, activation='relu', name='encoder_1'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim, activation='relu', name='encoder_2'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim, name='encoder_3')
        ], name='basic_encoder')
        
        self.enc_norm = layers.LayerNormalization(epsilon=1e-6, name='enc_norm')
        
        # Aggregation layers: W^l for each layer l
        self.aggregate_linears = []
        for l in range(n_layers):
            self.aggregate_linears.append(
                layers.Dense(n_heads, use_bias=False, name=f'aggregate_{l}')
            )
        
        # Nonlinear transformation modules: ϕ^l_h for each layer l
        self.nonlinear_transforms = []
        for l in range(n_layers):
            self.nonlinear_transforms.append(
                NonlinearTransformation(n_heads, embed_dim, dropout, 
                                       name=f'nonlinear_{l}')
            )
        
        # Final linear layer to compute utilities
        self.final_linear = layers.Dense(1, name='final_linear')
        
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary containing:
                - 'features': Tensor of shape (batch_size, n_items, input_dim)
                - 'availability': Tensor of shape (batch_size, n_items) with 1 for available items
            training: Boolean for dropout
            
        Returns:
            Dictionary containing:
                - 'logits': Utility scores of shape (batch_size, n_items)
                - 'probabilities': Choice probabilities (batch_size, n_items)
                - 'log_probabilities': Log probabilities (batch_size, n_items)
        """
        features = inputs['features']  # (B, n, input_dim)
        availability = inputs.get('availability')  # (B, n)
        
        batch_size = tf.shape(features)[0]
        n_items = tf.shape(features)[1]
        
        # Compute the number of available items per choice set
        if availability is not None:
            lengths = tf.reduce_sum(availability, axis=1)  # (B,)
        else:
            lengths = tf.cast(tf.fill([batch_size], n_items), tf.float32)
        
        # Step 1: Basic encoding χ(x_j)
        # Reshape for encoder: (B*n, input_dim)
        features_flat = tf.reshape(features, [-1, self.input_dim])
        Z = self.basic_encoder(features_flat, training=training)
        Z = tf.reshape(Z, [batch_size, n_items, self.embed_dim])
        Z = self.enc_norm(Z)  # z^0_j in the paper
        
        # Keep original embeddings for modulation
        X = Z  # z^0 is used in ϕ^l_h(z^0_j)
        
        # Step 2: Layer-wise context aggregation
        for layer_idx in range(self.n_layers):
            # Aggregate: Z̄^l = (1/|S|) Σ_k W^l z^{l-1}_k
            # Z shape: (B, n, embed)
            
            # Apply linear transformation: (B, n, embed) -> (B, n, H)
            Z_transformed = self.aggregate_linears[layer_idx](Z)  # (B, n, H)
            
            # Apply availability mask before aggregation
            if availability is not None:
                mask = tf.expand_dims(availability, axis=-1)  # (B, n, 1)
                Z_transformed = Z_transformed * mask
            
            # Sum over items: (B, H)
            Z_sum = tf.reduce_sum(Z_transformed, axis=1)
            
            # Divide by number of available items: (B, H)
            lengths_expanded = tf.expand_dims(lengths, axis=-1)  # (B, 1)
            Z_bar = Z_sum / lengths_expanded  # (B, H)
            
            # Reshape for broadcasting: (B, 1, H, 1)
            Z_bar = tf.reshape(Z_bar, [batch_size, 1, self.n_heads, 1])
            
            # Apply nonlinear transformation: phi = ϕ^l_h(z^0_j)
            phi = self.nonlinear_transforms[layer_idx](X, training=training)  # (B, n, H, embed)
            
            # Apply availability mask to phi
            if availability is not None:
                mask_phi = tf.reshape(availability, [batch_size, n_items, 1, 1])
                phi = phi * mask_phi
            
            # Context modulation: phi * Z̄^l
            # Z_bar: (B, 1, H, 1), phi: (B, n, H, embed)
            modulated = phi * Z_bar  # (B, n, H, embed)
            
            # Average over heads: (B, n, embed)
            modulated = tf.reduce_sum(modulated, axis=2) / self.n_heads
            
            # Residual connection: z^l_j = z^{l-1}_j + modulated
            Z = Z + modulated
        
        # Step 3: Compute utilities
        # (B, n, embed) -> (B, n, 1) -> (B, n)
        Z_flat = tf.reshape(Z, [-1, self.embed_dim])
        logits_flat = self.final_linear(Z_flat)
        logits = tf.reshape(logits_flat, [batch_size, n_items])
        
        # Apply availability mask to logits
        if availability is not None:
            # Set logits of unavailable items to large negative value
            logits = tf.where(
                tf.cast(availability, tf.bool),
                logits,
                tf.fill(tf.shape(logits), -1e9)
            )
        
        # Compute probabilities using softmax
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probabilities = tf.nn.log_softmax(logits, axis=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'log_probabilities': log_probabilities
        }
    
    def compute_negative_log_likelihood(self, inputs, choices):
        """
        Compute negative log-likelihood loss.
        
        Args:
            inputs: Input dictionary for the model
            choices: One-hot encoded choices of shape (batch_size, n_items)
            
        Returns:
            Scalar loss value
        """
        outputs = self(inputs, training=True)
        log_probs = outputs['log_probabilities']
        
        # Compute negative log-likelihood
        # log_probs: (B, n), choices: (B, n)
        nll = -tf.reduce_sum(choices * log_probs, axis=-1)  # (B,)
        return tf.reduce_mean(nll)


class DeepHaloFeatureless(keras.Model):
    """
    Deep Context-Dependent Choice Model (Featureless version)
    
    This is the specialized version for featureless settings as described
    in Section 4.2 of the paper. It uses residual blocks with polynomial
    activations to efficiently capture high-order interactions.
    
    Implements the recursion from Appendix A.2.3:
        y^l = y^{l-1} + Θ^l σ(y^{l-1})
    where σ is a quadratic activation function.
    """
    
    def __init__(self,
                 n_items,
                 depth=5,
                 width=None,
                 block_types=None,
                 name='DeepHaloFeatureless'):
        """
        Args:
            n_items: Number of items in the universe
            depth: Number of layers (L in the paper)
            width: Hidden dimension (J' in the paper). If None, uses n_items
            block_types: List of block types ('exa' or 'qua'). If None, uses 'qua' for all
        """
        super(DeepHaloFeatureless, self).__init__(name=name)
        
        self.n_items = n_items
        self.depth = depth
        self.width = width if width is not None else n_items
        
        if block_types is None:
            block_types = ['qua'] * (depth - 1)
        
        assert len(block_types) == depth - 1, \
            f"block_types must have length {depth-1}, got {len(block_types)}"
        
        self.block_types = block_types
        
        # Input layer: J -> J'
        self.in_linear = layers.Dense(self.width, use_bias=False, name='in_linear')
        
        # Output layer: J' -> J
        self.out_linear = layers.Dense(n_items, use_bias=False, name='out_linear')
        
        # Build residual blocks
        self.blocks = []
        for i, block_type in enumerate(block_types):
            if block_type == 'exa':
                self.blocks.append(
                    ExaResBlock(n_items, self.width, name=f'block_{i}_exa')
                )
            elif block_type == 'qua':
                self.blocks.append(
                    QuaResBlock(self.width, name=f'block_{i}_qua')
                )
            else:
                raise ValueError(f"Unknown block type: {block_type}")
    
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary containing:
                - 'items': One-hot encoded items of shape (batch_size, n_items)
                - 'availability': Binary mask of shape (batch_size, n_items)
            
        Returns:
            Dictionary with probabilities and logits
        """
        items = inputs['items']  # (B, J) - indicator vector e_S
        availability = inputs.get('availability', items)  # (B, J)
        
        # Store original input for ExaResBlock
        e0 = items
        
        # Input transformation
        e = self.in_linear(items)  # (B, J')
        
        # Apply residual blocks
        for block in self.blocks:
            if isinstance(block, ExaResBlock):
                e = block([e, e0], training=training)
            else:
                e = block(e, training=training)
        
        # Output transformation
        logits = self.out_linear(e)  # (B, J)
        
        # Apply availability mask
        logits = tf.where(
            tf.cast(availability, tf.bool),
            logits,
            tf.fill(tf.shape(logits), -1e9)
        )
        
        # Compute probabilities
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probabilities = tf.nn.log_softmax(logits, axis=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'log_probabilities': log_probabilities
        }
    
    def compute_negative_log_likelihood(self, inputs, choices):
        """Compute negative log-likelihood loss."""
        outputs = self(inputs, training=True)
        log_probs = outputs['log_probabilities']
        nll = -tf.reduce_sum(choices * log_probs, axis=-1)
        return tf.reduce_mean(nll)


class ExaResBlock(layers.Layer):
    """
    Exact Residual Block for featureless model.
    
    Implements: z^l = W^l_main(z^{l-1} ⊙ W^l_act(e_0)) + z^{l-1}
    where ⊙ is element-wise multiplication.
    """
    
    def __init__(self, input_dim, hidden_dim, name=None):
        super(ExaResBlock, self).__init__(name=name)
        self.linear_main = layers.Dense(hidden_dim, use_bias=False, name='linear_main')
        self.linear_act = layers.Dense(hidden_dim, use_bias=False, name='linear_act')
    
    def call(self, inputs, training=False):
        """
        Args:
            inputs: List [z_prev, z0] where
                - z_prev: Previous layer output (B, hidden_dim)
                - z0: Original input (B, input_dim)
        """
        z_prev, z0 = inputs
        
        # z_prev * W^l_act(z0)
        activation = self.linear_act(z0)
        modulated = z_prev * activation
        
        # W^l_main(modulated) + z_prev
        output = self.linear_main(modulated)
        return output + z_prev


class QuaResBlock(layers.Layer):
    """
    Quadratic Residual Block for featureless model.
    
    Implements: z^l = W^l(z^{l-1})^2 + z^{l-1}
    This enables exponential growth in interaction order.
    """
    
    def __init__(self, dim, name=None):
        super(QuaResBlock, self).__init__(name=name)
        self.linear = layers.Dense(dim, use_bias=False, name='linear')
    
    def call(self, x, training=False):
        """
        Args:
            x: Input tensor (B, dim)
        """
        # Quadratic activation
        x_squared = tf.square(x)
        
        # Linear transformation and residual
        return self.linear(x_squared) + x


# Utility functions for compatibility with choice-learn

def create_deephalo_model(model_type='feature_based', **kwargs):
    """
    Factory function to create DeepHalo models.
    
    Args:
        model_type: Either 'feature_based' or 'featureless'
        **kwargs: Model-specific parameters
        
    Returns:
        DeepHalo model instance
    """
    if model_type == 'feature_based':
        return DeepHaloFeatureBased(**kwargs)
    elif model_type == 'featureless':
        return DeepHaloFeatureless(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def masked_log_softmax(logits, mask):
    """
    Compute log-softmax with masking for unavailable items.
    
    Args:
        logits: Tensor of shape (batch_size, n_items)
        mask: Binary tensor of shape (batch_size, n_items) where 1 = available
        
    Returns:
        Log probabilities of shape (batch_size, n_items)
    """
    # Set logits of unavailable items to large negative value
    masked_logits = tf.where(
        tf.cast(mask, tf.bool),
        logits,
        tf.fill(tf.shape(logits), -1e9)
    )
    
    return tf.nn.log_softmax(masked_logits, axis=-1)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("DeepHalo Model Implementation Test")
    print("=" * 60)
    
    # Test feature-based model
    print("\n1. Testing Feature-Based DeepHalo Model")
    print("-" * 60)
    
    batch_size = 32
    n_items = 10
    input_dim = 25
    
    # Create dummy data
    features = tf.random.normal([batch_size, n_items, input_dim])
    availability = tf.ones([batch_size, n_items])
    choices = tf.one_hot(tf.random.uniform([batch_size], 0, n_items, dtype=tf.int32), n_items)
    
    # Create model
    model_fb = DeepHaloFeatureBased(
        input_dim=input_dim,
        n_items_max=n_items,
        embed_dim=64,
        n_layers=3,
        n_heads=4,
        dropout=0.1
    )
    
    # Forward pass
    inputs_fb = {'features': features, 'availability': availability}
    outputs_fb = model_fb(inputs_fb, training=False)
    
    print(f"Input features shape: {features.shape}")
    print(f"Output logits shape: {outputs_fb['logits'].shape}")
    print(f"Output probabilities shape: {outputs_fb['probabilities'].shape}")
    print(f"Probabilities sum (should be ~1.0): {tf.reduce_sum(outputs_fb['probabilities'][0]):.4f}")
    
    # Compute loss
    loss_fb = model_fb.compute_negative_log_likelihood(inputs_fb, choices)
    print(f"Negative log-likelihood: {loss_fb:.4f}")
    
    # Test featureless model
    print("\n2. Testing Featureless DeepHalo Model")
    print("-" * 60)
    
    n_items_fl = 20
    items_onehot = tf.one_hot(
        tf.random.uniform([batch_size, 15], 0, n_items_fl, dtype=tf.int32),
        n_items_fl
    )
    items_indicator = tf.reduce_sum(items_onehot, axis=1)  # (B, 20)
    
    # Create model
    model_fl = DeepHaloFeatureless(
        n_items=n_items_fl,
        depth=4,
        width=30,
        block_types=['qua', 'qua', 'qua']
    )
    
    # Forward pass
    inputs_fl = {'items': items_indicator}
    outputs_fl = model_fl(inputs_fl, training=False)
    
    print(f"Input items shape: {items_indicator.shape}")
    print(f"Output logits shape: {outputs_fl['logits'].shape}")
    print(f"Output probabilities shape: {outputs_fl['probabilities'].shape}")
    print(f"Probabilities sum: {tf.reduce_sum(outputs_fl['probabilities'][0]):.4f}")
    
    # Compute loss
    choices_fl = tf.one_hot(tf.argmax(items_indicator, axis=1), n_items_fl)
    loss_fl = model_fl.compute_negative_log_likelihood(inputs_fl, choices_fl)
    print(f"Negative log-likelihood: {loss_fl:.4f}")
    
    print("\n" + "=" * 60)
    print("Model implementation test completed successfully!")
    print("=" * 60)
