from flax import linen as nn
from typing import Sequence
import jax
import jax.numpy as jnp

class DeepONet(nn.Module):
    branch_layers: Sequence[int]  # MLP layers after CNN
    trunk_layers: Sequence[int]
    output_dim: int = 1
    dropout_rate: float = 0.01

    @nn.compact
    def __call__(self, branch_x, trunk_x, training=False):
        init = nn.initializers.glorot_normal()
        
        # Ensure branch_x has correct shape: (batch_size, 17, 17, 1)
        if branch_x.ndim == 3:
            branch_x = jnp.expand_dims(branch_x, axis=0)
        elif branch_x.ndim == 2:
            branch_x = branch_x[..., None][None, ...]
        
        # print(f"Branch input shape: {branch_x.shape}")  # Debugging line
        # print(f"Trunk input shape: {trunk_x.shape}")  # Debugging line

        input_t = branch_x.reshape(branch_x.shape[0], -1)  # Reshape to (batch_size, 17*17, 1)
        # CNN Branch Network
        # Conv layers with 2D kernels
        branch_x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(branch_x)
        branch_x = nn.relu(branch_x)
        # branch_x = nn.max_pool(branch_x, window_shape=(2, 2), strides=(2, 2))
        branch_x = nn.avg_pool(branch_x, window_shape=(2, 2), strides=(2, 2))

        
        branch_x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(branch_x)
        branch_x = nn.relu(branch_x)
        # branch_x = nn.max_pool(branch_x, window_shape=(2, 2), strides=(2, 2))
        branch_x = nn.avg_pool(branch_x, window_shape=(2, 2), strides=(2, 2))

        
        branch_x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME')(branch_x)
        branch_x = nn.relu(branch_x)
        
        # Flatten for MLP layers
        branch_x = branch_x.reshape(branch_x.shape[0], -1)
        
        # Dropout at CNN-to-MLP transition
        branch_x = nn.Dropout(rate=self.dropout_rate)(branch_x, deterministic=not training)
        
        # MLP layers after CNN
        for i, features in enumerate(self.branch_layers[:-1]):
            branch_x = nn.Dense(features, kernel_init=init)(branch_x)
            branch_x = nn.tanh(branch_x)
        
        branch_x = nn.Dense(self.branch_layers[-1], kernel_init=init)(branch_x)
        
        # Trunk Network (same as before)
        if trunk_x.ndim == 1:
            trunk_x = jnp.expand_dims(trunk_x, axis=0)
        
        for i, features in enumerate(self.trunk_layers):
            trunk_x = nn.Dense(features, kernel_init=init)(trunk_x)
            trunk_x = nn.tanh(trunk_x)
        
        # DeepONet combination
        result = jnp.einsum('ij,kj->ik', branch_x, trunk_x)
        
        # Add bias
        bias = self.param('output_bias', nn.initializers.zeros, (self.output_dim,))
        result += bias
        
        # return input_t + 0.01*result
        return result
