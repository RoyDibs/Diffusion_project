from flax import linen as nn
from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from torch.utils import data
from functools import partial
import time
import optax
import scipy.io
import os
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from jax import jvp
import pickle
from sklearn import metrics
from termcolor import colored
import random
import numpy as np
import math
from scipy.interpolate import griddata

import sys
import os

from DataGenerator import DataGenerator
from DeepONet import DeepONet
from helper_fun import mse, mse_single, apply_net, step
from save_model import save_model_params, load_model_params

# Go up two levels to reach Diffusion_project root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the function
from visualize_res import visualize_res


seed = 0
# n_train = 100
batch_size = 75
# n_test = 100
n_sensors = 1
branch_layers = [128]
branch_input_features = 1
trunk_layers = [128, 128, 128]
trunk_input_features = 2
hidden_dim = 100
# p_test = 100
result_dir = './'
epochs = 80000

lr = 1e-3
transition_steps = 2000
decay_rate = 0.9


# Load the data
print(f"="*60)
print(f"LOADING DIFFUSION SIMULATION DATA")
print(f"="*60)

# Load the training dataset

def load_for_training(filename):
    """
    Quick load function that returns just inputs and outputs for training
    """
    
    loaded = np.load(filename, allow_pickle=True)
    
    inputs = jnp.array(loaded['inputs'])
    outputs = jnp.array(loaded['outputs'])
    time_indices = jnp.array(loaded['time_indices'])
    mesh_points = jnp.array(loaded['mesh_points'])
    
    print(f"Loaded training data:")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Outputs: {outputs.shape}")
    print(f"  Time Indices: {time_indices.shape}")
    print(f"  Mesh Points: {mesh_points.shape}")


    # data_dict = {
    #     'inputs': inputs,
    #     'outputs': outputs,
    #     'time_indices': time_indices,
    #     'mesh_points': mesh_points,
    #     'time_steps': time_steps,
    #     'input_shape': inputs.shape,
    #     'output_shape': outputs.shape,
    #     'target_size': inputs.shape[1],  # Should be 17
    #     'num_samples': inputs.shape[0],
    #     'start_timestep': time_indices[0, 0],
    #     'end_timestep': time_indices[-1, 1],
    # }
    
    return inputs, outputs, time_indices, mesh_points

# Usage for training
train_inputs, train_outputs, time_indices, mesh_points = load_for_training("cnn_training_dataset.npz")

xi = jnp.linspace(jnp.min(mesh_points[0]), jnp.max(mesh_points[0]), 17)
yi = jnp.linspace(jnp.min(mesh_points[1]), jnp.max(mesh_points[1]), 17)
Xi, Yi = jnp.meshgrid(xi, yi, indexing='ij')
cord_mesh = jnp.column_stack([Xi.flatten(), Yi.flatten()])

key = jax.random.PRNGKey(seed)
# keys = jax.random.split(key, 1)

print('cord_mesh shape:', cord_mesh.shape)
data_dataset = DataGenerator(train_inputs, cord_mesh, train_outputs, batch_size, key)

# data_data = iter(data_dataset)

# data_batch = next(data_data)

def loss_fn(model_fn, params, data_batch):
    
    inputs, outputs = data_batch
    s_in, cord = inputs

    x = cord[:, 0]
    y = cord[:, 1]

    s_pred = apply_net(model_fn, params, s_in, x, y)

    outputs = outputs.reshape(outputs.shape[0], -1)  # Ensure outputs are in the correct shape
    loss_value = mse(outputs, s_pred)

    return loss_value 

# def compute_ssim_loss(targets, predictions, grid_size=17):
#     """Structural similarity loss"""
    
#     batch_size = targets.shape[0]
#     ssim_loss = 0.0
    
#     for i in range(batch_size):
#         target_2d = targets[i].reshape(grid_size, grid_size)
#         pred_2d = predictions[i].reshape(grid_size, grid_size)
        
#         # Simple structural similarity (can use skimage.metrics.structural_similarity for better version)
#         target_mean = jnp.mean(target_2d)
#         pred_mean = jnp.mean(pred_2d)
        
#         target_var = jnp.var(target_2d)
#         pred_var = jnp.var(pred_2d)
        
#         covariance = jnp.mean((target_2d - target_mean) * (pred_2d - pred_mean))
        
#         # SSIM-like term
#         ssim = (2 * target_mean * pred_mean + 1e-6) / (target_mean**2 + pred_mean**2 + 1e-6) * \
#                (2 * covariance + 1e-6) / (target_var + pred_var + 1e-6)
        
#         ssim_loss += (1 - ssim)
    
#     return ssim_loss / batch_size

# def plot_training_samples(s_in, outputs, s_pred, outputs_flat, num_samples=5):
#     """Plot input, target, and prediction for diagnostic purposes"""
    
#     batch_size = min(num_samples, s_in.shape[0])
    
#     fig, axes = plt.subplots(3, batch_size, figsize=(4*batch_size, 12))
#     if batch_size == 1:
#         axes = axes.reshape(3, 1)
    
#     for i in range(batch_size):
#         # Input (current timestep)
#         input_spatial = s_in[i, :, :, 0]  # (17, 17)
#         im1 = axes[0, i].imshow(input_spatial, cmap='viridis', origin='upper')
#         axes[0, i].set_title(f'Input {i}\nRange: [{jnp.min(input_spatial):.3f}, {jnp.max(input_spatial):.3f}]')
#         plt.colorbar(im1, ax=axes[0, i])
        
#         # Target output (next timestep)
#         target_spatial = outputs[i, :, :, 0]  # (17, 17)
#         im2 = axes[1, i].imshow(target_spatial, cmap='viridis', origin='upper')
#         axes[1, i].set_title(f'Target {i}\nRange: [{jnp.min(target_spatial):.3f}, {jnp.max(target_spatial):.3f}]')
#         plt.colorbar(im2, ax=axes[1, i])
        
#         # Model prediction (reshaped to 17x17)
#         pred_spatial = s_pred[i].reshape(17, 17)  # (17, 17)
#         im3 = axes[2, i].imshow(pred_spatial, cmap='viridis', origin='upper')
#         axes[2, i].set_title(f'Prediction {i}\nRange: [{jnp.min(pred_spatial):.3f}, {jnp.max(pred_spatial):.3f}]')
#         plt.colorbar(im3, ax=axes[2, i])
        
#         # Print sample-specific diagnostics
#         input_flat = input_spatial.flatten()
#         target_flat = outputs_flat[i]
#         pred_flat = s_pred[i]
        
#         sample_mse = jnp.mean((target_flat - pred_flat)**2)
#         input_target_change = jnp.mean(jnp.abs(target_flat - input_flat))
        
#         print(f"Sample {i}:")
#         print(f"  Input → Target change: {input_target_change:.6f}")
#         print(f"  Sample MSE: {sample_mse:.6f}")
#         print(f"  Prediction mean: {jnp.mean(pred_flat):.6f}")
#         print(f"  Target mean: {jnp.mean(target_flat):.6f}")
        
#         # Check if prediction is reasonable
#         if jnp.std(pred_flat) < 1e-6:
#             print(f"  ⚠️ WARNING: Prediction {i} is nearly constant!")
        
#         if abs(jnp.mean(pred_flat) - jnp.mean(target_flat)) > jnp.std(target_flat):
#             print(f"  ⚠️ WARNING: Prediction {i} mean very different from target!")
    
#     plt.tight_layout()
    
#     plt.savefig("training_diagnostic.png", dpi=150, bbox_inches='tight')
#     print("Saved diagnostic plot: training_diagnostic.png")
    
#     plt.close()

# Initialize model and params
# make sure trunk_layers and branch_layers are lists
trunk_layers = [trunk_layers] if isinstance(trunk_layers, int) else trunk_layers

branch_layers = [branch_layers] if isinstance(branch_layers, int) else branch_layers

# add output features to trunk and branch layers
trunk_layers = trunk_layers + [hidden_dim]
branch_layers = branch_layers + [hidden_dim]

# Convert list to tuples
trunk_layers = tuple(trunk_layers)
branch_layers = tuple(branch_layers)

num_outputs = 1
model = DeepONet(branch_layers, trunk_layers, num_outputs)

params = model.init(key, jnp.ones(shape=(1, train_inputs.shape[1], train_inputs.shape[2], 1)),
                    jnp.ones(shape=(1, trunk_input_features)))

# Print model from parameters
print('--- model_summary ---')
# count total params
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f'total params: {total_params}')
print('--- model_summary ---')

# model function
model_fn = jax.jit(model.apply)


# loss = loss_fn(model_fn, params, data_batch)

# print("Initial loss:", loss)

# Define optimizer with optax (ADAM)

lr_scheduler = optax.exponential_decay(lr, transition_steps, decay_rate)

# lr_scheduler = optax.constant_schedule(1e-3)

optimizer = optax.adam(learning_rate=lr_scheduler)
opt_state = optimizer.init(params)

# Data
data_data = iter(data_dataset)

# create dir for saving results
result_dir = os.path.join(os.getcwd(), os.path.join(result_dir, f"results_run_{seed}"))
log_file = os.path.join(result_dir, 'log.csv')
# Create directory
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if os.path.exists(os.path.join(result_dir, 'vis')):
    shutil.rmtree(os.path.join(result_dir, 'vis'))
if os.path.exists(log_file):
    os.remove(log_file)


with open(log_file, 'a') as f:
    f.write('epoch,loss,runtime\n')

"""# model save function"""

save = True
# Saving
if save:
    save_model_params(params, result_dir)

# Iterations
epochs = epochs  # Assuming 'epochs' is defined elsewhere
log_iter = 1000
best_loss = float('inf')  # Initialize with infinity
# Save initial model at 0th iteration
save_model_params(params, result_dir, filename='model_params_best.pkl')
print("Saved initial model at iteration 0")
# Training loop
for it in range(epochs):
    if it == 1:
        # start timer and exclude first iteration (compile time)
        start = time.time()
    # Fetch data
    data_batch = next(data_data)

    # Do Step
    loss, params, opt_state = step(optimizer, loss_fn, model_fn, opt_state,
                                   params, data_batch)
    if it % log_iter == 0:
        # Compute losses
        loss = loss_fn(model_fn, params, data_batch)

        # inputs, outputs = data_batch
        # s_in, cord = inputs

        # x = cord[:, 0]
        # y = cord[:, 1]

        # s_pred = apply_net(model_fn, params, s_in, x, y)
        # outputs_flat = outputs.reshape(outputs.shape[0], -1) 
        # plot_training_samples(s_in, outputs, s_pred, outputs_flat, num_samples=5)

        if loss < best_loss:
            best_loss = loss
            # Save the model as it's the best so far
            save_model_params(params, result_dir, filename='model_params_best.pkl')
            print(f"New best model saved at iteration {it} with loss MSE: {loss:.7f}")

        # get runtime
        if it == 0:
            runtime = 0
        else:
            runtime = time.time() - start

        # Print losses
        print(f"Iteration {it+1}/{epochs}")
        print(f"Trian_loss: {loss:.2e}, runtime: {runtime:06.2f}")

        # Save results
        with open(log_file, 'a') as f:
            f.write(f'{it+1}, {loss}, {runtime}\n')

# Save results
runtime = time.time() - start
# Save initial model at 0th iteration
save_model_params(params, result_dir, filename='model_params_last.pkl')
print("Saved model at end of training")
with open(log_file, 'a') as f:
    f.write(f'{it + 1}, {loss}, {runtime}\n')


"""# Loss plots"""

# Set the result directory
# result_dir = "/kaggle/input/csv-log"

# Read the CSV file
csv_file = os.path.join(result_dir, "log.csv")  # Assuming the file is named "log.csv"
df = pd.read_csv(csv_file)

# Create the figure with two subplots side by side
fig, (ax1) = plt.subplots(1, 1, figsize=(18, 8), dpi=100)

# Set color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Right plot: epoch vs loss and l2err_val
ax1.plot(df['epoch'], df['loss'], label='Training Loss', color=colors[0], linestyle='-')
ax1.set_yscale("log")
ax1.set_xlabel('Epoch', fontsize=22)
ax1.set_ylabel('Train Loss', fontsize=22)
ax1.set_title('Training Loss and Testing Loss over Epochs', fontsize=22)
ax1.legend(loc='best', fontsize=22)
ax1.tick_params(axis='both', which='major', labelsize=22)  # Set axis ticks font size

# Adjust layout and save the figure
plt.tight_layout()
output_file = os.path.join(result_dir, "train_and_test_loss_plots.png")
plt.savefig(output_file, dpi=100)
plt.show()
plt.close()

print(f"Plots saved to {output_file}")


def save_autoregressive_results(predictions, initial_condition, mesh_points, 
                               num_steps, filename="prediction_DON.npz"):
    """Save autoregressive prediction results"""
    
    print(f"Saving autoregressive predictions to: {filename}")
    
    # Create time steps
    time_steps = jnp.arange(num_steps) * 0.01  # Assuming dt = 0.01, adjust as needed
    
    # Convert JAX arrays to NumPy for saving
    predictions_np = np.array(predictions)
    initial_condition_np = np.array(initial_condition)
    mesh_points_np = np.array(mesh_points)
    time_steps_np = np.array(time_steps)
    
    # Prepare data for saving
    save_data = {
        'solutions': predictions_np,                   # (289, 100) - Main prediction data
        'mesh_points': mesh_points_np,                # (289, 2) - Spatial coordinates
        'time_steps': time_steps_np,                  # (100,) - Time values
        'initial_condition': initial_condition_np,    # (17, 17) - Starting condition
        'num_steps': num_steps,                       # 100
        'num_points': predictions_np.shape[0],       # 289
        'prediction_shape': predictions_np.shape,     # (289, 100)
        'method': 'CNN_DeepONet_Autoregressive'
    }
    
    # Save with compression using NumPy
    np.savez_compressed(filename, **save_data)
    
    # Calculate file size
    import os
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    
    print(f"Autoregressive predictions saved successfully!")
    print(f"File: {filename}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Shape: {predictions_np.shape} (num_points, num_steps)")

def autoregressive_prediction(model_fn, params, initial_condition, mesh_points, num_steps=100, 
                            target_size=17, save_filename="prediction_DON.npz"):
    """
    Perform autoregressive prediction using trained CNN-DeepONet
    
    Args:
        model_fn: Jitted model function
        params: Trained model parameters
        initial_condition: Initial spatial field, shape (17, 17, 1) or (17, 17)
        mesh_points: Coordinate points for prediction
        num_steps: Number of time steps to predict (default: 100)
        target_size: Spatial grid size (17)
        save_filename: Output filename
    
    Returns:
        predictions: Array of predictions, shape (289, 100)
    """
    
    print(f"="*60)
    print(f"AUTOREGRESSIVE PREDICTION")
    print(f"="*60)
    print(f"Number of prediction steps: {num_steps}")
    print(f"Initial condition shape: {initial_condition.shape}")
    print(f"Target spatial size: {target_size}×{target_size}")
    print(f"Output shape will be: (289, {num_steps})")
    
    # Ensure initial condition has correct shape
    if initial_condition.ndim == 2:  # (17, 17)
        current_state = initial_condition[..., None]  # Add channel: (17, 17, 1)
    else:  # Already (17, 17, 1)
        current_state = initial_condition
    
    # Add batch dimension for model input: (1, 17, 17, 1)
    current_state_batch = current_state[None, ...]
    
    # Storage for predictions - shape (289, num_steps)
    predictions = jnp.zeros((289, num_steps))
    
    # Prepare coordinates for trunk network

    xi = jnp.linspace(jnp.min(mesh_points[0]), jnp.max(mesh_points[0]), 17)
    yi = jnp.linspace(jnp.min(mesh_points[1]), jnp.max(mesh_points[1]), 17)
    Xi, Yi = jnp.meshgrid(xi, yi, indexing='ij')
    cord_mesh = jnp.column_stack([Xi.flatten(), Yi.flatten()])
    x_coords = cord_mesh[:, 0]
    y_coords = cord_mesh[:, 1]
    # x_coords = mesh_points[:, 0]
    # y_coords = mesh_points[:, 1]
    
    print(f"Starting autoregressive prediction loop...")
    
    for step in range(num_steps):
        print(f"Predicting step {step+1}/{num_steps}", end='\r')
        
        # Predict next state using apply_net function
        pred_flat = apply_net(model_fn, params, current_state_batch, x_coords, y_coords)
        # pred_flat shape: (1, 289)
        
        # Store flattened prediction (remove batch dimension)
        predictions = predictions.at[:, step].set(pred_flat[0])  # Store as column
        
        # Reshape prediction back to spatial grid for next input
        pred_spatial = pred_flat.reshape(1, target_size, target_size)  # (1, 17, 17)
        
        # Prepare next input (add channel dimension)
        current_state = pred_spatial[0][..., None]  # (17, 17, 1)
        current_state_batch = current_state[None, ...]  # (1, 17, 17, 1)
    
    print(f"\nAutoregressive prediction complete!")
    print(f"Predictions shape: {predictions.shape}")
    
    # Save results
    save_autoregressive_results(predictions, initial_condition, mesh_points, 
                              num_steps, save_filename)
    
    return predictions

# ============================================================================
# MAIN AUTOREGRESSIVE PREDICTION EXECUTION
# ============================================================================

# Load the best trained model
print("Loading best trained model...")
best_params = load_model_params(result_dir, filename='model_params_best.pkl')

# Choose initial condition - using first training sample
initial_condition = train_inputs[0, :, :, 0]  # Remove channel dimension: (17, 17)

print(f"Using initial condition with shape: {initial_condition.shape}")

# Set prediction parameters
num_prediction_steps = 100  # Predict 100 time steps
output_filename = "prediction_DON.npz"

# Perform autoregressive prediction
predictions = autoregressive_prediction(
    model_fn=model_fn,
    params=best_params,
    initial_condition=initial_condition,
    mesh_points=mesh_points,
    num_steps=num_prediction_steps,
    save_filename=output_filename
)

print(f"\n" + "="*60)
print(f"AUTOREGRESSIVE PREDICTION COMPLETE")
print(f"="*60)
print(f"Results saved to: {output_filename}")
print(f"Predictions shape: {predictions.shape}")  # Should be (289, 100)
print(f"Format: (num_points={predictions.shape[0]}, num_steps={predictions.shape[1]})")
print("="*60)

# Verify the saved file
print(f"\nVerifying saved predictions...")
loaded_results = np.load(output_filename, allow_pickle=True)
print(f"Loaded 'solutions' shape: {loaded_results['solutions'].shape}")  # Should be (289, 100)
print(f"Loaded 'mesh_points' shape: {loaded_results['mesh_points'].shape}")  # Should be (289, 2)
print(f"Loaded 'time_steps' shape: {loaded_results['time_steps'].shape}")  # Should be (100,)