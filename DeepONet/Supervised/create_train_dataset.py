import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

def process_all_timesteps_efficient(solution_data, mesh_points, target_size=17, verbose=True):
    """Process all timesteps efficiently by calculating mesh structure once"""
    
    num_grid_points, num_timesteps = solution_data.shape
    x_coords, y_coords = mesh_points[:, 0], mesh_points[:, 1]
    
    # Determine mesh structure ONCE
    unique_x = jnp.unique(x_coords)
    unique_y = jnp.unique(y_coords)
    
    if verbose:
        print(f"Processing all {num_timesteps} timesteps efficiently...")
        print(f"Using exact mapping to {len(unique_y)}×{len(unique_x)} grid for ALL timesteps")
    
    if len(unique_x) * len(unique_y) == len(x_coords):
        # Regular grid - create mapping indices once
        x_indices = jnp.array([jnp.where(unique_x == x)[0][0] for x in x_coords])
        y_indices = jnp.array([jnp.where(unique_y == y)[0][0] for y in y_coords])
        
        # Check if interpolation is needed (do this once)
        needs_interpolation = (len(unique_y), len(unique_x)) != (target_size, target_size)
        
        if needs_interpolation and verbose:
            print(f"Will interpolate from {(len(unique_y), len(unique_x))} to ({target_size}, {target_size})")
            # Prepare interpolation coordinates once
            xi = jnp.linspace(jnp.min(unique_x), jnp.max(unique_x), target_size)
            yi = jnp.linspace(jnp.min(unique_y), jnp.max(unique_y), target_size)
            Xi, Yi = jnp.meshgrid(xi, yi)
            X_orig, Y_orig = jnp.meshgrid(unique_x, unique_y)
            points_orig = jnp.column_stack([X_orig.flatten(), Y_orig.flatten()])
            points_target = jnp.column_stack([Xi.flatten(), Yi.flatten()])
        
        # Process all timesteps
        processed_grids = []
        
        for t in range(num_timesteps):
            if verbose and t % 20 == 0:
                print(f"Processing timestep {t}/{num_timesteps}", end='\r')
            
            # Create 2D grid for this timestep
            grid_2d = jnp.zeros((len(unique_y), len(unique_x)))
            solution_t = solution_data[:, t]
            
            # Vectorized assignment
            grid_2d = grid_2d.at[y_indices, x_indices].set(solution_t)
            
            # Flip y-axis
            grid_2d = jnp.flipud(grid_2d)
            
            # Interpolate if needed
            if needs_interpolation:
                grid_interp = griddata(
                    points_orig, jnp.flipud(grid_2d).flatten(),
                    points_target, method='cubic', fill_value=0
                ).reshape(target_size, target_size)
                spatial_2d = jnp.flipud(grid_interp)
            else:
                spatial_2d = grid_2d
            
            processed_grids.append(spatial_2d)
    
    else:
        # Irregular mesh
        if verbose:
            print(f"Interpolating irregular mesh to {target_size}×{target_size}")
        
        # Create interpolation grid once
        xi = jnp.linspace(jnp.min(x_coords), jnp.max(x_coords), target_size)
        yi = jnp.linspace(jnp.min(y_coords), jnp.max(y_coords), target_size)
        Xi, Yi = jnp.meshgrid(xi, yi)
        
        processed_grids = []
        for t in range(num_timesteps):
            if verbose and t % 20 == 0:
                print(f"Processing timestep {t}/{num_timesteps}", end='\r')
            
            solution_t = solution_data[:, t]
            spatial_2d = griddata(
                (x_coords, y_coords), solution_t,
                (Xi, Yi), method='cubic', fill_value=0
            )
            spatial_2d = jnp.flipud(spatial_2d)
            processed_grids.append(spatial_2d)
    
    # Convert to JAX array
    processed_grids = jnp.array(processed_grids)
    
    if verbose:
        print(f"\nAll timesteps processed! Shape: {processed_grids.shape}")
    
    return processed_grids

def create_cnn_dataset_for_training(solution_data, mesh_points, 
                                  start_timestep=0, end_timestep=None, 
                                  target_size=17, verbose=True):
    """
    Create CNN dataset for autoregressive training - EFFICIENT VERSION
    
    Args:
        solution_data: shape (num_grid_points, num_timesteps)
        mesh_points: mesh coordinates for proper spatial arrangement
        start_timestep: starting time step index
        end_timestep: ending time step index (None = use all available)
        target_size: spatial grid size for CNN (17x17)
        verbose: print progress information
    
    Returns:
        inputs: shape (num_samples, 17, 17, 1) - time steps t=start to end-1
        outputs: shape (num_samples, 17, 17, 1) - time steps t=start+1 to end
        time_indices: corresponding time step indices
    """
    
    num_grid_points, total_timesteps = solution_data.shape
    
    # Handle end_timestep
    if end_timestep is None:
        end_timestep = total_timesteps - 1
    
    # Validate inputs
    assert start_timestep >= 0, "start_timestep must be >= 0"
    assert end_timestep < total_timesteps, f"end_timestep must be < {total_timesteps}"
    assert start_timestep < end_timestep, "start_timestep must be < end_timestep"
    
    num_samples = end_timestep - start_timestep
    
    if verbose:
        print(f"="*60)
        print(f"CREATING CNN DATASET - EFFICIENT VERSION")
        print(f"="*60)
        print(f"Total time steps available: {total_timesteps}")
        print(f"Using time steps: {start_timestep} to {end_timestep}")
        print(f"Number of training samples: {num_samples}")
        print(f"Target spatial size: {target_size}×{target_size}")
        print(f"-"*60)
    
    # Step 1: Process ALL timesteps efficiently (only once!)
    all_processed_grids = process_all_timesteps_efficient(
        solution_data, mesh_points, target_size=target_size, verbose=verbose
    )
    
    # Step 2: Create input-output pairs from processed grids
    if verbose:
        print(f"Creating input-output pairs from processed grids...")
    
    # Extract the relevant time range and create pairs
    input_grids = all_processed_grids[start_timestep:end_timestep]      # Shape: (num_samples, 17, 17)
    output_grids = all_processed_grids[start_timestep+1:end_timestep+1] # Shape: (num_samples, 17, 17)
    
    # Add channel dimension
    inputs = input_grids[..., None]   # Shape: (num_samples, 17, 17, 1)
    outputs = output_grids[..., None] # Shape: (num_samples, 17, 17, 1)
    
    # Create time indices
    time_indices = jnp.array([(t, t+1) for t in range(start_timestep, end_timestep)])
    
    if verbose:
        print(f"Dataset creation complete!")
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Time indices shape: {time_indices.shape}")
        print(f"="*60)
    
    return inputs, outputs, time_indices

def create_correct_cnn_input(solution_data, mesh_points, timestep_idx=0, target_size=17):
    """Create properly structured CNN input using actual mesh coordinates"""
    
    x_coords, y_coords = mesh_points[:, 0], mesh_points[:, 1]
    solution_at_time = solution_data[:, timestep_idx]
    
    unique_x = jnp.unique(x_coords)
    unique_y = jnp.unique(y_coords)
    
    if len(unique_x) * len(unique_y) == len(x_coords):
        # Regular grid - use exact mapping
        print(f"Using exact mapping to {len(unique_y)}×{len(unique_x)} grid")
        
        grid_2d = jnp.zeros((len(unique_y), len(unique_x)))
        
        for i, (x, y, val) in enumerate(zip(x_coords, y_coords, solution_at_time)):
            x_idx = jnp.where(unique_x == x)[0][0]
            y_idx = jnp.where(unique_y == y)[0][0]
            grid_2d = grid_2d.at[y_idx, x_idx].set(val)
        
        # Flip y-axis to match standard image coordinates (0 at top)
        grid_2d = jnp.flipud(grid_2d)
        
        # If the grid is not the target size, interpolate
        if grid_2d.shape != (target_size, target_size):
            print(f"Interpolating from {grid_2d.shape} to ({target_size}, {target_size})")
            
            # Create target coordinates
            xi = jnp.linspace(jnp.min(unique_x), jnp.max(unique_x), target_size)
            yi = jnp.linspace(jnp.min(unique_y), jnp.max(unique_y), target_size)
            Xi, Yi = jnp.meshgrid(xi, yi)
            
            # Interpolate from the correct grid
            X_orig, Y_orig = jnp.meshgrid(unique_x, unique_y)
            points_orig = jnp.column_stack([X_orig.flatten(), Y_orig.flatten()])
            points_target = jnp.column_stack([Xi.flatten(), Yi.flatten()])
            
            grid_interp = griddata(
                points_orig, jnp.flipud(grid_2d).flatten(),  # Flip back for interpolation
                points_target, method='cubic', fill_value=0
            ).reshape(target_size, target_size)
            
            # Flip again after interpolation
            spatial_2d = jnp.flipud(grid_interp)
        else:
            spatial_2d = grid_2d
            
    else:
        # Irregular mesh - interpolate directly
        print(f"Interpolating irregular mesh to {target_size}×{target_size}")
        
        xi = jnp.linspace(jnp.min(x_coords), jnp.max(x_coords), target_size)
        yi = jnp.linspace(jnp.min(y_coords), jnp.max(y_coords), target_size)
        Xi, Yi = jnp.meshgrid(xi, yi)
        
        spatial_2d = griddata(
            (x_coords, y_coords), solution_at_time,
            (Xi, Yi), method='cubic', fill_value=0
        )
        
        # Flip y-axis for consistent orientation
        spatial_2d = jnp.flipud(spatial_2d)
    
    # Add channel and batch dimensions for CNN
    cnn_input = spatial_2d[..., None][None, ...]  # (1, target_size, target_size, 1)
    
    return jnp.asarray(cnn_input), spatial_2d

def visualize_dataset_samples(inputs, outputs, time_indices, num_samples=3, save_name="dataset_samples"):
    """Visualize some samples from the dataset"""
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Input (current time step)
        im1 = axes[0, i].imshow(inputs[i, :, :, 0], cmap='viridis', origin='upper')
        axes[0, i].set_title(f"Input: t={time_indices[i, 0]}")
        plt.colorbar(im1, ax=axes[0, i])
        
        # Output (next time step)
        im2 = axes[1, i].imshow(outputs[i, :, :, 0], cmap='viridis', origin='upper')
        axes[1, i].set_title(f"Output: t={time_indices[i, 1]}")
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Dataset samples saved to {save_name}.png")

def save_cnn_dataset(inputs, outputs, time_indices, mesh_points, time_steps, 
                     filename="cnn_training_dataset.npz", compress=True):
    """
    Save the CNN dataset to npz file
    
    Args:
        inputs: shape (num_samples, 17, 17, 1)
        outputs: shape (num_samples, 17, 17, 1) 
        time_indices: shape (num_samples, 2)
        mesh_points: original mesh coordinates
        time_steps: original time step values
        filename: output filename
        compress: whether to use compression
    """
    
    print(f"="*60)
    print(f"SAVING CNN DATASET")
    print(f"="*60)
    print(f"Filename: {filename}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Time indices shape: {time_indices.shape}")
    print(f"Compression: {'Yes' if compress else 'No'}")
    
    # Prepare data dictionary
    data_dict = {
        'inputs': inputs,
        'outputs': outputs,
        'time_indices': time_indices,
        'mesh_points': mesh_points,
        'time_steps': time_steps,
        'input_shape': inputs.shape,
        'output_shape': outputs.shape,
        'target_size': inputs.shape[1],  # Should be 17
        'num_samples': inputs.shape[0],
        'start_timestep': time_indices[0, 0],
        'end_timestep': time_indices[-1, 1],
    }
    
    # Save with or without compression
    if compress:
        np.savez_compressed(filename, **data_dict)
    else:
        np.savez(filename, **data_dict)
    
    # Calculate file size
    import os
    file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
    
    print(f"Dataset saved successfully!")
    print(f"File size: {file_size:.2f} MB")
    print(f"="*60)


# Load the data
print(f"="*60)
print(f"LOADING DIFFUSION SIMULATION DATA")
print(f"="*60)

data_file = '/home/droysar1/scr4_sgoswam4/Dibakar/Diffusion_project/diffusion_simulation_data.npz'
loaded = np.load(data_file, allow_pickle=True)

solutions = loaded['solutions']  # Shape: (num_grid_points, num_timesteps)
time_steps = loaded['time_steps']
params = loaded['simulation_params'].item()

print(f"Solutions shape: {solutions.shape}")
print(f"Time steps: {len(time_steps)} from {time_steps[0]:.3f} to {time_steps[-1]:.3f}")
print(f"Simulation parameters: {params}")
print(f"-"*60)

mesh_points = loaded['mesh_points']
x_coords, y_coords = mesh_points[:, 0], mesh_points[:, 1]
print(f"Using provided mesh points: {mesh_points.shape}")

# Create the dataset efficiently
inputs, outputs, time_indices = create_cnn_dataset_for_training(
    solutions, mesh_points, 
    start_timestep=0, 
    # end_timestep=8,  # Use all available
    end_timestep=None,  # Use all available
    target_size=17,
    verbose=True
)

# Visualize some samples
visualize_dataset_samples(
    inputs[:3], outputs[:3], time_indices[:3], 
    num_samples=3, save_name="training_dataset_samples"
)

# Save the CNN dataset
save_cnn_dataset(
    inputs, outputs, time_indices, mesh_points, time_steps,
    filename="cnn_training_dataset.npz",
    compress=True  # Use compression to save disk space
)

print(f"\nDataset ready for CNN training!")
print(f"Final shapes:")
print(f"  Inputs: {inputs.shape}")
print(f"  Outputs: {outputs.shape}")