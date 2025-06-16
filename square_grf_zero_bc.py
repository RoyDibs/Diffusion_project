import numpy as np
import GPy
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def create_square_mask(nx, ny):
    """
    Create a boolean mask for square geometry.
    
    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    
    Returns:
    --------
    mask : ndarray
        Boolean array where True indicates points inside square
    """
    return np.ones((ny, nx), dtype=bool)

def identify_square_boundary_points(mask):
    """
    Identify boundary points for a square domain.
    
    Parameters:
    -----------
    mask : ndarray
        Boolean mask where True indicates points inside the domain
    
    Returns:
    --------
    boundary_indices : list
        Linear indices of boundary points in the masked coordinate array
    """
    ny, nx = mask.shape
    boundary_mask = np.zeros_like(mask, dtype=bool)
    
    # For a square domain, boundary points are on the edges
    boundary_mask[0, :] = True      # Bottom edge
    boundary_mask[-1, :] = True     # Top edge
    boundary_mask[:, 0] = True      # Left edge
    boundary_mask[:, -1] = True     # Right edge
    
    # Convert to indices in the masked coordinate array
    mask_indices = np.where(mask.ravel())[0]
    boundary_mask_flat = boundary_mask.ravel()
    boundary_indices = []
    
    for idx, mask_idx in enumerate(mask_indices):
        if boundary_mask_flat[mask_idx]:
            boundary_indices.append(idx)
    
    return boundary_indices

def create_square_grf_initial_condition(env,
                                      nx=32, ny=32,
                                      lengthscale_x=0.1, lengthscale_y=0.1,
                                      variance=1.0, kernel_type='rbf',
                                      seed=None):
    """
    Create a zero-boundary GRF initial condition for square domain compatible with DiffusionEnv.
    
    Parameters:
    -----------
    env : DiffusionEnvironment
        The diffusion environment instance (should be square)
    nx, ny : int
        Grid resolution for GRF generation
    lengthscale_x, lengthscale_y : float
        Correlation lengthscales in x and y directions (relative to domain size)
    variance : float
        Signal variance of the random field
    kernel_type : str
        Type of covariance kernel ('rbf', 'exp', 'mat32', 'mat52')
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    initial_condition : callable
        Function f(x, y) -> float that can be passed to env.reset()
    grf_data : dict
        Dictionary containing the generated GRF data for analysis
    """
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get domain bounds from the environment
    bounds = env.get_domain_bounds()  # [x_min, x_max, y_min, y_max]
    x_min, x_max, y_min, y_max = bounds
    
    # Create regular grid covering the square domain
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create square mask (all points are inside)
    mask = create_square_mask(nx, ny)
    
    # Extract coordinates (all points are inside the square)
    X_masked = X[mask]
    Y_masked = Y[mask]
    coordinates = np.column_stack([X_masked, Y_masked])
    
    # Identify boundary points in the square
    boundary_indices = identify_square_boundary_points(mask)
    interior_indices = [i for i in range(len(coordinates)) if i not in boundary_indices]
    
    # Available kernels
    kernels = {
        'rbf': GPy.kern.RBF, 
        'exp': GPy.kern.Exponential, 
        'mat32': GPy.kern.Matern32, 
        'mat52': GPy.kern.Matern52
    }
    
    assert kernel_type in kernels.keys(), f"Kernel must be one of {list(kernels.keys())}"
    
    # Scale lengthscales relative to domain size
    domain_width = x_max - x_min
    domain_height = y_max - y_min
    actual_lengthscale_x = lengthscale_x * domain_width
    actual_lengthscale_y = lengthscale_y * domain_height
    
    # Set up GPy kernel
    kernel = kernels[kernel_type](
        input_dim=2,
        lengthscale=[actual_lengthscale_x, actual_lengthscale_y],
        variance=variance,
        ARD=True
    )
    
    # Compute covariance matrix for square domain points
    nugget = 1e-6
    Cov_full = kernel.K(coordinates) + nugget * np.eye(coordinates.shape[0])
    
    # Generate zero-boundary GRF using conditional sampling
    if len(boundary_indices) > 0 and len(interior_indices) > 0:
        # Extract submatrices for conditional sampling
        K_ii = Cov_full[np.ix_(interior_indices, interior_indices)]
        K_ib = Cov_full[np.ix_(interior_indices, boundary_indices)]
        K_bb = Cov_full[np.ix_(boundary_indices, boundary_indices)]
        
        try:
            # Conditional covariance for interior points given zero boundary
            K_bb_inv = np.linalg.inv(K_bb + nugget * np.eye(K_bb.shape[0]))
            K_conditional = K_ii - K_ib @ K_bb_inv @ K_ib.T
            
            # Ensure numerical stability
            K_conditional = K_conditional + nugget * np.eye(K_conditional.shape[0])
            
            # Check if matrix is positive definite
            eigenvals = np.linalg.eigvals(K_conditional)
            min_eigenval = np.min(eigenvals)
            
            if min_eigenval < -1e-10:
                # Add more regularization
                K_conditional = K_conditional + (abs(min_eigenval) + nugget) * np.eye(K_conditional.shape[0])
            
            # Sample from conditional distribution
            L_conditional = np.linalg.cholesky(K_conditional)
            z = np.random.randn(len(interior_indices))
            field_interior = L_conditional @ z
            
        except np.linalg.LinAlgError as e:
            # Fallback using eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(K_conditional)
            eigenvals = np.maximum(eigenvals, nugget)
            L_conditional = eigenvecs @ np.diag(np.sqrt(eigenvals))
            z = np.random.randn(len(interior_indices))
            field_interior = L_conditional @ z
        
        # Reconstruct full field
        field_values = np.zeros(len(coordinates))
        field_values[interior_indices] = field_interior
        # Boundary values remain zero
        
    elif len(interior_indices) > 0:
        # Sample on interior points only if no boundary points
        K_ii = Cov_full[np.ix_(interior_indices, interior_indices)]
        K_ii_reg = K_ii + nugget * np.eye(K_ii.shape[0])
        L_ii = np.linalg.cholesky(K_ii_reg)
        z = np.random.randn(len(interior_indices))
        field_interior = L_ii @ z
        
        field_values = np.zeros(len(coordinates))
        field_values[interior_indices] = field_interior
        
    else:
        field_values = np.zeros(len(coordinates))
    
    # Create full 2D field for visualization
    field_full = field_values.reshape(ny, nx)
    
    # Create interpolator for the square domain
    interpolator = RegularGridInterpolator((y, x), field_full, 
                                         bounds_error=False, fill_value=0.0)
    
    def initial_condition(x_eval, y_eval):
        """
        Initial condition function compatible with DiffusionEnv.
        Returns 0 outside the square domain.
        
        Parameters:
        -----------
        x_eval, y_eval : float or array
            Coordinates where to evaluate the field
            
        Returns:
        --------
        float or array
            Field value at (x_eval, y_eval)
        """
        # Handle both scalar and array inputs
        x_eval = np.atleast_1d(x_eval)
        y_eval = np.atleast_1d(y_eval)
        
        # Create points array for interpolation (note: interpolator expects (y, x) order)
        points = np.column_stack([y_eval.ravel(), x_eval.ravel()])
        
        # Interpolate
        result = interpolator(points)
        
        # Set values outside domain bounds to zero
        outside_mask = ((x_eval.ravel() < x_min) | (x_eval.ravel() > x_max) | 
                       (y_eval.ravel() < y_min) | (y_eval.ravel() > y_max))
        result[outside_mask] = 0.0
        
        # Return scalar if input was scalar
        if result.size == 1:
            return float(result[0])
        else:
            return result.reshape(x_eval.shape)
    
    # Compute statistics
    valid_values = field_values[np.abs(field_values) > 1e-12]  # Exclude zeros
    if len(valid_values) > 0:
        field_stats = {
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'min': np.min(field_values),
            'max': np.max(field_values),
            'n_interior_points': len(interior_indices),
            'n_boundary_points': len(boundary_indices),
            'n_total_points': len(coordinates)
        }
    else:
        field_stats = {
            'mean': 0, 'std': 0, 'min': 0, 'max': 0,
            'n_interior_points': len(interior_indices),
            'n_boundary_points': len(boundary_indices),
            'n_total_points': len(coordinates)
        }
    
    # Package data for analysis
    grf_data = {
        'field_values': field_values,
        'field_full': field_full,
        'coordinates': coordinates,
        'mask': mask,
        'x_grid': x,
        'y_grid': y,
        'boundary_indices': boundary_indices,
        'interior_indices': interior_indices,
        'statistics': field_stats,
        'domain_bounds': bounds,
        'kernel_info': {
            'type': kernel_type,
            'lengthscales': [actual_lengthscale_x, actual_lengthscale_y],
            'variance': variance
        }
    }
    
    return initial_condition, grf_data

def visualize_square_grf(grf_data, title="Square GRF Initial Condition"):
    """
    Visualize the generated square GRF data.
    
    Parameters:
    -----------
    grf_data : dict
        Data dictionary returned by create_square_grf_initial_condition
    title : str
        Plot title
    """
    
    field_full = grf_data['field_full']
    x_grid = grf_data['x_grid']
    y_grid = grf_data['y_grid']
    coordinates = grf_data['coordinates']
    boundary_indices = grf_data['boundary_indices']
    bounds = grf_data['domain_bounds']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create coordinate arrays for contourf
    X_full, Y_full = np.meshgrid(x_grid, y_grid)
    
    # Plot 1: Square domain with boundary points
    axes[0].contourf(X_full, Y_full, np.ones_like(field_full), 
                    levels=[0.5, 1.5], colors=['lightblue'], alpha=0.8)
    # Add boundary
    axes[0].contour(X_full, Y_full, np.ones_like(field_full), 
                   levels=[0.5], colors=['black'], linewidths=2)
    
    # Highlight boundary points
    if len(boundary_indices) > 0:
        boundary_coords = coordinates[boundary_indices]
        axes[0].scatter(boundary_coords[:, 0], boundary_coords[:, 1], 
                       c='red', s=20, alpha=0.8, label='Boundary (zero)')
        axes[0].legend()
    
    axes[0].set_title('Square Domain with Boundary Points')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_xlim(bounds[0], bounds[1])
    axes[0].set_ylim(bounds[2], bounds[3])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Plot 2: Random field on square with zero boundary
    levels = np.linspace(np.min(field_full), np.max(field_full), 20)
    cs1 = axes[1].contourf(X_full, Y_full, field_full, 
                          levels=levels, cmap='viridis')
    plt.colorbar(cs1, ax=axes[1], label='Field value')
    # Add domain boundary
    axes[1].contour(X_full, Y_full, np.ones_like(field_full), 
                   levels=[0.5], colors=['white'], linewidths=1, alpha=0.8)
    axes[1].set_title(f'{title}\n(Zero boundary conditions)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_xlim(bounds[0], bounds[1])
    axes[1].set_ylim(bounds[2], bounds[3])
    axes[1].set_aspect('equal')
    
    # Plot 3: Scatter plot of field values
    field_values = grf_data['field_values']
    
    scatter = axes[2].scatter(coordinates[:, 0], coordinates[:, 1], 
                             c=field_values, cmap='viridis', s=15, alpha=0.8)
    plt.colorbar(scatter, ax=axes[2], label='Field value')
    axes[2].set_title('Field Values at Square Grid Points')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_xlim(bounds[0], bounds[1])
    axes[2].set_ylim(bounds[2], bounds[3])
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()