import diffusion_env
import numpy as np
import matplotlib.pyplot as plt

# Create environment with moderate resolution for demonstration
env = diffusion_env.DiffusionEnvironment(refinement_level=4)

# Set initial condition
def gaussian_peak(x, y):
    return np.exp(-20 * ((x-0.3)**2 + (y-0.7)**2))

env.reset(gaussian_peak)

# Extract solution and geometry
solution = np.array(env.get_solution_data())
points = np.array(env.get_mesh_points())

print(f"Full solution vector shape: {solution.shape}")
print(f"Mesh points array shape: {points.shape}")
print(f"Solution value range: [{solution.min():.6f}, {solution.max():.6f}]")

# The points array has shape (N, 2) where N is number of degrees of freedom
x_coords = points[:, 0]  # All x-coordinates
y_coords = points[:, 1]  # All y-coordinates

print(f"Domain bounds: x=[{x_coords.min():.3f}, {x_coords.max():.3f}]")
print(f"               y=[{y_coords.min():.3f}, {y_coords.max():.3f}]")

# Create a spatial visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(x_coords, y_coords, c=solution, cmap='hot', s=20)
plt.colorbar(label='Solution Value')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Initial Condition: Spatial Distribution')
plt.axis('equal')

# Take a few time steps and show evolution
for step in range(20):
    env.step()

solution_evolved = np.array(env.get_solution_data())

plt.subplot(1, 2, 2)
plt.scatter(x_coords, y_coords, c=solution_evolved, cmap='hot', s=20)
plt.colorbar(label='Solution Value')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'After {env.get_timestep()} Time Steps')
plt.axis('equal')

plt.tight_layout()
plt.savefig('spatial_solution_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Spatial visualization saved as 'spatial_solution_evolution.png'")