import diffusion_env
import numpy as np

print('Creating finite element environment...')
env = diffusion_env.DiffusionEnvironment(refinement_level=3)
print(f'Environment created with {env.get_num_dofs()} degrees of freedom')

print('Testing mesh point extraction...')
mesh_points = env.get_mesh_points()
print(f'Extracted {len(mesh_points)} mesh points')
print(f'Mesh domain spans: x=[{min(p[0] for p in mesh_points):.3f}, {max(p[0] for p in mesh_points):.3f}]')
print(f'                   y=[{min(p[1] for p in mesh_points):.3f}, {max(p[1] for p in mesh_points):.3f}]')

print('Basic functionality verification completed successfully!')