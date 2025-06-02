import diffusion_env
import numpy as np

print('Testing complete simulation workflow...')
env = diffusion_env.DiffusionEnvironment(refinement_level=3, final_time=0.1)

# Define a simple initial condition for testing
def test_initial_condition(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

print('Setting initial condition...')
env.reset(test_initial_condition)

# Get initial state
initial_quantities = env.get_physical_quantities()
print('Initial physical quantities: ', str(initial_quantities))
print(f'Initial state: energy={initial_quantities[0]:.6f}, max_val={initial_quantities[1]:.6f}')

# Run a few time steps
step_count = 0
while not env.is_done() and step_count < 5:
    env.step()
    step_count += 1
    current_quantities = env.get_physical_quantities()
    print(f'Step {step_count}: time={env.get_time():.3f}, energy={current_quantities[0]:.6f}, max_val={current_quantities[1]:.6f}')

print(f'Simulation dynamics test completed after {step_count} steps!')