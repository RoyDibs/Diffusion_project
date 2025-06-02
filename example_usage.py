#!/usr/bin/env python3
"""
Complete example demonstrating how to use the diffusion_env module
for reinforcement learning applications.

This script shows several usage patterns:
1. Basic simulation with visualization
2. Multiple episodes with different initial conditions  
3. Parameter studies and physical analysis
4. Integration patterns for RL frameworks
5. Performance optimization techniques

Run this after building and installing the diffusion_env module.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, List, Tuple

# Import our wrapped finite element environment
try:
    import diffusion_env
    print("Successfully imported diffusion_env module!")
    print(f"Module version: {diffusion_env.__version__}")
except ImportError as e:
    print("Error importing diffusion_env module.")
    print("Make sure you've built and installed the module first:")
    print("  python setup.py build_ext --inplace")
    print(f"Error details: {e}")
    exit(1)

def example_1_basic_simulation():
    """
    Basic example: Run a single simulation and visualize the results.
    
    This demonstrates the fundamental workflow: create environment,
    set initial condition, step through time, extract data.
    """
    print("\n" + "="*50)
    print("Example 1: Basic Simulation")
    print("="*50)
    
    # Create the environment with a moderately fine mesh
    # Refinement level 4 gives 289 degrees of freedom - good for learning
    env = diffusion_env.DiffusionEnvironment(
        refinement_level=4,
        diffusion_coeff=0.1,
        dt=0.01,
        final_time=0.5
    )
    
    print(f"Created environment with {env.get_num_dofs()} degrees of freedom")
    
    # Define a smooth initial condition - sine wave pattern
    def sine_initial_condition(x: float, y: float) -> float:
        """Classic smooth initial condition for diffusion problems."""
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Reset environment with this initial condition
    env.reset(sine_initial_condition)
    
    # Collect data throughout the simulation
    times = []
    total_energies = []
    max_values = []
    
    # Run the simulation
    print("Running simulation...")
    while not env.is_done():
        # Store current state information
        times.append(env.get_time())
        physical_quantities = env.get_physical_quantities()
        total_energies.append(physical_quantities[0])
        max_values.append(physical_quantities[1])
        
        # Advance one time step
        env.step()
    
    # Final state
    times.append(env.get_time())
    physical_quantities = env.get_physical_quantities()
    total_energies.append(physical_quantities[0])
    max_values.append(physical_quantities[1])
    
    print(f"Simulation completed: {len(times)} time steps")
    print(f"Energy decreased from {total_energies[0]:.6f} to {total_energies[-1]:.6f}")
    print(f"Maximum value decreased from {max_values[0]:.6f} to {max_values[-1]:.6f}")
    
    # Visualize the temporal evolution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(times, total_energies, 'b-', linewidth=2, label='Total Energy')
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Energy Dissipation Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(times, max_values, 'r-', linewidth=2, label='Maximum Value')
    plt.xlabel('Time')
    plt.ylabel('Maximum Temperature')
    plt.title('Peak Temperature Decay')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('basic_simulation_results.png', dpi=150, bbox_inches='tight')
    print("Saved temporal evolution plot to 'basic_simulation_results.png'")
    
    # Visualize the final spatial distribution
    solution = env.get_solution_data()
    mesh_points = env.get_mesh_points()
    
    # Create a spatial plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mesh_points[:, 0], mesh_points[:, 1], 
                         c=solution, cmap='hot', s=20)
    plt.colorbar(scatter, label='Temperature')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title(f'Final Temperature Distribution (t = {env.get_time():.2f})')
    plt.axis('equal')
    plt.savefig('final_spatial_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved spatial distribution plot to 'final_spatial_distribution.png'")
    
    # Optional: write VTU file for advanced visualization
    env.write_vtk("final_solution.vtu")
    print("Saved VTU file 'final_solution.vtu' for VisIt/ParaView visualization")

def example_2_multiple_episodes():
    """
    Demonstrate running multiple episodes with different initial conditions.
    
    This pattern is essential for RL training where you need many different
    simulation episodes to train your agent effectively.
    """
    print("\n" + "="*50)
    print("Example 2: Multiple Episodes")
    print("="*50)
    
    # Create environment once - reuse for all episodes
    env = diffusion_env.DiffusionEnvironment(refinement_level=4)
    
    # Define several different initial conditions
    initial_conditions = {
        'sine_wave': lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
        'gaussian_center': lambda x, y: np.exp(-20 * ((x-0.5)**2 + (y-0.5)**2)),
        'corner_heat': lambda x, y: np.exp(-10 * (x**2 + y**2)),
        'dual_peaks': lambda x, y: np.exp(-20 * ((x-0.3)**2 + (y-0.3)**2)) + 
                                   np.exp(-20 * ((x-0.7)**2 + (y-0.7)**2)),
        'linear_gradient': lambda x, y: x * y,
    }
    
    episode_results = {}
    
    print(f"Running {len(initial_conditions)} episodes...")
    
    for episode_name, initial_condition in initial_conditions.items():
        print(f"  Episode: {episode_name}")
        
        # Reset environment for new episode
        env.reset(initial_condition)
        
        # Run this episode to completion
        step_count = 0
        while not env.is_done():
            env.step()
            step_count += 1
        
        # Extract final results
        final_solution = env.get_solution_data()
        final_quantities = env.get_physical_quantities()
        
        episode_results[episode_name] = {
            'steps': step_count,
            'final_energy': final_quantities[0],
            'final_max': final_quantities[1],
            'final_solution': final_solution
        }
        
        print(f"    Completed in {step_count} steps")
        print(f"    Final energy: {final_quantities[0]:.6f}")
    
    # Compare results across episodes
    print("\nEpisode Comparison:")
    print("-" * 40)
    for name, results in episode_results.items():
        print(f"{name:15s}: Energy={results['final_energy']:.6f}, "
              f"Max={results['final_max']:.6f}")
    
    # Visualize all final distributions
    mesh_points = env.get_mesh_points()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (name, results) in enumerate(episode_results.items()):
        if i < len(axes):
            scatter = axes[i].scatter(mesh_points[:, 0], mesh_points[:, 1], 
                                    c=results['final_solution'], cmap='hot', s=15)
            axes[i].set_title(f'{name}\nFinal Energy: {results["final_energy"]:.4f}')
            axes[i].set_aspect('equal')
            plt.colorbar(scatter, ax=axes[i])
    
    # Hide the last subplot if we have fewer episodes
    if len(episode_results) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('multiple_episodes_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved episode comparison to 'multiple_episodes_comparison.png'")

def example_3_parameter_study():
    """
    Study how different physical parameters affect the simulation.
    
    This demonstrates how to use the environment for parameter exploration,
    which might be useful if your RL agent controls physical parameters.
    """
    print("\n" + "="*50)
    print("Example 3: Parameter Study")
    print("="*50)
    
    # Study different diffusion coefficients
    diffusion_coeffs = [0.05, 0.1, 0.2, 0.5]
    
    # Fixed initial condition for fair comparison
    def standard_initial(x: float, y: float) -> float:
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    results = []
    
    print(f"Testing {len(diffusion_coeffs)} different diffusion coefficients...")
    
    for K in diffusion_coeffs:
        print(f"  Testing K = {K}")
        
        # Create environment with specific diffusion coefficient
        env = diffusion_env.DiffusionEnvironment(
            refinement_level=4,
            diffusion_coeff=K,
            dt=0.005,  # Smaller time step for accuracy
            final_time=0.3
        )
        
        env.reset(standard_initial)
        
        # Track energy decay over time
        times = []
        energies = []
        
        while not env.is_done():
            times.append(env.get_time())
            quantities = env.get_physical_quantities()
            energies.append(quantities[0])
            env.step()
        
        results.append({
            'K': K,
            'times': np.array(times),
            'energies': np.array(energies)
        })
    
    # Visualize parameter study results
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.plot(result['times'], result['energies'], 
                linewidth=2, label=f'K = {result["K"]}')
    
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Energy Decay for Different Diffusion Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale shows exponential decay clearly
    plt.savefig('parameter_study_results.png', dpi=150, bbox_inches='tight')
    print("Saved parameter study to 'parameter_study_results.png'")
    
    # Theoretical verification
    print("\nTheoretical vs. Numerical Comparison:")
    print("For the sine wave initial condition, theory predicts exponential decay")
    print("with rate constant 2π²K. Let's check our numerical results:")
    print("-" * 50)
    
    for result in results:
        K = result['K']
        theoretical_rate = 2 * np.pi**2 * K
        
        # Fit exponential to numerical data (later times)
        mask = result['times'] > 0.05  # Skip early transients
        if np.sum(mask) > 5:
            log_energies = np.log(result['energies'][mask])
            times_fit = result['times'][mask]
            # Linear fit to log(energy) vs time gives decay rate
            coeffs = np.polyfit(times_fit, log_energies, 1)
            numerical_rate = -coeffs[0]
            
            print(f"K = {K:4.2f}: Theory = {theoretical_rate:.4f}, "
                  f"Numerical = {numerical_rate:.4f}, "
                  f"Error = {abs(theoretical_rate - numerical_rate)/theoretical_rate*100:.1f}%")

def example_4_rl_integration_pattern():
    """
    Demonstrate integration patterns for reinforcement learning frameworks.
    
    This shows how to structure your code for efficient RL training,
    including state representation and reward computation.
    """
    print("\n" + "="*50)
    print("Example 4: RL Integration Pattern")
    print("="*50)
    
    class DiffusionRLEnvironment:
        """
        Wrapper that adapts our diffusion environment for RL frameworks.
        
        This class provides the standard RL interface (reset, step, etc.)
        while handling state representation and reward computation.
        """
        
        def __init__(self, **kwargs):
            self.env = diffusion_env.DiffusionEnvironment(**kwargs)
            self.initial_energy = None
            
        def reset(self, initial_condition_params: np.ndarray) -> np.ndarray:
            """
            Reset environment and return initial state.
            
            Args:
                initial_condition_params: Parameters defining initial condition
                
            Returns:
                state: Compact state representation for RL agent
            """
            # Convert parameters to initial condition function
            # This is where your RL agent's "actions" affect the environment
            center_x, center_y, width = initial_condition_params
            
            def parameterized_initial(x, y):
                return np.exp(-width * ((x - center_x)**2 + (y - center_y)**2))
            
            self.env.reset(parameterized_initial)
            
            # Store initial energy for reward computation
            quantities = self.env.get_physical_quantities()
            self.initial_energy = quantities[0]
            
            return self._get_state()
        
        def step(self) -> Tuple[np.ndarray, float, bool, dict]:
            """
            Advance simulation one step and return RL tuple.
            
            Returns:
                state: New state representation
                reward: Reward signal for RL agent  
                done: Whether episode is complete
                info: Additional debugging information
            """
            self.env.step()
            
            state = self._get_state()
            reward = self._compute_reward()
            done = self.env.is_done()
            info = {
                'time': self.env.get_time(),
                'timestep': self.env.get_timestep()
            }
            
            return state, reward, done, info
        
        def _get_state(self) -> np.ndarray:
            """
            Convert full finite element solution to compact state representation.
            
            This is crucial for RL efficiency - the full FE solution might have
            hundreds or thousands of values, but RL works better with compact states.
            """
            # Get physical quantities (5 values)
            quantities = self.env.get_physical_quantities()
            
            # Get some spatial statistics
            solution = self.env.get_solution_data()
            mesh_points = self.env.get_mesh_points()
            
            # Compute additional features
            solution_array = np.array(solution)
            std_dev = np.std(solution_array)
            skewness = self._compute_skewness(solution_array)
            
            # Spatial moment features
            weighted_x_variance = self._compute_spatial_variance(solution_array, mesh_points, 0)
            weighted_y_variance = self._compute_spatial_variance(solution_array, mesh_points, 1)
            
            # Combine into compact state vector
            state = np.array([
                quantities[0],  # total energy
                quantities[1],  # max value
                quantities[3],  # energy center x
                quantities[4],  # energy center y
                std_dev,        # spatial standard deviation
                skewness,       # solution skewness
                weighted_x_variance,  # x-direction spread
                weighted_y_variance,  # y-direction spread
                self.env.get_time()   # current time
            ])
            
            return state
        
        def _compute_reward(self) -> float:
            """
            Compute reward signal for RL agent.
            
            This is where you encode what you want the agent to learn.
            Different reward functions will lead to different behaviors.
            """
            quantities = self.env.get_physical_quantities()
            current_energy = quantities[0]
            max_value = quantities[1]
            
            # Example reward: maintain high energy longer (resist diffusion)
            energy_preservation_reward = current_energy / self.initial_energy
            
            # Example reward: maintain localized heat (high max value)
            localization_reward = max_value
            
            # Combine rewards with weights
            total_reward = 0.7 * energy_preservation_reward + 0.3 * localization_reward
            
            return total_reward
        
        def _compute_skewness(self, data: np.ndarray) -> float:
            """Compute skewness of solution distribution."""
            if np.std(data) < 1e-12:
                return 0.0
            normalized = (data - np.mean(data)) / np.std(data)
            return np.mean(normalized**3)
        
        def _compute_spatial_variance(self, solution: np.ndarray, 
                                    mesh_points: np.ndarray, axis: int) -> float:
            """Compute weighted spatial variance along specified axis."""
            weights = np.abs(solution)
            if np.sum(weights) < 1e-12:
                return 0.0
            
            coords = mesh_points[:, axis]
            weighted_mean = np.sum(weights * coords) / np.sum(weights)
            weighted_variance = np.sum(weights * (coords - weighted_mean)**2) / np.sum(weights)
            
            return weighted_variance
    
    # Demonstrate the RL environment wrapper
    print("Testing RL environment wrapper...")
    
    rl_env = DiffusionRLEnvironment(refinement_level=3, final_time=0.2)
    
    # Simulate a few RL episodes
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}:")
        
        # Random initial condition parameters (this is what RL agent would choose)
        params = np.array([
            np.random.uniform(0.2, 0.8),  # center_x
            np.random.uniform(0.2, 0.8),  # center_y  
            np.random.uniform(10, 30)     # width
        ])
        
        print(f"  Initial condition: center=({params[0]:.2f}, {params[1]:.2f}), width={params[2]:.1f}")
        
        # Reset and run episode
        state = rl_env.reset(params)
        total_reward = 0.0
        step_count = 0
        
        while True:
            state, reward, done, info = rl_env.step()
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        print(f"  Completed in {step_count} steps, total reward: {total_reward:.4f}")
        print(f"  Final state summary: energy={state[0]:.4f}, max_val={state[1]:.4f}")

def performance_benchmark():
    """
    Benchmark the environment performance for RL training.
    
    Understanding performance characteristics helps you optimize
    your RL training pipeline.
    """
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    # Test different refinement levels
    refinement_levels = [3, 4, 5]
    
    def simple_initial(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    for level in refinement_levels:
        print(f"\nRefinement level {level}:")
        
        env = diffusion_env.DiffusionEnvironment(
            refinement_level=level,
            final_time=0.1
        )
        
        print(f"  Degrees of freedom: {env.get_num_dofs()}")
        
        # Time environment creation (one-time cost)
        start_time = time.time()
        env.reset(simple_initial)
        reset_time = time.time() - start_time
        
        # Time stepping (repeated cost)
        start_time = time.time()
        step_count = 0
        while not env.is_done():
            env.step()
            step_count += 1
        stepping_time = time.time() - start_time
        
        # Time data extraction (RL interface cost)
        start_time = time.time()
        for _ in range(100):  # Simulate multiple extractions
            solution = env.get_solution_data()
            quantities = env.get_physical_quantities()
        extraction_time = (time.time() - start_time) / 100
        
        print(f"  Reset time: {reset_time*1000:.2f} ms")
        print(f"  Time per step: {stepping_time/step_count*1000:.2f} ms ({step_count} steps)")
        print(f"  Data extraction: {extraction_time*1000:.2f} ms per call")
        print(f"  Total episode time: {(reset_time + stepping_time)*1000:.2f} ms")

if __name__ == "__main__":
    """
    Run all examples to demonstrate the complete functionality.
    
    You can run individual examples by commenting out the ones you don't want.
    """
    print("Diffusion Environment - Complete Examples")
    print("="*60)
    
    # Run all examples
    example_1_basic_simulation()
    example_2_multiple_episodes() 
    example_3_parameter_study()
    example_4_rl_integration_pattern()
    performance_benchmark()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("Use the final VTU file with VisIt for 3D visualization.")
    print("="*60)