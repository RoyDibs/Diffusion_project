#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "DiffusionEnvironment.h"

namespace py = pybind11;

/**
 * This is the pybind11 module definition that creates the Python interface.
 * Each .def() call exposes a C++ method to Python with automatic type conversion.
 * The module name "diffusion_env" is what you'll use in Python: import diffusion_env
 */
PYBIND11_MODULE(diffusion_env, m) {
    // Module documentation that appears when users call help() in Python
    m.doc() = R"pbdoc(
        Deal.II based diffusion environment for reinforcement learning
        
        This module provides a high-performance finite element environment
        for solving transient diffusion equations. It's designed specifically
        for reinforcement learning applications where you need to run many
        simulation episodes efficiently.
        
        Example usage:
            import diffusion_env
            import numpy as np
            
            # Create environment
            env = diffusion_env.DiffusionEnvironment(refinement_level=4)
            
            # Define initial condition
            def initial_condition(x, y):
                return np.sin(np.pi * x) * np.sin(np.pi * y)
            
            # Run simulation
            env.reset(initial_condition)
            while not env.is_done():
                env.step()
                solution = env.get_solution_data()  # NumPy array
                points = env.get_mesh_points()      # NumPy array
    )pbdoc";

    // Expose the main DiffusionEnvironmentWrapper class to Python
    py::class_<TransientDiffusion::DiffusionEnvironmentWrapper>(m, "DiffusionEnvironment", R"pbdoc(
        High-performance transient diffusion solver for reinforcement learning.
        
        This class wraps a deal.II finite element solver in a Python-friendly interface.
        It's designed for efficient repeated use in RL training where you need to
        run thousands of simulation episodes.
        
        The solver uses implicit time stepping and linear finite elements on a
        structured quadrilateral mesh. Boundary conditions are zero Dirichlet
        (fixed temperature of zero) on all domain boundaries.
    )pbdoc")
    
        // Constructor with default arguments
        // pybind11 automatically converts Python arguments to C++ types
        .def(py::init<int, double, double, double>(),
             R"pbdoc(
                 Create a new diffusion environment.
                 
                 Args:
                     refinement_level (int): Mesh refinement level (default: 4).
                         Higher values give finer meshes but slower computation.
                         Level 4 gives 289 degrees of freedom, level 5 gives 1089.
                     diffusion_coeff (float): Physical diffusion coefficient (default: 0.1).
                         Higher values make diffusion happen faster.
                     dt (float): Time step size (default: 0.01).
                         Smaller values give more accurate solutions but require more steps.
                     final_time (float): Total simulation time (default: 1.0).
                 
                 The constructor performs all expensive one-time setup operations:
                 mesh generation, finite element space setup, and matrix assembly.
                 This cost is amortized over many episodes.
             )pbdoc",
             py::arg("refinement_level") = 4,
             py::arg("diffusion_coeff") = 0.1,
             py::arg("dt") = 0.01,
             py::arg("final_time") = 1.0)

        // Reset method - this is your RL environment's reset() function
        .def("reset", &TransientDiffusion::DiffusionEnvironmentWrapper::reset,
             R"pbdoc(
                 Reset the environment with a new initial condition.
                 
                 Args:
                     initial_condition (callable): A Python function that takes (x, y) coordinates
                         and returns the initial temperature at that point.
                         
                 Example:
                     # Gaussian heat source at center
                     def gaussian_initial(x, y):
                         return np.exp(-10 * ((x-0.5)**2 + (y-0.5)**2))
                     
                     env.reset(gaussian_initial)
                 
                 This method is very fast because it only changes the initial values,
                 not the underlying finite element structure.
             )pbdoc",
             py::arg("initial_condition"))

        // Step method - advances simulation by one time step
        .def("step", &TransientDiffusion::DiffusionEnvironmentWrapper::step,
             R"pbdoc(
                 Advance the simulation by one time step.
                 
                 This solves the linear system for the next time level using
                 implicit time stepping. The method is unconditionally stable
                 but requires solving a linear system at each step.
                 
                 If the simulation has reached the final time, this method
                 does nothing (check is_done() first).
             )pbdoc")

        // State query methods
        .def("get_time", &TransientDiffusion::DiffusionEnvironmentWrapper::get_time,
             "Get the current simulation time.")

        .def("is_done", &TransientDiffusion::DiffusionEnvironmentWrapper::is_done,
             "Check if the simulation has reached the final time.")

        .def("get_timestep", &TransientDiffusion::DiffusionEnvironmentWrapper::get_timestep,
             "Get the current time step number.")

        .def("get_num_dofs", &TransientDiffusion::DiffusionEnvironmentWrapper::get_num_dofs,
             "Get the number of degrees of freedom (mesh points) in the simulation.")

        // Data extraction methods - these return NumPy arrays
        .def("get_solution_data", &TransientDiffusion::DiffusionEnvironmentWrapper::get_solution_data,
             R"pbdoc(
                 Extract current solution values as a NumPy array.
                 
                 Returns:
                     numpy.ndarray: Solution values at all degrees of freedom.
                         The array has shape (num_dofs,) where num_dofs depends
                         on the mesh refinement level.
                 
                 The ordering of values corresponds to the mesh points returned
                 by get_mesh_points(). This gives you the complete state of the
                 system at the current time.
             )pbdoc")

        .def("get_mesh_points", &TransientDiffusion::DiffusionEnvironmentWrapper::get_mesh_points,
             R"pbdoc(
                 Get physical coordinates of all mesh points.
                 
                 Returns:
                     numpy.ndarray: Coordinate array with shape (num_dofs, 2).
                         Each row contains [x, y] coordinates of a mesh point.
                 
                 This tells you where each value in get_solution_data() is located
                 in the physical domain [0,1] x [0,1].
             )pbdoc")

        .def("get_physical_quantities", &TransientDiffusion::DiffusionEnvironmentWrapper::get_physical_quantities,
             R"pbdoc(
                 Get physically meaningful derived quantities.
                 
                 Returns:
                     numpy.ndarray: Array containing [total_energy, max_value, min_value,
                         energy_center_x, energy_center_y].
                 
                 These quantities are often more useful for RL state representation
                 than the raw finite element data, as they capture the essential
                 physics in a compact form.
                 
                 - total_energy: Integral of solution over the domain
                 - max_value, min_value: Extremal values in the solution
                 - energy_center_x, energy_center_y: Weighted center of the solution
             )pbdoc")

        // Optional output method
        .def("write_vtk", &TransientDiffusion::DiffusionEnvironmentWrapper::write_vtk,
             R"pbdoc(
                 Write current solution to VTK file for visualization.
                 
                 Args:
                     filename (str): Output filename (should end in .vtu).
                 
                 Use this sparingly during RL training - only for episodes you want
                 to examine in detail. For production training, extract data with
                 get_solution_data() instead.
             )pbdoc",
             py::arg("filename"))

        // Parameter modification method
        .def("set_diffusion_coefficient", &TransientDiffusion::DiffusionEnvironmentWrapper::set_diffusion_coefficient,
             R"pbdoc(
                 Change the diffusion coefficient during simulation.
                 
                 Args:
                     K_new (float): New diffusion coefficient (must be positive).
                 
                 This allows your RL agent to control physical parameters as actions.
                 The method rebuilds the system matrix, so use it sparingly within
                 a single episode.
             )pbdoc",
             py::arg("K_new"));

    // Module-level version information
    m.attr("__version__") = "1.0.0";
    
    // Add some useful constants
    m.attr("DEFAULT_REFINEMENT") = 4;
    m.attr("DEFAULT_DIFFUSION_COEFF") = 0.1;
    m.attr("DEFAULT_TIME_STEP") = 0.01;
    m.attr("DEFAULT_FINAL_TIME") = 1.0;
}