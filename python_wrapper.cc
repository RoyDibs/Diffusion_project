#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "DiffusionEnvironment.h"

namespace py = pybind11;

PYBIND11_MODULE(diffusion_env, m) {
    m.doc() = R"pbdoc(
        Deal.II based diffusion environment for reinforcement learning
        
        This module provides a high-performance finite element environment
        for solving transient diffusion equations with support for multiple
        geometry types and customizable domain sizes.
        
        Supported geometries:
        - Hyper cube: Square domain with customizable side length
        - Hyper rectangle: Rectangular domain with custom width and height  
        - Hyper ball: Circular domain with custom radius
        - Hyper shell: Annular domain with custom inner and outer radii
        - L-shaped: L-shaped domain with custom size
        - Quarter hyper ball: Quarter-circle domain with custom radius
        
        Example usage:
            import diffusion_env
            import numpy as np
            
            # Create a circular environment
            config = diffusion_env.GeometryConfig.hyper_ball(radius=2.0)
            env = diffusion_env.DiffusionEnvironment(config, refinement_level=4)
            
            # Define initial condition
            def initial_condition(x, y):
                return np.exp(-((x-1)**2 + (y-1)**2))
            
            # Run simulation
            env.reset(initial_condition)
            while not env.is_done():
                env.step()
                solution = env.get_solution_data()
    )pbdoc";

    // Expose the GeometryType enumeration
    py::enum_<TransientDiffusion::GeometryType>(m, "GeometryType", R"pbdoc(
        Enumeration of supported geometry types.
        
        Each geometry type has different parameters that control its size and shape.
    )pbdoc")
        .value("HYPER_CUBE", TransientDiffusion::GeometryType::HYPER_CUBE, 
               "Square domain with customizable side length")
        .value("HYPER_RECTANGLE", TransientDiffusion::GeometryType::HYPER_RECTANGLE,
               "Rectangular domain with custom width and height")
        .value("HYPER_BALL", TransientDiffusion::GeometryType::HYPER_BALL,
               "Circular domain with custom radius")
        .value("HYPER_SHELL", TransientDiffusion::GeometryType::HYPER_SHELL,
               "Annular (ring) domain with custom inner and outer radii")
        .value("L_SHAPED", TransientDiffusion::GeometryType::L_SHAPED,
               "L-shaped domain with custom size")
        .value("QUARTER_HYPER_BALL", TransientDiffusion::GeometryType::QUARTER_HYPER_BALL,
               "Quarter-circle domain with custom radius")
        .export_values();

    // Expose the GeometryConfig structure
    py::class_<TransientDiffusion::GeometryConfig>(m, "GeometryConfig", R"pbdoc(
        Configuration structure for specifying geometry type and parameters.
        
        Different geometry types use different subsets of the available parameters.
        Use the static factory methods for convenient configuration.
    )pbdoc")
        .def(py::init<>(), "Create default geometry configuration (unit hyper cube)")
        
        // Expose all fields
        .def_readwrite("type", &TransientDiffusion::GeometryConfig::type,
                      "Geometry type (GeometryType enum)")
        .def_readwrite("size", &TransientDiffusion::GeometryConfig::size,
                      "Side length for HYPER_CUBE or general size for L_SHAPED")
        .def_readwrite("width", &TransientDiffusion::GeometryConfig::width,
                      "Width for HYPER_RECTANGLE")
        .def_readwrite("height", &TransientDiffusion::GeometryConfig::height,
                      "Height for HYPER_RECTANGLE")
        .def_readwrite("radius", &TransientDiffusion::GeometryConfig::radius,
                      "Radius for HYPER_BALL and QUARTER_HYPER_BALL")
        .def_readwrite("inner_radius", &TransientDiffusion::GeometryConfig::inner_radius,
                      "Inner radius for HYPER_SHELL")
        .def_readwrite("outer_radius", &TransientDiffusion::GeometryConfig::outer_radius,
                      "Outer radius for HYPER_SHELL")
        .def_readwrite("l_size", &TransientDiffusion::GeometryConfig::l_size,
                      "Size parameter for L_SHAPED domain")
        
        // Static factory methods for convenient configuration
        .def_static("hyper_cube", &TransientDiffusion::GeometryConfig::hyper_cube,
                   R"pbdoc(
                       Create configuration for a square domain.
                       
                       Args:
                           side_length (float): Side length of the square (default: 1.0)
                       
                       Returns:
                           GeometryConfig: Configuration for hyper cube geometry
                   )pbdoc",
                   py::arg("side_length") = 1.0)
                   
        .def_static("hyper_rectangle", &TransientDiffusion::GeometryConfig::hyper_rectangle,
                   R"pbdoc(
                       Create configuration for a rectangular domain.
                       
                       Args:
                           width (float): Width of the rectangle
                           height (float): Height of the rectangle
                       
                       Returns:
                           GeometryConfig: Configuration for hyper rectangle geometry
                   )pbdoc",
                   py::arg("width"), py::arg("height"))
                   
        .def_static("hyper_ball", &TransientDiffusion::GeometryConfig::hyper_ball,
                   R"pbdoc(
                       Create configuration for a circular domain.
                       
                       Args:
                           radius (float): Radius of the circle
                           center (tuple): Center point as (x, y) coordinates (default: (0, 0))
                       
                       Returns:
                           GeometryConfig: Configuration for hyper ball geometry
                   )pbdoc",
                   py::arg("radius"), py::arg("center") = std::array<double, 2>{0.0, 0.0})
                   
        .def_static("hyper_shell", &TransientDiffusion::GeometryConfig::hyper_shell,
                   R"pbdoc(
                       Create configuration for an annular (ring) domain.
                       
                       Args:
                           inner_radius (float): Inner radius of the annulus
                           outer_radius (float): Outer radius of the annulus
                           center (tuple): Center point as (x, y) coordinates (default: (0, 0))
                       
                       Returns:
                           GeometryConfig: Configuration for hyper shell geometry
                   )pbdoc",
                   py::arg("inner_radius"), py::arg("outer_radius"), 
                   py::arg("center") = std::array<double, 2>{0.0, 0.0})
                   
        .def_static("l_shaped", &TransientDiffusion::GeometryConfig::l_shaped,
                   R"pbdoc(
                       Create configuration for an L-shaped domain.
                       
                       Args:
                           domain_size (float): Size parameter for the L-domain (default: 1.0)
                       
                       Returns:
                           GeometryConfig: Configuration for L-shaped geometry
                   )pbdoc",
                   py::arg("domain_size") = 1.0)
                   
        .def_static("quarter_hyper_ball", &TransientDiffusion::GeometryConfig::quarter_hyper_ball,
                   R"pbdoc(
                       Create configuration for a quarter-circle domain.
                       
                       Args:
                           radius (float): Radius of the quarter circle
                           center (tuple): Center point as (x, y) coordinates (default: (0, 0))
                       
                       Returns:
                           GeometryConfig: Configuration for quarter hyper ball geometry
                   )pbdoc",
                   py::arg("radius"), py::arg("center") = std::array<double, 2>{0.0, 0.0});

    // Expose the main DiffusionEnvironmentWrapper class
    py::class_<TransientDiffusion::DiffusionEnvironmentWrapper>(m, "DiffusionEnvironment", R"pbdoc(
        High-performance transient diffusion solver with multiple geometry support.
        
        This class wraps a deal.II finite element solver with support for various
        domain geometries including squares, rectangles, circles, annuli, and L-shaped domains.
    )pbdoc")
    
        // Main constructor with geometry configuration
        .def(py::init<const TransientDiffusion::GeometryConfig&, int, double, double, double>(),
             R"pbdoc(
                 Create a new diffusion environment with custom geometry.
                 
                 Args:
                     geometry_config (GeometryConfig): Geometry configuration specifying
                         domain type and dimensions
                     refinement_level (int): Mesh refinement level (default: 4)
                     diffusion_coeff (float): Physical diffusion coefficient (default: 0.1)
                     dt (float): Time step size (default: 0.01)
                     final_time (float): Total simulation time (default: 1.0)
                 
                 Example:
                     # Create a circular domain with radius 2.0
                     config = GeometryConfig.hyper_ball(2.0)
                     env = DiffusionEnvironment(config, refinement_level=5)
             )pbdoc",
             py::arg("geometry_config") = TransientDiffusion::GeometryConfig::hyper_cube(),
             py::arg("refinement_level") = 4,
             py::arg("diffusion_coeff") = 0.1,
             py::arg("dt") = 0.01,
             py::arg("final_time") = 1.0)

        // Backward compatibility constructor (creates unit hyper cube)
        .def(py::init<int, double, double, double>(),
             R"pbdoc(
                 Create a new diffusion environment with unit hyper cube geometry.
                 
                 This constructor is provided for backward compatibility.
                 For new code, prefer using the geometry_config constructor.
                 
                 Args:
                     refinement_level (int): Mesh refinement level
                     diffusion_coeff (float): Physical diffusion coefficient
                     dt (float): Time step size
                     final_time (float): Total simulation time
             )pbdoc",
             py::arg("refinement_level") = 4,
             py::arg("diffusion_coeff") = 0.1,
             py::arg("dt") = 0.01,
             py::arg("final_time") = 1.0)

        // All the existing methods
        .def("reset", &TransientDiffusion::DiffusionEnvironmentWrapper::reset,
             R"pbdoc(
                 Reset the environment with a new initial condition.
                 
                 Args:
                     initial_condition (callable): Function taking (x, y) coordinates
                         and returning the initial temperature at that point.
                         
                 The initial condition function should handle the coordinate system
                 of your chosen geometry. For example, a centered circular domain
                 will have coordinates roughly in [-radius, radius] x [-radius, radius].
             )pbdoc",
             py::arg("initial_condition"))

        .def("step", &TransientDiffusion::DiffusionEnvironmentWrapper::step,
             "Advance the simulation by one time step.")

        .def("get_time", &TransientDiffusion::DiffusionEnvironmentWrapper::get_time,
             "Get the current simulation time.")

        .def("is_done", &TransientDiffusion::DiffusionEnvironmentWrapper::is_done,
             "Check if the simulation has reached the final time.")

        .def("get_timestep", &TransientDiffusion::DiffusionEnvironmentWrapper::get_timestep,
             "Get the current time step number.")

        .def("get_num_dofs", &TransientDiffusion::DiffusionEnvironmentWrapper::get_num_dofs,
             "Get the number of degrees of freedom (mesh points) in the simulation.")

        .def("get_solution_data", &TransientDiffusion::DiffusionEnvironmentWrapper::get_solution_data,
             R"pbdoc(
                 Extract current solution values as a NumPy array.
                 
                 Returns:
                     numpy.ndarray: Solution values at all degrees of freedom.
                 
                 The coordinate locations for these values are given by get_mesh_points().
             )pbdoc")

        .def("get_mesh_points", &TransientDiffusion::DiffusionEnvironmentWrapper::get_mesh_points,
             R"pbdoc(
                 Get physical coordinates of all mesh points.
                 
                 Returns:
                     numpy.ndarray: Coordinate array with shape (num_dofs, 2).
                         Each row contains [x, y] coordinates of a mesh point.
                 
                 The coordinate ranges depend on your geometry choice:
                 - Hyper cube: [0, size] x [0, size]
                 - Hyper rectangle: [0, width] x [0, height]  
                 - Hyper ball: approximately [-radius, radius] x [-radius, radius]
                 - etc.
             )pbdoc")

        .def("get_physical_quantities", &TransientDiffusion::DiffusionEnvironmentWrapper::get_physical_quantities,
             R"pbdoc(
                 Get physically meaningful derived quantities.
                 
                 Returns:
                     numpy.ndarray: Array containing [total_energy, max_value, min_value,
                         energy_center_x, energy_center_y].
                 
                 These quantities provide a compact representation of the solution state
                 that's often more useful for RL than the raw finite element data.
             )pbdoc")

        .def("write_vtk", &TransientDiffusion::DiffusionEnvironmentWrapper::write_vtk,
             R"pbdoc(
                 Write current solution to VTK file for visualization.
                 
                 Args:
                     filename (str): Output filename (should end in .vtu).
                 
                 The VTK file can be opened in ParaView or other visualization tools
                 to examine the solution on your custom geometry.
             )pbdoc",
             py::arg("filename"))

        .def("set_diffusion_coefficient", &TransientDiffusion::DiffusionEnvironmentWrapper::set_diffusion_coefficient,
             R"pbdoc(
                 Change the diffusion coefficient during simulation.
                 
                 Args:
                     K_new (float): New diffusion coefficient (must be positive).
             )pbdoc",
             py::arg("K_new"))

        // New methods for geometry information
        .def("get_geometry_config", &TransientDiffusion::DiffusionEnvironmentWrapper::get_geometry_config,
             R"pbdoc(
                 Get the current geometry configuration.
                 
                 Returns:
                     GeometryConfig: The geometry configuration used to create this environment.
             )pbdoc")

        .def("get_geometry_description", &TransientDiffusion::DiffusionEnvironmentWrapper::get_geometry_description,
             R"pbdoc(
                 Get a human-readable description of the current geometry.
                 
                 Returns:
                     str: Descriptive string of the geometry type and parameters.
             )pbdoc")

        .def("get_domain_bounds", &TransientDiffusion::DiffusionEnvironmentWrapper::get_domain_bounds,
             R"pbdoc(
                 Get the bounding box of the computational domain.
                 
                 Returns:
                     list: [x_min, x_max, y_min, y_max] bounding the domain.
                 
                 This is useful for understanding the coordinate system of your geometry
                 and for setting up appropriate initial conditions.
             )pbdoc");

    // Module-level constants and information
    m.attr("__version__") = "1.1.0";
    
    // Useful defaults
    m.attr("DEFAULT_REFINEMENT") = 4;
    m.attr("DEFAULT_DIFFUSION_COEFF") = 0.1;
    m.attr("DEFAULT_TIME_STEP") = 0.01;
    m.attr("DEFAULT_FINAL_TIME") = 1.0;
    
    // Add some example configurations as module-level convenience functions
    m.def("create_square_env", 
          [](double size, int refinement) {
              auto config = TransientDiffusion::GeometryConfig::hyper_cube(size);
              return TransientDiffusion::DiffusionEnvironmentWrapper(config, refinement);
          },
          R"pbdoc(
              Convenience function to create a square domain environment.
              
              Args:
                  size (float): Side length of the square
                  refinement (int): Mesh refinement level
              
              Returns:
                  DiffusionEnvironment: Environment with square geometry
          )pbdoc",
          py::arg("size") = 1.0, py::arg("refinement") = 4);
          
    m.def("create_circle_env", 
          [](double radius, int refinement) {
              auto config = TransientDiffusion::GeometryConfig::hyper_ball(radius);
              return TransientDiffusion::DiffusionEnvironmentWrapper(config, refinement);
          },
          R"pbdoc(
              Convenience function to create a circular domain environment.
              
              Args:
                  radius (float): Radius of the circle
                  refinement (int): Mesh refinement level
              
              Returns:
                  DiffusionEnvironment: Environment with circular geometry
          )pbdoc",
          py::arg("radius") = 1.0, py::arg("refinement") = 4);

    m.def("create_rectangle_env", 
          [](double width, double height, int refinement) {
              auto config = TransientDiffusion::GeometryConfig::hyper_rectangle(width, height);
              return TransientDiffusion::DiffusionEnvironmentWrapper(config, refinement);
          },
          R"pbdoc(
              Convenience function to create a rectangular domain environment.
              
              Args:
                  width (float): Width of the rectangle
                  height (float): Height of the rectangle
                  refinement (int): Mesh refinement level
              
              Returns:
                  DiffusionEnvironment: Environment with rectangular geometry
          )pbdoc",
          py::arg("width") = 1.0, py::arg("height") = 1.0, py::arg("refinement") = 4);
}