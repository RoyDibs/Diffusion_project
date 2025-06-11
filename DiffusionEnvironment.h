#ifndef DIFFUSION_ENVIRONMENT_H
#define DIFFUSION_ENVIRONMENT_H

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <functional>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>

namespace TransientDiffusion {
using namespace dealii;

/**
 * Enumeration of supported geometry types.
 * Each geometry type has its own set of size parameters.
 */
enum class GeometryType {
    HYPER_CUBE,        // Unit cube scaled by size parameter
    HYPER_RECTANGLE,   // Rectangle with custom width and height
    HYPER_BALL,        // Circle/sphere with custom radius
    HYPER_SHELL,       // Annulus with inner and outer radius
    L_SHAPED,          // L-shaped domain with custom size
    QUARTER_HYPER_BALL // Quarter circle with custom radius
};

/**
 * Structure to hold geometry configuration parameters.
 * Different geometry types use different subsets of these parameters.
 */
struct GeometryConfig {
    GeometryType type = GeometryType::HYPER_CUBE;
    
    // For HYPER_CUBE: side length (default 1.0)
    double size = 1.0;
    
    // For HYPER_RECTANGLE: width and height (defaults 1.0, 1.0)
    double width = 1.0;
    double height = 1.0;
    
    // For HYPER_BALL and QUARTER_HYPER_BALL: radius (default 1.0)
    double radius = 1.0;
    
    // For HYPER_SHELL: inner and outer radius (defaults 0.5, 1.0)
    double inner_radius = 0.5;
    double outer_radius = 1.0;
    
    // For L_SHAPED: size of the L-domain (default 1.0)
    double l_size = 1.0;
    
    // Center point for geometries that support translation (default origin)
    Point<2> center = Point<2>(0.0, 0.0);
    
    // Default constructor
    GeometryConfig() = default;
    
    // Convenience constructors for common cases
    static GeometryConfig hyper_cube(double side_length = 1.0) {
        GeometryConfig config;
        config.type = GeometryType::HYPER_CUBE;
        config.size = side_length;
        return config;
    }
    
    static GeometryConfig hyper_rectangle(double w, double h) {
        GeometryConfig config;
        config.type = GeometryType::HYPER_RECTANGLE;
        config.width = w;
        config.height = h;
        return config;
    }
    
    static GeometryConfig hyper_ball(double r, Point<2> center_point = Point<2>(0.0, 0.0)) {
        GeometryConfig config;
        config.type = GeometryType::HYPER_BALL;
        config.radius = r;
        config.center = center_point;
        return config;
    }
    
    static GeometryConfig hyper_shell(double inner_r, double outer_r, Point<2> center_point = Point<2>(0.0, 0.0)) {
        GeometryConfig config;
        config.type = GeometryType::HYPER_SHELL;
        config.inner_radius = inner_r;
        config.outer_radius = outer_r;
        config.center = center_point;
        return config;
    }
    
    static GeometryConfig l_shaped(double domain_size = 1.0) {
        GeometryConfig config;
        config.type = GeometryType::L_SHAPED;
        config.l_size = domain_size;
        return config;
    }
    
    static GeometryConfig quarter_hyper_ball(double r, Point<2> center_point = Point<2>(0.0, 0.0)) {
        GeometryConfig config;
        config.type = GeometryType::QUARTER_HYPER_BALL;
        config.radius = r;
        config.center = center_point;
        return config;
    }
};

/**
 * Python-friendly wrapper around the transient diffusion solver.
 * Now supports multiple geometry types with customizable dimensions.
 */
class DiffusionEnvironmentWrapper {
public:
    /**
     * Constructor sets up the finite element problem structure.
     * 
     * @param geometry_config Configuration specifying geometry type and dimensions
     * @param refinement_level Controls mesh resolution (higher = finer mesh)
     * @param diffusion_coeff Physical diffusion coefficient K
     * @param dt Time step size
     * @param final_t Total simulation time
     */
    DiffusionEnvironmentWrapper(const GeometryConfig& geometry_config = GeometryConfig::hyper_cube(),
                               int refinement_level = 4, 
                               double diffusion_coeff = 0.1,
                               double dt = 0.01, 
                               double final_t = 1.0);

    /**
     * Alternative constructor for backward compatibility.
     * Creates a unit hyper_cube geometry.
     */
    DiffusionEnvironmentWrapper(int refinement_level, 
                               double diffusion_coeff,
                               double dt, 
                               double final_t);

    /**
     * Reset the simulation with a new initial condition.
     * The initial_condition function should take x,y coordinates and return
     * the initial value at that point.
     */
    void reset(const std::function<double(double, double)>& initial_condition);

    /**
     * Advance the simulation by one time step.
     */
    void step();

    /**
     * Get current simulation time.
     */
    double get_time() const { return time; }

    /**
     * Check if simulation has reached the final time.
     */
    bool is_done() const { return time >= final_time; }

    /**
     * Get the current timestep number.
     */
    unsigned int get_timestep() const { return timestep_number; }

    /**
     * Extract solution values as a vector.
     */
    std::vector<double> get_solution_data() const;

    /**
     * Get the physical coordinates of all mesh points.
     */
    std::vector<std::array<double, 2>> get_mesh_points() const;

    /**
     * Get number of degrees of freedom (mesh points).
     */
    size_t get_num_dofs() const { return dof_handler.n_dofs(); }

    /**
     * Write current solution to VTU file for visualization.
     */
    void write_vtk(const std::string& filename) const;

    /**
     * Get useful physical quantities for RL state representation.
     */
    std::vector<double> get_physical_quantities() const;

    /**
     * Allow modification of physical parameters during simulation.
     */
    void set_diffusion_coefficient(double K_new);

    /**
     * Get information about the current geometry configuration.
     */
    GeometryConfig get_geometry_config() const { return geometry_config_; }

    /**
     * Get a string description of the current geometry.
     */
    std::string get_geometry_description() const;

    /**
     * Get the bounding box of the computational domain.
     * Returns [x_min, x_max, y_min, y_max].
     */
    std::vector<double> get_domain_bounds() const;

private:
    // Core deal.II components
    Triangulation<2> triangulation;
    FE_Q<2> fe;
    DoFHandler<2> dof_handler;
    
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> mass_matrix, stiffness_matrix, system_matrix;
    Vector<double> solution, old_solution, system_rhs;
    std::map<types::global_dof_index, double> boundary_values;

    // Simulation parameters
    double time;
    double time_step;
    double final_time;
    unsigned int timestep_number;
    const double theta = 1.0;  // Implicit time stepping parameter
    double K;  // Diffusion coefficient

    // Geometry configuration
    GeometryConfig geometry_config_;

    // Pre-computed data for efficient Python interface
    std::vector<std::array<double, 2>> mesh_coordinates;
    bool system_initialized;

    // Private methods
    void setup_system();
    void assemble_system();
    void solve_time_step();
    void compute_mesh_coordinates();
    void rebuild_system_matrix();
    
    /**
     * Create the computational mesh based on geometry configuration.
     */
    void create_geometry();
    
    /**
     * Validate geometry configuration parameters.
     */
    void validate_geometry_config() const;
};

/**
 * Helper class to convert Python functions to deal.II Functions.
 */
class PythonInitialCondition : public Function<2> {
public:
    PythonInitialCondition(const std::function<double(double, double)>& func) 
        : Function<2>(), python_function(func) {}
    
    virtual double value(const Point<2> &p, const unsigned int = 0) const override {
        return python_function(p[0], p[1]);
    }

private:
    std::function<double(double, double)> python_function;
};

} // namespace TransientDiffusion

#endif // DIFFUSION_ENVIRONMENT_H