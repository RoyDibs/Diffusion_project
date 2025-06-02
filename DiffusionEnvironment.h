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

#include <functional>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iostream>

namespace TransientDiffusion {
using namespace dealii;

/**
 * Python-friendly wrapper around the transient diffusion solver.
 * This class is designed specifically to be exposed to Python via pybind11,
 * so all public methods use simple data types that can be easily converted.
 */
class DiffusionEnvironmentWrapper {
public:
    /**
     * Constructor sets up the finite element problem structure.
     * This does all the expensive one-time setup: mesh generation,
     * degree of freedom distribution, and matrix assembly.
     * 
     * @param refinement_level Controls mesh resolution (higher = finer mesh)
     * @param diffusion_coeff Physical diffusion coefficient K
     * @param dt Time step size
     * @param final_t Total simulation time
     */
    DiffusionEnvironmentWrapper(int refinement_level = 4, 
                               double diffusion_coeff = 0.1,
                               double dt = 0.01, 
                               double final_t = 1.0);

    /**
     * Reset the simulation with a new initial condition.
     * This is equivalent to the reset() method in RL environments.
     * The initial_condition function should take x,y coordinates and return
     * the initial value at that point.
     */
    void reset(const std::function<double(double, double)>& initial_condition);

    /**
     * Advance the simulation by one time step.
     * This solves the linear system for the next time level.
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
     * This returns the finite element solution at all degrees of freedom.
     * The ordering corresponds to the mesh points returned by get_mesh_points().
     */
    std::vector<double> get_solution_data() const;

    /**
     * Get the physical coordinates of all mesh points.
     * Returns a vector of [x,y] coordinate pairs.
     * This tells you where each solution value in get_solution_data() is located.
     */
    std::vector<std::array<double, 2>> get_mesh_points() const;

    /**
     * Get number of degrees of freedom (mesh points).
     * Useful for understanding the size of your state space.
     */
    size_t get_num_dofs() const { return dof_handler.n_dofs(); }

    /**
     * Optional: write current solution to VTU file for visualization.
     * Only use this for episodes you want to examine in detail.
     */
    void write_vtk(const std::string& filename) const;

    /**
     * Get some useful physical quantities for RL state representation.
     * Returns [total_energy, max_value, min_value, energy_center_x, energy_center_y]
     */
    std::vector<double> get_physical_quantities() const;

    /**
     * Allow modification of physical parameters during simulation.
     * Useful if your RL agent wants to control these as actions.
     */
    void set_diffusion_coefficient(double K_new);

private:
    // Core deal.II components - same as your original code but organized for reuse
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
    double K;  // Diffusion coefficient (now modifiable)

    // Pre-computed data for efficient Python interface
    std::vector<std::array<double, 2>> mesh_coordinates;
    bool system_initialized;

    // Private methods that handle the FEM mechanics
    void setup_system();
    void assemble_system();
    void solve_time_step();
    void compute_mesh_coordinates();
    void rebuild_system_matrix();  // Called when K changes
};

/**
 * Helper class to convert Python functions to deal.II Functions.
 * This bridges the gap between Python callable objects and deal.II's
 * Function interface.
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