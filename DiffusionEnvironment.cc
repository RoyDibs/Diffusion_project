#include "DiffusionEnvironment.h"

namespace TransientDiffusion {

DiffusionEnvironmentWrapper::DiffusionEnvironmentWrapper(int refinement_level, 
                                                       double diffusion_coeff,
                                                       double dt, 
                                                       double final_t)
    : fe(1), dof_handler(triangulation), time(0.0), time_step(dt), 
      final_time(final_t), timestep_number(0), K(diffusion_coeff), 
      system_initialized(false) {
    
    // Create the computational mesh - this is expensive but only done once
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(refinement_level);
    
    // Set up the finite element system structure
    setup_system();
    
    // Pre-compute mesh coordinates for efficient Python access
    compute_mesh_coordinates();
    
    system_initialized = true;
}

void DiffusionEnvironmentWrapper::setup_system() {
    // Distribute degrees of freedom - this assigns a unique number to each mesh vertex
    dof_handler.distribute_dofs(fe);
    
    // Create the sparsity pattern - this determines which matrix entries can be non-zero
    // This step is crucial for memory efficiency in large problems
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Initialize all matrices with the same sparsity pattern
    // Mass matrix: represents the time derivative term
    // Stiffness matrix: represents the spatial derivative (diffusion) term
    // System matrix: combination of mass and stiffness for implicit time stepping
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    // Initialize solution vectors
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Assemble the time-independent matrices
    // These represent the spatial discretization and don't change between time steps
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<2>(2), mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<2>(2), stiffness_matrix);
    
    // Build the system matrix for implicit time stepping
    // This combines mass and stiffness matrices according to the theta-method
    rebuild_system_matrix();

    // Set up boundary conditions - zero Dirichlet on all boundaries
    // This means the temperature is fixed at zero on the domain boundary
    VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ZeroFunction<2>(), boundary_values);
}

void DiffusionEnvironmentWrapper::rebuild_system_matrix() {
    // System matrix = M + theta * dt * K * A
    // where M is mass matrix, A is stiffness matrix, K is diffusion coefficient
    // This matrix gets solved at each time step
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * time_step * K, stiffness_matrix);
}

void DiffusionEnvironmentWrapper::compute_mesh_coordinates() {
    // Extract the physical coordinates of all degrees of freedom
    // This is computed once and reused for efficient Python access
    mesh_coordinates.resize(dof_handler.n_dofs());
    std::vector<Point<2>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler, support_points);
    
    for (size_t i = 0; i < support_points.size(); ++i) {
        mesh_coordinates[i][0] = support_points[i][0];
        mesh_coordinates[i][1] = support_points[i][1];
    }
}

void DiffusionEnvironmentWrapper::reset(const std::function<double(double, double)>& initial_condition) {
    if (!system_initialized) {
        throw std::runtime_error("System not properly initialized");
    }
    
    // Reset time variables - this starts a new episode
    time = 0.0;
    timestep_number = 0;
    
    // Apply the new initial condition
    // We create a deal.II Function wrapper around the Python function
    PythonInitialCondition initial_func(initial_condition);
    VectorTools::interpolate(dof_handler, initial_func, old_solution);
    solution = old_solution;
}

void DiffusionEnvironmentWrapper::step() {
    if (time >= final_time) {
        // Simulation is complete, no more steps to take
        return;
    }
    
    // Advance time first
    time += time_step;
    timestep_number++;
    
    // Solve for the new time level
    assemble_system();
    solve_time_step();
    
    // Update solution for next iteration
    old_solution = solution;
}

void DiffusionEnvironmentWrapper::assemble_system() {
    // Build the right-hand side for the linear system
    // This represents the known information from the previous time step
    
    // Start with zero right-hand side
    system_rhs = 0;
    
    // Add contribution from mass matrix times old solution
    // This represents the time derivative term
    mass_matrix.vmult(system_rhs, old_solution);
    
    // Add contribution from diffusion at previous time step
    // This handles the explicit part of the time stepping scheme
    Vector<double> tmp(dof_handler.n_dofs());
    stiffness_matrix.vmult(tmp, old_solution);
    system_rhs.add(-time_step * (1 - theta) * K, tmp);
    
    // Note: we're using theta=1 (fully implicit), so the second term is zero
    // But this structure allows for different time stepping schemes if needed
}

void DiffusionEnvironmentWrapper::solve_time_step() {
    // Apply boundary conditions to the linear system
    // This modifies the system matrix and right-hand side to enforce zero Dirichlet BCs
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
    
    // Solve the linear system using conjugate gradient method
    // For small problems, this is very fast. For large problems, you might want
    // to use more sophisticated preconditioners
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

std::vector<double> DiffusionEnvironmentWrapper::get_solution_data() const {
    // Convert deal.II Vector to std::vector for Python compatibility
    // This creates a copy of the data, which is safe but has memory overhead
    // For very large problems, you might want to consider view-based approaches
    std::vector<double> solution_data(solution.size());
    for (size_t i = 0; i < solution.size(); ++i) {
        solution_data[i] = solution[i];
    }
    return solution_data;
}

std::vector<std::array<double, 2>> DiffusionEnvironmentWrapper::get_mesh_points() const {
    // Return the pre-computed mesh coordinates
    // This avoids recomputing the coordinates every time Python asks for them
    return mesh_coordinates;
}

std::vector<double> DiffusionEnvironmentWrapper::get_physical_quantities() const {
    // Compute physically meaningful quantities that might be useful for RL
    // This gives your agent access to global properties rather than just local values
    
    std::vector<double> quantities(5);
    
    // Total energy in the system (integral of u over domain)
    // This is computed using the mass matrix as a quadrature rule
    Vector<double> energy_vector(solution.size());
    mass_matrix.vmult(energy_vector, solution);
    double total_energy = solution * energy_vector;
    quantities[0] = total_energy;
    
    // Maximum and minimum values
    double max_val = *std::max_element(solution.begin(), solution.end());
    double min_val = *std::min_element(solution.begin(), solution.end());
    quantities[1] = max_val;
    quantities[2] = min_val;
    
    // Center of energy (weighted average position)
    double weighted_x = 0.0, weighted_y = 0.0, total_weight = 0.0;
    for (size_t i = 0; i < solution.size(); ++i) {
        double weight = std::abs(solution[i]);
        weighted_x += weight * mesh_coordinates[i][0];
        weighted_y += weight * mesh_coordinates[i][1];
        total_weight += weight;
    }
    
    if (total_weight > 1e-12) {
        quantities[3] = weighted_x / total_weight;
        quantities[4] = weighted_y / total_weight;
    } else {
        quantities[3] = 0.5;  // Default to center if no energy
        quantities[4] = 0.5;
    }
    
    return quantities;
}

void DiffusionEnvironmentWrapper::write_vtk(const std::string& filename) const {
    // Standard deal.II output routine for visualization
    // Only use this for episodes you want to examine in detail
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "temperature");
    data_out.build_patches();
    
    std::ofstream output(filename);
    data_out.write_vtu(output);
}

void DiffusionEnvironmentWrapper::set_diffusion_coefficient(double K_new) {
    // Allow dynamic modification of the diffusion coefficient
    // This could be useful if your RL agent controls physical parameters
    if (K_new <= 0.0) {
        throw std::invalid_argument("Diffusion coefficient must be positive");
    }
    
    K = K_new;
    // Rebuild the system matrix with the new coefficient
    rebuild_system_matrix();
}

} // namespace TransientDiffusion