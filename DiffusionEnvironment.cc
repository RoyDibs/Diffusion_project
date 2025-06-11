#include "DiffusionEnvironment.h"
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

namespace TransientDiffusion {

DiffusionEnvironmentWrapper::DiffusionEnvironmentWrapper(const GeometryConfig& geometry_config,
                                                       int refinement_level, 
                                                       double diffusion_coeff,
                                                       double dt, 
                                                       double final_t)
    : fe(1), dof_handler(triangulation), time(0.0), time_step(dt), 
      final_time(final_t), timestep_number(0), K(diffusion_coeff), 
      geometry_config_(geometry_config), system_initialized(false) {
    
    // Validate the geometry configuration
    validate_geometry_config();
    
    // Create the computational mesh based on geometry configuration
    create_geometry();
    triangulation.refine_global(refinement_level);
    
    // Set up the finite element system structure
    setup_system();
    
    // Pre-compute mesh coordinates for efficient Python access
    compute_mesh_coordinates();
    
    system_initialized = true;
}

// Backward compatibility constructor
DiffusionEnvironmentWrapper::DiffusionEnvironmentWrapper(int refinement_level, 
                                                       double diffusion_coeff,
                                                       double dt, 
                                                       double final_t)
    : DiffusionEnvironmentWrapper(GeometryConfig::hyper_cube(1.0), refinement_level, 
                                 diffusion_coeff, dt, final_t) {
}

void DiffusionEnvironmentWrapper::validate_geometry_config() const {
    switch (geometry_config_.type) {
        case GeometryType::HYPER_CUBE:
            if (geometry_config_.size <= 0.0) {
                throw std::invalid_argument("Hyper cube size must be positive");
            }
            break;
            
        case GeometryType::HYPER_RECTANGLE:
            if (geometry_config_.width <= 0.0 || geometry_config_.height <= 0.0) {
                throw std::invalid_argument("Rectangle dimensions must be positive");
            }
            break;
            
        case GeometryType::HYPER_BALL:
        case GeometryType::QUARTER_HYPER_BALL:
            if (geometry_config_.radius <= 0.0) {
                throw std::invalid_argument("Ball radius must be positive");
            }
            break;
            
        case GeometryType::HYPER_SHELL:
            if (geometry_config_.inner_radius <= 0.0 || 
                geometry_config_.outer_radius <= geometry_config_.inner_radius) {
                throw std::invalid_argument("Shell radii must satisfy 0 < inner_radius < outer_radius");
            }
            break;
            
        case GeometryType::L_SHAPED:
            if (geometry_config_.l_size <= 0.0) {
                throw std::invalid_argument("L-shaped domain size must be positive");
            }
            break;
            
        default:
            throw std::invalid_argument("Unknown geometry type");
    }
}

void DiffusionEnvironmentWrapper::create_geometry() {
    switch (geometry_config_.type) {
        case GeometryType::HYPER_CUBE: {
            // Create unit cube and scale it
            GridGenerator::hyper_cube(triangulation, 0.0, geometry_config_.size);
            break;
        }
        
        case GeometryType::HYPER_RECTANGLE: {
            // Create rectangle with specified dimensions
            Point<2> bottom_left(0.0, 0.0);
            Point<2> top_right(geometry_config_.width, geometry_config_.height);
            GridGenerator::hyper_rectangle(triangulation, bottom_left, top_right);
            break;
        }
        
        case GeometryType::HYPER_BALL: {
            // Create ball (circle in 2D) with specified radius and center
            GridGenerator::hyper_ball(triangulation, geometry_config_.center, geometry_config_.radius);
            
            // For curved boundaries, we need to set the manifold
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold(0, SphericalManifold<2>(geometry_config_.center));
            break;
        }
        
        case GeometryType::HYPER_SHELL: {
            // Create annulus with specified inner and outer radii
            GridGenerator::hyper_shell(triangulation, geometry_config_.center, 
                                     geometry_config_.inner_radius, geometry_config_.outer_radius);
            
            // Set curved boundary manifolds
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold(0, SphericalManifold<2>(geometry_config_.center));
            break;
        }
        
        case GeometryType::L_SHAPED: {
            // Create L-shaped domain
            // This creates an L-shaped domain by removing the upper-right quadrant from a square
            std::vector<Point<2>> vertices = {
                Point<2>(0.0, 0.0),                                    // 0: bottom-left
                Point<2>(geometry_config_.l_size, 0.0),                // 1: bottom-right
                Point<2>(geometry_config_.l_size, geometry_config_.l_size/2), // 2: middle-right
                Point<2>(geometry_config_.l_size/2, geometry_config_.l_size/2), // 3: middle-center
                Point<2>(geometry_config_.l_size/2, geometry_config_.l_size),   // 4: middle-top
                Point<2>(0.0, geometry_config_.l_size)                 // 5: top-left
            };
            
            std::vector<CellData<2>> cells(3);
            
            // Cell 0: bottom-left rectangle
            cells[0].vertices[0] = 0; cells[0].vertices[1] = 1; 
            cells[0].vertices[2] = 3; cells[0].vertices[3] = 5;
            cells[0].material_id = 0;
            
            // Cell 1: bottom-right rectangle  
            cells[1].vertices[0] = 1; cells[1].vertices[1] = 2; 
            cells[1].vertices[2] = 3; cells[1].vertices[3] = 3; // Note: this creates a degenerate quad, we need to be more careful
            cells[1].material_id = 0;
            
            // Actually, let's use the built-in L-shaped grid generator if available,
            // or create it more carefully. For now, let's use a simpler approach:
            GridGenerator::hyper_L(triangulation, -1, 1);
            
            // Scale the L-domain to the desired size
            GridTools::scale(geometry_config_.l_size / 2.0, triangulation);
            break;
        }
        
        case GeometryType::QUARTER_HYPER_BALL: {
            // Create quarter circle with specified radius
            GridGenerator::quarter_hyper_ball(triangulation, geometry_config_.center, geometry_config_.radius);
            
            // Set curved boundary manifold
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold(0, SphericalManifold<2>(geometry_config_.center));
            break;
        }
        
        default:
            throw std::invalid_argument("Unsupported geometry type");
    }
}

void DiffusionEnvironmentWrapper::setup_system() {
    // Distribute degrees of freedom
    dof_handler.distribute_dofs(fe);
    
    // Create the sparsity pattern
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    // Initialize matrices and vectors
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    // Assemble the time-independent matrices
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<2>(2), mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<2>(2), stiffness_matrix);
    
    // Build the system matrix
    rebuild_system_matrix();

    // Set up boundary conditions - zero Dirichlet on all boundaries
    VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ZeroFunction<2>(), boundary_values);
}

void DiffusionEnvironmentWrapper::rebuild_system_matrix() {
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * time_step * K, stiffness_matrix);
}

void DiffusionEnvironmentWrapper::compute_mesh_coordinates() {
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
    
    time = 0.0;
    timestep_number = 0;
    
    PythonInitialCondition initial_func(initial_condition);
    VectorTools::interpolate(dof_handler, initial_func, old_solution);
    solution = old_solution;
}

void DiffusionEnvironmentWrapper::step() {
    if (time >= final_time) {
        return;
    }
    
    time += time_step;
    timestep_number++;
    
    assemble_system();
    solve_time_step();
    
    old_solution = solution;
}

void DiffusionEnvironmentWrapper::assemble_system() {
    system_rhs = 0;
    mass_matrix.vmult(system_rhs, old_solution);
    
    Vector<double> tmp(dof_handler.n_dofs());
    stiffness_matrix.vmult(tmp, old_solution);
    system_rhs.add(-time_step * (1 - theta) * K, tmp);
}

void DiffusionEnvironmentWrapper::solve_time_step() {
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
    
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

std::vector<double> DiffusionEnvironmentWrapper::get_solution_data() const {
    std::vector<double> solution_data(solution.size());
    for (size_t i = 0; i < solution.size(); ++i) {
        solution_data[i] = solution[i];
    }
    return solution_data;
}

std::vector<std::array<double, 2>> DiffusionEnvironmentWrapper::get_mesh_points() const {
    return mesh_coordinates;
}

std::vector<double> DiffusionEnvironmentWrapper::get_physical_quantities() const {
    std::vector<double> quantities(5);
    
    // Total energy in the system
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
        auto bounds = get_domain_bounds();
        quantities[3] = (bounds[0] + bounds[1]) / 2.0;  // Center x
        quantities[4] = (bounds[2] + bounds[3]) / 2.0;  // Center y
    }
    
    return quantities;
}

void DiffusionEnvironmentWrapper::write_vtk(const std::string& filename) const {
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "temperature");
    data_out.build_patches();
    
    std::ofstream output(filename);
    data_out.write_vtu(output);
}

void DiffusionEnvironmentWrapper::set_diffusion_coefficient(double K_new) {
    if (K_new <= 0.0) {
        throw std::invalid_argument("Diffusion coefficient must be positive");
    }
    
    K = K_new;
    rebuild_system_matrix();
}

std::string DiffusionEnvironmentWrapper::get_geometry_description() const {
    switch (geometry_config_.type) {
        case GeometryType::HYPER_CUBE:
            return "Hyper cube with side length " + std::to_string(geometry_config_.size);
            
        case GeometryType::HYPER_RECTANGLE:
            return "Rectangle " + std::to_string(geometry_config_.width) + 
                   " x " + std::to_string(geometry_config_.height);
                   
        case GeometryType::HYPER_BALL:
            return "Circle with radius " + std::to_string(geometry_config_.radius) +
                   " centered at (" + std::to_string(geometry_config_.center[0]) + 
                   ", " + std::to_string(geometry_config_.center[1]) + ")";
                   
        case GeometryType::HYPER_SHELL:
            return "Annulus with inner radius " + std::to_string(geometry_config_.inner_radius) +
                   " and outer radius " + std::to_string(geometry_config_.outer_radius) +
                   " centered at (" + std::to_string(geometry_config_.center[0]) + 
                   ", " + std::to_string(geometry_config_.center[1]) + ")";
                   
        case GeometryType::L_SHAPED:
            return "L-shaped domain with size " + std::to_string(geometry_config_.l_size);
            
        case GeometryType::QUARTER_HYPER_BALL:
            return "Quarter circle with radius " + std::to_string(geometry_config_.radius) +
                   " centered at (" + std::to_string(geometry_config_.center[0]) + 
                   ", " + std::to_string(geometry_config_.center[1]) + ")";
                   
        default:
            return "Unknown geometry type";
    }
}

std::vector<double> DiffusionEnvironmentWrapper::get_domain_bounds() const {
    // Find the bounding box of all mesh points
    if (mesh_coordinates.empty()) {
        return {0.0, 1.0, 0.0, 1.0};  // Default fallback
    }
    
    double x_min = mesh_coordinates[0][0];
    double x_max = mesh_coordinates[0][0];
    double y_min = mesh_coordinates[0][1];
    double y_max = mesh_coordinates[0][1];
    
    for (const auto& point : mesh_coordinates) {
        x_min = std::min(x_min, point[0]);
        x_max = std::max(x_max, point[0]);
        y_min = std::min(y_min, point[1]);
        y_max = std::max(y_max, point[1]);
    }
    
    return {x_min, x_max, y_min, y_max};
}

} // namespace TransientDiffusion