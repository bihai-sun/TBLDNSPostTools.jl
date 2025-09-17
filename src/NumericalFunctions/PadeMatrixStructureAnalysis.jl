#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               PadeMatrixStructureAnalysis.jl
#                               ---------------------------
#
#    @AUTHOR   : Created based on work by Julio Soria
#    date      : 08-06-2025
#    
#    This code contains functions to:
#        1. Generate A matrices for Pade differentiation with equal and unequal spacings
#        2. Compute and analyze the inverse of A matrices
#        3. Visualize the structure of A^(-1) using heat maps
#        4. Examine how the structure changes with the number of grid points
#       
#========================================================================================================#

#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra
using Printf
include("PadeDifferentiation.jl")  # Include the main Pade differentiation code

"""
    generate_A_matrix_uniform(Δ::T, nx::Int) where T<:Real

Generate the A matrix for Pade differentiation on a uniform grid.
This is the matrix on the LHS of the equation A f' = B f.

Arguments:
    Δ::T    : constant step size, i.e., spacing between function values
    nx::Int : number of function values, size of array f[1:nx]

Returns:
    A::Matrix{T} : The tridiagonal A matrix for the uniform Pade scheme
"""
function generate_A_matrix_uniform(Δ::T, nx::Int) where T<:Real
    # Get the Pade coefficients
    coef = compute_weights_6order_pade_uniform(Δ, nx)
    
    # Initialize the A matrix
    A = zeros(T, nx, nx)
    
    # Fill the main diagonal
    for i in 1:nx
        A[i, i] = 1.0
    end
    
    # Fill the subdiagonal (alpha)
    for i in 2:nx
        A[i, i-1] = coef[6, i]
    end
    
    # Fill the superdiagonal (beta)
    for i in 1:nx-1
        A[i, i+1] = coef[8, i]
    end
    
    return A
end

"""
    generate_A_matrix_non_uniform(y::AbstractArray{T,1}) where T<:Real

Generate the A matrix for Pade differentiation on a non-uniform grid.
This is the matrix on the LHS of the equation A f' = B f.

Arguments:
    y::AbstractArray{T,1} : 1D array of size ny containing the non-uniform grid points

Returns:
    A::Matrix{T} : The tridiagonal A matrix for the non-uniform Pade scheme
"""
function generate_A_matrix_non_uniform(y::AbstractArray{T,1}) where T<:Real
    ny = length(y)
    
    # Get the Pade coefficients
    coef = compute_weights_6order_pade_non_uniform(y)
    
    # Initialize the A matrix
    A = zeros(T, ny, ny)
    
    # Fill the main diagonal
    for i in 1:ny
        A[i, i] = 1.0
    end
    
    # Fill the subdiagonal (alpha)
    for i in 2:ny
        A[i, i-1] = coef[6, i]
    end
    
    # Fill the superdiagonal (beta)
    for i in 1:ny-1
        A[i, i+1] = coef[8, i]
    end
    
    return A
end

"""
    generate_B_matrix_uniform(Δ::T, nx::Int) where T<:Real

Generate the B matrix for Pade differentiation on a uniform grid.
This is the matrix on the RHS of the equation A f' = B f.

Arguments:
    Δ::T    : constant step size, i.e., spacing between function values
    nx::Int : number of function values, size of array f[1:nx]

Returns:
    B::Matrix{T} : The pentadiagonal B matrix for the uniform Pade scheme
"""
function generate_B_matrix_uniform(Δ::T, nx::Int) where T<:Real
    # Get the Pade coefficients
    coef = compute_weights_6order_pade_uniform(Δ, nx)
    
    # Initialize the B matrix
    B = zeros(T, nx, nx)
    
    # Fill the matrix with the 5-band structure
    for i in 1:nx
        # AA (far lower diagonal)
        if i > 2
            B[i, i-2] = coef[1, i]
        end
        
        # BB (lower diagonal)
        if i > 1
            B[i, i-1] = coef[2, i]
        end
        
        # CC (main diagonal)
        B[i, i] = coef[3, i]
        
        # DD (upper diagonal)
        if i < nx
            B[i, i+1] = coef[4, i]
        end
        
        # EE (far upper diagonal)
        if i < nx-1
            B[i, i+2] = coef[5, i]
        end
    end
    
    return B
end

"""
    generate_B_matrix_non_uniform(y::AbstractArray{T,1}) where T<:Real

Generate the B matrix for Pade differentiation on a non-uniform grid.
This is the matrix on the RHS of the equation A f' = B f.

Arguments:
    y::AbstractArray{T,1} : 1D array of size ny containing the non-uniform grid points

Returns:
    B::Matrix{T} : The pentadiagonal B matrix for the non-uniform Pade scheme
"""
function generate_B_matrix_non_uniform(y::AbstractArray{T,1}) where T<:Real
    ny = length(y)
    
    # Get the Pade coefficients
    coef = compute_weights_6order_pade_non_uniform(y)
    
    # Initialize the B matrix
    B = zeros(T, ny, ny)
    
    # Fill the matrix with the 5-band structure
    for i in 1:ny
        # AA (far lower diagonal)
        if i > 2
            B[i, i-2] = coef[1, i]
        end
        
        # BB (lower diagonal)
        if i > 1
            B[i, i-1] = coef[2, i]
        end
        
        # CC (main diagonal)
        B[i, i] = coef[3, i]
        
        # DD (upper diagonal)
        if i < ny
            B[i, i+1] = coef[4, i]
        end
        
        # EE (far upper diagonal)
        if i < ny-1
            B[i, i+2] = coef[5, i]
        end
    end
    
    return B
end

"""
    analyze_A_inverse_structure(grid_sizes::AbstractArray{Int,1}, uniform::Bool=true)

Analyze and visualize the structure of A^(-1) for different grid sizes.

Arguments:
    grid_sizes::AbstractArray{Int,1} : Array of grid sizes to analyze
    uniform::Bool : Whether to use uniform (true) or non-uniform (false) grid spacing

Returns:
    Nothing, but displays plots of the A^(-1) matrices
"""
function analyze_A_inverse_structure(grid_sizes::AbstractArray{Int,1}, uniform::Bool=true)
    for n in grid_sizes
        # Create appropriate grid
        if uniform
            Δ = 1.0  # Constant spacing
            A = generate_A_matrix_uniform(Δ, n)
            grid_type = "Uniform"
        else
            # Create a non-uniform grid with clustering near the boundaries
            y = zeros(Float64, n)
            # Use a hyperbolic tangent stretching
            for i in 1:n
                α = 1.5  # Stretching parameter
                y[i] = tanh(α * (2.0 * (i-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
            end
            A = generate_A_matrix_non_uniform(y)
            grid_type = "Non-uniform"
        end
        
        # Compute inverse
        A_inv = inv(A)
        
        # Normalize for better visualization
        max_val = maximum(abs.(A_inv))
        
        # Plot heatmap of A^(-1)
        p = heatmap(1:n, 1:n, A_inv, 
                    aspect_ratio=:equal,
                    c=:viridis,
                    clim=(-max_val, max_val),
                    title="$grid_type Grid: A^(-1) Structure (n=$n)",
                    xlabel="Column Index",
                    ylabel="Row Index",
                    colorbar_title="Value")
        
        display(p)
        
        # Save the plot if needed
        savefig(p, "A_inverse_$(grid_type)_n$(n).png")
        
        # Print some statistics
        println("\n--- $grid_type Grid: A^(-1) Statistics (n=$n) ---")
        println("Maximum absolute value: ", max_val)
        println("Condition number: ", cond(A))
        println("Sparsity (fraction of near-zero elements): ", 
                sum(abs.(A_inv) .< 1e-10) / (n*n))
    end
end

"""
    compare_A_inverse_decay(grid_sizes::AbstractArray{Int,1})

Compare how quickly the elements of A^(-1) decay away from the diagonal
for both uniform and non-uniform grids.

Arguments:
    grid_sizes::AbstractArray{Int,1} : Array of grid sizes to analyze

Returns:
    Nothing, but displays plots comparing the decay patterns
"""
function compare_A_inverse_decay(grid_sizes::AbstractArray{Int,1})
    for n in grid_sizes
        # Create uniform grid
        Δ = 1.0  # Constant spacing
        A_uniform = generate_A_matrix_uniform(Δ, n)
        A_inv_uniform = inv(A_uniform)
        
        # Create non-uniform grid with clustering near the boundaries
        y = zeros(Float64, n)
        for i in 1:n
            α = 1.5  # Stretching parameter
            y[i] = tanh(α * (2.0 * (i-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
        end
        A_nonuniform = generate_A_matrix_non_uniform(y)
        A_inv_nonuniform = inv(A_nonuniform)
        
        # Extract diagonal and off-diagonal decay patterns
        # For simplicity, let's look at the middle row
        mid_row = div(n, 2)
        
        # Plot the decay pattern
        # Replace zeros with small values to avoid log scale issues
        uniform_values = abs.(A_inv_uniform[mid_row, :])
        uniform_values[uniform_values .< 1e-15] .= 1e-15
        
        nonuniform_values = abs.(A_inv_nonuniform[mid_row, :])
        nonuniform_values[nonuniform_values .< 1e-15] .= 1e-15
        
        p = plot(1:n, uniform_values, 
                label="Uniform Grid", 
                xlabel="Column Index", 
                ylabel="Absolute Value", 
                yscale=:log10,
                lw=2, 
                marker=:circle,
                markersize=4,
                title="A^(-1) Decay Pattern from Middle Row (n=$n)")
        
        plot!(p, 1:n, nonuniform_values, 
              label="Non-uniform Grid", 
              lw=2, 
              marker=:square,
              markersize=4)
        
        display(p)
        
        # Save the plot if needed
        savefig(p, "A_inverse_decay_comparison_n$(n).png")
    end
end

"""
    analyze_condition_number_trend(grid_sizes::AbstractArray{Int,1})

Analyze how the condition number of A changes with grid size
for both uniform and non-uniform grids.

Arguments:
    grid_sizes::AbstractArray{Int,1} : Array of grid sizes to analyze

Returns:
    Nothing, but displays a plot of condition numbers vs grid size
"""
function analyze_condition_number_trend(grid_sizes::AbstractArray{Int,1})
    cond_uniform = zeros(Float64, length(grid_sizes))
    cond_nonuniform = zeros(Float64, length(grid_sizes))
    
    for (i, n) in enumerate(grid_sizes)
        # Uniform grid
        Δ = 1.0
        A_uniform = generate_A_matrix_uniform(Δ, n)
        cond_uniform[i] = cond(A_uniform)
        
        # Non-uniform grid
        y = zeros(Float64, n)
        for j in 1:n
            α = 1.5  # Stretching parameter
            y[j] = tanh(α * (2.0 * (j-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
        end
        A_nonuniform = generate_A_matrix_non_uniform(y)
        cond_nonuniform[i] = cond(A_nonuniform)
    end
    
    p = plot(grid_sizes, cond_uniform, 
            label="Uniform Grid", 
            xlabel="Grid Size", 
            ylabel="Condition Number", 
            yscale=:log10,
            lw=2, 
            marker=:circle,
            title="Condition Number vs Grid Size")
    
    plot!(p, grid_sizes, cond_nonuniform, 
          label="Non-uniform Grid", 
          lw=2, 
          marker=:square)
    
    display(p)
    
    # Save the plot if needed
    savefig(p, "condition_number_trend.png")
end

"""
    analyze_bandwidth_structure(grid_sizes::AbstractArray{Int,1})

Analyze the effective bandwidth of A^(-1) for different grid sizes.
This function identifies how quickly the matrix elements decrease away from the diagonal.

Arguments:
    grid_sizes::AbstractArray{Int,1} : Array of grid sizes to analyze

Returns:
    Nothing, but displays results about the bandwidth structure
"""
function analyze_bandwidth_structure(grid_sizes::AbstractArray{Int,1})
    println("\n=== Effective Bandwidth Analysis of A^(-1) ===")
    
    thresholds = [1e-2, 1e-4, 1e-6, 1e-8]
    
    # Create a table header
    println("Grid Size | Grid Type | ", join(["Bandwidth (threshold = $t)" for t in thresholds], " | "))
    println("----------|-----------|", join(["---------------------------" for _ in thresholds], "|"))
    
    for n in grid_sizes
        # Uniform grid
        Δ = 1.0
        A_uniform = generate_A_matrix_uniform(Δ, n)
        A_inv_uniform = inv(A_uniform)
        
        # Non-uniform grid
        y = zeros(Float64, n)
        for j in 1:n
            α = 1.5  # Stretching parameter
            y[j] = tanh(α * (2.0 * (j-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
        end
        A_nonuniform = generate_A_matrix_non_uniform(y)
        A_inv_nonuniform = inv(A_nonuniform)
        
        # Calculate effective bandwidth for each threshold
        uniform_bandwidths = []
        nonuniform_bandwidths = []
        
        for threshold in thresholds
            # For uniform grid
            max_val_uniform = maximum(abs.(A_inv_uniform))
            normalized_uniform = abs.(A_inv_uniform) ./ max_val_uniform
            bandwidth_uniform = 0
            
            # Check each diagonal
            for k in 0:n-1
                diag_vals = [normalized_uniform[i, i+k] for i in 1:n-k]
                if any(diag_vals .> threshold)
                    bandwidth_uniform = max(bandwidth_uniform, k)
                end
                
                if k > 0  # Check the other side of the diagonal
                    diag_vals = [normalized_uniform[i+k, i] for i in 1:n-k]
                    if any(diag_vals .> threshold)
                        bandwidth_uniform = max(bandwidth_uniform, k)
                    end
                end
            end
            push!(uniform_bandwidths, bandwidth_uniform)
            
            # For non-uniform grid
            max_val_nonuniform = maximum(abs.(A_inv_nonuniform))
            normalized_nonuniform = abs.(A_inv_nonuniform) ./ max_val_nonuniform
            bandwidth_nonuniform = 0
            
            # Check each diagonal
            for k in 0:n-1
                diag_vals = [normalized_nonuniform[i, i+k] for i in 1:n-k]
                if any(diag_vals .> threshold)
                    bandwidth_nonuniform = max(bandwidth_nonuniform, k)
                end
                
                if k > 0  # Check the other side of the diagonal
                    diag_vals = [normalized_nonuniform[i+k, i] for i in 1:n-k]
                    if any(diag_vals .> threshold)
                        bandwidth_nonuniform = max(bandwidth_nonuniform, k)
                    end
                end
            end
            push!(nonuniform_bandwidths, bandwidth_nonuniform)
        end
        
        # Print results
        println(@sprintf("%9d | %-9s | %s", n, "Uniform", join([@sprintf("%9d", bw) for bw in uniform_bandwidths], " | ")))
        println(@sprintf("%9d | %-9s | %s", n, "Non-uniform", join([@sprintf("%9d", bw) for bw in nonuniform_bandwidths], " | ")))
    end
end

"""
    visualize_diagonal_dominance(grid_sizes::AbstractArray{Int,1})

Visualize how diagonal dominance changes along the matrix for both uniform and non-uniform grids.

Arguments:
    grid_sizes::AbstractArray{Int,1} : Array of grid sizes to analyze

Returns:
    Nothing, but displays plots showing the diagonal dominance
"""
function visualize_diagonal_dominance(grid_sizes::AbstractArray{Int,1})
    for n in grid_sizes
        # Uniform grid
        Δ = 1.0
        A_uniform = generate_A_matrix_uniform(Δ, n)
        A_inv_uniform = inv(A_uniform)
        
        # Non-uniform grid
        y = zeros(Float64, n)
        for j in 1:n
            α = 1.5  # Stretching parameter
            y[j] = tanh(α * (2.0 * (j-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
        end
        A_nonuniform = generate_A_matrix_non_uniform(y)
        A_inv_nonuniform = inv(A_nonuniform)
        
        # Calculate the diagonal dominance for each row
        # (ratio of diagonal element to sum of absolute values of off-diagonal elements)
        diag_dominance_uniform = zeros(Float64, n)
        diag_dominance_nonuniform = zeros(Float64, n)
        
        for i in 1:n
            diag_val_uniform = abs(A_inv_uniform[i, i])
            offdiag_sum_uniform = sum(abs.(A_inv_uniform[i, :]) .- abs.([j == i ? A_inv_uniform[i, j] : 0.0 for j in 1:n]))
            diag_dominance_uniform[i] = diag_val_uniform / (offdiag_sum_uniform + 1e-15)
            
            diag_val_nonuniform = abs(A_inv_nonuniform[i, i])
            offdiag_sum_nonuniform = sum(abs.(A_inv_nonuniform[i, :]) .- abs.([j == i ? A_inv_nonuniform[i, j] : 0.0 for j in 1:n]))
            diag_dominance_nonuniform[i] = diag_val_nonuniform / (offdiag_sum_nonuniform + 1e-15)
        end
        
        # Normalize row position to [0,1] for comparison across different sizes
        row_positions = range(0, 1, length=n)
        
        # Plot the diagonal dominance
        p = plot(row_positions, diag_dominance_uniform, 
                label="Uniform Grid", 
                xlabel="Normalized Row Position", 
                ylabel="Diagonal Dominance Ratio", 
                lw=2, 
                marker=:circle,
                markersize=4,
                title="Diagonal Dominance in A^(-1) (n=$n)")
        
        plot!(p, row_positions, diag_dominance_nonuniform, 
              label="Non-uniform Grid", 
              lw=2, 
              marker=:square,
              markersize=4)
        
        display(p)
        
        # Save the plot
        savefig(p, "diagonal_dominance_n$(n).png")
    end
end

"""
    main()

Main function to execute the analysis.
"""
function main()
    println("=== Pade Matrix Structure Analysis ===")
    
    # Define grid sizes to analyze
    small_grid_sizes = [10, 20, 30]
    large_grid_sizes = [50, 100, 200]
    
    # Analyze A^(-1) structure for uniform grid
    println("\nAnalyzing A^(-1) structure for uniform grid...")
    analyze_A_inverse_structure(small_grid_sizes, true)
    
    # Analyze A^(-1) structure for non-uniform grid
    println("\nAnalyzing A^(-1) structure for non-uniform grid...")
    analyze_A_inverse_structure(small_grid_sizes, false)
    
    # Compare decay patterns
    println("\nComparing A^(-1) decay patterns...")
    compare_A_inverse_decay(small_grid_sizes)
    
    # Analyze bandwidth structure
    println("\nAnalyzing bandwidth structure...")
    analyze_bandwidth_structure(small_grid_sizes)
    
    # Visualize diagonal dominance
    println("\nVisualizing diagonal dominance...")
    visualize_diagonal_dominance(small_grid_sizes)
    
    # Analyze condition number trend
    println("\nAnalyzing condition number trend...")
    analyze_condition_number_trend(vcat(small_grid_sizes, large_grid_sizes))
    
    # Demonstration of direct computation
    println("\nDemonstrating direct computation of A and A^(-1)...")
    
    # For a specific grid size
    n = 20
    Δ = 1.0
    
    # Uniform grid
    A_uniform = generate_A_matrix_uniform(Δ, n)
    B_uniform = generate_B_matrix_uniform(Δ, n)
    A_inv_uniform = inv(A_uniform)
    
    # Create a test function for demonstration
    x_uniform = range(0, 1, length=n)
    f_uniform = sin.(2π * x_uniform)
    
    # Compute derivative directly using matrix operations
    df_direct = A_inv_uniform * (B_uniform * f_uniform)
    
    # Compute derivative using the Pade differentiation function
    df_pade = compute_PadeFD(f_uniform, Δ)
    
    # Compare results
    max_diff = maximum(abs.(df_direct - df_pade))
    println("Maximum difference between direct matrix computation and PadeFD: $max_diff")
    
    # Plot the derivatives for comparison
    p = plot(x_uniform, df_direct, 
             label="Direct Matrix (A^(-1)B)", 
             xlabel="x", 
             ylabel="df/dx", 
             lw=2,
             title="Comparison of Derivative Computation Methods")
    
    plot!(p, x_uniform, df_pade, 
          label="PadeFD Function", 
          lw=2, 
          ls=:dash)
    
    plot!(p, x_uniform, 2π * cos.(2π * x_uniform), 
          label="Analytical", 
          lw=2, 
          ls=:dot)
    
    display(p)
    savefig(p, "derivative_comparison.png")
end

# Execute the main function
main()
