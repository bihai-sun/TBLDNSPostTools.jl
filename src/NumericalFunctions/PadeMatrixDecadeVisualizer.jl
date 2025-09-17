#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               PadeMatrixDecadeVisualizer.jl
#                               --------------------------
#
#    @AUTHOR   : Created based on work by Julio Soria
#    date      : 08-06-2025
#    
#    This code contains simplified functions to visualize the A^(-1) matrix structure
#    for Pade differentiation using a decade-based logarithmic scale.
#       
#========================================================================================================#

#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra
include("PadeDifferentiation.jl")

"""
    generate_A_matrix_uniform(Δ::T, nx::Int) where T<:Real

Generate the A matrix for Pade differentiation on a uniform grid.
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
    visualize_matrix_decade_log(A_inv::Matrix{Float64}, n::Int, grid_type::String, decade_min::Int=-10)

Create a visualization of the A^(-1) matrix using logarithmic scale by decades.
This shows the structure more clearly by highlighting the order of magnitude of each element.
"""
function visualize_matrix_decade_log(A_inv::Matrix{Float64}, n::Int, grid_type::String, decade_min::Int=-10)
    # Normalize the matrix
    max_val = maximum(abs.(A_inv))
    A_normalized = A_inv ./ max_val
    
    # Create a matrix to hold decade values
    decade_matrix = zeros(Int, n, n)
    
    # Assign decade values (0 for 1.0 to 0.1, -1 for 0.1 to 0.01, etc.)
    for i in 1:n
        for j in 1:n
            val = abs(A_normalized[i, j])
            if val == 0
                decade_matrix[i, j] = decade_min - 1  # Special value for zero
            else
                decade = floor(Int, log10(val))
                decade_matrix[i, j] = max(decade, decade_min)  # Cap at minimum decade
            end
        end
    end
    
    # Create a custom colormap with distinct colors for each decade
    num_decades = abs(decade_min) + 1
    
    # Create the heatmap with custom color labels
    p = heatmap(1:n, 1:n, decade_matrix, 
                aspect_ratio=:equal,
                c=cgrad(:inferno, num_decades),
                clim=(decade_min, 0),
                title="$grid_type Grid: A^(-1) Decade Structure (n=$n)",
                xlabel="Column Index",
                ylabel="Row Index",
                colorbar_title="log₁₀ Decade",
                colorbar_ticks=(decade_min:0, ["10^$d" for d in decade_min:0]))
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_decade_log_n$(n).png")
    
    return p
end

"""
    visualize_zoomed_matrix_decade(A_inv::Matrix{Float64}, n::Int, grid_type::String, zoom_factor::Int=10, decade_min::Int=-10)

Create a zoomed-in visualization of the central portion of the A^(-1) matrix using decade-based log scale.
"""
function visualize_zoomed_matrix_decade(A_inv::Matrix{Float64}, n::Int, grid_type::String, zoom_factor::Int=10, decade_min::Int=-10)
    # Calculate the central region to focus on
    center = div(n, 2)
    half_window = div(n, zoom_factor * 2)
    
    # Define range to visualize
    idx_range = (center - half_window):(center + half_window)
    
    # Normalize the matrix
    max_val = maximum(abs.(A_inv))
    A_normalized = A_inv ./ max_val
    
    # Create a matrix to hold decade values for the zoomed region
    sub_matrix = A_normalized[idx_range, idx_range]
    decade_matrix = zeros(Int, length(idx_range), length(idx_range))
    
    # Assign decade values
    for i in 1:length(idx_range)
        for j in 1:length(idx_range)
            val = abs(sub_matrix[i, j])
            if val == 0
                decade_matrix[i, j] = decade_min - 1  # Special value for zero
            else
                decade = floor(Int, log10(val))
                decade_matrix[i, j] = max(decade, decade_min)  # Cap at minimum decade
            end
        end
    end
    
    # Create a custom colormap with distinct colors for each decade
    num_decades = abs(decade_min) + 1
    
    # Create zoomed-in plot
    p = heatmap(idx_range, idx_range, decade_matrix, 
                aspect_ratio=:equal,
                c=cgrad(:inferno, num_decades),
                clim=(decade_min, 0),
                title="$grid_type Grid: Zoomed A^(-1) Decade Structure (n=$n)",
                xlabel="Column Index",
                ylabel="Row Index",
                colorbar_title="log₁₀ Decade",
                colorbar_ticks=(decade_min:0, ["10^$d" for d in decade_min:0]))
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_zoomed_decade_log_n$(n).png")
    
    return p
end

"""
    main()

Main function to visualize A^(-1) matrix structure with decade-based log scale.
"""
function main()
    println("=== Pade Matrix Decade Visualizer ===")
    
    # Generate and visualize for n=1000
    n = 1000
    println("\nGenerating A matrices for n=$n...")
    
    # Uniform grid
    println("Generating uniform grid A matrix...")
    Δ = 1.0
    A_uniform = generate_A_matrix_uniform(Δ, n)
    
    println("Computing uniform grid A^(-1)...")
    A_inv_uniform = inv(A_uniform)
    
    println("Creating uniform grid visualization...")
    p_uniform = visualize_matrix_decade_log(A_inv_uniform, n, "Uniform")
    display(p_uniform)
    
    println("Creating zoomed uniform grid visualization...")
    p_zoom_uniform = visualize_zoomed_matrix_decade(A_inv_uniform, n, "Uniform")
    display(p_zoom_uniform)
    
    # Non-uniform grid
    println("\nGenerating non-uniform grid A matrix...")
    y = zeros(Float64, n)
    for i in 1:n
        α = 1.5  # Stretching parameter
        y[i] = tanh(α * (2.0 * (i-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
    end
    A_nonuniform = generate_A_matrix_non_uniform(y)
    
    println("Computing non-uniform grid A^(-1)...")
    A_inv_nonuniform = inv(A_nonuniform)
    
    println("Creating non-uniform grid visualization...")
    p_nonuniform = visualize_matrix_decade_log(A_inv_nonuniform, n, "Non-uniform")
    display(p_nonuniform)
    
    println("Creating zoomed non-uniform grid visualization...")
    p_zoom_nonuniform = visualize_zoomed_matrix_decade(A_inv_nonuniform, n, "Non-uniform")
    display(p_zoom_nonuniform)
    
    # Print statistics
    println("\n=== Matrix Statistics (n=$n) ===")
    println("Uniform Grid:")
    println("  Maximum absolute value: ", maximum(abs.(A_inv_uniform)))
    println("  Condition number: ", cond(A_uniform))
    println("Non-uniform Grid:")
    println("  Maximum absolute value: ", maximum(abs.(A_inv_nonuniform)))
    println("  Condition number: ", cond(A_nonuniform))
    
    println("\nAll visualizations have been saved as PNG files.")
end

# Execute the main function
main()
