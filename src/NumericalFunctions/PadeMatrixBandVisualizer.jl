#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               PadeMatrixBandVisualizer.jl
#                               -------------------------
#
#    @AUTHOR   : Created based on work by Julio Soria
#    date      : 08-06-2025
#    
#    This code creates a specialized visualization that highlights the band structure
#    of the A^(-1) matrix for Pade differentiation.
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
    visualize_band_structure(A_inv::Matrix{Float64}, n::Int, grid_type::String, num_diagonals::Int=25)

Create a visualization that focuses on the band structure of the A^(-1) matrix,
showing how the magnitude decreases as we move away from the main diagonal.

Arguments:
    A_inv::Matrix{Float64} : The A^(-1) matrix
    n::Int : Size of the matrix
    grid_type::String : "Uniform" or "Non-uniform"
    num_diagonals::Int : Number of diagonals to visualize (default: 25)
"""
function visualize_band_structure(A_inv::Matrix{Float64}, n::Int, grid_type::String, num_diagonals::Int=25)
    # Normalize the matrix
    max_val = maximum(abs.(A_inv))
    A_normalized = A_inv ./ max_val
    
    # Calculate the average magnitude of elements for each diagonal
    diag_avgs = zeros(Float64, 2*num_diagonals-1)
    diag_indices = -num_diagonals+1:num_diagonals-1
    
    # Index in diag_avgs: num_diagonals + k (where k is the diagonal offset)
    for k in diag_indices
        diag_vals = []
        
        if k >= 0
            # Upper diagonal
            for i in 1:n-k
                push!(diag_vals, abs(A_normalized[i, i+k]))
            end
        else
            # Lower diagonal
            for i in 1-k:n
                push!(diag_vals, abs(A_normalized[i, i+k]))
            end
        end
        
        idx = num_diagonals + k
        diag_avgs[idx] = mean(diag_vals)
    end
    
    # Create a matrix for the band visualization
    band_matrix = zeros(Float64, n, 2*num_diagonals-1)
    
    # Fill each column with the average value for that diagonal
    for (j, k) in enumerate(diag_indices)
        band_matrix[:, j] .= diag_avgs[j]
    end
    
    # Apply logarithmic scale with a minimum threshold
    min_threshold = 1e-10
    log_band_matrix = log10.(max.(band_matrix, min_threshold))
    
    # Create visualization
    p1 = heatmap(diag_indices, 1:n, log_band_matrix,
                aspect_ratio=:equal,
                c=:inferno,
                clim=(-10, 0),
                title="$grid_type Grid: A^(-1) Band Structure (n=$n)",
                xlabel="Diagonal Offset",
                ylabel="Row Index",
                colorbar_title="log₁₀|Avg. Value|")
    
    # Plot the average magnitude decay
    p2 = plot(diag_indices, log10.(max.(diag_avgs, min_threshold)),
             title="$grid_type Grid: Diagonal Decay in A^(-1) (n=$n)",
             xlabel="Diagonal Offset",
             ylabel="log₁₀|Avg. Value|",
             lw=2,
             marker=:circle,
             markersize=3,
             legend=false)
    
    # Combine plots
    p = plot(p1, p2, layout=(2,1), size=(800, 1000))
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_band_structure_n$(n).png")
    
    return p
end

"""
    visualize_effective_bandwidth(A_inv::Matrix{Float64}, n::Int, grid_type::String, thresholds::Vector{Float64})

Visualize the effective bandwidth of the A^(-1) matrix at different threshold levels.

Arguments:
    A_inv::Matrix{Float64} : The A^(-1) matrix
    n::Int : Size of the matrix
    grid_type::String : "Uniform" or "Non-uniform"
    thresholds::Vector{Float64} : Threshold levels for determining the effective bandwidth
"""
function visualize_effective_bandwidth(A_inv::Matrix{Float64}, n::Int, grid_type::String, 
                                       thresholds::Vector{Float64}=[1e-2, 1e-4, 1e-6, 1e-8])
    # Normalize the matrix
    max_val = maximum(abs.(A_inv))
    A_normalized = abs.(A_inv) ./ max_val
    
    # Create a matrix where each element is the maximum threshold it exceeds
    threshold_matrix = zeros(Float64, n, n)
    
    for i in 1:n
        for j in 1:n
            val = A_normalized[i, j]
            for threshold in thresholds
                if val >= threshold
                    threshold_matrix[i, j] = threshold
                    break
                end
            end
        end
    end
    
    # Map thresholds to colors
    colors = [:white, :blue, :green, :orange, :red]
    color_values = [0.0, thresholds...]
    
    # Create a threshold label matrix for visualization
    threshold_labels = ["< $(thresholds[1])"]
    for i in 1:length(thresholds)-1
        push!(threshold_labels, "$(thresholds[i]) - $(thresholds[i+1])")
    end
    push!(threshold_labels, ">= $(thresholds[end])")
    
    # Create visualization
    p = heatmap(1:n, 1:n, threshold_matrix,
               aspect_ratio=:equal,
               c=cgrad(colors, color_values, scale=:log10),
               title="$grid_type Grid: A^(-1) Effective Bandwidth (n=$n)",
               xlabel="Column Index",
               ylabel="Row Index",
               colorbar_title="Threshold",
               colorbar_ticks=(vcat([0.0], thresholds), threshold_labels))
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_effective_bandwidth_n$(n).png")
    
    return p
end

"""
    main()

Main function to visualize the band structure of A^(-1) matrix.
"""
function main()
    println("=== Pade Matrix Band Structure Visualizer ===")
    
    # Generate and visualize for n=1000
    n = 1000
    println("\nGenerating A matrices for n=$n...")
    
    # Uniform grid
    println("Generating uniform grid A matrix...")
    Δ = 1.0
    A_uniform = generate_A_matrix_uniform(Δ, n)
    
    println("Computing uniform grid A^(-1)...")
    A_inv_uniform = inv(A_uniform)
    
    println("Creating uniform grid band structure visualization...")
    p_band_uniform = visualize_band_structure(A_inv_uniform, n, "Uniform")
    display(p_band_uniform)
    
    println("Creating uniform grid effective bandwidth visualization...")
    p_bandwidth_uniform = visualize_effective_bandwidth(A_inv_uniform, n, "Uniform")
    display(p_bandwidth_uniform)
    
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
    
    println("Creating non-uniform grid band structure visualization...")
    p_band_nonuniform = visualize_band_structure(A_inv_nonuniform, n, "Non-uniform")
    display(p_band_nonuniform)
    
    println("Creating non-uniform grid effective bandwidth visualization...")
    p_bandwidth_nonuniform = visualize_effective_bandwidth(A_inv_nonuniform, n, "Non-uniform")
    display(p_bandwidth_nonuniform)
    
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
