#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               PadeMatrixInverseVisualizer.jl
#                               ---------------------------
#
#    @AUTHOR   : Created based on work by Julio Soria
#    date      : 08-06-2025
#    
#    This code contains simplified functions to:
#        1. Generate A matrices for Pade differentiation and compute their inverses
#        2. Visualize the structure of A^(-1) for both uniform and non-uniform grids
#        3. Save the results as images for analysis
#       
#========================================================================================================#

#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra
using ColorSchemes
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
    visualize_A_inverse(n::Int)

Create and visualize A^(-1) for both uniform and non-uniform grids of size n.
"""
function visualize_A_inverse(n::Int)
    # Uniform grid
    Δ = 1.0
    A_uniform = generate_A_matrix_uniform(Δ, n)
    A_inv_uniform = inv(A_uniform)
    
    # Non-uniform grid with clustering near boundaries
    y = zeros(Float64, n)
    for i in 1:n
        α = 1.5  # Stretching parameter
        y[i] = tanh(α * (2.0 * (i-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
    end
    A_nonuniform = generate_A_matrix_non_uniform(y)
    A_inv_nonuniform = inv(A_nonuniform)
    
    # Normalize for better visualization
    max_val_uniform = maximum(abs.(A_inv_uniform))
    max_val_nonuniform = maximum(abs.(A_inv_nonuniform))
    
    # Normalize matrix values to an absolute maximum of 1
    A_inv_uniform_normalized = A_inv_uniform ./ max_val_uniform
    A_inv_nonuniform_normalized = A_inv_nonuniform ./ max_val_nonuniform
    
    # Create log-scale versions with a minimum threshold to avoid log(0)
    min_threshold = 1e-10
    log_scale_uniform = log10.(max.(abs.(A_inv_uniform_normalized), min_threshold))
    log_scale_nonuniform = log10.(max.(abs.(A_inv_nonuniform_normalized), min_threshold))
    
    # Plot heatmap of uniform A^(-1) with logarithmic absolute values
    p1 = heatmap(1:n, 1:n, log_scale_uniform, 
               aspect_ratio=:equal,
               c=:viridis,
               clim=(-10.0, 0.0),  # log10 scale from 1e-10 to 1
               title="Uniform Grid: log₁₀|A⁻¹| (n=$n)",
               xlabel="Column Index",
               ylabel="Row Index",
               colorbar_title="log₁₀|Normalized Value|")
    
    # Plot heatmap of non-uniform A^(-1) with logarithmic absolute values
    p2 = heatmap(1:n, 1:n, log_scale_nonuniform, 
               aspect_ratio=:equal,
               c=:viridis,
               clim=(-10.0, 0.0),  # log10 scale from 1e-10 to 1
               title="Non-uniform Grid: log₁₀|A⁻¹| (n=$n)",
               xlabel="Column Index",
               ylabel="Row Index",
               colorbar_title="log₁₀|Normalized Value|")
    
    # Plot the middle row of each A^(-1) to show decay
    mid_row = div(n, 2)
    
    # Replace zeros with small values to avoid log scale issues
    uniform_values = abs.(A_inv_uniform[mid_row, :])
    uniform_values[uniform_values .< 1e-15] .= 1e-15
    
    nonuniform_values = abs.(A_inv_nonuniform[mid_row, :])
    nonuniform_values[nonuniform_values .< 1e-15] .= 1e-15
    
    p3 = plot(1:n, uniform_values, 
            label="Uniform Grid", 
            xlabel="Column Index", 
            ylabel="Absolute Value", 
            yscale=:log10,
            lw=2, 
            marker=:circle,
            markersize=3,
            title="A^(-1) Decay Pattern from Middle Row (n=$n)")
    
    plot!(p3, 1:n, nonuniform_values, 
          label="Non-uniform Grid", 
          lw=2, 
          marker=:square,
          markersize=3)
    
    # Combine all plots
    p = plot(p1, p2, p3, layout=(3,1), size=(800, 1200))
    
    # Save the plot
    savefig(p, "A_inverse_analysis_n$(n).png")
    
    # Create zoomed-in views
    p_zoom_uniform = visualize_zoomed_matrix(A_inv_uniform_normalized, n, "Uniform")
    p_zoom_nonuniform = visualize_zoomed_matrix(A_inv_nonuniform_normalized, n, "Non-uniform")
    
    # Create decade-based logarithmic visualizations
    p_decade_uniform = visualize_matrix_decade_log(A_inv_uniform, n, "Uniform")
    p_decade_nonuniform = visualize_matrix_decade_log(A_inv_nonuniform, n, "Non-uniform")
    
    # Print some statistics
    println("=== Grid Size n=$n ===")
    println("Uniform Grid:")
    println("  Maximum absolute value: ", max_val_uniform)
    println("  Condition number: ", cond(A_uniform))
    println("Non-uniform Grid:")
    println("  Maximum absolute value: ", max_val_nonuniform)
    println("  Condition number: ", cond(A_nonuniform))
    
    return p
end

"""
    visualize_zoomed_matrix(A_inv_normalized::Matrix{Float64}, n::Int, grid_type::String, zoom_factor::Int=10)

Create a zoomed-in visualization of the central portion of the A^(-1) matrix.

Arguments:
    A_inv_normalized::Matrix{Float64} : The normalized A^(-1) matrix
    n::Int : Size of the matrix
    grid_type::String : "Uniform" or "Non-uniform"
    zoom_factor::Int : Factor by which to zoom in (default: 10)
"""
function visualize_zoomed_matrix(A_inv_normalized::Matrix{Float64}, n::Int, grid_type::String, zoom_factor::Int=10)
    # Calculate the central region to focus on
    center = div(n, 2)
    half_window = div(n, zoom_factor * 2)
    
    # Define range to visualize
    idx_range = (center - half_window):(center + half_window)
    
    # Create logarithmic version with a minimum threshold to avoid log(0)
    min_threshold = 1e-10
    log_scale = log10.(max.(abs.(A_inv_normalized[idx_range, idx_range]), min_threshold))
    
    # Create zoomed-in plot
    p = heatmap(idx_range, idx_range, log_scale, 
                aspect_ratio=:equal,
                c=:viridis,
                clim=(-10.0, 0.0),  # log10 scale from 1e-10 to 1
                title="$grid_type Grid: Zoomed log₁₀|A⁻¹| (n=$n)",
                xlabel="Column Index",
                ylabel="Row Index",
                colorbar_title="log₁₀|Normalized Value|")
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_zoomed_log_n$(n).png")
    
    return p
end

"""
    analyze_bandwidth_pattern(A_inv::Matrix{Float64}, n::Int, grid_type::String)

Analyze and visualize the bandwidth pattern of A^(-1) by showing how 
the magnitude of elements decreases as we move away from the diagonal.

Arguments:
    A_inv::Matrix{Float64} : The A^(-1) matrix
    n::Int : Size of the matrix
    grid_type::String : "Uniform" or "Non-uniform"

Returns:
    Plot object with the bandwidth pattern visualization
"""
function analyze_bandwidth_pattern(A_inv::Matrix{Float64}, n::Int, grid_type::String)
    # Calculate the maximum absolute value for normalization
    max_val = maximum(abs.(A_inv))
    
    # Calculate the average magnitude of elements at each distance from the diagonal
    max_distance = n - 1
    avg_magnitudes = zeros(Float64, max_distance + 1)
    
    for k in 0:max_distance
        # Get elements at distance k from the diagonal
        diag_values = []
        
        for i in 1:n-k
            push!(diag_values, abs(A_inv[i, i+k]) / max_val)
        end
        
        # If k > 0, also consider elements below the diagonal
        if k > 0
            for i in 1:n-k
                push!(diag_values, abs(A_inv[i+k, i]) / max_val)
            end
        end
        
        # Calculate average magnitude
        avg_magnitudes[k+1] = mean(diag_values)
    end
    
    # Plot the bandwidth pattern
    p = plot(0:max_distance, avg_magnitudes,
             xlabel="Distance from Diagonal",
             ylabel="Average Normalized Magnitude",
             yscale=:log10,
             lw=2,
             marker=:circle,
             markersize=3,
             title="$grid_type Grid: A^(-1) Bandwidth Pattern (n=$n)")
    
    # Save the plot
    savefig(p, "A_inverse_$(grid_type)_bandwidth_n$(n).png")
    
    return p
end

"""
    visualize_matrix_decade_log(A_inv::Matrix{Float64}, n::Int, grid_type::String, decade_min::Int=-10)

Create a visualization of the A^(-1) matrix using logarithmic scale by decades.
This shows the structure more clearly by highlighting the order of magnitude of each element.

Arguments:
    A_inv::Matrix{Float64} : The A^(-1) matrix
    n::Int : Size of the matrix
    grid_type::String : "Uniform" or "Non-uniform"
    decade_min::Int : Minimum decade to display (default: -10, meaning 1e-10)
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
                c=cgrad(:viridis, num_decades),
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
    main()

Main function to execute the analysis.
"""
function main()
    println("=== Pade Matrix Inverse Visualizer ===")
    
    # Define grid sizes to analyze
    grid_sizes = [20, 50, 100]
    
    for n in grid_sizes
        println("\nVisualizing A^(-1) for grid size n=$n...")
        p = visualize_A_inverse(n)
        display(p)
    end
    
    # Add a larger example with n = 1000
    println("\nVisualizing A^(-1) for grid size n=1000...")
    println("This may take a while due to the large matrix size...")
    
    # For n=1000, only compute the non-uniform grid to save time
    # and memory since the uniform grid pattern is quite consistent
    
    # Non-uniform grid with clustering near boundaries
    n = 1000
    y = zeros(Float64, n)
    for i in 1:n
        α = 1.5  # Stretching parameter
        y[i] = tanh(α * (2.0 * (i-1) / (n-1) - 1.0)) / tanh(α) * 0.5 + 0.5
    end
    
    println("Generating A matrix...")
    A_nonuniform = generate_A_matrix_non_uniform(y)
    
    println("Computing A^(-1)...")
    A_inv_nonuniform = inv(A_nonuniform)
    
    max_val_nonuniform = maximum(abs.(A_inv_nonuniform))
    A_inv_nonuniform_normalized = A_inv_nonuniform ./ max_val_nonuniform
    
    # Create log-scale version with a minimum threshold to avoid log(0)
    min_threshold = 1e-10
    log_scale_nonuniform = log10.(max.(abs.(A_inv_nonuniform_normalized), min_threshold))
    
    println("Creating visualization...")
    p = heatmap(1:n, 1:n, log_scale_nonuniform, 
                aspect_ratio=:equal,
                c=:viridis,
                clim=(-10.0, 0.0),  # log10 scale from 1e-10 to 1
                title="Non-uniform Grid: log₁₀|A⁻¹| (n=1000)",
                xlabel="Column Index",
                ylabel="Row Index",
                colorbar_title="log₁₀|Normalized Value|")
    
    display(p)
    savefig(p, "A_inverse_analysis_n1000.png")
    
    # Create a zoomed-in view of the central portion
    p_zoom = visualize_zoomed_matrix(A_inv_nonuniform_normalized, n, "Non-uniform")
    display(p_zoom)
    
    # Create decade-based logarithmic visualization
    p_decade = visualize_matrix_decade_log(A_inv_nonuniform, n, "Non-uniform")
    display(p_decade)
    
    # Analyze the bandwidth pattern
    println("Analyzing bandwidth pattern...")
    p_bandwidth = analyze_bandwidth_pattern(A_inv_nonuniform, n, "Non-uniform")
    display(p_bandwidth)
    
    # Calculate effective bandwidth at different thresholds
    thresholds = [1e-2, 1e-4, 1e-6, 1e-8]
    println("\nEffective Bandwidth (Non-uniform grid, n=1000):")
    
    for threshold in thresholds
        # Normalize matrix values
        A_inv_norm = abs.(A_inv_nonuniform) ./ max_val_nonuniform
        
        # Find the maximum distance from diagonal with elements above threshold
        max_distance = 0
        for k in 0:n-1
            diag_vals = [A_inv_norm[i, i+k] for i in 1:n-k]
            if any(diag_vals .> threshold)
                max_distance = max(max_distance, k)
            end
            
            if k > 0  # Check the other side of the diagonal
                diag_vals = [A_inv_norm[i+k, i] for i in 1:n-k]
                if any(diag_vals .> threshold)
                    max_distance = max(max_distance, k)
                end
            end
        end
        
        println("  Threshold = $threshold: $max_distance")
    end
    
    println("Maximum absolute value for n=1000: ", max_val_nonuniform)
    println("Condition number for n=1000: ", cond(A_nonuniform))
    
    println("\nAll visualizations have been saved as PNG files.")
end

# Execute the main function
main()
