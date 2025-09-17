#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               test_CDFD_optimized.jl
#                               ---------------------
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 09-05-2025
#    
#    This code contains functions to test the optimized central difference differentiation:
#        1. Regular vs optimized implementation benchmark
#        2. Usage example with non-uniform grid
#       
#========================================================================================================#

using BenchmarkTools
using LinearAlgebra

# Include the FD-Differentiation.jl file
include("FD-Differentiation.jl")

#========================================================================================================#
# Test Function
#========================================================================================================#

function benchmark_CDFD_implementations(n_points=32, n_repeats=100)
    # Create a test function and grid
    x_uniform    = range(0, 2π, length=n_points)
    x_nonuniform = [0.0; sort(rand(n_points-2) * 2π); 2π]
    
    # Create a test function and its analytical derivative
    f                   = sin.(x_uniform)
    f_nonuniform        = sin.(x_nonuniform)
    df_exact            = cos.(x_uniform)
    df_exact_nonuniform = cos.(x_nonuniform)
    
    # Standard implementation
    time_standard = @belapsed compute_CDFD($f_nonuniform, $x_nonuniform) samples=n_repeats
    
    # Optimized implementation
    coeffs              = precompute_grid_coefficients(x_nonuniform)
    time_precompute     = @belapsed precompute_grid_coefficients($x_nonuniform) samples=n_repeats
    time_optimized      = @belapsed compute_CDFD_optimized($f_nonuniform, $x_nonuniform, $coeffs) samples=n_repeats
    time_optimized_safe = @belapsed compute_CDFD_optimized_safe($f_nonuniform, $x_nonuniform, $coeffs) samples=n_repeats
    
    # Calculate errors
    df_standard       = compute_CDFD(f_nonuniform, x_nonuniform)
    df_optimized      = compute_CDFD_optimized(f_nonuniform, x_nonuniform, coeffs)
    df_optimized_safe = compute_CDFD_optimized_safe(f_nonuniform, x_nonuniform, coeffs)
    
    # Calculate relative errors
    error_standard       = norm(df_standard - df_exact_nonuniform) / norm(df_exact_nonuniform)
    error_optimized      = norm(df_optimized - df_exact_nonuniform) / norm(df_exact_nonuniform)
    error_optimized_safe = norm(df_optimized_safe - df_exact_nonuniform) / norm(df_exact_nonuniform)
    
    # Calculate differences between implementations
    diff_standard_optimized = norm(df_standard - df_optimized) / norm(df_standard)
    diff_standard_safe      = norm(df_standard - df_optimized_safe) / norm(df_standard)
    
    # Print results
    println("============ Performance Comparison ============")
    println("Grid size:                              $n_points points")
    println("Standard implementation time:           $(time_standard * 1000) ms")
    println("Optimized: precompute time:             $(time_precompute * 1000) ms")
    println("Optimized: evaluation time:             $(time_optimized * 1000) ms")
    println("Optimized safe: evaluation time:        $(time_optimized_safe * 1000) ms")
    println("Total optimized time (first call):      $((time_precompute + time_optimized) * 1000) ms")
    println("Total optimized safe time (first call): $((time_precompute + time_optimized_safe) * 1000) ms")
    println("Speedup (regular optimized):            $(time_standard / time_optimized)x")
    println("Speedup (safe optimized):               $(time_standard / time_optimized_safe)x")
    println()
    println("============ Accuracy Comparison ============")
    println("Standard implementation error:                  $error_standard")
    println("Optimized implementation error:                 $error_optimized")
    println("Optimized safe implementation error:            $error_optimized_safe")
    println()
    println("Difference between standard and optimized:      $diff_standard_optimized")
    println("Difference between standard and optimized safe: $diff_standard_safe")
    
    # Plot the results
    p = plot(x_nonuniform, df_exact_nonuniform, label="Exact", 
         linewidth=3, title="Derivative Comparison (Non-uniform Grid)")
    plot!(p, x_nonuniform, df_standard, label="Standard", 
          linestyle=:dash)
    plot!(p, x_nonuniform, df_optimized_safe, label="Optimized Safe", 
          linestyle=:dot)
    display(p)
    savefig(p, "cdfd_comparison.png")
    
    # Plot errors
    err_standard = abs.(df_standard - df_exact_nonuniform)
    err_optimized = abs.(df_optimized - df_exact_nonuniform)
    err_optimized_safe = abs.(df_optimized_safe - df_exact_nonuniform)
    
    p_err = plot(x_nonuniform, err_standard, label="Standard Error", 
         linewidth=2, title="Error Comparison (log scale)", yscale=:log10)
    plot!(p_err, x_nonuniform, err_optimized, label="Optimized Error", 
          linestyle=:dash)
    plot!(p_err, x_nonuniform, err_optimized_safe, label="Optimized Safe Error", 
          linestyle=:dot)
    display(p_err)
    savefig(p_err, "cdfd_error_comparison.png")
    
    return time_standard, time_optimized, time_optimized_safe, error_standard, error_optimized, error_optimized_safe
end

function test_sine_function(n_points=32)
    """
    Test the derivative calculation on sin(x) function across a uniform and non-uniform grid.
    This is a well-controlled test case with known exact derivative.
    """
    # Create uniform and non-uniform grids
    x_uniform    = range(0, 2π, length=n_points)
    x_nonuniform = sort([0; rand(n_points-2) * 2π; 2π])
    
    # Create test function sin(x) and its analytical derivative cos(x)
    f_uniform           = sin.(x_uniform)
    f_nonuniform        = sin.(x_nonuniform)
    df_exact_uniform    = cos.(x_uniform)
    df_exact_nonuniform = cos.(x_nonuniform)
    
    # Calculate derivatives with all methods
    df_uniform_standard     = compute_CDFD(f_uniform, collect(x_uniform))
    
    coeffs_nonuniform       = precompute_grid_coefficients(x_nonuniform)
    df_nonuniform_standard  = compute_CDFD(f_nonuniform, x_nonuniform)
    df_nonuniform_optimized = compute_CDFD_optimized(f_nonuniform, x_nonuniform, coeffs_nonuniform)
    df_nonuniform_safe      = compute_CDFD_optimized_safe(f_nonuniform, x_nonuniform, coeffs_nonuniform)
    
    # Calculate errors
    error_uniform    = norm(df_uniform_standard - df_exact_uniform) / norm(df_exact_uniform)
    error_nonuniform = norm(df_nonuniform_standard - df_exact_nonuniform) / norm(df_exact_nonuniform)
    error_optimized  = norm(df_nonuniform_optimized - df_exact_nonuniform) / norm(df_exact_nonuniform)
    error_safe       = norm(df_nonuniform_safe - df_exact_nonuniform) / norm(df_exact_nonuniform)
    
    # Print results
    println("============ Sine Function Test ============")
    println("Grid size:                             $n_points points")
    println("Uniform grid error:                    $error_uniform")
    println("Non-uniform grid standard error:       $error_nonuniform")
    println("Non-uniform grid optimized error:      $error_optimized")
    println("Non-uniform grid optimized safe error: $error_safe")
    
    # Create plot of the function and its derivatives
    p1 = plot(x_uniform, f_uniform, label="sin(x)", linewidth=2, title="Function and Derivatives")
    plot!(p1, x_uniform, df_exact_uniform, label="cos(x) (exact)", linewidth=2, linestyle=:dash)
    plot!(p1, x_uniform, df_uniform_standard, label="Numerical (uniform)", linewidth=2, linestyle=:dot)
    display(p1)
    savefig(p1, "sine_function_test.png")
    
    # Plot errors
    max_err = max(
        maximum(abs.(df_uniform_standard - df_exact_uniform)),
        maximum(abs.(df_nonuniform_standard - df_exact_nonuniform)),
        maximum(abs.(df_nonuniform_optimized - df_exact_nonuniform)),
        maximum(abs.(df_nonuniform_safe - df_exact_nonuniform))
    )
    
    println("Maximum absolute error across all methods: $max_err")
    
    return error_uniform, error_nonuniform, error_optimized, error_safe
end

function test_non_monotonic_grid()
    # Create a non-monotonic grid (to test warning)
    x_bad = [0.0, 0.5, 0.3, 1.0]
    f_bad = sin.(x_bad)
    
    println("Testing non-monotonic grid (should show warning):")
    result = compute_CDFD(f_bad, x_bad)
    println("Computation completed despite warning.")
    
    # Create a grid with zero spacing (to test epsilon adjustment)
    x_zero = [0.0, 0.5, 0.5, 1.0]
    f_zero = sin.(x_zero)
    
    println("\nTesting zero-spacing grid (should show warning):")
    result = compute_CDFD(f_zero, x_zero)
    println("Computation completed with epsilon adjustment.")
end

# Test with sine function (analytical test)
N = 32
println("\n===== ANALYTICAL TEST WITH SINE FUNCTION =====")
test_sine_function(N)

# Run the performance benchmark
println("\n===== PERFORMANCE BENCHMARK =====")
benchmark_CDFD_implementations(N)

# Test error handling
println("\n===== TESTING ERROR HANDLING =====")
test_non_monotonic_grid()

println("\nAll tests completed. Check the plots for visual comparisons.")
