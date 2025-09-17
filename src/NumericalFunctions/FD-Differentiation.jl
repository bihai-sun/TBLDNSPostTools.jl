#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               FD-Differentiation.jl
#                               ---------------------
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 14-10-2024
#    updated   : 06-05-2025
#    
#    This code contains functions to compute::
#        1. central difference differentiation 
#       
#========================================================================================================#

#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra


#========================================================================================================#
# Function to central difference in on a uniform grid
#========================================================================================================#

function compute_CDFD(f::AbstractArray{T,1}, Δ::T, dfdx::Union{AbstractArray{T,1},Nothing}=nothing) where T<:Real
"""
function to compute the central difference of a 1D array f on a uniform grid

    Arguments:
        f:    input array
        Δ:    grid spacing
        dfdx: derivative of f with respect to x, if passed as nothing, it will be initialized to zeros

    Returns:
        dfdx: derivative of f with respect to x
"""
    N    = length(f)
    if dfdx === nothing
        dfdx = zeros(T, N)
    end
    
    # Interior points (2nd order central difference)
    dfdx[2:end-1] .= (f[3:end] .- f[1:end-2]) / (2 * Δ)
    
    # First point (2nd order forward difference)
    if N >= 3
        dfdx[1] = (-3 * f[1] + 4 * f[2] - f[3]) / (2 * Δ)
    else
        dfdx[1] = (f[2] - f[1]) / Δ  # Fall back to 1st order if array is too small
    end
    
    # Last point (2nd order backward difference)
    if N >= 3
        dfdx[N] = (3 * f[N] - 4 * f[N-1] + f[N-2]) / (2 * Δ)
    else
        dfdx[N] = (f[N] - f[N-1]) / Δ  # Fall back to 1st order if array is too small
    end
    
    return dfdx
end


function compute_CDFD(f::AbstractArray{T,2}, Δ::T, dfdxᵢ::Union{AbstractArray{T,2},Nothing}=nothing; dim::Int=1) where T<:Real
"""
function to compute the central difference of a 2D array f on a uniform grid

    Arguments:
        f:      input array
        Δ:      grid spacing
        dfdxᵢ:  derivative of f with respect to xᵢ, if passed as nothing, it will be initialized to zeros
        dim:    dimension to differentiate along (1 for rows, 2 for columns)

    Returns:
        dfdxᵢ: derivative of f with respect to xᵢ
"""
    if dim == 1
        N     = size(f, 1)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, N, size(f, 2))
        end        
        
        # Interior points (2nd order central difference)
        dfdxᵢ[2:end-1, :] .= (f[3:end, :] .- f[1:end-2, :]) / (2 * Δ)
        
        # First point (2nd order forward difference)
        if N >= 3
            dfdxᵢ[1, :] .= (-3 * f[1, :] .+ 4 * f[2, :] .- f[3, :]) / (2 * Δ)
        else
            dfdxᵢ[1, :] .= (f[2, :] .- f[1, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
        # Last point (2nd order backward difference)
        if N >= 3
            dfdxᵢ[N, :] .= (3 * f[N, :] .- 4 * f[N-1, :] .+ f[N-2, :]) / (2 * Δ)
        else
            dfdxᵢ[N, :] .= (f[N, :] .- f[N-1, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
    elseif dim == 2
        N    = size(f, 2)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), N)
        end
        
        # Interior points (2nd order central difference)
        dfdxᵢ[:, 2:end-1] .= (f[:, 3:end] .- f[:, 1:end-2]) / (2 * Δ)
        
        # First point (2nd order forward difference)
        if N >= 3
            dfdxᵢ[:, 1] .= (-3 * f[:, 1] .+ 4 * f[:, 2] .- f[:, 3]) / (2 * Δ)
        else
            dfdxᵢ[:, 1] .= (f[:, 2] .- f[:, 1]) / Δ  # Fall back to 1st order if array is too small
        end
        
        # Last point (2nd order backward difference)
        if N >= 3
            dfdxᵢ[:, N] .= (3 * f[:, N] .- 4 * f[:, N-1] .+ f[:, N-2]) / (2 * Δ)
        else
            dfdxᵢ[:, N] .= (f[:, N] .- f[:, N-1]) / Δ  # Fall back to 1st order if array is too small
        end
        
    else
        error("Invalid dimension specified. Use dim=1 for rows or dim=2 for columns.")
    end
    return dfdxᵢ
end


function compute_CDFD(f::AbstractArray{T,3}, Δ::T, dfdxᵢ::Union{AbstractArray{T,3},Nothing}=nothing; dim::Int=1) where T<:Real
"""
function to compute the central difference of a 3D array f on a uniform grid

    Arguments:
        f:      input array
        Δ:      grid spacing
        dfdxᵢ:  derivative of f with respect to xᵢ, if passed as nothing, it will be initialized to zeros
        dim:    dimension to differentiate along (1 for x₁, 2 for x₂, 3 for x₃)

    Returns:
        dfdxᵢ: derivative of f with respect to xᵢ
"""
    if dim == 1
        N     = size(f, 1)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, N, size(f, 2), size(f, 3))
        end
        
        # Interior points (2nd order central difference)
        dfdxᵢ[2:end-1, :, :] .= (f[3:end, :, :] .- f[1:end-2, :, :]) / (2 * Δ)
        
        # First point (2nd order forward difference)
        if N >= 3
            dfdxᵢ[1, :, :] .= (-3 * f[1, :, :] .+ 4 * f[2, :, :] .- f[3, :, :]) / (2 * Δ)
        else
            dfdxᵢ[1, :, :] .= (f[2, :, :] .- f[1, :, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
        # Last point (2nd order backward difference)
        if N >= 3
            dfdxᵢ[N, :, :] .= (3 * f[N, :, :] .- 4 * f[N-1, :, :] .+ f[N-2, :, :]) / (2 * Δ)
        else
            dfdxᵢ[N, :, :] .= (f[N, :, :] .- f[N-1, :, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
    elseif dim == 2
        N    = size(f, 2)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), N, size(f, 3))
        end
        
        # Interior points (2nd order central difference)
        dfdxᵢ[:, 2:end-1, :] .= (f[:, 3:end, :] .- f[:, 1:end-2, :]) / (2 * Δ)
        
        # First point (2nd order forward difference)
        if N >= 3
            dfdxᵢ[:, 1, :] .= (-3 * f[:, 1, :] .+ 4 * f[:, 2, :] .- f[:, 3, :]) / (2 * Δ)
        else
            dfdxᵢ[:, 1, :] .= (f[:, 2, :] .- f[:, 1, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
        # Last point (2nd order backward difference)
        if N >= 3
            dfdxᵢ[:, N, :] .= (3 * f[:, N, :] .- 4 * f[:, N-1, :] .+ f[:, N-2, :]) / (2 * Δ)
        else
            dfdxᵢ[:, N, :] .= (f[:, N, :] .- f[:, N-1, :]) / Δ  # Fall back to 1st order if array is too small
        end
        
    elseif dim == 3
        N    = size(f, 3)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), size(f, 2), N)
        end
        
        # Interior points (2nd order central difference)
        dfdxᵢ[:, :, 2:end-1] .= (f[:, :, 3:end] .- f[:, :, 1:end-2]) / (2 * Δ)
        
        # First point (2nd order forward difference)
        if N >= 3
            dfdxᵢ[:, :, 1] .= (-3 * f[:, :, 1] .+ 4 * f[:, :, 2] .- f[:, :, 3]) / (2 * Δ)
        else
            dfdxᵢ[:, :, 1] .= (f[:, :, 2] .- f[:, :, 1]) / Δ  # Fall back to 1st order if array is too small
        end
        
        # Last point (2nd order backward difference)
        if N >= 3
            dfdxᵢ[:, :, N] .= (3 * f[:, :, N] .- 4 * f[:, :, N-1] .+ f[:, :, N-2]) / (2 * Δ)
        else
            dfdxᵢ[:, :, N] .= (f[:, :, N] .- f[:, :, N-1]) / Δ  # Fall back to 1st order if array is too small
        end
        
    else
        error("Invalid dimension specified. Use dim=1 for x₁, dim=2 for x₂, or dim=3 for x₃.")
    end
    return dfdxᵢ
end

#====================== End Function to central difference in on a uniform grid =========================#



#========================================================================================================#
# Functions for central difference on a non-uniform grid
#========================================================================================================#

function precompute_grid_coefficients(xᵢ::AbstractArray{T,1}) where T<:Real
"""
    precompute_grid_coefficients(xᵢ::AbstractArray{T,1}) where T<:Real

Precompute grid spacing coefficients for non-uniform grid differentiation.
This function can improve performance when the same grid is used multiple times.

    Arguments:
        xᵢ: Grid point coordinates

    Returns:
        A tuple of arrays containing precomputed coefficients for interior, first, and last points.
"""
    N = length(xᵢ)
    
    if N < 2
        error("Grid must have at least 2 points")
    end
    
    # Check grid monotonicity
    if !all(diff(xᵢ) .> 0)
        @warn "Non-monotonic grid detected. Results may be incorrect."
    end
    
    # Coefficients for interior points
    interior_coeffs = Array{Tuple{T,T,T}}(undef, N-2)
    
    for i in 2:N-1
        h₁ = xᵢ[i] - xᵢ[i-1]    # spacing to previous point
        h₂ = xᵢ[i+1] - xᵢ[i]    # spacing to next point
        
        # Safety check
        if h₁ ≈ 0 || h₂ ≈ 0
            @warn "Near-zero grid spacing detected at index $i. Using epsilon adjustment."
            h₁ = max(h₁, eps(T))
            h₂ = max(h₂, eps(T))
        end
        
        # Precompute coefficients for f[i-1], f[i], f[i+1]
        denom = h₁ * h₂ * (h₁ + h₂)
        c_prev = -h₂^2 / denom
        c_curr = -(h₁^2 - h₂^2) / denom  # Note the negative sign here
        c_next = h₁^2 / denom
        
        interior_coeffs[i-1] = (c_prev, c_curr, c_next)
    end
    
    # Coefficients for first point (2nd order forward)
    first_coeffs = (T(0), T(0), T(0))
    if N >= 3
        h₁ = max(xᵢ[2] - xᵢ[1], eps(T))
        h₂ = max(xᵢ[3] - xᵢ[2], eps(T))
        
        # Second-order forward difference coefficients
        c_1 = -((2*h₁ + h₂)/(h₁*(h₁+h₂)))
        c_2 = ((h₁+h₂)/(h₁*h₂))
        c_3 = -(h₁/((h₁+h₂)*h₂))
        
        first_coeffs = (c_1, c_2, c_3)
    else
        h₁ = max(xᵢ[2] - xᵢ[1], eps(T))
        first_coeffs = (T(-1)/h₁, T(1)/h₁, T(0))
    end
    
    # Coefficients for last point (2nd order backward)
    last_coeffs = (T(0), T(0), T(0))
    if N >= 3
        h₁ = max(xᵢ[N-1] - xᵢ[N-2], eps(T))
        h₂ = max(xᵢ[N] - xᵢ[N-1], eps(T))
        
        # Second-order backward difference coefficients
        c_1 = (h₂/((h₁+h₂)*h₁))
        c_2 = -((h₁+h₂)/(h₁*h₂))
        c_3 = ((2*h₂+h₁)/(h₂*(h₁+h₂)))
        
        last_coeffs = (c_1, c_2, c_3)
    else
        h₂ = max(xᵢ[N] - xᵢ[N-1], eps(T))
        last_coeffs = (T(0), T(-1)/h₂, T(1)/h₂)
    end
    
    return (interior_coeffs, first_coeffs, last_coeffs)
end

function compute_CDFD(f::AbstractArray{T,1}, xᵢ::AbstractArray{T,1}, dfdx::Union{AbstractArray{T,1},Nothing}=nothing) where T<:Real
"""
function to compute the central difference of a 1D array f on a non-uniform grid

    Arguments:
        f:      input array
        xᵢ:     grid point coordinates
        dfdx:   derivative of f with respect to x, if passed as nothing, it will be initialized to zeros

    Returns:
        dfdx: derivative of f with respect to x
        
    Notes:
        Assumes xᵢ is a strictly monotonic increasing sequence of grid points.
        Will emit a warning if non-monotonic grid points are detected.
"""
    N    = length(f)
    if dfdx === nothing
        dfdx = zeros(T, N)
    end
    
    # Check grid monotonicity
    if !all(diff(xᵢ) .> 0)
        @warn "Non-monotonic grid detected. Results may be incorrect."
    end
    
    # Interior points using central difference formula for non-uniform grid
    for i in 2:N-1
        h₁ = xᵢ[i] - xᵢ[i-1]    # spacing to previous point
        h₂ = xᵢ[i+1] - xᵢ[i]    # spacing to next point
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            @warn "Near-zero grid spacing detected at index $i. Using fallback."
            dfdx[i] = (f[i+1] - f[i-1]) / (xᵢ[i+1] - xᵢ[i-1] + eps(T))
            continue
        end
        
        # Central difference formula for non-uniform grid
        dfdx[i] = (h₁^2 * f[i+1] - (h₁^2 - h₂^2) * f[i] - h₂^2 * f[i-1]) / (h₁ * h₂ * (h₁ + h₂))
    end
    
    # Forward difference for first point (second-order accurate)
    if N >= 3
        h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
        h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            @warn "Near-zero grid spacing detected at boundary. Using fallback."
            dfdx[1] = (f[2] - f[1]) / (xᵢ[2] - xᵢ[1] + eps(T))
        else
            # Second-order forward difference formula for non-uniform grid
            dfdx[1] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[1] + ((h₁+h₂)/(h₁*h₂)) * f[2] - (h₁/((h₁+h₂)*h₂)) * f[3])
        end
    else
        # Fall back to first-order if array is too small
        h₁ = xᵢ[2] - xᵢ[1]
        if h₁ ≈ 0
            @warn "Near-zero grid spacing detected at boundary. Using epsilon adjustment."
            h₁ += eps(T)
        end
        dfdx[1] = (f[2] - f[1]) / h₁
    end
    
    # Backward difference for last point (second-order accurate)
    if N >= 3
        h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
        h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            @warn "Near-zero grid spacing detected at boundary. Using fallback."
            dfdx[N] = (f[N] - f[N-1]) / (xᵢ[N] - xᵢ[N-1] + eps(T))
        else
            # Second-order backward difference formula for non-uniform grid
            dfdx[N] = ((h₂/((h₁+h₂)*h₁)) * f[N-2] - ((h₁+h₂)/(h₁*h₂)) * f[N-1] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[N])
        end
    else
        # Fall back to first-order if array is too small
        h₂ = xᵢ[N] - xᵢ[N-1]
        if h₂ ≈ 0
            @warn "Near-zero grid spacing detected at boundary. Using epsilon adjustment."
            h₂ += eps(T)
        end
        dfdx[N] = (f[N] - f[N-1]) / h₂
    end
    
    return dfdx
end


function compute_CDFD_optimized(f::AbstractArray{T,1}, xᵢ::AbstractArray{T,1}, 
                                coeffs::Tuple, 
                                dfdx::Union{AbstractArray{T,1},Nothing}=nothing) where T<:Real
"""
    compute_CDFD_optimized(f, xᵢ, coeffs; [dfdx])

Compute central differences using pre-computed grid coefficients for improved performance.

Arguments:
    f:      Input array
    xᵢ:     Grid point coordinates
    coeffs: Pre-computed grid coefficients (from precompute_grid_coefficients)
    dfdx:   Optional output array. If not provided, a new array will be allocated.

Returns:
    dfdx:   Derivative of f with respect to x
"""
    N = length(f)
    if length(xᵢ) != N
        error("Input array and grid must have the same length")
    end
    
    if dfdx === nothing
        dfdx = zeros(T, N)
    end
    
    interior_coeffs, first_coeffs, last_coeffs = coeffs
    
    # Apply coefficients to interior points
    for i in 2:N-1
        c_prev, c_curr, c_next = interior_coeffs[i-1]
        dfdx[i] = c_prev * f[i-1] + c_curr * f[i] + c_next * f[i+1]
    end
    
    # Apply coefficients to first point
    if N >= 3
        c_1, c_2, c_3 = first_coeffs
        dfdx[1] = c_1 * f[1] + c_2 * f[2] + c_3 * f[3]
    else
        c_1, c_2, _ = first_coeffs
        dfdx[1] = c_1 * f[1] + c_2 * f[2]
    end
    
    # Apply coefficients to last point
    if N >= 3
        c_1, c_2, c_3 = last_coeffs
        dfdx[N] = c_1 * f[N-2] + c_2 * f[N-1] + c_3 * f[N]
    else
        _, c_2, c_3 = last_coeffs
        dfdx[N] = c_2 * f[N-1] + c_3 * f[N]
    end
    
    return dfdx
end

# Alternative implementation with direct computation - more accurate but slightly less optimized
function compute_CDFD_optimized_safe(f::AbstractArray{T,1}, xᵢ::AbstractArray{T,1}, 
                                   coeffs::Tuple, 
                                   dfdx::Union{AbstractArray{T,1},Nothing}=nothing) where T<:Real
"""
    compute_CDFD_optimized_safe(f, xᵢ, coeffs; [dfdx])
Compute central differences using pre-computed grid coefficients for improved performance.

    Arguments:
        f:      Input array
        xᵢ:     Grid point coordinates
        coeffs: Pre-computed grid coefficients (from precompute_grid_coefficients)
        dfdx:   Optional output array. If not provided, a new array will be allocated.

    Returns:
        dfdx:   Derivative of f with respect to x
"""
    N = length(f)
    if length(xᵢ) != N
        error("Input array and grid must have the same length")
    end
    
    if dfdx === nothing
        dfdx = zeros(T, N)
    end
    
    # Interior points
    for i in 2:N-1
        h₁ = xᵢ[i] - xᵢ[i-1]    # spacing to previous point
        h₂ = xᵢ[i+1] - xᵢ[i]    # spacing to next point
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            h₁ = max(h₁, eps(T))
            h₂ = max(h₂, eps(T))
        end
        
        # Central difference formula for non-uniform grid
        dfdx[i] = (h₁^2 * f[i+1] - (h₁^2 - h₂^2) * f[i] - h₂^2 * f[i-1]) / (h₁ * h₂ * (h₁ + h₂))
    end
    
    # Forward difference for first point (second-order accurate)
    if N >= 3
        h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
        h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            h₁ = max(h₁, eps(T))
            h₂ = max(h₂, eps(T))
        end
        
        # Second-order forward difference formula for non-uniform grid
        dfdx[1] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[1] + ((h₁+h₂)/(h₁*h₂)) * f[2] - (h₁/((h₁+h₂)*h₂)) * f[3])
    else
        # Fall back to first-order if array is too small
        h₁ = max(xᵢ[2] - xᵢ[1], eps(T))
        dfdx[1] = (f[2] - f[1]) / h₁
    end
    
    # Backward difference for last point (second-order accurate)
    if N >= 3
        h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
        h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points
        
        # Safety check for zero spacing
        if h₁ ≈ 0 || h₂ ≈ 0
            h₁ = max(h₁, eps(T))
            h₂ = max(h₂, eps(T))
        end
        
        # Second-order backward difference formula for non-uniform grid
        dfdx[N] = ((h₂/((h₁+h₂)*h₁)) * f[N-2] - ((h₁+h₂)/(h₁*h₂)) * f[N-1] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[N])
    else
        # Fall back to first-order if array is too small
        h₂ = max(xᵢ[N] - xᵢ[N-1], eps(T))
        dfdx[N] = (f[N] - f[N-1]) / h₂
    end
    
    return dfdx
end


function compute_CDFD(f::AbstractArray{T,2}, xᵢ::AbstractArray{T,1}, dfdxᵢ::Union{AbstractArray{T,2},Nothing}=nothing; dim::Int=1) where T<:Real
"""
function to compute the central difference of a 2D array f on a non-uniform grid

    Arguments:
        f:      input array
        xᵢ:     grid point coordinates along the dimension to differentiate
        dfdxᵢ:  derivative of f with respect to xᵢ, if passed as nothing, it will be initialized to zeros
        dim:    dimension to differentiate along (1 for rows, 2 for columns)

    Returns:
        dfdxᵢ: derivative of f with respect to xᵢ
"""
    if dim == 1
        N     = size(f, 1)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, N, size(f, 2))
        end
        
        # For each column
        for j in 1:size(f, 2)
            # Interior points
            for i in 2:N-1
                h₁ = xᵢ[i] - xᵢ[i-1]    # spacing to previous point
                h₂ = xᵢ[i+1] - xᵢ[i]    # spacing to next point
                # Central difference formula for non-uniform grid
                dfdxᵢ[i, j] = (h₁^2 * f[i+1, j] - (h₁^2 - h₂^2) * f[i, j] - h₂^2 * f[i-1, j]) / (h₁ * h₂ * (h₁ + h₂))
            end
            
            # Forward difference for first point (second-order accurate)
            if N >= 3
                h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
                h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next   
                # Second-order forward difference formula for non-uniform grid
                dfdxᵢ[1, j] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[1, j] + ((h₁+h₂)/(h₁*h₂)) * f[2, j] - (h₁/((h₁+h₂)*h₂)) * f[3, j])
            else
                # Fall back to first-order if array is too small
                h₁          = xᵢ[2] - xᵢ[1]
                dfdxᵢ[1, j] = (f[2, j] - f[1, j]) / h₁
            end
            
            # Backward difference for last point (second-order accurate)
            if N >= 3
                h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
                h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points
                
                # Second-order backward difference formula for non-uniform grid
                dfdxᵢ[N, j] = ((h₂/((h₁+h₂)*h₁)) * f[N-2, j] - ((h₁+h₂)/(h₁*h₂)) * f[N-1, j] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[N, j])
            else
                # Fall back to first-order if array is too small
                h₂          = xᵢ[N] - xᵢ[N-1]
                dfdxᵢ[N, j] = (f[N, j] - f[N-1, j]) / h₂
            end
        end
        
    elseif dim == 2
        N     = size(f, 2)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), N)
        end
        
        # For each row
        for i in 1:size(f, 1)
            # Interior points
            for j in 2:N-1
                h₁ = xᵢ[j] - xᵢ[j-1]    # spacing to previous point
                h₂ = xᵢ[j+1] - xᵢ[j]    # spacing to next point
                # Central difference formula for non-uniform grid
                dfdxᵢ[i, j] = (h₁^2 * f[i, j+1] - (h₁^2 - h₂^2) * f[i, j] - h₂^2 * f[i, j-1]) / (h₁ * h₂ * (h₁ + h₂))
            end
            
            # Forward difference for first point (second-order accurate)
            if N >= 3
                h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
                h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next
                
                # Second-order forward difference formula for non-uniform grid
                dfdxᵢ[i, 1] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[i, 1] + ((h₁+h₂)/(h₁*h₂)) * f[i, 2] - (h₁/((h₁+h₂)*h₂)) * f[i, 3])
            else
                # Fall back to first-order if array is too small
                h₁          = xᵢ[2] - xᵢ[1]
                dfdxᵢ[i, 1] = (f[i, 2] - f[i, 1]) / h₁
            end
            
            # Backward difference for last point (second-order accurate)
            if N >= 3
                h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
                h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points
                
                # Second-order backward difference formula for non-uniform grid
                dfdxᵢ[i, N] = ((h₂/((h₁+h₂)*h₁)) * f[i, N-2] - ((h₁+h₂)/(h₁*h₂)) * f[i, N-1] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[i, N])
            else
                # Fall back to first-order if array is too small
                h₂          = xᵢ[N] - xᵢ[N-1]
                dfdxᵢ[i, N] = (f[i, N] - f[i, N-1]) / h₂
            end
        end
    else
        error("Invalid dimension specified. Use dim=1 for rows or dim=2 for columns.")
    end
    
    return dfdxᵢ
end


function compute_CDFD(f::AbstractArray{T,3}, xᵢ::AbstractArray{T,1}, dfdxᵢ::Union{AbstractArray{T,3},Nothing}=nothing; dim::Int=1) where T<:Real
"""
function to compute the central difference of a 3D array f on a non-uniform grid

    Arguments:
        f:      input array
        xᵢ:     grid point coordinates along the dimension to differentiate
        dfdxᵢ:  derivative of f with respect to xᵢ, if passed as nothing, it will be initialized to zeros
        dim:    dimension to differentiate along (1 for x₁, 2 for x₂, 3 for x₃)

    Returns:
        dfdxᵢ: derivative of f with respect to xᵢ
"""
    if dim == 1
        N     = size(f, 1)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, N, size(f, 2), size(f, 3))
        end
        
        # For each j,k combination
        for k in 1:size(f, 3)
            for j in 1:size(f, 2)
                # Interior points
                for i in 2:N-1
                    h₁ = xᵢ[i] - xᵢ[i-1]    # spacing to previous point
                    h₂ = xᵢ[i+1] - xᵢ[i]    # spacing to next point
                    # Central difference formula for non-uniform grid
                    dfdxᵢ[i, j, k] = (h₁^2 * f[i+1, j, k] - (h₁^2 - h₂^2) * f[i, j, k] - h₂^2 * f[i-1, j, k]) / (h₁ * h₂ * (h₁ + h₂))
                end
                
                # Forward difference for first point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
                    h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next
                    
                    # Second-order forward difference formula for non-uniform grid
                    dfdxᵢ[1, j, k] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[1, j, k] + ((h₁+h₂)/(h₁*h₂)) * f[2, j, k] - (h₁/((h₁+h₂)*h₂)) * f[3, j, k])
                else
                    # Fall back to first-order if array is too small
                    h₁             = xᵢ[2] - xᵢ[1]
                    dfdxᵢ[1, j, k] = (f[2, j, k] - f[1, j, k]) / h₁
                end
                
                # Backward difference for last point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
                    h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points
                    
                    # Second-order backward difference formula for non-uniform grid
                    dfdxᵢ[N, j, k] = ((h₂/((h₁+h₂)*h₁)) * f[N-2, j, k] - ((h₁+h₂)/(h₁*h₂)) * f[N-1, j, k] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[N, j, k])
                else
                    # Fall back to first-order if array is too small
                    h₂             = xᵢ[N] - xᵢ[N-1]
                    dfdxᵢ[N, j, k] = (f[N, j, k] - f[N-1, j, k]) / h₂
                end
            end
        end
        
    elseif dim == 2
        N     = size(f, 2)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), N, size(f, 3))
        end
        
        # For each i,k combination
        for k in 1:size(f, 3)
            for i in 1:size(f, 1)
                # Interior points
                for j in 2:N-1
                    h₁ = xᵢ[j] - xᵢ[j-1]    # spacing to previous point
                    h₂ = xᵢ[j+1] - xᵢ[j]    # spacing to next point
                    # Central difference formula for non-uniform grid
                    dfdxᵢ[i, j, k] = (h₁^2 * f[i, j+1, k] - (h₁^2 - h₂^2) * f[i, j, k] - h₂^2 * f[i, j-1, k]) /  (h₁ * h₂ * (h₁ + h₂))
                end
                
                # Forward difference for first point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
                    h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next   
                    # Second-order forward difference formula for non-uniform grid
                    dfdxᵢ[i, 1, k] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[i, 1, k] + ((h₁+h₂)/(h₁*h₂)) * f[i, 2, k] - (h₁/((h₁+h₂)*h₂)) * f[i, 3, k])
                else
                    # Fall back to first-order if array is too small
                    h₁             = xᵢ[2] - xᵢ[1]
                    dfdxᵢ[i, 1, k] = (f[i, 2, k] - f[i, 1, k]) / h₁
                end
                
                # Backward difference for last point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[N-1] - xᵢ[N-2]    # spacing between second-to-last and third-to-last points
                    h₂ = xᵢ[N] - xᵢ[N-1]      # spacing between last and second-to-last points 
                    # Second-order backward difference formula for non-uniform grid
                    dfdxᵢ[i, N, k] = ((h₂/((h₁+h₂)*h₁)) * f[i, N-2, k] - ((h₁+h₂)/(h₁*h₂)) * f[i, N-1, k] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[i, N, k])
                else
                    # Fall back to first-order if array is too small
                    h₂             = xᵢ[N] - xᵢ[N-1]
                    dfdxᵢ[i, N, k] = (f[i, N, k] - f[i, N-1, k]) / h₂
                end
            end
        end
        
    elseif dim == 3
        N     = size(f, 3)
        if dfdxᵢ === nothing
            dfdxᵢ = zeros(T, size(f, 1), size(f, 2), N)
        end

        # For each i,j combination
        for j in 1:size(f, 2)
            for i in 1:size(f, 1)
                # Interior points
                for k in 2:N-1
                    h₁ = xᵢ[k] - xᵢ[k-1]    # spacing to previous point
                    h₂ = xᵢ[k+1] - xᵢ[k]    # spacing to next point
                    # Central difference formula for non-uniform grid
                    dfdxᵢ[i, j, k] = (h₁^2 * f[i, j, k+1] - (h₁^2 - h₂^2) * f[i, j, k] - h₂^2 * f[i, j, k-1]) / (h₁ * h₂ * (h₁ + h₂))
                end
                
                # Forward difference for first point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[2] - xᵢ[1]    # spacing to next point
                    h₂ = xᵢ[3] - xᵢ[2]    # spacing to point after next
                    
                    # Second-order forward difference formula for non-uniform grid
                    dfdxᵢ[i, j, 1] = (-((2*h₁ + h₂)/(h₁*(h₁+h₂))) * f[i, j, 1] + ((h₁+h₂)/(h₁*h₂)) * f[i, j, 2] - (h₁/((h₁+h₂)*h₂)) * f[i, j, 3])
                else
                    # Fall back to first-order if array is too small
                    h₁             = xᵢ[2] - xᵢ[1]
                    dfdxᵢ[i, j, 1] = (f[i, j, 2] - f[i, j, 1]) / h₁
                end
                
                # Backward difference for last point (second-order accurate)
                if N >= 3
                    h₁ = xᵢ[N-1] - xᵢ[N-2] # spacing between second-to-last and third-to-last points
                    h₂ = xᵢ[N] - xᵢ[N-1]   # spacing between last and second-to-last points
                    # Second-order backward difference formula for non-uniform grid
                    dfdxᵢ[i, j, N] = ((h₂/((h₁+h₂)*h₁)) * f[i, j, N-2] - ((h₁+h₂)/(h₁*h₂)) * f[i, j, N-1] + ((2*h₂+h₁)/(h₂*(h₁+h₂))) * f[i, j, N])
                else
                    # Fall back to first-order if array is too small
                    h₂             = xᵢ[N] - xᵢ[N-1]
                    dfdxᵢ[i, j, N] = (f[i, j, N] - f[i, j, N-1]) / h₂
                end
            end
        end
    else
        error("Invalid dimension specified. Use dim=1 for x₁, dim=2 for x₂, or dim=3 for x₃.")
    end
    
    return dfdxᵢ
end

#==================== End Function to central difference in on a non-uniform grid =======================#


#========================================================================================================#
# Testing ...
#========================================================================================================#

function plot_scaled_error(x, exact, approx; plot_title="Error Comparison", scale_factor=10)
    """
    Create a plot showing the scaled error between exact and approximate derivatives.
    
    Arguments:
        x: Grid coordinates
        exact: Exact derivative values
        approx: Approximate derivative values
        plot_title: Title for the plot
        scale_factor: Scale factor for errors (default 10)
        
    Returns:
        Tuple of (plot_handle, scale_factor)
    """
    # Calculate error
    error = exact - approx
    
    # Get max error for display purposes
    max_error = maximum(abs.(error))
    max_scaled_error = max_error * scale_factor
    
    # Scale the error
    scaled_error = error * scale_factor
    
    # Create the plot
    plt = plot(x, exact, label="Exact", linewidth=2, title=plot_title)
    plot!(plt, x, approx, label="Numerical", linestyle=:dash, linewidth=2)
    plot!(plt, x, scaled_error, label="Error × $(scale_factor)", 
          linewidth=1, linestyle=:dot, color=:red)
    
    # Add a horizontal line at zero
    hline!(plt, [0], label=false, color=:black, linewidth=0.5, linestyle=:dot)
    
    # Add legend showing actual max error
    annotate!(plt, [(x[end], maximum(exact)/2, 
               text("Max error: $(round(max_error, digits=6))", 8, :right, :black))])
    
    return plt, scale_factor
end


function test_CDiff_1D(n::Int=20)
    # Test the central 1D difference functions
    f = zeros(Real, n)

    λ  = 2.0
    L  = 2π * λ
    Δ  = L/n
    x  = (0.0:Δ:L)[1:end-1]

    # Function
    f     = sin.(x)
    dfdx  = cos.(x)
    
    # compute the derivative using uniform grid coeffts:
    dfdx_uniform = compute_CDFD(f, Δ)
    
    # compute the derivative using non-uniform grid coeffts:
    dfdx_non_uniform = compute_CDFD(f, collect(x))

    # Plot derivatives
    fig1 = plot(x, dfdx, label="Exact")
    plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid")
    plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Grid")
    display(fig1)
    
    # Set scale factor for error plots
    scale_factor = 1
    
    # Plot scaled errors with fixed scale factors
    fig_error_uniform, factor_uniform = plot_scaled_error(
        x, dfdx, dfdx_uniform, 
        plot_title="Uniform Grid Error (scaled ×$(scale_factor))",
        scale_factor=scale_factor
    )
    display(fig_error_uniform)
    
    fig_error_nonuniform, factor_nonuniform = plot_scaled_error(
        x, dfdx, dfdx_non_uniform, 
        plot_title="Non-Uniform Grid Error (scaled ×$(scale_factor))",
        scale_factor=scale_factor
    )
    display(fig_error_nonuniform)
    
    # Print error statistics
    println("Uniform grid max absolute error: ", maximum(abs.(dfdx - dfdx_uniform)))
    println("Non-uniform grid max absolute error: ", maximum(abs.(dfdx - dfdx_non_uniform)))
end


function test_CDiff_3D(n₁::Int=64, n₂::Int=48, n₃::Int=32)
    """
    Comprehensive test of 3D differentiation on uniform and non-uniform grids
    
    Arguments:
        n₁, n₂, n₃: grid dimensions
    """
    println("=== Testing 3D differentiation with grid size: $(n₁) × $(n₂) × $(n₃) ===")
    
    # Create a function and its analytical derivatives
    λ₁ = 1.0
    L₁ = 2π * λ₁
    Δ₁ = L₁/(n₁-1)
    x₁ = (0.0:Δ₁:L₁)[1:end]

    λ₂ = 1.0
    L₂ = 2π * λ₂
    Δ₂ = L₂/(n₂-1)
    x₂ = (0.0:Δ₂:L₂)[1:end]

    λ₃ = 1.0
    L₃ = 2π * λ₃
    Δ₃ = L₃/(n₃-1)
    x₃ = (0.0:Δ₃:L₃)[1:end]

    # Create 3D arrays for coordinates
    X₁ = reshape(x₁, n₁, 1, 1)
    X₂ = reshape(x₂, 1, n₂, 1)
    X₃ = reshape(x₃, 1, 1, n₃)
    
    # Test function and analytical derivatives
    f           = sin.(X₁) .* cos.(X₂) .* sin.(X₃)
    dfdx₁_exact = cos.(X₁) .* cos.(X₂) .* sin.(X₃)
    dfdx₂_exact = sin.(X₁) .* (-sin.(X₂)) .* sin.(X₃)
    dfdx₃_exact = sin.(X₁) .* cos.(X₂) .* cos.(X₃)
    
    # 1. Test uniform grid differentiation
    println("\n--- Testing uniform grid differentiation ---")
    dfdx₁_uniform = compute_CDFD(f, Δ₁, dim=1)
    dfdx₂_uniform = compute_CDFD(f, Δ₂, dim=2)
    dfdx₃_uniform = compute_CDFD(f, Δ₃, dim=3)
    
    # Compute errors for uniform grid
    error₁_uniform = norm(dfdx₁_exact - dfdx₁_uniform) / norm(dfdx₁_exact)
    error₂_uniform = norm(dfdx₂_exact - dfdx₂_uniform) / norm(dfdx₂_exact)
    error₃_uniform = norm(dfdx₃_exact - dfdx₃_uniform) / norm(dfdx₃_exact)
    
    println("Uniform grid relative error (x₁): ", error₁_uniform)
    println("Uniform grid relative error (x₂): ", error₂_uniform)
    println("Uniform grid relative error (x₃): ", error₃_uniform)
    
    # 2. Test non-uniform grid differentiation
    # First, create a non-uniform grid by applying a transformation
    println("\n--- Testing non-uniform grid differentiation ---")
    
    # Create slightly non-uniform grids with a sine perturbation
    perturbation_factor = 0.05
    x₁_nonuniform = x₁ + perturbation_factor * L₁/n₁ * sin.(2π .* x₁ ./ L₁)
    x₂_nonuniform = x₂ + perturbation_factor * L₂/n₂ * sin.(2π .* x₂ ./ L₂)
    x₃_nonuniform = x₃ + perturbation_factor * L₃/n₃ * sin.(2π .* x₃ ./ L₃)
    
    # Make sure the non-uniform grids remain monotonically increasing
    @assert all(diff(x₁_nonuniform) .> 0) "x₁_nonuniform is not monotonically increasing!"
    @assert all(diff(x₂_nonuniform) .> 0) "x₂_nonuniform is not monotonically increasing!"
    @assert all(diff(x₃_nonuniform) .> 0) "x₃_nonuniform is not monotonically increasing!"
    
    # Generate the function values on the non-uniform grid (interpolate from uniform grid)
    # For demonstration, we'll use the same function values, interpolation would be more accurate
    
    # Compute derivatives with non-uniform grid methods
    dfdx₁_nonuniform = compute_CDFD(f, x₁_nonuniform, dim=1)
    dfdx₂_nonuniform = compute_CDFD(f, x₂_nonuniform, dim=2)
    dfdx₃_nonuniform = compute_CDFD(f, x₃_nonuniform, dim=3)
    
    # Compute errors for non-uniform grid
    error₁_nonuniform = norm(dfdx₁_exact - dfdx₁_nonuniform) / norm(dfdx₁_exact)
    error₂_nonuniform = norm(dfdx₂_exact - dfdx₂_nonuniform) / norm(dfdx₂_exact)
    error₃_nonuniform = norm(dfdx₃_exact - dfdx₃_nonuniform) / norm(dfdx₃_exact)
    
    println("Non-uniform grid relative error (x₁): ", error₁_nonuniform)
    println("Non-uniform grid relative error (x₂): ", error₂_nonuniform)
    println("Non-uniform grid relative error (x₃): ", error₃_nonuniform)
    
    # 3. Compare at a specific slice
    println("\n--- Creating plots comparing exact vs numerical derivatives ---")
    
    # Select middle indices for 2D slices
    mid₁ = div(n₁, 2)
    mid₂ = div(n₂, 2)
    mid₃ = div(n₃, 2)
    
    # Plot comparison for x₁ derivative (slice at mid₂, mid₃)
    fig1 = plot(x₁, dfdx₁_exact[:, mid₂, mid₃], label="Exact", title="∂f/∂x₁ at x₂=$(x₂[mid₂]), x₃=$(x₃[mid₃])",
                xlabel="x₁", ylabel="∂f/∂x₁", lw=2)
    plot!(fig1, x₁, dfdx₁_uniform[:, mid₂, mid₃], seriestype=:scatter, label="Uniform Grid", markersize=4)
    plot!(fig1, x₁, dfdx₁_nonuniform[:, mid₂, mid₃], seriestype=:scatter, label="Non-Uniform Grid", markersize=4)
    
    # Plot comparison for x₂ derivative (slice at mid₁, mid₃)
    fig2 = plot(x₂, dfdx₂_exact[mid₁, :, mid₃], label="Exact", title="∂f/∂x₂ at x₁=$(x₁[mid₁]), x₃=$(x₃[mid₃])",
                xlabel="x₂", ylabel="∂f/∂x₂", lw=2)
    plot!(fig2, x₂, dfdx₂_uniform[mid₁, :, mid₃], seriestype=:scatter, label="Uniform Grid", markersize=4)
    plot!(fig2, x₂, dfdx₂_nonuniform[mid₁, :, mid₃], seriestype=:scatter, label="Non-Uniform Grid", markersize=4)
    
    # Plot comparison for x₃ derivative (slice at mid₁, mid₂)
    fig3 = plot(x₃, dfdx₃_exact[mid₁, mid₂, :], label="Exact", title="∂f/∂x₃ at x₁=$(x₁[mid₁]), x₂=$(x₂[mid₂])",
                xlabel="x₃", ylabel="∂f/∂x₃", lw=2)
    plot!(fig3, x₃, dfdx₃_uniform[mid₁, mid₂, :], seriestype=:scatter, label="Uniform Grid", markersize=4)
    plot!(fig3, x₃, dfdx₃_nonuniform[mid₁, mid₂, :], seriestype=:scatter, label="Non-Uniform Grid", markersize=4)
    
    # Display the plots
    display(fig1)
    display(fig2)
    display(fig3)
    
    # Create error plots with scaled errors
    # For x₁ derivative
    fig_error_x1_uniform, factor_x1_uniform = plot_scaled_error(
        x₁, dfdx₁_exact[:, mid₂, mid₃], dfdx₁_uniform[:, mid₂, mid₃],
        plot_title="x₁ Uniform Grid Error (scaled ×10)"
    )
    
    fig_error_x1_nonuniform, factor_x1_nonuniform = plot_scaled_error(
        x₁, dfdx₁_exact[:, mid₂, mid₃], dfdx₁_nonuniform[:, mid₂, mid₃],
        plot_title="x₁ Non-Uniform Grid Error (scaled ×10)"
    )
    
    # For x₂ derivative
    fig_error_x2_uniform, factor_x2_uniform = plot_scaled_error(
        x₂, dfdx₂_exact[mid₁, :, mid₃], dfdx₂_uniform[mid₁, :, mid₃],
        plot_title="x₂ Uniform Grid Error (scaled ×10)"
    )
    
    fig_error_x2_nonuniform, factor_x2_nonuniform = plot_scaled_error(
        x₂, dfdx₂_exact[mid₁, :, mid₃], dfdx₂_nonuniform[mid₁, :, mid₃],
        plot_title="x₂ Non-Uniform Grid Error (scaled ×10)"
    )
    
    # For x₃ derivative
    fig_error_x3_uniform, factor_x3_uniform = plot_scaled_error(
        x₃, dfdx₃_exact[mid₁, mid₂, :], dfdx₃_uniform[mid₁, mid₂, :],
        plot_title="x₃ Uniform Grid Error (scaled ×10)"
    )
    
    fig_error_x3_nonuniform, factor_x3_nonuniform = plot_scaled_error(
        x₃, dfdx₃_exact[mid₁, mid₂, :], dfdx₃_nonuniform[mid₁, mid₂, :],
        plot_title="x₃ Non-Uniform Grid Error (scaled ×10)"
    )
    
    # Display error plots
    display(fig_error_x1_uniform)
    display(fig_error_x1_nonuniform)
    display(fig_error_x2_uniform)
    display(fig_error_x2_nonuniform)
    display(fig_error_x3_uniform)
    display(fig_error_x3_nonuniform)
    
    # Create combined error plots (uniform vs non-uniform)
    fig_combined_error_x1 = plot(
        fig_error_x1_uniform, fig_error_x1_nonuniform,
        layout=(2,1), size=(800, 800),
        plot_title="x₁ Direction Error Comparison"
    )
    
    fig_combined_error_x2 = plot(
        fig_error_x2_uniform, fig_error_x2_nonuniform,
        layout=(2,1), size=(800, 800),
        plot_title="x₂ Direction Error Comparison"
    )
    
    fig_combined_error_x3 = plot(
        fig_error_x3_uniform, fig_error_x3_nonuniform,
        layout=(2,1), size=(800, 800),
        plot_title="x₃ Direction Error Comparison"
    )
    
    display(fig_combined_error_x1)
    display(fig_combined_error_x2)
    display(fig_combined_error_x3)
    
    # 4. Combine into a single figure with subplots
    fig_combined = plot(fig1, fig2, fig3, layout=(3,1), size=(800, 1200),
                       plot_title="Comparison of Differentiation Methods")
    display(fig_combined)
    
    # Return error metrics
    return (
        uniform = (x₁=error₁_uniform, x₂=error₂_uniform, x₃=error₃_uniform),
        nonuniform = (x₁=error₁_nonuniform, x₂=error₂_nonuniform, x₃=error₃_nonuniform)
    )
end


function test_CDiff()
    # Test the central difference functions
    n₁ = 20
    n₂ = 10
    n₃ = 30

    # Start with 1D test
    test_CDiff_1D(64)
    
    # Then run 3D test
    test_CDiff_3D(64, 48, 32)
end


if abspath(PROGRAM_FILE) == @__FILE__
    test_CDiff()
end