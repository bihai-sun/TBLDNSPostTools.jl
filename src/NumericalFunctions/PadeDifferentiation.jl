#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               PadeDifferentiation.jl
#                               ----------------------
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 14-10-2024
#    updated   : 05-05-2025
#    
#    This code contains functions to compute::
#        1. multi-dimensional (1D, 2D, 3D) uniformly spaced 6th order Pade differentiation
#        2. multi-dimensional (1D, 2D, 3D) uneqaully spaced 6th order Pade differentiation 
#       
#========================================================================================================#


#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra
using Distributions



#========================================================================================================#
# Preliminary Pade Differentiation Functions
#========================================================================================================#

function compute_weights_6order_pade_non_uniform(y::AbstractArray{T,1}) where T<:Real
#=====================================================================================================#
#  function to compute the 6 order Pade-derivative weights for a non-uniform grid defined by the grid:
#        y[1:ny]
#    the distance between the grid points is given by:
#        h[i] = y[i+1] - y[i] for i = 1, ..., ny-1 
#        and is stored in h[1:ny-1]
#
#    The Pade-derviatives are computed 
#        A f' = B f 
#
#        A is tridiagonal (alpha/0, 1, beta/0)  while 
#        B is 5-banded denoted in each row by (AA, BB, CC, DD, EE)
#        
#    the values for each row of A & B for i = 1, ..., ny are stored in coef(8,ny):
#        coef[1, i] = AA
#        coef[2, i] = BB
#        coef[3, i] = CC
#        coef[4, i] = DD
#        coef[5, i] = EE
#        coef[6, i] = alpha or 0 for i = 1
#        coef[7, i] = 1
#        coef[8, i] = beta or 0 for i = ny
#
#    Note: - f[0:ny] and dfdy[0:ny], f[ny+1] is not used!!!        
#=====================================================================================================#
"""
    Arguments:
        y::AbstractArray{T,1} : 1D array of size ny containing the non-uniform grid points

    Returns:
        coef::Array{T,2} : 8 x ny array containing the coefficients for the Pade-derivative
"""
    # assign arrays
    ny   = length(y)
    h    = zeros(T, ny-1)
    coef = zeros(T, 8,ny)

    # compute spacing between grid points
    h[1:ny-1] .= y[2:ny] .- y[1:ny-1]

    # BOUNDARY SCHEME FOR I = 1
    a = h[1]
    b = a + h[2]
    c = b + h[3]
    d = c + h[4]
    # B matrix components:
    coef[1, 1] = -((a*b*c + a*b*d + a*c*d + 2*b*c*d)/(a*b*c*d))
    coef[2, 1] = -((b*c*d*(5*a^3 - 4*a^2*b - 4*a^2*c + 3*a*b*c - 4*a^2*d + 3*a*b*d + 3*a*c*d - 2*b*c*d))/
    (a*(a - b)^2*(a - c)^2*(a - d)^2))
    coef[3, 1] = (a^2 * c * d) / ((a - b)^2 * b * (b - c) * (b - d))
    coef[4, 1] = -((a^2 * b * d) / ((a - c)^2 * (b - c) * c * (c - d)))
    coef[5, 1] = (a^2 * b * c) / ((a - d)^2 * (b - d) * (c - d) * d)
    # Tridiagonal Matrix components:
    coef[7, 1] = 1.0
    coef[8, 1] = (b*c*d)/((-a + b)*(a - c)*(a - d)) # beta

    # BOUNDARY SCHEME FOR I = 2
    a = h[2]
    b = a + h[3]
    c = b + h[4]
    d = c + h[5]
    e = h[1]
    # B matrix components:
    coef[1, 2] = -((a^2 * b * c * (2*a*b*c + 3*a*b*e + 3*a*c*e + 4*b*c*e + 4*a*e^2 + 5*b*e^2 + 5*c*e^2 + 6*e^3)) /
    (e * (a + e)^3 * (b + e)^2 * (c + e)^2))
    coef[2, 2] = (2*a*b*c - a*b*e - a*c*e - 2*b*c*e)/(a*b*c*e)
    coef[3, 2] = (b*c*e^2 * (6*a^3 - 5*a^2*b - 5*a^2*c + 4*a*b*c + 4*a^2*e - 3*a*b*e - 3*a*c*e + 2*b*c*e)) / 
    (a * (a - b)^2 * (a - c)^2 * (a + e)^3)
    coef[4, 2] =  -((a^2 * c * e^2) / ((a - b)^2 * b * (b - c) * (b + e)^2))
    coef[5, 2] = (a^2 * b * e^2) / ((a - c)^2 * (b - c) * c * (c + e)^2)
    # Tridiagonal Matrix components:
    coef[6, 2] = (a^2 * b * c) / ((a + e)^2 * (b + e) * (c + e)) # alpha
    coef[7, 2] = 1.0
    coef[8, 2] = (b * c * e^2) / ((a - b) * (a - c) * (a + e)^2) # beta

    #INTERNAL SCHEME I = 3, 4, ..., ny-2
    for i = 3:ny-2
        a = h[i]
        b = a + h[i+1]
        e = h[i-1]
        f = e + h[i-2]
        # B matrix components:
        coef[1, i] = -((a^2 * b * e^2) / ((e - f)^2 * f * (a + f)^2 * (b + f)))
        coef[2, i] = (a^2 * b * f * (3*a*b*e + 4*a*e^2 + 5*b*e^2 + 6*e^3 - 2*a*b*f - 3*a*e*f - 4*b*e*f - 5*e^2*f)) /
        (e * (a + e)^3 * (b + e)^2 * (e - f)^2)
        coef[3, i] = (a*b*e + 2*a*b*f - a*e*f - 2*b*e*f)/(a*b*e*f)
        coef[4, i] = -((b * e^2 * f * (6*a^3 - 5*a^2*b + 4*a^2*e - 3*a*b*e + 5*a^2*f - 4*a*b*f + 3*a*e*f - 2*b*e*f)) /
        (a * (a - b)^2 * (a + e)^3 * (a + f)^2))
        coef[5, i] = (a^2 * e^2 * f) / ((a - b)^2 * b * (b + e)^2 * (b + f))
        # Tridiagonal Matrix components:
        coef[6, i] = -((a^2 * b * f) / ((a + e)^2 * (b + e) * (e - f))) # alpha
        coef[7, i] = 1.0
        coef[8, i] = -((b * e^2 * f) / ((a - b) * (a + e)^2 * (a + f))) # beta
    end

    # BOUNDARY SCHEME FOR I = ny-1
    e  = h[ny-2]
    f  = e + h[ny-3]
    gg = f + h[ny-4]
    a  = h[ny-1]
    # B matrix components:
    coef[1, ny-1] = -((a^2 * e^2 * f) / ((e - gg)^2 * (f - gg) * gg * (a + gg)^2))
    coef[2, ny-1] = (a^2 * e^2 * gg) / ((e - f)^2 * f * (a + f)^2 * (f - gg))
    coef[3, ny-1] = -((a^2 * f * gg * (4*a*e^2 + 6*e^3 - 3*a*e*f - 5*e^2*f - 3*a*e*gg - 5*e^2*gg + 2*a*f*gg + 4*e*f*gg)) /
    (e * (a + e)^3 * (e - f)^2 * (e - gg)^2))
    coef[4, ny-1] = (a*e*f + a*e*gg + 2*a*f*gg - 2*e*f*gg) / (a*e*f*gg)
    coef[5, ny-1] = (e^2*f*gg*(6*a^3 + 4*a^2*e + 5*a^2*f + 3*a*e*f + 5*a^2*gg + 3*a*e*gg + 4*a*f*gg + 2*e*f*gg))/(a*(a + e)^3*(a + f)^2*(a + gg)^2)
    # Tridiagonal Matrix components:
    coef[6, ny-1] = (a^2 * f * gg) / ((a + e)^2 * (e - f) * (e - gg)) # alpha
    coef[7, ny-1] = 1.0
    coef[8, ny-1] = (e^2 * f * gg) / ((a + e)^2 * (a + f) * (a + gg)) # beta

    # BOUNDARY SCHEME FOR I = ny
    e  = h[ny-1]
    f  = e + h[ny-2]
    gg = f + h[ny-3]
    hh = gg + h[ny-4]
    # B matrix components:
    coef[1, ny] = -((e^2 * f * gg) / ((e - hh)^2 * (f - hh) * (gg - hh) * hh))
    coef[2, ny] = (e^2 * f * hh) / ((e - gg)^2 * (f - gg) * gg * (gg - hh))
    coef[3, ny] = -((e^2 * gg * hh) / ((e - f)^2 * f * (f - gg) * (f - hh)))
    coef[4, ny] = (f*gg*hh*(5*e^3 - 4*e^2*f - 4*e^2*gg + 3*e*f*gg - 4*e^2*hh + 3*e*f*hh + 3*e*gg*hh - 2*f*gg*hh))/(e*(e - f)^2*(e - gg)^2*(e - hh)^2)
    coef[5, ny] = (e * f * gg + e * f * hh + e * gg * hh + 2 * f * gg * hh) / (e * f * gg * hh)
    # Tridiagonal Matrix components:
    coef[6, ny] = -((f * gg * hh) / ((e - f) * (e - gg) * (e - hh))) # alpha
    coef[7, ny] = 1.0
 
    # Reorganising the coeffs to save flops in the trid-solver (LU)
    for i = 2:ny
        coef[6, i] /= coef[7, i-1]
        coef[7, i] -= coef[6, i] * coef[8, i-1]
    end
    
    coef[7, :] .= 1.0 ./ coef[7, :]

    return coef
end


function compute_weights_6order_pade_uniform(Δ::T, nx::Int) where T<:Real
#=====================================================================================================#
# Compute weights for equally spaced coordinates for 6th order compact first derivative
#    Δ  = constant step size, i.e. spacing between functions
#    nx = number of function values, size of array f[1:nx]
#    
# The Pade-derviatives are computed 
#        A f' = B f 
#
#        A is tridiagonal (alpha/0, 1, beta/0)  while 
#        B is 5-banded denoted in each row by (AA, BB, CC, DD, EE)
#        
#    the values for each row of A & B for i = 1, ..., nx are stored in coef(8,nx):
#        coef[1, i] = AA
#        coef[2, i] = BB
#        coef[3, i] = CC
#        coef[4, i] = DD
#        coef[5, i] = EE
#        coef[6, i] = alpha or 0 for i = 1
#        coef[7, i] = 1
#        coef[8, i] = beta or 0 for i = nx    
#=====================================================================================================#
"""
    Arguments:
        Δ::T    : constant step size, i.e. spacing between functions
        nx::Int : number of function values, size of array f[1:nx]
    Returns:
        coef::Array{T,2} : 8 x nx array containing the coefficients for the Pade-derivative
"""  
    # assign array
    coef = zeros(T, 8, nx)

    # BOUNDARY SCHEME FOR I = 1
    # B matrix components:
    coef[1, 1] = -37.0/12.0
    coef[2, 1] = 2.0/3.0
    coef[3, 1] = 3.0
    coef[4, 1] = -2.0/3.0
    coef[5, 1] = 1.0/12.0
    # Tridiagonal Matrix components:
    coef[7, 1] = 1.0
    coef[8, 1] = 4.0 # beta
    
    # BOUNDARY SCHEME FOR I = 2
    # B matrix components:
    coef[1, 2] = -43.0/96.0
    coef[2, 2] = -5.0/6.0
    coef[3, 2] = 9.0/8.0
    coef[4, 2] = 1.0/6.0
    coef[5, 2] = -1.0/96.0
    # Tridiagonal Matrix components:
    coef[6, 2] = 1.0/8.0 # alpha
    coef[7, 2] = 1.0
    coef[8, 2] = 3.0/4.0 # beta
    
    # INTERNAL SCHEME For I = 3, 4, ..., NX-2
    # B matrix components:
    coef[1, 3:nx-2] .= -1.0/36.0
    coef[2, 3:nx-2] .= -7.0/9.0
    coef[3, 3:nx-2] .= 0.0
    coef[4, 3:nx-2] .= 7.0/9.0
    coef[5, 3:nx-2] .= 1.0/36.0
    # Tridiagonal Matrix components:
    coef[6, 3:nx-2] .= 1.0/3.0 # alpha
    coef[7, 3:nx-2] .= 1.0
    coef[8, 3:nx-2] .= 1.0/3.0 # beta

    # BOUNDARY SCHEME FOR I = NX-1
    # B matrix components:
    coef[1, nx-1] = 1.0/96.0
    coef[2, nx-1] = -1.0/6.0
    coef[3, nx-1] = -9.0/8.0
    coef[4, nx-1] = 5.0/6.0
    coef[5, nx-1] = 43.0/96.0
    # Tridiagonal Matrix components:
    coef[6, nx-1]  = 3.0/4.0 # alpha
    coef[7, nx-1]  = 1.0
    coef[8, nx-1]  = 1.0/8.0 # beta
    
    # BOUNDARY SCHEME FOR I = NX
    # B matrix components:
    coef[1, nx] = -1.0/12.0
    coef[2, nx] = 2.0/3.0
    coef[3, nx] = -3.0
    coef[4, nx] = -2.0/3.0
    coef[5, nx] = 37.0/12.0
    # Tridiagonal Matrix components:
    coef[6, nx] = 4.0 # alpha
    coef[7, nx] = 1.0
    
    #divide the B matrix components by the function spacing: Δ
    coef[1:5, :]  .= coef[1:5,:] ./ Δ

    # LU decomposition in Coef
    for i = 2:nx
        coef[6, i] /= coef[7, i-1]
        coef[7, i] -= coef[6, i] * coef[8, i-1]
    end
    coef[7, :] .= 1.0 ./ coef[7, :]
    
    return coef
end


function diff1d(f::AbstractVector{T}, coef::AbstractArray{T,2}, dfdx::Union{AbstractVector{T},Nothing}=nothing) where T<:Real
#=====================================================================================================================#
# 1d array: WORKHORSE !!!!
#           compute 1st derivative using compact 6th order FD for equally or unequally spaced function values 
#    
# ACHTUNG: this function can be used for either - equally spaced and unequally spaced grid points !!!
#   
#          A f' = B f
#    
# coef[8,nx] are computed using: 
#   - for equally spaced function values:    compute_weights_6order_pade_uniform(Δ, nx)
#   - for unequally spaced function values:  compute_weights_6order_pade_non_uniform(y)
#        
# f[1:n]    = array of size "n" containing equally/un-equally spaced f() values
#        
# dfdxᵢ[1:n] = derivative of f[1:n] at each sampling point where f is located - memory assigned inside this function
#=====================================================================================================================#
"""
    Arguments:
        f::AbstractVector{T}        : 1D array of size n containing the function values
        coef::AbstractArray{T,2}    : 8 x n array containing the coefficients for the Pade-derivative
        dfdx::AbstractVector{T}    : 1D array of size n containing the derivative of f[1:n] at each sampling point,
                                       if not passed in, memory is allocated inside this function
        
    Returns:
        dfdx::AbstractVector{T}  : 1D array of size n containing the derivative of f[1:n] at each sampling point
"""
    n = length(f)
    if dfdx === nothing
        dfdx = zeros(T, n)
    end

    dfdx[1]   = sum(coef[1:5, 1]   .* f[1:5])
    dfdx[2]   = sum(coef[1:5, 2]   .* f[1:5])
    dfdx[n-1] = sum(coef[1:5, n-1] .* f[n-4:n])
    dfdx[n]   = sum(coef[1:5, n]   .* f[n-4:n])
    for i = 3:n-2
        dfdx[i] = sum(coef[1:5, i] .* f[i-2:i+2])
    end
        
    # Resolution of tridiagonal: TDMA Algorithm. LU decomposition in Coeft
    for i = 2:n
        dfdx[i] -=  coef[6, i] * dfdx[i-1]
    end
    dfdx[n] *= coef[7, n]
    for i = n-1:-1:1
        dfdx[i] = (dfdx[i] - coef[8, i] * dfdx[i+1]) * coef[7, i]
    end
        
    return dfdx
end

#=========================================== End of Preliminary Pade Differentiation Functions =======================================#



#######################################################################################################################################
###################################################### UNIFORM PADE FINTE DIFFERENTIATION #############################################
#######################################################################################################################################

function compute_PadeFD(f::AbstractArray{T,1}, Δ::T, dfdx₁::Union{AbstractArray{T,1}, Nothing}=nothing) where T<:Real
#================================================================================================================================#
# Compute Pade FD derivative in the 1D for EQUALLY-SPACED points, separated by Δ of 1D array ....
#  - in this function the memory for the derivative is allocated within
#================================================================================================================================#
"""
    Arguments:
        Δ::T                        : constant step size, i.e. spacing between functions
        f::AbstractArray{T,1}       : 1D array of size n₁ containing the function values
        dfdx₁::AbstractArray{T,1}   : 1D array of size n₁ containing the derivative of f[1:n₁] at each sampling point,
                                       if not passed in, memory is allocated inside this function

    Returns:
        dfdx₁::AbstractArray{T,1} : 1D array of size n₁ containing the derivative of f[1:n₁] at each sampling point
"""
    n₁    = length(f)
    if dfdx₁ === nothing
        dfdx₁ = zeros(T, n₁)
    end  
    
    # compute compact pade coeffts for uniform grid:
    ld1x₁ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₁))
    
    # compute df/dx₁ ...
    dfdx₁ = diff1d(f, ld1x₁, dfdx₁)
    
    return dfdx₁
end
    
 
function compute_PadeFD(f::AbstractArray{T,2}, Δ::T, dfdxᵢ::Union{AbstractArray{T,2}, Nothing}=nothing; dim::Int=1) where T<:Real
#================================================================================================================================#
# Compute Pade FD derivative in the "dim" direction for EQUALLY-SPACED points, separated by Δ of 2D array ....
#  - in this function the memory for the derivative is passed as a parameter 
#================================================================================================================================#
"""
    Arguments:
        Δ::T                      : constant step size, i.e. spacing between functions
        f::AbstractArray{T,2}     : 2D array of size n₁ x n₂ containing the function values
        dfdxᵢ::AbstractArray{T,2} : 2D array of size n₁ x n₂ containing the derivative of f[1:n₁,1:n₂] at each sampling point,
                                       if not passed in, memory is allocated inside this function
        dimᵢ::Int                 : direction of differentiation (1 or 2)

    Returns:
        dfdx₁,₂::Array{T,2} : 2D array of size n₁ x n₂ containing the derivative of f[1:n₁,1:n₂] at each sampling point   
"""
    n₁, n₂ = size(f)
    if dfdxᵢ === nothing
        dfdxᵢ = zeros(T, n₁, n₂)
    end

    if dim == 1
        dfdx₁ = dfdxᵢ   
        # compute compact pade coeffts for uniform grid:
        ld1x₁ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₁))
        # compute df/dx₁ ...
        for j in 1:n₂
            dfdx₁[:,j] = diff1d(f[:,j], ld1x₁, dfdx₁[:,j])
        end
        return dfdx₁
    elseif dim == 2
        dfdx₂ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₂ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₂))
        # compute df/dx₂ ...
        for i in 1:n₁
            dfdx₂[i,:] = diff1d(f[i,:], ld1x₂, dfdx₂[i,:])
        end
        return dfdx₂
    else
        println("ACHTUNG: Array dimension not supported!!")
    end
end


function compute_PadeFD(f::AbstractArray{T,3}, Δ::T, dfdxᵢ::Union{AbstractArray{T,3}, Nothing}=nothing; dim::Int=1) where T<:Real
#================================================================================================================================#
# Compute Pade FD derivative in the "dim" direction for EQUALLY-SPACED points, separated by Δ of 3D array ....
#  - in this function the memory for the derivative is passed as a parameter 
#================================================================================================================================#
"""
    Arguments:
        Δ::T                      : constant step size, i.e. spacing between functions
        f::AbstractArray{T,3}     : 3D array of size n₁ x n₂ x n₃ containing the function values
        dfdxᵢ::AbstractArray{T,3} : 3D array of size n₁ x n₂ x n₃ containing the derivative of f[1:n₁,1:n₂,1:n₃] at each sampling point, 
                                       if not passed in, memory is allocated inside this function
        dim::Int                  : direction of differentiation (1, 2 or 3)

    Returns:
        dfdxᵢ::AbstractArray{T,3} : i ∈ (1,2,3) 3D array of size n₁ x n₂ x n₃ containing the derivative of f[1:n₁,1:n₂,1:n₃] wrt xᵢ at each sampling point

"""
    n₁, n₂, n₃ = size(f)
    if dfdxᵢ === nothing
        dfdxᵢ = zeros(T, n₁, n₂, n₃)
    end

    if dim == 1
        dfdx₁ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₁ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₁))
        # compute df/dx₁ ...
        for k in 1:n₃
            for j in 1:n₂
                dfdx₁[:,j,k] = diff1d(f[:,j,k], ld1x₁, dfdx₁[:,j,k])
            end
        end
        return dfdx₁
    elseif dim == 2
        dfdx₂ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₂ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₂))
        # compute df/dx₂ ...
        for k in 1:n₃
            for i in 1:n₁
                dfdx₂[i,:,k] = diff1d(f[i,:,k], ld1x₂, dfdx₂[i,:,k])
            end
        end
        return dfdx₂
    elseif dim == 3
        dfdx₃ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₃ = convert(Array{T}, compute_weights_6order_pade_uniform(Δ, n₃))
        # compute df/dx₃ ...
        for j in 1:n₂
            for i in 1:n₁
                dfdx₃[i,j,:] = diff1d(f[i,j,:], ld1x₃, dfdx₃[i,j,:])
            end
        end
        return dfdx₃
    else
        println("ACHTUNG: Array dimension not supported!!")
    end
end

#============================================== END UNIFORM PADE FINTE DIFFERENTIATION ===============================================#



#######################################################################################################################################
################################################# NON-UNIFORM PADE FINTE DIFFERENTIATION ##############################################
#######################################################################################################################################

function compute_PadeFD(f::AbstractArray{T,1}, xᵢ::AbstractArray{T,1}, dfdxᵢ::Union{AbstractArray{T,1}, Nothing}=nothing) where T<:Real
#======================================================================================================================================#
# Compute Pade FD derivative in the 1D direction for UNEQUALLY-SPACED points of 1D array f (x₁) ....
#  - in this function the memory for the derivative is passed in as a paramter 
#======================================================================================================================================#
"""
    Arguments:
        xᵢ::Array{T,1}     : 1D array of size n₁ containing the unequally spaced grid points
        f::Array{T,1}      : 1D array of size n₁ containing the function values
        dfdxᵢ::Array{T,1}  : 1D array of size n₁ containing the derivative of f[1:n₁] at each sampling point

    Returns:
        dfdx₁::Array{T,1}  : 1D array of size n₁ containing the derivative of f[1:n₁] at each sampling point    
""" 
    n = size(f,1)
    if dfdxᵢ === nothing
        dfdx₁ = zeros(T, n)
    end
    
    dfdx₁ = dfdxᵢ
    # compute compact pade coeffts for uniform grid:
    ld1x₁ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
    # compute df/dx₁ ...
    dfdx₁ = diff1d(f, ld1x₁, dfdx₁)
    return dfdx₁
end


function compute_PadeFD(f::AbstractArray{T,2}, xᵢ::AbstractArray{T,1}, dfdxᵢ::Union{AbstractArray{T,2}, Nothing}=nothing; 
                            dim::Int=1) where T<:Real
#==============================================================================================================================#
# Compute Pade FD derivative in the "dimᵢ" direction for UNEQUALLY-SPACED points of 2D array f (x₁,x₂) ....
#  - in this function the memory for the derivative is passed in as a paramter 
#==============================================================================================================================#
"""
    Arguments:
        xᵢ::AbstractArray{T,1}     : 1D array of size nᵢ ∈ (n₁, n₂) containing the unequally spaced grid points
        f::AbstractArray{T,2}      : 2D array of size n₁ x n₂ containing the function values
        dfdxᵢ::AbstractArray{T,2}  : 2D array of size n₁ x n₂ containing the derivative of f[1:n₁,1:n₂] at each sampling point, 
                                       if not passed in, memory is allocated inside this function
        dim::Int                   : direction of differentiation (1 or 2)

    Returns:
        dfdxᵢ::Array{T,2}  : i ∈{1,2}, 2D array of size n₁ x n₂ containing the derivative of f[1:n₁,1:n₂]  at each sampling point
"""
    n₁, n₂ = size(f)
    if dfdxᵢ === nothing
        dfdxᵢ = zeros(T, n₁, n₂)
    end
     
    if dim == 1
        dfdx₁ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₁ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
        # compute df/dx₁ ...
        for j in 1:n₂
            dfdx₁[:,j] = diff1d(f[:,j], ld1x₁, dfdx₁[:,j])
        end
        return dfdx₁
    elseif dim == 2
        dfdx₂ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₂ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
        # compute df/dx₂ ...
        for i in 1:n₁
            dfdx₂[i,:] = diff1d(f[i,:], ld1x₂, dfdx₂[i,:])
        end
        return dfdx₂
    else
        println("ACHTUNG: Array dimension not supported!!")
    end
end


function compute_PadeFD(f::AbstractArray{T,3}, xᵢ::AbstractArray{T,1}, dfdxᵢ::Union{AbstractArray{T,3}, Nothing}=nothing; 
                            dim::Int=1) where T<:Real
#==============================================================================================================================#
# Compute Pade FD derivative in the "dim" direction for UNEQUALLY-SPACED points of 3D array f (x₁,x₂,x₃) ....
#  - in this function the memory for the derivative is passed in as a paramter 
#==============================================================================================================================#
"""
    Arguments:
        xᵢ::Array{T,1}     : 1D array of size nᵢ ∈ (n₁, n₂, n₃) containing the unequally spaced grid points
        f::Array{T,3}      : 3D array of size n₁ x n₂ x n₃ containing the function values
        dfdxᵢ::Array{T,3}  : 3D array of size n₁ x n₂ x n₃ containing the derivative of f[1:n₁,1:n₂,1:n₃] at each sampling point
        dim::Int           : direction of differentiation (1, 2 or 3)

    Returns:
        dfdxᵢ::Array{T,3}  : i ∈ (1,2,3) 3D array of size n₁ x n₂ x n₃ containing the derivative of f[1:n₁,1:n₂,1:n₃] wrt xᵢ at each sampling point
"""
    n₁, n₂, n₃ = size(f)
    if dfdxᵢ === nothing
        dfdxᵢ = zeros(T, n₁, n₂, n₃)
    end

    if dim == 1
        dfdx₁ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₁ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
        # compute df/dx₁ ...
        for k in 1:n₃
            for j in 1:n₂
                dfdx₁[:,j,k] = diff1d(f[:,j,k], ld1x₁, dfdx₁[:,j,k])
            end
        end
        return dfdx₁
    elseif dim == 2
        dfdx₂ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₂ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
        # compute df/dx₂ ...
        for k in 1:n₃
            for i in 1:n₁
                dfdx₂[i,:,k] = diff1d(f[i,:,k], ld1x₂, dfdx₂[i,:,k])
            end
        end
        return dfdx₂
    elseif dim == 3
        dfdx₃ = dfdxᵢ
        # compute compact pade coeffts for uniform grid:
        ld1x₃ = convert(Array{T}, compute_weights_6order_pade_non_uniform(xᵢ))
        # compute df/dx₃ ...
        for j in 1:n₂
            for i in 1:n₁
                dfdx₃[i,j,:] = diff1d(f[i,j,:], ld1x₃, dfdx₃[i,j,:])
            end
        end
        return dfdx₃
    else
        println("ACHTUNG: Array dimension not supported!!")
    end
end 

#============================================== END NON-UNIFORM PADE FINTE DIFFERENTIATION =====================================#




################################################################################################################
#============================================= Test Functions =================================================#
################################################################################################################
function UnitFunction(x)
    """
    Unit function
    """
    f = ones(size(x))
end
    

function testDiffx_Uniform(nx::Int, freq = 1.0)
"""
    Test the compact 6th order Pade FD scheme for uniform grid

        Arguments:
            nx::Int     : number of grid points
            freq::Float : oscillatory frequency of the trig function to be differentiated

        Returns:    
            sum(err1^2) : sum of the squared error between the exact and numerical derivative
"""
    Δ = 1.0/nx
    x = 0.0:Δ:1.0
    dx = x[2] - x[1]
    NX = length(x)
    
    # function
    f                = sin.(2π*freq .* x)
    dfdx_exact       = 2π*freq * cos.(2π*freq .* x)
    dfdx_uniform     = zeros(NX)
    
    # compute compact pade coeffts for uniform grid:
    coef_uniform = compute_weights_6order_pade_uniform(Δ, NX) 

    # compute the derivative using uniform grid coeffts:
    diff1d(f, coef_uniform, dfdx_uniform)

    # Plot derivatives
    fig1 = plot(x, dfdx_exact, label="Exact")
    plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid")
    display(fig1)
    
    # Plot Error
    err1 = abs.(dfdx_uniform .- dfdx_exact) .+ 1e-30
    fig2 = plot(x, err1, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Uniform Grid",ylim=(1e-20, 1e-1))
    display(fig2)

    return sum(err1.^2)
end


function testDiffx(nx::Int; freq = 1.0)
"""
    Test the compact 6th order Pade FD scheme for uniform and non-uniform grid

        Arguments:
            nx::Int     : number of grid points
            freq::Float : oscillatory frequency of the trig function to be differentiated

        Returns:    
            sum(err1^2) : sum of the squared error between the exact and numerical derivative - uniform grid
            sum(err2^2) : sum of the squared error between the exact and numerical derivative - non-uniform grid
"""
    Δ = 1.0/nx
    x = 0.0:Δ:1.0
    dx = x[2] - x[1]
    NX = length(x)
    
    # function
    f                = sin.(2π*freq .* x)
    dfdx_exact       = 2π*freq * cos.(2π*freq .* x)
    dfdx_uniform     = zeros(NX)
    dfdx_non_uniform = zeros(NX)
    
    # compute compact pade coeffts for uniform grid:
    coef_uniform = compute_weights_6order_pade_uniform(Δ, NX) 

    # compute compact pade coeffts for non-uniform grid:
    coef_non_unform = compute_weights_6order_pade_non_uniform(x)

    # compute the derivative using uniform grid coeffts:
    diff1d(f, coef_uniform, dfdx_uniform)

    # compute the derivative using non-uniform grid coeffts:
    diff1d(f, coef_non_unform, dfdx_non_uniform)

    # Plot derivatives
    fig1 = plot(x, dfdx_exact, label="Exact")
    plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid")
    plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Grid")
    display(fig1)
    
    # Plot Error
    err1 = abs.(dfdx_uniform .- dfdx_exact) .+ 1e-30
    err2 = abs.(dfdx_non_uniform .- dfdx_exact) .+ 1e-30
    fig2 = plot(x, err1, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Uniform Grid",ylim=(1e-20, 1))
    plot!(fig2, x, err2, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Non-Uniform Grid", ylim=(1e-20, 1))
    display(fig2)

    println("max error uniform grid     ", maximum(abs.(err1)))
    println("max error non-uniform grid ", maximum(abs.(err2)))
    return norm(err1), norm(err2)
end


function testDiffx_NonUniform(nx::Int, freq = 1.0)
"""
    Test the compact 6th order Pade FD scheme for non-uniform grid

        Arguments:
            nx::Int     : number of grid points
            freq::Float : oscillatory frequency of the trig function to be differentiated

        Returns:    
            sum(err1^2) : sum of the squared error between the exact and numerical derivative - non-uniform grid    
"""
    # generate random non-uniform grid
    y                = zeros(nx)
    y[1]             = 0
    y[end]           = 1
    x                = rand(Uniform(eps(),1-eps()), 1,nx-2)
    y[2:nx-1]        = x
    x                = sort!(y)
    # compute function at random points
    f                = sin.(2π*freq .* x)
    dfdx_exact       = 2π*freq * cos.(2π*freq .* x)
    dfdx_non_uniform = zeros(nx)
    # compute compact pade coeffts for non-uniform grid:
    coef_non_unform  = compute_weights_6order_pade_non_uniform(x)
    # compute the derivative using non-uniform grid coeffts:
    diff1d(f, coef_non_unform, dfdx_non_uniform)
    # Plot derivatives:
    fig2 = plot(x, dfdx_exact, label="Exact")
    plot!(fig2, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Grid")
    display(fig2)
    # Plot Error
    fig3 = plot(x, abs.(dfdx_non_uniform .- dfdx_exact) .+ 1e-30, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Non-Uniform Grid", ylim=(1e-20, 1e-1))
    display(fig3)
    L2 = abs.(dfdx_non_uniform .- dfdx_exact) .+ 1e-30
    return sum(L2.^2)
end


function Plot_L2_vs_Grid_UniformGrid(M, norm = false)
"""
    Plot the L2 norm of the error for uniform and non-uniform grid

        Arguments:
            M::Int     : number of grid points
            norm::Bool : normalize the L2 norm by the number of grid points

        Returns:    
            fig3       : plot of the L2 norm of the error for uniform and non-uniform grid
"""
    NPTS = zeros(Int, M)
    L2_1 = zeros(M)
    L2_2 = zeros(M)
    N = 1:M
    for i in N
        NPTS[i]          = 5*2^i
        L2_1[i], L2_2[i] = testDiffx(NPTS[i])
        if(norm)
            L2_1[i] /= NPTS[i]
            L2_2[i] /= NPTS[i]
        end
    end

    fig3 = plot(NPTS, L2_1, xscale=:log10, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Uniform Grid", ylim=(1e-25, 1e-1))
    plot!(fig3, NPTS, L2_2, xscale=:log10, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Non-Uniform Grid", ylim=(1e-25, 1e-1))
    display(fig3)
end


function Plot_L2_vs_Grid_NonUniformGrid(M, norm = false)
"""
    Plot the L2 norm of the error for non-uniform grid

        Arguments:
            M::Int     : number of grid points
            norm::Bool : normalize the L2 norm by the number of grid points

        Returns:    
            fig4       : plot of the L2 norm of the error for non-uniform grid
"""
    NPTS = zeros(Int, M)
    L2   = zeros(M)
    N = 1:M
    for i in N
        NPTS[i] = 5*2^i
        if(norm)
            L2[i]   = testDiffx_NonUniform(NPTS[i])/NPTS[i]
        else
            L2[i]   = testDiffx_NonUniform(NPTS[i]) 
        end
    end
    fig4 = plot(NPTS, L2, xscale=:log10, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Non-Uniform Grid-random points")
    display(fig4)
end


function Plot_L2_vs_Grid_UniformGrid_Freq(N::Int)
"""

"""
    Nfreq = Int(N/8)
    freq  = 1:1:Nfreq
    L2_1  = zeros(Nfreq)
    L2_2  = zeros(Nfreq)
    
    for i in freq
        L2_1[i], L2_2[i] = testDiffx(N, i)
    end
    fig5 = plot(freq/Nfreq, L2_1, xscale=:log10, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Uniform Grid")
    plot!(fig5, freq/Nfreq, L2_2, xscale=:log10, yscale=:log10, minorgrid=true, seriestype=:scatter, label="Non-Uniform Grid")
    display(fig5)
end

#========================================================================================================#
# End of Functions 
#========================================================================================================#



# #=========================================================================================================#
# # main function to run TEST functions ...
# #=========================================================================================================#

# function main()
   
#     test = 2

#     if test == 1
#         # test Pade v. FFT differentiation
#         N  = 126 # number of spaces = number of total grid points - 1
#         λ  = 10
#         L  = 2π * λ
#         Δ  = L/N
#         x  = (0.0:Δ:L)[1:end-1]
    
#         # functions
#         f                 = sin.(x)
#         dfdx_exact        = cos.(x)
#         dfdx_uniform      = similar(dfdx_exact)
#         dfdx_non_uniform  = similar(dfdx_exact)
   
#         # compute compact pade coeffts for uniform grid:
#         coef_uniform = compute_weights_6order_pade_uniform(Δ, length(x))
#         # compute the derivative using uniform grid coeffts:
#         diff1d(f, coef_uniform, dfdx_uniform) 

#         # compute compact pade coeffts for non-uniform grid:
#         coef_non_unform = compute_weights_6order_pade_non_uniform(collect(x))
#         # compute the derivative using non-uniform grid coeffts:
#         diff1d(f, coef_non_unform, dfdx_non_uniform)

#         # compute the derivative using FFT:
#         dfdx_fft = compute_FT_Diff_FromPhysical(L, f)

#         # Plot derivatives
#         fig1 = plot(x, dfdx_exact, label="Exact")
#         plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid Pade")
#         plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Pade")
#         plot!(fig1, x, dfdx_fft, seriestype=:scatter, label="FFT")
#         display(fig1)

#     elseif test == 2
#         # do a 3D array test
#         n₁, n₂, n₃ =180, 90, 270
#         f = zeros(Real, n₁, n₂, n₃)

#         λ₁  = 2.0
#         L₁  = 2π * λ₁
#         Δ₁  = L₁/n₁
#         x₁  = (0.0:Δ₁:L₁)[1:end-1]

#         λ₂  = 1.0
#         L₂  = 2π * λ₂
#         Δ₂  = L₂/n₂
#         x₂  = (0.0:Δ₂:L₂)[1:end-1]

#         λ₃  = 3.0
#         L₃  = 2π * λ₃
#         Δ₃  = L₃/n₃
#         x₃  = (0.0:Δ₃:L₃)[1:end-1]

#         # Create a 3D grid using broadcasting
#         X₁ = reshape(x₁, n₁, 1, 1)
#         X₂ = reshape(x₂, 1, n₂, 1)
#         X₃ = reshape(x₃, 1, 1, n₃)

#         # Function
#         f     = sin.(X₁) .* cos.(X₂) .* sin.(X₃);
#         dfdx₁ = cos.(X₁) .* cos.(X₂) .* sin.(X₃);
#         dfdx₂ = sin.(X₁) .* (-sin.(X₂)) .* sin.(X₃);
#         dfdx₃ = sin.(X₁) .* cos.(X₂) .* cos.(X₃);

#         # compute the derivative using uniform grid coeffts:
#         dfdx₁_uniform = compute_PadeFD(f, Δ₁, dim=1);
#         dfdx₂_uniform = compute_PadeFD(f, Δ₂, dim=2);
#         dfdx₃_uniform = compute_PadeFD(f, Δ₃, dim=3);
#         # compute the derivative using non-uniform grid coeffts:
#         dfdx₁_non_uniform = compute_PadeFD(f, collect(x₁), dim=1);
#         dfdx₂_non_uniform = compute_PadeFD(f, collect(x₂), dim=2);
#         dfdx₃_non_uniform = compute_PadeFD(f, collect(x₃), dim=3);

#         # compute error of derivatives for uniform derivatives ...
#         println("norm error for uniform grid derivative wrt x₁    ", norm(dfdx₁ .- dfdx₁_uniform))
#         println("norm error for uniform grid derivative wrt x₂    ", norm(dfdx₂ .- dfdx₂_uniform))
#         println("norm error for uniform grid derivative wrt x₃    ", norm(dfdx₃ .- dfdx₃_uniform))

#         # compute error of derivatives for non-uniform derivatives ...
#         println("norm error for non-uniform grid derivative wrt x₁ ", norm(dfdx₁ .- dfdx₁_non_uniform))
#         println("norm error for non-uniform grid derivative wrt x₂ ", norm(dfdx₂ .- dfdx₂_non_uniform))
#         println("norm error for non-uniform grid derivative wrt x₃ ", norm(dfdx₃ .- dfdx₃_non_uniform))

#     elseif test == 3
#         # test Pade v. FFT differentiation
#         N  = 126 # number of spaces = number of total grid points - 1
#         λ  = 10
#         L  = 2π * λ
#         Δ  = L/N
#         x  = (0.0:Δ:L)[1:end-1]
    
#         # functions
#         f                 = sin.(x)
#         dfdx_exact        = cos.(x)
#         dfdx_uniform      = similar(dfdx_exact)
#         dfdx_non_uniform  = similar(dfdx_exact)
   
#         # compute the derivative using uniform grid coeffts:
#         dfdx_uniform = compute_PadeFD(f, Δ, dfdx_uniform)

#         # compute the derivative using non-uniform grid coeffts:
#         dfdx_non_uniform = compute_PadeFD(f, collect(x), dfdx_non_uniform)

#         # Plot derivatives
#         fig1 = plot(x, dfdx_exact, label="Exact")
#         plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid Pade")
#         plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Pade")
#         display(fig1)
#     else
#         # !!! ACHTUNG: for these high order scheme there is minimum number of points required, i.e. N > 10 !!!!
#         M    = 10
#         Plot_L2_vs_Grid_UniformGrid(M)
#         Plot_L2_vs_Grid_NonUniformGrid(M)
#     end
# end

# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end