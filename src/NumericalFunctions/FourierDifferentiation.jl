#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               FourierDifferentiation.jl
#                               -------------------------
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 14-10-2024
#    updated   : 05-05-2025
#    
#    This code contains functions to compute::
#        1. Fourier differentiation
#       
#========================================================================================================#


#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using LinearAlgebra
using Distributions
using FFTW


#################################################################################################################
########################################## FOURIER-BASED DIFFERENTIATION ########################################
#################################################################################################################


function compute_FTD(f::AbstractArray{T,1}, L::T) where T<:Real
#=======================================================================================================#
# Compute the derivative of a real function f(n) using the RFFT:
#    L         = physical length of the domain of interest
#    f[n]      = 1D array holding n elements @ physical equally spaced points
#    df[n]     = 1D array holding n elements of f'() @ physical equally spaced points
#        
#    wavenumber array:  
#    k = (2 \pi)/L * [0, 1, ..., n/2] 
#
#    dfdx = IFT[ FT[df] ] = IFT[ im * k * FT[f] ]
#       
#    ACHTUNG: No scale factor when using rfft and irfft in FFTW  !!!!
#=======================================================================================================#
"""
    Arguments:
        L::T                  : physical length of the domain
        f::AbstractArray{T,1} : 1D array of size n containing the function values

    Returns:
        dfdx::Array{T,1} : 1D array of size n containing the derivative of f[1:n] at each sampling point     
"""
    n = size(f, 1)
    # compute wavenumber array
    k    = (2π / L) * collect(0:floor(Int, div(n,2)))
    # Compute the real-input FFT
    F    = rfft(f)
    # compute the FT(df/dx) in Fourier space
    dF   = (im * k) .* F
    # Compute the inverse real-input FFT of FT(df/dx)
    dfdx = irfft(dF, n)   
    return convert(Array{T}, dfdx)
end


function compute_FTD(f::AbstractArray{T,2}, L::T ; dim::Int = 1) where T<:Real
#=======================================================================================================#
# Compute the derivative of a real function f(n₁, n₂) using the RFFT:
#    L             = physical length of the domain of interest - along direction of the derivative
#    f[n₁, n₂]     = 2D array holding (n₁, n₂) elements @ physical equally spaced points
#    df[n₁, n₂]    = 2D array holding (n₁, n₂) elements of f'() @ physical equally spaced points
#        
#    wavenumber array:  
#    kᵢ = (2 \pi)/Lᵢ * [0, 1, ..., nᵢ/2] 
#
#    dfdxᵢ = IFT[ FT[df] ] = IFT[ im * k * FT[f] ]
#       
#    ACHTUNG: No scale factor when using rfft and irfft in FFTW  !!!!
#=======================================================================================================#
"""
    Arguments:
        L::T                  : physical length of the domain along direction of the derivative, L ∈ (L₁, L₂)
        f::AbstractArray{T,2} : 2D array of size n₁ x n₂ containing the function values
        dim::Int              : direction of differentiation (1 or 2)

    Returns:
        dfdxᵢ::Array{T,2} : i ∈ (1,2) 2D array of size n₁ x n₂ containing the derivative of f[1:n₁,1:n₂] wrt xᵢ at each sampling point     
"""
    (n₁, n₂) = size(f)
    if dim == 1
        # compute wavenumber array
        k     = (2π / L) * collect(0:floor(Int, div(n₁,2)))
        # generate a 2D array to allow element by element multiplication
        k₁    = (reshape(k, length(k), 1) .* ones(1, n₂))
        # Compute the real-input FFT
        F     = rfft(f, 1)
        # compute the FT(df/dx₁) in Fourier space
        dF    = (im * k₁) .* F
        # Compute the inverse real-input FFT of FT(df/dx₁)
        dfdxᵢ = irfft(dF, n₁, 1)   
        return convert(Array{T}, dfdxᵢ)
    elseif dim == 2
        # compute wavenumber array
        k     = (2π / L) * collect(0:floor(Int, div(n₂,2)))
        # generate a 2D array to allow element by element multiplication
        k₂    = reshape(k, 1, length(k)) .* ones(n₁, 1)
        # Compute the real-input FFT
        F     = rfft(f, 2)
        # compute the FT(df/dx₂) in Fourier space
        dF    = (im * k₂) .* F
        # Compute the inverse real-input FFT of FT(df/dx₂)
        dfdxᵢ = irfft(dF, n₂, 2)   
        return convert(Array{T}, dfdxᵢ)
    else
        println("ACHTUNG: Array dimension not supported!!")
        return nothing
    end
end


function compute_FTD(f::AbstractArray{T,3}, L::T ; dim::Int = 1) where T<:Real
#=======================================================================================================#
# Compute the derivative of a real function f(n₁, n₂, n₃) using the RFFT:
#    L              = physical length of the domain of interest - along direction of the derivative
#    f[n₁, n₂, n₃]  = 3D array holding (n₁, n₂, n₃) elements @ pyhsical equally spaced points
#    df[n₁, n₂, n₃] = 3D array holding (n₁, n₂, n₃) elements of f'() @ pyhsical equally spaced points
#        
#    wavemenber array:  
#    kᵢ = (2 \pi)/(Lᵢ * [0, 1, ..., nᵢ/2] 
#
#    dfdxᵢ = IFT[ FT[df] ] = IFT[ im * k * FT[f] ]
#       
#    ACHTUNG: No scale factor when using rfft and irfft in FFTW  !!!!
#=======================================================================================================#
"""
    Arguments:
        L::T                  : physical length of the domain along direction of the derivative, L ∈ (L₁, L₂, L₃)
        f::AbstractArray{T,3} : 3D array of size n₁ x n₂ x n₃ containing the function values
        dim::Int              : direction of differentiation (1, 2 or 3)

    Returns:
        dfdxᵢ::Array{T,3} : i ∈ (1,2,3) 3D array of size n₁ x n₂ x n₃ containing the derivative of f[1:n₁,1:n₂,1:n₃] wrt xᵢ at each sampling point     
"""
    (n₁, n₂, n₃) = size(f)
    if dim == 1
        # compute wavenumber array
        k     = (2π / L) * collect(0:floor(Int, div(n₁,2)))
        # generate a 3D array to allow element by element multiplication
        k₁    = (reshape(k, length(k), 1, 1) .* ones(1, n₂, n₃))
        # Compute the real-input FFT
        F     = rfft(f, 1)
        # compute the FT(df/dx₁) in Fourier space
        dF    = (im * k₁) .* F
        # Compute the inverse real-input FFT of FT(df/dx₁)
        dfdxᵢ = irfft(dF, n₁, 1)   
        return convert(Array{T}, dfdxᵢ)
    elseif dim == 2
        # compute wavenumber array
        k     = (2π / L) * collect(0:floor(Int, div(n₂,2)))
        # generate a 3D array to allow element by element multiplication
        k₂    = reshape(k, 1, length(k), 1) .* ones(n₁, 1, n₃)
        # Compute the real-input FFT
        F     = rfft(f, 2)
        # compute the FT(df/dx₂) in Fourier space
        dF    = (im * k₂) .* F
        # Compute the inverse real-input FFT of FT(df/dx₂)
        dfdxᵢ = irfft(dF, n₂, 2)   
        return convert(Array{T}, dfdxᵢ)
    elseif dim == 3
        # compute wavenumber array
        k     = (2π / L) * collect(0:floor(Int, div(n₃,2)))
        # generate a 3D array to allow element by element multiplication
        k₃    = reshape(k, 1, 1, length(k)) .* ones(n₁, n₂, 1)
        # Compute the real-input FFT
        F     = rfft(f, 3)
        # compute the FT(df/dx₃) in Fourier space
        dF    = (im * k₃) .* F
        # Compute the inverse real-input FFT of FT(df/dx₃)
        dfdxᵢ = irfft(dF, n₃, 3)   
        return convert(Array{T}, dfdxᵢ)
    else
        println("ACHTUNG: Array dimension not supported!!")
        return nothing
    end
end
    

function compute_FT_Diff_FromFourierModes(F::AbstractArray{T,1}, L::T)  where T<:Real
#===========================================================================================================
    Compute the derivative of a function f() using the FFT:
        L = physical length of the domain
        F[M] = 1D array holding M (total = (all real) + (all imag)) elements of (real, imag; real, imag, ...) 
                FT of a real function f which has N = M-2, equally spaced elements
        df[M-2] = output derivative real array of size (M-2)
        
        ACHTUNG: FT[f] has M/2 = N/2-1 complex Fourier modes 
            
        k = (2 \pi)/L * [0, 1, ..., N/2] = (2 \pi)/L * [0, 1, ..., M/2-1]

        dfdx = IFT[ FT[df/dx] ] = IFT[ im * k * F ]
        
        ACHTUNG: assuming that F comes directly from the TBL h5 file, if you are computing physical -> 
                    Fourier first, check that (M-2) scale factor still applies !!!!
==========================================================================================================#
"""
    Arguments:
        L::T         : physical length of the domain
        F::AbstractArray{T,1}: 1D array holding M (total = (all real) + (all imag)) elements of (real, imag; real, imag, ...) 
                       FT of a real function f which has N = M-2, equally spaced elements
    Returns:
        df::Array{T,1}: output derivative real array of size (M-2)
"""
    # compute the required array sizes and number elements, etc.
    M = F.size[1]
    N = M - 2
    # compute wavenumber array k (2 \pi)/L * [0, 1, ..., M/2-1]
    k = (2π / L) * collect(0:floor(Int, div(M, 2)))    
    # allocate arrays ....
    fou     = zeros(Complex{Float32}, div(M, 2))
    # convert F -> complex:
    fou[:] .= F[1:2:end] + im * F[2:2:end]
    # compute im * k * F:
    fou     = im * k .* fou
    # compute dfdx = IFT[ im * k * F ]
    df    = FFTW.irfft(fou, M-2)*(M-2) # Note the factor (M-2) may not apply see warning above !!!
    return convert(Array{T}, df)
end


function compute_FT_Diff_FromFourierModes(fou::AbstractArray{T,1}, L)  where T<:Complex
#===========================================================================================================
    Compute the derivative of a function f() using the FFT:
        L = physical length of the domain
        fou[M/2] = 1D complex array holding M/2 complex modes 
                    FT of a real function f which has N = M-2, equally spaced elements
        df[M-2] = output derivative real array of size (M-2)
        
        ACHTUNG: FT[f] has M/2 = N/2-1 complex Fourier modes 
            
        k = (2 \pi)/L * [0, 1, ..., N/2] = (2 \pi)/L * [0, 1, ..., M/2-1]

        dfdx = IFT[ FT[df/dx] ] = IFT[ im * k * F ]
        
        ACHTUNG: assuming that F comes directly from the TBL h5 file, if you are computing physical -> 
                    Fourier first, check that (M-2) scale factor still applies !!!!
==========================================================================================================#
"""
    Arguments:
        L::T            : physical length of the domain
        fou::Array{T,1} : 1D complex array holding M/2 complex modes 
                          FT of a real function f which has N = M-2, equally spaced elements
    Returns:
        df::Array{T,1}: output derivative real array of size (M-2)  
"""
    # compute the required array sizes and number elements, etc.
    M   = 2*fou.size[1]
    N   = M - 2
    # compute wavenumber array k (2 \pi)/L * [0, 1, ..., M/2-1]
    k   = (2π / L) * collect(0:floor(Int, div(M, 2)))    
    # compute im * k * F:
    fou = (im * k) .* fou
    # compute dfdx = IFT[ im * k * F ]
    df  = FFTW.irfft(fou, M-2)*(M-2) # Note the factor (M-2) may not apply see warning above !!!
    return df
end


function compute_FT_Diff_FromPhysical(f::AbstractArray{T,1}, L::T) where T<:Real
#===================================================================================
    Compute the derivative of a function f() using the RFFT:
        L     = physical length of the domain
        f[N]  = 1D array holding N elements of f()  @ pyhsical equally spaced points
        df[N] = 1D array holding N elements of f'() @ pyhsical equally spaced points
        
        wavemenber array:  
        k = (2 \pi)/L * [0, 1, ..., N/2] 

        df = IFT[ FT[df] ] = IFT[ im * k * FT[f] ]
        
        ACHTUNG: No scale factor when using rfft and irfft in FFTW  !!!!
====================================================================================#
"""
    Arguments:
        L::T         : physical length of the domain
        f::AbstractArray{T,1}: 1D array holding N elements of f() @ pyhsical equally spaced points

    Returns:
        df::Array{T,1}: 1D array holding N elements of f'() @ pyhsical equally spaced points
"""
    # compute the required array size and number elements, etc.
    N  = size(f)
    # compute wavenumber array
    k  = (2π / L) * collect(0:floor(Int, div(N,2)))
    # Compute the real-input FFT
    F  = rfft(f)
    # compute the FT(df/dx)
    dF = (im * k) .* F
    # Compute the inverse real-input FFT of FT(df/dx)
    df = irfft(dF, N)   
    return df
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
#         # FFT differentiation
#         N  = 126 # number of spaces = number of total grid points - 1
#         λ  = 10
#         L  = 2π * λ
#         Δ  = L/N
#         x  = (0.0:Δ:L)[1:end-1]
    
#         # functions
#         f                 = sin.(x)
#         dfdx_exact        = cos.(x)

#         # compute the derivative using FFT:
#         dfdx_fft = compute_FT_Diff_FromPhysical(f, L)

#         # Plot derivatives
#         fig1 = plot(x, dfdx_exact, label="Exact")
#         plot!(fig1, x, dfdx_fft, seriestype=:scatter, label="FFT")
#         display(fig1)

#     elseif test == 2
#         # do a 3D array test
#         n₁, n₂, n₃ = 128, 64, 96
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
#         X₁ = reshape(x₁, n₁, 1, 1);
#         X₂ = reshape(x₂, 1, n₂, 1);
#         X₃ = reshape(x₃, 1, 1, n₃);

#         # Function
#         f     = sin.(X₁) .* cos.(X₂) .* sin.(X₃);
#         dfdx₁ = cos.(X₁) .* cos.(X₂) .* sin.(X₃);
#         dfdx₂ = sin.(X₁) .* (-sin.(X₂)) .* sin.(X₃);
#         dfdx₃ = sin.(X₁) .* cos.(X₂) .* cos.(X₃);

#         # compute the derivative using FFT:
#         dfdx₁_fft = compute_FTD(f, L₁, dim=1);
#         dfdx₂_fft = compute_FTD(f, L₂, dim=2);
#         dfdx₃_fft = compute_FTD(f, L₃, dim=3);

#         # compute error of derivatives for FFT derivatives ...
#         println("norm error for FFT grid derivative wrt x₁         ", norm(dfdx₁ .- dfdx₁_fft))
#         println("norm error for FFT grid derivative wrt x₂         ", norm(dfdx₂ .- dfdx₂_fft))
#         println("norm error for FFT grid derivative wrt x₃         ", norm(dfdx₃ .- dfdx₃_fft))

#     elseif test == 3
#         #  FFT differentiation
#         N  = 126 # number of spaces = number of total grid points - 1
#         λ  = 10
#         L  = 2π * λ
#         Δ  = L/N
#         x  = (0.0:Δ:L)[1:end-1]
    
#         # functions
#         f                 = sin.(x)
#         dfdx_exact        = cos.(x)
#         dfdx_non_uniform  = similar(dfdx_exact)

#         # compute the derivative using FFT:
#         dfdx_fft = compute_FT_Diff_FromPhysical(f, L)

#         # Plot derivatives
#         fig1 = plot(x, dfdx_exact, label="Exact")
#         plot!(fig1, x, dfdx_fft, seriestype=:scatter, label="FFT")
#         display(fig1)
#     end
# end

# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end