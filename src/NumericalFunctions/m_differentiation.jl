#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
# Code: COMPACT DIFFERENTIATION FUNCTIONS AND FOURIER DIFFERENTIATION FUNCTION
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 14-10-2024
#    updated   : 25-10-2024
#    
#    This code contains functions to compute::
#        1. multi-dimensional uniformly spaced 6th order Pade differentiation via PadeDifferentiation.jl
#        2. multi-dimensional uuneqaully spaced 6th order Pade differentiation via PadeDifferentiation.jl
#        3. Fourier differentiation via FourierDifferentiation.jl
#       
#========================================================================================================#


#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
include("PadeDifferentiation.jl")
include("FourierDifferentiation.jl")
include("FD-Differentiation.jl")


#=========================================================================================================#
# main function to run TEST functions ...
#=========================================================================================================#

function main()
   
    test = 2

    if test == 1
        # test Pade v. FFT differentiation
        N  = 126 # number of spaces = number of total grid points - 1
        λ  = 10
        L  = 2π * λ
        Δ  = L/N
        x  = (0.0:Δ:L)[1:end-1]
    
        # functions
        f                 = sin.(x)
        dfdx_exact        = cos.(x)
        dfdx_uniform      = similar(dfdx_exact)
        dfdx_non_uniform  = similar(dfdx_exact)
   
        # compute compact pade coeffts for uniform grid:
        coef_uniform = compute_weights_6order_pade_uniform(Δ, length(x))
        # compute the derivative using uniform grid coeffts:
        diff1d(f, coef_uniform, dfdx_uniform) 

        # compute compact pade coeffts for non-uniform grid:
        coef_non_unform = compute_weights_6order_pade_non_uniform(collect(x))
        # compute the derivative using non-uniform grid coeffts:
        diff1d(f, coef_non_unform, dfdx_non_uniform)

        # compute the derivative using FFT:
        dfdx_fft = compute_FT_Diff_FromPhysical(L, f)

        # Plot derivatives
        fig1 = plot(x, dfdx_exact, label="Exact")
        plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid Pade")
        plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Pade")
        plot!(fig1, x, dfdx_fft, seriestype=:scatter, label="FFT")
        display(fig1)

    elseif test == 2
        # do a 3D array test
        n₁, n₂, n₃ = 256, 64, 128
        f = zeros(Real, n₁, n₂, n₃)

        λ₁  = 2.0
        L₁  = 2π * λ₁
        Δ₁  = L₁/n₁
        x₁  = (0.0:Δ₁:L₁)[1:end-1]

        λ₂  = 1.0
        L₂  = 2π * λ₂
        Δ₂  = L₂/n₂
        x₂  = (0.0:Δ₂:L₂)[1:end-1]

        λ₃  = 3.0
        L₃  = 2π * λ₃
        Δ₃  = L₃/n₃
        x₃  = (0.0:Δ₃:L₃)[1:end-1]

        # Create a 3D grid using broadcasting
        X₁ = reshape(x₁, n₁, 1, 1)
        X₂ = reshape(x₂, 1, n₂, 1)
        X₃ = reshape(x₃, 1, 1, n₃)

        # Function
        f     = sin.(X₁) .* cos.(X₂) .* sin.(X₃)
        dfdx₁ = cos.(X₁) .* cos.(X₂) .* sin.(X₃)
        dfdx₂ = sin.(X₁) .* (-sin.(X₂)) .* sin.(X₃)
        dfdx₃ = sin.(X₁) .* cos.(X₂) .* cos.(X₃)

        # compute the derivative using uniform grid coeffts:
        dfdx₁_uniform = compute_PadeFD(f, Δ₁, dim=1)
        dfdx₂_uniform = compute_PadeFD(f, Δ₂, dim=2)
        dfdx₃_uniform = compute_PadeFD(f, Δ₃, dim=3)
        # compute the derivative using non-uniform grid coeffts:
        dfdx₁_non_uniform = compute_PadeFD(f, collect(x₁), dim=1)
        dfdx₂_non_uniform = compute_PadeFD(f, collect(x₂), dim=2)
        dfdx₃_non_uniform = compute_PadeFD(f, collect(x₃), dim=3)
        # compute the derivative using FFT:
        dfdx₁_fft = compute_FTD(f, L₁, dim=1)
        dfdx₂_fft = compute_FTD(f, L₂, dim=2)
        dfdx₃_fft = compute_FTD(f, L₃, dim=3)


        # compute error of derivatives for uniform derivatives ...
        println("norm error for uniform grid derivative wrt x₁    ", norm(dfdx₁ .- dfdx₁_uniform))
        println("norm error for uniform grid derivative wrt x₂    ", norm(dfdx₂ .- dfdx₂_uniform))
        println("norm error for uniform grid derivative wrt x₃    ", norm(dfdx₃ .- dfdx₃_uniform))

        # compute error of derivatives for non-uniform derivatives ...
        println("norm error for non-uniform grid derivative wrt x₁ ", norm(dfdx₁ .- dfdx₁_non_uniform))
        println("norm error for non-uniform grid derivative wrt x₂ ", norm(dfdx₂ .- dfdx₂_non_uniform))
        println("norm error for non-uniform grid derivative wrt x₃ ", norm(dfdx₃ .- dfdx₃_non_uniform))

        # compute error of derivatives for FFT derivatives ...
        println("norm error for FFT grid derivative wrt x₁         ", norm(dfdx₁ .- dfdx₁_fft))
        println("norm error for FFT grid derivative wrt x₂         ", norm(dfdx₂ .- dfdx₂_fft))
        println("norm error for FFT grid derivative wrt x₃         ", norm(dfdx₃ .- dfdx₃_fft))

    elseif test == 3
        # test Pade v. FFT differentiation
        N  = 32 # number of spaces = number of total grid points - 1
        λ  = 1
        L  = 2π * λ
        Δ  = L/N
        x  = (0.0:Δ:L)[1:end-1]
    
        # functions
        f                 = sin.(x)
        dfdx_exact        = cos.(x)
        dfdx_uniform      = similar(dfdx_exact)
        dfdx_non_uniform  = similar(dfdx_exact)
   
        # compute the derivative using uniform grid coeffts:
        dfdx_uniform = compute_PadeFD(f, Δ, dfdx_uniform)

        # compute the derivative using non-uniform grid coeffts:
        dfdx_non_uniform = compute_PadeFD(f, collect(x), dfdx_non_uniform)

        # compute using central difference unform grid
        dfdx_uniform_CD = compute_CDFD(f, Δ)
        
        # compute using central difference non-unform grid
        dfdx_non_uniform_CD = compute_CDFD(f, collect(x))

        # compute the derivative using FFT:
        dfdx_fft = compute_FTD(f, L)

        # Plot derivatives
        fig1 = plot(x, dfdx_exact, label="Exact")
        plot!(fig1, x, dfdx_uniform, seriestype=:scatter, label="Uniform Grid Pade")
        plot!(fig1, x, dfdx_non_uniform, seriestype=:scatter, label="Non-Uniform Pade")
        plot!(fig1, x, dfdx_uniform_CD, seriestype=:scatter, label="Uniform Grid CD")
        plot!(fig1, x, dfdx_non_uniform_CD, seriestype=:scatter, label="Non-Uniform CD")
        plot!(fig1, x, dfdx_fft, seriestype=:scatter, label="FFT")
        display(fig1)
    else
        # !!! ACHTUNG: for these high order scheme there is minimum number of points required, i.e. N > 10 !!!!
        M    = 10
        Plot_L2_vs_Grid_UniformGrid(M)
        Plot_L2_vs_Grid_NonUniformGrid(M)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end