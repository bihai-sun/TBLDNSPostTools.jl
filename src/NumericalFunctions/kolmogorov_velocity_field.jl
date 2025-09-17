#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================# 
#                               kolmogorov_velocity_field.jl
#                               ----------------------------
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 07-05-2025
#
#    
#    This code contains functions to:
#        1. generate a 3D velocity field with Kolmogorov energy spectrum (E(k) ∝ k^(-5/3))
#        2. compute the Fourier frequencies
#        3. test the Kolmogorov velocity field generation and plot the energy spectrum 
#       
#========================================================================================================#


#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using Random
using FFTW
using Statistics, StatsBase, KernelDensity, LinearAlgebra

include("PadeDifferentiation.jl")
include("FourierDifferentiation.jl")
include("FD-Differentiation.jl")


#========================================================================================================#
# Functions
#========================================================================================================#

function generate_kolmogorov_velocity_field(nx, ny, nz; kmin=1, kmax=nothing, L=2π, E0=1.0, seed=nothing)
"""
    generate_kolmogorov_velocity_field(nx, ny, nz; kmin=1, kmax=nothing, L=2π, E0=1.0, seed=nothing)

Generate a 3D velocity field with Kolmogorov energy spectrum (E(k) ∝ k^(-5/3)).

    Arguments
        - `nx`, `ny`, `nz`: Number of grid points in each direction
        - `kmin`: Minimum wavenumber for the spectrum (default: 1)
        - `kmax`: Maximum wavenumber for the spectrum (default: nx/3 to avoid aliasing)
        - `L`: Domain size (default: 2π)
        - `E0`: Energy scaling factor (default: 1.0)
        - `seed`: Random seed for reproducibility (default: nothing)

    Returns
        - `u`, `v`, `w`: Components of the 3D velocity field
        - `x`, `y`, `z`: Grid coordinates
        - `E`: Energy spectrum
        - `k_values`: Wavenumbers

    Examples
        ``` julia
            u, v, w, x, y, z, E, k_values = generate_kolmogorov_velocity_field(64, 64, 64)
        ```
"""
    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    # Default maximum wavenumber (to avoid aliasing)
    if isnothing(kmax)
        kmax = min(nx, ny, nz) ÷ 3
    end
    
    # Create grid
    dx, dy, dz = L/nx, L/ny, L/nz
    x = range(0, L-dx, length=nx)
    y = range(0, L-dy, length=ny)
    z = range(0, L-dz, length=nz)
    
    # Initialize velocity field in Fourier space
    u_hat = zeros(ComplexF64, nx, ny, nz)
    v_hat = zeros(ComplexF64, nx, ny, nz)
    w_hat = zeros(ComplexF64, nx, ny, nz)
    
    # Define wavenumbers
    kx = 2π/L * fftfreq(nx, nx)
    ky = 2π/L * fftfreq(ny, ny)
    kz = 2π/L * fftfreq(nz, nz)
    
    # Function to compute Kolmogorov energy spectrum: E(k) ∝ k^(-5/3)
    function kolmogorov_spectrum(k)
        if k < kmin || k > kmax
            return 0.0
        else
            return E0 * k^(-5/3)
        end
    end
    
    # Generate energy spectrum
    k_values = Float64[]
    E = Float64[]
    
    # Fill velocity field in Fourier space with Kolmogorov spectrum
    total_energy = 0.0
    for i in 1:nx
        for j in 1:ny
            for k in 1:nz
                # Calculate wavenumber magnitude
                k_squared = kx[i]^2 + ky[j]^2 + kz[k]^2
                k_mag = sqrt(k_squared)
                
                # Skip k=0 (mean flow)
                if k_mag ≈ 0.0
                    continue
                end
                
                # Get energy for this wavenumber from spectrum
                energy = kolmogorov_spectrum(k_mag)
                
                # Skip if no energy at this wavenumber
                if energy ≈ 0.0
                    continue
                end
                
                # Generate random phase
                phase = 2π * rand()
                
                # Create a random direction that is perpendicular to k
                # (to ensure divergence-free velocity field)
                kv = [kx[i], ky[j], kz[k]]
                
                # Generate random vector
                rv = randn(3)
                
                # Make it perpendicular to k (solenoidal)
                rv = rv - (rv⋅kv) * kv / (k_squared + 1e-10)
                rv_norm = norm(rv)
                
                if rv_norm > 1e-10
                    rv ./= rv_norm
                    
                    # Scale by energy
                    amplitude = sqrt(2 * energy)
                    
                    # Apply to Fourier components
                    u_hat[i,j,k] = amplitude * rv[1] * exp(im * phase)
                    v_hat[i,j,k] = amplitude * rv[2] * exp(im * phase)
                    w_hat[i,j,k] = amplitude * rv[3] * exp(im * phase)
                    
                    # Apply conjugate for negative frequencies to ensure real velocity field
                    i_neg = mod1(nx - i + 2, nx)
                    j_neg = mod1(ny - j + 2, ny)
                    k_neg = mod1(nz - k + 2, nz)
                    
                    if !(i == i_neg && j == j_neg && k == k_neg)
                        u_hat[i_neg,j_neg,k_neg] = conj(u_hat[i,j,k])
                        v_hat[i_neg,j_neg,k_neg] = conj(v_hat[i,j,k])
                        w_hat[i_neg,j_neg,k_neg] = conj(w_hat[i,j,k])
                    end
                    
                    # Add to total energy
                    total_energy += energy
                    
                    # Store for spectrum output
                    push!(k_values, k_mag)
                    push!(E, energy)
                end
            end
        end
    end
    
    # Convert back to physical space
    u = real.(ifft(u_hat))
    v = real.(ifft(v_hat))
    w = real.(ifft(w_hat))
    
    # Scale velocity field to match desired total energy
    actual_energy = 0.5 * mean(u.^2 + v.^2 + w.^2)
    scale_factor = sqrt(E0 / max(actual_energy, 1e-10))
    
    u .*= scale_factor
    v .*= scale_factor
    w .*= scale_factor
    
    return u, v, w, x, y, z, E, k_values
end


function fftfreq(n, d=1.0)
"""
    fftfreq(n, d=1.0)

Return the Discrete Fourier Transform sample frequencies.
Similar to numpy.fft.fftfreq.

    Arguments
        - `n`: Sample count
        - `d`: Sample spacing (default: 1.0)

    Returns:
        - Array of frequencies
"""
    val = 1.0 / (n * d)
    results = zeros(n)
    N = (n-1) ÷ 2 + 1
    p1 = 0:N-1
    results[1:N] = p1
    p2 = -n÷2:-1
    results[N+1:end] = p2
    return results
end


# New function to compute vorticity from velocity components
function compute_vorticity(x, y, z, u, v, w; method=:Fourier)
    """
    Compute vorticity vector from velocity components using different differentiation methods.
    
    Arguments:
        - x, y, z: Grid coordinates
        - u, v, w: Velocity components
        - method: Differentiation method, options are:
            * :Fourier - Fourier spectral differentiation
            * :PadeUniform - Padé scheme on uniform grid
            * :PadeNonUniform - Padé scheme on non-uniform grid
            * :FDUniform - Finite difference on uniform grid
            * :FDNonUniform - Finite difference on non-uniform grid
    
    Returns:
        - ωx, ωy, ωz: Vorticity components
        
    Example:
        ```julia
        ωx, ωy, ωz = compute_vorticity(x, y, z, u, v, w, method=:Fourier)
        ```
    """
    
    # Get dimensions
    nx, ny, nz = size(u)
    
    # Determine domain size
    L = 2π  # Assuming a standard 2π domain
    
    # Compute grid spacings for uniform methods
    dx, dy, dz = L/nx, L/ny, L/nz
    
    # Initialize vorticity components
    ωx = zeros(Float64, nx, ny, nz)
    ωy = zeros(Float64, nx, ny, nz)
    ωz = zeros(Float64, nx, ny, nz)
    
    # Compute derivatives based on the specified method
    if method == :Fourier
        # Using Fourier spectral differentiation
        dvdz = compute_FTD(v, L, dim=3)
        dwdy = compute_FTD(w, L, dim=2)
        
        dudz = compute_FTD(u, L, dim=3)
        dwdx = compute_FTD(w, L, dim=1)
        
        dudy = compute_FTD(u, L, dim=2)
        dvdx = compute_FTD(v, L, dim=1)
        
    elseif method == :PadeUniform
        # Using Padé scheme on uniform grid
        dvdz = compute_PadeFD(v, dz, dim=3)
        dwdy = compute_PadeFD(w, dy, dim=2)
        
        dudz = compute_PadeFD(u, dz, dim=3)
        dwdx = compute_PadeFD(w, dx, dim=1)
        
        dudy = compute_PadeFD(u, dy, dim=2)
        dvdx = compute_PadeFD(v, dx, dim=1)
        
    elseif method == :PadeNonUniform
        # Using Padé scheme on non-uniform grid
        dvdz = compute_PadeFD(v, collect(z), dim=3)
        dwdy = compute_PadeFD(w, collect(y), dim=2)
        
        dudz = compute_PadeFD(u, collect(z), dim=3)
        dwdx = compute_PadeFD(w, collect(x), dim=1)
        
        dudy = compute_PadeFD(u, collect(y), dim=2)
        dvdx = compute_PadeFD(v, collect(x), dim=1)
        
    elseif method == :FDUniform
        # Using finite difference on uniform grid
        dvdz = compute_CDFD(v, dz, dim=3)
        dwdy = compute_CDFD(w, dy, dim=2)
        
        dudz = compute_CDFD(u, dz, dim=3)
        dwdx = compute_CDFD(w, dx, dim=1)
        
        dudy = compute_CDFD(u, dy, dim=2)
        dvdx = compute_CDFD(v, dx, dim=1)
        
    elseif method == :FDNonUniform
        # Using finite difference on non-uniform grid
        dvdz = compute_CDFD(v, collect(z), dim=3)
        dwdy = compute_CDFD(w, collect(y), dim=2)
        
        dudz = compute_CDFD(u, collect(z), dim=3)
        dwdx = compute_CDFD(w, collect(x), dim=1)
        
        dudy = compute_CDFD(u, collect(y), dim=2)
        dvdx = compute_CDFD(v, collect(x), dim=1)
        
    else
        error("Unknown differentiation method: $method")
    end
    
    # Compute vorticity components: ω = ∇×u
    ωx .= dwdy .- dvdz  # ωx = dw/dy - dv/dz
    ωy .= dudz .- dwdx  # ωy = du/dz - dw/dx
    ωz .= dvdx .- dudy  # ωz = dv/dx - du/dy
    
    return ωx, ωy, ωz
end

#========================================================================================================#
# Plotting Functions 
#========================================================================================================#

function plot_scatter_of_divergence(dudx, dvdy, dwdz; label="Fourier")
    """
    Plot scatter plot of divergence components
    
    Arguments:
        - dudx, dvdy, dwdz: The partial derivatives
        - label: Method label for the plot (default: "Fourier")
        
    Returns:
        - p: The plot object with scatter plot of divergence components
    """
    p = scatter(
        vec(dudx .+ dvdy),
        vec(-dwdz),
        markersize=2,
        alpha=0.5,
        label=label,
        xlabel="∂u/∂x + ∂v/∂y",
        ylabel="-∂w/∂z",
        title="Divergence Components (should align on y=x line)"
    )
    
    # Add a reference line (y=x) for perfect divergence-free condition
    plot!(p, identity, linestyle=:dash, color=:black, label="y=x")
    
    return p
end


function plot_divergence_jpdf(dudx, dvdy, dwdz; label="Fourier", nbins=100)
"""
Plot joint probability density function of divergence components with probability contours

Arguments:
    - dudx, dvdy, dwdz: The partial derivatives
    - label: Method label for the plot (default: "Fourier")
    - nbins: Number of bins for the histogram (default: 100)
    
Returns:
    - p: The plot object with JPDF heatmap and probability contours
"""
    
     # Calculate components
    x_comp = vec(dudx .+ dvdy)
    y_comp = vec(-dwdz)
    
    # Determine plot range - extend a bit beyond min/max values
    range_x = extrema(x_comp)
    range_y = extrema(y_comp)
    x_min, x_max = range_x[1] - 0.1*abs(range_x[1]), range_x[2] + 0.1*abs(range_x[2])
    y_min, y_max = range_y[1] - 0.1*abs(range_y[1]), range_y[2] + 0.1*abs(range_y[2])
    
    # Create 2D histogram for the heatmap
    edges_x = range(x_min, x_max, length=nbins)
    edges_y = range(y_min, y_max, length=nbins)
    h = fit(Histogram, (x_comp, y_comp), (edges_x, edges_y))
    
    # Normalize to get PDF
    hist_pdf = normalize(h, mode=:pdf)
    
    # Create heatmap
    p = heatmap(hist_pdf.edges[1][1:end-1], hist_pdf.edges[2][1:end-1], hist_pdf.weights',
            xlabel="∂u/∂x + ∂v/∂y",
            ylabel="-∂w/∂z",
            title="Divergence Components JPDF ($label)",
            color=:viridis,
            colorbar_title="Probability Density")
    
    # Add a reference line (y=x) for perfect divergence-free condition
    plot!(p, identity, 
          xlims=(x_min, x_max), 
          ylims=(y_min, y_max),
          linestyle=:dash, 
          color=:white, 
          linewidth=2,
          label="y=x")
    
    # Compute 2D kernel density estimate for smoother contours
    kde_obj = kde((x_comp, y_comp))
    
    # Create a grid for evaluating the KDE
    grid_x = range(x_min, x_max, length=200)
    grid_y = range(y_min, y_max, length=200)
    
    # Evaluate KDE on the grid
    z = zeros(length(grid_x), length(grid_y))
    for (i, x) in enumerate(grid_x)
        for (j, y) in enumerate(grid_y)
            z[i, j] = pdf(kde_obj, [x, y])  # This uses the pdf method from KernelDensity
        end
    end
    
    # Normalize the density
    z = z / maximum(z)
    
    # Sort the density values to determine contour levels
    sorted_densities = sort(vec(z), rev=true)
    cumulative_prob = cumsum(sorted_densities) / sum(sorted_densities)
    
    # Find the density thresholds for the desired probability levels
    levels = Dict{Float64, Float64}()
    for prob in [0.5, 0.67, 0.9, 0.95]
        idx = findfirst(cumulative_prob .>= prob)
        levels[prob] = sorted_densities[idx]
    end
    
    # Add contour lines for the specified probability levels
    contour!(p, grid_x, grid_y, z', 
             levels=[levels[0.5], levels[0.67], levels[0.9], levels[0.95]],
             linestyles=[:solid, :solid, :solid, :solid],
             linewidths=[2, 2, 2, 2],
             linecolors=[:white, :yellow, :orange, :red],
             labels=["50%", "67%", "90%", "95%"])
    
    return p
end


function plot_vorticity_divergence_jpdf(dωxdx, dωydy, dωzdz; 
                              label="Fourier", 
                              nbins=100)
"""
Plot joint probability density function of vorticity divergence components with probability contours

Arguments:
    - dωxdx, dωydy, dωzdz: The partial derivatives of vorticity components
    - label: Method label for the plot (default: "Fourier")
    - nbins: Number of bins for the histogram (default: 100)
    
Returns:
    - p: The plot object with JPDF heatmap and probability contours
"""
    # Calculate components
    x_comp = vec(dωxdx .+ dωydy)
    y_comp = vec(-dωzdz)
    
    # Determine plot range - extend a bit beyond min/max values
    range_x      = extrema(x_comp)
    range_y      = extrema(y_comp)
    x_min, x_max = range_x[1] - 0.1*abs(range_x[1]), range_x[2] + 0.1*abs(range_x[2])
    y_min, y_max = range_y[1] - 0.1*abs(range_y[1]), range_y[2] + 0.1*abs(range_y[2])
    
    # Create 2D histogram for the heatmap
    edges_x = range(x_min, x_max, length=nbins)
    edges_y = range(y_min, y_max, length=nbins)
    h = fit(Histogram, (x_comp, y_comp), (edges_x, edges_y))
    
    # Normalize to get PDF
    hist_pdf = normalize(h, mode=:pdf)
    
    # Create heatmap with vorticity-specific labels
    p = heatmap(hist_pdf.edges[1][1:end-1], hist_pdf.edges[2][1:end-1], hist_pdf.weights',
            xlabel="∂ωₓ/∂x + ∂ωᵧ/∂y",
            ylabel="-∂ωᵤ/∂z",
            title="Vorticity Divergence Components JPDF ($label)",
            color=:viridis,
            colorbar_title="Probability Density")
    
    # Add a reference line (y=x) for perfect divergence-free condition
    plot!(p, identity, 
          xlims=(x_min, x_max), 
          ylims=(y_min, y_max),
          linestyle=:dash, 
          color=:black, 
          linewidth=2,
          label="∂ωₓ/∂x + ∂ωᵧ/∂y = -∂ωᵤ/∂z")
    
    # Compute 2D kernel density estimate for smoother contours
    kde_obj = kde((x_comp, y_comp))
    
    # Create a grid for evaluating the KDE
    grid_x = range(x_min, x_max, length=200)
    grid_y = range(y_min, y_max, length=200)
    
    # Evaluate KDE on the grid
    z = zeros(length(grid_x), length(grid_y))
    for (i, x) in enumerate(grid_x)
        for (j, y) in enumerate(grid_y)
            z[i, j] = pdf(kde_obj, x, y)  # KernelDensity requires two separate arguments
        end
    end
    
    # Normalize the density
    z = z / maximum(z)
    
    # Sort the density values to determine contour levels
    sorted_densities = sort(vec(z), rev=true)
    cumulative_prob  = cumsum(sorted_densities) / sum(sorted_densities)
    
    # Find the density thresholds for the desired probability levels
    levels = Dict{Float64, Float64}()
    for prob in [0.5, 0.67, 0.9, 0.95]
        idx = findfirst(cumulative_prob .>= prob)
        levels[prob] = sorted_densities[idx]
    end
    
    # Add contour lines for the specified probability levels
    contour!(p, grid_x, grid_y, z', 
             levels=[levels[0.5], levels[0.67], levels[0.9], levels[0.95]],
             linestyles=[:solid, :solid, :solid, :solid],
             linewidths=[2, 2, 2, 2],
             linecolors=[:blue, :green, :yellow, :red],
             labels=["50%", "67%", "90%", "95%"])
    
    return p
end


function plot_vorticity_scatter_of_divergence(dωxdx, dωydy, dωzdz; 
                                    label="Fourier")
    """
    Plot scatter plot of vorticity divergence components
    
    Arguments:
        - dωxdx, dωydy, dωzdz: The partial derivatives of vorticity components
        - label: Method label for the plot (default: "Fourier")
        
    Returns:
        - p: The plot object with scatter plot of vorticity divergence components
    """
    p = scatter(
        vec(dωxdx .+ dωydy),
        vec(-dωzdz),
        markersize=2,
        alpha=0.5,
        label=label,
        xlabel="∂ωₓ/∂x + ∂ωᵧ/∂y",
        ylabel="-∂ωᵤ/∂z",
        title="Vorticity Divergence Components ($label)"
    )
    
    # Add a reference line (y=x) for perfect divergence-free condition
    plot!(p, identity, linestyle=:dash, color=:black, label="y=x")
    
    return p
end

#========================================================================================================#


function test_kolmogorov_spectrum(n=64)
"""
    test_kolmogorov_spectrum(n=64)

Test the kolmogorov_velocity_field function and plot the energy spectrum.

    Arguments
        - `n`: Grid size (default: 64)

    Example
        ``` julia
            test_kolmogorov_spectrum(128)
        ```
"""
    println("Generating Kolmogorov velocity field on $(n)×$(n)×$(n) grid...")
    u, v, w, x, y, z, E, k_values = generate_kolmogorov_velocity_field(n, n, n, seed=42)
    
    # Create k and E arrays sorted by k
    sorted_indices = sortperm(k_values)
    k_sorted       = k_values[sorted_indices]
    E_sorted       = E[sorted_indices]
    
    # Group E values by wavenumber (binning)
    k_bins = Dict{Int, Vector{Float64}}()
    for i in 1:length(k_sorted)
        k_bin = round(Int, k_sorted[i])
        if !haskey(k_bins, k_bin)
            k_bins[k_bin] = Float64[]
        end
        push!(k_bins[k_bin], E_sorted[i])
    end
    
    # Compute mean energy for each wavenumber bin
    k_unique = sort(collect(keys(k_bins)))
    E_mean = [mean(k_bins[k]) for k in k_unique]
    
    # Compute theoretical Kolmogorov spectrum for comparison
    E_theory = [k^(-5/3) for k in k_unique]
    
    println("Statistics:")
    println("  Mean velocity: (", mean(u), ", ", mean(v), ", ", mean(w), ")")
    println("  Velocity std: (", std(u), ", ", std(v), ", ", std(w), ")")
    println("  Energy: ", 0.5 * mean(u.^2 + v.^2 + w.^2))
    
    println("\nUse the following code to plot the spectrum:")
    println("""
    
    # Plot energy spectrum
    k_unique = $k_unique
    E_mean   = $E_mean
    E_theory = $E_theory
    
    p = plot(k_unique, E_mean, 
        xscale=:log10, yscale=:log10, 
        xlabel="Wavenumber (k)", ylabel="Energy E(k)",
        label="Simulated spectrum", 
        marker=:circle, markersize=3, linealpha=0.5)
    
    plot!(p, k_unique, E_theory .* E_mean[1] / E_theory[1], 
        label="k^(-5/3)", 
        linestyle=:dash, lw=2)
    
    # Add a title
    title!(p, "Kolmogorov Energy Spectrum")
    
    savefig(p, "kolmogorov_spectrum.png")
    display(p)
    
    # Plot a slice of the velocity field
    heatmap(x, y, u[:,:,div(end,2)]', 
        xlabel="x", ylabel="y", 
        title="Velocity u component (z midplane)",
        aspect_ratio=:equal)
    savefig("velocity_slice.png")
    """)
    
    return k_unique, E_mean, E_theory, u, v, w, x, y, z, E, k_values
end


function test_derivative_of_velocity(n::Int=64)
"""
    test_derivative_of_velocity(u, v, w, dx, dy, dz)

"""
# compute Kolmogorov velocity field
    u, v, w, x, y, z, E, k_values = generate_kolmogorov_velocity_field(n, n, n, kmax=Int(n÷5));
    
    # Compute derivatives using Fourier differentiation
    L = 2π
    dx = L/n
    dy = L/n
    dz = L/n
    
    dudx_PD   = zeros(Float64, n, n, n)
    dudx_FD   = zeros(Float64, n, n, n)
    dudx_FTD  = compute_FTD(u, L, dim=1)
    dudx_PD   = compute_PadeFD(u, dx, dim=1, dudx_PD)
    dudx_FD   = compute_CDFD(u, dx, dim=1, dudx_FD)

    p1 = plot(x, dudx_FTD[:,div(end,2),div(end,2)], 
        label="Fourier", 
        xlabel="x", ylabel="dudx",
        title="Derivative of u component (y midplane, z midplane)",
        legend=:topright)
    plot!(p1, x, dudx_PD[:,div(end,2),div(end,2)],
        label="Pade", 
        linestyle=:dash)
    plot!(p1, x, dudx_FD[:,div(end,2),div(end,2)],
        label="FD", 
        linestyle=:dot)
    display(p1) 
    # plot the error of Pade and FD relative to Fourier as a function of x
    err_PD = (dudx_PD .- dudx_FTD) / (maximum(abs.(dudx_FTD)))
    err_FD = (dudx_FD .- dudx_FTD) / (maximum(abs.(dudx_FTD)))
    p1 = plot(x, err_PD[:,div(end,2),div(end,2)], 
        label="Pade", 
        xlabel="x", ylabel="dudx",
        title="Relative Error of derivative of u component (y midplane, z midplane)",
        legend=:topright)
    plot!(p1, x, err_FD[:,div(end,2),div(end,2)],
        label="FD", 
        linestyle=:dash)
    display(p1)

    dvdy_PD  = zeros(Float64, n, n, n)
    dvdy_FD  = zeros(Float64, n, n, n)
    dvdy_FTD = compute_FTD(v, L, dim=2)
    dvdy_PD  = compute_PadeFD(v, dy, dim=2, dvdy_PD)
    dvdy_FD  = compute_CDFD(v, dy, dim=2, dvdy_FD)
    p2 = plot(y, dvdy_FTD[div(end,2),:,div(end,2)], 
        label="Fourier", 
        xlabel="y", ylabel="dvdy",
        title="Derivative of v component (x midplane, z midplane)",
        legend=:topright)
    plot!(p2, y, dvdy_PD[div(end,2),:,div(end,2)],
        label="Pade", 
        linestyle=:dash)
    plot!(p2, y, dvdy_FD[div(end,2),:,div(end,2)],
        label="FD", 
        linestyle=:dot)
    display(p2)
    # plot the error of Pade and FD relative to Fourier as a function of x
    err_PD = (dvdy_PD .- dvdy_FTD) / (maximum(abs.(dvdy_FTD)))
    err_FD = (dvdy_FD .- dvdy_FTD) / (maximum(abs.(dvdy_FTD)))
    p2 = plot(y, err_PD[div(end,2),:,div(end,2)], 
        label="Pade", 
        xlabel="y", ylabel="dvdy",
        title="Relative Error of derivative of v component (x midplane, z midplane)",
        legend=:topright)
    plot!(p2, y, err_FD[div(end,2),:,div(end,2)],
        label="FD", 
        linestyle=:dash)
    display(p2)

    dwdz_PD  = zeros(Float64, n, n, n)
    dwdz_FD  = zeros(Float64, n, n, n)
    dwdz_FTD = compute_FTD(w, L, dim=3)
    dwdz_PD  = compute_PadeFD(w, dz, dim=3, dwdz_PD)
    dwdz_FD  = compute_CDFD(w, dz, dim=3, dwdz_FD)
    p3 = plot(z, dwdz_FTD[div(end,2),div(end,2),:], 
        label="Fourier", 
        xlabel="z", ylabel="dwdz",
        title="Derivative of w component (x midplane, y midplane)",
        legend=:topright)
    plot!(p3, z, dwdz_PD[div(end,2),div(end,2),:],
        label="Pade", 
        linestyle=:dash)
    plot!(p3, z, dwdz_FD[div(end,2),div(end,2),:],
        label="FD", 
        linestyle=:dot)
    display(p3)
    # plot the error of Pade and FD relative to Fourier as a function of x
    err_PD = (dwdz_PD .- dwdz_FTD) / (maximum(abs.(dwdz_FTD)))
    err_FD = (dwdz_FD .- dwdz_FTD) / (maximum(abs.(dwdz_FTD)))
    p3 = plot(z, err_PD[div(end,2),div(end,2),:], 
        label="Pade", 
        xlabel="z", ylabel="dwdz",
        title="Relative Error of derivative of w component (x midplane, y midplane)",
        legend=:topright)
    plot!(p3, z, err_FD[div(end,2),div(end,2),:],
        label="FD", 
        linestyle=:dash)
    display(p3)

    # compute the divergence of the velocity field
    div_vel_FTD  = zeros(Float64, n, n, n)
    div_vel_PD   = zeros(Float64, n, n, n)
    div_vel_FD   = zeros(Float64, n, n, n)
    div_vel_FTD  = dudx_FTD .+ dvdy_FTD .+ dwdz_FTD;
    div_vel_PD   = dudx_PD  .+ dvdy_PD  .+ dwdz_PD;
    div_vel_FD   = dudx_FD  .+ dvdy_FD  .+ dwdz_FD;
    p4 = plot(x, div_vel_FTD[:,div(end,2),div(end,2)], label="Fourier", xlabel="x", ylabel="div(u)", title="Divergence of velocity field (y midplane, z midplane)", legend=:topright)
    plot!(p4, x, div_vel_PD[:,div(end,2),div(end,2)], label="Pade", linestyle=:dash)
    plot!(p4, x, div_vel_FD[:,div(end,2),div(end,2)],label="FD", linestyle=:dot)
    display(p4)
    # plot the error of Pade and FD relative to Fourier as a function of x
    err_PD = div_vel_PD .- div_vel_FTD
    err_FD = div_vel_FD .- div_vel_FTD
    p4 = plot(x, err_PD[:,div(end,2),div(end,2)], label="Pade", xlabel="x", ylabel="div(u)", title="Error of divergence of velocity field (y midplane, z midplane)", legend=:topright)
    plot!(p4, x, err_FD[:,div(end,2),div(end,2)], label="FD", linestyle=:dash)
    display(p4)

    # plot a scatter plot of (dudx_FTD .+ dvdy_FTD) vs (-dwdz_FTD) for all values of the 3D array
    # Create scatter plots to compare the relationship between components
    p_scatter = scatter(
        vec(dudx_FTD .+ dvdy_FTD),
        vec(-dwdz_FTD),
        markersize=2,
        alpha=0.5,
        label="Fourier",
        xlabel="dudx + dvdy",
        ylabel="-dwdz",
        title="Divergence Components (should align on y=x line)"
    )
    # Add a reference line (y=x) that represents perfect divergence-free condition
    plot!(p_scatter, identity, linestyle=:dash, color=:black, label="y=x")
    display(p_scatter)
    savefig(p_scatter, "divergence_components_scatter_FT.png")
    
    # Plot data from Pade differentiation
    p_scatter = scatter( 
        vec(dudx_PD .+ dvdy_PD),
        vec(-dwdz_PD),
        markersize=2,
        alpha=0.5,
        label="Pade"
    )
    # add reerence line (y=x) that represents perfect divergence-free condition
    plot!(p_scatter, identity, linestyle=:dash, color=:black, label="y=x")
    display(p_scatter)
    savefig(p_scatter, "divergence_components_scatter_PD.png")


    # Plot data from finite difference differentiation
    p_scatter = scatter!(p_scatter, 
        vec(dudx_FD .+ dvdy_FD),
        vec(-dwdz_FD),
        markersize=2,
        alpha=0.5,
        label="FD"
    )
    # add reerence line (y=x) that represents perfect divergence-free condition
    plot!(p_scatter, identity, linestyle=:dash, color=:black, label="y=x")
    display(p_scatter)
    savefig(p_scatter, "divergence_components_scatter_FD.png")
    

    # compute the divergence of the velocity field     
end


function test_divergence_of_velocity(n::Int=64, fac::Int=1; plot_jpdf::Bool=true)
"""
    test_divergence_of_velocity(n::Int=64, fac::Int=1; plot_jpdf::Bool=true)
    Test the divergence of a Kolmogorov velocity field using different differentiation methods.
    Arguments:
        - n: Grid size (default: 64)
        - fac: Factor to reduce the grid size for testing (default: 1)
        - plot_jpdf: Flag to plot joint probability density function (default: true)
    Example:
        ```julia
            test_divergence_of_velocity(128, 2)
        ```
"""
# compute Kolmogorov velocity field
    Kmax = n÷3
    kmax = Int(Kmax ÷ fac)
    println()
    println("Generating Kolmogorov velocity field on $(n)×$(n)×$(n) grid ... with kₘₐₓ = $(kmax))")
    println()

    u, v, w, x, y, z, E, k_values = generate_kolmogorov_velocity_field(n, n, n, kmax=Int(round(n/5)));
    
    # Compute divergence using Fourier differentiation
    L = 2π
    dx = L/n
    dy = L/n
    dz = L/n
    
    div_vel  = zeros(Float64, n, n, n)

    # Compute divergence using Fourier differentiation
    dudx     = compute_FTD(u, L, dim=1)
    dvdy     = compute_FTD(v, L, dim=2)
    dwdz     = compute_FTD(w, L, dim=3)
    div_vel .= dudx .+ dvdy .+ dwdz;
    println("Maximum Divergence of velocity field using Fourier differentiation: ", maximum(abs.(div_vel)))
    if plot_jpdf
        p_jpdf = plot_divergence_jpdf(dudx, dvdy, dwdz, label="Fourier")
        display(p_jpdf)
        savefig(p_jpdf, "divergence_components_jpdf_FT.png")
    else
        p_scatter = plot_scatter_of_divergence(dudx, dvdy, dwdz, label="Fourier")
        display(p_scatter)
        savefig(p_scatter, "divergence_components_scatter_FT.png")
    end

    # Compute divergence using Pade differentiation with uniform grid
    dudx     = compute_PadeFD(u, dx, dim=1)
    dvdy     = compute_PadeFD(v, dy, dim=2)
    dwdz     = compute_PadeFD(w, dz, dim=3)
    div_vel .= dudx .+ dvdy .+ dwdz
    println("Maximum Divergence of velocity field using uniform grid Pade differentiation: ",     maximum(abs.(div_vel)))
    if plot_jpdf
        p_jpdf = plot_divergence_jpdf(dudx, dvdy, dwdz, label="Pade-uniform")
        display(p_jpdf)
        savefig(p_jpdf, "divergence_components_jpdf_PD_uniform.png")
    else
        p_scatter = plot_scatter_of_divergence(dudx, dvdy, dwdz, label="Pade-uniform")
        display(p_scatter)
        savefig(p_scatter, "divergence_components_scatter_PD_uniform.png")
    end    
    
    # Compute divergence using Pade differentiation with non-uniform grid
    dudx     = compute_PadeFD(u, collect(x), dim=1)
    dvdy     = compute_PadeFD(v, collect(y), dim=2)
    dwdz     = compute_PadeFD(w, collect(z), dim=3)
    div_vel .= dudx .+ dvdy .+ dwdz
    println("Maximum Divergence of velocity field using non-uniform grid Pade differentiation: ", maximum(abs.(div_vel)))
    if plot_jpdf
        p_jpdf = plot_divergence_jpdf(dudx, dvdy, dwdz, label="Pade-nonuniform")
        display(p_jpdf)
        savefig(p_jpdf, "divergence_components_jpdf_PD_nonuniform.png")
    else
        p_scatter = plot_scatter_of_divergence(dudx, dvdy, dwdz, label="Pade-nonuniform")
        display(p_scatter)
        savefig(p_scatter, "divergence_components_scatter_PD_nonuniform.png")
    end
    
    # Compute divergence using finite difference differentiation with uniform grid
    dudx     = compute_CDFD(u, dx, dim=1)
    dvdy     = compute_CDFD(v, dy, dim=2)
    dwdz     = compute_CDFD(w, dz, dim=3)
    div_vel .= dudx .+ dvdy .+ dwdz
    println("Maximum Divergence of velocity field using uniform grid finite difference: ",        maximum(abs.(div_vel)))
    if plot_jpdf
        p_jpdf = plot_divergence_jpdf(dudx, dvdy, dwdz, label="FD-uniform")
        display(p_jpdf)
        savefig(p_jpdf, "divergence_components_jpdf_FD_uniform.png")
    else
        p_scatter = plot_scatter_of_divergence(dudx, dvdy, dwdz, label="FD-uniform")
        display(p_scatter)
        savefig(p_scatter, "divergence_components_scatter_FD_uniform.png")
    end
    
    # Compute divergence using finite difference differentiation with non-uniform grid
    dudx     = compute_CDFD(u, collect(x), dim=1)
    dvdy     = compute_CDFD(v, collect(y), dim=2)
    dwdz     = compute_CDFD(w, collect(z), dim=3)
    div_vel .= dudx .+ dvdy .+ dwdz
    println("Maximum Divergence of velocity field using non-uniform grid finite difference: ",    maximum(abs.(div_vel)))
    if plot_jpdf
        p_jpdf = plot_divergence_jpdf(dudx, dvdy, dwdz, label="FD-nonuniform")
        display(p_jpdf)
        savefig(p_jpdf, "divergence_components_jpdf_FD_nonuniform.png")
    else
        p_scatter = plot_scatter_of_divergence(dudx, dvdy, dwdz, label="FD-nonuniform")
        display(p_scatter)
        savefig(p_scatter, "divergence_components_scatter_FD_nonuniform.png")
    end
end


function main()
    n = 64
    k_unique, E_mean, E_theory, u, v, w, x, y, z, E, k_values = test_kolmogorov_spectrum(n)

    p = plot(k_unique, E_mean, 
        xscale=:log10, yscale=:log10, 
        xlabel="Wavenumber (k)", ylabel="Energy E(k)",
        label="Simulated spectrum", 
        marker=:circle, markersize=3, linealpha=0.5)
    
    plot!(p, k_unique, E_theory .* E_mean[1] / E_theory[1], 
        label="k^(-5/3)", 
        linestyle=:dash, lw=2)
    
    # Add a title
    title!(p, "Kolmogorov Energy Spectrum")
    
    savefig(p, "kolmogorov_spectrum.png")
    display(p)
    
    # Plot a slice of the velocity field
    heatmap(x, y, u[:,:,div(end,2)]', 
        xlabel="x", ylabel="y", 
        title="Velocity u component (z midplane)",
        aspect_ratio=:equal)
    savefig("velocity_slice.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end