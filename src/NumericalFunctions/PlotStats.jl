#!/usr/bin/env julia

# include libraries
using HDF5
using Interpolations, BSplineKit
using QuadGK
using DelimitedFiles
using Printf

# include the necessary functions
# 1. Function to compute the x-derivative
include("m_differentiation.jl")


# --------------- FUNCTIONS ----------------

function readAveragesFromFile(file_name::String)
    """
    Read the streamwise-averaged data from a text file and return the data as arrays
    """
    # Skip the header line (#) by starting from the second line
    data_matrix = readdlm(file_name, '\t', skipstart=1)

    # Extract columns into separate arrays
    η     = data_matrix[:, 1]
    U_η   = data_matrix[:, 2]
    V_η   = data_matrix[:, 3]
    Ω_z_η = data_matrix[:, 4]
    uu_η  = data_matrix[:, 5]
    vv_η  = data_matrix[:, 6]
    ww_η  = data_matrix[:, 7]
    uv_η  = data_matrix[:, 8]

    return η, U_η, V_η, Ω_z_η, uu_η, vv_η, ww_η, uv_η
end

function PlotStats(η, U_η, V_η, Ω_z_η, uu_η, vv_η, ww_η, uv_η, β, ofile_dir="/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/")
    """
    """
    fig1 = plot(η[2:end], U_η[2:end], color=:black, label="U_η", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)", xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 100), legend=false, title="β = $β")
    # Create a secondary y-axis for uu
    uu_max = maximum(uu_η)
    plot!(twinx(), η[2:end], uu_η[2:end], color=:red, label="uu_η", ylabel="uu/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 100), legend=false)
    display(fig1)
    savefig(fig1, ofile_dir * "U_η_avg_β=$β.png")

    Vmax = maximum(V_η)    
    fig2 = plot(η[2:end], V_η[2:end], color=:black, label="V_η", xlabel="η = y/δ₁(x)", ylabel="V/Uₑ(x)", xscale=:log10, ylims=(0, 1.05*Vmax), xlims=(0.001, 100), legend=false, title="β = $β")
    # Create a secondary y-axis for uu
    vv_max = maximum(vv_η)
    plot!(twinx(), η[2:end], vv_η[2:end], color=:red, label="vv_η", ylabel="vv/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*vv_max), xlims=(0.001, 100),legend=false)
    display(fig2)
    savefig(fig2, ofile_dir * "V_η_avg_β=$β.png")
end


function PlotStats4AllBeta()
    # Define the list of β values
    β_values = [0, 1, 39]
    
    # Loop over each β value
    for β in β_values
        if β == 0
            local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/ZPG/"
            local file_path = dir_path * "Beta0_UVWPPuuvvwwuv_eta_avg.txt"
        elseif β == 1
            local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/Beta=1/"
            local file_path = dir_path * "Beta1_UVWPPuuvvwwuv_eta_avg.txt"
        elseif β == 39
            local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/Beta=39/"
            local file_path = dir_path * "Beta39_UVWPPuuvvwwuv_eta_avg.txt"
        else
            continue # Skip to the next iteration if β is not in the predefined list
        end

        η, U, V, Ω_z, uu, vv, ww, uv = readAveragesFromFile(file_path)

        PlotStats(η, U, V, Ω_z, uu, vv, ww, uv, β, dir_path)
    end
end


# Function to generate minor ticks between powers of 10
function log_minor_ticks(start_pow, end_pow)
    major_ticks = [10^i for i in start_pow:end_pow]
    minor_ticks = []
    for i in start_pow:end_pow-1
        minor_ticks = [minor_ticks; [j*10^i for j in 2:9]]
    end
    return sort([major_ticks; minor_ticks]), major_ticks
end

# --------------- MAIN ----------------

# Define the list of β values
β_values = [0, 39]
    
# Loop over each β value
for β in β_values
    if β == 0
        local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/ZPG/"
        local file_path = dir_path * "Beta0_UVWPPuuvvwwuv_eta_avg.txt"
        global η_0, U_0, V_0, Ω_z_0, uu_0, vv_0, ww_0, uv_0 = readAveragesFromFile(file_path)
    elseif β == 1
        local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/Beta=1/"
        local file_path = dir_path * "Beta1_UVWPPuuvvwwuv_eta_avg.txt"
        global η_1, U_1, V_1, Ω_z_1, uu_1, vv_1, ww_1, uv_1 = readAveragesFromFile(file_path)
    elseif β == 39
        local dir_path  = "/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/Beta=39/"
        local file_path = dir_path * "Beta39_UVWPPuuvvwwuv_eta_avg.txt"
        global η_39, U_39, V_39, Ω_z_39, uu_39, vv_39, ww_39, uv_39 = readAveragesFromFile(file_path)
    else
        continue # Skip to the next iteration if β is not in the predefined list
    end
end

uu_max = max(maximum(uu_0), maximum(uu_39))


# Plot Mean Velocity in semi-log standard Form:
# plot the mean velocity profiles

# Generating minor and major ticks from 10^0 to 10^3
all_ticks, major_ticks = log_minor_ticks(-3.0, 1.0)
# Creating custom tick labels, empty for minor ticks
tick_labels = [in(t, major_ticks) ? @sprintf("%.3f", t) : "" for t in all_ticks]

#fig1 = plot(η_0[2:end], U_39[2:end], color=:green, linewidth=4, label="β = 0", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)", xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14, xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 10), xticks=(all_ticks, tick_labels),legend = false)
fig1 = plot(η_0[2:end], U_0[2:end], color=:green, linewidth=4, label="β = 0", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)", xtickfontsize=12, ytickfontsize=12, xguidefontsize=14, yguidefontsize=14, xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 10), xticks=(all_ticks, tick_labels), legend=:left, legendfontsize=18)
plot!(fig1, η_39[2:end], U_39[2:end], color=:red, linewidth=4, label="β = 39", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)", xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 10),xticks=(all_ticks, tick_labels))
# plot streamwise turbulence intensities
plot!(twinx(), η_0[2:end], uu_0[2:end], color=:green, linestyle=:dash, linewidth=2, label="β = 0, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
plot!(twinx(), η_39[2:end], uu_39[2:end], color=:red, linestyle=:dash, linewidth=2, label="β = 39, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
display(fig1)
ofile_dir="/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/"
savefig(fig1, ofile_dir * "U_η_avg_β=0_39.png")


# Plot Mean Velocity in Deficit Form:
# plot the mean velocity profiles
fig1 = plot(η_0[2:end], 1 .- U_0[2:end], color=:green, linewidth=2, label="β = 0, U/Uₑ(x)", xlabel="η = y/δ₁(x)", ylabel="1-U/Uₑ(x)", xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 10), legend=:left)
plot!(fig1, η_39[2:end], 1 .- U_39[2:end], color=:red, linewidth=2, label="β = 39, U/Uₑ(x)", xlabel="η = y/δ₁(x)", ylabel="1-U/Uₑ(x)", xscale=:log10, ylims=(0, 1.05), xlims=(0.001, 10))
# plot streamwise turbulence intensities
plot!(twinx(), η_0[2:end], uu_0[2:end], color=:green, linestyle=:dash, linewidth=2, label="β = 0, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
plot!(twinx(), η_39[2:end], uu_39[2:end], color=:red, linestyle=:dash, linewidth=2, label="β = 39, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
display(fig1)
ofile_dir="/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/"
savefig(fig1, ofile_dir * "Deficit_U_η_avg_β=0_39.png")


# Plot Mean Velocity in semi-log log on vertical axis Form:
# Generating minor and major ticks from 10^0 to 10^3
all_ticks, major_ticks = log_minor_ticks(-1.0, 0)
# Creating custom tick labels, empty for minor ticks
tick_labels = [in(t, major_ticks) ? @sprintf("%.3f", t) : "" for t in all_ticks]
# plot the mean velocity profiles
fig1 = plot(η_0[2:end], 1 .- U_0[2:end], color=:green, linewidth=2, label="β = 0, U/Uₑ(x)", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)", yscale=:log10, ylims=(0.1, 1.0), xlims=(0., 6), yticks=(all_ticks, tick_labels))
plot!(fig1, η_39[2:end], 1 .- U_39[2:end], color=:red, linewidth=2, label="β = 39, U/Uₑ(x)", xlabel="η = y/δ₁(x)", ylabel="U/Uₑ(x)",  yscale=:log10, ylims=(0.1, 1.), xlims=(0., 6))
# plot streamwise turbulence intensities
#plot!(twinx(), η_0[2:end], uu_0[2:end], color=:green, linestyle=:dash, linewidth=2, label="β = 0, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", yscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
#plot!(twinx(), η_39[2:end], uu_39[2:end], color=:red, linestyle=:dash, linewidth=2, label="β = 39, <uu>/(Uₑ(x)²)", ylabel="<uu>/(Uₑ(x)²)", yscale=:log10, ylims=(0, 1.05*uu_max), xlims=(0.001, 10), legend=false)
display(fig1)
ofile_dir="/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/"
savefig(fig1, ofile_dir * "Log_U_η_avg_β=0_39.png")

uv_max = maximum(abs.(uv_39))
fig1 = plot(η_39[2:end], -uv_39[2:end], color=:red, linestyle=:solid, linewidth=2, label="β = 39, -<uv>/(Uₑ(x)²)", xlabel="η = y/δ₁(x)", ylabel="-<uv>/(Uₑ(x)²)", xscale=:log10, ylims=(0, 1.05*uv_max), xlims=(0.001, 10), legend=false)
display(fig1)
ofile_dir="/Users/jsoria/CTRSS24/Datasets/LTRAC_data/DataFromVas/"
savefig(fig1, ofile_dir * "uv_η_avg_β=39.png")