#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using TBLDNSPostTools
using MPI
using HDF5
using ArgParse
using Printf
using Dates

#=======================================================================================================================================================#
# Main code
# - Initialize MPI communication
#=======================================================================================================================================================#
MPI.Init()

comm      = MPI.COMM_WORLD
rank      = MPI.Comm_rank(comm)
no_cpus   = MPI.Comm_size(comm)
root      = 0  # Define the root rank
plot_flag = false

VERSION = "1.0.0"


#====================================================================================#
# Define global variables
#====================================================================================#
global counter

#====================================================================================#
# Define ArgParse settings
#====================================================================================#

function mpi_show_help(settings::ArgParseSettings, err, err_code::Int = 1)
    if rank == root
        ArgParse.show_help(stdout, settings; exit_when_done=false)  # only root prints
        flush(stdout)  # ensure output is displayed
    end
    mpi_exit()
end

function mpi_exit(err_code::Int = 0)
    MPI.Barrier(comm)   # sync so all ranks wait
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()      # clean shutdown only if MPI is active
    end
    exit(err_code)
end


s = ArgParseSettings()
s.description = "Convert and collocate u, v, w, p fields from spectral to physical space and compute z-averaged fields. Version $VERSION"
s.commands_are_required = false
s.version = VERSION
s.add_version = true
s.add_help = false
s.exc_handler = mpi_show_help

@add_arg_table s begin
    "--calcGrad", "-g"
        help = "flag to calculate gradients"
        action = :store_true

    "--var", "-V"
        help = "variable to process, has to be a combination of 'u', 'v', 'w', 'p'"
        arg_type = String
        default = "uvwp"

    "--output_prefix", "-o"
        help = "the location and the root name of the output files, if left blank, input_prefix will be used"
        arg_type = String
        default = ""

    "--idMin"
        help = "the minimum file index to process (inclusive), by default, start from the minimum index found"
        arg_type = Int
        default = 0

    "--idMax"
        help = "the maximum file index to process (inclusive), by default, end at the maximum index found"
        arg_type = Int
        default = 0

    "--help", "-h"
        help = "show this help message and exit"
        action = :store_true
    
    "--dry-run"
        help = "parse and print arguments without actual computation"
        action = :store_true

    "--input_prefix_list"
        help = "a text file containing a list of input prefixes to process. Separate the file prefixes by new lines, will overwrite --input_prefix if both are provided."
        arg_type = String
        default = ""

    "--input_prefix"
        help = "the location and the root name of the input files"
        arg_type = String
    
    "--xMin"
        help = "the minimum x-index to process (inclusive)"
        required = true
        arg_type = Int
        required = true

    "--xMax"
        help = "the maximum x-index to process (inclusive)"
        required = true
        arg_type = Int
        required = true

end



#====================================================================================#
# Handle command-line arguments
# testing input filename and x-plane to be converted and plotted
#====================================================================================#
if rank == root
    args = parse_args(ARGS, s)

    if args["output_prefix"] == ""
        args["output_prefix"] = args["input_prefix"] == "" ? "./output" : args["input_prefix"]
    end


    println("Number of cores             : $(no_cpus)")
    println("Input field root name       : $(args["input_prefix"])")
    println("Output field root name      : $(args["output_prefix"])")
    println("Variable to process         : $(args["var"])")
    println("Calculate gradients         : $(args["calcGrad"] ? "Yes" : "No")")

    process_file_list = []
    exit_flag = false  # Flag to signal if we should exit

    if args["input_prefix"] == "" && args["input_prefix_list"] == ""
        println("Either --input_prefix or --input_prefix_list must be provided.")
        exit_flag = true
    elseif args["input_prefix_list"] != ""
        if isfile(args["input_prefix_list"])
            println("Processing input prefixes from file: $(args["input_prefix_list"])")
            input_prefixes = readlines(args["input_prefix_list"])
            # Remove empty lines and whitespace
            input_prefixes = filter(x -> !isempty(strip(x)), input_prefixes)
            if length(input_prefixes) == 0
                println("No valid input prefixes found in the provided file.")
                exit_flag = true
            else
                process_file_list = input_prefixes
                idMin = args["idMin"] == 0 ? 1 : args["idMin"]
                idMax = args["idMax"] == 0 ? length(process_file_list) : args["idMax"]
                process_file_list = process_file_list[idMin:idMax]
            end
        else
            println("The specified input_prefix_list file does not exist: $(args["input_prefix_list"])")
            exit_flag = true
        end
    else
        directory, file_name = splitdir(args["input_prefix"])
        if directory == ""
            directory = "."
        end
        findices = sort(find_matching_files(directory, file_name))
        
        if length(findices) == 0
            println("\nNo matching files found with the given input_prefix: '$(args["input_prefix"])'")
            exit_flag = true
        else
            idMin = args["idMin"] == 0 ? minimum(findices) : args["idMin"]
            idMax = args["idMax"] == 0 ? maximum(findices) : args["idMax"]

            # Populate process_file_list with full file paths
            process_file_list = [joinpath(directory, file_name * @sprintf(".%03d", id)) for id in idMin:idMax]
        end
    end
    args["process_file_list"] = process_file_list
else
    # Non-root processes initialize variables as nothing
    args = nothing
    process_file_list = nothing
    exit_flag = false
end

# Broadcast the exit flag to all processes
exit_flag = MPI.bcast(exit_flag, root, comm)

# All processes exit if there was an error
if exit_flag
    mpi_exit(1)
end

args = MPI.bcast(args, root, comm)
process_file_list = MPI.bcast(process_file_list, root, comm)

# Continue with the rest only if no error occurred
if rank == root
    args["idMin"] = args["idMin"] == 0 ? minimum(findices) : args["idMin"]
    args["idMax"] = args["idMax"] == 0 ? maximum(findices) : args["idMax"]

    xMin  = args["xMin"]
    xMax  = args["xMax"]

    println("xMin                        : $(xMin)")
    println("xMax                        : $(xMax)\n")
    println("process_file_list           : $(process_file_list)")

    if args["dry-run"]
        println("\nDry run mode finished, no actual computation performed.")
        # Signal dry run completion to all processes
        dry_run_flag = true
    else
        dry_run_flag = false
    end
else
    dry_run_flag = false
end

# Broadcast dry run flag to all processes
dry_run_flag = MPI.bcast(dry_run_flag, root, comm)

# All processes exit if it's a dry run
if dry_run_flag
    mpi_exit()
end



#====================================================================================#
# Open HDF5 files and read data, create storage (main) and broadcast parameters
#====================================================================================#
if rank == root
    # Read h5 pertinant paramters using the u file: 
    Re, lx, ly, lz, y, NX0, NY, NZ = read_para_h5((process_file_list[1]*".u"*".h5"))

    xMin = max(1, xMin)
    xMax = min(NX0, xMax)
    NX   = xMax - xMin + 1
    println("Updated NX, NY, NZ          : $NX, $NY, $NZ\n")

    # compute grids ...
    xgrid, ygrid, zgrid, xmidp, ymidp = compute_grids(lx, xMin, xMax, NX0, NX, y, lz, NZ)
else
    (Re, lx, ly, lz, NX0, NX, NY, NZ, y, xMin, xMax, idMin, idMax, xgrid, ygrid, zgrid, xmidp, ymidp) = 
        (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

# Broadcast parameters
Re       = MPI.bcast(Re,    root, comm)
lx       = MPI.bcast(lx,    root, comm)
ly       = MPI.bcast(ly,    root, comm)
lz       = MPI.bcast(lz,    root, comm)
NX0      = MPI.bcast(NX0,   root, comm)
NX       = MPI.bcast(NX,    root, comm)
NY       = MPI.bcast(NY,    root, comm)
NZ       = MPI.bcast(NZ,    root, comm)
y        = MPI.bcast(y,     root, comm)
xMin     = MPI.bcast(xMin,  root, comm)
xMax     = MPI.bcast(xMax,  root, comm)
xgrid    = MPI.bcast(xgrid, root, comm)
ygrid    = MPI.bcast(ygrid, root, comm)
zgrid    = MPI.bcast(zgrid, root, comm)
xmidp    = MPI.bcast(xmidp, root, comm)
ymidp    = MPI.bcast(ymidp, root, comm)




if rank == root
    # allocate storage for mean fields for each processed file (Nfiles, NY-1, NX-1)
    # !!! ACHTUNG: Collocation is on array size (NZ-2, NY-1, NX-1) !!!!
    Nfiles = length(process_file_list)
    u_mean = zeros(Float32, Nfiles, NY-1, NX-1) 
    v_mean = zeros(Float32, Nfiles, NY-1, NX-1)
    w_mean = zeros(Float32, Nfiles, NY-1, NX-1)
    p_mean = zeros(Float32, Nfiles, NY-1, NX-1)

    # allocate storage for Mean Squared fields for each processed file (Nfiles, NY-1, NX-1)
    #==
    uu_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    vv_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    ww_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    uv_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    uw_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    vw_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    pp_mean  = zeros(Float32, Nfiles, NY-1, NX-1)
    ==#

    global counter
    counter = 1
else
    (u_mean, v_mean, w_mean, p_mean, counter) = 
        (nothing, nothing, nothing, nothing, nothing)
end

# Process all the files writing, u, v, w, p collocated into the output h5 file

for process_file_name in process_file_list

    var = args["var"]
    grad_flag = args["calcGrad"]
    if rank == root
        println("\n[$(Dates.now())] Processing file: $process_file_name\n")
        # input filename for u, v, w & p
        filename_u = process_file_name * ".u" * ".h5"
        filename_v = process_file_name * ".v" * ".h5"
        filename_w = process_file_name * ".w" * ".h5"
        filename_p = process_file_name * ".p" * ".h5"

        file_index = split(process_file_name, ".")[end]
        # output filename to store u, v, w & p collocated in space
        var = args["var"]
        grad_flag = args["calcGrad"]
        grad_output_name = grad_flag ? "_grad" : ""



        if NX < NX0
            ofile_name = args["output_prefix"] * ".$(file_index)" * ".$(var)" * "$(grad_output_name)" *".phys.$(string(xMin))-$(string(xMax))" * ".h5"
            
        else
            ofile_name = args["output_prefix"] * ".$(file_index)" * ".$(var)" * "$(grad_output_name)" *".phys" * ".h5"
        end
        println("\nOutput file                 : $ofile_name\n")
        # open the output file
        fo = h5open(ofile_name, "w")
        
        # Write grid dimensions and domain lengths to the output file
        # This ensures lx, ly, lz are correctly stored in each individual file
        dset = create_dataset(fo, "/xgrid", datatype(eltype(xgrid[1:end-1])), dataspace(size(xgrid[1:end-1])))
        write(dset, collect(xgrid[1:end-1]))
        dset = create_dataset(fo, "/ygrid", datatype(eltype(ygrid)), dataspace(size(ygrid)))
        write(dset, ygrid)
        dset = create_dataset(fo, "/zgrid", datatype(eltype(zgrid[1:end-2])), dataspace(size(zgrid[1:end-2])))
        write(dset, collect(zgrid[1:end-2]))
        dset = create_dataset(fo, "/lx", datatype(eltype(lx)), dataspace(size(lx)))
        write(dset, lx)
        dset = create_dataset(fo, "/ly", datatype(eltype(ly)), dataspace(size(ly)))
        write(dset, ly)
        dset = create_dataset(fo, "/lz", datatype(eltype(lz)), dataspace(size(lz)))
        write(dset, lz)
        dset = create_dataset(fo, "/Re", datatype(eltype(Re)), dataspace(size(Re)))
        write(dset, Re)
    else
        (filename_u, filename_v, filename_w, filename_p, ofile_name, fo) = (nothing, nothing, nothing, nothing, nothing, nothing)
    end
    MPI.Barrier(comm)

    if occursin("u", var)

        #====================================================================================#
        # processing the u-velocity:
        #====================================================================================#
        if rank == root
            println("\nDoing U-kind variable\n")
            global_array = read_value_h5(filename_u, xMin, xMax)
            println("\nFinished reading from '$filename_u': $(size(global_array))\n")
        else
            global_array = nothing
        end
        # stores the Fourier field for[xMin, xMax]

        # interpolate in x, interpolate in y & convert Fourier -> Phys: u
        u_phys_results = toPhysAndCollocate_u_VGT_h5(global_array, xgrid, xmidp, ygrid, ymidp, zgrid, root, comm, fo = fo, VGT = grad_flag)
        MPI.Barrier(comm)
        # extract just the u_phys field (first element) from the returned tuple
        u_phys = isa(u_phys_results, Tuple) ? u_phys_results[1] : u_phys_results
        # compute the z-averaged field and add it to u_mean storage
        u_buf = compute_z_average_field(u_phys, root, comm)
        MPI.Barrier(comm)
        # only root stores the z-averaged field
        if rank == root
            u_mean[counter, :, :] = u_buf
            u_buf                 = nothing
        end
        MPI.Barrier(comm)
        # free memory
        u_phys       = nothing
        global_array = nothing   
    
    end

    if occursin("v", var)

        #====================================================================================#
        # processing the v-velocity:
        #====================================================================================#
        if rank == root
            println("\nDoing V-kind variable\n")
            global_array = read_value_h5(filename_v, xMin, xMax)
            println("\nFinished reading from '$filename_v': $(size(global_array))\n")
        else
            global_array = nothing
        end
        # convert Fourier -> Phys: v
        v_phys_results = toPhysAndCollocate_v_VGT_h5(global_array, ygrid, zgrid, xgrid, root, comm, fo = fo, VGT = grad_flag)
        MPI.Barrier(comm)
        # extract just the v_phys field (first element) from the returned tuple
        v_phys = isa(v_phys_results, Tuple) ? v_phys_results[1] : v_phys_results
        # compute the z-averaged field and add it to v_mean storage
        v_buf = compute_z_average_field(v_phys, root, comm)
        MPI.Barrier(comm)
        # only root stores the z-averaged field
        if rank == root
            v_mean[counter, :, :] = v_buf
            v_buf                 = nothing
        end
        MPI.Barrier(comm)
        # free memory
        v_phys       = nothing
        global_array = nothing

    end

    if occursin("w", var) 
        #====================================================================================#       
        # processing the w-velocity:
        #====================================================================================#
        if rank == root
            println("\nDoing W-kind variable\n")
            global_array = read_value_h5(filename_w, xMin, xMax)
            println("\nFinished reading from '$filename_w': $(size(global_array))\n")
        else
            global_array = nothing
        end
        # interpolate in y & convert Fourier -> Phys: w, p, T, E_T
        w_phys_results = toPhysAndCollocate_w_VGT_h5(global_array, ygrid, ymidp, zgrid, xgrid, root, comm, pflag = false, fo = fo, VGT = grad_flag)
        MPI.Barrier(comm)
        # extract just the w_phys field (first element) from the returned tuple
        w_phys = isa(w_phys_results, Tuple) ? w_phys_results[1] : w_phys_results
        # compute the z-averaged field and add it to w_mean storage
        w_buf = compute_z_average_field(w_phys, root, comm)
        MPI.Barrier(comm)
        # only root stores the z-averaged field
        if rank == root
            w_mean[counter, :, :] = w_buf
            w_buf                 = nothing
        end
        MPI.Barrier(comm)
        # free memory
        w_phys       = nothing
        global_array = nothing
    end

    if occursin("p", var)

        #====================================================================================#
        # processing the pressure:
        #====================================================================================#
        if rank == root
            println("\nDoing P-kind variable\n")
            global_array = read_value_h5(filename_p, xMin, xMax)
            println("\nFinished reading from '$filename_p': $(size(global_array))\n")
        else
            global_array = nothing
        end
        # interpolate in y & convert Fourier -> Phys: w, p, T, E_T
        p_phys_results = toPhysAndCollocate_w_VGT_h5(global_array, ygrid, ymidp, zgrid, xgrid, root, comm, pflag = true, fo = fo, VGT = grad_flag)
        MPI.Barrier(comm)
        # extract just the p_phys field (first element) from the returned tuple
        p_phys = isa(p_phys_results, Tuple) ? p_phys_results[1] : p_phys_results
        # compute the z-averaged field and add it to p_mean storage
        p_buf = compute_z_average_field(p_phys, root, comm)
        MPI.Barrier(comm)
        # only root stores the z-averaged field
        if rank == root
            p_mean[counter, :, :] = p_buf
            p_buf                 = nothing
        end
        MPI.Barrier(comm)
        # free memory
        p_phys       = nothing
        global_array = nothing
    end

    #====================================================================================#
    # write out additional parameters to the output file
    #====================================================================================#
    if rank == root
        # use collect on the following to avoid HDF5 complaints !!!
        println("\nWriting to output h5 file: grid and parameter info")
        dset = safe_create_dataset(fo, "/x", datatype(eltype(xgrid)), dataspace(size(xgrid)))
        write(dset, collect(xgrid))
        dset = safe_create_dataset(fo, "/xgrid", datatype(eltype(xgrid[1:end-1])), dataspace(size(xgrid[1:end-1])))
        write(dset, collect(xgrid[1:end-1]))  # should strictly speaking be xmidp[1:end]
        dset = safe_create_dataset(fo, "/y", datatype(eltype(y)), dataspace(size(y)))
        write(dset, collect(y))
        dset = safe_create_dataset(fo, "/ygrid", datatype(eltype(ygrid)), dataspace(size(ygrid)))
        write(dset, ygrid)
        dset = safe_create_dataset(fo, "/z", datatype(eltype(zgrid[1:end-1])), dataspace(size(zgrid[1:end-1])))
        write(dset, collect(zgrid[1:end-1]))
        dset = safe_create_dataset(fo, "/zgrid", datatype(eltype(zgrid[1:end-2])), dataspace(size(zgrid[1:end-2])))
        write(dset, collect(zgrid[1:end-2]))
        dset = safe_create_dataset(fo, "/lx", datatype(eltype(lx)), dataspace(size(lx)))
        write(dset, lx)
        dset = safe_create_dataset(fo, "/lz", datatype(eltype(lz)), dataspace(size(lz)))
        write(dset, lz)
        dset = safe_create_dataset(fo, "/ly", datatype(eltype(ly)), dataspace(size(ly)))
        write(dset, ly)
        dset = safe_create_dataset(fo, "/Re", datatype(eltype(Re)), dataspace(size(Re)))
        write(dset, Re)
        close(fo)
    end

    # Increment counter
    if rank == root
        global counter
        counter += 1
    end
end


#===============================================================================================#
# compute the z-averaged fields for all the processed files & write them to the output HDF5 file
#===============================================================================================#
if rank == root
    println("\nComputing z-averaged fields for all files")
end

# Process each field separately instead of passing them as a tuple
U = compute_z_average_field(u_mean, root, comm)
V = compute_z_average_field(v_mean, root, comm)
W = compute_z_average_field(w_mean, root, comm)
P = compute_z_average_field(p_mean, root, comm)
MPI.Barrier(comm)

if rank == root
    println("\nDone\n")
    NoSamples = length(findices) * length(zgrid[1:end-2])
    println("Number of samples           : $NoSamples\n")
    # output file name to store the mean fields
    if NX < NX0
        ofile_name_mean = ARGS[1] * raw".Mean_uvwp_phys__" * string(xMin) * ":" * string(xMax) * ".h5"
    else
        ofile_name_mean = ARGS[1] * raw".Mean_uvwp_phys" * ".h5"
    end
    println("\nOutput file for Mean Fields : $ofile_name_mean\n") 
    # open the output file
    of = h5open(ofile_name_mean, "w")
    # write the mean fields to the output file
    println("\nWriting to output h5 file: Mean Fields")
    dset = safe_create_dataset(of, "/U", datatype(eltype(U)), dataspace(size(U)))
    write(dset, U)
    dset = safe_create_dataset(of, "/V", datatype(eltype(V)), dataspace(size(V)))
    write(dset, V)
    dset = safe_create_dataset(of, "/W", datatype(eltype(W)), dataspace(size(W)))
    write(dset, W)
    dset = safe_create_dataset(of, "/P", datatype(eltype(P)), dataspace(size(P)))
    write(dset, P)
    # write grids and parameters to the output file
    dset = safe_create_dataset(of, "/xgrid", datatype(eltype(xgrid[1:end-1])), dataspace(size(xgrid[1:end-1])))
    write(dset, collect(xgrid[1:end-1]))
    dset = safe_create_dataset(of, "/ygrid", datatype(eltype(ygrid)), dataspace(size(ygrid)))
    write(dset, ygrid)
    dset = safe_create_dataset(of, "/lx", datatype(eltype(lx)), dataspace(size(lx)))
    write(dset, lx)
    dset = safe_create_dataset(of, "/lz", datatype(eltype(lz)), dataspace(size(lz)))
    write(dset, lz)
    dset = safe_create_dataset(of, "/ly", datatype(eltype(ly)), dataspace(size(ly)))
    write(dset, ly)
    dset = safe_create_dataset(of, "/Re", datatype(eltype(Re)), dataspace(size(Re)))
    write(dset, Re)
    # write the number of samples to the output file
    dset = safe_create_dataset(of, "/NoSamples", datatype(eltype(NoSamples)), dataspace(size(NoSamples)))
    write(dset, NoSamples)
    close(of)
end


# Measure and print RAM usage at the end of execution
if rank == root
    total_memory = Sys.total_memory()
    free_memory  = Sys.free_memory()
    used_memory  = total_memory - free_memory
    println("Total memory: ", total_memory, " bytes")
    println("Free memory : ", free_memory, " bytes")
    println("Used memory : ", used_memory, " bytes")
    println(@sprintf("Used memory : %.2f GB", used_memory / 1e9))
end

# Finalize MPI
MPI.Finalize()


