#!/usr/bin/env julia
# -*- coding: utf-8 -*-

#copied from Prof. Soria's TBL-PostProc-Julia package and only functions are retained, no main program here.


#========================================================================================================#
# Code to convert Fourier -> Physical space ALL TBL HDF5 files
#
#    @AUTHOR   : Julio Soria
#    email     : julio.soria@monash.edu
#    date      : 14-10-2024
#
#    This code converts u, v, w, p Fourier -> Collocated Physical Space Points:
#        1.     Import necessary Julia packages.
#        2.     Initialize MPI communication.
#        3.     Handle command-line arguments.
#        4.     Open HDF5 files and read data.
#        5.     Broadcast necessary parameters.
#        6.     Create grids for interpolation.
#        7.     Handle all cases based on the variable type (u, v, w, temp, totalenergy).
#        8.     Allocate memory and perform interpolation.
#        9.     Perform IRFFT operations.
#        10.    Write results to ONE HDF5 file for each restart file ID
#        11.    Close files and finalize.
#========================================================================================================#


#========================================================================================================#
# Import necessary Julia packages
#========================================================================================================#
using MPI
using HDF5
using LinearAlgebra
using Dierckx
using Printf
using Interpolations
using StatsBase
using StaticArrays
using Statistics
using FFTW


include("./NumericalFunctions/PadeDifferentiation.jl")
include("./NumericalFunctions/FourierDifferentiation.jl")


#========================================================================================================#
# FUNCTIONS
#========================================================================================================#

########################################### HELPER FUNCTIONS #############################################

function print_3d_array(array::Array{T, 3}) where T
#====================================================================================#
# Print 3D array
#====================================================================================#
"""
    Print the dimensions and slices of a 3D array.
    Arguments:
        - array: The 3D array to print.

    Returns:
        - None
"""
    dims = size(array)
    println("dimensions = $dims")
    for i in 1:dims[3]
        println("x-Slice $i:")
        for j in 1:dims[1]
            println(array[j, :, i])
        end
        println()
    end
end


function split_count(N::Integer, n::Integer)
#====================================================================================#
# split_count(N::Integer, n::Integer)
#
# Return a vector of `n` integers that are approximately equally sized and sum to `N`.
#====================================================================================#
"""
This function splits the integer N into n parts, distributing the remainder evenly.
    This function is useful for parallel processing or distributing work among
    multiple tasks.

    Arguments:
        - N: The total number to be split.
        - n: The number of parts to split into.

    Returns:
        - A vector of integers representing the split counts.
"""
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i = 1:n]
end


function bcast_eltype(variable, root, comm)
#====================================================================================#
# Function to broadcast the element type of a variable
#====================================================================================#
"""
Broadcast the element type of a variable across MPI processes.
This function is useful for ensuring that all processes have the same understanding
of the variable's type, especially when dealing with distributed data.

    Arguments:
        - variable: The variable whose element type is to be broadcasted.
        - root:     The rank of the root process.
        - comm:     The MPI communicator.

    Returns:
        - eltype_type: The element type of the variable.
"""
    rank = MPI.Comm_rank(comm)

    if rank == root
        eltype_str = string(eltype(variable))
    else
        eltype_str = ""
    end

    # Broadcast the string representation of the element type
    eltype_str = MPI.bcast(eltype_str, root, comm)

    # Convert the string back to the type
    eltype_type = eval(Meta.parse(eltype_str))

    return eltype_type
end


function cast_to_higher_resolution(f, g)
#====================================================================================#
# Cast a 3D array type to a higher resolution, e.g. Float32 to Float64
#====================================================================================#
"""
Cast a 3D array type to a higher resolution, e.g. Float32 to Float64.
    This function ensures that both arrays have the same element type,
    and if they differ, it casts the lower resolution array to the higher resolution type.

    Arguments:
        - f: The first array to cast.
        - g: The second array to cast.

    Returns:
        - f: The casted first array.
        - g: The casted second array.
"""
    if eltype(f) != eltype(g)
        if Base.elsize(f) < Base.elsize(g)
            # f has lower resolution, cast f to the type of g
            f = convert(Array{eltype(g)}, f)
        else
            # g has lower resolution, cast g to the type of f
            g = convert(Array{eltype(f)}, g)
        end
    end
    return f, g
end


function cast_to_lower_resolution(f, g)
#====================================================================================#
# Cast a 3D array type to a lower resolution, e.g. Float64 to Float32
#====================================================================================#
"""
Cast a 3D array type to a lower resolution, e.g. Float64 to Float32.
    This function ensures that both arrays have the same element type,
    and if they differ, it casts the higher resolution array to the lower resolution type.

    Arguments:
        - f: The first array to cast.
        - g: The second array to cast.

    Returns:
        - f: The casted first array.
        - g: The casted second array.
"""
    if eltype(f) != eltype(g)
        if Base.elsize(f) > Base.elsize(g)
            # f has lower resolution, cast f to the type of g
            f = convert(Array{eltype(g)}, f)
        else
            # g has lower resolution, cast g to the type of f
            g = convert(Array{eltype(f)}, g)
        end
    end
    return f, g
end


function eltype_to_symbol(element_type::Type)
"""
    Convert a Julia type to its corresponding symbol representation.
    Arguments:
        - element_type: The Julia type to convert.
    Returns:
        - A symbol representing the type.
"""
    if element_type == Int8
        return :Int8
    elseif element_type == Int16
        return :Int16
    elseif element_type == Int32
        return :Int32
    elseif element_type == Int64
        return :Int64
    elseif element_type == UInt8
        return :UInt8
    elseif element_type == UInt16
        return :UInt16
    elseif element_type == UInt32
        return :UInt32
    elseif element_type == UInt64
        return :UInt64
    elseif element_type == Float16
        return :Float16
    elseif element_type == Float32
        return :Float32
    elseif element_type == Float64
        return :Float64
    elseif isdefined(Base, :Float8) && element_type == Base.Float8
        return :Float8
    else
        error("Unsupported element type.")
    end
end


function get_int_type_symbol(x)
"""
    Get the symbol representation of the integer type of an array.
    Arguments:
        - x: The array whose integer type is to be determined.
    Returns:
        - A symbol representing the integer type.
"""
    int_type = eltype(x)
    if int_type == Int8
        return :Int8
    elseif int_type == Int16
        return :Int16
    elseif int_type == Int32
        return :Int32
    elseif int_type == Int64
        return :Int64
    else
        error("Unsupported integer type.")
    end
end


function get_float_type_symbol(x)
"""
    Get the symbol representation of the floating-point type of an array.
    Arguments:
        - x: The array whose floating-point type is to be determined.
    Returns:
        - A symbol representing the floating-point type.
"""
    float_type = eltype(x)
    if float_type == Float16
        return :Float16
    elseif float_type == Float32
        return :Float32
    elseif float_type == Float64
        return :Float64
    elseif isdefined(Base, :Float8) && float_type == Base.Float8
        return :Float8
    else
        error("Unsupported floating-point type.")
    end
end


function convert_to_integer_array(array::Array{T}, int_type::Symbol) where T<:AbstractFloat
"""
    Convert a floating-point array to an unsigned integer array, automatically scaling to the full range of the target type.
    Arguments:
        - array:        The floating-point array to convert.
        - int_type:     The desired unsigned integer type (e.g., :UInt8, :UInt16, :UInt32).
    Returns:
        - int_array:    The converted unsigned integer array.
        - min_val:      The minimum value of the original array.
        - max_val:      The maximum value of the original array.
    Note:
        - The input array is first normalized to the range [0, 1].
        - The conversion is done by scaling the normalized values to the maximum range of the specified unsigned integer type and rounding down using floor.
        - The resulting unsigned integer array will have the same dimensions as the input array. 
"""
    # normalize the array to the range [0, 1] using its minimum and maximum values:
    min_val = minimum(array)
    max_val = maximum(array)
    # Ensure max_val is different from min_val to avoid division by zero
    if max_val == min_val
        # Handle the case where all elements are the same
        @warn "All elements in the array are identical. Normalization results in zeros."
        normalized_array = zeros(T, size(array)) # Use original type T or a default like Float64
    else
        normalized_array = (array .- min_val) ./ (max_val - min_val)
    end
    
    # Determine the unsigned integer type and the corresponding scale factor
    int_array  = nothing
    int_factor = 1.0
    if int_type == :UInt8
        int_factor = Float64(typemax(UInt8)) # 255.0
        int_array  = UInt8.(round.(normalized_array .* int_factor))
    elseif int_type == :UInt16
        int_factor = Float64(typemax(UInt16)) # 65535.0
        int_array  = UInt16.(round.(normalized_array .* int_factor))
    elseif int_type == :UInt32
        int_factor = Float64(typemax(UInt32)) # 4294967295.0
        int_array  = UInt32.(round.(normalized_array .* int_factor))
    else
        error("Unsupported unsigned integer type. Use :UInt8, :UInt16, or :UInt32.")
    end
    return int_array, min_val, max_val, int_factor
end


function convert_to_float_array(int_array::Array{T}, float_type::Symbol, min_val::Float64, max_val::Float64; int_factor::Float64=0.0) where T<:Unsigned
"""
    Convert an unsigned integer array back to a floating-point array using original min/max values.
    Arguments:
        - int_array:     The unsigned integer array to convert.
        - float_type:    The desired floating-point type (e.g., :Float8, :Float16, :Float32, :Float64).
        - min_val:       The minimum value of the original floating-point array.
        - max_val:       The maximum value of the original floating-point array.
    Returns:
        - float_array:   The converted floating-point array.
    Note:
        - The conversion scales the integer values based on the target integer type's maximum value and then denormalizes using min_val and max_val.
        - The resulting floating-point array will have the same dimensions as the input array.
"""
    # Determine the scale factor based on the integer type if not provided as a parameter
    int_type = eltype(int_array)
    if int_factor == 0
        if int_type == UInt8
            scale_factor = Float64(typemax(UInt8))
        elseif int_type == UInt16
            scale_factor = Float64(typemax(UInt16))
        elseif int_type == UInt32
            scale_factor = Float64(typemax(UInt32))
        else
            error("Unsupported unsigned integer type in input array.")
        end
    else
        scale_factor = int_factor
    end

    # Determine the floating-point type based on the label and perform conversion
    float_array      = nothing
    temp_float_array = Float64.(int_array) ./ scale_factor # Convert to intermediate Float64 for precision
    
    # Denormalize the array
    denormalized_array = temp_float_array .* (max_val - min_val) .+ min_val

    # Convert to the target float type
    if float_type == :Float8
        @assert isdefined(Base, :Float8) "Float8 is not defined in this Julia version."
        float_array = Base.Float8.(denormalized_array)
    elseif float_type == :Float16
        float_array = Float16.(denormalized_array)
    elseif float_type == :Float32
        float_array = Float32.(denormalized_array)
    elseif float_type == :Float64
        float_array = Float64.(denormalized_array) # Already Float64, but explicit conversion is fine
    else
        error("Unsupported floating-point type. Use :Float8, :Float16, :Float32, or :Float64.")
    end
    return float_array
end


function get_chunk_dims(n₁, n₂, n₃, eltype_size::Int, optimal_chunk_size::Int=1024^2)
"""
Calculate optimal chunk sizes for a 3D array.
    This function calculates the optimal chunk sizes for a 3D array based on the
    dimensions of the array and the size of each element type. The function aims to
    balance the chunk sizes across all three dimensions while keeping the total size
    within a specified optimal range.

    Arguments:
        - n₁:                   Number of elements in the 1st-index-dimension.
        - n₂:                   Number of elements in the 2nd-index-dimension.
        - n₃:                   Number of elements in the 3rd-index-dimension.
        - eltype_size:          Size of each element type in bytes.
        - optimal_chunk_size:   Desired size for each chunk in bytes.

    Returns:
        - A tuple containing the optimal chunk sizes for each dimension (x₁, x₂, x₃).
"""
    # Calculate the total size of the array in bytes
    total_size     = n₁ * n₂ * n₃ * eltype_size

    # Determine the scaling factor to bring the chunk size within the optimal range
    scaling_factor = max(1, ceil(Int, total_size / optimal_chunk_size))

    # Simple chunk sizing strategy - try to make chunks approximately cubic
    # but ensure they're not larger than the original dimensions
    chunk_size_x₁  = min(n₁, max(1, ceil(Int, n₁ / scaling_factor^(1/3))))
    chunk_size_x₂  = min(n₂, max(1, ceil(Int, n₂ / scaling_factor^(1/3))))
    chunk_size_x₃  = min(n₃, max(1, ceil(Int, n₃ / scaling_factor^(1/3))))

    return (chunk_size_x₁, chunk_size_x₂, chunk_size_x₃)
end



############################################ I/O FUNCTIONS ############################################

function find_matching_files(directory, base_name)
"""
Find all files in the given directory that match the specified base name and pattern.
    This function searches for files in the specified directory that match the given
    base name and a specific pattern.
    The pattern is defined as files that start with the base name, followed by a dot,
    three digits, and end with ".u.h5".

    Argument:
        - directory: The directory to search for files.
        - base_name: The base name of the files to match.

    Returns:
        - A vector of indices extracted from the matching filenames.
"""
    # List all files in the directory
    files = readdir(directory)

    # Define the regular expression pattern to match the filenames
    pattern = r"^" * base_name * r"\.\d{3}\.u\.h5$"

    # Find all matching files and extract the indices
    matching_files = filter(f -> occursin(pattern, f), files)
    indices = [parse(Int, match(r"\d{3}", f).match) for f in matching_files]

    return indices
end


function get_hdf5_variable_size(filename, variable)
"""
Get the size of a variable in an HDF5 file.
    This function opens an HDF5 file, retrieves the specified variable, and returns its dimensions.
    The function is useful for checking the size of variables in HDF5 files without loading the entire dataset.
    This is particularly useful for large datasets where loading the entire variable may be inefficient.
        
    Arguments:
        - filename: The name of the HDF5 file.
        - variable: The name of the variable to check.

    Returns:
        - dims: The dimensions of the variable in the HDF5 file.
"""
    # Open the HDF5 file
    fin = h5open(filename, "r")
    # Get the dataset
    dset = fin[variable]
    # Get the size of the dataset
    dims = size(dset)
    # Close the HDF5 file
    close(fin)
    return dims
end


function read_para_h5(filename)
#====================================================================================#
# Read perinant paramters from h5 file from TBL DNS ...
#====================================================================================#
"""
Read parameters from an HDF5 file.
    This function opens an HDF5 file, retrieves the specified parameters, and returns them.
    The function is useful for extracting simulation parameters from HDF5 files.
    
    Arguments:
        - filename: The name of the HDF5 file to read.

    Returns:
        - Re: Reynolds number = 1/nu = 1/(kinematic viscosity).
        - lx: Length in x-direction.
        - ly: Length in y-direction.
        - lz: Length in z-direction.
        - y:  y-coordinates.
        - NX: Number of points in the x-direction.
        - NY: Number of points in the y-direction.
        - NZ: Number of points in the z-direction.
"""
    println("Reading parameters from file: $filename\n")

    # Read data from HDF5
    fin          = h5open(filename, "r")
    Re           = read(fin["/Re"])[1]
    lx           = read(fin["/lx"])[1]
    ly           = read(fin["/ly"])[1]
    lz           = read(fin["/lz"])[1]
    y            = read(fin["/y"])
    y            = SVector{y.size[1], Float64}(y...)
    close(fin)
    (NZ, NY, NX) = get_hdf5_variable_size(filename, "/value")
    println("lX, lY, lZ                  : $lx $ly $lz")
    println("NX, NY, NZ                  : $NX $NY $NZ")
    return Re, lx, ly, lz, y, NX, NY, NZ
end


function read_value_h5(filename, xMin, xMax)
#====================================================================================#
# Read h5 file from TBL DNS ...
# achtung: reads the 'value' complex field between x \in[xNin, xMax]
# value[1:NY, 1:NZ, 1:NX] = read(fin["/value"])  # reads the entire 'value' complex field ...
#
#====================================================================================#
"""
Read the 'value' field from an HDF5 file.
    This function opens an HDF5 file, retrieves the 'value' field, and returns it.
    The function is useful for extracting the 'value' field from HDF5 files.
    
    Arguments:
        - filename: The name of the HDF5 file to read.
        - xMin: Minimum x-coordinate.
        - xMax: Maximum x-coordinate.

    Returns:
        - value: The 'value' field in the specified range.
"""
    println("Reading field (only) from file: ", filename)
    # Read data from HDF5
    fin   = h5open(filename, "r")
    value = fin["/value"][ :, :, xMin:xMax]  # reads the 'value' complex field between x \in[xNin, xMax]...
    close(fin)
    return value
end


function read_h5(filename)
#====================================================================================#
# Read h5 file from TBL DNS ...
#====================================================================================#
"""
Read parameters and value from an HDF5 file.
    This function opens an HDF5 file, retrieves the specified parameters and value,
    and returns them. The function is useful for extracting simulation parameters
    and the 'value' field from HDF5 files.

    Arguments:
        - filename: The name of the HDF5 file to read.

    Returns:
        - lx: Length in x-direction.
        - ly: Length in y-direction.
        - lz: Length in z-direction.
        - y:  y-coordinates.
        - NX: Number of points in the x-direction.
        - NY: Number of points in the y-direction.
        - NZ: Number of points in the z-direction.
"""
    println("Reading data from file: ", filename)
    # Read data from HDF5
    fin           = h5open(filename, "r")
    lx            = read(fin["/lx"])[1]
    ly            = read(fin["/ly"])[1]
    lz            = read(fin["/lz"])[1]
    y             = read(fin["/y"])
    y             = SVector{y.size[1], Float64}(y...)
    value         = read(fin["/value"])               # reads the entire 'value' complex field ...
    close(fin)
    (NZ, NY, NX)  = value.size # z-stored first, y-stored second, x-stored last

    println("lX, lY, lZ: $lx $ly $lz")
    println("NX, NY, NZ: $NX $NY $NZ")
    return lx, ly, lz, y, NX, NY, NZ, value
end


function determine_best_unsigned_type(array::Array{T}) where T<:AbstractFloat
#====================================================================================#
# Determine the most appropriate unsigned integer type for data compression
# based on the dynamic range of the data
#====================================================================================#
"""
Determine the most appropriate unsigned integer type for data compression.
    This function determines the most appropriate unsigned integer type for compressing
    the input array based on the dynamic range of the data.
    The function estimates the number of bits required to adequately represent the data's
    precision and returns the corresponding unsigned integer type.

    Arguments:
        - array: The array to analyze.

    Returns:
        - A symbol representing the recommended unsigned integer type (:UInt8, :UInt16, or :UInt32).
"""
    # Get the range of values in the array
    min_val    = minimum(array)
    max_val    = maximum(array)
    data_range = abs(max_val - min_val)
    
    if data_range == 0
        # If all values are the same, UInt8 is enough
        return :UInt8
    end
    
    # Estimate the precision required (in bits)
    # For floating point data, analyze the significant digits
    if T == Float32
        # Float32 has ~7 decimal digits of precision, which needs ~24 bits
        precision_bits = 24
    elseif T == Float64
        # Float64 has ~15-17 decimal digits of precision, which needs ~53 bits
        precision_bits = 53
    else
        # Default for other float types
        precision_bits = 16
    end
    
    # Analyze the actual data to see if we can use fewer bits
    non_integer_values = any(x -> x != floor(x), array)
    
    if !non_integer_values
        # If all values are effectively integers, we can use much fewer bits
        max_integer = max(abs(max_val), abs(min_val))
        if max_integer <= 127
            return :UInt8  # 8 bits can represent values 0-255
        elseif max_integer <= 32767
            return :UInt16 # 16 bits can represent values 0-65535
        else
            return :UInt32 # 32 bits for larger integers
        end
    end
    
    # For floating point data with fractional components
    # Check for the ratio of max value to the smallest non-zero difference
    # to determine the dynamic range
    sample = array[1:min(1000, length(array))]  # Sample to avoid expensive operation
    sorted_unique = sort(unique(sample))
    if length(sorted_unique) > 1
        diffs = diff(sorted_unique)
        min_diff = minimum(filter(x -> x > 0, diffs))
        dynamic_range = data_range / min_diff
        
        # Determine bits needed based on dynamic range
        bits_needed = ceil(Int, log2(dynamic_range))
        
        if bits_needed <= 8
            return :UInt8
        elseif bits_needed <= 16
            return :UInt16
        else
            return :UInt32
        end
    end
    
    # Default based on original data type
    if T == Float32
        return :UInt16  # Typically good for Float32
    else
        return :UInt32  # Better preservation for Float64
    end
end


function write_compressed_dataset(fo, dataset_name, data; compressed::Bool=false, compression_level::Int=9, auto_select_type::Bool=true)
#====================================================================================#
# Function to write a dataset to an HDF5 file with optional compression
# - If compression is requested, the function converts the data to an unsigned integer format
# - It also stores metadata to allow reconstruction of the original data
# - Uses optimal chunking to maximize performance
# - Can automatically select the best unsigned integer type for compression
#====================================================================================#
"""
Write a dataset to an HDF5 file with optional compression.
    This function writes a dataset to an HDF5 file with optional compression.
    If compression is requested, the function converts the data to an unsigned integer format
    and stores metadata to allow reconstruction of the original data.
    The function also uses optimal chunking to maximize performance.
    It can automatically select the best unsigned integer type for compression based on
    the data's characteristics.

    Arguments:
        - fo:               The HDF5 file object to write to.
        - dataset_name:     The name of the dataset to create (e.g., "/u", "/v", "/w").
        - data:             The data to write to the dataset.
        - compressed:       Optional argument to indicate whether to use compression.
        - compression_level: Optional argument to specify the compression level (0-9).
        - auto_select_type: Optional argument to indicate whether to automatically select the
                          best unsigned integer type for compression (true by default).

    Returns:
        - Nothing
"""
    # Get dimensions and element type
    dims      = size(data)
    data_type = eltype(data)
    
    # Calculate optimal chunk sizes
    chunk_dims = get_chunk_dims(dims..., sizeof(data_type))
    
    if compressed
        # Check if compression would be effective
        if !(data_type <: Unsigned)
            # Determine the best integer type for compression if auto_select_type is true
            int_type = auto_select_type ? determine_best_unsigned_type(data) : :UInt16
            
            # Convert to compressed integer format
            int_array, min_val, max_val, scale_factor = convert_to_integer_array(data, int_type)
            
            # Store reconstruction attributes
            attrs(fo)[dataset_name * "_DataType"]          = string(data_type)
            attrs(fo)[dataset_name * "_COMPRESSION_SCALE"] = scale_factor
            attrs(fo)[dataset_name * "_Min_Value"]         = min_val
            attrs(fo)[dataset_name * "_Max_Value"]         = max_val
            attrs(fo)[dataset_name * "_Original_Size"]     = sizeof(data)
            attrs(fo)[dataset_name * "_Compressed_Size"]   = sizeof(int_array)
            attrs(fo)[dataset_name * "_Compression_Type"]  = string(int_type)
            
            # Write array with chunking and compression filter
            fo[dataset_name, chunk=chunk_dims, compress=compression_level] = int_array
            
            # Print compression statistics
            original_size     = sizeof(data)
            compressed_size   = sizeof(int_array)
            compression_ratio = original_size / compressed_size
            println("Compression statistics for $dataset_name:")
            println("  Original size: $(original_size/1024^2) MB")
            println("  Compressed size: $(compressed_size/1024^2) MB")
            println("  Compression ratio: $(round(compression_ratio, digits=2)):1")
            println("  Using type: $int_type")
        else
            # Already an unsigned type, write directly with compression
            fo[dataset_name, chunk=chunk_dims, compress=compression_level] = data
        end
    else
        # Write array with chunking, no compression
        fo[dataset_name, chunk=chunk_dims] = data
    end
end


function read_compressed_dataset(fin, dataset_name)
#====================================================================================#
# Function to read a dataset from an HDF5 file, decompressing if necessary
# - Automatically detects if the dataset was compressed
# - Uses metadata stored in attributes to decompress the data
# - Returns the original data as the specified type
#====================================================================================#
"""
Read a dataset from an HDF5 file, decompressing if necessary.
    This function reads a dataset from an HDF5 file and decompresses it if it was
    compressed using the write_compressed_dataset function.
    The function automatically detects if the dataset was compressed and uses the
    metadata stored in the attributes to decompress it.
    The function returns the original data as the specified type.

    Arguments:
        - fin:           The HDF5 file object to read from.
        - dataset_name:  The name of the dataset to read (e.g., "/u", "/v", "/w").

    Returns:
        - The decompressed dataset if it was compressed, or the original dataset if it wasn't.
"""
    # Check if the dataset exists
    if !haskey(fin, dataset_name)
        error("Dataset $dataset_name does not exist in the file.")
    end
    
    # Read the dataset
    data = read(fin[dataset_name])
    
    # Check if the dataset has compression attributes
    compression_attrs = [
        dataset_name * "_DataType",
        dataset_name * "_COMPRESSION_SCALE",
        dataset_name * "_Min_Value",
        dataset_name * "_Max_Value"
    ]
    
    if all(attr -> haskey(attrs(fin), attr), compression_attrs)
        # Dataset is compressed, read the attributes
        data_type_str = read(attrs(fin)[dataset_name * "_DataType"])
        scale_factor  = read(attrs(fin)[dataset_name * "_COMPRESSION_SCALE"])
        min_val       = read(attrs(fin)[dataset_name * "_Min_Value"])
        max_val       = read(attrs(fin)[dataset_name * "_Max_Value"])
        
        # Determine the float type
        float_type_sym = if data_type_str == "Float32"
            :Float32
        elseif data_type_str == "Float64"
            :Float64
        elseif data_type_str == "Float16"
            :Float16
        else
            error("Unsupported float type: $data_type_str")
        end
        
        # Decompress the data
        decompressed_data = convert_to_float_array(data, float_type_sym, min_val, max_val, int_factor=scale_factor)
        
        # Print decompression statistics if available
        if haskey(attrs(fin), dataset_name * "_Original_Size") && haskey(attrs(fin), dataset_name * "_Compressed_Size")
            original_size     = read(attrs(fin)[dataset_name * "_Original_Size"])
            compressed_size   = read(attrs(fin)[dataset_name * "_Compressed_Size"])
            compression_ratio = original_size / compressed_size
            println("Decompression statistics for $dataset_name:")
            println("  Original size: $(original_size/1024^2) MB")
            println("  Compressed size: $(compressed_size/1024^2) MB")
            println("  Compression ratio: $(round(compression_ratio, digits=2)):1")
            if haskey(attrs(fin), dataset_name * "_Compression_Type")
                compression_type = read(attrs(fin)[dataset_name * "_Compression_Type"])
                println("  Compression type: $compression_type")
            end
        end
        
        return decompressed_data
    else
        # Dataset is not compressed
        return data
    end
end


function compress_all_datasets(input_file, output_file; compression_level::Int=9, auto_select_type::Bool=true, 
                                    include_patterns::Vector{String}=String[], exclude_patterns::Vector{String}=String[])
#====================================================================================#
# Function to convert all datasets in an HDF5 file to a compressed format
# - Reads all datasets from the input file
# - Compresses each dataset using the write_compressed_dataset function
# - Writes the compressed datasets to the output file
# - Preserves all attributes from the input file
#====================================================================================#
"""
Convert all datasets in an HDF5 file to a compressed format.
    This function reads all datasets from the input file, compresses them using the
    write_compressed_dataset function, and writes them to the output file.
    The function preserves all attributes from the input file.
    The function can filter which datasets to compress based on include and exclude patterns.

    Arguments:
        - input_file:           The path to the input HDF5 file.
        - output_file:          The path to the output HDF5 file.
        - compression_level:    Optional argument to specify the compression level (0-9).
        - auto_select_type:     Optional argument to indicate whether to automatically select the
                                best unsigned integer type for compression (true by default).
        - include_patterns:     Optional argument to specify patterns for dataset names to include.
                                If empty, all datasets will be included.
        - exclude_patterns:     Optional argument to specify patterns for dataset names to exclude.
                                If empty, no datasets will be excluded.

    Returns:
        - A dictionary with statistics about the compression.
"""
    # Open the input and output files
    fin  = h5open(input_file, "r")
    fout = h5open(output_file, "w")
    
    # Initialize statistics
    total_original_size   = 0
    total_compressed_size = 0
    datasets_processed    = 0
    compression_stats     = Dict()
    
    # Create a function to recursively process groups and datasets
    function process_group(group_name)
        # Copy all attributes from the input group to the output group
        if group_name != ""
            # Create the group if it doesn't exist
            if !haskey(fout, group_name)
                g_out = create_group(fout, group_name)
            else
                g_out = fout[group_name]
            end
            
            # Copy all attributes
            for attr_name in names(attrs(fin[group_name]))
                attrs(g_out)[attr_name] = read(attrs(fin[group_name])[attr_name])
            end
        end
        
        # Process all datasets in this group
        for obj_name in names(fin[group_name])
            full_path = group_name == "" ? obj_name : group_name * "/" * obj_name
            
            # Check if it's a dataset or a group
            if isa(fin[full_path], HDF5.Dataset)
                # Check include/exclude patterns
                should_include = isempty(include_patterns) || any(pattern -> occursin(pattern, full_path), include_patterns)
                should_exclude = !isempty(exclude_patterns) && any(pattern -> occursin(pattern, full_path), exclude_patterns)
                
                if should_include && !should_exclude
                    # Read the dataset
                    data = read(fin[full_path])
                    
                    # Record original size
                    orig_size            = sizeof(data)
                    total_original_size += orig_size
                    
                    # Compress and write the dataset
                    write_compressed_dataset(fout, full_path, data, 
                        compressed=true, 
                        compression_level=compression_level,
                        auto_select_type=auto_select_type)
                    
                    # Record compressed size
                    comp_data              = read(fout[full_path])
                    comp_size              = sizeof(comp_data)
                    total_compressed_size += comp_size
                    
                    # Update statistics
                    datasets_processed          += 1
                    compression_stats[full_path] = Dict(
                        "original_size" => orig_size,
                        "compressed_size" => comp_size,
                        "ratio" => orig_size / comp_size
                    )
                else
                    # Simply copy the dataset without compression
                    fout[full_path] = read(fin[full_path])
                    
                    # Copy all attributes
                    for attr_name in names(attrs(fin[full_path]))
                        attrs(fout[full_path])[attr_name] = read(attrs(fin[full_path])[attr_name])
                    end
                end
            elseif isa(fin[full_path], HDF5.Group)
                # Recursively process this group
                process_group(full_path)
            end
        end
    end
    
    # Start processing from the root group
    process_group("")
    
    # Close the files
    close(fin)
    close(fout)
    
    # Print overall statistics
    if datasets_processed > 0
        overall_ratio = total_original_size / total_compressed_size
        println("\nOverall compression statistics:")
        println("  Original size:             $(total_original_size/1024^2) MB")
        println("  Compressed size:           $(total_compressed_size/1024^2) MB")
        println("  Overall compression ratio: $(round(overall_ratio, digits=2)):1")
        println("  Datasets processed:        $datasets_processed")
    else
        println("No datasets were compressed.")
    end
    
    return Dict(
        "total_original_size" => total_original_size,
        "total_compressed_size" => total_compressed_size,
        "overall_ratio" => total_original_size / total_compressed_size,
        "datasets_processed" => datasets_processed,
        "dataset_stats" => compression_stats
    )
end


function safe_create_dataset(file, name, datatype, dataspace)
#====================================================================================#
# Safely create a dataset in an HDF5 file, handling existing datasets
#====================================================================================#
"""
Safely create a dataset in an HDF5 file.
    This function attempts to create a dataset in an HDF5 file. If the dataset already exists,
    it will delete the existing dataset first and then create a new one.
    This prevents errors when trying to create datasets that already exist.

    Arguments:
        - file:      The HDF5 file object to create the dataset in.
        - name:      The name of the dataset to create.
        - datatype:  The datatype of the dataset.
        - dataspace: The dataspace of the dataset.

    Returns:
        - The created dataset object.
"""
    # Check if the dataset already exists and delete it if it does
    if haskey(file, name)
        delete_object(file, name)
    end
    
    # Create and return the dataset
    return create_dataset(file, name, datatype, dataspace)
end

function compute_grids(lx, xMin, xMax, NX0, NX, y, lz, NZ)
#====================================================================================#
# compute grids ...
#====================================================================================#
"""
Compute the x, y, and z grids for the given parameters.
    This function generates the x, y, and z grids based on the provided parameters.
    The x-grid is computed based on the length in the x-direction, minimum and maximum
    x-coordinates, and the number of points in the original and new grids.
    The y-grid is taken from the provided y-coordinates, and the z-grid is computed
    based on the length in the z-direction and the number of points in the z-direction.
    The midpoints in the x and y directions are also computed.  
    The function returns the computed x, y, and z grids, as well as the midpoints.

    Arguments:
        - lx: Length in x-direction.

        - xMin: Minimum x-coordinate index (1-based) to use from the original grid.
        - xMax: Maximum x-coordinate index (1-based) to use from the original grid.
        - NX0: Number of points in the original grid.
        - NX: Number of points in the new grid.
        - y: y-coordinates.
        - lz: Length in z-direction.
        - NZ: Number of real,imag, real, imag, ... values in the z-direction.

    Returns:
        - xgrid: The x-coordinates of the x-faces of the computational cell - staggered u is defined, Note: xgrid[1] = 0
        - ygrid: The y-coordinates of the y-faces of the computational cell - staggered v is defined, Note: ygrid[1] = 0
        - zgrid: The z-coordinates of the centre of the computational cell  - staggered w,p is defined & collocation 
                                                                              z-coordinates for u,v,w,p.
        - xmidp: The x-coordinates of the centre of the computational cell  - collocation x-coordinates for u,v,w,p.
        - ymidp: The y-coordinates of the centre of the computational cell  - staggered u, w, p are defined, Note: ymidp[1] < 0
"""
    # Calculate the physical x-coordinates based on xMin and xMax indices
    # For the full domain (xMin=1, xMax=NX0), this gives range from 0 to lx
    # For a subset (e.g., xMin=10, xMax=100), this gives the corresponding physical range
    
# Convert indices to physical coordinates in the original grid
    dx_original = lx / (NX0 - 1)            # Grid spacing in the original grid
    x_start     = (xMin - 1) * dx_original  # Physical coordinate corresponding to xMin
    x_end       = (xMax - 1) * dx_original  # Physical coordinate corresponding to xMax
    
# Generate the grid with NX points from x_start to x_end
    xgrid = range(x_start, x_end, length = NX)
    
    ygrid = y[2:end-1]
    zgrid = range(0, lz, length=NZ-2) # number of points in physcial z = NZ-2
    xmidp = 0.5 .* (xgrid[2:end] .+ xgrid[1:end-1])
    ymidp = 0.5 .* (y[2:end]     .+ y[1:end-1])

    return xgrid, ygrid, zgrid, xmidp, ymidp
end


#=======================================================================================================================================================#
# CORE PROCESSING FUNCTIONS 
#=======================================================================================================================================================#

function InterpolateAtXmidPt(global_array, xgrid, xmidp, root, comm)
#====================================================================================#
# divide the work up among the cpu in y: i.e. NZ x NX planes to interpolate in x ....
# - allocate the necessary memory in every task Pencil
# - scatter the data to the tasks
# - interpolate the data in x
# - gather the data back to the root task
#====================================================================================#
"""
Interpolate the input array at xmidp points.
    This function performs interpolation on the input array at specified xgrid points.
    The function divides the work among multiple CPUs, scattering the data to each task,
    performing the interpolation, and then gathering the results back to the root task.
    The interpolation is done using B-splines of degree 5.
    The function also handles the MPI communication for scattering and gathering the data.
        
    Arguments:
        - global_array: The input array to interpolate.
        - xgrid:        The x-coordinates for the staggered u-velocity grid = x-cell face coordinates - [1:NX]
        - xmidp:        The x-coordinate of the centre of the computational cell - collocation x-coordinates for u,v,w,p - [1:NX-1]
        - root:         The rank of the root process.
        - comm:         The MPI communicator.

    Returns:
        - output_array: The interpolated array at xmidp points.
"""
### get the number of CPUs
    no_cpus = MPI.Comm_size(comm)

### Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the global array
    if rank == root
        println("InterpolateAtXmidPt: Interpolating to X mid-points ...")
        (NZ, NY, NX) = size(global_array)
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)  # Dummy values for non-root ranks    
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)    
        
    ### Number of y slices per rank
    local_ny = split_count(NY, no_cpus)

    if rank == root
        # allocate storage for final result on root:
        output_array = zeros(Float32, NZ, NY, NX-1)  # Placeholder for gathering the final result
                                                     # ACHTUNG: this array is different once we interpolate, only (NX-1) in x
        # Prepare sizes for scattering along the second dimension
        slice_size = NZ # Size of each 1D pencil corresponding to each y for scattering/gathering
        # Store the number of elements each rank will receive (in 1D flattened view)
        counts = [ny * slice_size for ny in local_ny]
    end

    # Define Local arrays:
    # Local input 3D array slices to each rank
    local_size  = (NZ, local_ny[rank + 1], NX)
    local_array = zeros(Float32, local_size)  # Local slice storage in on each rank

    # scatter the 3D array, one x-plane i.e. z-y plane, at a time
    out_array = zeros(Float32, (NZ, local_ny[rank + 1]))
    for i in 1:NX
        # Define data to scatter and its VBuffer for scattering
        if rank == root
            in_array    = global_array[:,:,i]
            global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
        else
            global_vbuf = nothing
        end
        MPI.Scatterv!(global_vbuf, out_array, root, comm)
        local_array[:,:,i] = out_array
    end
    MPI.Barrier(comm)
    
    # interpolate in x:
    if rank == 0 println("Interpolating in X\n") end
    interpolated_in_x = zeros(Float32, NZ, local_ny[rank + 1], NX-1)
    # note: x-interpolated field is @ centre of the CV.
    #       Hence, has only NX-1 elements
    for (k, j) in Iterators.product(1:NZ, 1:local_ny[rank + 1])      # for each k,j a pencil of lenth NX
        stride                          = local_array[k, j, :]       # a x-pencil for each (k,j)
        # Fit a B-spline with degree 5 to the data
        spline = Spline1D(xgrid, stride, k=5)
        # Evaluate the spline at xmidp points
        interpolated_in_x[k, j, 1:end] .= spline.(xmidp)
    end

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ, NY))
        output_vbuf = VBuffer(out_array, counts) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified slices back to the root process, one x-plane, i.e. z-y plane, at the time
    for i in 1:NX-1
        MPI.Gatherv!(interpolated_in_x[:,:,i], output_vbuf, root, comm)
        if rank == 0
            output_array[:,:,i] = out_array
        end
    end
    if rank == root
        return output_array
    else
        return nothing
    end
end


function InterpolateAtY_IRFFT_2_Phys(input_array, ygrid, ymidp, root, comm)
#====================================================================================#
# divide the work up among the cpu in x: i.e. NZ x NY planes to interpolate in y ....
# Note: after interpolating in y, there are only NY-1 y-planes!!!
# Number of x planes per rank is stored in local_nx[]
#====================================================================================#
"""
Interpolate the input array at ymidp points and perform IRFFT in z-direction.
    This function performs interpolation on the input array at specified ymidp points
    and then performs an Inverse Real Fast Fourier Transform (IRFFT) in the z-direction.
    The function divides the work among multiple CPUs, scattering the data to each task,
    performing the interpolation and IRFFT, and then gathering the results back to the root task.
    The interpolation is done using B-splines of degree 5.
    The function also handles the MPI communication for scattering and gathering the data.
    
    Arguments:
        - input_array: The input array to interpolate.
        - ygrid:       The y-coordinates of the y-faces of the computational cell - staggered v is defined = y-cell face coordinates - [1:NY-1]
        - ymidp:       The y-coordinates of the computational cell centre, u, w, p are defined there on the staggered grid including ghost cells for u & w. on both sides but only for y < 0 for p [1:NY]
        - root:        The rank of the root process.
        - comm:        The MPI communicator.

    Returns:
        - output_array: The interpolated array at ymidp points and IRFFT in z-direction.
"""
### Get the number of CPUs
    no_cpus = MPI.Comm_size(comm)       

###  Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the input array
    if rank == root
        println("InterpolateAtY_IRFFT_2_Phys: Interpolating to Y mid-points and IRFFTing in Z ...")
        (NZ, NY, NX) = size(input_array)
        NYo          = length(ygrid)
    else
        (NZ, NY, NX, NYo) = (nothing, nothing, nothing, nothing)  # Dummy values for non-root ranks
    end
    NZ  = MPI.bcast(NZ, root, comm)
    NY  = MPI.bcast(NY, root, comm)
    NX  = MPI.bcast(NX, root, comm)
    NYo = MPI.bcast(NYo, root, comm)
    MPI.Barrier(comm)       

    local_nx = split_count(NX, no_cpus)
    if rank == root
        slice_size   = NZ*NY        # Size of each 2D plane corresponding to each x for scattering
        slice_size_o = (NZ-2)*NYo   # Size of each 2D plane corresponding to each x for gathering
        counts   = [nx * slice_size   for nx in local_nx]
        counts_o = [nx * slice_size_o for nx in local_nx]
    end
    # Define Local arrays:
    # Local 3D slices to each rank
    local_size  = (NZ, NY, local_nx[rank + 1])
    local_array = zeros(Float32, local_size)
    # scatter the 3D array ...
    # Define data to scatter and its VBuffer for scattering
    if rank == root
        in_array = input_array
        global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
    else
        global_vbuf = nothing
    end
    MPI.Scatterv!(global_vbuf, local_array, root, comm)

    # interpolate in y:
    if rank == 0 println("Interpolating in Y\n") end
    interpolated_in_y = zeros(Float32, NZ, NYo, local_nx[rank + 1])
    for (k, i) in Iterators.product(1:NZ, 1:local_nx[rank + 1])
        stride = local_array[k, :, i]
        # Fit a B-spline with degree 5 to the data
        spline = Spline1D(ymidp[1:NY], stride, k=5) # Note: ymidp[1:NY] is explicitly used to enable the correct size of ymidp when used for u, w or p 
        # Evaluate the spline at xmidp points
        interpolated_in_y[k, 1:end, i] .= spline.(ygrid)
    end

    # IRFFTing in z
    if rank == 0 println("IRFFTing in Z\n") end
    value_out = zeros(Float32,          NZ-2,       NYo, local_nx[rank + 1])
    fou       = zeros(Complex{Float32}, div(NZ, 2), NYo)
    for i in 1:local_nx[rank + 1]
        fou[:,:]               .= interpolated_in_y[1:2:end, :, i] + im * interpolated_in_y[2:2:end, :, i]
        value_out[1:end, :, i] .= FFTW.irfft(fou, NZ-2, 1)*(NZ-2) # 3rd para = 1 => IRFFT along dim = 1, NZ
    end

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ-2, NYo, NX))
        output_vbuf = VBuffer(out_array, counts_o) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified blocks back to the root process, one block, i.e. number of x-plane
    MPI.Gatherv!(value_out, output_vbuf, root, comm)
    if rank == root
        return out_array
    else
        return nothing
    end
end


function IRFFT_2_Phys(input_array, root, comm)
#====================================================================================#
# divide the work up among the cpu in x: i.e. NZ x NY planes to IRFFT in z ....
# Note: NY is the correct one for collocation, NOT (NY-1) !!!!
# Note: Only the first NX-1 x-planes count and are used as these x values corresponds to xmidp[]!!!
# Number of x slices per rank are stored in local_nx[]
#
# ACHTUNG: input_array has dimension (NZ, NY-1, NX-1) !!!!
#====================================================================================#
"""
Perform IRFFT in z-direction on the input array.
    This function performs an Inverse Real Fast Fourier Transform (IRFFT) in the z-direction
    on the input array. The function divides the work among multiple CPUs, scattering
    the data to each task, performing the IRFFT, and then gathering the results back to the root task.
    The IRFFT is performed on the input array, which is expected to have dimensions (NZ, NY-1, NX-1).
    The function also handles the MPI communication for scattering and gathering the data.
        
    Arguments:
        - input_array: The input array to perform IRFFT on.
        - root:        The rank of the root process.
        - comm:        The MPI communicator.

    Returns:
        - output_array: The IRFFT result in physical space.
"""
### Get the number of CPUs
    no_cpus = MPI.Comm_size(comm)

### Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the input array
    if rank == root
        println("IRFFT_2_Phys: IRFFTing in Z ...")
        (NZ, NY, NX) = size(input_array)
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)  # Dummy values for non-root ranks
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)

    local_nx = split_count(NX, no_cpus)
    if rank == root
        slice_size   = NZ*NY     # Size of each 2D plane corresponding to each x for scattering
        slice_size_o = (NZ-2)*NY # Size of each 2D plane corresponding to each x for gathering
        counts   = [nx * slice_size   for nx in local_nx]
        counts_o = [nx * slice_size_o for nx in local_nx]
    end
    # Define Local arrays:
    # Local 3D slices to each rank
    local_size  = (NZ, NY, local_nx[rank + 1])
    local_array = zeros(Float32, local_size)
    # scatter the 3D array ...
    # Define data to scatter and its VBuffer for scattering
    if rank == root
        in_array = input_array
        global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
    else
        global_vbuf = nothing
    end
    MPI.Scatterv!(global_vbuf, local_array, root, comm)

    # IRFFTing in z
    if rank == 0 println("IRFFTing in Z\n") end
    value_out = zeros(Float32,          NZ-2,       NY, local_nx[rank + 1])
    fou       = zeros(Complex{Float32}, div(NZ, 2), NY)
    for i in 1:local_nx[rank + 1]
        fou[:,:]               .= local_array[1:2:end, :, i] + im * local_array[2:2:end, :, i]
        value_out[1:end, :, i] .= FFTW.irfft(fou, NZ-2, 1)*(NZ-2) # 3rd para = 1 => IRFFT along dim = 1, NZ
    end

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ-2, NY, NX))
        output_vbuf = VBuffer(out_array, counts_o) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified blocks back to the root process, one block, i.e. number of x-plane
    MPI.Gatherv!(value_out, output_vbuf, root, comm)
    if rank == root
        return out_array
    else
        return nothing
    end
end


function xDerivative(f, Δx, root, comm, no_cpus)
#==========================================================================================#
# Function to compute the x derivative of field f[NZ, NY, NX] using compact Pade FD
#  - divide the work up among the cpu in y: i.e. NZ x NX planes to differenteate in x ....
#       - allocate the necessary memory in every task Pencil
#       - scatter the data to the tasks
#       - differentiate the data in x
#       - gather the data back to the root task
#===========================================================================================#
"""
Compute the x derivative of the input array using compact finite difference.
    This function computes the x-derivative of the input array using Pade compact finite 
        difference.
    The function divides the work among multiple CPUs, scattering the data to each task,
    performing the differentiation, and then gathering the results back to the root task.
    The differentiation is done using the Pade finite difference method.
    The function also handles the MPI communication for scattering and gathering the data.

    Arguments:
        - f:        The input array to differentiate, expected to have dimensions (NZ, NY, NX).
        - Δx:      The grid spacing in the x-direction.
        - root:     The rank of the root process.
        - comm:     The MPI communicator.
        - no_cpus:  Number of CPUs. 

    Returns:
        - output_array: The x-derivative of the input array.
"""
    # Get the rank of the current process
    rank = MPI.Comm_rank(comm)
    
    if rank == root
        (NZ, NY, NX) = f.size
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)

    # Number of y slices per rank
    local_ny = split_count(NY, no_cpus)

    if rank == root
        # allocate storage for final result on root:
        output_array = zeros(Float32, NZ, NY, NX)  # Placeholder for gathering the final result
        # Prepare sizes for scattering along the second dimension
        slice_size = NZ # Size of each 1D pencil corresponding to each y for scattering/gathering
        # Store the number of elements each rank will receive (in 1D flattened view)
        counts = [ny * slice_size for ny in local_ny]
    end

    # Define Local arrays:
    # Local input 3D array slices to each rank
    local_size  = (NZ, local_ny[rank + 1], NX)
    local_array = zeros(Float32, local_size)  # Local slice storage in on each rank

    # scatter the 3D array, one x-plane i.e. z-y plane, at a time
    out_array = zeros(Float32, (NZ, local_ny[rank + 1]))
    for i in 1:NX
        # Define data to scatter and its VBuffer for scattering
        if rank == root
            in_array    = f[:,:,i]
            global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
        else
            global_vbuf = nothing
        end
        MPI.Scatterv!(global_vbuf, out_array, root, comm)
        local_array[:,:,i] = out_array
    end

    MPI.Barrier(comm)
    # differentiate in x:
    if rank == 0 println("\nDifferentiate in X\n") end
    local_array_type = eltype(local_array)
    Δx               = convert(local_array_type, Δx)
    dfdx             = compute_PadeFD(local_array, Δx , dim=3)

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ, NY))
        output_vbuf = VBuffer(out_array, counts) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified slices back to the root process, one x-plane, i.e. z-y plane, at the time
    for i in 1:NX
        MPI.Gatherv!(dfdx[:,:,i], output_vbuf, root, comm)
        if rank == 0
            output_array[:,:,i] = out_array
        end
    end
    MPI.Barrier(comm)

    if rank == root
        return output_array
    else
        return nothing
    end
end


function yDerivative(f, ygrid, root, comm, no_cpus)
#==========================================================================================#
# Function to compute the y derivative of field f[NZ, NY, NX] using compact FD
#  - divide the work up among the cpu in x: i.e. NZ x NY planes to differenteiate in y ....
#       - allocate the necessary memory in every task Pencil
#       - scatter the data to the tasks
#       - differentiate the data in y
#       - gather the data back to the root task
#==========================================================================================#
"""
Compute the y derivative of the input array using compact finite difference.
    This function computes the y-derivative of the input array using Pade compact finite 
        difference.
    The function divides the work among multiple CPUs, scattering the data to each task,
    performing the differentiation, and then gathering the results back to the root task.
    The differentiation is done using the Pade finite difference method.
    The function also handles the MPI communication for scattering and gathering the data.
    Note: The y-derivative is computed using the y-coordinates provided in the ygrid argument.

    Arguments:
        - f:        The input array to differentiate.
        - ygrid:    The y-coordinates for the original grid.
        - root:     The rank of the root process.
        - comm:     The MPI communicator.
        - no_cpus:  Number of CPUs.

    Returns:
        - output_array: The y-derivative of the input array.
"""
    # Get the rank of the current process
    rank = MPI.Comm_rank(comm)
    
    if rank == root
        (NZ, NY, NX) = f.size
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)

    # Number of x slices per rank
    local_nx = split_count(NX, no_cpus)

    if rank == root
        slice_size = NZ*NY      # Size of each 2D plane corresponding to each x for scattering
        counts     = [nx * slice_size   for nx in local_nx]
    end
    # Define Local arrays:
    # Local 3D slices to each rank
    local_size  = (NZ, NY, local_nx[rank + 1])
    local_array = zeros(Float32, local_size)
    # scatter the 3D array ...
    # Define data to scatter and its VBuffer for scattering
    if rank == root
        in_array    = f
        global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
    else
        global_vbuf = nothing
    end
    MPI.Scatterv!(global_vbuf, local_array, root, comm)

    # interpolate in y:
    if rank == 0 println("\nDifferentiating in Y\n") end
    local_array_type = eltype(local_array)
    ygrid            = convert(Vector{local_array_type}, ygrid)
    value_out        = compute_PadeFD(local_array, ygrid, dim=2)

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ, NY, NX))
        output_vbuf = VBuffer(out_array, counts) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified blocks back to the root process, one block, i.e. number of x-plane
    MPI.Gatherv!(value_out, output_vbuf, root, comm)
    if rank == root
        return out_array
    else
        return nothing
    end
end


function zDerivative(f, L, root, comm, no_cpus)
#==========================================================================================#
# Function to compute the z derivative of field f[NZ, NY, NX] using FFT differentiation
#  - divide the work up among the cpu in x: i.e. NZ x NY planes to differenteate in z ....
#       - allocate the necessary memory in every task Pencil
#       - scatter the data to the tasks
#       - differentiate the data in z
#       - gather the data back to the root task
#==========================================================================================#
"""
Compute the z derivative of the input array using FFT differentiation.
    This function computes the z-derivative of the input array using Fast Fourier Transform (FFT)
        differentiation.
    The function divides the work among multiple CPUs, scattering the data to each task,
    performing the differentiation, and then gathering the results back to the root task.
    The differentiation is done using the FFT method.
    The function also handles the MPI communication for scattering and gathering the data.

    Arguments:
        - f:        The input array to differentiate.
        - L:        The length in the z-direction.
        - root:     The rank of the root process.
        - comm:     The MPI communicator.
        - no_cpus:  Number of CPUs.

    Returns:
        - output_array: The z-derivative of the input array.
"""
    # Get the rank of the current process
    rank = MPI.Comm_rank(comm)
    
    if rank == root
        (NZ, NY, NX) = f.size
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)

    # Number of x slices per rank
    local_nx = split_count(NX, no_cpus)

    if rank == root
        slice_size = NZ*NY      # Size of each 2D plane corresponding to each x for scattering
        counts     = [nx * slice_size   for nx in local_nx]
    end
    # Define Local arrays:
    # Local 3D slices to each rank
    local_size  = (NZ, NY, local_nx[rank + 1])
    local_array = zeros(Float32, local_size)
    # scatter the 3D array ...
    # Define data to scatter and its VBuffer for scattering
    if rank == root
        in_array    = f
        global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
    else
        global_vbuf = nothing
    end
    MPI.Scatterv!(global_vbuf, local_array, root, comm)

    # differentiate in z:
    if rank == 0 println("\nDifferentiating in Z\n") end
    local_array_type = eltype(local_array)
    L_converted      = convert(local_array_type, L)
    value_out        = compute_FTD(local_array, L_converted, dim=1) # dim = 1 -> z-dirn, NZ

    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(Float32, (NZ, NY, NX))
        output_vbuf = VBuffer(out_array, counts) # Create VBuffers for gathering
    else
        out_array = nothing
        output_vbuf = nothing
    end
    # Gather the modified blocks back to the root process, one block, i.e. number of x-plane
    MPI.Gatherv!(value_out, output_vbuf, root, comm)
    if rank == root
        return out_array
    else
        return nothing
    end
end


function toPhysAndCollocate_u_VGT_h5(global_array, xgrid, xmidp, ygrid, ymidp, zgrid, root, comm; 
                                                                    fo=nothing, VGT::Bool = false)
#====================================================================================#
# Function to convert u_fourier -> u_physial and collocate u_physical at:
# - xmidp, ygrid, zgrid
# 1. Interpolate u_fourier from xgrid -> xmidp resulting in NX-1 x-locations
# 2. Interpolate u_fourier from ymidp -> ygrid (v-velocity grid) resulting in NY-1
#    y-locations
# 3. Perform IRFFT in z-direction to get u_physical @ zgrid resulting in NZ-2
#    z-locations
# 4. Write physcial field u_physical to the h5 file
# 5. if VGT: Compute du/dx, du/dy, du/dz and write them to the h5 file
# 6. Return u_physical
#====================================================================================#
"""
Convert the input array from Fourier space to physical space and collocate it at 
specified grid points.
    This function performs the conversion of the input array from Fourier space 
        to physical space and collocates it at specified grid points:
        - x-interpolate from xgrid[1:NX] -> xmidp[1:NX-1] : u(xgrid[1:NX],   ymidp[1:NY],   kz[1:NZ]) -> u(xmidp[1:NX-1], ymidp[1:NY],   kz[1:NZ])
        - y-interpolate from ymidp[1:NY] -> ygrid[1:NY-1] : u(xmidp[1:NX-1], ymidp[1:NY],   kz[1:NZ]) -> u(xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ]) 
        - IRFFT in z from NZ values      -> zgrid[1:NZ-2] : u(xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ]) -> u(xmidp[1:NX-1], ygrid[1:NY-1], zgrid[1:NZ-2]) 
    The function handles the MPI communication for scattering and gathering the data. 
    The final result is written to the specified HDF5 file. 
    The function also computes the derivatives of the physical field and writes them to the HDF5 file.
    
    Note: The input array is expected to have dimensions (NZ, NY, NX) and the output
            physical array will have dimensions (NZ-2, NY-1, NX-1).

    Arguments:
        - global_array: The input array to convert, dimensions (NZ, NY, NX) (LOCAL)
        - xgrid:        The x-coordinates for the original grid.
        - xmidp:        Midpoints in the x-direction.
        - ygrid:        The y-coordinates for the original grid.
        - ymidp:        Midpoints in the y-direction.
        - zgrid:        The z-coordinates for the original grid.
        - fo:           The HDF5 file object to write to, if fo == nothing, 
                        the function will not write to the file.
        - root:         The rank of the root process.
        - comm:         The MPI communicator.
        - VGT:          Optional argument to indicate whether to calculate the spatial derivatives of the collocated physical field.
                        Note:   Derivates are computed using xDerivative, yDerivative, zDerivative functions.
                                If VGT is true, the function will write the physical field to the HDF5 file.

    Returns:
        - u_phys:       The collocated physical representation of the input array at specified grid points, dimensions (NZ-2, NY-1, NX-1). 
"""
### Get the number of CPUs
    no_cpus = MPI.Comm_size(comm)

### Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the input array
    if rank == root
        (NZ, NY, NX) = size(global_array)
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)
    
### INTERPOLATION IN X:
    interpolated_in_x = InterpolateAtXmidPt(global_array, xgrid, xmidp, root, comm)

### INTERPOLATION IN y and IRFFTing in z
    u_phys            = InterpolateAtY_IRFFT_2_Phys(interpolated_in_x, ygrid, ymidp, root, comm)

### Write the final result to the output file
    if rank == root && fo !== nothing
        # Write the final result to the output file
        println("Writing to output h5 file: u_physical")
        write_compressed_dataset(fo, "/u", u_phys, compressed=false)
    end

# compute derivatives of u_phys and write them to the h5 file:
    if VGT == true   
        # du/dx:
        u_phys_type = eltype(u_phys)
        Δx          = convert(u_phys_type, xgrid[2]-xgrid[1])
        dudx        = xDerivative(u_phys, Δx, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: du/dx")
            write_compressed_dataset(fo, "/dudx", dudx, compressed=false)
        end

        # du/dy:
        ygrid  = convert(Vector{u_phys_type}, ygrid)
        dudy   = yDerivative(u_phys, ygrid, root, comm, no_cpus)
        if rank == root  && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: du/dy")
            write_compressed_dataset(fo, "/dudy", dudy, compressed=false)
        end

        # du/dz
        Lz   = convert(u_phys_type, zgrid[NZ-2])
        dudz = zDerivative(u_phys, Lz, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: du/dz")
            write_compressed_dataset(fo, "/dudz", dudz, compressed=false)
        end
    end

# return u_phys
    # Need to ensure all ranks return the same type of result
    if rank == root
        if VGT == true
            return (u_phys, dudx, dudy, dudz)
        else
            return u_phys   
        end
    else
        if VGT == true
            # Return empty arrays with same structure when not on root
            return (nothing, nothing, nothing, nothing)
        else
            return nothing
        end
    end
end


function toPhysAndCollocate_v_VGT_h5(global_array, ygrid, zgrid, xgrid, root, comm; 
                                                    fo = nothing, VGT::Bool = false)
#====================================================================================#
# Function to convert v_fourier -> v_physial and collocate v_physical at:
# - xmidp, ygrid, zgrid
# 1. v is already located @ xmidp + upstream ghost cell
# 2. v is already located @ ygrid (v-velocity grid) resulting in NY
#    y-locations
# 3. Perform IRFFT in z-direction to get w_physical @ zgrid resulting in 
#    NZ-2 z-locations
# 4. Write physcial field v_physical to the h5 file
# 5. If VGT: Compute dv/dx, dv/dy, dv/dz and write them to the h5 file
# 6. Return v_physical
#====================================================================================#
"""
Convert the input array from Fourier space to physical space and collocate it at
specified grid points.
    This function performs the conversion of the input array from Fourier space
    to physical space and collocates it at specified grid points:
        - remove the first v-component in x, which is a ghost cell value, ghost+xmidp[1:NX-1] -> xmidp[1:NX-1] : v(ghost+xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ]) -> v(xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ])
        - IRFFT in z from NZ values                                                           -> zgrid[1:NZ-2] : v(xmidp[1:NX-1],       ygrid[1:NY-1], kz[1:NZ]) -> v(xmidp[1:NX-1], ygrid[1:NY-1], zgrid[1:NZ-2])
    The function handles the MPI communication for scattering and gathering the data.
    The final result is written to the specified HDF5 file.
    The function also computes the derivatives of the physical field and writes them to the HDF5 file.
    Note: The input array is expected to have dimensions (NZ, NY, NX) and the output
            physical array will have dimensions (NZ-2, NY-1, NX-1).

    Arguments:
        - global_array: The input array to convert, dimensions (NZ, NY, NX)
        - ygrid:        The y-coordinates for the original grid.
        - zgrid:        The z-coordinates for the original grid.
        - fo:           The HDF5 file object to write to, if fo == nothing,
                        the function will not write to the file.
        - root:         The rank of the root process.
        - comm:         The MPI communicator.
        - VGT:          Optional argument to indicate whether to calculate the spatial derivatives of the collocated physical v field.
                        Note:   Derivates are computed using xDerivative, yDerivative, zDerivative functions.
                                If VGT is true, the function will write the physical v field to the HDF5 file.

    Returns:
        - v_phys:       The physical representation of the input array at specified grid points,
                        dimensions (NZ-2, NY-1, NX-1).
"""
### Get the number of CPUs
    no_cpus = MPI.Comm_size(comm)

### Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the input array
    if rank == root
        (NZ, NY, NX) = size(global_array)
        # remove the first x-component of v-velocity component which is at the centre of the ghost cell
        global_array = global_array[:, :, 2:NX] # remove the first x-component of v-velocity component which is at the centre of the ghost cell
        NX           = size(global_array, 3) # NX is now the number of x-coordinates in the output grid 
    else
        (NZ, NY, NX) = (nothing, nothing, nothing)
    end
    NZ = MPI.bcast(NZ, root, comm)
    NY = MPI.bcast(NY, root, comm)
    NX = MPI.bcast(NX, root, comm)
    MPI.Barrier(comm)

### IRFFTing in z
    v_phys = IRFFT_2_Phys(global_array, root, comm)

### Write the final result to the output file
    if rank == root && fo !== nothing
        # Write the final result to the output file
        println("Writing to output h5 file: v_physical")
        write_compressed_dataset(fo, "/v", v_phys, compressed=false)
    end

    # compute derivatives of v_phys and write them to the h5 file:
    if VGT == true
        # dv/dx:
        v_phys_type = eltype(v_phys)
        Δx          = convert(v_phys_type, xgrid[2]-xgrid[1])
        dvdx        = xDerivative(v_phys, Δx, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dv/dx")
            write_compressed_dataset(fo, "/dvdx", dvdx, compressed=false)
        end

        # dv/dy:
        ygrid = convert(Vector{v_phys_type}, ygrid)
        dvdy  = yDerivative(v_phys, ygrid, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dv/dy")
            write_compressed_dataset(fo, "/dvdy", dvdy, compressed=false)
        end

        # dv/dz
        Lz   = convert(v_phys_type, zgrid[NZ-2])
        dvdz = zDerivative(v_phys, Lz, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dv/dz")
            write_compressed_dataset(fo, "/dvdz", dvdz, compressed=false)
        end
    end

    # return v_phys
    if rank == root
        if VGT == true
            return (v_phys, dvdx, dvdy, dvdz) # Fixed to dvdx since it was incorrectly using dvdy twice
        else
            return v_phys
        end
    else
        if VGT == true
            # Return empty arrays with same structure when not on root
            return (nothing, nothing, nothing, nothing)
        else
            return nothing
        end
    end
end


function toPhysAndCollocate_w_VGT_h5(global_array, ygrid, ymidp, zgrid, xgrid, root, comm; 
                                        pflag::Bool = false, fo = nothing, VGT::Bool = false)
#====================================================================================#
# Function to convert w_fourier -> w_physial and collocate w_physical at:
# - xmidp, ygrid, zgrid
# 1. w is already located @ xmidp + upstream ghost cell
# 2. Interpolate w_fourier from ymidp -> ygrid (v-velocity grid) resulting in NY-1
#    y-locations
# 3. Perform IRFFT in z-direction to get w_physical @ zgrid resulting in NZ-2
#    z-locations
# 4. Write physcial field w_physical/pressure field to the h5 file
# 4. If VGT: Compute dw/dx, dw/dy, dw/dz and write them to the h5 file
# 5. Return w_physical/pressure field
#====================================================================================#
"""
Convert the input array from Fourier space to physical space and collocate it at
    specified grid points.
    This function performs the conversion of the input array from Fourier space
    to physical space and collocates it at specified grid points:
    - w:
        - remove the first w-component in x, which is a ghost cell value, ghost+xmidp[1:NX-1] -> xmidp[1:NX-1] : w(ghost+xmidp[1:NX-1], ymidp[1:NY],   kz[1:NZ]) -> w(xmidp[1:NX-1], ymidp[1:NY],   kz[1:NZ]) 
        - y-interpolate from ymidp[1:NY]                                                      -> ygrid[1:NY-1] : w(xmidp[1:NX-1],       ymidp[1:NY],   kz[1:NZ]) -> w(xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ])
        - IRFFT in z from NZ values                                                           -> zgrid[1:NZ-2] : w(xmidp[1:NX-1],       ygrid[1:NY-1], kz[1:NZ]) -> w(xmidp[1:NX-1], ygrid[1:NY-1], zgrid[1:NZ-2]) 

    - p:
        - remove the first p-component in x, which is a ghost cell value, ghost+xmidp[1:NX-1] -> xmidp[1:NX-1] : p(ghost+xmidp[1:NX-1], ymidp[1:NY-1], kz[1:NZ]) -> p(xmidp[1:NX-1], ymidp[1:NY-1], kz[1:NZ])
        - y-interpolate from ymidp[1:NY-1]                                                    -> ygrid[1:NY-1] : p(xmidp[1:NX-1],       ymidp[1:NY-1], kz[1:NZ]) -> p(xmidp[1:NX-1], ygrid[1:NY-1], kz[1:NZ]) 
        - IRFFT in z from NZ values                                                           -> zgrid[1:NZ-2] : p(xmidp[1:NX-1],       ygrid[1:NY-1], kz[1:NZ]) -> p(midp[1:NX-1],  ygrid[1:NY-1], zgrid[1:NZ-2])
    The function handles the MPI communication for scattering and gathering the data.
    The final result is written to the specified HDF5 file.
    The function also computes the derivatives of the physical field and writes them to the HDF5 file.
    Note: The input array is expected to have dimensions (NZ, NY, NX) (LOCAL) = (NZ, NY/(NY-1), NX) (GLOBAL) and the output
            physical array will have dimensions (NZ-2, NY-1, NX-1) (LOCAL = .
    The function also has an optional pflag argument to indicate whether to write
    the physical field or the pressure field to the HDF5 file.

    Arguments:
        - global_array: The input array to convert, dimensions (NZ, NY, NX) 
        - ygrid:        The y-coordinates for the original grid.
        - ymidp:        Midpoints in the y-direction.
        - zgrid:        The z-coordinates for the original grid.
        - fo:           The HDF5 file object to write to, if fo == nothing,
                        the function will not write to the file.
        - root:         The rank of the root process.
        - comm:         The MPI communicator.
        - pflag:        Optional argument to indicate whether to write the physical field
                        or the pressure field to the HDF5 file.
        - VGT:          Optional argument to indicate whether to calculate the spatial derivatives of the collocated physical w field.
                        Note:   Derivates are computed using xDerivative, yDerivative, zDerivative functions.
                                If VGT is true, the function will write the physical w field to the HDF5 file.

    Returns:
        - w_phys:       The physical w/p representation of the input array at specified grid points, 
                        dimensions (NZ-2, NY-1, NX-1) (GLOBAL) = (NZ-2, NYo, NX-1) (LOCAL).
"""
### Get the number of CPUs
    no_cpus = MPI.Comm_size(comm)

### Get the rank of the current process
    rank = MPI.Comm_rank(comm)

### Get the dimensions of the input array
    if rank == root
        (NZ, NY, NX) = size(global_array)
        # remove the first x-component of v-velocity component which is at the centre of the ghost cell
        global_array = global_array[:, :, 2:NX] # remove the first x-component of w-velocity component which is at the centre of the ghost cell
        NX           = size(global_array, 3) # NX is now the number of x-coordinates in the output grid
        NYo          = length(ygrid) # NYo is the number of y-coordinates in the output grid
    else
        (NZ, NY, NX, NYo) = (nothing, nothing, nothing, nothing)
    end
    NZ  = MPI.bcast(NZ, root, comm)
    NY  = MPI.bcast(NY, root, comm)
    NX  = MPI.bcast(NX, root, comm)
    NYo = MPI.bcast(NYo, root, comm)    
    MPI.Barrier(comm)   

### INTERPOLATION IN y and IRFFTing in z, w or p:
    w_phys = InterpolateAtY_IRFFT_2_Phys(global_array, ygrid, ymidp, root, comm)

### Write the final result to the output file
    if rank == root && fo !== nothing
        # Write the final result to the output file
        if pflag == true  # write pressure field
            println("Writing to output h5 file: p_physical")
            write_compressed_dataset(fo, "/p", w_phys, compressed=false)
        else              # write w field
            println("Writing to output h5 file: w_physical")
            write_compressed_dataset(fo, "/w", w_phys, compressed=false)
        end
    end

    # compute derivatives of w_phys (w only) and write them to the h5 file:
    if pflag != true && VGT == true
        # dw/dx:
        w_phys_type = eltype(w_phys)
        Δx          = convert(w_phys_type, xgrid[2]-xgrid[1])
        dwdx        = xDerivative(w_phys, Δx, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dw/dx")
            write_compressed_dataset(fo, "/dwdx", dwdx, compressed=false)
        end

        # dw/dy:
        ygrid = convert(Vector{w_phys_type}, ygrid)
        dwdy  = yDerivative(w_phys, ygrid, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dw/dy")
            write_compressed_dataset(fo, "/dwdy", dwdy, compressed=false)
        end

        # dw/dz
        Lz   = convert(w_phys_type, zgrid[NZ-2])
        dwdz = zDerivative(w_phys, Lz, root, comm, no_cpus)
        if rank == root && fo !== nothing
            # Write the final result to the output file
            println("Writing to output h5 file: dw/dz")
            write_compressed_dataset(fo, "/dwdz", dwdz, compressed=false)
        end
    end

    # return w_phys or p_phys
    if rank == root
        if pflag == true
            return w_phys # p_phys
        elseif VGT == true
            return (w_phys, dwdx, dwdy, dwdz) # w_phys and its derivatives
        else        
            return w_phys
        end
    else
        if VGT == true
            # Return empty arrays with same structure when not on root
            return (nothing, nothing, nothing, nothing)
        else
            return nothing
        end
    end
end


function compute_z_average_field(f, root, comm::MPI.Comm)
#===========================================================================#
# Function to average the field f in the z-direction (homogeneous direction)
# Note: f is only passed in via the root cpu and scattered for processing
#===========================================================================#
    rank    = MPI.Comm_rank(comm)
    no_cpus = MPI.Comm_size(comm)
    
    # Set up data information
    if rank == root
        # Handle case where f is a tuple (returned from VGT functions)
        if isa(f, Tuple)
            # Extract just the first element (velocity field) from the tuple
            f_array = f[1]
            (NZ, NY, NX) = size(f_array)
            eltype_f     = eltype(f_array)
        else
            (NZ, NY, NX) = size(f)
            eltype_f     = eltype(f)
        end
    else
        (NZ, NY, NX, eltype_f) = (nothing, nothing, nothing, nothing)
    end 
    NZ       = MPI.bcast(NZ, root, comm)
    NY       = MPI.bcast(NY, root, comm)
    NX       = MPI.bcast(NX, root, comm)
    
    # Broadcast the element type as a string and parse it back to a type
    if rank == root
        eltype_str = string(eltype_f)
    else
        eltype_str = ""
    end
    eltype_str = MPI.bcast(eltype_str, root, comm)
    eltype_f = eval(Meta.parse(eltype_str))
    
    MPI.Barrier(comm)

    # Number of x slices per rank
    local_nx = split_count(NX, no_cpus)

    if rank == root
        slice_size   = NZ*NY      # Size of each 2D plane corresponding to each x for scattering
        slice_size_o = NY         # Size of each 1D plane corresponding to each x for gathering
        counts       = [nx * slice_size   for nx in local_nx]
        counts_o     = [nx * slice_size_o for nx in local_nx]
    end
    # Define Local arrays:
    # Local 3D slices to each rank
    local_size  = (NZ, NY, local_nx[rank + 1])
    local_array = zeros(eltype_f, local_size)   

    # scatter the 3D array ...  
    # Define data to scatter and its VBuffer for scattering
    if rank == root
        # Extract the correct array from f if it's a tuple
        in_array = isa(f, Tuple) ? f[1] : f
        global_vbuf = VBuffer(in_array, counts)  # Create VBuffers for scattering
    else
        global_vbuf = nothing
    end
    MPI.Scatterv!(global_vbuf, local_array, root, comm)

    # Compute the local average along the NZ dimension
    local_avg_f = reshape(mean(local_array, dims=1), NY, local_nx[rank + 1])
    
    # Define storage and VBuffer for gathering
    if rank == root
        out_array   = zeros(eltype_f, (NY, NX))
        output_vbuf = VBuffer(out_array, counts_o) # Create VBuffers for gathering
    else
        output_vbuf = nothing
    end
    # Gather the modified blocks back to the root process, one block, i.e. number of x-plane 
    MPI.Gatherv!(local_avg_f, output_vbuf, root, comm)
    
    if rank == 0
        return out_array
    else
        return nothing
    end
end

#=======================================================================================================================================================#
# End of Functions 
#=======================================================================================================================================================#

