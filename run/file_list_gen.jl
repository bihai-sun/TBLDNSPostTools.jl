
##
base_dir = "../../rawdata_long_interval"

function list_files_recursive(dir_path::AbstractString)
    file_list = String[]
    for (root, dirs, files) in walkdir(dir_path)
        println("Directory: $root")

        for f in files
            push!(file_list, joinpath(root, f))
        end
    end
    return file_list
end


raw_file_names = list_files_recursive(base_dir)

# Find all files matching *.u.h5 pattern
u_files = filter(name -> endswith(name, ".u.h5"), raw_file_names)

# Display the results
println("Found $(length(u_files)) files matching *.u.h5:")
for file in u_files
    println("  $file")
end

# Save to text file with full path and stripped extension
output_file = "files_list.txt"
open(output_file, "w") do io
    for file in u_files
        # Create full path and strip .u.h5 extension
        name_without_ext = replace(file, ".u.h5" => "")
        println(io, name_without_ext)
    end
end

println("\nSaved $(length(u_files)) file paths to $output_file")
println("Each entry contains the full path with .u.h5 extension stripped")

