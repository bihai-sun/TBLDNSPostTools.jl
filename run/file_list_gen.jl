
##
base_dir = "../../fields/time_resolved_run1_dt40_steps"


raw_file_names = readdir(base_dir)

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
        full_path = joinpath(base_dir, file)
        name_without_ext = replace(full_path, ".u.h5" => "")
        println(io, name_without_ext)
    end
end

println("\nSaved $(length(u_files)) file paths to $output_file")
println("Each entry contains the full path with .u.h5 extension stripped")

