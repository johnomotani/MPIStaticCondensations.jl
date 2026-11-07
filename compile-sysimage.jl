using PackageCompiler

# Create a sysimage 'mpisc.so'.
create_sysimage(; sysimage_path="mpisc.so",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3 -t1`, # Assume if we are precompiling we want an optimized, production build
               )
