using PackageCompiler

# Create the sysimage 'mpistaticcondensations.so'
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-compile if you change anything.
create_sysimage(; sysimage_path="mpistaticcondensations.so",
                precompile_execution_file="_compile-run-MPIStaticCondensations.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3`, # Assume if we are precompiling we want an optimized, production build
               )
