include("benchmark-MUMPS-shared.jl")

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("MUMPS multi-variable benchmark")
    println("==============================\n")
end
benchmark(run_MUMPS, params_multivariable_1d, seed_multivariable_1d, "MUMPS_multivariable_1d", true, true; use_shared=false)
benchmark(run_MUMPS, params_multivariable_2d, seed_multivariable_2d, "MUMPS_multivariable_2d", true, true; use_shared=false)
benchmark(run_MUMPS, params_multivariable_3d, seed_multivariable_3d, "MUMPS_multivariable_3d", true, true; use_shared=false)
