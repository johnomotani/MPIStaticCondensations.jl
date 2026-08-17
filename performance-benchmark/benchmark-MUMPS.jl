include("benchmark-MUMPS-shared.jl")

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("MUMPS benchmark")
    println("================\n")
end
benchmark(run_MUMPS, params_1d, seed_1d, "MUMPS_1d", true, true; use_shared=false)
benchmark(run_MUMPS, params_2d, seed_2d, "MUMPS_2d", true, true; use_shared=false)
benchmark(run_MUMPS, params_3d, seed_3d, "MUMPS_3d", true, true; use_shared=false)
