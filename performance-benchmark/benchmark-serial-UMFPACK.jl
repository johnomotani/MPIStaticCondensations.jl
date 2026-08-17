include("benchmark-serial-UMFPACK-shared.jl")

println("UMFPACK benchmark")
println("=================\n")
benchmark(run_UMFPACK, params_1d, seed_1d, "UMFPACK_1d", true, true; use_shared=false)
benchmark(run_UMFPACK, params_2d, seed_2d, "UMFPACK_2d", true, true; use_shared=false)
benchmark(run_UMFPACK, params_3d, seed_3d, "UMFPACK_3d", true, true; use_shared=false)
