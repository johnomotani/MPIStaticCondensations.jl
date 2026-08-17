include("benchmark-serial-UMFPACK-shared.jl")

println("UMFPACK multi-variable benchmark")
println("================================\n")
benchmark(run_UMFPACK, params_multivariable_1d, seed_multivariable_1d, "UMFPACK_multivariable_1d", true, true; use_shared=false)
benchmark(run_UMFPACK, params_multivariable_2d, seed_multivariable_2d, "UMFPACK_multivariable_2d", true, true; use_shared=false)
benchmark(run_UMFPACK, params_multivariable_3d, seed_multivariable_3d, "UMFPACK_multivariable_3d", true, true; use_shared=false)
