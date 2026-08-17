include("benchmark-MPIStaticCondensations-shared.jl")

function run_benchmarks()
    BLAS.set_num_threads(1)

    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("MPIStaticCondensations multi-variable benchmark")
        println("===============================================\n")
    end

    benchmark(run_MSC, params_multivariable_1d, seed_multivariable_1d, "MPIStaticCondensations_multivariable_1d", false, false)
    benchmark(run_MSC, params_multivariable_2d, seed_multivariable_2d, "MPIStaticCondensations_multivariable_2d", false, false)
    benchmark(run_MSC, params_multivariable_3d, seed_multivariable_3d, "MPIStaticCondensations_multivariable_3d", false, false)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
