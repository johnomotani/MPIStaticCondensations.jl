include("benchmark-MPIStaticCondensations-shared.jl")

function run_benchmarks()
    BLAS.set_num_threads(1)

    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("MPIStaticCondensations benchmark")
        println("================================\n")
    end

    benchmark(run_MSC, params_1d, seed_1d, "MPIStaticCondensations_1d", false, false)
    benchmark(run_MSC, params_2d, seed_2d, "MPIStaticCondensations_2d", false, false)
    benchmark(run_MSC, params_3d, seed_3d, "MPIStaticCondensations_3d", false, false)

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
