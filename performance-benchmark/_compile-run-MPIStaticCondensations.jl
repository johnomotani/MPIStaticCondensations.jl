include("common.jl")
include("benchmark-MPIStaticCondensations.jl")

compile_params = BenchmarkParams([4, 4, 4], [3, 3, 3], true)

level_multiplier = 2

function compile_run()
    if !MPI.Initialized()
        MPI.Init()
    end

    BLAS.set_num_threads(1)

    comm_size = MPI.Comm_size(MPI.COMM_WORLD)
    comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    n_shared = comm_size

    run_benchmark(run_MSC, compile_params, 42, nothing, n_shared, true, level_multiplier)

    return nothing
end

compile_run()
