using TimerOutputs

include("common.jl")
include("benchmark-MPIStaticCondensations.jl")

timing_params = BenchmarkParams([32, 32], [5, 5], true)

function timing_run()
    if !MPI.Initialized()
        MPI.Init()
    end

    BLAS.set_num_threads(1)

    timer = TimerOutput()

    run_benchmark(run_MSC, timing_params, 42, nothing, parse(Int64, ARGS[1]), true, timer)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        display(timer)
    end

    return nothing
end

timing_run()
