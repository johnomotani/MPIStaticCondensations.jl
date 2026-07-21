using TimerOutputs
using TimerOutputComparisons

include("common.jl")
include("benchmark-MPIStaticCondensations.jl")
include("print_git_info.jl")

timing_params = BenchmarkParams([32, 32], [5, 5], true)

level_multiplier = 2

function timing_run()
    if !MPI.Initialized()
        MPI.Init()
    end

    BLAS.set_num_threads(1)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        print_git_info()
    end

    comm_size = MPI.Comm_size(MPI.COMM_WORLD)
    comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    n_shared = parse(Int64, ARGS[1])
    timer = TimerOutput()

    run_benchmark(run_MSC, timing_params, 42, nothing, n_shared, true, level_multiplier, timer)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        save_timer("timing-$comm_size-$n_shared.jld", timer)
        display(timer)
        #display(TimerOutputs.flatten(timer))
    end
    open("timing-proc$comm_rank.txt", "w") do io
        show(io, timer)
        println(io)
    end

    return nothing
end

timing_run()
