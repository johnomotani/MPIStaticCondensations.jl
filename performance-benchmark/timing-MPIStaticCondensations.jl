using TimerOutputs
using TimerOutputComparisons
using Debugger

include("common.jl")
include("benchmark-MPIStaticCondensations.jl")

#timing_params = BenchmarkParams([2], [3], true)
#timing_params = BenchmarkParams([8], [3], true; mumps_fill_in_threshold=0.1, sparse_C_blocks=true)
#timing_params = BenchmarkParams([32, 32], [5, 5], true)
#timing_params = BenchmarkParams([32, 32], [9, 9], false)
#timing_params = BenchmarkParams([16, 8, 16], [5, 5, 5], true)
#timing_params = BenchmarkParams([4, 4, 4], [5, 5, 5], true)
#timing_params = BenchmarkParams([8, 4, 8], [5, 5, 5], true)
#timing_params = BenchmarkParams([8, 8, 8], [3, 4, 5], true)
#timing_params = BenchmarkParams([8, 8, 8], [5, 5, 5], true)
#timing_params = BenchmarkParams([8, 8, 8], [5, 5, 5], true; sparse_C_blocks=true)
#timing_params = BenchmarkParams([8, 4, 16], [5, 5, 5], true)
#timing_params = BenchmarkParams([16, 8, 16], [5, 5, 5], false)
#timing_params = BenchmarkParams([16, 8, 16], [5, 5, 5], true)
#timing_params = BenchmarkParams([16, 16, 16], [5, 5, 5], true)
#timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true)
#timing_params = BenchmarkParams([32, 16, 32], [5, 5, 5], true)
#timing_params = BenchmarkParams([16, 16, 16], [5, 5, 5], true; sparse_C_blocks=true)
timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true; sparse_C_blocks=true)
#timing_params = BenchmarkParams([32, 16, 32], [5, 5, 5], true; sparse_C_blocks=true)
#timing_params = BenchmarkParams([32, 16, 32], [3, 3, 3], true)
#timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true; mumps_fill_in_threshold=1.0e-4, sparse_C_blocks=true)
#timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true; mumps_fill_in_threshold=1.0e-3, sparse_C_blocks=true)
#timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true; mumps_fill_in_threshold=1.0e-2, sparse_C_blocks=true)
#timing_params = BenchmarkParams([16, 16, 32], [5, 5, 5], true; mumps_fill_in_threshold=1.0e-1, sparse_C_blocks=true)

level_multiplier = 2

using Debugger, Cthulhu
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
