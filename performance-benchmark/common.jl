using Combinatorics
using Dates
using MPIStaticCondensations
using Primes
using TimerOutputs

const nmat = 4
const nrhs = 10
const matrix_repeats = 4
const rhs_repeats = 10

struct BenchmarkParams
    nelement_list::Vector{Int64}
    ngrid_list::Vector{Int64}
    sparse_stencils::Bool
    periodic_list::Vector{Bool}
    remove_boundaries_list::Vector{Bool}

    function BenchmarkParams(nelement_list, ngrid_list, sparse_stencils,
                             periodic_list=nothing, remove_boundaries_list=nothing)
        n = length(nelement_list)
        if periodic_list === nothing
            periodic_list = fill(false, n)
        end
        if remove_boundaries_list === nothing
            remove_boundaries_list = fill(false, n)
        end

        if !(length(nelement_list) == length(ngrid_list) == length(periodic_list) == length(remove_boundaries_list))
            error("length of all parameter lists must be the same")
        end

        return new(nelement_list, ngrid_list, sparse_stencils, periodic_list,
                   remove_boundaries_list)
    end
end

const params_1d = (
    BenchmarkParams([32], [5], true),
    BenchmarkParams([64], [9], true),
    BenchmarkParams([128], [17], true),
)
const seed_1d = 111

const params_2d = (
    BenchmarkParams([8, 8], [5, 5], true),
    BenchmarkParams([16, 16], [9, 9], true),
    BenchmarkParams([32, 32], [5, 5], true),
    BenchmarkParams([32, 32], [9, 9], true),
)
const seed_2d = 222

const params_3d = (
    BenchmarkParams([8, 8, 8], [5, 5, 5], true),
    BenchmarkParams([16, 8, 16], [9, 9, 9], true),
    BenchmarkParams([32, 16, 32], [5, 5, 5], true),
    BenchmarkParams([32, 16, 32], [9, 9, 9], true),
)
const seed_3d = 333

include("../test/utils.jl")
include("../test/generate_finite_element_matrices.jl")

function get_matrix(dimensions, sparse_stencils, rng, comm, distributed_comm, shared_comm,
                    allocate_shared_float, allocate_shared_int)

    _, data, global_i, global_j, local_i, local_j =
        assemble_and_scatter_global_matrix(dimensions, comm, distributed_comm,
                                           shared_comm, allocate_shared_float,
                                           allocate_shared_int, rng, sparse_stencils;
                                           return_sparse=true)
    return data, global_i, global_j, local_i, local_j
end

function get_rhs(dimensions, rng, comm, distributed_comm, shared_comm,
                 allocate_shared_float)
    rhs_global, rhs =
        assemble_and_scatter_global_rhs(dimensions, comm, distributed_comm, shared_comm,
                                        allocate_shared_float, rng)
    return rhs, rhs_global
end

function run_benchmark(run_solver::T, params, seed, label, n_shared, use_shared, timer=nothing) where T
    rng = StableRNG(seed)

    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    nproc = distributed_nproc * shared_nproc
    ndim = length(params.nelement_list)

    if distributed_rank == 0 && shared_rank == 0
        println(now(), "\nRunning nproc=$nproc, n_shared=$n_shared, n_threads=$(Threads.nthreads()), $params")
    end

    # For now, only distribute the last dimension.
    if distributed_nproc > params.nelement_list[end] || params.nelement_list[end] % distributed_nproc != 0
        # Cannot parallelise in this way, so skip.
        if distributed_rank == 0 && shared_rank == 0
            println("Parallelisation does not fit this grid, skipping...\n")
        end
        return nothing
    end
    nrank_list = ones(Int64, ndim)
    nrank_list[end] = distributed_nproc
    irank_list = get_iranks(nrank_list, distributed_rank)
    dimensions = [create_dimension(; nelement, ngrid, nrank, irank, periodic, remove_boundaries)
                  for (nelement, ngrid, irank, nrank, periodic, remove_boundaries)
                  ∈ zip(params.nelement_list, params.ngrid_list, irank_list, nrank_list,
                        params.periodic_list, params.remove_boundaries_list)]

    # First run ensures solver is compiled for these parameters. Do not save these timings
    # as we do not want to measure compilation time.
    data, global_i, global_j, local_i, local_j =
        get_matrix(dimensions, params.sparse_stencils, rng, comm, distributed_comm,
                   shared_comm, allocate_shared_float, allocate_shared_int)
    rhs, rhs_global = get_rhs(dimensions, rng, comm, distributed_comm, shared_comm,
                              allocate_shared_float)
    x_temp = allocate_shared_float(length(rhs))
    run_solver(x_temp, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
               dimensions, comm, distributed_comm, shared_comm, allocate_shared_float,
               allocate_shared_int, 1, 1, timer)

    if timer !== nothing
        reset_timer!(timer)
    end
    t_setup = Float64[]
    t_lu = Float64[]
    t_solve = Float64[]
    for imat ∈ 1:nmat
        data, global_i, global_j, local_i, local_j =
            get_matrix(dimensions, params.sparse_stencils, rng, comm, distributed_comm,
                       shared_comm, allocate_shared_float, allocate_shared_int)
        for irhs ∈ 1:nrhs
            rhs, rhs_global = get_rhs(dimensions, rng, comm, distributed_comm,
                                      shared_comm, allocate_shared_float)
            x = allocate_shared_float(length(rhs))
            this_t_setup, this_t_lu, this_t_solve =
                run_solver(x, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
                           dimensions, comm, distributed_comm, shared_comm,
                           allocate_shared_float, allocate_shared_int, nmat, nrhs, timer)
            push!(t_setup, this_t_setup)
            push!(t_lu, this_t_lu)
            push!(t_solve, this_t_solve)
        end

        if local_win_store_float !== nothing
            # Free the MPI.Win objects, because if they are free'd by the garbage collector
            # it may cause an MPI error or hang.
            for w ∈ local_win_store_float
                MPI.free(w)
            end
        end
        if local_win_store_int !== nothing
            # Free the MPI.Win objects, because if they are free'd by the garbage collector
            # it may cause an MPI error or hang.
            for w ∈ local_win_store_int
                MPI.free(w)
            end
        end
        MPI.Barrier(shared_comm)
    end

    # Average over different matrices and rhs.
    mean_setup = mean(t_setup)
    mean_lu = mean(t_lu)
    mean_solve = mean(t_solve)

    if distributed_rank == 0 && shared_rank == 0
        println("  setup = $mean_setup ms; LU = $mean_lu ms; solve = $mean_solve ms\n")
        if label !== nothing
            run_dir = mkpath("results-benchmark")
            total_size = prod(d.n for d ∈ dimensions)
            if use_shared
                ns = n_shared
            else
                ns = Threads.nthreads()
            end
            open(joinpath(run_dir, "benchmarks_$label.txt"), "a") do io
                println(io, "$nproc $ns $ndim $total_size $mean_setup $mean_lu $mean_solve $(params.nelement_list) $(params.ngrid_list) $(params.periodic_list) $(params.remove_boundaries_list)")
            end
        end
    end

    return nothing
end

function benchmark(run_solver::T, params, seed, label; use_shared=true) where T
    if !MPI.Initialized()
        MPI.Init()
    end

    comm_size = MPI.Comm_size(MPI.COMM_WORLD)

    if use_shared
        n_shared_values = [prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
    else
        n_shared_values = 1
    end
    for n_shared ∈ n_shared_values
        for p ∈ params
            run_benchmark(run_solver, p, seed, label, n_shared, use_shared)
            seed += 1
        end
    end

    return nothing
end

