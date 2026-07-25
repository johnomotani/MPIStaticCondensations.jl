using Combinatorics
using Dates
using MPIStaticCondensations
using Primes
using TimerOutputs

include("print_git_info.jl")

const nmat = 1
const nrhs = 1
const matrix_repeats = 4
const rhs_repeats = 100

const variable_dimensions_1d = (nothing, [1], [1], [1])
const variable_dimensions_2d = (nothing, [2], [2], [2])
const variable_dimensions_3d = (nothing, [3], [3], [3])
const stencil_matrix = ["element" "empty"   "point"   "element";
                        "element" "point"   "element" "empty";
                        "empty"   "element" "element" "point";
                        "point"   "element" "empty"   "element"]

const results_directory = "results-benchmark"

struct BenchmarkParams
    nelement_list::Vector{Int64}
    ngrid_list::Vector{Int64}
    variable_dimensions::Union{Vector{Vector{Int64}},Nothing}
    stencil_matrix::Union{Matrix{String},Nothing}
    sparse_stencils::Bool
    periodic_list::Vector{Bool}
    remove_boundaries_list::Vector{Bool}
    sparse_C_blocks::Bool
    mumps_fill_in_threshold::Float64
    block_sizes_heuristic::Union{MPIStaticCondensations.BlockSizesHeuristic,Vector{Vector{Int64}}}

    function BenchmarkParams(nelement_list, ngrid_list, sparse_stencils;
                             variable_dimensions=nothing, stencil_matrix=nothing,
                             periodic_list=nothing, remove_boundaries_list=nothing,
                             sparse_C_blocks=false, mumps_fill_in_threshold=1.0,
                             block_sizes_heuristic=MPIStaticCondensations.FastSlow())
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

        if variable_dimensions !== nothing
            nd = length(nelement_list)
            variable_dimensions = [vdims === nothing ? (1:nd) : vdims
                                   for vdims ∈ variable_dimensions]
        end

        return new(nelement_list, ngrid_list, variable_dimensions, stencil_matrix,
                   sparse_stencils, periodic_list, remove_boundaries_list,
                   sparse_C_blocks, mumps_fill_in_threshold, block_sizes_heuristic)
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
    BenchmarkParams([8, 4, 8], [5, 5, 5], true),
    BenchmarkParams([16, 8, 16], [5, 5, 5], true),
    BenchmarkParams([32, 16, 32], [5, 5, 5], true),
)
const seed_3d = 333

const params_multivariable_1d = (
    BenchmarkParams([32], [5], true; variable_dimensions=variable_dimensions_1d, stencil_matrix),
    BenchmarkParams([64], [9], true; variable_dimensions=variable_dimensions_1d, stencil_matrix),
    BenchmarkParams([128], [17], true; variable_dimensions=variable_dimensions_1d, stencil_matrix),
)
const seed_multivariable_1d = 112

const params_multivariable_2d = (
    BenchmarkParams([8, 8], [5, 5], true; variable_dimensions=variable_dimensions_2d, stencil_matrix),
    BenchmarkParams([16, 16], [9, 9], true; variable_dimensions=variable_dimensions_2d, stencil_matrix),
    BenchmarkParams([32, 32], [5, 5], true; variable_dimensions=variable_dimensions_2d, stencil_matrix),
    BenchmarkParams([32, 32], [9, 9], true; variable_dimensions=variable_dimensions_2d, stencil_matrix),
)
const seed_multivariable_2d = 223

const params_multivariable_3d = (
    BenchmarkParams([8, 4, 8], [5, 5, 5], true; variable_dimensions=variable_dimensions_3d, stencil_matrix),
    BenchmarkParams([16, 8, 16], [5, 5, 5], true; variable_dimensions=variable_dimensions_3d, stencil_matrix),
    BenchmarkParams([32, 16, 32], [5, 5, 5], true; variable_dimensions=variable_dimensions_3d, stencil_matrix),
)
const seed_multivariable_3d = 334

include("../test/utils.jl")
include("../test/generate_finite_element_matrices.jl")

function get_matrix(dimensions, variable_dimensions, stencil_matrix, sparse_stencils, rng,
                    comm, distributed_comm, shared_comm, allocate_shared_float,
                    allocate_shared_int, matrix_return_separate, matrix_combine_blocks)

    if variable_dimensions === nothing
        return assemble_and_scatter_global_matrix(
                   dimensions, comm, distributed_comm, shared_comm, allocate_shared_float,
                   allocate_shared_int, rng, sparse_stencils;
                   return_separate=matrix_return_separate)
    else
        return assemble_and_scatter_global_multi_variable_matrix(
                   dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                   allocate_shared_float, allocate_shared_int, rng, sparse_stencils;
                   return_separate=matrix_return_separate, stencil_matrix=stencil_matrix,
                   combine_blocks=matrix_combine_blocks)
    end
end

function get_rhs(dimensions, variable_dimensions, rng, comm, distributed_comm,
                 shared_comm, allocate_shared_float)
    if variable_dimensions === nothing
        rhs_global, rhs =
            assemble_and_scatter_global_rhs(
                dimensions, comm, distributed_comm, shared_comm, allocate_shared_float,
                rng)
    else
        rhs_global, rhs =
            assemble_and_scatter_global_multi_variable_rhs(
                dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                allocate_shared_float, rng)
    end
    return rhs, rhs_global
end

function run_benchmark(run_solver::T, params, seed, label, n_shared, use_shared,
                       matrix_return_separate, matrix_combine_blocks,
                       timer=nothing) where T
    rng = StableRNG(seed)

    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    if use_shared
        ns = n_shared
    else
        ns = Threads.nthreads()
    end
    nproc = distributed_nproc * n_shared * Threads.nthreads()
    ndim = length(params.nelement_list)

    if distributed_rank == 0 && shared_rank == 0
        println(now(), "\nRunning nproc=$nproc, n_shared=$n_shared, n_threads=$(Threads.nthreads()), $params")
    end

    nrank_list = ones(Int64, ndim)
    # For now, only distribute the last dimension.
    if distributed_nproc > params.nelement_list[end] || params.nelement_list[end] % distributed_nproc != 0
        # Cannot parallelise in this way, so skip.
        if distributed_rank == 0 && shared_rank == 0
            println("Parallelisation does not fit this grid, skipping...\n")
        end
        return nothing
    end
    nrank_list[end] = distributed_nproc
    irank_list = get_iranks(nrank_list, distributed_rank)
    dimensions = [create_dimension(; name="d$i", nelement, ngrid, nrank, irank, periodic,
                                   remove_boundaries)
                  for (i, (nelement, ngrid, irank, nrank, periodic, remove_boundaries))
                  ∈ enumerate(zip(params.nelement_list, params.ngrid_list, irank_list,
                                  nrank_list, params.periodic_list,
                                  params.remove_boundaries_list))]

    # First run ensures solver is compiled for these parameters. Do not save these timings
    # as we do not want to measure compilation time.
    matrix_data =
        get_matrix(dimensions, params.variable_dimensions, params.stencil_matrix,
                   params.sparse_stencils, rng, comm, distributed_comm, shared_comm,
                   allocate_shared_float, allocate_shared_int, matrix_return_separate,
                   matrix_combine_blocks)
    rhs, rhs_global = get_rhs(dimensions, params.variable_dimensions, rng, comm,
                              distributed_comm, shared_comm, allocate_shared_float)
    x_temp = allocate_shared_float(length(rhs))
    run_solver(x_temp, matrix_data, rhs, rhs_global, dimensions,
               params.variable_dimensions, params.sparse_C_blocks,
               params.mumps_fill_in_threshold, params.block_sizes_heuristic, comm,
               distributed_comm, shared_comm, allocate_shared_float, allocate_shared_int,
               1, 1, 1, 1, timer)

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

    if timer !== nothing
        reset_timer!(timer)
    end
    t_setup = Float64[]
    t_lu = Float64[]
    t_solve = Float64[]
    for imat ∈ 1:nmat
        matrix_data =
            get_matrix(dimensions, params.variable_dimensions, params.stencil_matrix,
                       params.sparse_stencils, rng, comm, distributed_comm, shared_comm,
                       allocate_shared_float, allocate_shared_int, matrix_return_separate,
                       matrix_combine_blocks)
        for irhs ∈ 1:nrhs
            rhs, rhs_global = get_rhs(dimensions, params.variable_dimensions, rng, comm,
                                      distributed_comm, shared_comm,
                                      allocate_shared_float)
            x = allocate_shared_float(length(rhs))
            this_t_setup, this_t_lu, this_t_solve =
                run_solver(x, matrix_data, rhs, rhs_global, dimensions,
                           params.variable_dimensions, params.sparse_C_blocks,
                           params.mumps_fill_in_threshold, params.block_sizes_heuristic,
                           comm, distributed_comm, shared_comm, allocate_shared_float,
                           allocate_shared_int, nmat, nrhs, matrix_repeats, rhs_repeats,
                           timer)
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
            run_dir = mkpath(results_directory)
            total_size = prod(d.n for d ∈ dimensions)
            function vec2string(v)
                if v === nothing
                    return "nothing"
                elseif !isa(v, AbstractVector)
                    return v
                elseif eltype(v) <: AbstractVector
                    return "[" * join([vec2string(x) for x ∈ v], ",") * "]"
                else
                    return "[" * join(v, ",") * "]"
                end
            end
            open(joinpath(run_dir, "benchmarks_$label.txt"), "a") do io
                println(io, "$nproc $ns $ndim $total_size $mean_setup $mean_lu $mean_solve $(vec2string(params.nelement_list)) $(vec2string(params.ngrid_list)) $(vec2string(params.periodic_list)) $(vec2string(params.remove_boundaries_list)) $(params.sparse_C_blocks) $(params.mumps_fill_in_threshold) $(vec2string(params.block_sizes_heuristic))")
            end
        end
    end

    return nothing
end

function benchmark(run_solver::T, params, seed, label, matrix_return_separate,
                   matrix_combine_blocks; use_shared=true) where T
    if !MPI.Initialized()
        MPI.Init()
    end

    if label !== nothing && MPI.Comm_rank(MPI.COMM_WORLD) == 0 && !isempty(params)
        run_dir = mkpath(results_directory)
        open(joinpath(run_dir, "provenance_$label.txt"), "a") do io
            println(io, round(now(), Dates.Second))
            print_git_info(io)
            println(io, "="^100)
            println(io)
        end
    end

    comm_size = MPI.Comm_size(MPI.COMM_WORLD)

    if use_shared
        n_shared_values = comm_size #[prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
    else
        # When use_shared=false, we set up the matrix with shared-memory, but then divide
        # it into distributed chunks to pass to MUMPS.
        n_shared_values = comm_size #[prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
    end
    for n_shared ∈ n_shared_values
        for p ∈ params
            run_benchmark(run_solver, p, seed, label, n_shared, use_shared,
                          matrix_return_separate, matrix_combine_blocks)
            seed += 1
        end
    end

    return nothing
end

