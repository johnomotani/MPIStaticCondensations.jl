using LinearAlgebra
using MPI
using MPIStaticCondensations
using MUMPS
using SparseArrays
using StableRNGs
using StatsBase
using TimerOutputs

include("common.jl")

function get_block_sizes(outer_nelement, outer_ngrid, inner_dims_length)
    # Can represent a continuous finite-element matrix a block-structured way. The matrix
    # entries where both row and column are in the interior of an element of the 'outer'
    # (last) dimension are 'a', where both are an element boundary are 'd', and the rest
    # are 'b' and 'c'.
    # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
    # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
    # a  a  a  │  b  │  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  ⋅
    # ─────────┼─────┼────────┼─────┼─────────
    # c  c  c  │  d  │  c  c  │  d  │  ⋅  ⋅  ⋅
    # ─────────┼─────┼────────┼─────┼─────────
    # ⋅  ⋅  ⋅  │  b  │  a  a  │  b  │  ⋅  ⋅  ⋅
    # ⋅  ⋅  ⋅  │  b  │  a  a  │  b  │  ⋅  ⋅  ⋅
    # ─────────┼─────┼────────┼─────┼─────────
    # ⋅  ⋅  ⋅  │  d  │  c  c  │  d  │  c  c  c
    # ─────────┼─────┼────────┼─────┼─────────
    # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a
    # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a
    # ⋅  ⋅  ⋅  │  ⋅  │  ⋅  ⋅  │  b  │  a  a  a

    if outer_nelement == 1
        outer_block_sizes = [outer_ngrid]
    else
        outer_block_sizes = [outer_ngrid - 1]
        push!(outer_block_sizes, 1)
        for ielement ∈ 2:outer_nelement-1
            push!(outer_block_sizes, outer_ngrid - 2)
            push!(outer_block_sizes, 1)
        end
        push!(outer_block_sizes, outer_ngrid - 1)
    end

    block_sizes = outer_block_sizes .* inner_dims_length

    # Need one 'off diagonal' block on either side inside elements, but two 'off
    # diagonal' blocks for element boundaries.
    off_diagonals = [(i - 1) % 2 + 1 for i ∈ 1:2*outer_nelement-1]
    return block_sizes, off_diagonals
end

function run_MSC(x, matrix_data, rhs, rhs_global, dimensions, variable_dimensions,
                 sparse_C_blocks, mumps_fill_in_threshold, block_sizes_heuristic, comm,
                 distributed_comm, shared_comm, allocate_shared_float,
                 allocate_shared_int, nmat, nrhs, matrix_repeats, rhs_repeats, timer)

    global_matrix, A = matrix_data

    comm_rank = MPI.Comm_rank(comm)
    outer_dim_steps = prod(d.n for d ∈ dimensions[1:end-1]; init=1)
    nelement_local = dimensions[end].nelement ÷ dimensions[end].nrank
    block_sizes, off_diagonals = get_block_sizes(nelement_local, dimensions[end].ngrid,
                                                 outer_dim_steps)

    t1 = time_ns()
    Alu = mpi_static_condensation(dimensions; variable_dimensions, block_sizes_heuristic,
                                  sparse_C_blocks, mumps_fill_in_threshold, comm,
                                  distributed_comm, shared_comm, allocate_shared_float,
                                  allocate_shared_int, schur_tile_size=nothing,
                                  separate_Ainv_B=false, timer, check_lu=false)
    t2 = time_ns()
    t_setup = (t2 - t1) * 1e-6 # in ms

    # The mpi_static_condensation() constructor is not type stable, as the solver type
    # depends on the number of levels and on the options chosen. Therefore the main
    # performance test must be in a separate inner function, that can be compiled knowing
    # the concrete type of Alu.

    return t_setup, run_MSC_inner(Alu, A, x, rhs, matrix_repeats, rhs_repeats, comm_rank,
                                  global_matrix)...
end

function run_MSC_inner(Alu, A, x, rhs, matrix_repeats, rhs_repeats, comm_rank,
                       global_matrix)
    t_lu = Inf
    t_solve = Inf
    # Run once before the loop to try to ensure that sparse buffer arrays have been filled
    # with the right number of entries, so that we are not timing initial array
    # allocation.
    if Alu.timer !== nothing
        disable_timer!(Alu.timer)
    end
    lu!(Alu, A)
    if Alu.timer !== nothing
        enable_timer!(Alu.timer)
    end
    for _ ∈ 1:matrix_repeats
        t1 = time_ns()
        lu!(Alu, A)
        t2 = time_ns()
        t_lu = min(t_lu, (t2 - t1) * 1e-6)

        for _ ∈ 1:rhs_repeats
            t1 = time_ns()
            ldiv!(x, Alu, rhs)
            t2 = time_ns()
            t_solve = min(t_solve, (t2 - t1) * 1e-6)

            # Check solution, just to be on the safe side...
            Alu.synchronize_shared()
            if comm_rank == 0
                max_error = maximum(abs.(global_matrix * x - rhs))
                if max_error > 1.0e-3
                    println("Solution incorrect? Max error $max_error.")
                    MPI.Abort(MPI.COMM_WORLD, -1)
                end
            end
        end
    end

    return t_lu, t_solve
end

function run_benchmarks()
    BLAS.set_num_threads(1)

    if !MPI.Initialized()
        MPI.Init()
    end
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("MPIStaticCondensations benchmark")
        println("================================\n")
    end

    benchmark(run_MSC, params_1d, seed_1d, "MPIStaticCondensations_1d")
    benchmark(run_MSC, params_2d, seed_2d, "MPIStaticCondensations_2d")
    benchmark(run_MSC, params_3d, seed_3d, "MPIStaticCondensations_3d")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
