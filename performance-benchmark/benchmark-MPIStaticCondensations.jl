using BlockBandedMatrices
using LinearAlgebra
using MPI
using MPIStaticCondensations
using SparseArrays
using StableRNGs
using StatsBase

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

function run_MSC(x, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
                 dimensions, comm, distributed_comm, shared_comm, allocate_shared_float,
                 allocate_shared_int, nmat, nrhs, timer)

    outer_dim_steps = prod(d.n for d ∈ dimensions[1:end-1]; init=1)
    nelement_local = dimensions[end].nelement ÷ dimensions[end].nrank
    block_sizes, off_diagonals = get_block_sizes(nelement_local, dimensions[end].ngrid,
                                                 outer_dim_steps)
    n_total = sum(block_sizes)
    # May not need BlockSkylineMatrix for this test, but it is what we use in
    # moment_kinetics, so is the most relevant choice.
    A = BlockSkylineMatrix{Float64}(BlockBandedMatrices.Zeros(n_total, n_total),
                                    block_sizes, block_sizes,
                                    (off_diagonals, off_diagonals))

    for (entry, i, j) ∈ zip(data, local_i, local_j)
        A[i,j] = entry
    end

    t1 = time_ns()
    Alu = mpi_static_condensation(dimensions; comm, distributed_comm, shared_comm,
                                  allocate_shared_float, allocate_shared_int,
                                  schur_tile_size=nothing, use_sparse=true,
                                  separate_Ainv_B=false,
#separate_Ainv_B=true,
                                  optimize_schur_complement_size=true, timer,
                                  check_lu=false)
    t2 = time_ns()
    t_setup = (t2 - t1) * 1e-6 # in ms

    # The mpi_static_condensation() constructor is not type stable, as the solver type
    # depends on the number of levels and on the options chosen. Therefore the main
    # performance test must be in a separate inner function, that can be compiled knowing
    # the concrete type of Alu.

    return t_setup, run_MSC_inner(Alu, A, x, rhs)...
end

function run_MSC_inner(Alu, A, x, rhs)
    t_lu = Inf
    t_solve = Inf
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
    #benchmark(run_MSC, params_3d, seed_3d, "MPIStaticCondensations_3d")

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
