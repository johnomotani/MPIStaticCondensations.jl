using BlockArrays
using LinearAlgebra
using MPI
using MPISchurComplements
using MPIStaticCondensations
using MPIStaticCondensations: OuterBSubmatrix, OuterCSubmatrix
using SparseArrays
using StableRNGs
using StatsBase
using TimerOutputs

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

function run_MSC(x, matrix, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
                 dimensions, level_multiplier, sparse_C_blocks, comm, distributed_comm,
                 shared_comm, allocate_shared_float, allocate_shared_int, nmat, nrhs,
                 matrix_repeats, rhs_repeats, timer)

    if matrix === nothing
        outer_dim_steps = prod(d.n for d ∈ dimensions[1:end-1]; init=1)
        nelement_local = dimensions[end].nelement ÷ dimensions[end].nrank
        block_sizes, off_diagonals = get_block_sizes(nelement_local, dimensions[end].ngrid,
                                                     outer_dim_steps)
        A = sparse(local_i, local_j, data)

        t1 = time_ns()
        Alu = mpi_static_condensation(dimensions; level_multiplier, sparse_C_blocks, comm,
                                      distributed_comm, shared_comm, allocate_shared_float,
                                      allocate_shared_int, schur_tile_size=nothing,
                                      use_sparse=true, separate_Ainv_B=false, timer,
                                      check_lu=false)
        t2 = time_ns()
    else
        # Handle a block-structured matrix that represents coupling between several
        # 'variables'.
        A = matrix
        n_blocks = length(matrix)
        Bmat = matrix[1][2:end]

        Cmat = Tuple(matrix[i][1] for i ∈ 2:n_blocks)
        #Dmat = Matrix(mortar(reshape([matrix[(i-1)%(n_blocks-1)+1][(i-1)÷(n_blocks-1)+1] for i ∈ 1:(n_blocks-1)^2], n_blocks - 1, n_blocks - 1)))

        t1 = time_ns()
        top_vector_indices = get_flat_global_indices(dimensions)
        bottom_vector_indices =
            get_flat_global_indices([dimensions[end:end] for _ ∈ 2:n_blocks])
        bottom_dimensions = dimensions[end:end]
        outer_A = mpi_static_condensation(dimensions; level_multiplier, sparse_C_blocks,
                                          comm, distributed_comm, shared_comm,
                                          allocate_shared_float, allocate_shared_int,
                                          schur_tile_size=nothing, use_sparse=true,
                                          separate_Ainv_B=false, timer, check_lu=false)
        outer_B = OuterBSubmatrix(outer_A, dimensions, bottom_dimensions, shared_comm,
                                  allocate_shared_float, allocate_shared_int, Bmat)
        outer_C = OuterCSubmatrix(outer_A, bottom_dimensions, dimensions, shared_comm,
                                  allocate_shared_float, allocate_shared_int, Cmat)

        Alu = mpi_schur_complement(outer_A, nothing, nothing, nothing, top_vector_indices,
                                   bottom_vector_indices; comm, shared_comm,
                                   distributed_comm, allocate_shared_float,
                                   allocate_shared_int, use_sparse=true,
                                   sparse_Ainv_B=true, Ainv_dot_B_buffer=outer_B,
                                   C_buffer=outer_C, skip_factorization=true,
                                   check_lu=false)
        t2 = time_ns()
    end
    t_setup = (t2 - t1) * 1e-6 # in ms

    # The mpi_static_condensation() constructor is not type stable, as the solver type
    # depends on the number of levels and on the options chosen. Therefore the main
    # performance test must be in a separate inner function, that can be compiled knowing
    # the concrete type of Alu.

    return t_setup, run_MSC_inner(Alu, A, x, rhs, matrix_repeats, rhs_repeats)...
end

# MPISchurComplements does not provide this function, because MPISchurComplements requires
# A to be split up into blocks.
import LinearAlgebra: lu!
function lu!(Alu::MPISchurComplement, A)
    n_blocks = length(A)
    new_A = A[1][1]
    new_B = mortar(reshape(collect(A[1][2:end]), 1, length(A[1]) - 1))
    new_C = mortar(reshape([A[i][1] for i ∈ 2:n_blocks], length(A) - 1, 1))
    new_D = Matrix(mortar(reshape([A[(i-1)%(n_blocks-1)+2][(i-1)÷(n_blocks-1)+2] for i ∈ 1:(n_blocks-1)^2], n_blocks - 1, n_blocks - 1)))
    return update_schur_complement!(Alu, new_A, new_B, new_C, new_D)
end

function run_MSC_inner(Alu, A, x, rhs, matrix_repeats, rhs_repeats)
    t_lu = Inf
    t_solve = Inf
    if isa(Alu, MPISchurComplement)
        ntop = length(Alu.owned_top_vector_entries)
        nbottom = length(Alu.owned_bottom_vector_entries)
        top_range = 1:ntop
        bottom_range = ntop+1:ntop+nbottom
    end
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
            if isa(Alu, MPISchurComplement)
                @views ldiv!(x[top_range], x[bottom_range], Alu, rhs[top_range],
                             rhs[bottom_range])
            else
                ldiv!(x, Alu, rhs)
            end
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
    benchmark(run_MSC, params_3d, seed_3d, "MPIStaticCondensations_3d")

    return nothing
end
