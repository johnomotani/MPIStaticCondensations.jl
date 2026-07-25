using LinearAlgebra
using MPI
using SparseArrays
using StableRNGs
using StatsBase

include("common.jl")

function run_UMFPACK(x, matrix_data, rhs, rhs_global, dimensions, variable_dimensions,
                     sparse_C_blocks, mumps_fill_in_threshold, block_sizes_heuristic,
                     comm, distributed_comm, shared_comm, allocate_shared_float,
                     allocate_shared_int, nmat, nrhs, matrix_repeats, rhs_repeats, timer)
    if MPI.Comm_size(comm) > 1
        error("UMFPACK can only run in serial")
    end

    global_data, global_i, global_j, data, this_block_global_i, this_block_global_j,
        local_i, local_j = matrix_data
    A = sparse(global_i, global_j, global_data)

    t1 = time_ns()
    Alu = lu(A)
    t2 = time_ns()
    t_setup = (t2 - t1) * 1e-6 # in ms

    t_lu = Inf
    t_solve = Inf
    for _ ∈ 1:matrix_repeats
        t1 = time_ns()
        lu!(Alu, A; reuse_symbolic=false)
        t2 = time_ns()
        t_lu = min(t_lu, (t2 - t1) * 1e-6)

        for _ ∈ 1:rhs_repeats
            t1 = time_ns()
            ldiv!(x, Alu, rhs)
            t2 = time_ns()
            t_solve = min(t_solve, (t2 - t1) * 1e-6)
        end
    end

    return t_setup, t_lu, t_solve
end

BLAS.set_num_threads(1)
