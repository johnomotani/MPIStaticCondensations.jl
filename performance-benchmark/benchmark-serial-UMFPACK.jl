using LinearAlgebra
using MPI
using SparseArrays
using StableRNGs
using StatsBase

include("common.jl")

function run_UMFPACK(x, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
                     dimensions, comm, distributed_comm, shared_comm,
                     allocate_shared_float, allocate_shared_int, nmat, nrhs, timer)
    if MPI.Comm_size(comm) > 1
        error("UMFPACK can only run in serial")
    end

    A = sparse(global_i, global_j, data)

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

println("UMFPACK benchmark")
println("=================\n")

benchmark(run_UMFPACK, params_1d, seed_1d, "UMFPACK_1d")
benchmark(run_UMFPACK, params_2d, seed_2d, "UMFPACK_2d")
benchmark(run_UMFPACK, params_3d, seed_3d, "UMFPACK_3d")
