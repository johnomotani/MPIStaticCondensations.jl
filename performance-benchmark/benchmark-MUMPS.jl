using LinearAlgebra
using MPI
using StableRNGs
using StatsBase

using MUMPS
using MUMPS: set_job!, invoke_mumps!, finalize!

include("common.jl")

function set_matrix!(mumps, data, global_i, global_j)
    n = length(data)
    mumps.nnz_loc = n
    mumps.irn_loc = pointer(global_i)
    mumps.jcn_loc = pointer(global_j)
    mumps.a_loc = pointer(data)
    return nothing
end

function set_rhs_solution!(mumps, x, isol, rhs, irhs)
    mumps.sol_loc = pointer(x)
    mumps.isol_loc = pointer(isol)
    mumps.rhs_loc = pointer(rhs)
    mumps.irhs_loc = pointer(irhs)
    mumps.nloc_rhs = length(rhs)
    mumps.lrhs_loc = length(rhs)
    return nothing
end

function set_global_rhs!(mumps, rhs_global)
    mumps.rhs = pointer(rhs_global)
    mumps.lrhs = length(rhs_global)
    return nothing
end

function run_MUMPS(x, data, global_i, global_j, local_i, local_j, rhs, rhs_global,
                   dimensions, comm, distributed_comm, shared_comm, allocate_shared_float,
                   allocate_shared_int, nmat, nrhs, timer)

    total_size = prod(d.n for d ∈ dimensions)
    is_root = (MPI.Comm_rank(comm) == 0)

    # The row/column indices need to be 32-bit integers for MUMPS.
    global_i = Cint.(global_i)
    global_j = Cint.(global_j)

    # The locally-owned vector entries should be given by the min/max of the matrix
    # indices in global_i or global_j.
    indrange = extrema(global_i)
    irhs = collect(indrange[1]:indrange[2])
    #isol = similar(irhs)

    t1 = time_ns()
    icntl = copy(default_icntl)
    icntl[4] = 1 # Non-verbose, only error messages.
    icntl[18] = 3 # User-provided distributed matrix pattern.
    #icntl[20] = 11 # Distributed RHS (also 10, not sure which value is best)
    icntl[20] = 0 # Centralised RHS.
    #icntl[21] = 1 # Solution is kept distributed.
    icntl[21] = 0 # Solution is gathered centrally.
    icntl[4] = 1 # Use 'tree parallelism' when multi-threaded.
    cntl = copy(default_cntl64)
    Alu = Mumps{Float64}(0, icntl, cntl)
    Alu.n = total_size
    set_matrix!(Alu, data, global_i, global_j)
    t2 = time_ns()
    t_setup = (t2 - t1) * 1e-6 # in ms

    t_lu = Inf
    t_solve = Inf
    for _ ∈ 1:matrix_repeats
        t1 = time_ns()
        set_job!(Alu, 4)
        invoke_mumps!(Alu)
        t2 = time_ns()
        t_lu = min(t_lu, (t2 - t1) * 1e-6)
        if Alu.info[1] != 0
            error("some MUMPS error occured: $(Alu.info[1:2])")
        end

        nsol_loc = Alu.info[23]
        isol_loc = zeros(Cint, nsol_loc)
        x_loc = fill(NaN, nsol_loc)
        Alu.lsol_loc = nsol_loc

        for _ ∈ 1:rhs_repeats
            t1 = time_ns()
            #set_rhs_solution!(Alu, x_loc, isol_loc, rhs, irhs)
            if is_root
                set_global_rhs!(Alu, rhs_global)
            end
            set_job!(Alu, 3)
            invoke_mumps!(Alu)
            t2 = time_ns()
            t_solve = min(t_solve, (t2 - t1) * 1e-6)
            if Alu.info[1] != 0
                error("some MUMPS error occured: $(Alu.info[1:2])")
            end
        end
    end
    if Alu.info[1] != 0
        # This conditional should never be entered, but seems to prevent global_i and
        # global_j from being garbage collected (which would cause errors in MUMPS).
        println("global_i=$(extrema(global_i)), global_j=$(extrema(global_j))")
    end

    finalize!(Alu)

    return t_setup, t_lu, t_solve
end

BLAS.set_num_threads(Threads.nthreads())

if !MPI.Initialized()
    MPI.Init()
end
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("MUMPS benchmark")
    println("================\n")
end

benchmark(run_MUMPS, params_1d, seed_1d, "MUMPS_1d"; use_shared=false)
benchmark(run_MUMPS, params_2d, seed_2d, "MUMPS_2d"; use_shared=false)
#benchmark(run_MUMPS, params_3d, seed_3d, "MUMPS_3d"; use_shared=false)
