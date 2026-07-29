module MumpsExt

import LinearAlgebra: lu!, ldiv!
using MPI
using MPIStaticCondensations: Dimension, MPIStaticCondensation, SharedSparseBuffer,
                              @sc_timeit
import MPIStaticCondensations: get_mumps_solver, finalize_mpi_static_condensation!
using MUMPS
using MUMPS: set_job!, invoke_mumps!, finalize!
using TimerOutputs

function get_mumps_solver(dimensions::Vector{<:Dimension},
                          matrix_buffer::NTuple{Nvar,NTuple{Nvar,SharedSparseBuffer{Tf,Ti}}},
                          comm, synchronize_shared::Fs,
                          timer) where {Nvar,Tf,Ti,Fs}
    return MPIStaticCondensationMUMPS(matrix_buffer, comm, synchronize_shared, timer)
end

# Convert an MPI.Comm to a Fortran communicator.
# From https://discourse.julialang.org/t/can-i-pass-mpi-jl-communicator-to-the-fortran-side/76547/2
@static if MPI.MPI_Comm === Cint
    # some MPI libraries don't define MPI_Comm_c2f, they use a Fortran-like integer
    # communicator even in the C api
    comm2f(comm::MPI.Comm) = comm.val
else
    comm2f(comm::MPI.Comm) =
        ccall((:MPI_Comm_c2f, MPI.libmpi), Cint, (MPI.MPI_Comm,), comm)
end

struct MPIStaticCondensationMUMPS{Tf<:AbstractFloat,Ti<:Integer,Tmumps<:Mumps{Tf},Tsync,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    n::Ti
    mumps::Tmumps
    is_root::Bool
    copy_range::UnitRange{Ti}
    global_i::Vector{Cint}
    global_j::Vector{Cint}
    synchronize_shared::Tsync
    timer::Ttimer

    function MPIStaticCondensationMUMPS(matrix_buffer::NTuple{Nvar,NTuple{Nvar,SharedSparseBuffer{Tf,Ti}}},
                                        comm, synchronize_shared::Fs,
                                        timer) where {Nvar,Tf,Ti,Fs}
        if Nvar > 1
            error("MPIStaticCondensationMUMPS does not yet support solves with more than one variable")
        else
            matrix_buffer = matrix_buffer[1][1]
        end

        comm_rank = MPI.Comm_rank(comm)
        comm_size = MPI.Comm_size(comm)
        is_root = (comm_rank == 0)

        colptr = matrix_buffer.colptr
        rowval_list = matrix_buffer.rowval_list
        nzval = matrix_buffer.nzval

        # The matrix is stored in shared memory. For passing to MUMPS, select a subset of
        # columns to be 'locally owned' - not sure whether this is more or less efficient
        # than for example passing the whole matrix on the root process.
        ncol = size(matrix_buffer, 2)
        cols_per_proc = (ncol + comm_size - 1) ÷ comm_size
        local_col_range = min(comm_rank*cols_per_proc+1,ncol+1):min((comm_rank+1)*cols_per_proc,ncol)
        local_flat_range = colptr[first(local_col_range)]:colptr[last(local_col_range)+1]-1

        # The row/column indices need to be 32-bit integers for MUMPS.
        # Note we store global_i and global_j in the MPIStaticCondensationMUMPS struct to
        # ensure that the memory backing these arrays (that we pass a pointer to to MUMPS)
        # is not garbage-collected.
        global_i = vcat((Cint.(rv) for rv ∈ @view(rowval_list[local_col_range]))...)
        global_j = vcat((fill(Cint(j), colptr[j+1] - colptr[j]) for j ∈ local_col_range)...)
        local_nzval = @view nzval[local_flat_range]

        t1 = time_ns()
        icntl = copy(default_icntl)
        icntl[4] = 1 # Non-verbose, only error messages.
        icntl[6] = 1 # A pivoting strategy based only on the pattern of non-zeros - does not require values of matrix entries - so analysis can be done once, and different matrices (with the same non-zero pattern) can be factorised without re-doing analysis.
        icntl[14] = 100 # Percentage increase in the estimated working space (default is between 25 and 35).
        icntl[18] = 3 # User-provided distributed matrix pattern.
        #icntl[20] = 11 # Distributed RHS (also 10, not sure which value is best)
        icntl[20] = 0 # Centralised RHS.
        #icntl[21] = 1 # Solution is kept distributed.
        icntl[21] = 0 # Solution is gathered centrally.
        icntl[4] = 1 # Use 'tree parallelism' when multi-threaded.
        cntl = copy(default_cntl64)
        mumps = Mumps{Float64}(0, icntl, cntl; comm=comm2f(comm))
        mumps.n = ncol

        # Pass matrix storage to MUMPS (does not need to be initialised yet).
        n = length(local_nzval)
        mumps.nnz_loc = n
        mumps.irn_loc = pointer(global_i)
        mumps.jcn_loc = pointer(global_j)
        mumps.a_loc = pointer(local_nzval)

        if is_root
            # Set size of rhs/solution vector.
            mumps.lrhs = ncol
        end

        # Perform analysis phase without using matrix values.
        set_job!(mumps, 1)
        invoke_mumps!(mumps)

        return new{Tf,Ti,typeof(mumps),Fs,typeof(timer)}(
                   n, mumps, is_root, local_col_range, global_i, global_j,
                   synchronize_shared, timer)
    end
end
Base.size(Alu::MPIStaticCondensationMUMPS) = (Alu.n, Alu.n)
Base.size(Alu::MPIStaticCondensationMUMPS, d::Integer) = size(Alu)[d]

function lu!(solver::MPIStaticCondensationMUMPS, A)
    @sc_timeit solver.timer "MUMPS lu! $(size(solver))" begin
        # `A` is the same as the `matrix_buffer` that was used to initialise `solver`, so
        # we do not need to pass/copy anything here.
        mumps = solver.mumps
        set_job!(mumps, 2)
        invoke_mumps!(mumps)
    end
    return nothing
end

function ldiv!(solver::MPIStaticCondensationMUMPS{T}, U::AbstractVector{T}) where T
    @sc_timeit solver.timer "MUMPS ldiv!(Alu,U) $(size(solver, 1))" begin
        mumps = solver.mumps
        if solver.is_root
            mumps.rhs = pointer(U)
        end
        set_job!(mumps, 3)
        invoke_mumps!(mumps)
    end
    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationMUMPS{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "MUMPS ldiv!(X,Alu,U) $(size(solver, 1))" begin
        # MUMPS solves in-place, so copy U into X.
        copy_range = solver.copy_range
        @views X[copy_range] .= U[copy_range]
        solver.synchronize_shared()
        return ldiv!(solver, X)
    end
end

function finalize_mpi_static_condensation!(solver::MPIStaticCondensationMUMPS)
    finalize!(solver.mumps)
    return nothing
end

end
