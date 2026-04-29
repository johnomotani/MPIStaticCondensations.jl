"""
Does a direct solve for matrix systems where the right-hand-side and solution vectors can
be decomposed into locally-coupled blocks and joining elements, so that any element in a
'locally-coupled block' is not coupled (by a non-zero matrix entry) to any other
'locally-coupled block' except its own, but may be coupled to any of the 'joining elements'.

Matrices of this type often result from finite element discretizations, where the degrees
of freedom within the volume of an element (or contiguous group of elements) are coupled
to themselves, but only couple to another element via the degrees of freedom on the
surface shared by both elements. The 'locally coupled blocks' are then the interiors of
the elements, and the 'joining elements' are those on the surfaces of elements.

Using an algorithm suggested by the MFEM documentation
(https://docs.mfem.org/html/classmfem_1_1StaticCondensation.html), write the full matrix
system as
```math
\\begin{align}
A\\cdot X &= U
```
By reordering the entries of X and B so that the 'local blocks' are the first entries,
with each local block being a continuous chunk, followed by the 'joining elements', the
matrix system can be rewritten as
```math
\\begin{align}
\\left(\\begin{array}{cc}
a & b\\\\
c & d\\\\
\\end{array}\\right)\\cdot\\left(\\begin{array}{c}
x\\\\
y\\\\
\\end{array}\\right)=\\left(\\begin{array}{c}
u\\\\
v\\\\
\\end{array}\\right)
\\end{align}
```
In this form, \$a\$ is block-diagonal so \$a\\cdot x = u\$ can be solved efficiently, and
parallelised. The remaining part of the solution is found by forming the Schur complement
of \$a\$, doing a matrix-solve using that, and back-substituting, as follows.
```math
\\begin{align}
& a\\cdot x + b \\cdot y = u \\\\
& x = A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y \\\\
& c\\cdot x + d\\cdot y = v \\\\
& c\\cdot (A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y) + d\\cdot y = v \\\\
& (d - c\\cdot A^{-1} \\cdot b) \\cdot y = v - c\\cdot A^{-1}\\cdot u \\\\
& s\\cdot y = v - A^{-1}\\cdot u \\\\
\\end{align}
```
where \$s = (d - c\\cdot A^{-1} \\cdot b)\$ is the 'Schur complement' of \$a\$. Once \$y\$
is known, we can substitute back into the expression above for \$x\$
```math
\\begin{align}
& x = A^{-1}\\cdot u - A^{-1} \\cdot b \\cdot y \\\\
\\end{align}
```

The solve is implemented by
[MPISchurComplements.jl](https://github.com/johnomotani/MPISchurComplements.jl). This
package handles splitting up the matrix into blocks, and assigning MPI communicators to
solve each block. To minimise the size of each Schur complement matrix, the decomposition
is done recursively. The total set of processes is divided into groups, where succesive
divisions are by successive prime factors of the total number of processes. At each stage
the matrix is divided into as many 'local blocks' as there are processes, until at the
final level each 'local block' is solved in serial.
"""
module MPIStaticCondensations

using LinearAlgebra
using MPI
using MPISchurComplements
using Primes
using SparseArrays
using TimerOutputs

import LinearAlgebra: lu!, ldiv!

abstract type MPIStaticCondensation end

struct MPIStaticCondensationSerialSparse{Tf,Ti,Ttimer} <: MPIStaticCondensation
    local_block_solver::SparseArrays.UMFPACK.UmfpackLU{Tf,Ti}
    timer::Ttimer
    check_lu::Bool
end

struct MPIStaticCondensationSerialDense{Tf,Ti,Ttimer} <: MPIStaticCondensation
    local_block_solver::LU{Tf,Matrix{Tf},Vector{Ti}}
    timer::Ttimer
    check_lu::Bool
end

struct MPIStaticCondensationParallel{Tsolver<:MPISchurComplement,Ttimer} <: MPIStaticCondensation
    local_block_solver::Tsolver
    timer::Ttimer
end

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolver{Tsolver<:MPIStaticCondensation}
    local_block_solver::Tsolver
    local_block_indices::Vector{Int64}
end

macro sc_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

"""

`comm` is divided into equally sized shared-memory blocks. `shared_comm` represents the
shared-memory block that this process belongs to - it must be a subset of `comm`, and its
members must be able to create shared-memory arrays.
"""
function mpi_static_condensation(comm::MPI.Comm, shared_comm::MPI.Comm; use_sparse=true,
                                 timer::Union{Nothing,TimerOutput}=nothing,
                                 schur_tile_size=nothing, check_lu=false)

    data_type = Float64

    comm_size = MPI.Comm_size(comm)
    shared_comm_size = MPI.Comm_size(shared_comm)

    if comm_size % shared_comm_size != 0
        error("Size of shared_comm ($shared_comm_size) does not divide the size of comm "
              * "($comm_size).")
    end
    n_blocks = comm_size ÷ shared_comm_size

    n_blocks_factors = factor(Vector, n_blocks)
    shared_comm_size_factors = factor(Vector, shared_comm_size_factors)

    # Create lowest level solver
    this_n = 42
    identity = Matrix{Float64}(undef, this_n, this_n)
    identity .= I
    if use_sparse
        lowest_level_solver =
            MPIStaticCondensationSerialSparse(lu(sparse(identity); check=check_lu), timer,
                                              check_lu)
    else
        lowest_level_solver =
            MPIStaticCondensationSerialDense(lu(identity; check=check_lu), timer,
                                             check_lu)
    end

    this_level_solver = lowest_level_solver
    for group_size ∈ reverse(shared_comm_size_factors)
        A_block_solver = BlockDiagonalSolver(this_level_solver, local_top_vector_entries)

        # Use a parallelized dense-matrix LU solver for the Schur complement solve as long
        # as the Schur complement matrix is not too small.
        level_parallel_schur = length(level_bottom_vector_entries) ≥ 1024

        this_level_sc =
            mpi_schur_complement(A_block_solver, data_type, data_type, data_type,
                                 level_top_vector_entries, level_bottom_vector_entries;
                                 comm=level_comm, shared_comm=level_shared_comm,
                                 distributed_comm=level_distributed_comm,
                                 allocate_shared_float=level_allocate_shared_float,
                                 allocate_shared_int=level_allocate_shared_int,
                                 synchronize_shared=synchronize_shared,
                                 use_sparse=use_sparse, separate_Ainv_B=false,
                                 parallel_schur=level_parallel_schur,
                                 skip_factorization=false,
                                 schur_tile_size=schur_tile_size, check_lu=check_lu,
                                 timer=timer)
        this_level_solver = MPIStaticCondensationParallel(this_level_sc, timer)
    end
end

function lu!(block_diagonal_solver::BlockDiagonalSolver, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    inds = block_diagonal_solver.local_block_indices
    lu!(solver, @view(A[inds,inds]))
    return nothing
end

function ldiv!(block_diagonal_solver::BlockDiagonalSolver, v::AbstractVector)
    solver = block_diagonal_solver.local_block_solver
    inds = block_diagonal_solver.local_block_indices
    ldiv!(solver, @view(v[inds]))
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialSparse, A::AbstractMatrix)
    # For simplicity assume non-zero pattern might change, so pass reuse_symbolic=false.
    lu!(solver.local_block_solver, sparse(A); reuse_symbolic=false, check=solver.check_lu)
    return nothing
end

function ldiv!(solver::MPIStaticCondensationSerialSparse, v::AbstractVector)
    ldiv!(solver.local_block_solver, v)
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialDense, A::AbstractMatrix)
    # Re-use the arrays to avoid allocating.
    mat_storage = solver.local_block_solver.factors
    ipiv = solver.local_block_solver.ipiv
    check = solver.check
    mat_storage .= A
    LAPACK.getrf!(mat_storage, ipiv; check=check)
    return nothing
end

function ldiv!(solver::MPIStaticCondensationSerialDense, v::AbstractVector)
    ldiv!(solver.local_block_solver, v)
    return nothing
end

end
