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

export mpi_static_condensation, create_dimension

using LinearAlgebra
using LinearAlgebra.LAPACK: getrf!
using MPI
using MPIDenseLUs
using MPISchurComplements
using MPISchurComplements: MPISchurComplementAFactorization,
                           MPISchurComplementBlockAinvDotB, MPISchurComplementBlockC
import MPISchurComplements: ldiv_Bmatrix!, copy_B_submatrix!, Ainv_dot_B_dot_y!,
                            copy_C_submatrix!, mul_C_Ainv_dot_B!, mul_C_dot_Ainv_dot_u!
using Primes
using SparseArrays
using SparseArrays: FixedSparseCSC, AbstractSparseMatrixCSC
using SparseArrays.UMFPACK: UmfpackLU
using TimerOutputs

import LinearAlgebra: lu!, ldiv!

macro sc_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

const AbstractVectorOrMatrix{T} = Union{AbstractVector{T},AbstractMatrix{T}}

abstract type MPIStaticCondensation{Tf<:AbstractFloat} <: Factorization{Tf} end

struct MPIStaticCondensationNull{Tf<:AbstractFloat} <: MPIStaticCondensation{Tf} end

struct MPIStaticCondensationParallel{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:MPISchurComplement{Tf},Tranget,Trangeatab,Trangeabs,Trangeb,Trangebs,Tsync,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_top_vector_indices::Tranget
    all_local_top_vector_a_block_indices::Trangeatab
    partial_local_top_vector_a_block_indices::Trangeatab
    partial_a_block_sub_selection_indices::Trangeabs
    local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_vector_no_overlap_indices::Trangeb
    this_shared_local_bottom_sub_selection_indices::Trangebs
    this_shared_local_bottom_sub_selection_no_overlap_indices::Trangeb
    this_shared_local_bottom_vector_repeat_indices::Trangeb
    this_shared_local_bottom_periodic_pairs::Matrix{Ti}
    u_buffer::Vector{Tf}
    v_buffer::Vector{Tf}
    has_periodic::Bool
    synchronize_shared::Tsync
    timer::Ttimer
end
Base.size(Alu::MPIStaticCondensationParallel) = (Alu.n, Alu.n)
Base.size(Alu::MPIStaticCondensationParallel, d::Integer) = size(Alu)[d]

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolverSerial{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},Nothing},Trange,Tsparse} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Vector{Tsolver}
    block_indices::Trange
    lu_selection_indices::Trange
    sparse_buffers::Vector{Tsparse}
    x_buffer::Vector{Tf}
    u_buffer::Vector{Tf}
    B_column_indices::Trange
    B_buffers_out::Vector{Matrix{Tf}}
    B_buffers_in::Vector{Matrix{Tf}}
    check_lu::Bool
    function BlockDiagonalSolverSerial{Tf}(n::Ti, block_indices, lu_selection_indices,
                                           B_column_indices, use_sparse, timer,
                                           check_lu) where {Tf, Ti <: Integer}
        # Don't need a solver for any empty entries in block_indices, as these blocks have
        # no interior points.
        block_indices = [bi for bi ∈ block_indices if !isempty(bi)]
        lu_selection_indices = [li for li ∈ lu_selection_indices if !isempty(li)]
        B_column_indices = [Bc for (Bc, bi) ∈ zip(B_column_indices, block_indices)
                            if !isempty(bi)]
        block_sizes = [length(bi) for bi ∈ block_indices]
        block_size = maximum(block_sizes; init=0)
        function get_identity(bs)
            if use_sparse
                identity = spzeros(Tf, bs, bs)
            else
                identity = zeros(Tf, bs, bs)
            end
            copyto!(identity, I)
            return identity
        end
        if block_size > 0
            local_block_solver = [lu(get_identity(length(bi))) for bi ∈ block_indices]
            if use_sparse
                sparse_buffers = [spzeros(Tf, bs, bs) for bs ∈ block_sizes]
            else
                sparse_buffers = [nothing for _ ∈ block_indices]
            end
        else
            local_block_solver = [nothing]
            sparse_buffers = [nothing]
        end
        x_buffer = fill(NaN, block_size)
        u_buffer = fill(NaN, block_size)
        B_buffers_out = [zeros(length(bi), length(Bc))
                         for (bi, Bc) ∈ zip(block_indices, B_column_indices)]
        if use_sparse
            B_buffers_in = deepcopy(B_buffers_out)
        else
            B_buffers_in = Matrix{Tf}[]
        end
        return new{Tf,Ti,eltype(local_block_solver),typeof(block_indices),eltype(sparse_buffers)}(
                   n, local_block_solver, block_indices, lu_selection_indices,
                   sparse_buffers, x_buffer, u_buffer, B_column_indices, B_buffers_out,
                   B_buffers_in, check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverSerial) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverSerial, d::Integer) = size(Alu)[d]

# When this solver is used there are more processes than blocks, so we use multiple
# processes to solve each block, with shared-memory parallelism.
struct BlockDiagonalSolverShared{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},MPIDenseLU{Tf},Nothing},Tserialsolver<:Union{Factorization{Tf},Nothing},Tm,Trange,Tsync} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_block_serial_solver::Tserialsolver
    factors::Tm
    block_indices::Trange
    lu_selection_indices::Trange
    partial_lu_selection_indices::Trange
    partial_col_range::UnitRange{Ti}
    x_buffer::Vector{Tf}
    u_buffer::Vector{Tf}
    B_column_indices::Trange
    block_comm_rank::Ti
    synchronize_shared::Tsync
    check_lu::Bool
    function BlockDiagonalSolverShared{Tf}(n::Ti, block_indices, lu_selection_indices,
                                           B_column_indices, block_comm,
                                           allocate_shared_float, allocate_shared_int,
                                           synchronize_shared::F, timer,
                                           check_lu) where {Tf, Ti <: Integer, F}
        block_size = length(block_indices)
        block_comm_rank = MPI.Comm_rank(block_comm)
        block_comm_size = MPI.Comm_size(block_comm)

        if block_size == 0
            local_block_solver = nothing
            local_block_serial_solver = nothing
            factors = nothing
            x_buffer = fill(NaN, block_size)
            u_buffer = fill(NaN, block_size)
        elseif block_comm_size > 1 && block_size > 1024
            # Have multiple processes working on this block, and the block size is
            # big enough to be worth using a parallel dense-matrix LU solver.
            factors = allocate_shared_float(block_size, block_size)
            if MPI.Comm_rank(block_comm) == 0
                copyto!(factors, I)
            end

            local_block_solver =
                mpi_dense_lu(factors, 128, block_comm, block_comm, MPI.COMM_SELF,
                             allocate_shared_float, allocate_shared_int;
                             synchronize_shared=synchronize_shared,
                             skip_factorization=true, check_lu=check_lu, timer=timer)
            local_block_serial_solver = LU(factors,
                                           local_block_solver.factorization_shared_lu.ipiv,
                                           block_size)
            x_buffer = allocate_shared_float(block_size)
            u_buffer = allocate_shared_float(block_size)
            if block_comm_rank == 0
                x_buffer .= NaN
                u_buffer .= NaN
            end
        else
            factors = allocate_shared_float(block_size, block_size)
            ipiv = allocate_shared_int(block_size)
            if MPI.Comm_rank(block_comm) == 0
                copyto!(factors, I)
                getrf!(factors, ipiv; check=check_lu)
                local_block_solver = LU(factors, ipiv, block_size)
                local_block_serial_solver = local_block_solver
            else
                local_block_solver = nothing
                local_block_serial_solver = LU(factors, ipiv, block_size)
            end
            x_buffer = fill(NaN, block_size)
            u_buffer = fill(NaN, block_size)
        end

        cols_per_proc = (block_size + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,block_size)
        partial_lu_selection_indices = lu_selection_indices[partial_col_range]

        return new{Tf,Ti,typeof(local_block_solver),typeof(local_block_serial_solver),typeof(factors),typeof(block_indices),F}(
                   n, local_block_solver, local_block_serial_solver, factors,
                   block_indices, lu_selection_indices, partial_lu_selection_indices,
                   partial_col_range, x_buffer, u_buffer, B_column_indices,
                   block_comm_rank, synchronize_shared, check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverShared) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverShared, d::Integer) = size(Alu)[d]

struct BlockAinvDotBSerial{Tf,Ti} <: MPISchurComplementBlockAinvDotB
    blocks::Vector{Matrix{Tf}}
    block_rowinds::Vector{Vector{Ti}}
    block_colinds::Vector{Vector{Ti}}
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}

    function BlockAinvDotBSerial{Tf}(block_rowinds::Vector{Vector{Ti}},
                                     block_colinds::Vector{Vector{Ti}}) where {Tf,Ti}
        non_empty_blocks = [!isempty(ri) && !isempty(ci)
                            for (ri, ci) ∈ zip(block_rowinds, block_colinds)]
        block_rowinds = block_rowinds[non_empty_blocks]
        block_colinds = block_colinds[non_empty_blocks]
        blocks = Matrix{Tf}[]
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out = Vector{Tf}[]
        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = length(ri)
            ncol = length(ci)
            push!(blocks, zeros(Tf, nrow, ncol))
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
        end
        return new{Tf,Ti}(blocks, block_rowinds, block_colinds, vector_buffer_blocks_in,
                          vector_buffer_blocks_out)
    end
end

# This version has a single block, and operations are parallelised using shared-memory
# MPI.
struct BlockAinvDotBShared{Tf,Ti,Tm,Tsync} <: MPISchurComplementBlockAinvDotB
    block::Tm
    partial_block::Matrix{Tf}
    block_rowinds::Vector{Ti}
    block_partial_rowinds::Vector{Ti}
    block_colinds::Vector{Ti}
    block_partial_colinds::Vector{Ti}
    buffer::Tm
    partial_col_range::UnitRange{Ti}
    partial_row_range::UnitRange{Ti}
    vector_buffer_block_in::Vector{Tf}
    vector_buffer_block_out::Vector{Tf}
    synchronize_shared::Tsync

    function BlockAinvDotBShared{Tf}(block_rowinds::Vector{Ti}, block_colinds::Vector{Ti},
                                     block_comm_rank::Integer, block_comm_size::Integer,
                                     allocate_shared_float::Fa,
                                     synchronize_shared::Fs) where {Tf,Ti,Fa,Fs}
        if isempty(block_rowinds) || isempty(block_colinds)
            return new{Tf,Ti,Matrix{Tf},Fs}(zeros(Tf, 0, 0), zeros(Tf, 0, 0),
                                            block_rowinds, zeros(Ti, 0), block_colinds,
                                            zeros(Ti, 0), zeros(Tf, 0, 0), 1:0, 1:0,
                                            zeros(Tf, 0), zeros(Tf, 0),
                                            synchronize_shared)
        end

        nrow = length(block_rowinds)
        ncol = length(block_colinds)
        block = allocate_shared_float(length(block_rowinds), length(block_colinds))
        buffer = allocate_shared_float(length(block_rowinds), length(block_colinds))
        cols_per_proc = (ncol + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,ncol)
        block_partial_colinds = block_colinds[partial_col_range]
        rows_per_proc = (nrow + block_comm_size - 1) ÷ block_comm_size
        partial_row_range = block_comm_rank*rows_per_proc+1:min((block_comm_rank+1)*rows_per_proc,nrow)
        partial_nrow = length(partial_row_range)
        block_partial_rowinds = block_rowinds[partial_row_range]
        vector_buffer_block_in = allocate_shared_float(ncol)
        vector_buffer_block_out = zeros(Tf, partial_nrow)
        partial_block = zeros(Tf, partial_nrow, ncol)

        block[:,partial_col_range] .= 0.0
        vector_buffer_block_in[partial_col_range] .= 0.0
        vector_buffer_block_out .= 0.0

        return new{Tf,Ti,typeof(block),Fs}(block, partial_block, block_rowinds,
                                           block_partial_rowinds, block_colinds,
                                           block_partial_colinds, buffer,
                                           partial_col_range, partial_row_range,
                                           vector_buffer_block_in,
                                           vector_buffer_block_out, synchronize_shared)
    end
end

struct BlockCSerial{Tf,Ti,Tib,Tbc,Tir,Fsb<:Function,Fs<:Function} <: MPISchurComplementBlockC
    blocks::Vector{Matrix{Tf}}
    block_rowinds::Vector{Vector{Ti}}
    block_colinds::Vector{Vector{Ti}}
    block_hypercube_positions::Vector{Ti}
    right_multiplication_buffer_blocks::Vector{Matrix{Tf}}
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}
    vector_intermediate_buffer::Tib
    vector_range::UnitRange{Ti}
    vector_init_range::Tir
    matrix_init_range::Tir
    buffer_position::Tbc
    block_synchronize_shared::Fsb
    synchronize_shared::Fs

    function BlockCSerial{Tf}(block_rowinds::Vector{Vector{Ti}},
                              block_colinds::Vector{Vector{Ti}},
                              block_hypercube_positions::Vector{Ti},
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti},
                              vector_init_range::Union{UnitRange{Ti},Nothing},
                              matrix_init_range::Union{UnitRange{Ti},Nothing},
                              buffer_position::Union{Ti,Nothing}, comm_rank::Ti,
                              block_synchronize_shared::Fsb,
                              synchronize_shared::Fs) where {Tf,Ti,Fsb<:Function,Fs<:Function}
        non_empty_blocks = [!isempty(ri) && !isempty(ci)
                            for (ri, ci) ∈ zip(block_rowinds, block_colinds)]
        block_rowinds = block_rowinds[non_empty_blocks]
        block_colinds = block_colinds[non_empty_blocks]
        blocks = Matrix{Tf}[]
        right_multiplication_buffer_blocks = Matrix{Tf}[]
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out = Vector{Tf}[]
        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = length(ri)
            ncol = length(ci)
            push!(blocks, zeros(Tf, nrow, ncol))
            push!(right_multiplication_buffer_blocks, zeros(Tf, nrow, nrow))
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
        end
        return new{Tf,Ti,typeof(vector_intermediate_buffer),typeof(buffer_position),typeof(vector_init_range),Fsb,Fs}(
                   blocks, block_rowinds, block_colinds, block_hypercube_positions,
                   right_multiplication_buffer_blocks, vector_buffer_blocks_in,
                   vector_buffer_blocks_out, vector_intermediate_buffer, vector_range,
                   vector_init_range, matrix_init_range, buffer_position,
                   block_synchronize_shared, synchronize_shared)
    end
end

struct BlockCShared{Tf,Ti,Tbi,Tbuff,Tib,Tir,Fbs<:Function,Fs<:Function} <: MPISchurComplementBlockC
    block::Matrix{Tf}
    block_rowinds::Vector{Ti}
    block_colinds::Vector{Ti}
    partial_block_colinds::Vector{Ti}
    partial_col_range::UnitRange{Ti}
    block_hypercube_position::Ti
    right_multiplication_buffer_block::Matrix{Tf}
    block_right_multiplication_output_colinds::Vector{Ti}
    vector_buffer_block_in::Tbi
    vector_buffer_block_out::Vector{Tf}
    vector_intermediate_buffer_local::Tbuff
    vector_intermediate_buffer::Tib
    vector_range::UnitRange{Ti}
    buffer_column_per_subgroup::Bool
    vector_init_range::Tir
    matrix_init_range::Tir
    block_synchronize_shared::Fbs
    synchronize_shared::Fs

    # When multiplying a vector or a BlockAinvDotBShared matrix by a BlockCShared
    # block-structured C matrix, the output from each block can overlap as the outputs are
    # on the 'boundary points' of the grid, not the decoupled 'interior points'. To deal
    # with this, we first write the results from each block into an intermediate buffer
    # (`vector_intermediate_buffer`, or a buffer provided by MPISchurComplements), which
    # has several columns that collect different contributions to the result (where the
    # blocks written to a single column do not have overlapping results, unless they come
    # from the same process and therefore cannot conflict). The columns are summed to give
    # the final result.
    # To minimise memory bandwidth and computational time, we would like to minimise the
    # number of columns in the intermediate buffer.
    # When the number of processes is less than 2^d, where d is the number of dimensions,
    # we use one column per process.
    # When the number of processes is ≥2^d, we restrict the buffer to 2^d columns by
    # choosing the output column for each block in such a way that the blocks in a single
    # column never overlap. At any level of the solver, the grid is divided into blocks.
    # Blocks that are adjacent in any dimension share a face/edge/corner/etc. and
    # therefore have an overlap in 'C'. We group the blocks into 2x2x...
    # squares/cubes/hypercubes. Use 3d language for simplicity in the rest of this note,
    # but the same argument applies in any number of dimensions. A block in a certain
    # position within a cube cannot overlap with the blocks in the same position in
    # adjacent cubes, because they are fully separated by another block (cannot share even
    # an edge or a corner). Therefore if we put the outputs from all blocks in one
    # position in the cubes in one column, there are no overlaps (and so no conflicts
    # between outputs from different processes). The number of positions in a cube is 2^3
    # (or 2^d in d dimensions), so we need 2^d columns. We also need to keep track for
    # each block of which position it has in its cube, which translates to the column its
    # output should be written to in the intermediate buffer.
    # To find the block's position within its cube, get the block index in un-flattened
    # form. Transforming the index in each dimension to 0 for even values and 1 for odd
    # values, the binary number formed by the string of 0s and 1s (ordered in the same way
    # as the dimensions) is translated back to an integer to give the intermediate buffer
    # column.
    function BlockCShared{Tf}(block_rowinds_full::Vector{Ti},
                              partial_row_range::UnitRange{Ti}, block_colinds::Vector{Ti},
                              block_hypercube_position::Ti,
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti},
                              buffer_column_per_subgroup::Bool,
                              vector_init_range::Union{UnitRange{Ti},Nothing},
                              matrix_init_range::Union{UnitRange{Ti},Nothing},
                              subgroup_i::Ti, block_allocate_shared_float::Fa,
                              block_synchronize_shared::Fbs, block_comm_rank::Integer,
                              block_comm_size::Integer,
                              synchronize_shared::Fs) where {Tf,Ti,Fa<:Function,Fbs<:Function,Fs<:Function}
        block_rowinds = block_rowinds_full[partial_row_range]
        nrow_full = length(block_rowinds_full)
        nrow = length(block_rowinds)
        ncol = length(block_colinds)
        block = zeros(Tf, nrow, ncol)
        cols_per_proc = (ncol + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,ncol)
        partial_block_colinds = block_colinds[partial_col_range]
        right_multiplication_buffer_block = zeros(Tf, nrow, nrow_full)
        vector_buffer_block_in = block_allocate_shared_float(ncol)
        vector_buffer_block_out = zeros(Tf, nrow)
        if subgroup_i < 0
            vector_intermediate_buffer_local = zeros(Tf, 0)
        else
            vector_intermediate_buffer_local = @view vector_intermediate_buffer[block_hypercube_position,:]
        end
        return new{Tf,Ti,typeof(vector_buffer_block_in),typeof(vector_intermediate_buffer_local),typeof(vector_intermediate_buffer),typeof(vector_init_range),Fbs,Fs}(
                   block, block_rowinds, block_colinds, partial_block_colinds,
                   partial_col_range, block_hypercube_position,
                   right_multiplication_buffer_block, block_rowinds_full,
                   vector_buffer_block_in, vector_buffer_block_out,
                   vector_intermediate_buffer_local, vector_intermediate_buffer,
                   vector_range, buffer_column_per_subgroup, vector_init_range,
                   matrix_init_range, block_synchronize_shared, synchronize_shared)
    end
end

function get_C_buffer_init_ranges(vector_n, schur_complement_buffer,
                                  C_buffer_column_per_subgroup, C_buffer_ncopies,
                                  subgroup_i, n_subgroups, subgroup_size, block_comm_rank)
    matrix_n = length(schur_complement_buffer.nzval)

    if C_buffer_column_per_subgroup
        vector_points_per_proc = (vector_n + subgroup_size - 1) ÷ subgroup_size
        C_vector_init_range = block_comm_rank*vector_points_per_proc+1:min((block_comm_rank+1)*vector_points_per_proc,vector_n)
        matrix_points_per_proc = (matrix_n + subgroup_size - 1) ÷ subgroup_size
        C_matrix_init_range = block_comm_rank*matrix_points_per_proc+1:min((block_comm_rank+1)*matrix_points_per_proc,matrix_n)
        if 0 ≤ subgroup_i ≤ C_buffer_ncopies
            C_buffer_column = subgroup_i + 1
        else
            C_buffer_column = nothing
        end
    else
        C_vector_init_range = nothing
        C_matrix_init_range = nothing
        C_buffer_column = nothing
    end

    return C_vector_init_range, C_matrix_init_range, C_buffer_column
end

function get_C_hypercube_position(iblock)
    return sum(((i - 1) % 2) * 2^(d-1) for (d, i) ∈ enumerate(iblock)) + 1
end

struct Dimension{Ti<:Integer}
    n::Ti
    n_local::Ti
    nelement::Ti
    ngrid::Ti
    nrank::Ti
    irank::Ti
    global_inds::Vector{Ti}
    periodic::Bool
    #has_lower_boundary::Bool
    #has_upper_boundary::Bool
    remove_boundaries::Bool

    function Dimension(; nelement::Ti, ngrid::Ti, nrank::Ti, irank::Ti, periodic::Bool,
                       remove_boundaries::Bool) where Ti <: Integer

        if nelement % nrank != 0
            error("`nrank=$nrank` does not divide nelement=$nelement")
        end
        if nelement < 0
            error("nelement=$nelement cannot be negative")
        end
        if ngrid < 0
            error("ngrid=$ngrid cannot be negative")
        end
        if nrank < 1
            error("nrank=$nrank must be positive")
        end
        if irank < 0
            error("irank=$irank cannot be negative")
        end

        nelement_local = nelement ÷ nrank

        # Assume a continuous-Galerkin finite element discretization where adjacent
        # elements share a boundary point. `ngrid` counts the points in a single element,
        # but two of these are shared (except at the ends of the grid).
        if nelement == 0
            n = 0
        else
            n = nelement * (ngrid - 1) + 1
        end
        if nelement_local == 0
            n_local = 0
        else
            n_local = nelement_local * (ngrid - 1) + 1
        end
        first_global_ind = irank * nelement_local * (ngrid - 1) + 1
        last_global_ind = (irank + 1) * nelement_local * (ngrid - 1) + 1

        #if !has_lower_boundary
        #    if nelement > 0
        #        n -= 1
        #    end
        #    if irank == 0
        #        if nelement_local > 0
        #            n_local -= 1
        #        end
        #        first_global_ind += 1
        #    end
        #end
        #if !has_upper_boundary
        #    if nelement > 0
        #        n -= 1
        #    end
        #    if irank == nrank - 1
        #        if nelement_local > 0
        #            n_local -= 1
        #        end
        #        last_global_ind -= 1
        #    end
        #end

        global_inds = collect(first_global_ind:last_global_ind)
        if periodic && irank == nrank - 1
            global_inds[end] = 1
        end

        return new{Ti}(n, n_local, nelement, ngrid, nrank, irank, global_inds, periodic,
                       remove_boundaries)
    end
end

"""
    create_dimension(; nelement::Integer, ngrid::Integer, nrank::Integer,
                     irank::Integer, periodic::Bool, remove_boundaries::Bool=false)

Create a `Dimension` object for input to the `dimensions` argument of
`mpi_static_condensation()`.

Assume a continuous-Galerkin finite element discretization where there are `nelement`
elements and `ngrid` points in each element. The points at the boundary between two
elements are shared by both elements, so that the total number of grid points is
`nelement * (ngrid - 1) + 1`. When `periodic=true`, the grid is periodic and the last grid
point is a copy of the first. When the grid is distributed over different MPI blocks, the
point on the boundary between the blocks is duplicated on both blocks.

The number of shared-memory blocks that this dimension is divided into is given by
`nrank`, and the rank of the block that this process belongs to is `irank`.

`remove_boundaries=true` can be passed if the grid at the boundary in this dimension does
not fit in to the sparsity pattern of the rest of the grid. In this case, the boundary
points can be included in the 'bottom vector' part of the Schur complement split on the
top level of the static-condensation solve, in order to ensure that the 'top vector' part
can be split by removing any element boundary.
"""
function create_dimension(; nelement::Integer, ngrid::Integer, nrank::Integer,
                          irank::Integer, periodic::Bool, remove_boundaries::Bool=false)
    return Dimension(; nelement, ngrid, nrank, irank, periodic, remove_boundaries)
end

function get_global_indices(dimensions::Vector{<:Dimension}, local_inds::Vector{<:Integer})
    n_local_tuple = Tuple(d.n_local for d ∈ dimensions)
    return get_global_indices(dimensions, local_inds, n_local_tuple)
end
function get_global_indices(dimensions::Vector{<:Dimension},
                            local_inds::Vector{<:Integer}, n_local_tuple)
    global_inds = similar(local_inds)
    cartinds = CartesianIndices(n_local_tuple)
    for (i, ind) ∈ enumerate(local_inds)
        cart_i = cartinds[ind]
        global_i = 0
        for (d, di) ∈ zip(reverse(dimensions), reverse(Tuple(cart_i)))
            global_i = global_i * d.n + d.global_inds[di] - 1
        end
        global_i += 1
        global_inds[i] = global_i
    end
    return global_inds
end

function apply_periodicity_to_indices(dimensions::Vector{<:Dimension},
                                      inds::Vector{<:Integer})
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return apply_periodicity_to_indices(dimensions, inds, n_tuple)
end
function apply_periodicity_to_indices(dimensions::Vector{<:Dimension},
                                      inds::Vector{Ti}, n_tuple) where Ti <: Integer
    if !any(d.periodic for d ∈ dimensions)
        # No periodic dimensions to account for.
        return copy(inds)
    end
    periodic_inds = similar(inds)
    periodic_pairs_first = Ti[]
    periodic_pairs_second = Ti[]
    cartinds = CartesianIndices(n_tuple)
    for (i, ind) ∈ enumerate(inds)
        cart_i = cartinds[ind]
        global_i = 0
        global_i_ignore_periodic = 0
        for (d, di, n) ∈ zip(reverse(dimensions), reverse(Tuple(cart_i)), reverse(n_tuple))
            global_i_ignore_periodic = global_i_ignore_periodic * n + di - 1
            if di == n && d.periodic
                di = 1
            end
            global_i = global_i * n + di - 1
        end
        global_i += 1
        global_i_ignore_periodic += 1
        periodic_inds[i] = global_i
        if global_i != global_i_ignore_periodic
            push!(periodic_pairs_second, global_i_ignore_periodic)
            push!(periodic_pairs_first, global_i)
        end
    end
    periodic_pairs = transpose(hcat(periodic_pairs_first, periodic_pairs_second))
    return periodic_inds, periodic_pairs
end

function get_non_repeated_indices_and_repeats(dimensions::Vector{<:Dimension},
                                              inds::Vector{<:Integer})
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return get_non_repeated_indices_and_repeats(dimensions, inds, n_tuple)
end
function get_non_repeated_indices_and_repeats(dimensions::Vector{<:Dimension},
                                              inds::Vector{Ti}, n_tuple) where Ti <: Integer
    if !any(d.periodic for d ∈ dimensions)
        # No periodic dimensions to account for.
        return copy(inds)
    end
    unique_inds = Ti[]
    repeat_inds = Ti[]
    cartinds = CartesianIndices(n_tuple)
    for (i, ind) ∈ enumerate(inds)
        cart_i = cartinds[ind]
        global_i = 0
        global_i_ignore_periodic = 0
        for (d, di, n) ∈ zip(reverse(dimensions), reverse(Tuple(cart_i)), reverse(n_tuple))
            global_i_ignore_periodic = global_i_ignore_periodic * n + di - 1
            if di == n && d.periodic
                di = 1
            end
            global_i = global_i * n + di - 1
        end
        global_i += 1
        global_i_ignore_periodic += 1
        if global_i == global_i_ignore_periodic
            push!(unique_inds, global_i)
        else
            push!(repeat_inds, global_i_ignore_periodic)
        end
    end
    return unique_inds, repeat_inds
end

function get_dim_indices!(dimensions, block_sizes, flat_i)
    block_inds = zeros(Int64, length(dimensions))
    inner_inds = zeros(Int64, length(dimensions))
    for (i, d) ∈ enumerate(dimensions)
        flat_i, dim_i = divrem(flat_i, d.n_local)
        this_block_npoints = block_sizes[i] * (d.ngrid - 1)
        block_inds[i], inner_inds[i] = divrem(dim_i, this_block_npoints) .+ 1
    end
    return block_inds, inner_inds
end
function add_row_inds!(rv, idim, dimensions, block_sizes, nblock_list, row_indices,
                       block_inds, inner_inds, rowind, count, row_count)
    if idim == 0
        # rowind is constructed as a 0-based index for convenience. Convert to
        # 1-based before adding to `rv`.
        rowind += 1
        # Only add row indices that are contained in row_indices. For each column,
        # we iterate through the rows in order so we use row_count to avoid
        # searching row_indices here.
        while row_count[] ≤ length(row_indices) && rowind > row_indices[row_count[]]
            row_count[] += 1
        end
        if row_count[] ≤ length(row_indices) && rowind == row_indices[row_count[]]
            push!(rv, row_count[])
            count[] += 1
            row_count[] += 1
        end
        return nothing
    end
    d = dimensions[idim]
    block_npoints = block_sizes[idim] * (d.ngrid - 1)
    iblock = block_inds[idim]
    iinner = inner_inds[idim]
    rowind *= d.n_local
    if iinner == 1 && iblock > 1
        # Is a block boundary, so include points from previous block.
        row_offset = (iblock - 2) * block_npoints
        for row_inner ∈ 1:block_npoints
            add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list, row_indices,
                          block_inds, inner_inds, rowind + row_offset + row_inner - 1,
                          count, row_count)
        end
    end
    row_offset = (iblock - 1) * block_npoints
    if iblock > nblock_list[idim]
        # Creating entries for the last grid point, this is 'really'
        # iel=nelement_local-1, col_igr=ngrid, so only need to add the row_igr=1
        # point.
        add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list, row_indices,
                      block_inds, inner_inds, rowind + row_offset, count, row_count)
    else
        for row_inner ∈ 1:block_npoints+1
            add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list, row_indices,
                          block_inds, inner_inds, rowind + row_offset + row_inner - 1,
                          count, row_count)
        end
    end
    return nothing
end
function get_shared_sparse_matrix_csc_buffer(dimensions::Vector{<:Dimension},
                                             block_sizes::Vector{<:Integer},
                                             row_indices::Vector{<:Integer},
                                             column_indices::Vector{<:Integer},
                                             shared_comm, allocate_shared_float::F1,
                                             allocate_shared_int::F2) where {F1, F2}
    n_local_list = [d.n_local for d ∈ dimensions]
    nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
    nblock_list = [(nel + bs - 1) ÷ bs for (nel, bs) ∈ zip(nelement_local_list, block_sizes)]
    m = length(row_indices)
    n = length(column_indices)
    if m == 0 || n == 0
        # FixedSparseCSC constructor errors when one of the matrix sizes is zero, and
        # there are no entries anyway so do not need shared-memory allocation.
        return spzeros(m, n)
    end

    shared_comm_rank = MPI.Comm_rank(shared_comm)
    n_colptr = Ref(-1)
    n_rowval = Ref(-1)
    if shared_comm_rank == 0
        cp = Int64[]
        rv = Int64[]
        count = Ref(1)
        row_count = Ref(1)

        for col ∈ column_indices
            push!(cp, count[])
            block_inds, inner_inds = get_dim_indices!(dimensions, block_sizes, col - 1)
            row_count[] = 1
            add_row_inds!(rv, length(dimensions), dimensions, block_sizes, nblock_list,
                          row_indices, block_inds, inner_inds, 0, count, row_count)
        end
        push!(cp, count[])

        n_colptr[] = length(cp)
        n_rowval[] = length(rv)

        MPI.Bcast!(n_colptr, shared_comm; root=0)
        MPI.Bcast!(n_rowval, shared_comm; root=0)

        colptr = allocate_shared_int(n_colptr[])
        rowval = allocate_shared_int(n_rowval[])
        nzval = allocate_shared_float(n_rowval[])

        colptr .= cp
        rowval .= rv
        nzval .= 0.0
    else
        MPI.Bcast!(n_colptr, shared_comm; root=0)
        MPI.Bcast!(n_rowval, shared_comm; root=0)

        colptr = allocate_shared_int(n_colptr[])
        rowval = allocate_shared_int(n_rowval[])
        nzval = allocate_shared_float(n_rowval[])
    end

    MPI.Barrier(shared_comm)

    # Use the 'experimental' FixedSparseCSC instead of SparseMatrixCSC to ensure that the
    # Vectors are not resized, reallocated, etc.
    return FixedSparseCSC(m, n, colptr, rowval, nzval)
end

struct FakeComm
    rank::Int64
    size::Int64
end
MPI.Comm_rank(comm::FakeComm) = comm.rank
MPI.Comm_size(comm::FakeComm) = comm.size
MPI.Comm_split(comm::FakeComm, color, key) = comm
MPI.Allreduce!(buff, op, comm::FakeComm) = buff # This is not a sensible result!
MPI.Bcast!(buff, comm::FakeComm; root=nothing) = buff # This is not a sensible result!
MPI.Barrier(comm::FakeComm) = nothing

@kwdef struct LevelInfo{Ti,Tcomm<:Union{MPI.Comm,FakeComm}}
    #level_dimensions::Vector{Dimension{Ti}}
    has_periodic::Bool
    block_sizes::Vector{Ti}
    global_size::Ti
    global_bottom_vector_size::Ti
    top_vector_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    iblock_list::Matrix{Ti}
    all_local_top_vector_a_block_indices::Vector{Ti}
    local_top_vector_a_block_indices::Vector{Vector{Ti}}
    all_a_block_sub_selection_indices::Vector{Ti}
    a_block_sub_selection_indices::Vector{Vector{Ti}}
    a_block_lu_selection_indices::Vector{Vector{Ti}}
    a_block_B_column_indices::Vector{Vector{Ti}}
    n_subgroups::Ti
    subgroup_i::Ti
    subgroup_size::Ti
    block_comm::Tcomm
    bottom_vector_indices::Vector{Ti}
    local_bottom_vector_indices::Vector{Ti}
    local_bottom_vector_no_overlap_indices::Vector{Ti}
    local_bottom_vector_no_overlap_sub_selection_indices::Vector{Ti}
    local_bottom_vector_repeat_indices::Vector{Ti}
    local_bottom_vector_periodic_pairs::Matrix{Ti}
    #level_comm::Tcomm
    #level_distributed_comm::Tdcomm
    level_shared_comm::Tcomm
end

# Use `FakeComm` values for comm/distributed_comm/shared_comm to skip the comm splitting,
# for testing of the index generation.
function split_matrix(dimensions::Vector{<:Dimension}, level_indices::Vector{Ti},
                      block_sizes::Vector{Ti}, global_size::Ti, is_top_level::Bool,
                      is_bottom_level::Bool,
                      distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
                      shared_comm::Union{MPI.Comm,FakeComm}) where Ti <: Integer
    if length(dimensions) != length(block_sizes)
        error("dimensions and block_sizes should be the same length")
    end

    has_periodic = any(d.periodic for d ∈ dimensions)

    if shared_comm == MPI.COMM_NULL
        # This processor does no work on this level, so just fill level_info with dummy
        # values.
        return LevelInfo(; has_periodic, block_sizes, global_size=0,
                         global_bottom_vector_size=0, top_vector_indices=Ti[],
                         local_top_vector_indices=Ti[],
                         all_local_top_vector_a_block_indices=Ti[],
                         local_top_vector_a_block_indices=Vector{Ti}[],
                         iblock_list=zeros(Ti, 2, 0),
                         all_a_block_sub_selection_indices=Ti[],
                         a_block_sub_selection_indices=Vector{Ti}[],
                         a_block_lu_selection_indices=Vector{Ti}[],
                         a_block_B_column_indices=Vector{Ti}[], n_subgroups=0,
                         subgroup_i=-1, subgroup_size=0, block_comm=shared_comm,
                         bottom_vector_indices=Ti[], local_bottom_vector_indices=Ti[],
                         local_bottom_vector_no_overlap_indices=Ti[],
                         local_bottom_vector_no_overlap_sub_selection_indices=Ti[],
                         local_bottom_vector_repeat_indices=Ti[],
                         local_bottom_vector_periodic_pairs=zeros(Ti,2,0),
                         level_shared_comm=shared_comm)
    end

    # Divide the grid into blocks where the number of elements in a block in each
    # dimension is given by `block_sizes`.
    boundary_indices = Ti[]
    function get_boundary_indices!(idim, this_dim, flat_i)
        if this_dim ≤ 0
            push!(boundary_indices, flat_i + 1)
            return nothing
        end

        next_dim = this_dim - 1
        d = dimensions[this_dim]
        n = d.n
        n_local = d.n_local
        flat_i *= n

        # Add offset for distributed blocks.
        flat_i += d.irank * (n_local - 1)

        if idim == this_dim
            bs = block_sizes[idim]
            nelement_local = d.nelement ÷ d.nrank
            ngrid = d.ngrid

            if d.remove_boundaries || d.periodic
                # Always add first and last points to 'boundary points'.
                get_boundary_indices!(idim, next_dim, flat_i)
                get_boundary_indices!(idim, next_dim, flat_i + n_local - 1)
            else
                # Keep boundary points on first/last shared-memory blocks of processes.
                if d.irank > 0
                    get_boundary_indices!(idim, next_dim, flat_i)
                end
                if d.irank < d.nrank - 1
                    get_boundary_indices!(idim, next_dim, flat_i + n_local - 1)
                end
            end

            # Add the interior boundary points
            nblocks = (nelement_local + bs - 1) ÷ bs
            for b ∈ 1:nblocks-1
                # Note we do not `+1` to boundary here because it is more convenient to
                # construct `flat_i` as a 0-based index, and only convert to 1-based just
                # before pushing into `boundary_indices`.
                boundary = b * bs * (ngrid - 1)
                if boundary < n_local
                    get_boundary_indices!(idim, next_dim, flat_i + boundary)
                end
            end
        else
            # Add all points from `d`.
            for i ∈ 0:n_local-1
                get_boundary_indices!(idim, next_dim, flat_i + i)
            end
        end

        return nothing
    end
    for idim ∈ 1:length(dimensions)
        get_boundary_indices!(idim, length(dimensions), 0)
    end
    # There will be duplicated points in block_boundary_indices. Sort the list and remove
    # the duplicates.
    sort!(boundary_indices)
    unique!(boundary_indices)

    # Interior indices are all the indices in level_indices that are not boundary indices.
    interior_indices = setdiff(level_indices, boundary_indices)

    # Get interior indices of the blocks that should be inverted by this processor.
    nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
    nblocks_list = [(nelement + bs - 1) ÷ bs
                    for (nelement, bs) ∈ zip(nelement_local_list, block_sizes)]
    total_nblocks = prod(nblocks_list)
    shared_comm_rank = MPI.Comm_rank(shared_comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    if is_top_level
        # At the top level we might use sparse blocks, which are not supported by
        # shared-memory parallelised blocks. Usually the top level should have more blocks
        # than processes, so lack of parallelisation should not matter.
        subgroup_size = 1
    else
        subgroup_size = max(shared_comm_size ÷ total_nblocks, 1)
    end
    if shared_comm_rank ≥ total_nblocks * subgroup_size
        subgroup_i = -1
    else
        subgroup_i = shared_comm_rank ÷ subgroup_size
    end
    n_subgroups = min(shared_comm_size ÷ subgroup_size, total_nblocks)
    block_comm = MPI.Comm_split(shared_comm, subgroup_i < 0 ? nothing : subgroup_i, 0)
    blocks_per_proc = (total_nblocks + shared_comm_size - 1) ÷ shared_comm_size
    if subgroup_i < 0
        this_proc_blocks = 1:0
    else
        this_proc_blocks = subgroup_i*blocks_per_proc+1:min((subgroup_i+1)*blocks_per_proc,total_nblocks)
    end
    block_interior_indices = Vector{Vector{Ti}}(undef, length(this_proc_blocks))
    block_boundary_indices = Vector{Vector{Ti}}(undef, length(this_proc_blocks))
    iblock_list = Matrix{Ti}(undef, length(nblocks_list), length(this_proc_blocks))
    function get_block_points!(bi, b)
        iblock = zeros(Ti, length(dimensions))
        temp = b - 1
        for (idim, nb) ∈ enumerate(nblocks_list)
            temp, iblock[idim] = divrem(temp, nb)
        end
        iblock .+= 1
        iblock_list[:,bi] .= iblock
        this_bii = Ti[]
        block_interior_indices[bi] = this_bii
        function get_block_interior_indices_from_dim!(this_dim, flat_i)
            if this_dim ≤ 0
                push!(this_bii, flat_i + 1)
                return nothing
            end
            next_dim = this_dim - 1
            d = dimensions[this_dim]
            n = d.n
            ngrid = d.ngrid
            nelement_local = d.nelement ÷ d.nrank
            flat_i *= n

            # Add offset for distributed blocks.
            flat_i += d.irank * (d.n_local - 1)

            this_block = iblock[this_dim]
            bs = block_sizes[this_dim]
            first_element = (this_block - 1) * bs + 1
            last_element = min(this_block * bs, nelement_local)
            if d.irank == 0 && first_element == 1 && !(d.remove_boundaries || d.periodic)
                first_interior_point = 1
            else
                first_interior_point = (first_element - 1) * (ngrid - 1) + 2
            end
            if d.irank == d.nrank - 1 && last_element == nelement_local && !(d.remove_boundaries || d.periodic)
                last_interior_point = n
            else
                last_interior_point = last_element * (ngrid - 1)
            end
            for i ∈ first_interior_point:last_interior_point
                get_block_interior_indices_from_dim!(next_dim, flat_i + i - 1)
            end
            return nothing
        end
        get_block_interior_indices_from_dim!(length(dimensions), 0)

        this_bbi = Ti[]
        block_boundary_indices[bi] = this_bbi
        function get_block_boundary_indices_from_dim!(this_dim, flat_i, boundary_dim)
            if this_dim ≤ 0
                push!(this_bbi, flat_i + 1)
                return nothing
            end
            next_dim = this_dim - 1
            d = dimensions[this_dim]
            n = d.n
            ngrid = d.ngrid
            nelement_local = d.nelement ÷ d.nrank
            flat_i *= n

            # Add offset for distributed blocks.
            flat_i += d.irank * (d.n_local - 1)

            this_block = iblock[this_dim]
            bs = block_sizes[this_dim]
            first_element = (this_block - 1) * bs + 1
            last_element = min(this_block * bs, nelement_local)
            if this_dim == boundary_dim
                if d.irank == 0 && first_element == 1 && !(d.remove_boundaries || d.periodic)
                    # No first boundary point.
                else
                    first_boundary_point = (first_element - 1) * (ngrid - 1) + 1
                    get_block_boundary_indices_from_dim!(next_dim, flat_i + first_boundary_point - 1,
                                                boundary_dim)
                end
                if d.irank == d.nrank - 1 && last_element == nelement_local && !(d.remove_boundaries || d.periodic)
                    # No last boundary point.
                else
                    last_boundary_point = last_element * (ngrid - 1) + 1
                    get_block_boundary_indices_from_dim!(next_dim, flat_i + last_boundary_point - 1,
                                                boundary_dim)
                end
            else
                if d.irank == 0 && first_element == 1 && !(d.remove_boundaries || d.periodic)
                    first_point = 1
                else
                    first_point = (first_element - 1) * (ngrid - 1) + 1
                end
                if d.irank == d.nrank - 1 && last_element == nelement_local && !(d.remove_boundaries || d.periodic)
                    last_point = n
                else
                    last_point = last_element * (ngrid - 1) + 1
                end
                for i ∈ first_point:last_point
                    get_block_boundary_indices_from_dim!(next_dim, flat_i + i - 1,
                                                         boundary_dim)
                end
            end
            return nothing
        end
        for boundary_dim ∈ 1:length(dimensions)
            get_block_boundary_indices_from_dim!(length(dimensions), 0, boundary_dim)
        end
        return nothing
    end
    for (bi, b) ∈ enumerate(this_proc_blocks)
        get_block_points!(bi, b)
    end
    for bii ∈ block_interior_indices
        sort!(bii)
        unique!(bii)
    end
    for bbi ∈ block_boundary_indices
        sort!(bbi)
        unique!(bbi)
    end
    all_block_interior_indices = sort!(vcat(block_interior_indices))
    # Find the points from interior_indices that are part of block_interior_indices.
    # Generally this will not be all the points in block_interior_indices.
    all_local_top_vector_a_block_indices = Ti[]
    local_top_vector_a_block_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
    all_a_block_sub_selection_indices = Ti[]
    a_block_sub_selection_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
    # The following search relies on both `interior_indices` and `block_interior_indices`
    # being sorted.
    for (this_block_interior_indices, this_local_top_vector_a_block_indices, this_a_block_sub_selection_indices) ∈ zip(block_interior_indices, local_top_vector_a_block_indices, a_block_sub_selection_indices)
        i_count = 1
        bi_count = 1
        while (i_count ≤ length(interior_indices)
               && bi_count ≤ length(this_block_interior_indices))
            i = interior_indices[i_count]
            bi = this_block_interior_indices[bi_count]
            if i == bi
                push!(all_local_top_vector_a_block_indices, i)
                push!(this_local_top_vector_a_block_indices, i)
                push!(all_a_block_sub_selection_indices, i_count)
                push!(this_a_block_sub_selection_indices, i_count)
                i_count += 1
                bi_count += 1
            elseif i < bi
                i_count += 1
            else
                bi_count += 1
            end
        end
    end
    sort!(all_local_top_vector_a_block_indices)
    sort!(all_a_block_sub_selection_indices)

    a_block_lu_selection_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
    # The following search relies on both `all_a_block_sub_selection_indices` and
    # `a_block_sub_selection_indices` being sorted.
    for (this_a_block_sub_selection_indices, this_a_block_lu_selection_indices) ∈ zip(a_block_sub_selection_indices, a_block_lu_selection_indices)
        i_count = 1
        bi_count = 1
        while (i_count ≤ length(all_a_block_sub_selection_indices)
               && bi_count ≤ length(this_a_block_sub_selection_indices))
            i = all_a_block_sub_selection_indices[i_count]
            bi = this_a_block_sub_selection_indices[bi_count]
            if i == bi
                push!(this_a_block_lu_selection_indices, i_count)
                i_count += 1
                bi_count += 1
            elseif i < bi
                i_count += 1
            else
                bi_count += 1
            end
        end
    end

    if is_bottom_level && has_periodic
        block_boundary_indices = [get_non_repeated_indices_and_repeats(dimensions, bbi)[1]
                                  for bbi ∈ block_boundary_indices]
    end

    # Simplest way to get the global_bottom_vector_size is to first calculate the size of
    # the 'top vector' then subtract it from `global_size`. This is simplest because the
    # 'top vector' does not have any points that are duplicated between different
    # shared-memory blocks of processes, so we don't have to worry about double-counting.
    top_vector_size = Ref(length(interior_indices))
    if shared_comm_rank == 0
        MPI.Allreduce!(top_vector_size, +, distributed_comm)
    end
    MPI.Bcast!(top_vector_size, shared_comm; root=0)
    global_bottom_vector_size = global_size - top_vector_size[]

    #global_top_vector_indices, _, _ = apply_periodicity_to_indices(dimensions, interior_indices)
    # Top vector indices can never be on periodic boundaries, so no need to apply
    # periodicity.
    global_top_vector_indices = interior_indices
    if is_top_level && is_bottom_level && has_periodic
        # need to handle periodicity
        global_bottom_vector_indices, _ =
            apply_periodicity_to_indices(dimensions, boundary_indices)
        global_bottom_vector_no_overlap_indices, global_bottom_vector_repeat_inds =
            get_non_repeated_indices_and_repeats(dimensions, boundary_indices)
        global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
    elseif is_bottom_level && has_periodic
        # need to handle periodicity
        global_bottom_vector_indices, global_bottom_vector_periodic_pairs =
            apply_periodicity_to_indices(dimensions, boundary_indices)
        global_bottom_vector_no_overlap_indices, _ =
            get_non_repeated_indices_and_repeats(dimensions, boundary_indices)
        global_bottom_vector_repeat_inds = Ti[]
    elseif is_top_level && has_periodic
        # need to handle periodicity
        global_bottom_vector_no_overlap_indices, global_bottom_vector_repeat_inds =
            get_non_repeated_indices_and_repeats(dimensions, boundary_indices)
        global_bottom_vector_indices = boundary_indices
        global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
    else
        global_bottom_vector_indices = boundary_indices
        global_bottom_vector_no_overlap_indices = boundary_indices
        global_bottom_vector_repeat_inds = Ti[]
        global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
    end

    # Get the index within boundary_indices of the entries in block_boundary_indices.
    # The following search relies on both `a_block_B_column_indices` and
    # `block_boundary_indices` being sorted.
    a_block_B_column_indices = [Ti[] for _ ∈ 1:length(block_boundary_indices)]
    if is_bottom_level && has_periodic
        B_column_boundary_indices = get_non_repeated_indices_and_repeats(dimensions, boundary_indices)[1]
    else
        B_column_boundary_indices = boundary_indices
    end
    for (this_a_block_B_column_indices, this_block_boundary_indices) ∈ zip(a_block_B_column_indices, block_boundary_indices)
        nbbi = length(this_block_boundary_indices)
        if nbbi == 0
            continue
        end
        b_count = max(searchsortedlast(B_column_boundary_indices, first(this_block_boundary_indices)) - 1, 1)
        bb_count = 1
        while b_count ≤ length(B_column_boundary_indices) && bb_count ≤ nbbi
            i = B_column_boundary_indices[b_count]
            bi = this_block_boundary_indices[bb_count]
            if i == bi
                push!(this_a_block_B_column_indices, b_count)
                b_count += 1
                bb_count += 1
            elseif i < bi
                b_count += 1
            else
                bb_count += 1
            end
        end
    end

    # Sort the periodic pairs by the 'destination' indices. This turns out to be
    # convenient in a couple of places.
    local_bottom_vector_periodic_pairs = sortslices(global_bottom_vector_periodic_pairs;
                                                    dims=2, lt=(x,y)->(x[1]<y[1]))

    # The level local indices need to be actually the indices of those entries within
    # level_indices.
    local_top_vector_indices = Ti[]
    t_count = 1
    nt = length(interior_indices)
    all_a_block_indices = Ti[]
    a_count = 1
    na = length(all_local_top_vector_a_block_indices)
    local_bottom_vector_indices = Ti[]
    b_count = 1
    nb = length(boundary_indices)
    local_bottom_vector_no_overlap_indices = Ti[]
    local_bottom_vector_no_overlap_sub_selection_indices = Ti[]
    bno_count = 1
    nbno = length(global_bottom_vector_no_overlap_indices)
    local_bottom_vector_repeat_indices = Ti[]
    r_count = 1
    nr = length(global_bottom_vector_repeat_inds)
    p_count = 1
    np = size(local_bottom_vector_periodic_pairs, 2)
    count = 1
    n = length(level_indices)
    while (t_count ≤ nt || a_count ≤ na || b_count ≤ nb || bno_count ≤ nbno || r_count ≤ nr || p_count ≤ np) && count ≤ n
        if t_count ≤ nt && b_count ≤ nb && interior_indices[t_count] == boundary_indices[b_count]
            error("interior_indices and boundary_indices should not overlap, got "
                  * "interior_indices[$t_count]=$(interior_indices[t_count]) and "
                  * "boundary_indices[$b_count]=$(boundary_indices[b_count]).")
        end
        if t_count ≤ nt && interior_indices[t_count] == level_indices[count]
            push!(local_top_vector_indices, count)
            t_count += 1
        end
        if a_count ≤ na && all_local_top_vector_a_block_indices[a_count] == level_indices[count]
            push!(all_a_block_indices, count)
            a_count += 1
        end
        if r_count ≤ nr && global_bottom_vector_repeat_inds[r_count] == boundary_indices[b_count]
            push!(local_bottom_vector_repeat_indices, b_count)
            r_count += 1
        end
        # Need to loop for p_count as there may be repeated entries in local_bottom_vector_periodic_pairs[1,:].
        while p_count ≤ np && local_bottom_vector_periodic_pairs[1,p_count] ≤ level_indices[count]
            if local_bottom_vector_periodic_pairs[1,p_count] == level_indices[count]
                local_bottom_vector_periodic_pairs[1,p_count] = b_count
            end
            p_count += 1
        end
        if bno_count ≤ nbno && global_bottom_vector_no_overlap_indices[bno_count] == level_indices[count]
            push!(local_bottom_vector_no_overlap_indices, count)
            push!(local_bottom_vector_no_overlap_sub_selection_indices, b_count)
            bno_count += 1
        end
        if b_count ≤ nb && boundary_indices[b_count] == level_indices[count]
            push!(local_bottom_vector_indices, count)
            b_count += 1
        end
        count += 1
    end
    if t_count != nt + 1 || a_count != na + 1 || b_count != nb + 1 || r_count != nr + 1 || p_count != np + 1
        error("Did not find all indices in search. t_count=$t_count while nt+1=$(nt+1). "
              * "t_count=$a_count while nt+1=$(nt+1), "
              * "a_count=$a_count while na+1=$(na+1), "
              * "b_count=$b_count while nb+1=$(nb+1), "
              * "r_count=$a_count while nr+1=$(nr+1), "
              * "p_count=$p_count while np+1=$(np+1).")
    end

    # The second row of entries (the 'source' points of the repeated pairs) are not
    # sorted, so cannot be found in the loop above. Need to search the whole of
    # `level_indices` for each second-row entry.
    for i ∈ 1:size(local_bottom_vector_periodic_pairs, 2)
        local_bottom_vector_periodic_pairs[2,i] = searchsortedfirst(level_indices, local_bottom_vector_periodic_pairs[2,i])
    end

    if is_bottom_level && any(d.periodic && d.nrank > 1 for d ∈ dimensions)
        println("Error: periodicity not properly supported for MPI-distributed "
                * "dimensions yet.")
        local_bottom_vector_periodic_pairs = fill(Ti(-1), 2, 1)
    end

    a_block_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
    for (abi, lti) ∈ zip(a_block_indices, local_top_vector_a_block_indices)
        na = length(lti)
        if na == 0
            continue
        end
        count = max(searchsortedlast(level_indices, first(lti)) - 1, 1)
        a_count = 1
        while a_count ≤ na && count ≤ n
            if a_count ≤ na && lti[a_count] == level_indices[count]
                push!(abi, count)
                a_count += 1
            end
            count += 1
        end
        if a_count != na + 1
            error("Did not find all indices in search. a_count=$a_count while na+1=$(na+1).")
        end
    end

    return LevelInfo(; has_periodic, block_sizes, global_size, global_bottom_vector_size,
                     top_vector_indices=global_top_vector_indices,
                     local_top_vector_indices=local_top_vector_indices,
                     iblock_list=iblock_list,
                     all_local_top_vector_a_block_indices=all_a_block_indices,
                     local_top_vector_a_block_indices=a_block_indices,
                     all_a_block_sub_selection_indices=all_a_block_sub_selection_indices,
                     a_block_sub_selection_indices=a_block_sub_selection_indices,
                     a_block_lu_selection_indices=a_block_lu_selection_indices,
                     a_block_B_column_indices=a_block_B_column_indices,
                     n_subgroups=n_subgroups, subgroup_i=subgroup_i,
                     subgroup_size=subgroup_size, block_comm=block_comm,
                     bottom_vector_indices=global_bottom_vector_indices,
                     local_bottom_vector_indices=local_bottom_vector_indices,
                     local_bottom_vector_no_overlap_indices=local_bottom_vector_no_overlap_indices,
                     local_bottom_vector_no_overlap_sub_selection_indices=local_bottom_vector_no_overlap_sub_selection_indices,
                     local_bottom_vector_repeat_indices=local_bottom_vector_repeat_indices,
                     local_bottom_vector_periodic_pairs=local_bottom_vector_periodic_pairs,
                     level_shared_comm=shared_comm)
end

function get_block_diagonal_solver(level_info, data_type, use_sparse, is_top_level,
                                   use_shared_blocks, timer, check_lu,
                                   block_allocate_shared_float=nothing,
                                   block_allocate_shared_int=nothing,
                                   block_synchronize_shared=nothing)
    # The A blocks may be sparse at the top level, but will generally be dense on lower
    # levels, so only use a sparse LU solver when is_top_level=true.
    if isempty(level_info.a_block_sub_selection_indices)
        return MPIStaticCondensationNull{data_type}()
    elseif use_shared_blocks
        return BlockDiagonalSolverShared{data_type}(level_info.global_size - level_info.global_bottom_vector_size,
                                                    level_info.a_block_sub_selection_indices[1],
                                                    level_info.a_block_lu_selection_indices[1],
                                                    level_info.a_block_B_column_indices[1],
                                                    level_info.block_comm,
                                                    block_allocate_shared_float,
                                                    block_allocate_shared_int,
                                                    block_synchronize_shared, timer,
                                                    check_lu)
    else
        return BlockDiagonalSolverSerial{data_type}(level_info.global_size - level_info.global_bottom_vector_size,
                                                    level_info.a_block_sub_selection_indices,
                                                    level_info.a_block_lu_selection_indices,
                                                    level_info.a_block_B_column_indices,
                                                    use_sparse && is_top_level, timer, check_lu)
    end
end

"""
    mpi_static_condensation(dimensions::Vector{<:Dimension};
                            level_multiplier::Integer=2,
                            reduce_proc_count_with_blocks::Bool=false,
                            comm::MPI.Comm=MPI.COMM_WORLD,
                            distributed_comm::Union{MPI.Comm,Nothing}=missing,
                            shared_comm::MPI.Comm=MPI.COMM_SELF,
                            allocate_shared_float::Union{Function,Nothing}=nothing,
                            allocate_shared_int::Union{Function,Nothing}=nothing,
                            synchronize_shared::Union{Function,Nothing}=nothing,
                            schur_tile_size::Union{Nothing,Integer}=nothing,
                            use_sparse::Bool=true, separate_Ainv_B::Bool=false,
                            optimize_schur_complement_size::Bool=true,
                            timer::Union{Nothing,TimerOutput}=nothing,
                            check_lu::Bool=false)

`dimensions` is a length-\$d\$ Vector of `Dimension` objects, which can be created with
`create_dimension()`, which describe the structure of the \$d\$-dimensional
continuous-finite-element grid.  The right-hand-side and solution vectors are flattened
(aka. linear-indexed) versions of the \$d\$-dimensional array representing a variable on
the finite element grid. The order of `dimensions` corresponds to the order of the indices
in the multi-dimensional array. For a description of the discretization, see the
`create_dimensions()` docstring.

`level_multiplier` gives the factor by which the block size is increased in each dimension
at each level.

`reduce_proc_count_with_blocks` sets whether the number of processes involved in the solve
at each level is reduced when the number of blocks at that level is less than the total
number of processes. Usually reducing the number of processes is probably not helpful
(hence the default is `false`), but if MPI communication cost is the dominant bottleneck
it might be faster.

`comm` is divided into equally sized shared-memory blocks. `shared_comm` represents the
shared-memory block that this process belongs to - it must be a subset of `comm`, and its
members must be able to create shared-memory arrays.

`allocate_shared_float`, `allocate_shared_int`, and `synchronize_shared` are as required
by `mpi_schur_complement()`. `schur_tile_size` is passed to the `tile_size` argument of
`mpi_schur_complement()`.

`use_sparse` indicates whether to use a sparse-matrix solver as the lowest-level LU
solver, and within the MPISchurComplement solvers.

`separate_Ainv_B` is passed through to the MPISchurComplement constructors.

`optimize_schur_complement_size` sets the strategy used to pick which dimension to split
at each level, when splitting between shared-memory processes (for distributed-memory the
group size is always required to exactly divide the number of elements, to be consistent
with the distributed-memory domain-decomposition). The default strategy (`true`) splits
the largest (according to value of `n`) dimension remaining at each level, in order to
minimise the size of the Schur complement block. The alternative strategy (`false`) tries
to optimise load balance by considering first dimensions whose remaining `nelement` value
can be exactly divided by the group size (picking the largest of these), and only
considering other dimensions if no dimension can be exactly divided. In either case,
dimensions that are distributed over different shared-memory MPI blocks are divided first,
until the locally-owned parts of all dimensions are contained within the same
shared-memory MPI block. The two strategies will be equivalent as long as the largest
dimension at each level is anyway exactly divisible, which may often be the case (e.g. if
the number of processes is a power of 2, and `nelement` of the dimensions contain enough
factors of 2).

`timer` can be passed a `TimerOutput` object to collect run timings.

`check_lu=true` can be passed to activate extra checks that all values are finite in
matrices being factorized.
"""
function mpi_static_condensation(dimensions::Vector{<:Dimension};
                                 level_multiplier::Integer=2,
                                 reduce_proc_count_with_blocks::Bool=false,
                                 comm::MPI.Comm=MPI.COMM_WORLD,
                                 distributed_comm::Union{MPI.Comm,Nothing}=missing,
                                 shared_comm::MPI.Comm=MPI.COMM_SELF,
                                 allocate_shared_float::F1=nothing,
                                 allocate_shared_int::F2=nothing,
                                 synchronize_shared::F3=nothing,
                                 schur_tile_size::Union{Nothing,Integer}=nothing,
                                 use_sparse::Bool=true, separate_Ainv_B::Bool=false,
                                 timer::Union{Nothing,TimerOutput}=nothing,
                                 check_lu::Bool=false) where {F1<:Union{Function,Nothing}, F2<:Union{Function,Nothing}, F3<:Union{Function,Nothing}}

    data_type = Float64
    ind_type = Int64

    comm_size = MPI.Comm_size(comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    shared_comm_rank = MPI.Comm_rank(shared_comm)

    if distributed_comm === missing
        # Create default distributed_comm
        distributed_comm = MPI.Comm_split(comm, shared_comm_rank == 0 ? 0 : nothing, 0)
    end

    if comm_size % shared_comm_size != 0
        error("Size of shared_comm ($shared_comm_size) does not divide the size of comm "
              * "($comm_size).")
    end
    n_blocks = comm_size ÷ shared_comm_size

    n_blocks_factors = factor(Vector, n_blocks)
    shared_comm_size_factors = factor(Vector, shared_comm_size)

    # Not sure if this is necessarily the most efficient choice for all grids and numbers
    # of processes - everything following should work for any set of block_sizes, so other
    # choices could be made here.
    block_sizes_list = [ones(ind_type, length(dimensions))]
    nelement_list = [d.nelement for d ∈ dimensions]
    nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
    total_local_nblock = prod(nelement_local_list)
    total_local_nblock_list = [total_local_nblock]
    while total_local_nblock > 1
        previous_block_sizes = block_sizes_list[end]
        this_block_sizes = @. min(previous_block_sizes .* level_multiplier, nelement_local_list)
        local_nblock_list = @. (nelement_local_list + this_block_sizes - 1) ÷ this_block_sizes
        total_local_nblock = prod(local_nblock_list)
        push!(total_local_nblock_list, total_local_nblock)
        push!(block_sizes_list, this_block_sizes)
    end

    dimensions_without_periodic = [Dimension(; nelement=d.nelement, ngrid=d.ngrid,
                                             nrank=d.nrank, irank=d.irank, periodic=false,
                                             remove_boundaries=(d.periodic || d.remove_boundaries))
                                   for d ∈ dimensions]

    n_levels = length(block_sizes_list)
    level_info_list = Vector{LevelInfo{ind_type,typeof(shared_comm)}}(undef, n_levels)
    level_indices = get_global_indices(dimensions_without_periodic,
                                       collect(1:prod(d.n_local for d ∈ dimensions)))
    level_global_size = prod(d.n for d ∈ dimensions)
    level_shared_comm = shared_comm
    level_shared_comm_size = shared_comm_size
    for (level, (block_sizes, total_local_nblock)) ∈ enumerate(zip(block_sizes_list,
                                                                   total_local_nblock_list))
        if level == 1 || level == n_levels
            # Only handle periodicity on the final level
            dims = dimensions
        else
            dims = dimensions_without_periodic
        end
        if reduce_proc_count_with_blocks && level_shared_comm_size > total_local_nblock
            # Not enough blocks to divide among processes in existing level_shared_comm,
            # which probably indicates that the parallel efficiency of continuing the
            # solve on that many proceses. We therefore decrease the number of processes
            # being used.
            if level_shared_comm != MPI.COMM_NULL
                level_shared_comm =
                    MPI.Comm_split(level_shared_comm,
                                   MPI.Comm_rank(level_shared_comm) < total_local_nblock ? 0 : nothing,
                                   0)
            end
            level_shared_comm_size = total_local_nblock
        end
        # Keep selecting the subset of `1:prod(d.n_local for d ∈ dimensions)` that is
        # involved in each successive level.
        this_level_info = split_matrix(dims, level_indices, block_sizes,
                                       level_global_size, level==1, level==n_levels,
                                       distributed_comm, level_shared_comm)
        level_info_list[level] = this_level_info
        level_indices = this_level_info.bottom_vector_indices
        level_global_size = this_level_info.global_bottom_vector_size
    end

    # Create lowest level MPISchurComplement solver
    # Use a parallelized dense-matrix LU solver for the last Schur complement solve as
    # long as the last Schur complement matrix is not too small.
    last_level_info = level_info_list[end]
    if last_level_info.level_shared_comm != MPI.COMM_NULL
        last_use_shared_blocks = (length(level_info_list) > 1
                                  && length(last_level_info.local_top_vector_a_block_indices) == 1
                                  && MPI.Comm_size(last_level_info.block_comm) > 1)
        if last_use_shared_blocks
            block_comm_rank = MPI.Comm_rank(last_level_info.block_comm)
            block_comm_size = MPI.Comm_size(last_level_info.block_comm)
            if block_comm_size == shared_comm_size
                last_block_allocate_shared_float = allocate_shared_float
                last_block_allocate_shared_int = allocate_shared_int
                if synchronize_shared === nothing
                    last_block_synchronize_shared = () -> MPI.Barrier(last_level_info.block_comm)
                else
                    last_block_synchronize_shared = synchronize_shared
                end
            else
                last_block_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=last_level_info.block_comm)
                last_block_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=last_level_info.block_comm)
                last_block_synchronize_shared = () -> MPI.Barrier(last_level_info.block_comm)
            end
            last_A_block_solver = get_block_diagonal_solver(last_level_info, data_type,
                                                            use_sparse,
                                                            length(level_info_list) == 1,
                                                            true, timer, check_lu,
                                                            last_block_allocate_shared_float,
                                                            last_block_allocate_shared_int,
                                                            last_block_synchronize_shared)
        else
            block_comm_rank = 0
            block_comm_size = 1
            last_A_block_solver = get_block_diagonal_solver(last_level_info, data_type,
                                                            use_sparse,
                                                            length(level_info_list) == 1,
                                                            false, timer, check_lu)
        end
        last_level_shared_comm = last_level_info.level_shared_comm
        level_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=last_level_shared_comm)
        level_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=last_level_shared_comm)
        last_parallel_schur = last_level_info.global_bottom_vector_size ≥ 1024
        if reduce_proc_count_with_blocks || synchronize_shared === nothing
            level_synchronize_shared = () -> MPI.Barrier(last_level_shared_comm)
        else
            level_synchronize_shared = synchronize_shared
        end
        this_level_sc =
            mpi_schur_complement(last_A_block_solver, data_type, data_type, data_type,
                                 last_level_info.top_vector_indices,
                                 last_level_info.bottom_vector_indices; comm=comm,
                                 shared_comm=last_level_shared_comm,
                                 distributed_comm=distributed_comm,
                                 allocate_shared_float=level_allocate_shared_float,
                                 allocate_shared_int=level_allocate_shared_int,
                                 synchronize_shared=level_synchronize_shared,
                                 use_sparse=false, sparse_Ainv_B=false,
                                 parallel_schur=last_parallel_schur,
                                 copy_input_to_dense_buffers=(use_sparse && last_level_info.has_periodic),
                                 skip_factorization=true, schur_tile_size=schur_tile_size,
                                 check_lu=check_lu, timer=timer)
    else
        this_level_sc = MPIStaticCondensationNull{data_type}()
    end

    this_level_schur_solver = nothing
    for (level, this_level_info) ∈ reverse(collect(enumerate(level_info_list)))
        if this_level_info.level_shared_comm == MPI.COMM_NULL
            this_level_schur_solver = MPIStaticCondensationNull{data_type}()
            continue
        end
        this_level_shared_comm = this_level_info.level_shared_comm
        level_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=this_level_shared_comm)
        level_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=this_level_shared_comm)
        this_level_comm_size = MPI.Comm_size(this_level_shared_comm)
        this_level_comm_rank = MPI.Comm_rank(this_level_shared_comm)
        if level < n_levels
            if reduce_proc_count_with_blocks || synchronize_shared === nothing
                level_synchronize_shared = () -> MPI.Barrier(this_level_shared_comm)
            else
                level_synchronize_shared = synchronize_shared
            end

            if this_level_info.block_comm == MPI.COMM_NULL
                block_comm_rank = 0
                block_comm_size = 1
            else
                block_comm_rank = MPI.Comm_rank(this_level_info.block_comm)
                block_comm_size = MPI.Comm_size(this_level_info.block_comm)
            end
            use_shared_blocks = (level > 1 && length(this_level_info.local_top_vector_a_block_indices) == 1
                                 && block_comm_size > 1)
            if block_comm_size == shared_comm_size
                block_allocate_shared_float = allocate_shared_float
                block_allocate_shared_int = allocate_shared_int
                if synchronize_shared === nothing
                    block_synchronize_shared = () -> MPI.Barrier(this_level_info.block_comm)
                else
                    block_synchronize_shared = synchronize_shared
                end
            else
                block_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=this_level_info.block_comm)
                block_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=this_level_info.block_comm)
                block_synchronize_shared = () -> MPI.Barrier(this_level_info.block_comm)
            end

            C_buffer = nothing
            C_buffer_ncopies = nothing
            if use_sparse
                schur_complement_buffer =
                    get_shared_sparse_matrix_csc_buffer(dimensions,
                                                        this_level_info.block_sizes,
                                                        this_level_info.bottom_vector_indices,
                                                        this_level_info.bottom_vector_indices,
                                                        this_level_shared_comm,
                                                        level_allocate_shared_float,
                                                        level_allocate_shared_int)
                if use_shared_blocks
                    if this_level_info.block_comm == MPI.COMM_NULL
                        this_A_block_solver = MPIStaticCondensationNull{data_type}()
                        Ainv_dot_B_buffer = nothing
                        C_buffer = nothing
                    else
                        this_A_block_solver = get_block_diagonal_solver(this_level_info,
                                                                        data_type, use_sparse,
                                                                        level==1,
                                                                        use_shared_blocks,
                                                                        timer, check_lu,
                                                                        block_allocate_shared_float,
                                                                        block_allocate_shared_int,
                                                                        block_synchronize_shared)
                        Ainv_dot_B_buffer =
                            BlockAinvDotBShared{data_type}(this_level_info.a_block_sub_selection_indices[1],
                                                           this_level_info.a_block_B_column_indices[1],
                                                           block_comm_rank, block_comm_size,
                                                           block_allocate_shared_float,
                                                           block_synchronize_shared)
                        C_buffer_column_per_subgroup = this_level_info.n_subgroups < 2^length(dimensions)
                        if C_buffer_column_per_subgroup
                            C_buffer_ncopies = this_level_info.n_subgroups
                        else
                            C_buffer_ncopies = 2^length(dimensions)
                        end
                        C_vector_intermediate_buffer =
                            level_allocate_shared_float(C_buffer_ncopies,
                                                        this_level_info.global_bottom_vector_size)
                        if this_level_comm_rank == 0
                            C_vector_intermediate_buffer .= 0.0
                        end
                        C_vector_points_per_proc = (this_level_info.global_bottom_vector_size + this_level_comm_size - 1) ÷ this_level_comm_size
                        C_vector_range = this_level_comm_rank*C_vector_points_per_proc+1:min((this_level_comm_rank+1)*C_vector_points_per_proc,this_level_info.global_bottom_vector_size)
                        C_block_row_inds_full = this_level_info.a_block_B_column_indices[1]

                       C_nrow = length(C_block_row_inds_full)
                       C_rows_per_proc = (C_nrow + block_comm_size - 1) ÷ block_comm_size
                       if isempty(this_level_info.a_block_sub_selection_indices[1])
                           # There are no entries in the block handled by this process, so
                           # to avoid accessing zero-length vectors, set the row range to
                           # be empty also.
                           C_partial_row_range = 1:0
                       else
                           C_partial_row_range = block_comm_rank*C_rows_per_proc+1:min((block_comm_rank+1)*C_rows_per_proc,C_nrow)
                       end

                        C_vector_init_range, C_matrix_init_range, C_buffer_column =
                            get_C_buffer_init_ranges(this_level_info.global_bottom_vector_size,
                                                     schur_complement_buffer,
                                                     C_buffer_column_per_subgroup,
                                                     C_buffer_ncopies,
                                                     this_level_info.subgroup_i,
                                                     this_level_info.n_subgroups,
                                                     this_level_info.subgroup_size,
                                                     block_comm_rank)

                        block_hypercube_position =
                            get_C_hypercube_position(this_level_info.iblock_list[:,1])

                        if !C_buffer_column_per_subgroup
                            C_buffer_column = block_hypercube_position
                        end

                        C_buffer =
                            BlockCShared{data_type}(C_block_row_inds_full,
                                                    C_partial_row_range,
                                                    this_level_info.a_block_sub_selection_indices[1],
                                                    C_buffer_column,
                                                    C_vector_intermediate_buffer,
                                                    C_vector_range,
                                                    C_buffer_column_per_subgroup,
                                                    C_vector_init_range,
                                                    C_matrix_init_range,
                                                    this_level_info.subgroup_i,
                                                    block_allocate_shared_float,
                                                    block_synchronize_shared,
                                                    block_comm_rank, block_comm_size,
                                                    level_synchronize_shared)
                    end
                else
                    this_A_block_solver = get_block_diagonal_solver(this_level_info,
                                                                    data_type, use_sparse,
                                                                    level==1, false,
                                                                    timer, check_lu)
                    Ainv_dot_B_buffer =
                        BlockAinvDotBSerial{data_type}(this_level_info.a_block_sub_selection_indices,
                                                       this_level_info.a_block_B_column_indices)
                    C_buffer_column_per_subgroup = this_level_info.n_subgroups < 2^length(dimensions)
                    if C_buffer_column_per_subgroup
                        C_buffer_ncopies = this_level_info.n_subgroups
                    else
                        C_buffer_ncopies = 2^length(dimensions)
                    end
                    C_vector_intermediate_buffer =
                        level_allocate_shared_float(C_buffer_ncopies,
                                                    this_level_info.global_bottom_vector_size)
                    if this_level_comm_rank == 0
                        C_vector_intermediate_buffer .= 0.0
                    end
                    C_vector_points_per_proc = (this_level_info.global_bottom_vector_size + this_level_comm_size - 1) ÷ this_level_comm_size
                    C_vector_range = this_level_comm_rank*C_vector_points_per_proc+1:min((this_level_comm_rank+1)*C_vector_points_per_proc, this_level_info.global_bottom_vector_size)
                    C_block_row_inds = this_level_info.a_block_B_column_indices

                    C_vector_init_range, C_matrix_init_range, C_buffer_column =
                        get_C_buffer_init_ranges(this_level_info.global_bottom_vector_size,
                                                 schur_complement_buffer,
                                                 C_buffer_column_per_subgroup,
                                                 C_buffer_ncopies,
                                                 this_level_info.subgroup_i,
                                                 this_level_info.n_subgroups,
                                                 this_level_info.subgroup_size,
                                                 block_comm_rank)

                    C_block_hypercube_positions =
                        [get_C_hypercube_position(iblock)
                         for iblock ∈ eachcol(this_level_info.iblock_list)]

                    C_buffer =
                        BlockCSerial{data_type}(C_block_row_inds,
                                                this_level_info.a_block_sub_selection_indices,
                                                C_block_hypercube_positions,
                                                C_vector_intermediate_buffer,
                                                C_vector_range, C_vector_init_range,
                                                C_matrix_init_range, C_buffer_column,
                                                this_level_info.subgroup_i,
                                                block_synchronize_shared,
                                                level_synchronize_shared)
                end
            else
                this_A_block_solver = get_block_diagonal_solver(this_level_info,
                                                                data_type, use_sparse,
                                                                level==1,
                                                                use_shared_blocks, timer,
                                                                check_lu,
                                                                block_allocate_shared_float,
                                                                block_allocate_shared_int,
                                                                block_synchronize_shared)
                Ainv_dot_B_buffer = nothing
                schur_complement_buffer = nothing
            end
            this_level_sc =
                mpi_schur_complement(this_A_block_solver, data_type, data_type, data_type,
                                     this_level_info.top_vector_indices,
                                     this_level_info.bottom_vector_indices; comm=comm,
                                     shared_comm=this_level_shared_comm,
                                     distributed_comm=distributed_comm,
                                     allocate_shared_float=level_allocate_shared_float,
                                     allocate_shared_int=level_allocate_shared_int,
                                     synchronize_shared=level_synchronize_shared,
                                     Ainv_dot_B_buffer=Ainv_dot_B_buffer,
                                     C_buffer=C_buffer,
                                     C_dot_Ainv_dot_B_buffer_ncopies=C_buffer_ncopies,
                                     schur_complement_buffer=schur_complement_buffer,
                                     use_sparse=use_sparse, sparse_Ainv_B=use_sparse,
                                     parallel_schur=this_level_schur_solver,
                                     skip_factorization=true,
                                     schur_tile_size=schur_tile_size, check_lu=check_lu,
                                     timer=timer)
        end
        level_shared_comm_rank = MPI.Comm_rank(this_level_shared_comm)
        level_shared_comm_size = MPI.Comm_size(this_level_shared_comm)
        this_u_buffer = level_allocate_shared_float(length(this_level_info.local_top_vector_indices))
        this_v_buffer = level_allocate_shared_float(length(this_level_info.local_bottom_vector_indices))

        # Need to create a version of local_top_vector_indices and
        # local_bottom_vector_indices that is split into ranges to be handled in parallel
        # by all the processes in the shared-memory block.
        ntop = length(this_level_info.local_top_vector_indices)
        top_points_per_proc = (ntop + level_shared_comm_size - 1) ÷ level_shared_comm_size
        top_subset = level_shared_comm_rank*top_points_per_proc+1:min((level_shared_comm_rank+1)*top_points_per_proc,ntop)

        nbottom = length(this_level_info.local_bottom_vector_indices)
        bottom_points_per_proc = (nbottom + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset = level_shared_comm_rank*bottom_points_per_proc+1:min((level_shared_comm_rank+1)*bottom_points_per_proc,nbottom)
        this_shared_local_bottom_vector_indices = this_level_info.local_bottom_vector_indices[bottom_subset]
        this_shared_local_bottom_sub_selection_indices = (1:length(this_level_info.local_bottom_vector_indices))[bottom_subset]

        nbottom_no_overlap = length(this_level_info.local_bottom_vector_no_overlap_indices)
        bottom_points_per_proc_no_overlap = (nbottom_no_overlap + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_no_overlap = level_shared_comm_rank*bottom_points_per_proc_no_overlap+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_no_overlap,nbottom_no_overlap)
        this_shared_local_bottom_vector_no_overlap_indices = this_level_info.local_bottom_vector_no_overlap_indices[bottom_subset_no_overlap]
        this_shared_local_bottom_sub_selection_no_overlap_indices = this_level_info.local_bottom_vector_no_overlap_sub_selection_indices[bottom_subset_no_overlap]

        nbottom_repeats = length(this_level_info.local_bottom_vector_repeat_indices)
        bottom_points_per_proc_repeats = (nbottom_repeats + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_repeats = level_shared_comm_rank*bottom_points_per_proc_repeats+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_repeats,nbottom_repeats)
        this_shared_local_bottom_vector_repeat_indices = this_level_info.local_bottom_vector_repeat_indices[bottom_subset_repeats]

        nbottom_periodic_pairs = size(this_level_info.local_bottom_vector_periodic_pairs, 2)
        bottom_points_per_proc_periodic_pairs = (nbottom_periodic_pairs + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_periodic_pairs = level_shared_comm_rank*bottom_points_per_proc_periodic_pairs+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_periodic_pairs,nbottom_periodic_pairs)

        # On this processor, handle the overlap pairs where the destination of the overlap
        # is in this_shared_local_bottom_sub_selection_no_overlap_indices.
        local_bottom_vector_periodic_pairs = this_level_info.local_bottom_vector_periodic_pairs
        this_proc_pairs_inds = ind_type[]
        pair_count = 1
        bottom_count = 1
        while pair_count ≤ size(local_bottom_vector_periodic_pairs, 2) && bottom_count ≤ length(this_shared_local_bottom_sub_selection_no_overlap_indices)
            if local_bottom_vector_periodic_pairs[1,pair_count] == this_shared_local_bottom_sub_selection_no_overlap_indices[bottom_count]
                push!(this_proc_pairs_inds, pair_count)
                pair_count += 1
                # Note that local_bottom_vector_periodic_pairs may have repeated first-row entries, so do not
                # increment bottom_count here.
            elseif local_bottom_vector_periodic_pairs[1,pair_count] < this_shared_local_bottom_sub_selection_no_overlap_indices[bottom_count]
                pair_count += 1
            else
                bottom_count += 1
            end
        end
        this_shared_local_bottom_periodic_pairs = local_bottom_vector_periodic_pairs[:,this_proc_pairs_inds]

        na = length(this_level_info.all_local_top_vector_a_block_indices)
        a_points_per_block_proc = (na + block_comm_size - 1) ÷ block_comm_size
        partial_a_block_range = block_comm_rank*a_points_per_block_proc+1:min((block_comm_rank+1)*a_points_per_block_proc,na)

        this_level_schur_solver =
            MPIStaticCondensationParallel(this_level_info.global_size, this_level_sc,
                                          this_level_info.local_top_vector_indices,
                                          this_level_info.all_local_top_vector_a_block_indices,
                                          this_level_info.all_local_top_vector_a_block_indices[partial_a_block_range],
                                          this_level_info.all_a_block_sub_selection_indices[partial_a_block_range],
                                          this_level_info.local_bottom_vector_indices,
                                          this_shared_local_bottom_vector_indices,
                                          this_shared_local_bottom_vector_no_overlap_indices,
                                          this_shared_local_bottom_sub_selection_indices,
                                          this_shared_local_bottom_sub_selection_no_overlap_indices,
                                          this_shared_local_bottom_vector_repeat_indices,
                                          this_shared_local_bottom_periodic_pairs,
                                          this_u_buffer, this_v_buffer,
                                          this_level_info.has_periodic,
                                          level_synchronize_shared, timer)
    end
    # The level-1 MPIStaticCondensationParallel is not a 'Schur complement solver', but
    # the full matrix solver.
    solver = this_level_schur_solver

    return solver
end

"""
    update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti},
                          new_A::AbstractSparseMatrixCSC{Tf,Ti}, rowinds,
                          colinds) where {Tf,Ti}
    update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::AbstractMatrix{Tf},
                          rowinds, colinds) where {Tf,Ti}
    update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::SubArray{Tf,2},
                          rowinds, colinds) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors. For this FixedMatrixCSC version, also filter out
zeros because FixedMatrixCSC was probably defined with a maximal stencil, which might
contain extra zeros.

`rowinds` gives the subset of rows in `new_A` that should be copied into `A`.

`colinds` gives the subset of columns in `new_A` that should be copied into `A`.
"""
update_sparse_matrix!

function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti},
                               new_A::AbstractSparseMatrixCSC{Tf,Ti}, rowinds,
                               colinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(colptr, 1)
    resize!(rowval, 0)
    resize!(nzval, 0)
    count = 1
    n_rowinds = length(rowinds)
    for col ∈ colinds
        colstart = new_colptr[col]
        colend = new_colptr[col+1] - 1
        if colend < colstart
            continue
        end
        row_count = max(searchsortedlast(rowinds, new_rowval[colstart]) - 1, 1)
        for new_i ∈ colstart:colend
            rv = new_rowval[new_i]
            while row_count ≤ n_rowinds && rowinds[row_count] < rv
                row_count += 1
            end
            if row_count > n_rowinds
                break
            end
            if rowinds[row_count] == rv
                newval = new_nzval[new_i]
                if !iszero(newval)
                    push!(rowval, row_count)
                    push!(nzval, newval)
                    count += 1
                    row_count += 1
                end
            end
        end
        push!(colptr, count)
    end

    return nothing
end
function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::AbstractMatrix{Tf},
                               rowinds, colinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    resize!(colptr, 1)
    resize!(rowval, 0)
    resize!(nzval, 0)
    count = 1

    for (j1, j2) ∈ enumerate(colinds)
        for (i1, i2) ∈ enumerate(rowinds)
            val = new_A[i2,j2]
            if val != zero(Tf)
                push!(rowval, i1)
                push!(nzval, val)
            end
        end
        push!(colptr, count)
    end

    return nothing
end
@inline function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::SubArray{Tf,2},
                               rowinds, colinds) where {Tf,Ti}
    full_rowinds, full_colinds = new_A.indices
    return @views update_sparse_matrix!(A, parent(new_A), full_rowinds[rowinds],
                                        full_colinds[colinds])
end

function lu!(block_diagonal_solver::BlockDiagonalSolverSerial, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    check_lu = block_diagonal_solver.check_lu
    if solver != [nothing]
        for (s, inds, buffer) ∈ zip(solver, block_diagonal_solver.lu_selection_indices,
                                    block_diagonal_solver.sparse_buffers)
            if isa(s, UmfpackLU)
                update_sparse_matrix!(buffer, A, inds, inds)
                lu!(s, buffer; reuse_symbolic=false, check=check_lu)
            else
                factors = s.factors
                for (j1, j2) ∈ enumerate(inds), (i1, i2) ∈ enumerate(inds)
                    factors[i1,j1] = A[i2,j2]
                end
                getrf!(factors, s.ipiv; check=check_lu)
            end
        end
    end
    return nothing
end
function lu!(block_diagonal_solver::BlockDiagonalSolverShared, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    factors = block_diagonal_solver.factors
    lu_selection_indices = block_diagonal_solver.lu_selection_indices
    partial_lu_selection_indices = block_diagonal_solver.partial_lu_selection_indices
    partial_col_range = block_diagonal_solver.partial_col_range
    synchronize_shared = block_diagonal_solver.synchronize_shared

# Could make this branch more efficient when A is a (view of a) sparse matrix?
    for (j1, j2) ∈ zip(partial_col_range, partial_lu_selection_indices), (i1, i2) ∈ enumerate(lu_selection_indices)
        factors[i1,j1] = A[i2,j2]
    end

    synchronize_shared()

    if isa(solver, MPIDenseLU)
        # Note that this would not work if we were using distributed MPI in the MPIDenseLU
        # `solver`, as in the distributed-MPI case, `factors` is not factorised directly,
        # and we require that it is for `local_block_serial_solver` to work.
        lu!(solver, factors)
    elseif isa(solver, LU)
        getrf!(factors, solver.ipiv; check=block_diagonal_solver.check_lu)
    end
    return nothing
end

function ldiv!(x::AbstractVector{T}, block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::AbstractVector{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if solvers != [nothing]
        x_buffer = block_diagonal_solver.x_buffer
        u_buffer = block_diagonal_solver.u_buffer
        for (bi, s) ∈ zip(block_diagonal_solver.block_indices, solvers)
            n = length(bi)
            this_u_buffer = @view u_buffer[1:n]
            this_x_buffer = @view x_buffer[1:n]
            for (i1, i2) ∈ enumerate(bi)
                this_u_buffer[i1] = u[i2]
            end
            ldiv!(this_x_buffer, s, this_u_buffer)
            for (i2, i1) ∈ enumerate(bi)
                x[i1] = this_x_buffer[i2]
            end
        end
    end
    return nothing
end
function ldiv!(x::AbstractVector{T}, block_diagonal_solver::BlockDiagonalSolverShared{T},
               u::AbstractVector{T}) where T
    solver = block_diagonal_solver.local_block_solver
    x_buffer = block_diagonal_solver.x_buffer
    block_comm_rank = block_diagonal_solver.block_comm_rank
    block_indices = block_diagonal_solver.block_indices
    synchronize_shared = block_diagonal_solver.synchronize_shared

    # Need to synchronize here as `u_buffer` is filled only on block_comm_rank==0, but `u`
    # was filled in parallel. Maybe it would be worth filling `u_buffer` in parallel? Then
    # would need to synchronize before `ldiv!()` call.
    synchronize_shared()

    if solver === nothing
        # Nothing to do.
    elseif isa(solver, MPIDenseLU)
        u_buffer = block_diagonal_solver.u_buffer
        if block_comm_rank == 0
            for (i1, i2) ∈ enumerate(block_indices)
                u_buffer[i1] = u[i2]
            end
        end
        ldiv!(x_buffer, solver, u_buffer)
        if block_comm_rank == 0
            for (i2, i1) ∈ enumerate(block_indices)
                x[i1] = x_buffer[i2]
            end
        end
    else
        if block_comm_rank == 0
            for (i1, i2) ∈ enumerate(block_indices)
                x_buffer[i1] = u[i2]
            end
            ldiv!(solver, x_buffer)
            for (i2, i1) ∈ enumerate(block_indices)
                x[i1] = x_buffer[i2]
            end
        end
    end
    return nothing
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractVector{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(x::AbstractMatrix{T},
               block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractMatrix{T}) where T
    if block_diagonal_solver.local_block_solver !== nothing
        for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
            ldiv!(this_x, block_diagonal_solver, this_u)
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::Matrix{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if length(solvers) == 1 && length(block_diagonal_solver.block_indices[1]) == size(u, 1)
        # There is only one block, so do not need to select range out of x/u.
        ldiv!(x, solvers[1], u)
    else
        if block_diagonal_solver.local_block_solver !== nothing
            for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
                ldiv!(this_x, block_diagonal_solver, this_u)
            end
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverShared{T},
               u::Matrix{T}) where T
    for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
        ldiv!(this_x, block_diagonal_solver, this_u)
    end
    return nothing
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractMatrix{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverSerial{T}, u::Matrix{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if length(solvers) == 1 && length(block_diagonal_solver.block_indices[1]) == size(u, 1)
        # There is only one block, so do not need to select range out of u.
        ldiv!(solvers[1], u)
        return nothing
    else
        return ldiv!(u, block_diagonal_solver, u)
    end
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverShared{T}, u::Matrix{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function sparse_column_has_overlap(rowval, bi)
    r_count = 1
    b_count = 1
    while r_count ≤ length(rowval) && b_count ≤ length(bi)
        if rowval[r_count] == bi[b_count]
            return true
        elseif rowval[r_count] < bi[b_count]
            r_count += 1
        else
            b_count += 1
        end
    end
    return false
end
function ldiv!(x::AbstractSparseMatrixCSC{T},
               block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::AbstractSparseMatrixCSC{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if solvers != [nothing]
        m = size(u, 2)
        u_colptr = u.colptr
        u_rowval = u.rowval
        x_colptr = x.colptr
        x_rowval = x.rowval
        x_nzval = x.nzval
        u_buffer = block_diagonal_solver.u_buffer
        x_buffer = block_diagonal_solver.x_buffer
        for (bi, s) ∈ zip(block_diagonal_solver.block_indices, solvers)
            block_start = first(bi)
            block_end = last(bi)
            block_size = length(bi)
            this_u_buffer = @view u_buffer[1:block_size]
            if eltype(solvers) <: LU
                this_x_buffer = this_u_buffer
            else
                this_x_buffer = @view x_buffer[1:block_size]
            end
            for col ∈ 1:m
                u_flat_start = u_colptr[col]
                u_flat_end = u_colptr[col+1] - 1
                if u_flat_end < u_flat_start
                    # Column is empty.
                    continue
                end
                if sparse_column_has_overlap(@view(u_rowval[u_flat_start:u_flat_end]), bi)
                    # Column has non-zero row entries for this block.
                    u_column = @view u[:,col]
                    for (i1, i2) ∈ enumerate(bi)
                        this_u_buffer[i1] = u_column[i2]
                    end
                    if eltype(solvers) <: LU
                        # Dense-matrix LU solver, most efficient to solve in-place
                        ldiv!(s, this_u_buffer)
                    else
                        ldiv!(this_x_buffer, s, this_u_buffer)
                    end
                    x_flat_start = x_colptr[col]
                    x_flat_end = x_colptr[col+1] - 1
                    x_col_rowval = @view x_rowval[x_flat_start:x_flat_end]
                    nxr = x_flat_end - x_flat_start + 1
                    count = max(searchsortedlast(x_col_rowval, first(bi)) - 1, 1)
                    for (i2, i1) ∈ enumerate(bi)
                        # Assume that the structural non-zero entries of `x` are enough to
                        # contain all the non-zero entries of the solve. Note that the
                        # entries in this_x_buffer that should be structurally zero might
                        # only be zero up to floating-point precision.
                        while count ≤ nxr && x_col_rowval[count] < i1
                            count += 1
                        end
                        if count > nxr
                            break
                        end
                        if i1 == x_col_rowval[count]
                            x_nzval[x_flat_start+count-1] = this_x_buffer[i2]
                            count += 1
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# Specialized implementations to be used for A^{-1}.B
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                       B::AbstractMatrix{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if solvers != [nothing]
        if eltype(solvers) <: LU
            for (bi, s, Bbuff, Bcols) ∈ zip(block_diagonal_solver.block_indices, solvers,
                                            block_diagonal_solver.B_buffers_out,
                                            block_diagonal_solver.B_column_indices)
                for (j1, j2) ∈ enumerate(Bcols), (i1, i2) ∈ enumerate(bi)
                    Bbuff[i1,j1] = B[i2,j2]
                end
                ldiv!(s, Bbuff)
                for (j2, j1) ∈ enumerate(Bcols), (i2, i1) ∈ enumerate(bi)
                    B[i1,j1] = Bbuff[i2,j2]
                end
            end
        else
            for (bi, s, Bbuff_out, Bbuff_in, Bcols) ∈
                    zip(block_diagonal_solver.block_indices, solvers,
                        block_diagonal_solver.B_buffers_out,
                        block_diagonal_solver.B_buffers_in,
                        block_diagonal_solver.B_column_indices)
                for (j1, j2) ∈ enumerate(Bcols), (i1, i2) ∈ enumerate(bi)
                    Bbuff_in[i1,j1] = B[i2,j2]
                end
                ldiv!(Bbuff_out, s, Bbuff_in)
                for (j2, j1) ∈ enumerate(Bcols), (i2, i1) ∈ enumerate(bi)
                    B[i1,j1] = Bbuff_out[i2,j2]
                end
            end
        end
    end
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                       B::AbstractSparseMatrixCSC{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if solvers != [nothing]
        if eltype(solvers) <: LU
            for (bi, s, Bbuff, Bcols) ∈ zip(block_diagonal_solver.block_indices, solvers,
                                            block_diagonal_solver.B_buffers_out,
                                            block_diagonal_solver.B_column_indices)
                B_colptr = B.colptr
                B_rowval = B.rowval
                B_nzval = B.nzval
                firstrow = first(bi)
                for (j1, j2) ∈ enumerate(Bcols)
                    first_i = B_colptr[j2]
                    last_i = B_colptr[j2+1] - 1
                    col_rv = @view B_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                    for (i1, i2) ∈ enumerate(bi)
                        while flat_i ≤ last_i && B_rowval[flat_i] < i2
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                        if B_rowval[flat_i] == i2
                            Bbuff[i1,j1] = B_nzval[flat_i]
                        end
                    end
                end
                ldiv!(s, Bbuff)
                for (j1, j2) ∈ enumerate(Bcols)
                    first_i = B_colptr[j2]
                    last_i = B_colptr[j2+1] - 1
                    col_rv = @view B_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                    for (i1, i2) ∈ enumerate(bi)
                        while flat_i ≤ last_i && B_rowval[flat_i] < i2
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                        if B_rowval[flat_i] == i2
                            B_nzval[flat_i] = Bbuff[i1,j1]
                        end
                    end
                end
            end
        else
            for (bi, s, Bbuff_out, Bbuff_in, Bcols) ∈
                    zip(block_diagonal_solver.block_indices, solvers,
                        block_diagonal_solver.B_buffers_out,
                        block_diagonal_solver.B_buffers_in,
                        block_diagonal_solver.B_column_indices)
                B_colptr = B.colptr
                B_rowval = B.rowval
                B_nzval = B.nzval
                firstrow = first(bi)
                for (j1, j2) ∈ enumerate(Bcols)
                    first_i = B_colptr[j2]
                    last_i = B_colptr[j2+1] - 1
                    col_rv = @view B_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                    for (i1, i2) ∈ enumerate(bi)
                        while flat_i ≤ last_i && B_rowval[flat_i] < i2
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                        if B_rowval[flat_i] == i2
                            Bbuff_in[i1,j1] = B_nzval[flat_i]
                        end
                    end
                end
                ldiv!(Bbuff_out, s, Bbuff_in)
                for (j1, j2) ∈ enumerate(Bcols)
                    first_i = B_colptr[j2]
                    last_i = B_colptr[j2+1] - 1
                    col_rv = @view B_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                    for (i1, i2) ∈ enumerate(bi)
                        while flat_i ≤ last_i && B_rowval[flat_i] < i2
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                        if B_rowval[flat_i] == i2
                            B_nzval[flat_i] = Bbuff_out[i1,j1]
                        end
                    end
                end
            end
        end
    end
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                       B::BlockAinvDotBSerial{T}) where T
    for (solver, block) ∈ zip(block_diagonal_solver.local_block_solver, B.blocks)
        ldiv!(solver, block)
    end
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{T},
                       B::BlockAinvDotBShared{T}) where T
    solver = block_diagonal_solver.local_block_serial_solver
    if solver !== nothing
        block = B.block
        partial_block = B.partial_block
        partial_col_range = B.partial_col_range
        partial_row_range = B.partial_row_range
        synchronize_shared = B.synchronize_shared

        # Probably more efficient to parallelise over columns in `block` than to use a
        # parallelised `ldiv!()` on the full block.
        ldiv!(solver, @view(block[:,partial_col_range]))

        synchronize_shared()

        partial_block .= @view block[partial_row_range,:]
    end
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{T},
                       B::Matrix{T}) where T
    # When not using BlockAinvDotBShared, this function will use a different
    # parallelisation than copy_B_submatrix!(), so need to synchronize.
    block_diagonal_solver.synchronize_shared()
    return ldiv!(block_diagonal_solver, B)
end
function ldiv_Bmatrix!(::MPIStaticCondensationNull, B)
    return nothing
end

function lu!(solver::MPIStaticCondensationNull, A::AbstractMatrix)
    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationNull{T},
               U::AbstractVector{T}) where T
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationNull{T},
               U::AbstractMatrix{T}) where T
    return nothing
end
function ldiv!(solver::MPIStaticCondensationNull{T},
               U::AbstractVectorOrMatrix{T}) where T
    return nothing
end

# Don't think dense-matrix version is needed?
#function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial, B::AbstractMatrix)
#    blocks = Ainv_dot_B.blocks
#    if length(blocks) == 0
#        # Nothing to do.
#        return nothing
#    end
#
#    block_rowinds = Ainv_dot_B.block_rowinds
#    block_colinds = Ainv_dot_B.block_colinds
#    for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
#        for (j1, j2) ∈ enumerate(colinds), (i1, i2) ∈ enumerate(rowinds)
#            block[i1,j1] = B[i2,j2]
#        end
#    end
#    return nothing
#end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial, B::AbstractSparseMatrixCSC,
                           B_rowinds, B_colinds)
    blocks = Ainv_dot_B.blocks
    if length(blocks) == 0
        # Nothing to do.
        return nothing
    end

    block_rowinds = Ainv_dot_B.block_rowinds
    block_colinds = Ainv_dot_B.block_colinds
    B_colptr = B.colptr
    B_rowval = B.rowval
    B_nzval = B.nzval
    for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
        block_nrow = length(rowinds)
        first_row = first(rowinds)
        for (j1, j2) ∈ enumerate(colinds)
            B_col = B_colinds[j2]
            first_i = B_colptr[B_col]
            last_i = B_colptr[B_col+1] - 1
            col_rv = @view B_rowval[first_i:last_i]
            flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
            i1 = 1
            while flat_i ≤ last_i && i1 ≤ block_nrow
                B_row = B_rowval[flat_i]
                block_global_row = B_rowinds[rowinds[i1]]
                if B_row == block_global_row
                    block[i1,j1] = B_nzval[flat_i]
                    i1 += 1
                    flat_i += 1
                elseif B_row > block_global_row
                    block[i1,j1] = 0.0
                    i1 += 1
                else
                    flat_i += 1
                end
            end
        end
    end
    return nothing
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBShared, B::AbstractSparseMatrixCSC,
                           B_rowinds, B_colinds)
    block_rowinds = Ainv_dot_B.block_rowinds
    block_colinds = Ainv_dot_B.block_colinds
    if isempty(block_rowinds) || isempty(block_colinds)
        # Nothing to do.
        return nothing
    end
    block = Ainv_dot_B.block
    partial_col_range = Ainv_dot_B.partial_col_range
    B_colptr = B.colptr
    B_rowval = B.rowval
    B_nzval = B.nzval

    block_nrow = length(block_rowinds)
    first_row = first(block_rowinds)
    for j1 ∈ partial_col_range
        j2 = block_colinds[j1]
        B_col = B_colinds[j2]
        first_i = B_colptr[B_col]
        last_i = B_colptr[B_col+1] - 1
        col_rv = @view B_rowval[first_i:last_i]
        flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
        i1 = 1
        while flat_i ≤ last_i && i1 ≤ block_nrow
            B_row = B_rowval[flat_i]
            block_global_row = B_rowinds[block_rowinds[i1]]
            if B_row == block_global_row
                block[i1,j1] = B_nzval[flat_i]
                i1 += 1
                flat_i += 1
            elseif B_row > block_global_row
                block[i1,j1] = 0.0
                i1 += 1
            else
                flat_i += 1
            end
        end
    end

    return nothing
end
@inline function copy_B_submatrix!(Ainv_dot_B::Union{BlockAinvDotBSerial,BlockAinvDotBShared},
                                   B::SubArray)
    return copy_B_submatrix!(Ainv_dot_B, B.parent, B.indices[1], B.indices[2])
end

# copy_C_submatrix!() is identical to copy_B_submatrix!(), but keep as a separate function
# instead of having a single implementation for both in case we want to experiment with
# using a transposed representation of the C blocks at some point.
# Don't think dense-matrix version is needed?
#function copy_C_submatrix!(block_C::BlockCSerial, C::AbstractMatrix)
#    blocks = block_C.blocks
#    if length(blocks) == 0
#        # Nothing to do.
#        return nothing
#    end
#
#    block_rowinds = block_C.block_rowinds
#    block_colinds = block_C.block_colinds
#    for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
#        for (j1, j2) ∈ enumerate(colinds), (i1, i2) ∈ enumerate(rowinds)
#            block[i1,j1] = C[i2,j2]
#        end
#    end
#    return nothing
#end
function copy_C_submatrix!(block_C::BlockCSerial, C::AbstractSparseMatrixCSC, C_rowinds,
                           C_colinds)
    blocks = block_C.blocks
    if length(blocks) == 0
        # Nothing to do.
        return nothing
    end

    block_rowinds = block_C.block_rowinds
    block_colinds = block_C.block_colinds
    C_colptr = C.colptr
    C_rowval = C.rowval
    C_nzval = C.nzval
    for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
        block_nrow = length(rowinds)
        first_row = first(rowinds)
        for (j1, j2) ∈ enumerate(colinds)
            C_col = C_colinds[j2]
            first_i = C_colptr[C_col]
            last_i = C_colptr[C_col+1] - 1
            col_rv = @view C_rowval[first_i:last_i]
            flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
            i1 = 1
            while flat_i ≤ last_i && i1 ≤ block_nrow
                C_row = C_rowval[flat_i]
                block_global_row = C_rowinds[rowinds[i1]]
                if C_row == block_global_row
                    block[i1,j1] = C_nzval[flat_i]
                    i1 += 1
                    flat_i += 1
                elseif C_row > block_global_row
                    block[i1,j1] = 0.0
                    i1 += 1
                else
                    flat_i += 1
                end
            end
        end
    end
    return nothing
end
function copy_C_submatrix!(block_C::BlockCShared, C::AbstractSparseMatrixCSC, C_rowinds,
                           C_colinds)
    block_rowinds = block_C.block_rowinds
    block_colinds = block_C.block_colinds
    if isempty(block_rowinds) || isempty(block_colinds)
        # Nothing to do.
        return nothing
    end
    block = block_C.block
    C_colptr = C.colptr
    C_rowval = C.rowval
    C_nzval = C.nzval

    block_nrow = length(block_rowinds)
    first_row = first(block_rowinds)
    for (j1, j2) ∈ enumerate(block_colinds)
        C_col = C_colinds[j2]
        first_i = C_colptr[C_col]
        last_i = C_colptr[C_col+1] - 1
        col_rv = @view C_rowval[first_i:last_i]
        flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
        i1 = 1
        while flat_i ≤ last_i && i1 ≤ block_nrow
            C_row = C_rowval[flat_i]
            block_global_row = C_rowinds[block_rowinds[i1]]
            if C_row == block_global_row
                block[i1,j1] = C_nzval[flat_i]
                i1 += 1
                flat_i += 1
            elseif C_row > block_global_row
                block[i1,j1] = 0.0
                i1 += 1
            else
                flat_i += 1
            end
        end
    end
    return nothing
end
@inline function copy_C_submatrix!(block_C::Union{BlockCSerial,BlockCShared}, C::SubArray)
    return copy_C_submatrix!(block_C, C.parent, C.indices[1], C.indices[2])
end

# Note that combining all the contributions to C_dot_Ainv_dot_B from different processes
# is taken care of by MPISchurComplements.
function mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::NamedTuple, C::BlockCSerial,
                           Ainv_dot_B::BlockAinvDotBSerial)
    C_blocks = C.blocks
    if length(C_blocks) == 0
        # Nothing to do.
        return nothing
    end

    mul_blocks = C.right_multiplication_buffer_blocks
    Ainv_dot_B_blocks = Ainv_dot_B.blocks
    block_output_inds = C.block_rowinds # This is identical to Ainv_dot_B.block_colinds
    buffer_position = C.buffer_position

    colptr = C_dot_Ainv_dot_B.colptr
    rowval = C_dot_Ainv_dot_B.rowval
    C_dot_Ainv_dot_B_storage = C_dot_Ainv_dot_B.storage

    if buffer_position !== nothing
        block_synchronize_shared = C.block_synchronize_shared

        nzval = @view C_dot_Ainv_dot_B_storage[buffer_position,:]

        # Initialise buffer to zero.
        matrix_init_range = C.matrix_init_range
        for i ∈ matrix_init_range
            nzval[i] = 0.0
        end

        block_synchronize_shared()

        for (mb, Cb, AiBb, output_inds) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks,
                                              block_output_inds)
            mul!(mb, Cb, AiBb, -1.0, 0.0)

            # Copy result from mb into the sparse output buffer C_dot_Ainv_dot_B.
            first_row = first(output_inds)
            nrows = length(output_inds)
            for (j, col) ∈ enumerate(output_inds)
                first_i = colptr[col]
                last_i = colptr[col+1] - 1
                col_rv = @view rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
                i = 1
                while flat_i ≤ last_i && i ≤ nrows
                    if rowval[flat_i] == output_inds[i]
                        nzval[flat_i] += mb[i,j]
                        flat_i += 1
                        i += 1
                    else
                        # rowval[flat_i] must be less than output_inds[i]
                        flat_i += 1
                    end
                end
            end
        end
    else
        # When choosing the column by block_hypercube_position, there are no overlaps, so
        # we can directly set entries, instead of adding to them, and so do not need to
        # zero-initialise the output buffer.
        block_hypercube_positions = C.block_hypercube_positions
        for (mb, Cb, AiBb, output_inds, bhp) ∈ zip(mul_blocks, C_blocks,
                                                   Ainv_dot_B_blocks, block_output_inds,
                                                   block_hypercube_positions)
            nzval = @view C_dot_Ainv_dot_B_storage[bhp,:]

            mul!(mb, Cb, AiBb, -1.0, 0.0)

            # Copy result from mb into the sparse output buffer C_dot_Ainv_dot_B.
            first_row = first(output_inds)
            nrows = length(output_inds)
            for (j, col) ∈ enumerate(output_inds)
                first_i = colptr[col]
                last_i = colptr[col+1] - 1
                col_rv = @view rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
                i = 1
                while flat_i ≤ last_i && i ≤ nrows
                    if rowval[flat_i] == output_inds[i]
                        nzval[flat_i] = mb[i,j]
                        flat_i += 1
                        i += 1
                    else
                        # rowval[flat_i] must be less than output_inds[i]
                        flat_i += 1
                    end
                end
            end
        end
    end
    return nothing
end
function mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::NamedTuple, C::BlockCShared,
                           Ainv_dot_B::BlockAinvDotBShared)
    C_block = C.block
    mul_block = C.right_multiplication_buffer_block
    block_output_inds = C.block_rowinds
    block_output_colinds = C.block_right_multiplication_output_colinds
    Ainv_dot_B_block = Ainv_dot_B.block
    matrix_init_range = C.matrix_init_range

    if isempty(block_output_inds) || isempty(block_output_colinds)
        return nothing
    end

    colptr = C_dot_Ainv_dot_B.colptr
    rowval = C_dot_Ainv_dot_B.rowval
    nzval = @view C_dot_Ainv_dot_B.storage[C.block_hypercube_position,:]

    if matrix_init_range !== nothing
        # C.block_hypercube_position is actually the subgroup index, as there are fewer
        # subgroups than 2^d where d is the number of dimensions, and each subgroup writes
        # to a single column of the output buffer.

        block_synchronize_shared = C.block_synchronize_shared

        # Initialise buffer to zero.
        for i ∈ matrix_init_range
            nzval[i] = 0.0
        end

        block_synchronize_shared()

        if !isempty(block_output_inds)
            mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)

            # Copy result from mul_block into the sparse output buffer C_dot_Ainv_dot_B.
            first_row = first(block_output_inds)
            nrows = length(block_output_inds)
            for (j, col) ∈ enumerate(block_output_colinds)
                first_i = colptr[col]
                last_i = colptr[col+1] - 1
                col_rv = @view rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
                i = 1
                while flat_i ≤ last_i && i ≤ nrows
                    if rowval[flat_i] == block_output_inds[i]
                        nzval[flat_i] += mul_block[i,j]
                        flat_i += 1
                        i += 1
                    else
                        # rowval[flat_i] must be less than block_output_inds[i].
                        flat_i += 1
                    end
                end
            end
        end
    elseif !isempty(block_output_inds)
        # Output buffer columns are divided by 'hypercube position' so there are no
        # overlaps, and we can directly set entries, instead of adding to them, and so do
        # not need to zero-initialise the output buffer.
        mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)

        # Copy result from mul_block into the sparse output buffer C_dot_Ainv_dot_B.
        first_row = first(block_output_inds)
        nrows = length(block_output_inds)
        for (j, col) ∈ enumerate(block_output_colinds)
            first_i = colptr[col]
            last_i = colptr[col+1] - 1
            col_rv = @view rowval[first_i:last_i]
            flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
            i = 1
            while flat_i ≤ last_i && i ≤ nrows
                if rowval[flat_i] == block_output_inds[i]
                    nzval[flat_i] = mul_block[i,j]
                    flat_i += 1
                    i += 1
                else
                    # rowval[flat_i] must be less than block_output_inds[i].
                    flat_i += 1
                end
            end
        end
    end
    return nothing
end

function lu!(solver::MPIStaticCondensationParallel, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        local_top_vector_indices = solver.local_top_vector_indices
        all_local_top_vector_a_block_indices = solver.all_local_top_vector_a_block_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        a = @view A[all_local_top_vector_a_block_indices,all_local_top_vector_a_block_indices]
        b = @view A[local_top_vector_indices,local_bottom_vector_indices]
        c = @view A[local_bottom_vector_indices,local_top_vector_indices]
        d = @view A[local_bottom_vector_indices,local_bottom_vector_indices]
        update_schur_complement!(solver.local_block_solver, a, b, c, d)
    end
    return nothing
end

function Ainv_dot_B_dot_y!(top_vec_buffer::AbstractVector,
                           Ainv_dot_B::BlockAinvDotBSerial, global_y::AbstractVector)
    blocks = Ainv_dot_B.blocks
    if length(blocks) == 0
        # Nothing to do.
        return nothing
    end

    for (vec_buffer_in, vec_buffer_out, rowinds, colinds, block) ∈
            zip(Ainv_dot_B.vector_buffer_blocks_in, Ainv_dot_B.vector_buffer_blocks_out,
                Ainv_dot_B.block_rowinds, Ainv_dot_B.block_colinds, blocks)
        for (i1, i2) ∈ enumerate(colinds)
            vec_buffer_in[i1] = global_y[i2]
        end
        mul!(vec_buffer_out, block, vec_buffer_in)
        for (i2, i1) ∈ enumerate(rowinds)
            top_vec_buffer[i1] = vec_buffer_out[i2]
        end
    end
    return nothing
end
function Ainv_dot_B_dot_y!(top_vec_buffer::AbstractVector,
                           Ainv_dot_B::BlockAinvDotBShared, global_y::AbstractVector)
    partial_block = Ainv_dot_B.partial_block
    vector_buffer_block_in = Ainv_dot_B.vector_buffer_block_in
    vector_buffer_block_out = Ainv_dot_B.vector_buffer_block_out
    block_partial_rowinds = Ainv_dot_B.block_partial_rowinds
    block_partial_colinds = Ainv_dot_B.block_partial_colinds
    partial_col_range = Ainv_dot_B.partial_col_range
    synchronize_shared = Ainv_dot_B.synchronize_shared

    for (i1, i2) ∈ zip(partial_col_range, block_partial_colinds)
        vector_buffer_block_in[i1] = global_y[i2]
    end
    synchronize_shared()

    mul!(vector_buffer_block_out, partial_block, vector_buffer_block_in)
    for (i2, i1) ∈ enumerate(block_partial_rowinds)
        top_vec_buffer[i1] = vector_buffer_block_out[i2]
    end
    return nothing
end

function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCSerial,
                               Ainv_dot_u::AbstractVector)

    blocks = C.blocks
    vector_range = C.vector_range
    vector_intermediate_buffer = C.vector_intermediate_buffer
    buffer_position = C.buffer_position
    synchronize_shared = C.synchronize_shared

    if buffer_position !== nothing
        block_synchronize_shared = C.block_synchronize_shared
        vector_init_range = C.vector_init_range
        vector_intermediate_buffer_local = @view vector_intermediate_buffer[buffer_position,:]

        # Initialise buffer to zero.
        for i ∈ vector_init_range
            vector_intermediate_buffer_local[i] = 0.0
        end

        block_synchronize_shared()

        if length(blocks) > 0
            for (vec_buffer_in, vec_buffer_out, rowinds, colinds, block) ∈
                    zip(C.vector_buffer_blocks_in, C.vector_buffer_blocks_out, C.block_rowinds,
                        C.block_colinds, blocks)
                for (i1, i2) ∈ enumerate(colinds)
                    vec_buffer_in[i1] = Ainv_dot_u[i2]
                end
                mul!(vec_buffer_out, block, vec_buffer_in)
                for (i2, i1) ∈ enumerate(rowinds)
                    vector_intermediate_buffer_local[i1] -= vec_buffer_out[i2]
                end
            end
        end
    else
        # When choosing the column by block_hypercube_position, there are no overlaps, so
        # we can directly set entries, instead of adding to them, and so do not need to
        # zero-initialise the intermediate buffer.
        block_hypercube_positions = C.block_hypercube_positions
        if length(blocks) > 0
            for (vec_buffer_in, vec_buffer_out, rowinds, colinds, block, bhp) ∈
                    zip(C.vector_buffer_blocks_in, C.vector_buffer_blocks_out, C.block_rowinds,
                        C.block_colinds, blocks, block_hypercube_positions)
                vector_intermediate_buffer_local = @view vector_intermediate_buffer[bhp,:]
                for (i1, i2) ∈ enumerate(colinds)
                    vec_buffer_in[i1] = Ainv_dot_u[i2]
                end
                mul!(vec_buffer_out, block, vec_buffer_in)
                for (i2, i1) ∈ enumerate(rowinds)
                    vector_intermediate_buffer_local[i1] = -vec_buffer_out[i2]
                end
            end
        end
    end

    synchronize_shared()

    # Sum contributions from all processes into the output.
    if !isempty(vector_range)
        @views sum!(C_dot_Ainv_dot_u[vector_range]', vector_intermediate_buffer[:,vector_range])
    end

    return nothing
end
function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCShared,
                               Ainv_dot_u::AbstractVector)

    block = C.block
    vector_range = C.vector_range
    vector_intermediate_buffer = C.vector_intermediate_buffer
    synchronize_shared = C.synchronize_shared
    vector_intermediate_buffer_local = C.vector_intermediate_buffer_local
    vector_init_range = C.vector_init_range
    vec_buffer_block_in = C.vector_buffer_block_in
    vec_buffer_block_out = C.vector_buffer_block_out
    block_rowinds = C.block_rowinds
    partial_block_colinds = C.partial_block_colinds
    partial_col_range = C.partial_col_range
    block_synchronize_shared = C.block_synchronize_shared
    synchronize_shared = C.synchronize_shared

    if vector_init_range !== nothing
        # C.block_hypercube_position is actually the subgroup index, as there are fewer
        # subgroups than 2^d where d is the number of dimensions, and each subgroup writes
        # to a single column of the intermediate buffer.

        # Initialise buffer to zero.
        for i ∈ vector_init_range
            vector_intermediate_buffer_local[i] = 0.0
        end

        for (i1, i2) ∈ zip(partial_col_range, partial_block_colinds)
            vec_buffer_block_in[i1] = Ainv_dot_u[i2]
        end

        block_synchronize_shared()

        mul!(vec_buffer_block_out, block, vec_buffer_block_in)
        for (i2, i1) ∈ enumerate(block_rowinds)
            vector_intermediate_buffer_local[i1] -= vec_buffer_block_out[i2]
        end
    else
        # Output buffer columns are divided by 'hypercube position' so there are no
        # overlaps, and we can directly set entries, instead of adding to them, and so do
        # not need to zero-initialise the intermediate buffer.
        for (i1, i2) ∈ zip(partial_col_range, partial_block_colinds)
            vec_buffer_block_in[i1] = Ainv_dot_u[i2]
        end

        block_synchronize_shared()

        mul!(vec_buffer_block_out, block, vec_buffer_block_in)
        for (i2, i1) ∈ enumerate(block_rowinds)
            vector_intermediate_buffer_local[i1] = -vec_buffer_block_out[i2]
        end
    end

    synchronize_shared()

    # Sum contributions from all processes into the output.
    if !isempty(vector_range)
        @views sum!(C_dot_Ainv_dot_u[vector_range]', vector_intermediate_buffer[:,vector_range])
    end

    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        # It is slightly faster to copy the data to/from local buffers than to use @view
        # with Vector{Int64} indices.
        partial_local_top_vector_a_block_indices = solver.partial_local_top_vector_a_block_indices
        partial_a_block_sub_selection_indices = solver.partial_a_block_sub_selection_indices
        this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
        this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
        this_shared_local_bottom_vector_no_overlap_indices = solver.this_shared_local_bottom_vector_no_overlap_indices
        this_shared_local_bottom_sub_selection_no_overlap_indices = solver.this_shared_local_bottom_sub_selection_no_overlap_indices
        this_shared_local_bottom_vector_repeat_indices = solver.this_shared_local_bottom_vector_repeat_indices
        this_shared_local_bottom_periodic_pairs = solver.this_shared_local_bottom_periodic_pairs
        u = solver.u_buffer
        v = solver.v_buffer
        # Use the a_block_indices here so that no shared-memory synchronization is needed
        # before the ldiv!() call for the A subblock with the
        # BlockDiagonalSolverSerial/BlockDiagonalSolverShared inside the
        # MPISchurComplement ldiv!().
        for (i1, i2) ∈ zip(partial_a_block_sub_selection_indices, partial_local_top_vector_a_block_indices)
            u[i1] = U[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
            # This loop uses 'no overlap' indices
            # (`this_shared_local_bottom_vector_no_overlap_indices`) because when there
            # are periodic dimensions, at the top level (and only the top level, not any
            # intermediate levels) the right-hand-side entries need to be taken only from
            # the non-repeated points, with the repeated points being zero-ed out.
            v[i1] = U[i2]
        end
        for i ∈ this_shared_local_bottom_vector_repeat_indices
            # Zero out repeated points at the top level
            v[i] = 0.0
        end
        if solver.has_periodic
            for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                # At the bottom level, need to add any contributions that the top and
                # intermediate levels have added to repeated points into the non-repeated
                # points.
                v[i1] += U[i2]
            end
        end
        ldiv!(u, v, solver.local_block_solver, u, v)
        for (i1, i2) ∈ zip(partial_local_top_vector_a_block_indices, partial_a_block_sub_selection_indices)
            X[i1] = u[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices, this_shared_local_bottom_sub_selection_indices)
            X[i1] = v[i2]
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationParallel{T}, U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        # It is slightly faster to copy the data to/from local buffers than to use @view
        # with Vector{Int64} indices.
        partial_local_top_vector_a_block_indices = solver.partial_local_top_vector_a_block_indices
        partial_a_block_sub_selection_indices = solver.partial_a_block_sub_selection_indices
        this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
        this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
        this_shared_local_bottom_vector_no_overlap_indices = solver.this_shared_local_bottom_vector_no_overlap_indices
        this_shared_local_bottom_sub_selection_no_overlap_indices = solver.this_shared_local_bottom_sub_selection_no_overlap_indices
        this_shared_local_bottom_vector_repeat_indices = solver.this_shared_local_bottom_vector_repeat_indices
        this_shared_local_bottom_periodic_pairs = solver.this_shared_local_bottom_periodic_pairs
        u = solver.u_buffer
        v = solver.v_buffer
        # Use the a_block_indices here so that no shared-memory synchronization is needed
        # before the ldiv!() call for the A subblock with the
        # BlockDiagonalSolverSerial/BlockDiagonalSolverShared inside the
        # MPISchurComplement ldiv!().
        for (i1, i2) ∈ zip(partial_a_block_sub_selection_indices, partial_local_top_vector_a_block_indices)
            u[i1] = U[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
            # This loop uses 'no overlap' indices
            # (`this_shared_local_bottom_vector_no_overlap_indices`) because when there
            # are periodic dimensions, at the top level (and only the top level, not any
            # intermediate levels) the right-hand-side entries need to be taken only from
            # the non-repeated points, with the repeated points being zero-ed out.
            v[i1] = U[i2]
        end
        for i ∈ this_shared_local_bottom_vector_repeat_indices
            # Zero out repeated points at the top level
            v[i] = 0.0
        end
        if solver.has_periodic
            for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                # At the bottom level, need to add any contributions that the top and
                # intermediate levels have added to repeated points into the non-repeated
                # points.
                v[i1] += U[i2]
            end
        end
        ldiv!(u, v, solver.local_block_solver, u, v)
        for (i1, i2) ∈ zip(partial_local_top_vector_a_block_indices, partial_a_block_sub_selection_indices)
            U[i1] = u[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices, this_shared_local_bottom_sub_selection_indices)
            U[i1] = v[i2]
        end
    end
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        for (this_X, this_U) ∈ zip(eachcol(X), eachcol(U))
            ldiv!(this_X, solver, this_U)
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationParallel{T}, U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        for this_U ∈ eachcol(U)
            ldiv!(solver, this_U)
        end
    end
    return nothing
end

end
