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
using MPI
using MPISchurComplements
using Primes
using SparseArrays
using SparseArrays: FixedSparseCSC, AbstractSparseMatrixCSC
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

struct MPIStaticCondensationSerialSparse{Tf<:AbstractFloat,Ti<:Integer,Tndi,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    local_block_solver::SparseArrays.UMFPACK.UmfpackLU{Tf,Ti}
    U_buffer::Vector{Tf}
    X_buffer::Vector{Tf}
    non_duplicate_indices::Tndi
    periodic_index_pairs::Matrix{Ti}
    timer::Ttimer
    check_lu::Bool
end
Base.size(Alu::MPIStaticCondensationSerialSparse) = size(Alu.local_block_solver)
Base.size(Alu::MPIStaticCondensationSerialSparse, d::Integer) = size(Alu)[d]

struct MPIStaticCondensationSerialDense{Tf<:AbstractFloat,Ti<:Integer,Tndi,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    local_block_solver::LU{Tf,Matrix{Tf},Vector{Ti}}
    X_buffer::Vector{Tf}
    non_duplicate_indices::Tndi
    periodic_index_pairs::Matrix{Ti}
    timer::Ttimer
    check_lu::Bool
end
Base.size(Alu::MPIStaticCondensationSerialDense) = size(Alu.local_block_solver)
Base.size(Alu::MPIStaticCondensationSerialDense, d::Integer) = size(Alu)[d]

struct MPIStaticCondensationParallel{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:MPISchurComplement{Tf},Tranget,Trangeatab,Trangeabs,Trangeb,Trangebs,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_top_vector_indices::Tranget
    all_local_top_vector_a_block_indices::Trangeatab
    all_a_block_sub_selection_indices::Trangeabs
    local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_sub_selection_indices::Trangebs
    u_buffer::Vector{Tf}
    v_buffer::Vector{Tf}
    timer::Ttimer
end
Base.size(Alu::MPIStaticCondensationParallel) = (Alu.n, Alu.n)
Base.size(Alu::MPIStaticCondensationParallel, d::Integer) = size(Alu)[d]

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolver{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},Nothing},Trange}
    n::Ti
    local_block_solver::Vector{Tsolver}
    block_indices::Trange
    lu_selection_indices::Trange
    x_buffer::Vector{Tf}
    u_buffer::Vector{Tf}
    function BlockDiagonalSolver{Tf}(n::Ti, block_indices, lu_selection_indices) where {Tf, Ti <: Integer}
        # Don't need a solver for any empty entries in block_indices, as these blocks have
        # no interior points.
        block_indices = [bi for bi ∈ block_indices if !isempty(bi)]
        block_sizes = [length(bi) for bi ∈ block_indices]
        block_size = maximum(block_sizes; init=0)
        function get_identity(bs)
            identity = spzeros(Tf, block_size, block_size)
            copyto!(identity, I)
            return identity
        end
        if block_size > 0
            local_block_solver = [lu(get_identity(length(bi))) for bi ∈ block_indices]
        else
            local_block_solver = [nothing]
        end
        x_buffer = fill(NaN, block_size)
        u_buffer = fill(NaN, block_size)
        return new{Tf,Ti,eltype(local_block_solver),typeof(block_indices)}(
                   n, local_block_solver, block_indices, lu_selection_indices, x_buffer,
                   u_buffer)
    end
end
Base.size(Alu::BlockDiagonalSolver) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolver, d::Integer) = size(Alu)[d]

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
                       #has_lower_boundary::Bool, has_upper_boundary::Bool,
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

        #return new{Ti}(n, n_local, nelement, ngrid, nrank, irank, global_inds, periodic,
        #               has_lower_boundary, has_upper_boundary, remove_boundaries)
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
    # As this function creates the top-level Dimension, it always includes boundary
    # points.
    #return Dimension(; nelement, ngrid, nrank, irank, periodic, has_lower_boundary=true,
    #                 has_upper_boundary=true, remove_boundaries)
    return Dimension(; nelement, ngrid, nrank, irank, periodic, remove_boundaries)
end

# Find the index of the last instance of the maximum in `x`.
# This function is only used in `pick_dimension_to_split` and called with small
# collections of integers, so efficiency is not important.
function last_argmax(x)
    i = argmax(reverse(collect(x)))
    return length(x) - i + 1
end

function pick_dimension_to_split(dimensions::Vector{<:Dimension}, n_groups::Integer,
                                 optimise_schur_complement_size::Bool)
    if all(d.nelement == 1 for d ∈ dimensions)
        error("All dimensions contain one element, and so cannot be split. This probably "
              * "means too many MPI processes are being used for the size of the grid.")
    end
    if n_groups ≤ 1
        error("Cannot split a dimension when n_groups≤1. Got n_groups=$n_groups.")
    end

    distributed_dims = findall(d -> d.nrank > 1, dimensions)
    if optimise_schur_complement_size
        if !isempty(distributed_dims)
            # When a distributed dimension is being split, require that the dimension can
            # be split exactly by n_groups - this is not in principle strictly necessary,
            # but if it is not true then load balance becomes tricky because, for example,
            # some shared-memory block may own parts of two decoupled sections of the 'A
            # matrix' (top left block of the 2x2 block matrix), and would then have to
            # participate in the solves for both 'a blocks'.
            candidate_dimensions = [i for i ∈ distributed_dims
                                    if dimensions[i].nrank % n_groups == 0]
            idim = last_argmax(d.n for d ∈ dimensions[candidate_dimensions])
            return candidate_dimensions[idim]
        else
            dims_to_divide = findall(d.nelement > 1 for d ∈ dimensions)
            idim = last_argmax(d.n for d ∈ dimensions[dims_to_divide])
            return dims_to_divide[idim]
        end
    else
        if !isempty(distributed_dims)
            # When dimensions are distributed, splits must be on block boundaries, not
            # just on element boundaries.
            distributed_dims_to_divide = findall(d.nrank % n_groups == 0
                                                 for d ∈ dimensions[distributed_dims])
            dims_to_divide = distributed_dims[distributed_dims_to_divide]
            if !isempty(dims_to_divide)
                idim = last_argmax(d.n for d ∈ dimensions[dims_to_divide])
                return dims_to_divide[idim]
            else
                idim = last_argmax(d.n for d ∈ dimensions[distributed_dims])
                return distributed_dims[idim]
            end
        else
            dims_to_divide = findall(d.nelement % n_groups == 0 for d ∈ dimensions)
            if !isempty(dims_to_divide)
                idim = last_argmax(d.n for d ∈ dimensions[dims_to_divide])
                return dims_to_divide[idim]
            else
                dims_to_divide = findall(d.nelement > 1 for d ∈ dimensions)
                idim = last_argmax(d.n for d ∈ dimensions[dims_to_divide])
                return dims_to_divide[idim]
            end
        end
    end
    error("Case not handled - this should never happen")
end

function get_local_flattened_index(indices::CartesianIndex, dim_sizes::Tuple)
    flat_i = 0
    for (i, n) ∈ zip(reverse(Tuple(indices)), reverse(dim_sizes))
        flat_i = flat_i * n + i - 1
    end
    # So far constructed a 0-based index, so convert to 1-based.
    flat_i += 1
    return flat_i
end

function get_local_ind_slice(dimensions::Vector{<:Dimension}, dim_to_slice::Integer,
                             slice_inds::OrdinalRange{<:Integer})
    dim_sizes = Tuple(d.n_local for d ∈ dimensions)
    result_ranges = Tuple(i == dim_to_slice ? slice_inds : 1:dim_sizes[i]
                          for i ∈ 1:length(dimensions))
    return get_local_ind_slice(Tuple(dimensions), dim_to_slice, slice_inds, dim_sizes,
                               result_ranges)
end
function get_local_ind_slice(dimensions::Tuple, dim_to_slice::Integer,
                             slice_inds::OrdinalRange{<:Integer}, dim_sizes::Tuple,
                             result_ranges::Tuple)
    inds = fill(eltype(slice_inds)(-1), prod(length(r) for r ∈ result_ranges))
    for (local_flat_i, i) ∈ enumerate(CartesianIndices(result_ranges))
        inds[local_flat_i] = get_local_flattened_index(i, dim_sizes)
    end
    return inds
end

function get_local_ind_slice(dimensions::Vector{<:Dimension}, dim_to_slice::Integer,
                             slice_inds::Vector{<:Integer})
    result_ranges_left = Tuple(1:dimensions[i].n_local for i ∈ 1:dim_to_slice-1)
    result_ranges_right = Tuple(1:dimensions[i].n_local for i ∈ dim_to_slice+1:length(dimensions))
    dim_sizes = Tuple(d.n_local for d ∈ dimensions)
    return get_local_ind_slice(slice_inds, result_ranges_left, result_ranges_right,
                               dim_sizes)
end
function get_local_ind_slice(slice_inds::Vector{<:Integer}, result_ranges_left::Tuple,
                             result_ranges_right::Tuple, dim_sizes::Tuple)
    # When `slice_inds` is a Vector, not an OrdinalRange, cannot use CartesianIndices on
    # it, so have to do more complicated loops.
    inds = fill(eltype(slice_inds)(-1),
                prod(length(r) for r ∈ result_ranges_left; init=1) * length(slice_inds) *
                prod(length(r) for r ∈ result_ranges_right; init=1))
    local_flat_i = 0
    for i_right ∈ CartesianIndices(result_ranges_right), i_slice ∈ slice_inds,
            i_left ∈ CartesianIndices(result_ranges_left)
        local_flat_i += 1
        indices = CartesianIndex(i_left, i_slice, i_right)
        inds[local_flat_i] = get_local_flattened_index(indices, dim_sizes)
    end
    return inds
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
                                      inds::Vector{<:Integer}, n_tuple)
    if !any(d.periodic for d ∈ dimensions)
        # No periodic dimensions to account for.
        return copy(inds)
    end
    periodic_inds = similar(inds)
    cartinds = CartesianIndices(n_tuple)
    for (i, ind) ∈ enumerate(inds)
        cart_i = cartinds[ind]
        global_i = 0
        for (d, di, n) ∈ zip(reverse(dimensions), reverse(Tuple(cart_i)), reverse(n_tuple))
            if di == n && d.periodic
                di = 1
            end
            global_i = global_i * n + di - 1
        end
        global_i += 1
        periodic_inds[i] = global_i
    end
    return periodic_inds
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
    n_colptr = allocate_shared_int(1)
    n_rowval = allocate_shared_int(1)
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

        MPI.Barrier(shared_comm)

        colptr = allocate_shared_int(n_colptr[])
        rowval = allocate_shared_int(n_rowval[])
        nzval = allocate_shared_float(n_rowval[])

        colptr .= cp
        rowval .= rv
        nzval .= 0.0
    else
        MPI.Barrier(shared_comm)

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
#MPI.Comm_split(comm::FakeComm, color, key) = comm
MPI.Allreduce!(buff, op, comm::FakeComm) = buff # This is not a sensible result!
MPI.Bcast!(buff, comm::FakeComm; root=nothing) = buff # This is not a sensible result!
MPI.Barrier(comm::FakeComm) = nothing

#@kwdef struct LevelInfo{Ti,Tasub,Tcomm<:Union{MPI.Comm,FakeComm},Tdcomm<:Union{MPI.Comm,Nothing,FakeComm}}
@kwdef struct LevelInfo{Ti,Tcomm<:Union{MPI.Comm,FakeComm}}
    #level_dimensions::Vector{Dimension{Ti}}
    block_sizes::Vector{Ti}
    global_size::Ti
    global_bottom_vector_size::Ti
    top_vector_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    all_local_top_vector_a_block_indices::Vector{Ti}
    local_top_vector_a_block_indices::Vector{Vector{Ti}}
    all_a_block_sub_selection_indices::Vector{Ti}
    a_block_sub_selection_indices::Vector{Vector{Ti}}
    a_block_lu_selection_indices::Vector{Vector{Ti}}
    bottom_vector_indices::Vector{Ti}
    local_bottom_vector_indices::Vector{Ti}
    #level_comm::Tcomm
    #level_distributed_comm::Tdcomm
    level_shared_comm::Tcomm
end

function split_matrix(dimensions::Vector{<:Dimension}, level_indices::Vector{Ti},
                      block_sizes::Vector{Ti}, global_size::Ti,
                      distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
                      shared_comm::Union{MPI.Comm,FakeComm}) where Ti <: Integer
    if length(dimensions) != length(block_sizes)
        error("dimensions and block_sizes should be the same length")
    end
    if shared_comm == MPI.COMM_NULL
        # This processor does no work on this level, so just fill level_info with dummy
        # values.
        return LevelInfo(; block_sizes, global_size=0, global_bottom_vector_size=0,
                         top_vector_indices=Ti[], local_top_vector_indices=Ti[],
                         all_local_top_vector_a_block_indices=Ti[],
                         local_top_vector_a_block_indices=Vector{Ti}[],
                         all_a_block_sub_selection_indices=Ti[],
                         a_block_sub_selection_indices=Vector{Ti}[],
                         a_block_lu_selection_indices=Vector{Ti}[],
                         bottom_vector_indices=Ti[], local_bottom_vector_indices=Ti[],
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
    blocks_per_proc = (total_nblocks + shared_comm_size - 1) ÷ shared_comm_size
    this_proc_blocks = shared_comm_rank*blocks_per_proc+1:min((shared_comm_rank+1)*blocks_per_proc,total_nblocks)
    block_interior_indices = Vector{Vector{Ti}}(undef, length(this_proc_blocks))
    function get_block_interior_points!(bi, b)
        iblock = zeros(Ti, length(dimensions))
        temp = b - 1
        for (idim, nb) ∈ enumerate(nblocks_list)
            temp, iblock[idim] = divrem(temp, nb)
        end
        iblock .+= 1
        this_bii = Ti[]
        block_interior_indices[bi] = this_bii
        function get_interior_from_dim!(this_dim, flat_i)
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
                get_interior_from_dim!(next_dim, flat_i + i - 1)
            end
            return nothing
        end
        get_interior_from_dim!(length(dimensions), 0)
        return nothing
    end
    for (bi, b) ∈ enumerate(this_proc_blocks)
        get_block_interior_points!(bi, b)
    end
    for bii ∈ block_interior_indices
        sort!(bii)
        unique!(bii)
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
    for (this_a_block_sub_selection_indices, this_a_block_ldiv_selection_indices) ∈ zip(a_block_sub_selection_indices, a_block_lu_selection_indices)
        i_count = 1
        bi_count = 1
        while (i_count ≤ length(all_a_block_sub_selection_indices)
               && bi_count ≤ length(this_a_block_sub_selection_indices))
            i = all_a_block_sub_selection_indices[i_count]
            bi = this_a_block_sub_selection_indices[bi_count]
            if i == bi
                push!(this_a_block_ldiv_selection_indices, i_count)
                i_count += 1
                bi_count += 1
            elseif i < bi
                i_count += 1
            else
                bi_count += 1
            end
        end
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

    global_top_vector_indices = apply_periodicity_to_indices(dimensions, interior_indices)
    global_bottom_vector_indices = apply_periodicity_to_indices(dimensions,
                                                                boundary_indices)

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
    count = 1
    n = length(level_indices)
    while (t_count ≤ nt || a_count ≤ na || b_count ≤ nb) && count ≤ n
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
        if b_count ≤ nb && boundary_indices[b_count] == level_indices[count]
            push!(local_bottom_vector_indices, count)
            b_count += 1
        end
        count += 1
    end
    if t_count != nt + 1 || a_count != na + 1 || b_count != nb + 1
        error("Did not find all indices in search. t_count=$t_count while nt+1=$(nt+1). "
              * "a_count=$a_count while na+1=$(na+1), "
              * "b_count=$b_count while nb+1=$(nb+1).")
    end
    a_block_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
    for (abi, lti) ∈ zip(a_block_indices, local_top_vector_a_block_indices)
        count = 1
        a_count = 1
        na = length(lti)
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

    return LevelInfo(; block_sizes, global_size, global_bottom_vector_size,
                     top_vector_indices=global_top_vector_indices,
                     local_top_vector_indices=local_top_vector_indices,
                     all_local_top_vector_a_block_indices=all_a_block_indices,
                     local_top_vector_a_block_indices=a_block_indices,
                     all_a_block_sub_selection_indices=all_a_block_sub_selection_indices,
                     a_block_sub_selection_indices=a_block_sub_selection_indices,
                     a_block_lu_selection_indices=a_block_lu_selection_indices,
                     bottom_vector_indices=global_bottom_vector_indices,
                     local_bottom_vector_indices=local_bottom_vector_indices,
                     level_shared_comm=shared_comm)
end

# Use `FakeComm` values for comm/distributed_comm/shared_comm to skip the comm splitting,
# for testing of the index generation.
function split_dimension(dimensions::Vector{<:Dimension}, n_groups::Integer,
                         optimize_schur_complement_size::Bool,
                         level_comm::Union{MPI.Comm,FakeComm},
                         level_distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
                         level_shared_comm::Union{MPI.Comm,FakeComm})
    ind_type = typeof(n_groups)
    level_dimensions = copy(dimensions)
    next_comm = level_comm
    next_distributed_comm = level_distributed_comm
    next_shared_comm = level_shared_comm
    comm_rank = MPI.Comm_rank(level_comm)
    shared_comm_rank = MPI.Comm_rank(level_shared_comm)
    shared_comm_size = MPI.Comm_size(level_shared_comm)
    distributed_comm_rank = comm_rank ÷ shared_comm_size
    local_bottom_vector_indices = ind_type[]

    slice_i = pick_dimension_to_split(dimensions, n_groups,
                                      optimize_schur_complement_size)

    slice_dim = level_dimensions[slice_i]
    slice_remove_boundaries = slice_dim.periodic || slice_dim.remove_boundaries
    slice_dim_n_local = slice_dim.n_local
    slice_dim_nelement = slice_dim.nelement
    slice_irank = slice_dim.irank
    slice_nrank = slice_dim.nrank
    last_slice_ind = length(slice_dim.global_inds)
    elements_per_group = (slice_dim_nelement + n_groups - 1) ÷ n_groups
    n_active_groups = (slice_dim_nelement + elements_per_group - 1) ÷ elements_per_group
    top_vector_slice_dim_n = slice_dim.n - (n_active_groups - 1)
    if slice_remove_boundaries
        # Once dimension has been sliced at least once, the periodic boundary is removed,
        # so the dimension is effectively no longer periodic, and also does not include
        # lower and upper boundaries.
        slice_dim = Dimension(; nelement=slice_dim.nelement, ngrid=slice_dim.ngrid,
                              nrank=slice_dim.nrank, irank=slice_irank, periodic=false,
                              has_lower_boundary=false, has_upper_boundary=false,
                              remove_boundaries=false)
        top_vector_slice_dim_n -= 2
    end

    is_distributed_slice = slice_dim.nrank > 1
    if is_distributed_slice
        # When dimension is distributed, split on block boundaries.
        blocks_per_group = (slice_dim.nrank + n_groups - 1) ÷ n_groups
        group_rank = slice_dim.irank ÷ blocks_per_group
        next_comm = MPI.Comm_split(next_comm, group_rank, 0)
        if shared_comm_rank == 0
            next_distributed_comm = MPI.Comm_split(next_distributed_comm, group_rank, 0)
        end
        if slice_dim.nelement % slice_dim.nrank != 0
            error("Number of elements in dimension should split equally among blocks."
                  * "Dimension $slice_i has $(slice_dim.nelement) elements and "
                  * "$(slice_dim.nrank) blocks.")
        end
        elements_per_block = slice_dim.nelement ÷ slice_dim.nrank
        if group_rank == n_groups - 1
            this_group_nelement = slice_dim.nelement - group_rank * blocks_per_group * elements_per_block
            this_group_nrank = slice_dim.nrank - group_rank * blocks_per_group
        else
            this_group_nelement = blocks_per_group * elements_per_block
            this_group_nrank = blocks_per_group
        end
        this_group_irank = slice_irank - group_rank * blocks_per_group
        block_boundaries = [i_group * blocks_per_group for i_group ∈ 1:n_groups-1]
        if (slice_irank ∈ block_boundaries) || (slice_remove_boundaries && slice_irank == 0)
            # Lower boundary on this block is a split.
            local_bottom_vector_indices =
                vcat(local_bottom_vector_indices,
                     get_local_ind_slice(level_dimensions, slice_i, 1:1))
            first_top_vector_slice_ind = 2
        else
            first_top_vector_slice_ind = 1
        end
        if group_rank > 0
            slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                                  nrank=this_group_nrank, irank=this_group_irank,
                                  periodic=slice_dim.periodic, has_lower_boundary=false,
                                  has_upper_boundary=slice_dim.has_upper_boundary,
                                  remove_boundaries=false)
        end
        if (slice_irank + 1 ∈ block_boundaries) || (slice_remove_boundaries && slice_irank == slice_nrank - 1)
            # Upper boundary on this block is a split.
            local_bottom_vector_indices =
                vcat(local_bottom_vector_indices,
                     get_local_ind_slice(level_dimensions, slice_i,
                                         last_slice_ind:last_slice_ind))
            last_top_vector_slice_ind = last_slice_ind - 1
        else
            last_top_vector_slice_ind = last_slice_ind
        end
        if group_rank != n_groups - 1
            slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                                  nrank=this_group_nrank, irank=this_group_irank,
                                  periodic=slice_dim.periodic,
                                  has_lower_boundary=slice_dim.has_lower_boundary,
                                  has_upper_boundary=false, remove_boundaries=false)
        end
        local_top_vector_indices =
            get_local_ind_slice(level_dimensions, slice_i,
                                first_top_vector_slice_ind:last_top_vector_slice_ind)
        slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                              nrank=this_group_nrank, irank=this_group_irank,
                              periodic=slice_dim.periodic,
                              has_lower_boundary=slice_dim.has_lower_boundary,
                              has_upper_boundary=slice_dim.has_upper_boundary,
                              remove_boundaries=false)
    else
        ngrid = slice_dim.ngrid
        procs_per_group = (shared_comm_size + n_groups - 1) ÷ n_groups
        group_rank = shared_comm_rank ÷ procs_per_group
        next_comm = MPI.Comm_split(next_comm, group_rank, 0)
        next_shared_comm = MPI.Comm_split(next_shared_comm, group_rank ≥ n_active_groups ? nothing : group_rank, 0)
        if next_shared_comm != MPI.COMM_NULL && MPI.Comm_rank(next_shared_comm) == 0
            next_distributed_comm = MPI.COMM_SELF
        else
            next_distributed_comm = MPI.COMM_NULL
        end
        # Want to include the last processes, that have no work to do, in the 'last
        # group', which we can do by hacking `this_group_nelement` to be the same for the
        # processes that have no work as it was for the last group that did have work.
        last_group_rank = n_active_groups - 1
        last_group_nelement = (min((last_group_rank + 1) * elements_per_group,
                                   slice_dim.nelement)
                               - min(last_group_rank * elements_per_group,
                                     slice_dim.nelement))
        if group_rank ≥ last_group_rank
            this_group_nelement = last_group_nelement
        else
            this_group_nelement = elements_per_group
        end
        slice_step = elements_per_group * (ngrid - 1)
        if slice_remove_boundaries
            n_slices = (slice_dim_nelement + elements_per_group - 1) ÷ elements_per_group
            slice_points = [min(s * slice_step + 1, last_slice_ind) for s ∈ 0:n_slices]
        else
            offset = slice_dim.has_lower_boundary ? 1 : 0
            slice_points = slice_step+offset:slice_step:min(slice_step*(n_groups-1)+offset,slice_dim_n_local-1)
        end
        local_bottom_vector_indices =
            vcat(local_bottom_vector_indices,
                 get_local_ind_slice(level_dimensions, slice_i, slice_points))
        if slice_remove_boundaries
            first_local_top_vector_slice_ind = slice_points[min(group_rank+1,end)] + 1
            has_lower_boundary = false
            last_local_top_vector_slice_ind = slice_points[min(group_rank+2,end)] - 1
            has_upper_boundary = false
            first_top_vector_a_block_slice_ind = group_rank * slice_step + 2
            last_top_vector_a_block_slice_ind = min((group_rank + 1) * slice_step,
                                                    last_local_top_vector_slice_ind)
        else
            is_last_group_in_slice_dim = (group_rank + 1) * elements_per_group ≥ slice_dim_nelement
            if group_rank == 0
                first_local_top_vector_slice_ind = 1
                has_lower_boundary = slice_dim.has_lower_boundary
                first_top_vector_a_block_slice_ind = 1
            else
                first_local_top_vector_slice_ind = slice_points[min(group_rank,end)] + 1
                has_lower_boundary = false
                first_top_vector_a_block_slice_ind = group_rank * slice_step + 1
                if slice_dim.has_lower_boundary
                    first_top_vector_a_block_slice_ind += 1
                end
            end
            if is_last_group_in_slice_dim
                has_upper_boundary = slice_dim.has_upper_boundary
            else
                has_upper_boundary = false
            end
            if group_rank < length(slice_points)
                last_local_top_vector_slice_ind = slice_points[group_rank+1] - 1
            else
                last_local_top_vector_slice_ind = last_slice_ind
            end
            # Maximum last 'block slice-dimension ind' is the total slice dimension
            # size minus the number of slice points (=n_groups-1).
            if slice_dim.has_lower_boundary
                offset = 1
            else
                offset = 0
            end
            last_top_vector_a_block_slice_ind =
                min((group_rank + 1) * slice_step + offset, last_local_top_vector_slice_ind)
        end
        all_top_vector_slice_inds = [i for i ∈ 1:last_slice_ind if i ∉ slice_points]
        local_top_vector_indices = get_local_ind_slice(level_dimensions, slice_i,
                                                       all_top_vector_slice_inds)
        local_top_vector_a_block_indices =
            get_local_ind_slice(level_dimensions, slice_i,
                                first_top_vector_a_block_slice_ind:last_top_vector_a_block_slice_ind)
        slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                              nrank=slice_dim.nrank, irank=slice_irank, periodic=false,
                              has_lower_boundary=has_lower_boundary,
                              has_upper_boundary=has_upper_boundary,
                              remove_boundaries=false)
    end

    if any(collect(d.remove_boundaries for d ∈ level_dimensions))
        new_dimensions = copy(level_dimensions)
        extra_local_bottom_vector_indices = ind_type[]
        for i_dim ∈ 1:length(level_dimensions)
            if i_dim == slice_i
                continue
            end
            d = level_dimensions[i_dim]
            if d.remove_boundaries
                if d.has_lower_boundary
                    if d.irank == 0
                        extra_local_bottom_vector_indices =
                            vcat(extra_local_bottom_vector_indices,
                                 get_local_ind_slice(level_dimensions, i_dim, 1:1))
                    end
                    has_lower_boundary = false
                else
                    has_lower_boundary = d.has_lower_boundary
                end
                if d.has_upper_boundary
                    if d.irank == d.nrank - 1
                        last_ind = length(d.global_inds)
                        extra_local_bottom_vector_indices =
                            vcat(extra_local_bottom_vector_indices,
                                 get_local_ind_slice(level_dimensions, i_dim, last_ind:last_ind))
                    end
                    has_upper_boundary = false
                else
                    has_upper_boundary = d.has_upper_boundary
                end
                new_d = Dimension(; nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                                  irank=d.irank, periodic=false,
                                  has_lower_boundary=has_lower_boundary,
                                  has_upper_boundary=has_upper_boundary,
                                  remove_boundaries=false)
                new_dimensions[i_dim] = new_d
            end
        end
        local_top_vector_indices = setdiff(local_top_vector_indices,
                                           extra_local_bottom_vector_indices)
        if !is_distributed_slice
            local_top_vector_a_block_indices = setdiff(local_top_vector_a_block_indices,
                                                       extra_local_bottom_vector_indices)
        end
        local_bottom_vector_indices = vcat(local_bottom_vector_indices,
                                           extra_local_bottom_vector_indices)
        sort!(local_bottom_vector_indices)
        unique!(local_bottom_vector_indices)
        level_dimensions = new_dimensions
    else
        sort!(local_bottom_vector_indices)
    end

    if is_distributed_slice
        local_top_vector_a_block_indices = local_top_vector_indices
        a_block_sub_selection_indices = 1:length(local_top_vector_a_block_indices)
    else
        a_block_sub_selection_indices = fill(typeof(n_groups)(-1), length(local_top_vector_a_block_indices))
        if length(a_block_sub_selection_indices) > 0
            counter = 1
            for (i, ind) ∈ enumerate(local_top_vector_indices)
                if ind == local_top_vector_a_block_indices[counter]
                    a_block_sub_selection_indices[counter] = i
                    counter += 1
                    if counter > length(local_top_vector_a_block_indices)
                        break
                    end
                end
            end
        end
    end

    global_size = prod(d.n for d ∈ dimensions)
    global_top_vector_size =
        top_vector_slice_dim_n * prod(level_dimensions[i].n
                                      for i ∈ 1:length(level_dimensions) if i ≠ slice_i;
                                      init=1)

    level_dimensions[slice_i] = slice_dim

    bottom_vector_indices = get_global_indices(dimensions, local_bottom_vector_indices)
    top_vector_indices = get_global_indices(dimensions, local_top_vector_indices)

    return LevelInfo(; level_dimensions, global_size, global_top_vector_size,
                     top_vector_indices, local_top_vector_indices,
                     local_top_vector_a_block_indices, a_block_sub_selection_indices,
                     bottom_vector_indices, local_bottom_vector_indices, level_comm,
                     level_distributed_comm, level_shared_comm),
           level_dimensions, next_comm, next_distributed_comm, next_shared_comm
end

function get_lowest_level_duplicates(ind_type::Type,
                                     lowest_level_dimensions::Vector{<:Dimension})
    n_tuple = Tuple(d.n for d ∈ lowest_level_dimensions)
    return get_lowest_level_duplicates(ind_type, lowest_level_dimensions, n_tuple)
end
function get_lowest_level_duplicates(ind_type::Type,
                                     lowest_level_dimensions::Vector{<:Dimension},
                                     n_tuple::Tuple)
    lowest_level_non_duplicate_indices = ind_type[]
    periodic_pairs = Tuple{ind_type,ind_type}[]
    level_cartinds = CartesianIndices(n_tuple)
    for (flat_i, inds) ∈ enumerate(level_cartinds)
        has_duplicate = false
        if any(d.periodic && d.has_lower_boundary && d.has_upper_boundary && i == d.n for (d, i) ∈ zip(lowest_level_dimensions, Tuple(inds)))
            has_duplicate = true
            pair_i = 0
            for (d, i) ∈ zip(reverse(lowest_level_dimensions), reverse(Tuple(inds)))
                n = d.periodic && d.has_lower_boundary && d.has_upper_boundary ? d.n - 1 : d.n
                if d.periodic && d.has_lower_boundary && d.has_upper_boundary && i == d.n
                    # pair_i corresponds to the first index in this dimension.
                    pair_i = pair_i * n
                else
                    pair_i = pair_i * n + i - 1
                end
            end
            pair_i += 1
            push!(periodic_pairs, (pair_i, flat_i))
        end
        if !has_duplicate
            push!(lowest_level_non_duplicate_indices, flat_i)
        end
    end

    return lowest_level_non_duplicate_indices, periodic_pairs
end

function generate_possible_multipliers(multiplier_list, v::Val{N}, ind_type::Type) where N
    sizes = ntuple(i->length(multiplier_list), v)
    possible_multipliers = Vector{Union{ind_type,Nothing}}[]
    for inds ∈ CartesianIndices(sizes)
        push!(possible_multipliers, [multiplier_list[i] for i ∈ Tuple(inds)])
    end
    return possible_multipliers
end

function multiply_block_sizes(multiplier, previous_block_sizes, nelement_local_list)
    block_sizes = similar(previous_block_sizes)
    for (i, (m, p, nel)) ∈ enumerate(zip(multiplier, previous_block_sizes, nelement_local_list))
        if m === nothing
            block_sizes[i] = nel
        else
            block_sizes[i] = m * p
        end
    end
    return block_sizes
end

"""
    mpi_static_condensation(dimensions::Vector{<:Dimension};
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
    level_indices = get_global_indices(dimensions,
                                       collect(1:prod(d.n_local for d ∈ dimensions)))
    level_global_size = prod(d.n for d ∈ dimensions)
    level_shared_comm = shared_comm
    level_shared_comm_size = shared_comm_size
    for (level, (block_sizes, total_local_nblock)) ∈ enumerate(zip(block_sizes_list,
                                                                   total_local_nblock_list))
        if level == n_levels
            # Only handle periodicity on the final level
            dims = dimensions
        else
            dims = dimensions_without_periodic
        end
        if level_shared_comm_size > total_local_nblock
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
                                       level_global_size, distributed_comm,
                                       level_shared_comm)
        level_info_list[level] = this_level_info
        level_indices = this_level_info.bottom_vector_indices
        level_global_size = this_level_info.global_bottom_vector_size
    end

    # Create lowest level MPISchurComplement solver
    # Use a parallelized dense-matrix LU solver for the last Schur complement solve as
    # long as the last Schur complement matrix is not too small.
    last_level_info = level_info_list[end]
    if last_level_info.level_shared_comm != MPI.COMM_NULL
        last_A_block_solver =
            BlockDiagonalSolver{data_type}(last_level_info.global_size - last_level_info.global_bottom_vector_size,
                                           last_level_info.a_block_sub_selection_indices,
                                           last_level_info.a_block_lu_selection_indices)
        last_level_shared_comm = last_level_info.level_shared_comm
        level_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=last_level_shared_comm)
        level_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=last_level_shared_comm)
        last_parallel_schur = last_level_info.global_bottom_vector_size ≥ 1024
        this_level_sc =
            mpi_schur_complement(last_A_block_solver, data_type, data_type, data_type,
                                 last_level_info.top_vector_indices,
                                 last_level_info.bottom_vector_indices; comm=comm,
                                 shared_comm=last_level_shared_comm,
                                 distributed_comm=distributed_comm,
                                 allocate_shared_float=level_allocate_shared_float,
                                 allocate_shared_int=level_allocate_shared_int,
                                 use_sparse=use_sparse, sparse_Ainv_B=true,
                                 parallel_schur=last_parallel_schur,
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
        if level < n_levels
            this_A_block_solver =
                BlockDiagonalSolver{data_type}(this_level_info.global_size - this_level_info.global_bottom_vector_size,
                                               this_level_info.a_block_sub_selection_indices,
                                               this_level_info.a_block_lu_selection_indices)
            Ainv_dot_B_buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions,
                                                    this_level_info.block_sizes,
                                                    this_level_info.top_vector_indices,
                                                    this_level_info.bottom_vector_indices,
                                                    this_level_shared_comm,
                                                    level_allocate_shared_float,
                                                    level_allocate_shared_int)
            schur_complement_buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions,
                                                    this_level_info.block_sizes,
                                                    this_level_info.bottom_vector_indices,
                                                    this_level_info.bottom_vector_indices,
                                                    this_level_shared_comm,
                                                    level_allocate_shared_float,
                                                    level_allocate_shared_int)
            this_level_sc =
                mpi_schur_complement(this_A_block_solver, data_type, data_type, data_type,
                                     this_level_info.top_vector_indices,
                                     this_level_info.bottom_vector_indices; comm=comm,
                                     shared_comm=this_level_shared_comm,
                                     distributed_comm=distributed_comm,
                                     allocate_shared_float=level_allocate_shared_float,
                                     allocate_shared_int=level_allocate_shared_int,
                                     Ainv_dot_B_buffer=Ainv_dot_B_buffer,
                                     schur_complement_buffer=schur_complement_buffer,
                                     use_sparse=use_sparse, sparse_Ainv_B=true,
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
        this_level_schur_solver =
            MPIStaticCondensationParallel(this_level_info.global_size, this_level_sc,
                                          this_level_info.local_top_vector_indices,
                                          this_level_info.all_local_top_vector_a_block_indices,
                                          this_level_info.all_a_block_sub_selection_indices,
                                          this_level_info.local_bottom_vector_indices,
                                          this_shared_local_bottom_vector_indices,
                                          this_shared_local_bottom_sub_selection_indices,
                                          this_u_buffer, this_v_buffer, timer)
    end
    # The level-1 MPIStaticCondensationParallel is not a 'Schur complement solver', but
    # the full matrix solver.
    solver = this_level_schur_solver

    return solver
end

function lu!(block_diagonal_solver::BlockDiagonalSolver, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    if solver != [nothing]
        for (s, inds) ∈ zip(solver, block_diagonal_solver.lu_selection_indices)
            lu!(s, sparse(@view A[inds,inds]); reuse_symbolic=false)
        end
    end
    return nothing
end

function ldiv!(x::AbstractVector{T}, block_diagonal_solver::BlockDiagonalSolver{T},
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
function ldiv!(block_diagonal_solver::BlockDiagonalSolver{T}, u::AbstractVector{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(x::AbstractMatrix{T}, block_diagonal_solver::BlockDiagonalSolver{T},
               u::AbstractMatrix{T}) where T
    if block_diagonal_solver.local_block_solver !== nothing
        for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
            ldiv!(this_x, block_diagonal_solver, this_u)
        end
    end
    return nothing
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolver{T}, u::AbstractMatrix{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(x::AbstractSparseMatrixCSC{T},
               block_diagonal_solver::BlockDiagonalSolver{T},
               u::AbstractSparseMatrixCSC{T}) where T
    solvers = block_diagonal_solver.local_block_solver
    if solvers !== nothing
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
            this_x_buffer = @view x_buffer[1:block_size]
            for col ∈ 1:m
                u_flat_start = u_colptr[col]
                u_flat_end = u_colptr[col+1] - 1
                if u_flat_end < u_flat_start
                    # Column is empty.
                    continue
                end
                u_row_start = u_rowval[u_flat_start]
                u_row_end = u_rowval[u_flat_end]
                if u_row_start ≤ block_end && u_row_end ≥ block_start
                    # Column has non-zero row entries for this block.
                    u_column = @view u[:,col]
                    for (i1, i2) ∈ enumerate(bi)
                        this_u_buffer[i1] = u_column[i2]
                    end
                    ldiv!(this_x_buffer, s, this_u_buffer)
                    x_flat_start = x_colptr[col]
                    x_flat_end = x_colptr[col+1] - 1
                    count = x_flat_start
                    while x_rowval[count] < first(bi) && count ≤ x_flat_end
                        count += 1
                    end
                    for (i2, i1) ∈ enumerate(bi)
                        # Assume that the structural non-zero entries of `x` are enough to
                        # contain all the non-zero entries of the solve. Note that the
                        # entries in this_x_buffer that should be structurally zero might
                        # only be zero up to floating-point precision.
                        if i1 == x_rowval[count]
                            x_nzval[count] = this_x_buffer[i2]
                            count += 1
                        end
                    end
                end
            end
        end
    end
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

function lu!(solver::MPIStaticCondensationSerialSparse, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        non_duplicate_indices = solver.non_duplicate_indices
        periodic_index_pairs = solver.periodic_index_pairs
        for (j1, j2) ∈ eachcol(periodic_index_pairs)
            @views A[:,non_duplicate_indices[j1]] .+= A[:,j2]
        end
        for (i1, i2) ∈ eachcol(periodic_index_pairs)
            @views A[non_duplicate_indices[i1],non_duplicate_indices] .+= A[i2,non_duplicate_indices]
        end
        # For simplicity assume non-zero pattern might change, so pass reuse_symbolic=false.
        lu!(solver.local_block_solver,
            sparse(@view(A[non_duplicate_indices,non_duplicate_indices]));
            reuse_symbolic=false, check=solver.check_lu)
    end
    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationSerialSparse{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation serial sparse ldiv! $(size(solver, 1))" begin
        non_duplicate_indices = solver.non_duplicate_indices
        # Note if X or U are views that were indexed with Vector{<:Integer}, then we need
        # to replace them with contiguous-in-memory buffers.
        if isa(X, StridedVector) && isa(non_duplicate_indices, Colon)
            this_X = X
        else
            this_X = solver.X_buffer
        end
        if isa(U, StridedVector) && isa(non_duplicate_indices, Colon)
            this_U = U
        else
            this_U = solver.U_buffer
            if isa(non_duplicate_indices, Colon)
                for i ∈ eachindex(this_U, U)
                    this_U[i] = U[i]
                end
            else
                for (i1, i2) ∈ enumerate(non_duplicate_indices)
                    this_U[i1] = U[i2]
                end
            end
        end
        ldiv!(this_X, solver.local_block_solver, this_U)
        if !(isa(X, StridedVector) && isa(non_duplicate_indices, Colon))
            if isa(non_duplicate_indices, Colon)
                for i ∈ eachindex(X, this_X)
                    X[i] = this_X[i]
                end
            else
                for (i1, i2) ∈ enumerate(non_duplicate_indices)
                    X[i2] = this_X[i1]
                end
            end
            for (i1, i2) ∈ eachcol(solver.periodic_index_pairs)
                X[i2] = this_X[i1]
            end
        end
    end
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationSerialSparse{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! serial sparse $(size(solver, 1))" begin
        # Note if X or U are views that were indexed with Vector{<:Integer}, then we need
        # to fall back to the AbstractVector function which can replace them with
        # contiguous-in-memory buffers.
        local_block_solver = solver.local_block_solver
        if !isa(X, StridedMatrix) || !isa(U, StridedMatrix) || !isa(solver.non_duplicate_indices, Colon)
            for (this_X, this_U) ∈ zip(eachcol(X), eachcol(U))
                ldiv!(this_X, solver, this_U)
            end
        else
            ldiv!(X, local_block_solver, U)
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationSerialSparse{T}, U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! serial sparse $(size(solver, 1))" begin
        non_duplicate_indices = solver.non_duplicate_indices
        U_buffer = solver.U_buffer
        U_buffer .= @view U[non_duplicate_indices]
        if isa(U, StridedVector) && isa(non_duplicate_indices, Colon)
            this_X = U
        else
            # Note if U is a view that was indexed with Vector{<:Integer}, then we need to
            # replace it with a contiguous-in-memory buffer.
            this_X = solver.X_buffer
        end
        ldiv!(this_X, solver.local_block_solver, U_buffer)
        if !(isa(U, StridedVector) && isa(non_duplicate_indices, Colon))
            @views U[non_duplicate_indices] .= this_X
            for (i1, i2) ∈ eachcol(solver.periodic_index_pairs)
                U[i2] = this_X[i1]
            end
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationSerialSparse{T}, U::AbstractMatrix{T}) where T
    for col ∈ eachcol(U)
        ldiv!(solver, col)
    end
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialDense, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        # Re-use the arrays to avoid allocating.
        mat_storage = solver.local_block_solver.factors
        ipiv = solver.local_block_solver.ipiv
        non_duplicate_indices = solver.non_duplicate_indices
        periodic_index_pairs = solver.periodic_index_pairs
        check = solver.check_lu
        mat_storage .= @view A[non_duplicate_indices,non_duplicate_indices]
        for (j1, j2) ∈ eachcol(periodic_index_pairs)
            @views mat_storage[:,j1] .+= A[non_duplicate_indices,j2]
        end
        for (i1, i2) ∈ eachcol(periodic_index_pairs)
            @views mat_storage[i1,:] .+= A[i2,non_duplicate_indices]
        end
        for (j1, j2) ∈ eachcol(periodic_index_pairs), (i1, i2) ∈ eachcol(periodic_index_pairs)
            mat_storage[i1,j1] += A[i2,j2]
        end
        LAPACK.getrf!(mat_storage, ipiv; check=check)
    end
    return nothing
end

function ldiv!(solver::MPIStaticCondensationSerialDense{T}, U::AbstractVectorOrMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! serial dense $(size(solver, 1))" begin
        local_block_solver = solver.local_block_solver
        non_duplicate_indices = solver.non_duplicate_indices
        if isa(U, StridedVecOrMat) && isa(non_duplicate_indices, Colon)
            ldiv!(local_block_solver, U)
        elseif isa(U, AbstractMatrix)
            # Note if U is a view that was indexed with Vector{<:Integer}, then we need to
            # fall back to the AbstractVector function which can replace it with a
            # contiguous-in-memory buffer.
            for this_U ∈ eachcol(U)
                ldiv!(solver, this_U)
            end
        else # U is an AbstractVector
            # Note if U is a view that was indexed with Vector{<:Integer}, then we need to
            # replace it with a contiguous-in-memory buffer.
            X_buffer = solver.X_buffer
            X_buffer .= @view U[non_duplicate_indices]
            ldiv!(local_block_solver, X_buffer)
            @views U[non_duplicate_indices] .= X_buffer
            for (i1, i2) ∈ eachcol(solver.periodic_index_pairs)
                U[i2] = X_buffer[i1]
            end
        end
    end
    return nothing
end
function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationSerialDense{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! serial dense $(size(solver, 1))" begin
        non_duplicate_indices = solver.non_duplicate_indices
        if (isa(X, StridedVector) && isa(non_duplicate_indices, Colon))
            this_X = X
        else
            # Note if X is a view that was indexed with Vector{<:Integer}, then we need to
            # replace it with a contiguous-in-memory buffer.
            this_X = solver.X_buffer
        end
        ldiv!(this_X, solver.local_block_solver, @view(U[non_duplicate_indices]))
        if !(isa(X, StridedVector) && isa(non_duplicate_indices, Colon))
            @views X[non_duplicate_indices] .= this_X
            for (i1, i2) ∈ eachcol(solver.periodic_index_pairs)
                X[i2] = this_X[i1]
            end
        end
    end
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationSerialDense{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! serial dense $(size(solver, 1))" begin
        local_block_solver = solver.local_block_solver
        if isa(X, StridedMatrix) && isa(solver.non_duplicate_indices, Colon)
            ldiv!(X, local_block_solver, U)
        else
            # Note if X is a view that was indexed with Vector{<:Integer}, then we need to
            # fall back to the AbstractVector function which can replace it with a
            # contiguous-in-memory buffer.
            for (this_X, this_U) ∈ zip(eachcol(X), eachcol(U))
                ldiv!(this_X, solver, this_U)
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

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        # It is slightly faster to copy the data to/from local buffers than to use @view
        # with Vector{Int64} indices.
        all_local_top_vector_a_block_indices = solver.all_local_top_vector_a_block_indices
        all_a_block_sub_selection_indices = solver.all_a_block_sub_selection_indices
        this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
        this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
        u = solver.u_buffer
        v = solver.v_buffer
        # Use the a_block_indices here so that no shared-memory synchronization is needed
        # before the ldiv!() call for the A subblock with the BlockDiagonalSolver inside
        # the MPISchurComplement ldiv!().
        for (i1, i2) ∈ zip(all_a_block_sub_selection_indices, all_local_top_vector_a_block_indices)
            u[i1] = U[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_indices, this_shared_local_bottom_vector_indices)
            v[i1] = U[i2]
        end
        ldiv!(u, v, solver.local_block_solver, u, v)
        for (i1, i2) ∈ zip(all_local_top_vector_a_block_indices, all_a_block_sub_selection_indices)
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
        all_local_top_vector_a_block_indices = solver.all_local_top_vector_a_block_indices
        all_a_block_sub_selection_indices = solver.all_a_block_sub_selection_indices
        this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
        this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
        u = solver.u_buffer
        v = solver.v_buffer
        # Use the a_block_indices here so that no shared-memory synchronization is needed
        # before the ldiv!() call for the A subblock with the BlockDiagonalSolver inside
        # the MPISchurComplement ldiv!().
        for (i1, i2) ∈ zip(all_a_block_sub_selection_indices, all_local_top_vector_a_block_indices)
            u[i1] = U[i2]
        end
        for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_indices, this_shared_local_bottom_vector_indices)
            v[i1] = U[i2]
        end
        ldiv!(u, v, solver.local_block_solver, u, v)
        for (i1, i2) ∈ zip(all_local_top_vector_a_block_indices, all_a_block_sub_selection_indices)
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
