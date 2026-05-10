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

struct MPIStaticCondensationSerialNull{Tf<:AbstractFloat} <: MPIStaticCondensation{Tf} end

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

struct MPIStaticCondensationParallel{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:MPISchurComplement{Tf},Tranget,Trangetab,Trangeb,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_top_vector_indices::Tranget
    local_top_vector_a_block_indices::Trangetab
    local_bottom_vector_indices::Trangeb
    timer::Ttimer
end
Base.size(Alu::MPIStaticCondensationParallel) = (Alu.n, Alu.n)
Base.size(Alu::MPIStaticCondensationParallel, d::Integer) = size(Alu)[d]

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolver{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:MPIStaticCondensation{Tf},Trange}
    n::Ti
    local_block_solver::Tsolver
    block_indices::Trange
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
    has_lower_boundary::Bool
    has_upper_boundary::Bool
    remove_boundaries::Bool

    function Dimension(; nelement::Ti, ngrid::Ti, nrank::Ti, irank::Ti, periodic::Bool,
                       has_lower_boundary::Bool, has_upper_boundary::Bool,
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

        if !has_lower_boundary
            if nelement > 0
                n -= 1
            end
            if irank == 0
                if nelement_local > 0
                    n_local -= 1
                end
                first_global_ind += 1
            end
        end
        if !has_upper_boundary
            if nelement > 0
                n -= 1
            end
            if irank == nrank - 1
                if nelement_local > 0
                    n_local -= 1
                end
                last_global_ind -= 1
            end
        end

        global_inds = collect(first_global_ind:last_global_ind)
        if periodic && irank == nrank - 1
            global_inds[end] = 1
        end

        return new{Ti}(n, n_local, nelement, ngrid, nrank, irank, global_inds, periodic,
                       has_lower_boundary, has_upper_boundary, remove_boundaries)
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
    return Dimension(; nelement, ngrid, nrank, irank, periodic, has_lower_boundary=true,
                     has_upper_boundary=true, remove_boundaries)
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
            idim = last_argmax(d.n for d ∈ dimensions[distributed_dims])
            return distributed_dims[idim]
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

function get_flattened_index(indices::CartesianIndex, dimensions::Vector{<:Dimension})
    flat_i = 0
    for (i, dim) ∈ zip(reverse(Tuple(indices)), reverse(dimensions))
        flat_i = flat_i * dim.n + dim.global_inds[i] - 1
    end
    # So far constructed a 0-based index, so convert to 1-based.
    flat_i += 1
    return flat_i
end

function get_local_flattened_index(indices::CartesianIndex, dim_sizes::Vector{<:Integer})
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
    dimensions = copy(dimensions)
    dim_sizes = [d.n_local for d ∈ dimensions]
    result_ranges = Tuple(i == dim_to_slice ? slice_inds : 1:dim_sizes[i] for i ∈ 1:length(dimensions))
    inds = fill(eltype(slice_inds)(-1), prod(length(r) for r ∈ result_ranges))
    for (local_flat_i, i) ∈ enumerate(CartesianIndices(result_ranges))
        inds[local_flat_i] = get_local_flattened_index(i, dim_sizes)
    end
    return inds
end

function get_local_ind_slice(dimensions::Vector{<:Dimension}, dim_to_slice::Integer,
                             slice_inds::Vector{<:Integer})
    # When `slice_inds` is a Vector, not an OrdinalRange, cannot use CartesianIndices on
    # it, so have to do more complicated loops.
    dimensions = copy(dimensions)
    result_ranges_left = Tuple(1:dimensions[i].n_local for i ∈ 1:dim_to_slice-1)
    result_ranges_right = Tuple(1:dimensions[i].n_local for i ∈ dim_to_slice+1:length(dimensions))
    inds = fill(eltype(slice_inds)(-1),
                prod(length(r) for r ∈ result_ranges_left; init=1) * length(slice_inds) *
                prod(length(r) for r ∈ result_ranges_right; init=1))
    dim_sizes = [d.n_local for d ∈ dimensions]
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
    global_inds = similar(local_inds)
    cartinds = CartesianIndices(Tuple(d.n_local for d ∈ dimensions))
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

struct FakeComm
    rank::Int64
    size::Int64
end
MPI.Comm_rank(comm::FakeComm) = comm.rank
MPI.Comm_size(comm::FakeComm) = comm.size
MPI.Comm_split(comm::FakeComm, color, key) = comm

@kwdef struct LevelInfo{Ti,Tasub,Tcomm<:Union{MPI.Comm,FakeComm},Tdcomm<:Union{MPI.Comm,Nothing,FakeComm}}
    level_dimensions::Vector{Dimension{Ti}}
    global_top_vector_size::Ti
    top_vector_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    local_top_vector_a_block_indices::Vector{Ti}
    a_block_sub_selection_indices::Tasub
    bottom_vector_indices::Vector{Ti}
    local_bottom_vector_indices::Vector{Ti}
    level_comm::Tcomm
    level_distributed_comm::Tdcomm
    level_shared_comm::Tcomm
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
    slice_irank = slice_dim.irank
    slice_nrank = slice_dim.nrank
    last_slice_ind = length(slice_dim.global_inds)
    top_vector_slice_dim_n = slice_dim.n - (n_groups - 1)
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

    elements_per_group = (slice_dim.nelement + n_groups - 1) ÷ n_groups

    if elements_per_group * (n_groups - 1) ≥ slice_dim.nelement && slice_remove_boundaries
        # The last element does not actually contain any points, so the last 'boundary'
        # point is actually the final grid point in slice_dim, which was already removed
        # by slice_remove_boundaries, so we have removed one point to many in
        # `top_vector_slice_dim_n`.
        top_vector_slice_dim_n += 1
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
        if group_rank == n_groups - 1
            this_group_nelement = slice_dim.nelement - group_rank * elements_per_group
        else
            this_group_nelement = elements_per_group
        end
        slice_step = elements_per_group * (ngrid - 1)
        if slice_remove_boundaries
            skip_last = ((n_groups - 1) * slice_step + 1 == last_slice_ind)
            slice_points = [min(s * slice_step + 1, last_slice_ind) for s ∈ 0:n_groups-skip_last]
        else
            slice_points = slice_step:slice_step:slice_step*(n_groups-1)
            if slice_dim.has_lower_boundary
                slice_points = slice_points .+ 1
            end
        end
        local_bottom_vector_indices =
            vcat(local_bottom_vector_indices,
                 get_local_ind_slice(level_dimensions, slice_i, slice_points))
        if slice_remove_boundaries
            first_local_top_vector_slice_ind = slice_points[group_rank+1] + 1
            has_lower_boundary = false
            last_local_top_vector_slice_ind = slice_points[min(group_rank+2,end)] - 1
            has_upper_boundary = false
            first_top_vector_block_slice_ind = group_rank * slice_step + 1
            # Maximum last 'block slice-dimension ind' is the total slice dimension size
            # minus the number of slice points (=n_groups-1), minus the two boundary
            # points that are removed.
            last_top_vector_block_slice_ind = min((group_rank + 1) * slice_step,
                                                  slice_dim.n - (n_groups - 1) - 2)
        else
            if group_rank == 0
                first_local_top_vector_slice_ind = 1
                has_lower_boundary = slice_dim.has_lower_boundary
                first_top_vector_block_slice_ind = 1
            else
                first_local_top_vector_slice_ind = slice_points[group_rank] + 1
                has_lower_boundary = false
                first_top_vector_block_slice_ind = group_rank * slice_step + 1
                if slice_dim.has_lower_boundary
                    first_top_vector_block_slice_ind += 1
                end
            end
            if group_rank == n_groups - 1
                last_local_top_vector_slice_ind = last_slice_ind
                has_upper_boundary = slice_dim.has_upper_boundary
            else
                last_local_top_vector_slice_ind = slice_points[group_rank+1] - 1
                has_upper_boundary = false
            end
            # Maximum last 'block slice-dimension ind' is the total slice dimension
            # size minus the number of slice points (=n_groups-1).
            if slice_dim.has_lower_boundary
                offset = 1
            else
                offset = 0
            end
            last_top_vector_block_slice_ind = min((group_rank + 1) * slice_step + offset,
                                                  slice_dim.n - (n_groups - 1))
        end
        all_top_vector_slice_inds = [i for i ∈ 1:last_slice_ind if i ∉ slice_points]
        local_top_vector_indices = get_local_ind_slice(level_dimensions, slice_i,
                                                       all_top_vector_slice_inds)
        local_top_vector_a_block_indices =
            get_local_ind_slice(level_dimensions, slice_i,
                                first_local_top_vector_slice_ind:last_local_top_vector_slice_ind)
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
                                  irank=d.irank, periodic=d.periodic,
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

    global_top_vector_size =
        top_vector_slice_dim_n * prod(level_dimensions[i].n
                                      for i ∈ 1:length(level_dimensions) if i ≠ slice_i;
                                      init=1)

    level_dimensions[slice_i] = slice_dim

    bottom_vector_indices = get_global_indices(dimensions, local_bottom_vector_indices)
    top_vector_indices = get_global_indices(dimensions, local_top_vector_indices)

    return LevelInfo(; level_dimensions, global_top_vector_size, top_vector_indices,
                     local_top_vector_indices, local_top_vector_a_block_indices,
                     a_block_sub_selection_indices, bottom_vector_indices,
                     local_bottom_vector_indices, level_comm, level_distributed_comm,
                     level_shared_comm),
           level_dimensions, next_comm, next_distributed_comm, next_shared_comm
end

"""

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
at each level. The default strategy (`true`) splits the largest (according to value of
`n`) dimension remaining at each level, in order to minimise the size of the Schur
complement block. The alternative strategy (`false`) tries to optimise load balance by
considering first dimensions whose remaining `nelement` value can be exactly divided by
the group size (picking the largest of these), and only considering other dimensions if no
dimension can be exactly divided. In either case, dimensions that are distributed over
different shared-memory MPI blocks are divided first, until the locally-owned parts of all
dimensions are contained within the same shared-memory MPI block. The two strategies will
be equivalent as long as the largest dimension at each level is anyway exactly divisible,
which may often be the case (e.g. if the number of processes is a power of 2, and
`nelement` of the dimensions contain enough factors of 2).

`timer` can be passed a `TimerOutput` object to collect run timings.

`check_lu=true` can be passed to activate extra checks that all values are finite in
matrices being factorized.
"""
function mpi_static_condensation(dimensions::Vector{<:Dimension};
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

    n_levels = length(n_blocks_factors) + length(shared_comm_size_factors) + 1

    if n_levels == 1
        lowest_level_n = prod(d.n for d ∈ dimensions)
        lowest_level_dimensions = dimensions
    else
        this_level_dimensions = dimensions
        this_level_comm = comm
        this_level_distributed_comm = distributed_comm
        this_level_shared_comm = shared_comm

        # Vector{LevelInfo} is not type stable because of the unspecified type parameters
        # of LevelInfo, but that does not matter here because this is just a constructor
        # function, and the nested nature of the solve means that the final
        # `MPIStaticCondensationParallel` could not have its type fully specified by the
        # types of the input arguments anyway.
        level_info_list = Vector{LevelInfo}(undef, n_levels - 1)
        level = 0
        for (level, n_groups) ∈ enumerate(vcat(n_blocks_factors,
                                               shared_comm_size_factors))
            this_level_info, this_level_dimensions, this_level_comm,
            this_level_distributed_comm, this_level_shared_comm =
                split_dimension(this_level_dimensions, n_groups,
                                optimize_schur_complement_size, this_level_comm,
                                this_level_distributed_comm, this_level_shared_comm)
            level_info_list[level] = this_level_info
        end

        lowest_level_n = length(level_info_list[end].local_top_vector_a_block_indices)
        lowest_level_dimensions = this_level_dimensions
    end

    # Create lowest level solver
    if any(d.periodic && (d.has_lower_boundary || d.has_upper_boundary) for d ∈ lowest_level_dimensions)
        for d ∈ lowest_level_dimensions
            if d.periodic && ((d.has_lower_boundary && !d.has_upper_boundary)
                              || (!d.has_lower_boundary && d.has_upper_boundary))
                error("Any periodic dimension (that has not already been split up before "
                      * "the lowest level) should have both boundaries or neither")
            end
        end
        lowest_level_non_duplicate_indices = ind_type[]
        periodic_pairs = Tuple{ind_type,ind_type}[]
        level_cartinds = CartesianIndices(Tuple(d.n for d ∈ lowest_level_dimensions))
        for (flat_i, inds) ∈ enumerate(level_cartinds)
            has_duplicate = false
            if any(d.periodic && d.has_lower_boundary && d.has_upper_boundary && i == d.n for (d, i) ∈ zip(lowest_level_dimensions, Tuple(inds)))
                has_duplicate = true
                pair_i = 0
                for (d, i) ∈ zip(reverse(dimensions), reverse(Tuple(inds)))
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
        lowest_level_periodic_index_pairs = zeros(ind_type, 2, length(periodic_pairs))
        for (i, pair) ∈ enumerate(periodic_pairs)
            lowest_level_periodic_index_pairs[:,i] .= pair
        end
        lowest_level_n = length(lowest_level_non_duplicate_indices)
    else
        lowest_level_non_duplicate_indices = (:)
        lowest_level_periodic_index_pairs = zeros(ind_type, 2, 0)
    end
    identity = Matrix{data_type}(undef, lowest_level_n, lowest_level_n)
    copyto!(identity, I)
    if lowest_level_n == 0
        lowest_level_solver = MPIStaticCondensationSerialNull{data_type}()
    elseif use_sparse
        lowest_level_solver =
            MPIStaticCondensationSerialSparse(lu(sparse(identity); check=check_lu),
                                              Vector{data_type}(undef, lowest_level_n),
                                              Vector{data_type}(undef, lowest_level_n),
                                              lowest_level_non_duplicate_indices,
                                              lowest_level_periodic_index_pairs,
                                              timer, check_lu)
    else
        lowest_level_solver =
            MPIStaticCondensationSerialDense(lu(identity; check=check_lu),
                                             Vector{data_type}(undef, lowest_level_n),
                                             lowest_level_non_duplicate_indices,
                                             lowest_level_periodic_index_pairs,
                                             timer, check_lu)
    end

    this_level_solver = lowest_level_solver
    for level ∈ n_levels-1:-1:1
        level_info = level_info_list[level]

        # A_block_solver has its `lu!()` function called from within the
        # MPISchurComplement solver. At that point a view that gives the top-left 'A'
        # block of the matrix has already been constructed (using
        # `level_info.top_vector_indices`), and is passed to A_block_solver, which needs
        # to select its block out of that.
        A_block_solver = BlockDiagonalSolver(level_info.global_top_vector_size,
                                             this_level_solver,
                                             level_info.a_block_sub_selection_indices)

        # Use a parallelized dense-matrix LU solver for the Schur complement solve as long
        # as the Schur complement matrix is not too small.
        level_parallel_schur = length(level_info.bottom_vector_indices) ≥ 1024

        level_shared_comm = level_info.level_shared_comm

        if allocate_shared_float === nothing
            level_allocate_shared_float = nothing
        else
            level_allocate_shared_float =
                (args...) -> allocate_shared_float(args...; comm=level_shared_comm)
        end

        if allocate_shared_int === nothing
            level_allocate_shared_int = nothing
        else
            level_allocate_shared_int =
                (args...) -> allocate_shared_int(args...; comm=level_shared_comm)
        end

        this_level_sc =
            mpi_schur_complement(A_block_solver, data_type, data_type, data_type,
                                 level_info.top_vector_indices,
                                 level_info.bottom_vector_indices;
                                 comm=level_info.level_comm,
                                 shared_comm=level_shared_comm,
                                 distributed_comm=level_info.level_distributed_comm,
                                 allocate_shared_float=level_allocate_shared_float,
                                 allocate_shared_int=level_allocate_shared_int,
                                 synchronize_shared=synchronize_shared,
                                 use_sparse=use_sparse, separate_Ainv_B=separate_Ainv_B,
                                 parallel_schur=level_parallel_schur,
                                 skip_factorization=true, schur_tile_size=schur_tile_size,
                                 check_lu=check_lu, timer=timer)
        this_level_solver =
            MPIStaticCondensationParallel(length(level_info.top_vector_indices),
                                          this_level_sc,
                                          level_info.local_top_vector_indices,
                                          level_info.local_top_vector_a_block_indices,
                                          level_info.local_bottom_vector_indices, timer)
    end

    return this_level_solver
end

function lu!(block_diagonal_solver::BlockDiagonalSolver, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    lu!(solver, A)
    return nothing
end

function ldiv!(x::AbstractVector{T}, block_diagonal_solver::BlockDiagonalSolver{T},
               u::AbstractVector{T}) where T
    solver = block_diagonal_solver.local_block_solver
    block_indices = block_diagonal_solver.block_indices
    @views ldiv!(x[block_indices], solver, u[block_indices])
    return nothing
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolver{T}, u::AbstractVector{T}) where T
    solver = block_diagonal_solver.local_block_solver
    block_indices = block_diagonal_solver.block_indices
    @views ldiv!(solver, u[block_indices])
    return nothing
end
function ldiv!(x::AbstractMatrix{T}, block_diagonal_solver::BlockDiagonalSolver{T},
               u::AbstractMatrix{T}) where T
    solver = block_diagonal_solver.local_block_solver
    block_indices = block_diagonal_solver.block_indices
    @views ldiv!(x[block_indices,:], solver, u[block_indices,:])
    return nothing
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolver{T}, u::AbstractMatrix{T}) where T
    solver = block_diagonal_solver.local_block_solver
    block_indices = block_diagonal_solver.block_indices
    @views ldiv!(solver, u[block_indices,:])
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialNull, A::AbstractMatrix)
    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationSerialNull{T},
               U::AbstractVector{T}) where T
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationSerialNull{T},
               U::AbstractMatrix{T}) where T
    return nothing
end
function ldiv!(solver::MPIStaticCondensationSerialNull{T},
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
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
            this_U .= @view U[non_duplicate_indices]
        end
        ldiv!(this_X, solver.local_block_solver, this_U)
        if !(isa(X, StridedVector) && isa(non_duplicate_indices, Colon))
            @views X[non_duplicate_indices] .= this_X
            for (i1, i2) ∈ eachcol(solver.periodic_index_pairs)
                X[i2] = this_X[i1]
            end
        end
    end
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationSerialSparse{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
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
        local_top_vector_a_block_indices = solver.local_top_vector_a_block_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        a = @view A[local_top_vector_a_block_indices,local_top_vector_a_block_indices]
        b = @view A[local_top_vector_indices,local_bottom_vector_indices]
        c = @view A[local_bottom_vector_indices,local_top_vector_indices]
        d = @view A[local_bottom_vector_indices,local_bottom_vector_indices]
        update_schur_complement!(solver.local_block_solver, a, b, c, d)
    end
    return nothing
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        local_top_vector_indices = solver.local_top_vector_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        x = @view X[local_top_vector_indices]
        u = @view U[local_top_vector_indices]
        y = @view X[local_bottom_vector_indices]
        v = @view U[local_bottom_vector_indices]
        ldiv!(x, y, solver.local_block_solver, u, v)
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationParallel{T}, U::AbstractVector{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        local_top_vector_indices = solver.local_top_vector_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        u = @view U[local_top_vector_indices]
        v = @view U[local_bottom_vector_indices]
        ldiv!(u, v, solver.local_block_solver, u, v)
    end
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        local_top_vector_indices = solver.local_top_vector_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        for (this_X, this_U) ∈ zip(eachcol(X), eachcol(U))
            x = @view this_X[local_top_vector_indices]
            u = @view this_U[local_top_vector_indices]
            y = @view this_X[local_bottom_vector_indices]
            v = @view this_U[local_bottom_vector_indices]
            ldiv!(x, y, solver.local_block_solver, u, v)
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationParallel{T}, U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        local_top_vector_indices = solver.local_top_vector_indices
        local_bottom_vector_indices = solver.local_bottom_vector_indices
        for this_U ∈ eachcol(U)
            u = @view this_U[local_top_vector_indices]
            v = @view this_U[local_bottom_vector_indices]
            ldiv!(u, v, solver.local_block_solver, u, v)
        end
    end
    return nothing
end

end
