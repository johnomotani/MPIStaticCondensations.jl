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

export mpi_static_condensation, create_dimension, finalize_mpi_static_condensation!
public BlockSizesHeuristic, LevelMultiplier, FastSlow

using BlockArrays
using LinearAlgebra
using LinearAlgebra.LAPACK: getrf!
using MPI
using MPIDenseLUs
using MPISchurComplements
using MPISchurComplements: MPISchurComplementAFactorization
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

abstract type MPIStaticCondensation{Tf<:AbstractFloat} <: Factorization{Tf} end

struct MPIStaticCondensationNull{Tf<:AbstractFloat} <: MPIStaticCondensation{Tf} end

function get_partial_FixedSparseCSC_buffer(row_indices, col_indices,
                                           existing_buffer::NTuple{Nvar,NTuple{Nvar,Tbuff}},
                                           float_type=Float64) where {Nvar,Tbuff}
    # Initialize buffer with the same non-zero pattern as existing_buffer, but only for a
    # subset of rows given by row_indices and columns given by col_indices.
    @inbounds begin
        nrow = sum(length(ri) for ri ∈ row_indices)
        ncol = sum(length(ci) for ci ∈ col_indices)
        ind_type = eltype(row_indices[1])
        if nrow == 0 || ncol == 0
            return FixedSparseCSC(nrow, ncol, ones(ind_type, ncol + 1), ind_type[],
                                  zeros(eltype(existing_buffer[1][1]), 0))
        end

        colptr = ind_type[1]
        rowval = ind_type[]
        for (jvar, ci) ∈ enumerate(col_indices)
            for j ∈ ci
                row_offset = 0
                for (ivar, ri) ∈ enumerate(row_indices)
                    if isempty(ri)
                        continue
                    end
                    eb = existing_buffer[ivar][jvar]
                    existing_colptr = eb.colptr
                    if isa(eb, SharedSparseBuffer)
                        existing_rowval_list = eb.rowval_list
                    else
                        existing_rowval = eb.rowval
                    end
                    firstrow = first(ri)
                    lastrow = last(ri)
                    existing_col_start = existing_colptr[j]
                    existing_col_end = existing_colptr[j+1]-1
                    if isa(eb, SharedSparseBuffer)
                        existing_col_rowval = existing_rowval_list[j]
                    else
                        existing_col_rowval = @view existing_rowval[existing_col_start:existing_col_end]
                    end
                    n_existing = existing_col_end - existing_col_start + 1
                    if n_existing == 0 || first(existing_col_rowval) > lastrow || last(existing_col_rowval) < firstrow
                        # Definitely no overlapping entries in this variable block of the
                        # column, so skip.
                        row_offset += length(ri)
                        continue
                    end
                    count = max(searchsortedlast(existing_col_rowval, firstrow) - 1, 1)
                    for (i, i_global) ∈ enumerate(ri)
                        while count ≤ n_existing && existing_col_rowval[count] < i_global
                            count += 1
                        end
                        if count > n_existing
                            break
                        end
                        if existing_col_rowval[count] == i_global
                            push!(rowval, row_offset + i)
                        end
                    end
                    row_offset += length(ri)
                end
                push!(colptr, length(rowval) + 1)
            end
        end
        nzval = zeros(float_type, length(rowval))

        buffer = FixedSparseCSC(nrow, ncol, colptr, rowval, nzval)
        return buffer
    end
end

struct Dimension{Ti<:Integer}
    name::String
    n::Ti
    n_local::Ti
    nelement::Ti
    ngrid::Ti
    nrank::Ti
    irank::Ti
    global_inds::Vector{Ti}
    periodic::Bool
    dense_boundaries::Bool
    #has_lower_boundary::Bool
    #has_upper_boundary::Bool
    remove_boundaries::Bool

    function Dimension(; name::String, nelement::Ti, ngrid::Ti, nrank::Ti, irank::Ti,
                       periodic::Bool, dense_boundaries::Bool,
                       remove_boundaries::Union{Bool,Nothing}) where Ti <: Integer

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

        if remove_boundaries === nothing
            remove_boundaries = periodic || dense_boundaries
        end

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

        return new{Ti}(name, n, n_local, nelement, ngrid, nrank, irank, global_inds,
                       periodic, dense_boundaries, remove_boundaries)
    end
end

"""
    create_dimension(; name::String, nelement::Integer, ngrid::Integer,
                     nrank::Integer, irank::Integer, periodic::Bool,
                     dense_boundaries::Bool=false,
                     remove_boundaries::Union{Bool,Nothing}=nothing)

Create a `Dimension` object for input to the `dimensions` argument of
`mpi_static_condensation()`.

`name` is used to test whether two dimensions are the same - different dimensions should
not ever be given the same `name`.

Assume a continuous-Galerkin finite element discretization where there are `nelement`
elements and `ngrid` points in each element. The points at the boundary between two
elements are shared by both elements, so that the total number of grid points is
`nelement * (ngrid - 1) + 1`. When `periodic=true`, the grid is periodic and the last grid
point is a copy of the first. When the grid is distributed over different MPI blocks, the
point on the boundary between the blocks is duplicated on both blocks.

The number of shared-memory blocks that this dimension is divided into is given by
`nrank`, and the rank of the block that this process belongs to is `irank`.

`periodic` indicates whether the dimension is (`true`) or is not (`false`) periodic.

`dense_boundaries=true` can be passed to indicate that at a boundary of this dimension
each point can be coupled to every other point in the dimensions to the left of this one
in the list of dimensions. For example, if there are both velocity and spatial dimensions
\${v_\\parallel,v_\\perp,z,r}\$ and \$z\$ has `dense_boundaries=true`, then at the
\$z\$-boundaries the matrix may couple all points in \$v_\\parallel\$ and \$v_\\perp\$.

`remove_boundaries=true` can be passed if the grid at the boundary in this dimension does
not fit in to the sparsity pattern of the rest of the grid. In this case, the boundary
points can be included in the 'bottom vector' part of the Schur complement split on the
top level of the static-condensation solve, in order to ensure that the 'top vector' part
can be split by removing any element boundary. `remove_boundaries` is set by default to
`true` if `periodic=true` or `dense_boundaries=true`, so it should not usually be
necessary to pass `remove_boundaries` explicitly.
"""
function create_dimension(; name::String, nelement::Integer, ngrid::Integer,
                          nrank::Integer, irank::Integer, periodic::Bool,
                          dense_boundaries::Bool=false,
                          remove_boundaries::Union{Bool,Nothing}=nothing)
    return Dimension(; name, nelement, ngrid, nrank, irank, periodic, dense_boundaries,
                     remove_boundaries)
end

"""
    BlockSizesHeuristic

Abstract type for algorithms that specify or generate the list of block sizes at each
level in MPIStaticCondensations.
"""
abstract type BlockSizesHeuristic end

"""
    FastSlow{Ti<:Integer} <: BlockSizesHeuristic
    FastSlow(fast_multiplier::Integer=2, slow_threshold::Ti=8)

Algorithm for generating the block sizes at each level in MPIStaticCondensations.

First in the 'fast' phase, the block size at each level in each dimension is increased by
`fast_multiplier` from that at the previous level until it is greater than or equal to the
number of elements in that dimension (if the block size would be greater than the number
of elements, it is reduced to the number of elements). If the block sizes in all
dimensions reach the number of elements or if the current (maximum) block size is greater
than or equal to `slow_threshold`, stop.

Second in the 'slow' phase, the block size in one dimension is increased by a factor of
two at each level. The dimension chosen at each level:
1) Must have a block size less than the number of elements.
2) Of the dimensions satisfying (1), has the smallest block size (to keep the blocks as
   equally sized as possible).
3) Of the dimensions satisfying (2), has the most room left to expand (largest nelement).
4) Of the dimensions satisfying (3), is the left-most dimension, as this means the
   combined blocks will be as close as possible in the global index-space, which might
   marginally improve cache efficiency sometimes (??).
"""
struct FastSlow{Ti<:Integer} <: BlockSizesHeuristic
    fast_multiplier::Ti
    slow_threshold::Ti

    function FastSlow(fast_multiplier::Ti=2, slow_threshold::Ti=8) where Ti
        return new{Ti}(fast_multiplier, slow_threshold)
    end
end

function get_block_sizes(fs::FastSlow, dimensions::Vector{<:Dimension},
                         nelement_local_list::Vector{Ti}) where Ti <: Integer
    fm = fs.fast_multiplier
    st = fs.slow_threshold

    # 'Fast' expansion - all block sizes increase by the same factor at each level.
    current_size = Ti(1)
    block_sizes_list = [fill(current_size, length(dimensions))]
    max_fast = min(st, maximum(nelement_local_list))
    while current_size < max_fast
        current_size *= fm
        this_block_sizes = @. min(current_size, nelement_local_list)
        push!(block_sizes_list, this_block_sizes)
    end

    # 'Slow' expansion - one block size increases a factor of 2 at each level.
    while any(block_sizes_list[end] .< nelement_local_list)
        previous_block_size = block_sizes_list[end]

        # Dimension to have block size increased must not already be at the maximum block
        # size.
        candidate_dims = findall(previous_block_size .< nelement_local_list)

        # Try to keep block equally sized in each dimension, so pick from the dimensions
        # that have the smallest current block size.
        not_full_block_sizes = previous_block_size[candidate_dims]
        smallest_block_size = minimum(not_full_block_sizes)
        candidate_dims = candidate_dims[findall(not_full_block_sizes .== smallest_block_size)]

        # Try to increase first the dimensions where the block has the most space left to
        # expand, i.e. the ones with the largest number of elements.
        largest_nelement = maximum(nelement_local_list[candidate_dims])
        candidate_dims = candidate_dims[findall(nelement_local_list[candidate_dims] .== largest_nelement)]

        if isempty(candidate_dims)
            error("This should not happen - maybe the 'while' condition is incorrect?")
        end

        # For the tie-breaker, increase the left-most dimension, as then the entries being
        # combined are closer together in the global index-space, which might improve
        # cache efficiency (?? although this is probably a small effect if any!).
        dim_to_increase = first(candidate_dims)

        block_size = copy(previous_block_size)
        block_size[dim_to_increase] = min(block_size[dim_to_increase] * 2,
                                          nelement_local_list[dim_to_increase])

        push!(block_sizes_list, block_size)
    end

    return block_sizes_list
end

"""
    LevelMultiplier{Ti<:Integer} <: BlockSizesHeuristic
    LevelMultiplier(multiplier::Integer=2)

Algorithm for generating the block sizes at each level in MPIStaticCondensations.

The block size at each level in each dimension is increased by `multiplier` from that at
the previous level until it is greater than or equal to the number of elements in that
dimension (if the block size would be greater than the number of elements, it is reduced
to the number of elements). The last level is where block sizes in all dimensions reach
the number of elements.
"""
struct LevelMultiplier{Ti<:Integer} <: BlockSizesHeuristic
    multiplier::Ti

    function LevelMultiplier(multiplier::Ti=2) where Ti
        return new{Ti}(multiplier)
    end
end

function get_block_sizes(lm::LevelMultiplier, dimensions::Vector{<:Dimension},
                         nelement_local_list::Vector{Ti}) where Ti <: Integer
    m = lm.multiplier

    current_size = Ti(1)
    block_sizes_list = [fill(current_size, length(dimensions))]
    max_nelement = maximum(nelement_local_list)
    while current_size < max_nelement
        current_size *= m
        this_block_sizes = @. min(current_size, nelement_local_list)
        push!(block_sizes_list, this_block_sizes)
    end

    return block_sizes_list
end

DefaultBlockSizesHeuristic = FastSlow

include("shared_sparse_buffers.jl")
include("block_S.jl")
include("block_C.jl")
include("block_B.jl")
include("block_diagonal_solvers.jl")
include("blocked_schur_complement.jl")

# Function with no methods that we can import in the MUMPS extension.
function get_mumps_solver end

struct MPIStaticCondensationParallel{Nvar,Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{MPISchurComplement{Tf},BlockedSchurComplementSolver{Tf},MPIStaticCondensation{Tf}},Tranget,Trangept,Trangeb,Trangebs,Tdbr,Tdbb,Tbuff,Tsync,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
    nvar::Val{Nvar}
    n::Ti
    schur_complement_solver::Tsolver
    local_top_vector_indices::Tranget
    partial_local_top_vector_indices::Trangept
    partial_top_sub_range::UnitRange{Ti}
    local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_vector_indices::Trangeb
    this_shared_local_bottom_vector_no_overlap_indices::Trangeb
    this_shared_local_bottom_sub_selection_indices::Trangebs
    this_shared_local_bottom_sub_selection_no_overlap_indices::Trangeb
    this_shared_local_bottom_vector_repeat_indices::Trangeb
    this_shared_local_bottom_periodic_pairs::Matrix{Ti}
    dense_boundaries_ranges::Tdbr
    dense_boundaries_partial_ranges::Tdbr
    dense_boundaries_partial_buffer_ranges::Tdbr
    dense_boundaries_offsets::Matrix{Ti}
    dense_boundaries_buffers::Tdbb
    u_buffer::Tbuff
    v_buffer::Tbuff
    y_buffer::Tbuff
    has_periodic::Bool
    synchronize_shared::Tsync
    timer::Ttimer
end
Base.size(Alu::MPIStaticCondensationParallel) = (Alu.n, Alu.n)
Base.size(Alu::MPIStaticCondensationParallel, d::Integer) = size(Alu)[d]

function get_global_indices(dimensions::Vector{<:Dimension}, local_inds::Vector{<:Integer})
    n_local_tuple = Tuple(d.n_local for d ∈ dimensions)
    return get_global_indices(dimensions, local_inds, n_local_tuple)
end
function get_global_indices(dimensions::Vector{<:Dimension},
                            local_inds::Vector{<:Integer}, n_local_tuple)
    @inbounds begin
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
end

function apply_periodicity_to_indices(dimensions::Vector{<:Dimension},
                                      inds::Vector{<:Integer})
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return apply_periodicity_to_indices(dimensions, inds, n_tuple)
end
function apply_periodicity_to_indices(dimensions::Vector{<:Dimension},
                                      inds::Vector{Ti}, n_tuple) where Ti <: Integer
    @inbounds begin
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
end

function get_non_repeated_indices_and_repeats(dimensions::Vector{<:Dimension},
                                              inds::Vector{<:Integer})
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return get_non_repeated_indices_and_repeats(dimensions, inds, n_tuple)
end
function get_non_repeated_indices_and_repeats(dimensions::Vector{<:Dimension},
                                              inds::Vector{Ti}, n_tuple) where Ti <: Integer
    @inbounds begin
        if !any(d.periodic for d ∈ dimensions)
            # No periodic dimensions to account for.
            return copy(inds), Ti[]
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
end

function get_hypercube_position(iblock::AbstractVector{<:Integer}, nblock)
    # Use `block_sizes .> 1` filter so that we only increment the 'hypercube position' in
    # dimensions with a block size greater than one. For dimensions where the block size
    # is 1, there cannot be overlap between different blocks (there is only one block!),
    # so there is no need to allow for different positions in the output buffer.
    if all(nblock .== 1)
        return 1
    else
        return sum(((i - 1) % 2) * 2^(d-1) for (d, i) ∈ enumerate(iblock[nblock .> 1])) + 1
    end
end
function get_hypercube_position(flat_iblock::Integer, nblock)
    iblock = zeros(typeof(flat_iblock), length(nblock))

    # Convert flat_iblock to 0-based index for convenience.
    flat_iblock -= 1

    for (d, n) ∈ collect(enumerate(nblock))
        flat_iblock, iblock[d] = divrem(flat_iblock, n)
    end

    # Convert iblock back to 1-based indices.
    iblock .+= 1

    return get_hypercube_position(iblock, nblock)
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
    nblock::Vector{Ti}
    global_size::Ti
    global_offset::Ti
    local_offset::Ti
    global_bottom_vector_size::Ti
    local_bottom_vector_offset::Ti
    top_vector_indices::Vector{Ti}
    top_vector_offset_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    local_top_vector_offset_indices::Vector{Ti}
    iblock_list::Matrix{Ti}
    local_top_vector_a_block_indices::Vector{Vector{Ti}}
    local_top_vector_a_block_offset_indices::Vector{Vector{Ti}}
    a_block_off_diagonal_indices::Vector{Vector{Ti}}
    a_block_off_diagonal_bottom_vector_indices::Vector{Vector{Ti}}
    a_block_off_diagonal_bottom_vector_offset_indices::Vector{Vector{Ti}}
    n_subgroups::Ti
    subgroup_i::Ti
    subgroup_size::Ti
    block_comm::Tcomm
    bottom_vector_indices::Vector{Ti}
    bottom_vector_offset_indices::Vector{Ti}
    local_bottom_vector_indices::Vector{Ti}
    local_bottom_vector_offset_indices::Vector{Ti}
    local_bottom_vector_no_overlap_indices::Vector{Ti}
    local_bottom_vector_no_overlap_offset_indices::Vector{Ti}
    local_bottom_vector_no_overlap_sub_selection_indices::Vector{Ti}
    local_bottom_vector_no_overlap_sub_selection_offset_indices::Vector{Ti}
    local_bottom_vector_repeat_indices::Vector{Ti}
    local_bottom_vector_repeat_offset_indices::Vector{Ti}
    local_bottom_vector_periodic_pairs::Matrix{Ti}
    local_bottom_vector_offset_periodic_pairs::Matrix{Ti}
    #level_comm::Tcomm
    #level_distributed_comm::Tdcomm
    level_shared_comm::Tcomm
end

# Use `FakeComm` values for comm/distributed_comm/shared_comm to skip the comm splitting,
# for testing of the index generation.
function get_level_info_for_variable(
             dimensions::Vector{<:Dimension}, variable_dimensions::AbstractVector{Ti},
             level_indices::Vector{Ti}, block_sizes::Vector{Ti}, nblock::Vector{Ti},
             global_size::Ti, global_offset::Ti, local_offset::Ti,
             local_bottom_vector_offset::Ti, is_top_level::Bool, is_bottom_level::Bool,
             distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
             shared_comm::Union{MPI.Comm,FakeComm}) where Ti <: Integer
    @inbounds begin
        if length(dimensions) != length(block_sizes)
            error("dimensions and block_sizes should be the same length")
        end

        this_var_dims = dimensions[variable_dimensions]
        has_periodic = any(d.periodic for d ∈ this_var_dims)

        if shared_comm == MPI.COMM_NULL
            # This processor does no work on this level, so just fill level_info with dummy
            # values.
            return LevelInfo(; has_periodic, block_sizes, nblock, global_size=0,
                             global_offset=0, local_offset=0, global_bottom_vector_size=0,
                             local_bottom_vector_offset=0, top_vector_indices=Ti[],
                             top_vector_offset_indices=Ti[],
                             local_top_vector_indices=Ti[],
                             local_top_vector_offset_indices=Ti[],
                             local_top_vector_a_block_indices=Vector{Ti}[],
                             local_top_vector_a_block_offset_indices=Vector{Ti}[],
                             iblock_list=zeros(Ti, 2, 0),
                             a_block_off_diagonal_indices=Vector{Ti}[],
                             a_block_off_diagonal_bottom_vector_indices=Vector{Ti}[],
                             a_block_off_diagonal_bottom_vector_offset_indices=Vector{Ti}[],
                             n_subgroups=0, subgroup_i=-1, subgroup_size=0,
                             block_comm=shared_comm, bottom_vector_indices=Ti[],
                             bottom_vector_offset_indices=Ti[],
                             local_bottom_vector_indices=Ti[],
                             local_bottom_vector_offset_indices=Ti[],
                             local_bottom_vector_no_overlap_indices=Ti[],
                             local_bottom_vector_no_overlap_offset_indices=Ti[],
                             local_bottom_vector_no_overlap_sub_selection_indices=Ti[],
                             local_bottom_vector_no_overlap_sub_selection_offset_indices=Ti[],
                             local_bottom_vector_repeat_indices=Ti[],
                             local_bottom_vector_repeat_offset_indices=Ti[],
                             local_bottom_vector_periodic_pairs=zeros(Ti,2,0),
                             local_bottom_vector_offset_periodic_pairs=zeros(Ti,2,0),
                             level_shared_comm=shared_comm)
        end

        other_dimensions = setdiff(1:length(dimensions), variable_dimensions)
        this_var_block_sizes = block_sizes[variable_dimensions]

        # Divide the grid into blocks where the number of elements in a block in each
        # dimension is given by `block_sizes`.
        # The grid for a certain variable can only be divided once the block size reaches
        # the total number of elements in all of the 'other dimensions' - otherwise the
        # off-diagonal coupling from the variable (which couples all points in the 'other
        # dimensions') would couple the interiors of different blocks.
        # Also, if any of `other_dimensions` has `dense_boundaries=true`, we cannot split
        # this variable because the off-diagonal coupling to the 'dense boundary' puts
        # non-zeros into the matrix in places that would be missed by the B/C blocks.
        # Note `all()` returns `true` when the generator expression has no entries, i.e.
        # when there are no 'other dimensions'.
        split_variable = all(block_sizes[id] == dimensions[id].nelement && !dimensions[id].dense_boundaries
                             for id ∈ other_dimensions)
        if !split_variable
            boundary_indices = level_indices
            interior_indices = Ti[]
        else
            boundary_indices = Ti[]
            function get_boundary_indices!(idim, this_dim, flat_i)
                if this_dim ≤ 0
                    push!(boundary_indices, flat_i + 1)
                    return nothing
                end

                next_dim = this_dim - 1
                d = this_var_dims[this_dim]
                n = d.n
                n_local = d.n_local
                flat_i *= n

                # Add offset for distributed blocks.
                flat_i += d.irank * (n_local - 1)

                if idim == this_dim
                    bs = this_var_block_sizes[idim]
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
            for idim ∈ 1:length(this_var_dims)
                get_boundary_indices!(idim, length(this_var_dims), 0)
            end
            # There will be duplicated points in boundary_indices. Sort the list and remove
            # the duplicates.
            sort!(boundary_indices)
            unique!(boundary_indices)

            # Interior indices are all the indices in level_indices that are not boundary indices.
            interior_indices = setdiff(level_indices, boundary_indices)
        end

        # Get interior indices of the blocks that should be inverted by this processor.
        nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
        nblocks_list = [(nelement + bs - 1) ÷ bs
                        for (nelement, bs) ∈ zip(nelement_local_list, block_sizes)]
        total_nblocks = prod(nblocks_list)
        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)
        subgroup_size = max(shared_comm_size ÷ total_nblocks, 1)
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
            # To ensure the best possible load balance, we want to have as close as
            # possible to the same number of blocks from each hypercube position on each
            # process (when there are multiple processes per subgroup there is only one
            # block per subgroup, so this load balance optimisation is only relevant when
            # there is one process per subgroup and multiple blocks per subgroup). To
            # achieve this, first sort the blocks by hypercube position, then assign the
            # blocks to subgroups in round-robin style from the sorted list.
            flat_iblock_list = collect(1:total_nblocks)
            all_hypercube_positions = [get_hypercube_position(iblock, nblock)
                                       for iblock ∈ flat_iblock_list]
            hp_sortinds = sortperm(all_hypercube_positions)
            flat_iblock_list = flat_iblock_list[hp_sortinds]
            this_proc_blocks = flat_iblock_list[subgroup_i+1:n_subgroups:end]
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
                if this_dim ∈ other_dimensions
                    # This variable does not depend on this_dim, so 'skip'.
                    return get_block_interior_indices_from_dim!(next_dim, flat_i)
                end
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
            # When split_variable=false, this variable does not contribute to block
            # interiors.
            if split_variable
                get_block_interior_indices_from_dim!(length(dimensions), 0)
            end

            this_bbi = Ti[]
            block_boundary_indices[bi] = this_bbi
            function get_block_boundary_indices_from_dim!(this_dim, flat_i, boundary_dim)
                if this_dim ≤ 0
                    push!(this_bbi, flat_i + 1)
                    return nothing
                end
                next_dim = this_dim - 1
                if this_dim ∈ other_dimensions
                    # This variable does not depend on this_dim, so 'skip'.
                    if split_variable && this_dim == boundary_dim
                        return nothing
                    end
                    return get_block_boundary_indices_from_dim!(next_dim, flat_i,
                                                                boundary_dim)
                end
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
        local_top_vector_a_block_indices = [Ti[] for _ ∈ 1:length(block_interior_indices)]
        # The following search relies on both `interior_indices` and `block_interior_indices`
        # being sorted.
        for (this_block_interior_indices, this_local_top_vector_a_block_indices) ∈ zip(block_interior_indices, local_top_vector_a_block_indices)
            i_count = 1
            bi_count = 1
            while (i_count ≤ length(interior_indices)
                   && bi_count ≤ length(this_block_interior_indices))
                i = interior_indices[i_count]
                bi = this_block_interior_indices[bi_count]
                if i == bi
                    push!(this_local_top_vector_a_block_indices, i)
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
            block_boundary_indices = [get_non_repeated_indices_and_repeats(this_var_dims, bbi)[1]
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

        #global_top_vector_indices, _, _ = apply_periodicity_to_indices(this_var_dims, interior_indices)
        # Top vector indices can never be on periodic boundaries, so no need to apply
        # periodicity.
        global_top_vector_indices = interior_indices
        if is_top_level && is_bottom_level && has_periodic
            # need to handle periodicity
            global_bottom_vector_indices, _ =
                apply_periodicity_to_indices(this_var_dims, boundary_indices)
            global_bottom_vector_no_overlap_indices, global_bottom_vector_repeat_inds =
                get_non_repeated_indices_and_repeats(this_var_dims, boundary_indices)
            global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
        elseif is_bottom_level && has_periodic
            # need to handle periodicity
            global_bottom_vector_indices, global_bottom_vector_periodic_pairs =
                apply_periodicity_to_indices(this_var_dims, boundary_indices)
            global_bottom_vector_no_overlap_indices, _ =
                get_non_repeated_indices_and_repeats(this_var_dims, boundary_indices)
            global_bottom_vector_repeat_inds = Ti[]
        elseif is_top_level && has_periodic
            # need to handle periodicity
            global_bottom_vector_no_overlap_indices, global_bottom_vector_repeat_inds =
                get_non_repeated_indices_and_repeats(this_var_dims, boundary_indices)
            global_bottom_vector_indices = boundary_indices
            global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
        else
            global_bottom_vector_indices = boundary_indices
            global_bottom_vector_no_overlap_indices = boundary_indices
            global_bottom_vector_repeat_inds = Ti[]
            global_bottom_vector_periodic_pairs = zeros(Ti, 2, 0)
        end

        # Get the index within level_indices of the entries in block_boundary_indices.
        # The following search relies on both `a_block_off_diagonal_indices` and
        # `level_indices` being sorted.
        a_block_off_diagonal_indices = [Ti[] for _ ∈ 1:length(block_boundary_indices)]
        for (this_a_block_B_column_indices, this_block_boundary_indices) ∈ zip(a_block_off_diagonal_indices, block_boundary_indices)
            nbbi = length(this_block_boundary_indices)
            if nbbi == 0
                continue
            end
            l_count = max(searchsortedlast(level_indices, first(this_block_boundary_indices)) - 1, 1)
            bb_count = 1
            while l_count ≤ length(level_indices) && bb_count ≤ nbbi
                i = level_indices[l_count]
                bi = this_block_boundary_indices[bb_count]
                if i == bi
                    push!(this_a_block_B_column_indices, l_count)
                    l_count += 1
                    bb_count += 1
                elseif i < bi
                    l_count += 1
                else
                    bb_count += 1
                end
            end
        end

        # Get the index within bottom_vector_indices of the entries in
        # block_boundary_indices.  The following search relies on both
        # `a_block_off_diagonal_bottom_vector_indices` and `boundary_indices`
        # being sorted.
        a_block_off_diagonal_bottom_vector_indices = [Ti[] for _ ∈ 1:length(block_boundary_indices)]
        for (this_a_block_B_column_indices, this_block_boundary_indices) ∈ zip(a_block_off_diagonal_bottom_vector_indices, block_boundary_indices)
            nbbi = length(this_block_boundary_indices)
            if nbbi == 0
                continue
            end
            b_count = max(searchsortedlast(boundary_indices, first(this_block_boundary_indices)) - 1, 1)
            bb_count = 1
            while b_count ≤ length(boundary_indices) && bb_count ≤ nbbi
                i = boundary_indices[b_count]
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
        while (t_count ≤ nt || b_count ≤ nb || bno_count ≤ nbno || r_count ≤ nr || p_count ≤ np) && count ≤ n
            if t_count ≤ nt && b_count ≤ nb && interior_indices[t_count] == boundary_indices[b_count]
                error("interior_indices and boundary_indices should not overlap, got "
                      * "interior_indices[$t_count]=$(interior_indices[t_count]) and "
                      * "boundary_indices[$b_count]=$(boundary_indices[b_count]).")
            end
            if t_count ≤ nt && interior_indices[t_count] == level_indices[count]
                push!(local_top_vector_indices, count)
                t_count += 1
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
        if t_count != nt + 1 || b_count != nb + 1 || bno_count != nbno + 1 || r_count != nr + 1 || p_count != np + 1 || count != n + 1
            error("Did not find all indices in search. t_count=$t_count while nt+1=$(nt+1), "
                  * "b_count=$b_count while nb+1=$(nb+1), "
                  * "bno_count=$bno_count while nbno+1=$(nbno+1), "
                  * "r_count=$r_count while nr+1=$(nr+1), "
                  * "p_count=$p_count while np+1=$(np+1), "
                  * "count=$count while n+1=$(n+1).")
        end

        # The second row of entries (the 'source' points of the repeated pairs) are not
        # sorted, so cannot be found in the loop above. Need to search the whole of
        # `level_indices` for each second-row entry.
        for i ∈ 1:size(local_bottom_vector_periodic_pairs, 2)
            local_bottom_vector_periodic_pairs[2,i] = searchsortedfirst(level_indices, local_bottom_vector_periodic_pairs[2,i])
        end

        if is_bottom_level && any(d.periodic && d.nrank > 1 for d ∈ this_var_dims)
            println("Error: periodicity not properly supported for MPI-distributed "
                    * "this_var_dims yet.")
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

        local_bottom_vector_offset_periodic_pairs = copy(local_bottom_vector_periodic_pairs)
        local_bottom_vector_offset_periodic_pairs[1,:] .+= local_bottom_vector_offset
        local_bottom_vector_offset_periodic_pairs[2,:] .+= local_offset

        return LevelInfo(; has_periodic, block_sizes, nblock, global_size, global_offset,
                         local_offset, global_bottom_vector_size,
                         local_bottom_vector_offset,
                         top_vector_indices=global_top_vector_indices,
                         top_vector_offset_indices=global_top_vector_indices.+global_offset,
                         local_top_vector_indices=local_top_vector_indices,
                         local_top_vector_offset_indices=local_top_vector_indices.+local_offset,
                         iblock_list=iblock_list,
                         local_top_vector_a_block_indices=a_block_indices,
                         local_top_vector_a_block_offset_indices=[x .+ local_offset for x ∈ a_block_indices],
                         a_block_off_diagonal_indices=a_block_off_diagonal_indices,
                         a_block_off_diagonal_bottom_vector_indices=a_block_off_diagonal_bottom_vector_indices,
                         a_block_off_diagonal_bottom_vector_offset_indices=[x .+ local_bottom_vector_offset for x ∈ a_block_off_diagonal_bottom_vector_indices],
                         n_subgroups=n_subgroups, subgroup_i=subgroup_i,
                         subgroup_size=subgroup_size, block_comm=block_comm,
                         bottom_vector_indices=global_bottom_vector_indices,
                         bottom_vector_offset_indices=global_bottom_vector_indices.+global_offset,
                         local_bottom_vector_indices=local_bottom_vector_indices,
                         local_bottom_vector_offset_indices=local_bottom_vector_indices.+local_offset,
                         local_bottom_vector_no_overlap_indices=local_bottom_vector_no_overlap_indices,
                         local_bottom_vector_no_overlap_offset_indices=local_bottom_vector_no_overlap_indices.+local_offset,
                         local_bottom_vector_no_overlap_sub_selection_indices=local_bottom_vector_no_overlap_sub_selection_indices,
                         local_bottom_vector_no_overlap_sub_selection_offset_indices=local_bottom_vector_no_overlap_sub_selection_indices.+local_bottom_vector_offset,
                         local_bottom_vector_repeat_indices=local_bottom_vector_repeat_indices,
                         local_bottom_vector_repeat_offset_indices=local_bottom_vector_repeat_indices.+local_bottom_vector_offset,
                         local_bottom_vector_periodic_pairs=local_bottom_vector_periodic_pairs,
                         local_bottom_vector_offset_periodic_pairs,
                         level_shared_comm=shared_comm)
    end
end

function get_dense_boundaries_ranges_inner(outer_cartinds, outer_dims, nb, dense_dim_n,
                                           has_first_point, has_distinct_last_point,
                                           ind_type)
    db_ranges = UnitRange{ind_type}[]
    offset_step_size = prod(d.n_local for d ∈ outer_dims; init=1)
    for (count, outer_i) ∈ enumerate(outer_cartinds)
println("count=$count, outer_i=$outer_i")
        skip = false
        for (i, d) ∈ zip(Tuple(outer_i), outer_dims)
println("i=$i, d.name=", d.name)
            if (i == 1 && d.dense_boundaries && d.irank == 0) ||
                    (i == d.n_local && d.dense_boundaries && d.irank == d.nrank - 1)
                # Don't need to include points already included in the dense
                # boundary of an outer dimension.
println("skipping")
                skip = true
            end
        end
        if skip
            continue
        end
        offset = (count - 1) * offset_step_size
        if has_first_point
            push!(db_ranges, offset+1:offset+nb)
        end
        if has_distinct_last_point
            push!(db_ranges, offset+(dense_dim_n-1)*nb+1:offset+dense_dim_n*nb)
        end
println("has_first_point=$has_first_point, has_distinct_last_point=$has_distinct_last_point, db_ranges=$db_ranges")
    end
    return db_ranges
end

function get_dense_boundaries_ranges(idim, ivar, dimensions, variable_dimensions,
                                     ind_type)
    if !dimensions[idim].dense_boundaries
        error("In get_dense_boundaries_ranges(), "
              * "dimensions[idim].dense_boundaries should always be true.")
    end
    dense_dim = dimensions[idim]
    dense_dim_n = dense_dim.n
    vdims = variable_dimensions[ivar]
    this_var_idim = searchsortedfirst(vdims, idim)
    nb = prod(d.n for d ∈ dimensions[vdims[1:idim-1]]; init=1)
    if idim ∈ vdims
        outer_dims = Tuple(dimensions[vdims[1:idim-1]])
    else
        outer_dims = Tuple(dimensions[vdims[1:idim]])
    end
    return get_dense_boundaries_ranges_inner(
               CartesianIndices(Tuple(d.n_local for d ∈ outer_dims)), outer_dims,
               nb, dense_dim_n, dense_dim.irank == 0,
               dense_dim.irank == dense_dim.nrank - 1 && dense_dim_n > 1, ind_type)
end

function get_dense_boundaries_partial_range(full_ranges, shared_comm_rank,
                                            shared_comm_size, get_buffer_indices)
    range_lengths = collect(length(r) for r ∈ full_ranges)
    total_size = sum(range_lengths)
    local_entries_per_proc = (total_size + shared_comm_size - 1) ÷ shared_comm_size
    local_entries = shared_comm_rank*local_entries_per_proc+1:min((shared_comm_rank+1)*local_entries_per_proc,total_size)
    offsets = cumsum(vcat(0, range_lengths[1:end-1]))
    partial_buffer_ranges = [intersect(1:length(full_r), local_entries .- offset)
                             for (full_r, offset) ∈ zip(full_ranges, offsets)]
    if get_buffer_indices
        filter!(!isempty, partial_buffer_ranges)
        return partial_buffer_ranges
    else
        partial_ranges =
            [full_r[br] for (full_r, br) ∈ zip(full_ranges, partial_buffer_ranges)
             if !isempty(br)]
        return partial_ranges
    end
end

"""
    mpi_static_condensation(dimensions::Vector{<:Dimension};
                            variable_dimensions::Tuple=(nothing,),
                            block_sizes_heuristic::Union{BlockSizesHeuristic,Vector{<:Vector{<:Integer}}}=$DefaultBlockSizesHeuristic,
                            reduce_proc_count_with_blocks::Bool=false,
                            sparse_C_blocks::Bool=true,
                            comm::MPI.Comm=MPI.COMM_WORLD,
                            distributed_comm::Union{MPI.Comm,Nothing}=missing,
                            shared_comm::MPI.Comm=MPI.COMM_SELF,
                            allocate_shared_float::Union{Function,Nothing}=nothing,
                            allocate_shared_int::Union{Function,Nothing}=nothing,
                            synchronize_shared::Union{Function,Nothing}=nothing,
                            schur_tile_size::Union{Nothing,Integer}=nothing,
                            separate_Ainv_B::Bool=false,
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

When multiple variables are to be solved for, variables can be defined on grids given by
different subsets of `dimensions`. Which dimensions each variable is a function of is
specified by `variable_dimensions`, which is a Tuple with one entry per variable. Each
entry may be `nothing` (the whole of `dimensions` is used for that variable), or a
`Vector{<:Integer}` or `AbstractRange` that selects the dimensions for that variable from
`dimensions`.

`block_sizes_heuristic` determines how to set the block sizes at each level. A heuristic
(an instance of BlockSizesHeuristic) can be passed; currently `FastSlow` and
`LevelMultiplier` are available (default is `$DefaultBlockSizesHeuristic`). Alternatively
a list of block sizes for each level can be passed a `Vector{<:Vector{<:Integer}}` - the
block size for each dimension must increase or stay the same at each level, and must be an
integer multiple of the block sizes at the previous level, and at the final level the
block size in every dimension must be equal to the number of elements.

`reduce_proc_count_with_blocks` sets whether the number of processes involved in the solve
at each level is reduced when the number of blocks at that level is less than the total
number of processes. Usually reducing the number of processes is probably not helpful
(hence the default is `false`), but if MPI communication cost is the dominant bottleneck
it might be faster.

`sparse_C_blocks=false` can be passed to disable using sparse-matrix storage for the
non-zero blocks of the 'C' sub-matrices, on levels where the sub-blocks are not dense
matrices. Using sparse 'C' blocks should save some memory usage, and seems to improve the
'solve' performance slightly (although it should also give a small cost in the
matrix-processing part, `lu!()`).

When the 'fill in' of the matrix (number of non-zeros divided by total number of entries)
at some level exceeds `mumps_fill_in_threshold`, MUMPS is used to factorize/solve the
schur_complement matrix at that level instead of using another
`MPIStaticCondensationParallel`.

`comm` is divided into equally sized shared-memory blocks. `shared_comm` represents the
shared-memory block that this process belongs to - it must be a subset of `comm`, and its
members must be able to create shared-memory arrays.

`allocate_shared_float`, `allocate_shared_int`, and `synchronize_shared` are as required
by `mpi_schur_complement()`. `schur_tile_size` is passed to the `tile_size` argument of
`mpi_schur_complement()`.

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
                                 variable_dimensions::Tuple=(nothing,),
                                 block_sizes_heuristic::Union{BlockSizesHeuristic,Vector{<:Vector{<:Integer}}}=DefaultBlockSizesHeuristic,
                                 reduce_proc_count_with_blocks::Bool=false,
                                 sparse_C_blocks::Bool=true,
                                 mumps_fill_in_threshold::Number=1.0,
                                 comm::MPI.Comm=MPI.COMM_WORLD,
                                 distributed_comm::Union{MPI.Comm,Nothing}=missing,
                                 shared_comm::MPI.Comm=MPI.COMM_SELF,
                                 allocate_shared_float::F1=nothing,
                                 allocate_shared_int::F2=nothing,
                                 synchronize_shared::F3=nothing,
                                 schur_tile_size::Union{Nothing,Integer}=nothing,
                                 separate_Ainv_B::Bool=false,
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

    if mumps_fill_in_threshold < 1.0
        if reduce_proc_count_with_blocks
            error("reduce_proc_count_with_blocks=true is not compatible with using a "
                  * "MUMPS solver for the lowest level.")
        end
        if Base.get_extension(MPIStaticCondensations, :MumpsExt) === nothing
            error("MUMPS must be loaded when `mumps_fill_in_threshold` is set to a value "
                  * "less than 1.")
        end
        if any(d.periodic for d ∈ dimensions)
            error("MPIStaticCondensationMUMPS does not currently support periodicity.")
        end
    end

    nd = length(dimensions)
    variable_dimensions = Tuple(vdim === nothing ? (1:nd) : vdim
                                for vdim ∈ variable_dimensions)
    Nvar = length(variable_dimensions)

    n_blocks = comm_size ÷ shared_comm_size

    n_blocks_factors = factor(Vector, n_blocks)
    shared_comm_size_factors = factor(Vector, shared_comm_size)

    nelement_list = [ind_type(d.nelement) for d ∈ dimensions]
    nelement_local_list = [ind_type(d.nelement ÷ d.nrank) for d ∈ dimensions]
    if isa(block_sizes_heuristic, BlockSizesHeuristic)
        block_sizes_list = get_block_sizes(block_sizes_heuristic, dimensions,
                                           nelement_local_list)
    else
        block_sizes_list = block_sizes_heuristic
        # Check consistency of passed-in list.
        for i ∈ 1:length(block_sizes_list)-1
            bs = block_sizes_list[i]
            bs_next = block_sizes_list[i+1]
            if !all(bs_next .≥ bs)
                error("Block size for each dimension must not decrease at any level. "
                      * "Got block_sizes_list=$block_sizes_list.")
            end
            if !all(bs_next .% bs .== 0)
                error("Block size for each dimension must be an integer multiple of the "
                      * "block size at the previous level. "
                      * "Got block_sizes_list=$block_sizes_list.")
            end
        end
    end
    nblock_list = [(nelement_local_list .+ bs .- 1) .÷ bs for bs ∈ block_sizes_list]
    for (nb, bs) ∈ zip(nblock_list, block_sizes_list)
        if !all(nb .> 0)
            error("nb not positive for block_sizes=$bs.")
        end
    end
    total_local_nblock_list = [prod(nb) for nb ∈ nblock_list]

    dimensions_without_periodic = [Dimension(; name=d.name, nelement=d.nelement,
                                             ngrid=d.ngrid, nrank=d.nrank, irank=d.irank,
                                             periodic=false,
                                             dense_boundaries=d.dense_boundaries,
                                             remove_boundaries=(d.periodic || d.remove_boundaries))
                                   for d ∈ dimensions]

    n_levels = length(block_sizes_list)
    level_info_list = Vector{NTuple{Nvar,LevelInfo{ind_type,typeof(shared_comm)}}}(undef, n_levels)
    level_indices = Tuple(get_global_indices(dimensions_without_periodic[vdims],
                                             collect(1:prod(d.n_local for d ∈ dimensions[vdims])))
                          for vdims ∈ variable_dimensions)
    total_global_size = [prod(d.n for d ∈ dimensions[vdims]) for vdims ∈ variable_dimensions]
    level_global_size = total_global_size
    level_shared_comm = shared_comm
    level_shared_comm_size = shared_comm_size
    # When variable_dimensions has duplicate indices, we can re-use a single LevelInfo for
    # each of the duplicates.
    duplicate_var_first_position = Tuple(findfirst(variable_dimensions .== (vdim,))
                                         for vdim ∈ variable_dimensions)
    for (level, (block_sizes, nblock, total_local_nblock)) ∈
            enumerate(zip(block_sizes_list, nblock_list, total_local_nblock_list))
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
        this_level_info_list = Vector{LevelInfo{ind_type,typeof(level_shared_comm)}}(undef, Nvar)
        global_offset = 0
        local_offset = 0
        local_bottom_vector_offset = 0
        for (ivar, (this_var_dims, this_var_level_indices, this_var_level_global_size)) ∈
                enumerate(zip(variable_dimensions, level_indices, level_global_size))
            vfirst = duplicate_var_first_position[ivar]
            if vfirst < ivar
                # Copy this_level_info_list[vfirst] but need to add different offsets for
                # this variable.
                level_info_to_copy = this_level_info_list[vfirst]
                local_bottom_vector_offset_periodic_pairs = copy(level_info_to_copy.local_bottom_vector_periodic_pairs)
                local_bottom_vector_offset_periodic_pairs[1,:] .+= local_bottom_vector_offset
                local_bottom_vector_offset_periodic_pairs[2,:] .+= local_offset
                this_level_info =
                    LevelInfo(; has_periodic=level_info_to_copy.has_periodic,
                              block_sizes=level_info_to_copy.block_sizes,
                              nblock=level_info_to_copy.nblock,
                              global_size=level_info_to_copy.global_size, global_offset,
                              local_offset,
                              global_bottom_vector_size=level_info_to_copy.global_bottom_vector_size,
                              local_bottom_vector_offset,
                              top_vector_indices=level_info_to_copy.top_vector_indices,
                              top_vector_offset_indices=level_info_to_copy.top_vector_indices.+global_offset,
                              local_top_vector_indices=level_info_to_copy.local_top_vector_indices,
                              local_top_vector_offset_indices=level_info_to_copy.local_top_vector_indices.+local_offset,
                              iblock_list=level_info_to_copy.iblock_list,
                              local_top_vector_a_block_indices=level_info_to_copy.local_top_vector_a_block_indices,
                              local_top_vector_a_block_offset_indices=Vector{ind_type}[x .+ local_offset for x ∈ level_info_to_copy.local_top_vector_a_block_indices],
                              a_block_off_diagonal_indices=level_info_to_copy.a_block_off_diagonal_indices,
                              a_block_off_diagonal_bottom_vector_indices=level_info_to_copy.a_block_off_diagonal_bottom_vector_indices,
                              a_block_off_diagonal_bottom_vector_offset_indices=Vector{ind_type}[x .+ local_bottom_vector_offset for x ∈ level_info_to_copy.a_block_off_diagonal_bottom_vector_indices],
                              n_subgroups=level_info_to_copy.n_subgroups, subgroup_i=level_info_to_copy.subgroup_i,
                              subgroup_size=level_info_to_copy.subgroup_size, block_comm=level_info_to_copy.block_comm,
                              bottom_vector_indices=level_info_to_copy.bottom_vector_indices,
                              bottom_vector_offset_indices=level_info_to_copy.bottom_vector_indices.+global_offset,
                              local_bottom_vector_indices=level_info_to_copy.local_bottom_vector_indices,
                              local_bottom_vector_offset_indices=level_info_to_copy.local_bottom_vector_indices.+local_offset,
                              local_bottom_vector_no_overlap_indices=level_info_to_copy.local_bottom_vector_no_overlap_indices,
                              local_bottom_vector_no_overlap_offset_indices=level_info_to_copy.local_bottom_vector_no_overlap_indices.+local_offset,
                              local_bottom_vector_no_overlap_sub_selection_indices=level_info_to_copy.local_bottom_vector_no_overlap_sub_selection_indices,
                              local_bottom_vector_no_overlap_sub_selection_offset_indices=level_info_to_copy.local_bottom_vector_no_overlap_sub_selection_indices.+local_bottom_vector_offset,
                              local_bottom_vector_repeat_indices=level_info_to_copy.local_bottom_vector_repeat_indices,
                              local_bottom_vector_repeat_offset_indices=level_info_to_copy.local_bottom_vector_repeat_indices.+local_bottom_vector_offset,
                              local_bottom_vector_periodic_pairs=level_info_to_copy.local_bottom_vector_periodic_pairs,
                              local_bottom_vector_offset_periodic_pairs,
                              level_shared_comm=shared_comm)
            else
                this_level_info = get_level_info_for_variable(
                                      dims, this_var_dims, this_var_level_indices,
                                      block_sizes, nblock, this_var_level_global_size,
                                      global_offset, local_offset,
                                      local_bottom_vector_offset, level==1,
                                      level==n_levels, distributed_comm,
                                      level_shared_comm)
            end
            this_level_info_list[ivar] = this_level_info
            global_offset += total_global_size[ivar]
            local_offset += length(this_level_info.local_top_vector_indices) + length(this_level_info.local_bottom_vector_indices)
            local_bottom_vector_offset += length(this_level_info.local_bottom_vector_indices)
        end
        level_info_list[level] = tuple(this_level_info_list...)
        level_indices = Tuple(li.bottom_vector_indices for li ∈ level_info_list[level])
        level_global_size = [li.global_bottom_vector_size for li ∈ this_level_info_list]
    end

    level_allocate_shared_float_list =
        [(args...) -> allocate_shared_float(args...; comm=li[1].level_shared_comm)
         for li ∈ level_info_list]
    level_allocate_shared_int_list =
        [(args...) -> allocate_shared_int(args...; comm=li[1].level_shared_comm)
         for li ∈ level_info_list]
    schur_complement_buffer_info_list = []
    final_sc_solver_is_mumps = false
    final_level = n_levels
    for (level, (li, lai)) ∈ enumerate(zip(level_info_list[1:end-2],
                                           level_allocate_shared_int_list[1:end-2]))
        first_sc_info =
            get_shared_sparse_matrix_info(dimensions, li[1].level_shared_comm, lai,
                                          li[1].block_sizes, li[1].bottom_vector_indices,
                                          li[1].bottom_vector_indices,
                                          variable_dimensions[1], variable_dimensions[1];
                                          include_dense_boundaries=false, ind_type)
        sc_info = Matrix{typeof(first_sc_info)}(undef, Nvar, Nvar)
        sc_info[1,1] = first_sc_info
        for ivar ∈ 2:Nvar
            vfirst = duplicate_var_first_position[ivar]
            if vfirst < ivar
                sc_info[ivar,1] = sc_info[vfirst,1]
            else
                sc_info[ivar,1] =
                    get_shared_sparse_matrix_info(dimensions, li[1].level_shared_comm,
                                                  lai, li[1].block_sizes,
                                                  li[ivar].bottom_vector_indices,
                                                  li[1].bottom_vector_indices,
                                                  variable_dimensions[ivar],
                                                  variable_dimensions[1];
                                                  include_dense_boundaries=false,
                                                  ind_type)
            end
        end
        for jvar ∈ 2:Nvar
            vfirst = duplicate_var_first_position[jvar]
            if vfirst < jvar
                sc_info[:,jvar] .= @view sc_info[:,vfirst]
            else
                for ivar ∈ 1:Nvar
                    vfirst = duplicate_var_first_position[ivar]
                    if vfirst < ivar
                        sc_info[ivar,jvar] = sc_info[vfirst,jvar]
                    else
                        sc_info[ivar,jvar] =
                            get_shared_sparse_matrix_info(dimensions,
                                                          li[1].level_shared_comm, lai,
                                                          li[1].block_sizes,
                                                          li[ivar].bottom_vector_indices,
                                                          li[jvar].bottom_vector_indices,
                                                          variable_dimensions[ivar],
                                                          variable_dimensions[jvar];
                                                          include_dense_boundaries=false,
                                                          ind_type)
                    end
                end
            end
        end

        push!(schur_complement_buffer_info_list, sc_info)

        if level < n_levels && sum(sci.nzval_length for sci ∈ sc_info) / (sum(sci.m for sci ∈ sc_info[:,1]) * sum(sci.n for sci ∈ sc_info[1,:])) > mumps_fill_in_threshold
            final_sc_solver_is_mumps = true
            final_level = level + 1
            break
        end
    end

    schur_complement_nnz_list = [sum(var_block_sc.nzval_length for var_block_sc ∈ sc)
                                 for sc ∈ schur_complement_buffer_info_list]
    odd_buffer_size = Ref(maximum(schur_complement_nnz_list[1:2:end]; init=0))
    even_buffer_size = Ref(maximum(schur_complement_nnz_list[2:2:end]; init=0))
    if final_level > 1 && !final_sc_solver_is_mumps
        if level_info_list[end-1][1].level_shared_comm != MPI.COMM_NULL
            nbuff = sum(length(li.bottom_vector_indices) for li ∈ level_info_list[end-1])
            if n_levels % 2 == 0
                odd_buffer_size[] = max(odd_buffer_size[], nbuff^2)
            else
                even_buffer_size[] = max(even_buffer_size[], nbuff^2)
            end
        end
    end
    MPI.Allreduce!(odd_buffer_size, max, shared_comm)
    MPI.Allreduce!(even_buffer_size, max, shared_comm)
    if odd_buffer_size[] > 0
        odd_buffer = allocate_shared_float(odd_buffer_size[])
    else
        odd_buffer = zeros(data_type, 0)
    end
    if even_buffer_size[] > 0
        even_buffer = allocate_shared_float(even_buffer_size[])
    else
        even_buffer = zeros(data_type, 0)
    end
    schur_complement_buffer_list =
        [get_shared_sparse_buffer(bi, i % 2 == 0 ? even_buffer : odd_buffer)
         for (i, bi) ∈ enumerate(schur_complement_buffer_info_list)]
    if final_level > 1 && !final_sc_solver_is_mumps
        if level_info_list[final_level-1][1].level_shared_comm != MPI.COMM_NULL
            if final_level % 2 == 0
                second_last_buffer = odd_buffer
            else
                second_last_buffer = even_buffer
            end
            second_last_schur_complement_buffer =
                reshape(@view(second_last_buffer[1:nbuff^2]), nbuff, nbuff)
        else
            second_last_schur_complement_buffer = nothing
        end
    else
        second_last_schur_complement_buffer = nothing
    end

    # Create lowest level schur complement solver.
    # Use MUMPS if `mumps_fill_in_threshold` was exceeded.  Otherwise, use a parallelized
    # dense-matrix LU solver for the last Schur complement solve as long as the last Schur
    # complement matrix is not too small.
    if final_sc_solver_is_mumps
        if synchronize_shared === nothing
            level_synchronize_shared = () -> MPI.Barrier(shared_comm)
        else
            level_synchronize_shared = synchronize_shared
        end
        this_level_sc =
            get_mumps_solver(dimensions, schur_complement_buffer_list[end], comm,
                             level_synchronize_shared, timer)
    elseif level_info_list[end][1].level_shared_comm != MPI.COMM_NULL
        last_level_info = level_info_list[end]
        # Always use 'shared memory' solver on last level
        if last_level_info[1].block_comm != MPI.COMM_NULL
            block_comm_rank = MPI.Comm_rank(last_level_info[1].block_comm)
            block_comm_size = MPI.Comm_size(last_level_info[1].block_comm)
            if block_comm_size == shared_comm_size
                last_block_allocate_shared_float = allocate_shared_float
                last_block_allocate_shared_int = allocate_shared_int
                if synchronize_shared === nothing
                    last_block_synchronize_shared = () -> MPI.Barrier(last_level_info[1].block_comm)
                else
                    last_block_synchronize_shared = synchronize_shared
                end
            else
                last_block_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=last_level_info[1].block_comm)
                last_block_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=last_level_info[1].block_comm)
                last_block_synchronize_shared = () -> MPI.Barrier(last_level_info[1].block_comm)
            end

            if !all(nblock_list[length(level_info_list)] .== 1)
                # In principle we could have multiple blocks on the last level, but we
                # would need a more complicated setup for the `fake_level_info` below to
                # support that. It does not seem likely that it is a useful feature
                # (probably slower than continuing to combine down to one block), so for
                # simplicity just error if the last entry of block_sizes_list does not
                # define a single block covering the whole grid.
                last_level = length(level_info_list)
                error("Last entry of block_sizes_list should include the whole grid in "
                      * "one block. Last entry was $(block_sizes_list[last_level]), "
                      * "which corresponds to nblock=$(nblock_list[last_level]).")
            end
            # Fake the LevelInfo argument here, because this solver will be passed
            # matrices and rhs/solution vectors that do not need the 'top vector' entries
            # selecting out of them.
            ntop = sum(length(li.local_top_vector_indices) for li ∈ last_level_info)
            fake_level_info = ((global_size=ntop, global_bottom_vector_size=0,
                               local_top_vector_a_block_indices=(1:ntop,),
                               local_top_vector_a_block_offset_indices=(1:ntop,),
                               a_block_off_diagonal_indices=(1:0,),
                               block_comm=last_level_info[1].block_comm),)
            last_A_block_solver = get_block_diagonal_solver(fake_level_info, data_type,
                                                            true, timer, check_lu,
                                                            last_block_allocate_shared_float,
                                                            last_block_allocate_shared_int,
                                                            last_block_synchronize_shared)
        else
            last_A_block_solver = MPIStaticCondensationNull{data_type}()
        end
        last_level_shared_comm = last_level_info[1].level_shared_comm
        level_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=last_level_shared_comm)
        level_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=last_level_shared_comm)
        last_parallel_schur = sum(li.global_bottom_vector_size for li ∈ last_level_info) ≥ 1024
        if reduce_proc_count_with_blocks || synchronize_shared === nothing
            level_synchronize_shared = () -> MPI.Barrier(last_level_shared_comm)
        else
            level_synchronize_shared = synchronize_shared
        end
        this_level_sc =
            mpi_schur_complement(last_A_block_solver, data_type, data_type, data_type,
                                 vcat((li.top_vector_offset_indices for li ∈ last_level_info)...),
                                 vcat((li.bottom_vector_offset_indices for li ∈ last_level_info)...);
                                 comm=comm, shared_comm=last_level_shared_comm,
                                 distributed_comm=distributed_comm,
                                 allocate_shared_float=level_allocate_shared_float,
                                 allocate_shared_int=level_allocate_shared_int,
                                 synchronize_shared=level_synchronize_shared,
                                 use_sparse=false, sparse_Ainv_B=false,
                                 parallel_schur=last_parallel_schur,
                                 copy_input_to_dense_buffers=(n_levels == 1 && last_level_info[1].has_periodic),
                                 skip_factorization=true, schur_tile_size=schur_tile_size,
                                 check_lu=check_lu, timer=timer)
    else
        this_level_sc = MPIStaticCondensationNull{data_type}()
    end

    if any(d.dense_boundaries
           && (d.irank == 0 || d.irank == d.nrank - 1) for d ∈ dimensions)
        # The 'dense boundaries' entries are not needed until the lowest level, so we copy
        # them into a separate buffer before running the top level, and then add them back
        # into the lowest level matrix. This is more efficient than storing/copying them
        # at every level.
        dense_boundaries_ranges = [get_dense_boundaries_ranges(idim, ivar, dimensions,
                                                               variable_dimensions,
                                                               ind_type)
                                   for ivar ∈ 1:Nvar, idim ∈ 1:nd
                                   if dimensions[idim].dense_boundaries]

        dense_boundaries_partial_ranges =
            [get_dense_boundaries_partial_range(r, shared_comm_rank, shared_comm_size, false)
             for r ∈ dense_boundaries_ranges]
        dense_boundaries_partial_buffer_ranges =
            [get_dense_boundaries_partial_range(r, shared_comm_rank, shared_comm_size, true)
             for r ∈ dense_boundaries_ranges]

        buffer_sizes = [(sum(length(r[1]) for r ∈ dbr), length(dbr[1]))
                        for dbr ∈ eachcol(dense_boundaries_ranges)]
        dense_boundaries_offsets = hcat([cumsum(vcat(0, [length(r[1]) for r ∈ dbr[1:end-1]]))
                                         for dbr ∈ eachcol(dense_boundaries_ranges)]...)
        dense_boundaries_buffers = [allocate_shared_float(bs[1], bs[1], bs[2]) for bs ∈ buffer_sizes]
        if shared_comm_rank == 0
            for b ∈ dense_boundaries_buffers
                b .= 0.0
            end
        end
    else
        dense_boundaries_ranges = nothing
        dense_boundaries_partial_ranges = nothing
        dense_boundaries_partial_buffer_ranges = nothing
        dense_boundaries_partial_offsets = nothing
        dense_boundaries_buffers = nothing
    end

    this_level_schur_solver = nothing
    right_multiplication_buffer_storage = zeros(data_type, 0)
    C_dense_buffer_storage = zeros(data_type, 0)
    for (level, this_level_info) ∈ reverse(collect(enumerate(level_info_list[1:final_level])))
        if this_level_info[1].level_shared_comm == MPI.COMM_NULL
            this_level_schur_solver = MPIStaticCondensationNull{data_type}()
            continue
        end
        this_level_shared_comm = this_level_info[1].level_shared_comm
        level_allocate_shared_float = level_allocate_shared_float_list[level]
        level_allocate_shared_int = level_allocate_shared_int_list[level]
        this_level_comm_size = MPI.Comm_size(this_level_shared_comm)
        this_level_comm_rank = MPI.Comm_rank(this_level_shared_comm)

        if dense_boundaries_ranges === nothing
            this_dense_boundaries_ranges = nothing
            this_dense_boundaries_partial_ranges = nothing
            this_dense_boundaries_partial_buffer_ranges = nothing
            this_dense_boundaries_offsets = zeros(ind_type, 0, 0)
            this_dense_boundaries_buffers = nothing
        elseif n_levels == 1
            # Only one level, so only one element - no need to handle dense buffers.
            this_dense_boundaries_ranges = nothing
            this_dense_boundaries_partial_ranges = nothing
            this_dense_boundaries_partial_buffer_ranges = nothing
            this_dense_boundaries_offsets = zeros(ind_type, 0, 0)
            this_dense_boundaries_buffers = nothing
        elseif level == 1
            this_dense_boundaries_ranges = dense_boundaries_ranges
            this_dense_boundaries_partial_ranges = dense_boundaries_partial_ranges
            this_dense_boundaries_partial_buffer_ranges = dense_boundaries_partial_buffer_ranges
            this_dense_boundaries_offsets = dense_boundaries_offsets
            this_dense_boundaries_buffers = dense_boundaries_buffers
        elseif level == n_levels
            this_dense_boundaries_offsets = dense_boundaries_offsets
            this_dense_boundaries_buffers = dense_boundaries_buffers

            local_inds = [li.local_bottom_vector_indices for li ∈ level_info_list[level-1]]
            offsets = [li.local_offset for li ∈ this_level_info]
            # Need to get ranges within this, last level's (dense) matrix.
            # Indices of points in dense boundaries are always still present in the
            # indices at the bottom level, so don't need to check whether indices are
            # present.
            this_dense_boundaries_ranges =
                [[searchsortedfirst(li,first(r))+offset:searchsortedfirst(li,last(r))+offset for r ∈ dbr]
                 for (dbr, li, offset) ∈ zip(dense_boundaries_ranges, local_inds, offsets)]
            this_dense_boundaries_partial_ranges =
                [[searchsortedfirst(li,first(r))+offset:searchsortedfirst(li,last(r))+offset for r ∈ dbr]
                 for (dbr, li, offset) ∈ zip(dense_boundaries_partial_ranges, local_inds, offsets)]
            this_dense_boundaries_partial_buffer_ranges = dense_boundaries_partial_buffer_ranges
        else
            this_dense_boundaries_ranges = nothing
            this_dense_boundaries_partial_ranges = nothing
            this_dense_boundaries_partial_buffer_ranges = nothing
            this_dense_boundaries_offsets = zeros(ind_type, 0, 0)
            this_dense_boundaries_buffers = nothing
        end

        if level < final_level
            if reduce_proc_count_with_blocks || synchronize_shared === nothing
                level_synchronize_shared = () -> MPI.Barrier(this_level_shared_comm)
            else
                level_synchronize_shared = synchronize_shared
            end

            if this_level_info[1].block_comm == MPI.COMM_NULL
                block_comm_rank = 0
                block_comm_size = 1
            else
                block_comm_rank = MPI.Comm_rank(this_level_info[1].block_comm)
                block_comm_size = MPI.Comm_size(this_level_info[1].block_comm)
            end
            use_shared_blocks = this_level_info[1].subgroup_size > 1
            if block_comm_size == 1
                # No shared-memory parallelism.
                block_allocate_shared_float = (args...) -> Vector{data_type}(undef, args...)
                block_allocate_shared_int = (args...) -> Vector{ind_type}(undef, args...)
                block_synchronize_shared = () -> nothing
            elseif block_comm_size == shared_comm_size
                block_allocate_shared_float = allocate_shared_float
                block_allocate_shared_int = allocate_shared_int
                if synchronize_shared === nothing
                    block_synchronize_shared = () -> MPI.Barrier(this_level_info[1].block_comm)
                else
                    block_synchronize_shared = synchronize_shared
                end
            else
                block_allocate_shared_float = (args...) -> allocate_shared_float(args...; comm=this_level_info[1].block_comm)
                block_allocate_shared_int = (args...) -> allocate_shared_int(args...; comm=this_level_info[1].block_comm)
                block_synchronize_shared = () -> MPI.Barrier(this_level_info[1].block_comm)
            end

            if level == 1
                level_sparse_C_blocks = sparse_C_blocks
            else
                # If block_size_change==2, then only two blocks on the previous level
                # combined into each block on this level, in which case the C blocks will
                # be dense, and there is no point using 'sparse C blocks'.
                block_size_change = prod(block_sizes_list[level] .÷ block_sizes_list[level-1])
                level_sparse_C_blocks = block_size_change > 2 && sparse_C_blocks
            end
            this_level_sc =
                BlockedSchurComplementSolver(dimensions, level, this_level_info,
                                             schur_complement_buffer_list,
                                             second_last_schur_complement_buffer,
                                             this_level_schur_solver, use_shared_blocks,
                                             level_sparse_C_blocks,
                                             this_level_shared_comm,
                                             level_synchronize_shared,
                                             level_allocate_shared_float,
                                             block_synchronize_shared,
                                             block_allocate_shared_float,
                                             block_allocate_shared_int,
                                             right_multiplication_buffer_storage,
                                             C_dense_buffer_storage,
                                             this_dense_boundaries_ranges,
                                             this_dense_boundaries_partial_ranges,
                                             check_lu)
        end
        level_shared_comm_rank = MPI.Comm_rank(this_level_shared_comm)
        level_shared_comm_size = MPI.Comm_size(this_level_shared_comm)

        all_local_top_vector_offset_indices = vcat((li.local_top_vector_offset_indices for li ∈ this_level_info)...)
        all_local_bottom_vector_offset_indices = vcat((li.local_bottom_vector_offset_indices for li ∈ this_level_info)...)

        ntop = length(all_local_top_vector_offset_indices)
        nbottom = length(all_local_bottom_vector_offset_indices)

        if level == n_levels
            this_u_buffer = level_allocate_shared_float(ntop)
        else
            this_u_buffer = level_allocate_shared_float(0)
        end
        this_v_buffer = level_allocate_shared_float(nbottom)
        if level == n_levels
            this_y_buffer = level_allocate_shared_float(0)
        else
            this_y_buffer = level_allocate_shared_float(nbottom)
        end

        # Need to create a version of local_top_vector_offset_indices and
        # local_bottom_vector_offset_indices that is split into ranges to be handled in parallel
        # by all the processes in the shared-memory block.
        top_points_per_proc = (ntop + level_shared_comm_size - 1) ÷ level_shared_comm_size
        partial_top_sub_range = level_shared_comm_rank*top_points_per_proc+1:min((level_shared_comm_rank+1)*top_points_per_proc,ntop)

        bottom_points_per_proc = (nbottom + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset = level_shared_comm_rank*bottom_points_per_proc+1:min((level_shared_comm_rank+1)*bottom_points_per_proc,nbottom)
        this_shared_local_bottom_vector_offset_indices = all_local_bottom_vector_offset_indices[bottom_subset]
        this_shared_local_bottom_sub_selection_indices = (1:nbottom)[bottom_subset]

        nbottom_no_overlap = sum(length(li.local_bottom_vector_no_overlap_offset_indices) for li ∈ this_level_info)
        bottom_points_per_proc_no_overlap = (nbottom_no_overlap + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_no_overlap = level_shared_comm_rank*bottom_points_per_proc_no_overlap+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_no_overlap,nbottom_no_overlap)
        this_shared_local_bottom_vector_no_overlap_offset_indices = vcat((li.local_bottom_vector_no_overlap_offset_indices for li ∈ this_level_info)...)[bottom_subset_no_overlap]
        this_shared_local_bottom_sub_selection_no_overlap_offset_indices = vcat((li.local_bottom_vector_no_overlap_sub_selection_offset_indices for li ∈ this_level_info)...)[bottom_subset_no_overlap]

        nbottom_repeats = sum(length(li.local_bottom_vector_repeat_offset_indices) for li ∈ this_level_info)
        bottom_points_per_proc_repeats = (nbottom_repeats + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_repeats = level_shared_comm_rank*bottom_points_per_proc_repeats+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_repeats,nbottom_repeats)
        this_shared_local_bottom_vector_repeat_offset_indices = vcat((li.local_bottom_vector_repeat_offset_indices for li ∈ this_level_info)...)[bottom_subset_repeats]

        nbottom_periodic_pairs = sum(size(li.local_bottom_vector_periodic_pairs, 2) for li ∈ this_level_info)
        bottom_points_per_proc_periodic_pairs = (nbottom_periodic_pairs + level_shared_comm_size - 1) ÷ level_shared_comm_size
        bottom_subset_periodic_pairs = level_shared_comm_rank*bottom_points_per_proc_periodic_pairs+1:min((level_shared_comm_rank+1)*bottom_points_per_proc_periodic_pairs,nbottom_periodic_pairs)

        # On this processor, handle the overlap pairs where the destination of the overlap
        # is in this_shared_local_bottom_sub_selection_no_overlap_offset_indices.
        local_bottom_vector_periodic_pairs = hcat((li.local_bottom_vector_offset_periodic_pairs for li ∈ this_level_info)...)
        this_proc_pairs_inds = ind_type[]
        pair_count = 1
        bottom_count = 1
        this_local_offset = vcat((fill(li.local_offset, size(li.local_bottom_vector_periodic_pairs, 2)) for li ∈ this_level_info)...)
        this_local_bottom_vector_offset = vcat((fill(li.local_bottom_vector_offset, length(li.local_bottom_vector_no_overlap_sub_selection_offset_indices)) for li ∈ this_level_info)...)
        while pair_count ≤ size(local_bottom_vector_periodic_pairs, 2) && bottom_count ≤ length(this_shared_local_bottom_sub_selection_no_overlap_offset_indices)
            if local_bottom_vector_periodic_pairs[1,pair_count] == this_shared_local_bottom_sub_selection_no_overlap_offset_indices[bottom_count]
                push!(this_proc_pairs_inds, pair_count)
                pair_count += 1
                # Note that local_bottom_vector_periodic_pairs may have repeated first-row entries, so do not
                # increment bottom_count here.
            elseif local_bottom_vector_periodic_pairs[1,pair_count] < this_shared_local_bottom_sub_selection_no_overlap_offset_indices[bottom_count]
                pair_count += 1
            else
                bottom_count += 1
            end
        end
        this_shared_local_bottom_periodic_pairs = local_bottom_vector_periodic_pairs[:,this_proc_pairs_inds]

        this_level_schur_solver =
            MPIStaticCondensationParallel(Val(Nvar),
                                          sum(li.global_size for li ∈ this_level_info),
                                          this_level_sc, all_local_top_vector_offset_indices,
                                          @view(all_local_top_vector_offset_indices[partial_top_sub_range]),
                                          partial_top_sub_range,
                                          all_local_bottom_vector_offset_indices,
                                          this_shared_local_bottom_vector_offset_indices,
                                          this_shared_local_bottom_vector_no_overlap_offset_indices,
                                          this_shared_local_bottom_sub_selection_indices,
                                          this_shared_local_bottom_sub_selection_no_overlap_offset_indices,
                                          this_shared_local_bottom_vector_repeat_offset_indices,
                                          this_shared_local_bottom_periodic_pairs,
                                          this_dense_boundaries_ranges,
                                          this_dense_boundaries_partial_ranges,
                                          this_dense_boundaries_partial_buffer_ranges,
                                          this_dense_boundaries_offsets,
                                          this_dense_boundaries_buffers, this_u_buffer,
                                          this_v_buffer, this_y_buffer,
                                          any(li.has_periodic for li ∈ this_level_info),
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
    @inbounds begin
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
end
function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::SharedSparseBuffer,
                               rowinds, colinds) where {Tf,Ti}
    @inbounds begin
        colptr = A.colptr
        rowval = A.rowval
        nzval = A.nzval
        new_colptr = new_A.colptr
        new_rowval_list = new_A.rowval_list
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
            col_new_rowval = new_rowval_list[col]
            row_count = max(searchsortedlast(rowinds, col_new_rowval[1]) - 1, 1)
            for (row_i, new_i) ∈ enumerate(colstart:colend)
                rv = col_new_rowval[row_i]
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
end
function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::AbstractMatrix{Tf},
                               rowinds, colinds) where {Tf,Ti}
    @inbounds begin
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
end
@inline function update_sparse_matrix!(A::SparseMatrixCSC{Tf,Ti}, new_A::SubArray{Tf,2},
                                       rowinds, colinds) where {Tf,Ti}
    @inbounds begin
        full_rowinds, full_colinds = new_A.indices
        return @views update_sparse_matrix!(A, parent(new_A), full_rowinds[rowinds],
                                            full_colinds[colinds])
    end
end

function ldiv_Bmatrix!(::MPIStaticCondensationNull, B)
    return nothing
end

function lu!(solver::MPIStaticCondensationNull, A)
    return nothing
end

# Here `X` might be `Vector{T}` or `Vector{Vector{T}}`, so don't make type specification
# for it any stricter.
function ldiv!(X::AbstractVector, solver::MPIStaticCondensationNull{T},
               U::AbstractVector{T}) where T
    return nothing
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensationNull{T},
               U::AbstractMatrix{T}) where T
    return nothing
end
function ldiv!(solver::MPIStaticCondensationNull{T},
               U::AbstractVector{T}) where T
    return nothing
end
function ldiv!(solver::MPIStaticCondensationNull{T},
               U::AbstractMatrix{T}) where T
    return nothing
end

function lu!(solver::MPIStaticCondensationParallel{Nvar}, A) where Nvar
    @inbounds begin
        schur_complement_solver = solver.schur_complement_solver
        if isa(schur_complement_solver, MPISchurComplement)
            if isa(A, NTuple)
                # This is inefficient, but should only happen for 1-element grids where
                # there is only one level - this is only relevant for testing.
                @sc_timeit solver.timer "Static condensation lu! level-1 MPISchurComplement special handling" begin
                    this_A = Matrix(mortar(reshape([A[i%Nvar+1][i÷Nvar+1] for i ∈ 0:Nvar^2-1], Nvar, Nvar)))
                end
            else
                this_A = A

                dense_boundaries_buffers = solver.dense_boundaries_buffers
                if dense_boundaries_buffers !== nothing
                    # Add 'dense boundaries' matrix entries, that were removed from the
                    # matrix at the top level, back into this lowest-level matrix.
                    @sc_timeit solver.timer "Static condensation lu! $(size(A)) copy dense boundaries" begin
                        for (ranges, partial_ranges, partial_buffer_ranges, offsets, buffer) ∈
                                zip(eachcol(solver.dense_boundaries_ranges),
                                    eachcol(solver.dense_boundaries_partial_ranges),
                                    eachcol(solver.dense_boundaries_partial_buffer_ranges),
                                    solver.dense_boundaries_offsets,
                                    dense_boundaries_buffers)
                            for (col_ranges, col_buffer_ranges, col_offset) ∈ zip(partial_ranges, partial_buffer_ranges, offsets)
                                for (row_ranges, row_offset) ∈ zip(ranges, offsets)
                                    for (count, (cr, cbr, rr)) ∈ enumerate(zip(col_ranges,
                                                                               col_buffer_ranges,
                                                                               row_ranges))
                                        for (j, buffer_j) ∈ zip(cr, cbr .+ col_offset)
                                            for (i, buffer_i) ∈ zip(rr, row_offset+1:row_offset+length(rr))
                                                this_A[i,j] += buffer[buffer_i,buffer_j,count]
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
                local_top_vector_indices = solver.local_top_vector_indices
                local_bottom_vector_indices = solver.local_bottom_vector_indices
                a = ((@view(this_A[local_top_vector_indices,local_top_vector_indices]),),)
                b = @view this_A[local_top_vector_indices,local_bottom_vector_indices]
                c = @view this_A[local_bottom_vector_indices,local_top_vector_indices]
                d = @view this_A[local_bottom_vector_indices,local_bottom_vector_indices]
                update_schur_complement!(schur_complement_solver, a, b, c, d)
            end
        else
            if isa(A, AbstractMatrix) && Nvar == 1
                lu!(solver, ((A,),))
            else
                if solver.dense_boundaries_ranges !== nothing
                    # At the top level (this is the only level where
                    # dense_boundaries_ranges!==nothing), we copy the 'dense boundaries'
                    # entries into a separate buffer, as they are not needed until the
                    # lowest level, so it is more efficient not to store/copy them at
                    # every level.
                    @sc_timeit solver.timer "Static condensation lu! $(size(solver)) copy dense boundaries" begin
                        for (ranges, partial_ranges, partial_buffer_ranges, offsets, buffer) ∈
                                zip(eachcol(solver.dense_boundaries_ranges),
                                    eachcol(solver.dense_boundaries_partial_ranges),
                                    eachcol(solver.dense_boundaries_partial_buffer_ranges),
                                    solver.dense_boundaries_offsets,
                                    solver.dense_boundaries_buffers)
                            for (jvar, col_ranges, col_buffer_ranges, col_offset) ∈ zip(1:Nvar, partial_ranges, partial_buffer_ranges, offsets)
                                for (ivar, row_ranges, row_offset) ∈ zip(1:Nvar, ranges, offsets)
                                    var_A = A[ivar][jvar]
                                    colptr = var_A.colptr
                                    rowval = var_A.rowval
                                    nzval = var_A.nzval
                                    for (count, (cr, cbr, rr)) ∈ enumerate(zip(col_ranges,
                                                                               col_buffer_ranges,
                                                                               row_ranges))
                                        row_start = first(rr)
                                        row_end = last(rr)
                                        for (j, buffer_j) ∈ zip(cr, cbr .+ col_offset)
                                            col_start = colptr[j]
                                            col_end = colptr[j+1] - 1
                                            first_flat_i = searchsortedfirst(@view(rowval[col_start:col_end]), row_start) + col_start - 1
                                            for flat_i ∈ first_flat_i:col_end
                                                i = rowval[flat_i]
                                                if i > row_end
                                                    break
                                                end
                                                buffer_i = i - row_start + 1 + row_offset
                                                buffer[buffer_i,buffer_j,count] = nzval[flat_i]
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                @sc_timeit solver.timer "Static condensation lu! $(size(solver))" begin
                    lu!(schur_complement_solver, A)
                end
            end
        end
        return nothing
    end
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationParallel{Nvar,T},
               U::AbstractVector{T}) where {Nvar, T}
    @inbounds begin
        @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
            # MPISchurComplement allows the RHS and solution vectors to be the same array.
            # It is slightly faster to copy the data to/from local buffers than to use
            # @view with Vector{Int64} indices.
            schur_complement_solver = solver.schur_complement_solver
            partial_local_top_vector_indices = solver.partial_local_top_vector_indices
            partial_top_sub_range = solver.partial_top_sub_range
            this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
            this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
            this_shared_local_bottom_vector_no_overlap_indices = solver.this_shared_local_bottom_vector_no_overlap_indices
            this_shared_local_bottom_sub_selection_no_overlap_indices = solver.this_shared_local_bottom_sub_selection_no_overlap_indices
            this_shared_local_bottom_vector_repeat_indices = solver.this_shared_local_bottom_vector_repeat_indices
            this_shared_local_bottom_periodic_pairs = solver.this_shared_local_bottom_periodic_pairs
            y = solver.y_buffer
            v = solver.v_buffer
            if isa(schur_complement_solver, BlockedSchurComplementSolver)
                for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
                    # This loop uses 'no overlap' indices
                    # (`this_shared_local_bottom_vector_no_overlap_indices`) because when
                    # there are periodic dimensions, at the top level (and only the top level,
                    # not any intermediate levels) the right-hand-side entries need to be
                    # taken only from the non-repeated points, with the repeated points being
                    # zero-ed out.
                    v[i1] = U[i2]
                end
                for i ∈ this_shared_local_bottom_vector_repeat_indices
                    # Zero out repeated points at the top level
                    v[i] = 0.0
                end
                if solver.has_periodic
                    for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                        # At the bottom level, need to add any contributions that the top and
                        # intermediate levels have added to repeated points into the
                        # non-repeated points.
                        v[i1] += U[i2]
                    end
                end
                ldiv!(X, y, schur_complement_solver, U, v)
                for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices, this_shared_local_bottom_sub_selection_indices)
                    X[i1] = y[i2]
                end
            elseif isa(schur_complement_solver, MPISchurComplement)
                u = solver.u_buffer
                for (i1, i2) ∈ zip(partial_top_sub_range, partial_local_top_vector_indices)
                    u[i1] = U[i2]
                end
                for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
                    # This loop uses 'no overlap' indices
                    # (`this_shared_local_bottom_vector_no_overlap_indices`) because when
                    # there are periodic dimensions, at the top level (and only the top level,
                    # not any intermediate levels) the right-hand-side entries need to be
                    # taken only from the non-repeated points, with the repeated points being
                    # zero-ed out.
                    v[i1] = U[i2]
                end
                for i ∈ this_shared_local_bottom_vector_repeat_indices
                    # Zero out repeated points at the top level
                    v[i] = 0.0
                end
                if solver.has_periodic
                    for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                        # At the bottom level, need to add any contributions that the top and
                        # intermediate levels have added to repeated points into the
                        # non-repeated points.
                        v[i1] += U[i2]
                    end
                end
                solver.synchronize_shared()
                ldiv!(u, v, schur_complement_solver, u, v)
                for (i1, i2) ∈ zip(partial_local_top_vector_indices, partial_top_sub_range)
                    X[i1] = u[i2]
                end
                for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices, this_shared_local_bottom_sub_selection_indices)
                    X[i1] = v[i2]
                end
            else
                ldiv!(X, schur_complement_solver, U)
            end
        end
        return nothing
    end
end
function ldiv!(solver::MPIStaticCondensationParallel{Nvar,T}, U::AbstractVector{T}) where {Nvar, T}
    @inbounds begin
        @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
            # MPISchurComplement allows the RHS and solution vectors to be the same array.
            # It is slightly faster to copy the data to/from local buffers than to use
            # @view with Vector{Int64} indices.
            schur_complement_solver = solver.schur_complement_solver
            partial_local_top_vector_indices = solver.partial_local_top_vector_indices
            partial_top_sub_range = solver.partial_top_sub_range
            this_shared_local_bottom_vector_indices = solver.this_shared_local_bottom_vector_indices
            this_shared_local_bottom_sub_selection_indices = solver.this_shared_local_bottom_sub_selection_indices
            this_shared_local_bottom_vector_no_overlap_indices = solver.this_shared_local_bottom_vector_no_overlap_indices
            this_shared_local_bottom_sub_selection_no_overlap_indices = solver.this_shared_local_bottom_sub_selection_no_overlap_indices
            this_shared_local_bottom_vector_repeat_indices = solver.this_shared_local_bottom_vector_repeat_indices
            this_shared_local_bottom_periodic_pairs = solver.this_shared_local_bottom_periodic_pairs
            v = solver.v_buffer
            if isa(schur_complement_solver, BlockedSchurComplementSolver)
                y = solver.y_buffer
                for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
                    # This loop uses 'no overlap' indices
                    # (`this_shared_local_bottom_vector_no_overlap_indices`) because when
                    # there are periodic dimensions, at the top level (and only the top level,
                    # not any intermediate levels) the right-hand-side entries need to be
                    # taken only from the non-repeated points, with the repeated points being
                    # zero-ed out.
                    v[i1] = U[i2]
                end
                for i ∈ this_shared_local_bottom_vector_repeat_indices
                    # Zero out repeated points at the top level
                    v[i] = 0.0
                end
                if solver.has_periodic
                    for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                        # At the bottom level, need to add any contributions that the top and
                        # intermediate levels have added to repeated points into the
                        # non-repeated points.
                        v[i1] += U[i2]
                    end
                end
                ldiv!(U, y, schur_complement_solver, U, v)
                for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices,
                                   this_shared_local_bottom_sub_selection_indices)
                    U[i1] = y[i2]
                end
            elseif isa(schur_complement_solver, MPISchurComplement)
                u = solver.u_buffer
                for (i1, i2) ∈ zip(partial_top_sub_range, partial_local_top_vector_indices)
                    u[i1] = U[i2]
                end
                for (i1, i2) ∈ zip(this_shared_local_bottom_sub_selection_no_overlap_indices, this_shared_local_bottom_vector_no_overlap_indices)
                    # This loop uses 'no overlap' indices
                    # (`this_shared_local_bottom_vector_no_overlap_indices`) because when
                    # there are periodic dimensions, at the top level (and only the top level,
                    # not any intermediate levels) the right-hand-side entries need to be
                    # taken only from the non-repeated points, with the repeated points being
                    # zero-ed out.
                    v[i1] = U[i2]
                end
                for i ∈ this_shared_local_bottom_vector_repeat_indices
                    # Zero out repeated points at the top level
                    v[i] = 0.0
                end
                if solver.has_periodic
                    for (i1, i2) ∈ eachcol(this_shared_local_bottom_periodic_pairs)
                        # At the bottom level, need to add any contributions that the top and
                        # intermediate levels have added to repeated points into the
                        # non-repeated points.
                        v[i1] += U[i2]
                    end
                end
                solver.synchronize_shared()
                ldiv!(u, v, schur_complement_solver, u, v)
                for (i1, i2) ∈ zip(partial_local_top_vector_indices, partial_top_sub_range)
                    U[i1] = u[i2]
                end
                for (i1, i2) ∈ zip(this_shared_local_bottom_vector_indices,
                                   this_shared_local_bottom_sub_selection_indices)
                    U[i1] = v[i2]
                end
            else
                ldiv!(schur_complement_solver, U)
            end
        end
        return nothing
    end
end
function ldiv!(X::AbstractMatrix{T}, solver::MPIStaticCondensation{T},
               U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        for (this_X, this_U) ∈ zip(eachcol(X), eachcol(U))
            ldiv!(this_X, solver, this_U)
        end
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensation{T}, U::AbstractMatrix{T}) where T
    @sc_timeit solver.timer "Static condensation ldiv! $(size(solver, 1))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        for this_U ∈ eachcol(U)
            ldiv!(solver, this_U)
        end
    end
    return nothing
end

function finalize_mpi_static_condensation!(::MPIStaticCondensationNull)
    return nothing
end
function finalize_mpi_static_condensation!(solver::MPIStaticCondensation)
    schur_complement_solver = solver.schur_complement_solver
    if isa(schur_complement_solver, Union{MPIStaticCondensation,BlockedSchurComplementSolver})
        finalize_mpi_static_condensation!(schur_complement_solver)
    end
    return nothing
end
function finalize_mpi_static_condensation!(solver::BlockedSchurComplementSolver)
    schur_complement_solver = solver.schur_complement_solver
    if isa(schur_complement_solver, MPIStaticCondensation)
        finalize_mpi_static_condensation!(schur_complement_solver)
    end
    return nothing
end

end
