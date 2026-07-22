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

function get_partial_FixedSparseCSC_buffer(row_range, col_range, existing_buffer,
                                           float_type=Float64)
    # Initialize buffer with the same non-zero pattern as existing_buffer, but only for a
    # subset of rows given by row_range and columns given by col_range.
    @inbounds begin
        nrow = length(row_range)
        ncol = length(col_range)
        ind_type = eltype(row_range)
        if nrow == 0 || ncol == 0
            return FixedSparseCSC(nrow, ncol, ones(ind_type, ncol + 1), ind_type[],
                                  zeros(eltype(existing_buffer), 0))
        end

        colptr = ind_type[1]
        rowval = ind_type[]
        firstrow = first(row_range)
        lastrow = last(row_range)
        existing_colptr = existing_buffer.colptr
        if isa(existing_buffer, SharedSparseBuffer)
            existing_rowval_list = existing_buffer.rowval_list
        else
            existing_rowval = existing_buffer.rowval
        end
        for j ∈ col_range
            existing_col_start = existing_colptr[j]
            existing_col_end = existing_colptr[j+1]-1
            if isa(existing_buffer, SharedSparseBuffer)
                existing_col_rowval = existing_rowval_list[j]
            else
                existing_col_rowval = @view existing_rowval[existing_col_start:existing_col_end]
            end
            n_existing = existing_col_end - existing_col_start + 1
            if n_existing == 0 || first(existing_col_rowval) > lastrow || last(existing_col_rowval) < firstrow
                # Definitely no overlapping entries in this column, so skip.
                push!(colptr, length(rowval) + 1)
                continue
            end
            count = max(searchsortedlast(existing_col_rowval, firstrow) - 1, 1)
            for (i, i_global) ∈ enumerate(row_range)
                while count ≤ n_existing && existing_col_rowval[count] < i_global
                    count += 1
                end
                if count > n_existing
                    break
                end
                if existing_col_rowval[count] == i_global
                    push!(rowval, i)
                end
            end
            push!(colptr, length(rowval) + 1)
        end
        nzval = zeros(float_type, length(rowval))

        buffer = FixedSparseCSC(nrow, ncol, colptr, rowval, nzval)
        return buffer
    end
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
    dense_boundaries::Bool
    #has_lower_boundary::Bool
    #has_upper_boundary::Bool
    remove_boundaries::Bool

    function Dimension(; nelement::Ti, ngrid::Ti, nrank::Ti, irank::Ti, periodic::Bool,
                       dense_boundaries::Bool,
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

        return new{Ti}(n, n_local, nelement, ngrid, nrank, irank, global_inds, periodic,
                       dense_boundaries, remove_boundaries)
    end
end

"""
    create_dimension(; nelement::Integer, ngrid::Integer, nrank::Integer,
                     irank::Integer, periodic::Bool, dense_boundaries::Bool=false,
                     remove_boundaries::Bool=nothing)

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
function create_dimension(; nelement::Integer, ngrid::Integer, nrank::Integer,
                          irank::Integer, periodic::Bool, dense_boundaries::Bool=false,
                          remove_boundaries::Union{Bool,Nothing}=nothing)
    return Dimension(; nelement, ngrid, nrank, irank, periodic, dense_boundaries,
                     remove_boundaries)
end

include("shared_sparse_buffers.jl")
include("block_S.jl")
include("block_C.jl")
include("block_B.jl")
include("block_diagonal_solvers.jl")
include("blocked_schur_complement.jl")
include("mumps_solver.jl")

struct MPIStaticCondensationParallel{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{MPISchurComplement{Tf},BlockedSchurComplementSolver{Tf},MPIStaticCondensationMUMPS{Tf}},Tranget,Trangept,Trangeb,Trangebs,Tbuff,Tsync,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation{Tf}
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
    global_bottom_vector_size::Ti
    top_vector_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    iblock_list::Matrix{Ti}
    local_top_vector_a_block_indices::Vector{Vector{Ti}}
    a_block_off_diagonal_indices::Vector{Vector{Ti}}
    a_block_off_diagonal_bottom_vector_indices::Vector{Vector{Ti}}
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
                      block_sizes::Vector{Ti}, nblock::Vector{Ti}, global_size::Ti,
                      is_top_level::Bool, is_bottom_level::Bool,
                      distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
                      shared_comm::Union{MPI.Comm,FakeComm}) where Ti <: Integer
    @inbounds begin
        if length(dimensions) != length(block_sizes)
            error("dimensions and block_sizes should be the same length")
        end

        has_periodic = any(d.periodic for d ∈ dimensions)

        if shared_comm == MPI.COMM_NULL
            # This processor does no work on this level, so just fill level_info with dummy
            # values.
            return LevelInfo(; has_periodic, block_sizes, nblock, global_size=0,
                             global_bottom_vector_size=0, top_vector_indices=Ti[],
                             local_top_vector_indices=Ti[],
                             local_top_vector_a_block_indices=Vector{Ti}[],
                             iblock_list=zeros(Ti, 2, 0),
                             a_block_off_diagonal_indices=Vector{Ti}[],
                             a_block_off_diagonal_bottom_vector_indices=Vector{Ti}[],
                             n_subgroups=0, subgroup_i=-1, subgroup_size=0,
                             block_comm=shared_comm, bottom_vector_indices=Ti[],
                             local_bottom_vector_indices=Ti[],
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
        # There will be duplicated points in boundary_indices. Sort the list and remove
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
        if t_count != nt + 1 || b_count != nb + 1 || r_count != nr + 1 || p_count != np + 1
            error("Did not find all indices in search. t_count=$t_count while nt+1=$(nt+1). "
                  * "t_count=$t_count while nt+1=$(nt+1), "
                  * "b_count=$b_count while nb+1=$(nb+1), "
                  * "r_count=$r_count while nr+1=$(nr+1), "
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

        return LevelInfo(; has_periodic, block_sizes, nblock, global_size,
                         global_bottom_vector_size,
                         top_vector_indices=global_top_vector_indices,
                         local_top_vector_indices=local_top_vector_indices,
                         iblock_list=iblock_list,
                         local_top_vector_a_block_indices=a_block_indices,
                         a_block_off_diagonal_indices=a_block_off_diagonal_indices,
                         a_block_off_diagonal_bottom_vector_indices=a_block_off_diagonal_bottom_vector_indices,
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
end

"""
    mpi_static_condensation(dimensions::Vector{<:Dimension};
                            level_multiplier::Integer=2,
                            reduce_proc_count_with_blocks::Bool=false,
                            sparse_C_blocks::Bool=false,
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

`level_multiplier` gives the factor by which the block size is increased in each dimension
at each level.

`reduce_proc_count_with_blocks` sets whether the number of processes involved in the solve
at each level is reduced when the number of blocks at that level is less than the total
number of processes. Usually reducing the number of processes is probably not helpful
(hence the default is `false`), but if MPI communication cost is the dominant bottleneck
it might be faster.

`sparse_C_blocks=true` can be passed to use sparse-matrix storage for the non-zero blocks
of the 'C' sub-matrices. This will save some memory usage, but probably comes with a
slight performance penalty.

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
                                 level_multiplier::Integer=2,
                                 reduce_proc_count_with_blocks::Bool=false,
                                 sparse_C_blocks::Bool=false,
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
    n_blocks = comm_size ÷ shared_comm_size

    n_blocks_factors = factor(Vector, n_blocks)
    shared_comm_size_factors = factor(Vector, shared_comm_size)

    # Not sure if this is necessarily the most efficient choice for all grids and numbers
    # of processes - everything following should work for any set of block_sizes, so other
    # choices could be made here.
    block_sizes_list = [ones(ind_type, length(dimensions))]
    nelement_list = [d.nelement for d ∈ dimensions]
    nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
    nblock_list = [nelement_local_list]
    total_local_nblock = prod(nelement_local_list)
    total_local_nblock_list = [total_local_nblock]
    while total_local_nblock > 1
        previous_block_sizes = block_sizes_list[end]
        this_block_sizes = @. min(previous_block_sizes .* level_multiplier, nelement_local_list)
        local_nblock_list = @. (nelement_local_list + this_block_sizes - 1) ÷ this_block_sizes
        total_local_nblock = prod(local_nblock_list)
        push!(total_local_nblock_list, total_local_nblock)
        push!(nblock_list, local_nblock_list)
        push!(block_sizes_list, this_block_sizes)
    end

    dimensions_without_periodic = [Dimension(; nelement=d.nelement, ngrid=d.ngrid,
                                             nrank=d.nrank, irank=d.irank, periodic=false,
                                             dense_boundaries=d.dense_boundaries,
                                             remove_boundaries=(d.periodic || d.remove_boundaries))
                                   for d ∈ dimensions]

    n_levels = length(block_sizes_list)
    level_info_list = Vector{LevelInfo{ind_type,typeof(shared_comm)}}(undef, n_levels)
    level_indices = get_global_indices(dimensions_without_periodic,
                                       collect(1:prod(d.n_local for d ∈ dimensions)))
    level_global_size = prod(d.n for d ∈ dimensions)
    level_shared_comm = shared_comm
    level_shared_comm_size = shared_comm_size
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
        this_level_info = split_matrix(dims, level_indices, block_sizes, nblock,
                                       level_global_size, level==1, level==n_levels,
                                       distributed_comm, level_shared_comm)
        level_info_list[level] = this_level_info
        level_indices = this_level_info.bottom_vector_indices
        level_global_size = this_level_info.global_bottom_vector_size
    end

    level_allocate_shared_float_list =
        [(args...) -> allocate_shared_float(args...; comm=li.level_shared_comm)
         for li ∈ level_info_list]
    level_allocate_shared_int_list =
        [(args...) -> allocate_shared_int(args...; comm=li.level_shared_comm)
         for li ∈ level_info_list]
    schur_complement_buffer_info_list = []
    final_sc_solver_is_mumps = false
    final_level = n_levels
    for (level, (li, lai)) ∈ enumerate(zip(level_info_list[1:end-2],
                                           level_allocate_shared_int_list[1:end-2]))
        sc_info =
            get_shared_sparse_matrix_info(dimensions, li.level_shared_comm, lai,
                                          li.block_sizes, li.bottom_vector_indices,
                                          li.bottom_vector_indices; ind_type)
        push!(schur_complement_buffer_info_list, sc_info)

        if 1 < level < n_levels && sc_info.nzval_length / (sc_info.m * sc_info.n) > mumps_fill_in_threshold
            final_sc_solver_is_mumps = true
            final_level = level + 1
            break
        end
    end

    schur_complement_nnz_list = [sc.nzval_length
                                 for sc ∈ schur_complement_buffer_info_list]
    odd_buffer_size = Ref(maximum(schur_complement_nnz_list[1:2:end]; init=0))
    even_buffer_size = Ref(maximum(schur_complement_nnz_list[2:2:end]; init=0))
    if final_level > 1 && !final_sc_solver_is_mumps
        if level_info_list[end-1].level_shared_comm != MPI.COMM_NULL
            nbuff = length(level_info_list[end-1].bottom_vector_indices)
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
        if level_info_list[final_level-1].level_shared_comm != MPI.COMM_NULL
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
        if reduce_proc_count_with_blocks
            error("reduce_proc_count_with_blocks=true is not compatible with using a "
                  * "MUMPS solver for the lowest level.")
        end
        if synchronize_shared === nothing
            level_synchronize_shared = () -> MPI.Barrier(shared_comm)
        else
            level_synchronize_shared = synchronize_shared
        end
        this_level_sc =
            MPIStaticCondensationMUMPS(schur_complement_buffer_list[end], comm,
                                       level_synchronize_shared, timer)
    elseif level_info_list[end].level_shared_comm != MPI.COMM_NULL
        last_level_info = level_info_list[end]
        last_use_shared_blocks = (length(level_info_list) > 1
                                  && length(last_level_info.local_top_vector_a_block_indices) == 1
                                  && MPI.Comm_size(last_level_info.block_comm) > 1)
        # Always use 'shared memory' solver on last level
        if last_level_info.block_comm != MPI.COMM_NULL
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
            # Fake the LevelInfo argument here, because this solver will be passed
            # matrices and rhs/solution vectors that do not need the 'top vector' entries
            # selecting out of them.
            ntop = length(last_level_info.local_top_vector_indices)
            fake_level_info = (global_size=ntop, global_bottom_vector_size=0,
                               local_top_vector_a_block_indices=(1:ntop,),
                               a_block_off_diagonal_indices=(1:0,),
                               block_comm=last_level_info.block_comm)
            last_A_block_solver = get_block_diagonal_solver(fake_level_info, data_type,
                                                            length(level_info_list) == 1,
                                                            true, timer, check_lu,
                                                            last_block_allocate_shared_float,
                                                            last_block_allocate_shared_int,
                                                            last_block_synchronize_shared)
        else
            last_A_block_solver = MPIStaticCondensationNull{data_type}()
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
                                 copy_input_to_dense_buffers=(n_levels == 1 && last_level_info.has_periodic),
                                 skip_factorization=true, schur_tile_size=schur_tile_size,
                                 check_lu=check_lu, timer=timer)
    else
        this_level_sc = MPIStaticCondensationNull{data_type}()
    end

    this_level_schur_solver = nothing
    right_multiplication_buffer_storage = zeros(data_type, 0)
    C_dense_buffer_storage = zeros(data_type, 0)
    for (level, this_level_info) ∈ reverse(collect(enumerate(level_info_list[1:final_level])))
        if this_level_info.level_shared_comm == MPI.COMM_NULL
            this_level_schur_solver = MPIStaticCondensationNull{data_type}()
            continue
        end
        this_level_shared_comm = this_level_info.level_shared_comm
        level_allocate_shared_float = level_allocate_shared_float_list[level]
        level_allocate_shared_int = level_allocate_shared_int_list[level]
        this_level_comm_size = MPI.Comm_size(this_level_shared_comm)
        this_level_comm_rank = MPI.Comm_rank(this_level_shared_comm)
        if level < final_level
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
            if block_comm_size == 1
                # No shared-memory parallelism.
                block_allocate_shared_float = (args...) -> Vector{data_type}(undef, args...)
                block_allocate_shared_int = (args...) -> Vector{ind_type}(undef, args...)
                block_synchronize_shared = () -> nothing
            elseif block_comm_size == shared_comm_size
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

            this_level_sc =
                BlockedSchurComplementSolver(dimensions, level, this_level_info,
                                             schur_complement_buffer_list,
                                             second_last_schur_complement_buffer,
                                             this_level_schur_solver, use_shared_blocks,
                                             sparse_C_blocks, this_level_shared_comm,
                                             level_synchronize_shared,
                                             level_allocate_shared_float,
                                             level_allocate_shared_int,
                                             block_synchronize_shared,
                                             block_allocate_shared_float,
                                             block_allocate_shared_int,
                                             right_multiplication_buffer_storage,
                                             C_dense_buffer_storage, check_lu)
        end
        level_shared_comm_rank = MPI.Comm_rank(this_level_shared_comm)
        level_shared_comm_size = MPI.Comm_size(this_level_shared_comm)
        if level == n_levels
            this_u_buffer = level_allocate_shared_float(length(this_level_info.local_top_vector_indices))
        else
            this_u_buffer = level_allocate_shared_float(0)
        end
        this_v_buffer = level_allocate_shared_float(length(this_level_info.local_bottom_vector_indices))
        if level == n_levels
            this_y_buffer = level_allocate_shared_float(0)
        else
            this_y_buffer = level_allocate_shared_float(length(this_level_info.local_bottom_vector_indices))
        end

        # Need to create a version of local_top_vector_indices and
        # local_bottom_vector_indices that is split into ranges to be handled in parallel
        # by all the processes in the shared-memory block.
        ntop = length(this_level_info.local_top_vector_indices)
        top_points_per_proc = (ntop + level_shared_comm_size - 1) ÷ level_shared_comm_size
        partial_top_sub_range = level_shared_comm_rank*top_points_per_proc+1:min((level_shared_comm_rank+1)*top_points_per_proc,ntop)

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

        this_level_schur_solver =
            MPIStaticCondensationParallel(this_level_info.global_size, this_level_sc,
                                          this_level_info.local_top_vector_indices,
                                          @view(this_level_info.local_top_vector_indices[partial_top_sub_range]),
                                          partial_top_sub_range,
                                          this_level_info.local_bottom_vector_indices,
                                          this_shared_local_bottom_vector_indices,
                                          this_shared_local_bottom_vector_no_overlap_indices,
                                          this_shared_local_bottom_sub_selection_indices,
                                          this_shared_local_bottom_sub_selection_no_overlap_indices,
                                          this_shared_local_bottom_vector_repeat_indices,
                                          this_shared_local_bottom_periodic_pairs,
                                          this_u_buffer, this_v_buffer, this_y_buffer,
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

function lu!(solver::MPIStaticCondensationParallel, A)
    @inbounds begin
        @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
            schur_complement_solver = solver.schur_complement_solver
            if isa(schur_complement_solver, MPISchurComplement)
                local_top_vector_indices = solver.local_top_vector_indices
                local_bottom_vector_indices = solver.local_bottom_vector_indices
                a = @view A[local_top_vector_indices,local_top_vector_indices]
                b = @view A[local_top_vector_indices,local_bottom_vector_indices]
                c = @view A[local_bottom_vector_indices,local_top_vector_indices]
                d = @view A[local_bottom_vector_indices,local_bottom_vector_indices]
                update_schur_complement!(schur_complement_solver, a, b, c, d)
            else
                lu!(schur_complement_solver, A)
            end
        end
        return nothing
    end
end

function ldiv!(X::AbstractVector{T}, solver::MPIStaticCondensationParallel{T},
               U::AbstractVector{T}) where T
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
            if isa(schur_complement_solver, MPIStaticCondensationMUMPS)
                ldiv!(X, schur_complement_solver, U)
            elseif isa(schur_complement_solver, BlockedSchurComplementSolver)
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
            else
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
            end
        end
        return nothing
    end
end
function ldiv!(solver::MPIStaticCondensationParallel{T}, U::AbstractVector{T}) where T
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
            if isa(schur_complement_solver, MPIStaticCondensationMUMPS)
                ldiv!(schur_complement_solver, U)
            elseif isa(schur_complement_solver, BlockedSchurComplementSolver)
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
            else
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

end
