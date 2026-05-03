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

abstract type MPIStaticCondensation end

struct MPIStaticCondensationSerialSparse{Tf<:AbstractFloat,Ti<:Integer,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation
    local_block_solver::SparseArrays.UMFPACK.UmfpackLU{Tf,Ti}
    buffer::Vector{Tf}
    timer::Ttimer
    check_lu::Bool
end

struct MPIStaticCondensationSerialDense{Tf<:AbstractFloat,Ti<:Integer,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation
    local_block_solver::LU{Tf,Matrix{Tf},Vector{Ti}}
    timer::Ttimer
    check_lu::Bool
end

struct MPIStaticCondensationParallel{Tsolver<:MPISchurComplement,Tranget,Trangeb,Ttimer<:Union{Nothing,TimerOutput}} <: MPIStaticCondensation
    local_block_solver::Tsolver
    top_vector_indices::Tranget
    bottom_vector_indices::Trangeb
    timer::Ttimer
end

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolver{Tsolver<:MPIStaticCondensation,Ti<:Integer}
    local_block_solver::Tsolver
    local_block_indices::Vector{Ti}
end

struct Dimension{Ti<:Integer}
    n::Ti
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

        nelement_local = nelement ÷ nrank

        # Assume a continuous-Galerkin finite element discretization where adjacent
        # elements share a boundary point. `ngrid` counts the points in a single element,
        # but two of these are shared (except at the ends of the grid).
        n = nelement * (ngrid - 1) + 1
        first_global_ind = irank * nelement_local * (ngrid - 1) + 1
        last_global_ind = (irank + 1) * nelement_local * (ngrid - 1) + 1

        if !has_lower_boundary
            n -= 1
            if irank == 0
                first_global_ind += 1
            end
        end
        if !has_upper_boundary
            n -= 1
            if irank == nrank - 1
                last_global_ind -= 1
            end
        end

        global_inds = collect(first_global_ind:last_global_ind)
        if periodic && irank == nrank - 1
            global_inds[end] = 1
        end

        return new{Ti}(n, nelement, ngrid, nrank, irank, global_inds, periodic,
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

function pick_dimension_to_split(dimensions::Vector{<:Dimension}, n_groups::Integer,
                                 optimise_schur_complement_size::Bool)
    distributed_dims = findall(d -> d.nrank > 1, dimensions)
    if optimise_schur_complement_size
        if !isempty(distributed_dims)
            idim = argmax(d.n for d ∈ dimensions[distributed_dims])
            return distributed_dims[idim]
        else
            return argmax(d.n for d ∈ dimensions)
        end
    else
        if !isempty(distributed_dims)
            # When dimensions are distributed, splits must be on block boundaries, not
            # just on element boundaries.
            distributed_dims_to_divide = findall(d.nrank % n_groups == 0
                                                 for d ∈ dimensions[distributed_dims])
            dims_to_divide = distributed_dims[distributed_dims_to_divide]
            if !isempty(dims_to_divide)
                idim = argmax(d.n for d ∈ dimensions[dims_to_divide])
                return dims_to_divide[idim]
            else
                idim = argmax(d.n for d ∈ dimensions[distributed_dims])
                return distributed_dims[idim]
            end
        else
            dims_to_divide = findall(d.nelement % n_groups == 0 for d ∈ dimensions)
            if !isempty(dims_to_divide)
                idim = argmax(d.n for d ∈ dimensions[dims_to_divide])
                return dims_to_divide[idim]
            else
                return argmax(d.n for d ∈ dimensions)
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

function get_ind_slice(dimensions::Vector{<:Dimension}, dim_to_slice::Integer,
                       slice_inds::OrdinalRange{<:Integer})
    dimensions = copy(dimensions)
    result_ranges = Tuple(i == dim_to_slice ? slice_inds : 1:dimensions[i].n for i ∈ 1:length(dimensions))
    inds = fill(eltype(slice_inds)(-1), prod(length(r) for r ∈ result_ranges))
    for (local_flat_i, i) ∈ enumerate(CartesianIndices(result_ranges))
        inds[local_flat_i] = get_flattened_index(i, dimensions)
    end
    return inds
end

function get_ind_slice(dimensions::Vector{<:Dimension}, dim_to_slice::Integer,
                       slice_inds::Vector{<:Integer})
    # When `slice_inds` is a Vector, not an OrdinalRange, cannot use CartesianIndices on
    # it, so have to do more complicated loops.
    dimensions = copy(dimensions)
    result_ranges_left = Tuple(1:dimensions[i].n for i ∈ 1:dim_to_slice-1)
    result_ranges_right = Tuple(1:dimensions[i].n for i ∈ dim_to_slice+1:length(dimensions))
    inds = fill(eltype(slice_inds)(-1),
                prod(length(r) for r ∈ result_ranges_left; init=1) * length(slice_inds) *
                prod(length(r) for r ∈ result_ranges_right; init=1))
    local_flat_i = 0
    for i_left ∈ CartesianIndices(result_ranges_left), i_slice ∈ slice_inds,
            i_right ∈ CartesianIndices(result_ranges_right)
        local_flat_i += 1
        indices = CartesianIndex(i_left, i_slice, i_right)
        inds[local_flat_i] = get_flattened_index(indices, dimensions)
    end
    return inds
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
                             slice_inds::Union{UnitRange{<:Integer},Vector{<:Integer}})
    dimensions = copy(dimensions)
    dim_sizes = [d.n for d ∈ dimensions]
    result_ranges = Tuple(i == dim_to_slice ? slice_inds : 1:dim_sizes[i] for i ∈ 1:length(dimensions))
    inds = fill(eltype(slice_inds)(-1), prod(length(r) for r ∈ result_ranges))
    for (local_flat_i, i) ∈ enumerate(CartesianIndices(result_ranges))
        inds[local_flat_i] = get_local_flattened_index(i, dim_sizes)
    end
    return inds
end

struct FakeComm
    rank::Int64
    size::Int64
end
MPI.Comm_rank(comm::FakeComm) = comm.rank
MPI.Comm_size(comm::FakeComm) = comm.size
MPI.Comm_split(comm::FakeComm, color, key) = comm

@kwdef struct LevelInfo{Ti,Tcomm<:Union{MPI.Comm,FakeComm},Tdcomm<:Union{MPI.Comm,Nothing,FakeComm}}
    new_dimensions::Vector{Dimension{Ti}}
    top_vector_indices::Vector{Ti}
    local_top_vector_indices::Vector{Ti}
    bottom_vector_indices::Vector{Ti}
    new_comm::Tcomm
    new_distributed_comm::Tdcomm
    new_shared_comm::Tcomm
end

# Use `FakeComm` values for comm/distributed_comm/shared_comm to skip the comm splitting,
# for testing of the index generation.
function split_dimension(dimensions::Vector{<:Dimension}, n_groups::Integer,
                         optimize_schur_complement_size::Bool,
                         comm::Union{MPI.Comm,FakeComm},
                         distributed_comm::Union{MPI.Comm,Nothing,FakeComm},
                         shared_comm::Union{MPI.Comm,FakeComm})
    ind_type = typeof(n_groups)
    new_dimensions = copy(dimensions)
    new_comm = comm
    new_distributed_comm = distributed_comm
    new_shared_comm = shared_comm
    comm_rank = MPI.Comm_rank(new_comm)
    shared_comm_rank = MPI.Comm_rank(new_shared_comm)
    shared_comm_size = MPI.Comm_size(new_shared_comm)
    distributed_comm_rank = comm_rank ÷ shared_comm_size
    bottom_vector_indices = ind_type[]

    if any(collect(d.remove_boundaries for d ∈ new_dimensions))
        for i_dim ∈ 1:length(new_dimensions)
            d = new_dimensions[i_dim]
            if d.remove_boundaries
                if d.has_lower_boundary
                    if d.irank == 0
                        bottom_vector_indices =
                            vcat(bottom_vector_indices,
                                 get_ind_slice(new_dimensions, i_dim, 1:1))
                    end
                    d = Dimension(; nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                                  irank=d.irank, periodic=d.periodic,
                                  has_lower_boundary=false,
                                  has_upper_boundary=d.has_upper_boundary,
                                  remove_boundaries=false)
                    new_dimensions[i_dim] = d
                end
                if d.has_upper_boundary
                    if d.irank == d.nrank - 1
                        last_ind = length(d.global_inds)
                        bottom_vector_indices =
                            vcat(bottom_vector_indices,
                                 get_ind_slice(new_dimensions, i_dim, last_ind:last_ind))
                    end
                    d = Dimension(; nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                                  irank=d.irank, periodic=d.periodic,
                                  has_lower_boundary=d.has_lower_boundary,
                                  has_upper_boundary=false, remove_boundaries=false)
                    new_dimensions[i_dim] = d
                end
            end
        end
    end

    slice_i = pick_dimension_to_split(dimensions, n_groups,
                                      optimize_schur_complement_size)
    slice_dim = new_dimensions[slice_i]
    slice_periodic = slice_dim.periodic
    slice_irank = slice_dim.irank
    slice_nrank = slice_dim.nrank
    last_slice_ind = length(slice_dim.global_inds)
    if slice_periodic
        # Once dimension has been sliced at least once, the periodic boundary is removed,
        # so the dimension is effectively no longer periodic, and also does not include
        # lower and upper boundaries.
        slice_dim = Dimension(; nelement=slice_dim.nelement, ngrid=slice_dim.ngrid,
                              nrank=slice_dim.nrank, irank=slice_irank,
                              periodic=false, has_lower_boundary=false,
                              has_upper_boundary=false, remove_boundaries=false)
    end
    if slice_dim.nrank > 1
        # When dimension is distributed, split on block boundaries.
        blocks_per_group = (slice_dim.nrank + n_groups - 1) ÷ n_groups
        group_rank = distributed_comm_rank ÷ blocks_per_group
        new_comm = MPI.Comm_split(new_comm, group_rank, 0)
        if shared_comm_rank == 0
            new_distributed_comm = MPI.Comm_split(new_distributed_comm, group_rank, 0)
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
        if (slice_irank ∈ block_boundaries) || (slice_periodic && slice_irank == 0)
            # Lower boundary on this block is a split.
            bottom_vector_indices =
                vcat(bottom_vector_indices,
                     get_ind_slice(new_dimensions, slice_i, 1:1))
            first_top_vector_slice_ind = 2
            slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                                  nrank=this_group_nrank, irank=this_group_irank,
                                  periodic=slice_dim.periodic, has_lower_boundary=false,
                                  has_upper_boundary=slice_dim.has_upper_boundary,
                                  remove_boundaries=false)
        else
            first_top_vector_slice_ind = 1
        end
        if (slice_irank + 1 ∈ block_boundaries) || (slice_periodic && slice_irank == slice_nrank - 1)
            # Upper boundary on this block is a split.
            bottom_vector_indices =
                vcat(bottom_vector_indices,
                     get_ind_slice(new_dimensions, slice_i,
                                   last_slice_ind:last_slice_ind))
            last_top_vector_slice_ind = last_slice_ind - 1
            slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                                  nrank=this_group_nrank, irank=this_group_irank,
                                  periodic=slice_dim.periodic,
                                  has_lower_boundary=slice_dim.has_lower_boundary,
                                  has_upper_boundary=false, remove_boundaries=false)
        else
            last_top_vector_slice_ind = last_slice_ind
        end
        top_vector_indices =
            get_ind_slice(new_dimensions, slice_i,
                          first_top_vector_slice_ind:last_top_vector_slice_ind)
        local_top_vector_indices =
            get_local_ind_slice(new_dimensions, slice_i,
                                first_top_vector_slice_ind:last_top_vector_slice_ind)
        slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                              nrank=this_group_nrank, irank=this_group_irank,
                              periodic=slice_dim.periodic,
                              has_lower_boundary=slice_dim.has_lower_boundary,
                              has_upper_boundary=slice_dim.has_upper_boundary,
                              remove_boundaries=false)
    else
        ngrid = slice_dim.ngrid
        elements_per_group = (slice_dim.nelement + n_groups - 1) ÷ n_groups
        procs_per_group = (shared_comm_size + n_groups - 1) ÷ n_groups
        group_rank = shared_comm_rank ÷ procs_per_group
        if group_rank == n_groups - 1
            this_group_nelement = slice_dim.nelement - group_rank * elements_per_group
        else
            this_group_nelement = elements_per_group
        end
        slice_step = elements_per_group * (ngrid - 1)
        if slice_periodic
            slice_points = 1:slice_step:last_slice_ind
        else
            slice_points = slice_step:slice_step:slice_step*(n_groups-1)
            if slice_dim.has_lower_boundary
                slice_points = slice_points .+ 1
            end
        end
        bottom_vector_indices =
            vcat(bottom_vector_indices,
                 get_ind_slice(new_dimensions, slice_i, slice_points))
        if slice_periodic
            first_local_top_vector_slice_ind = slice_points[group_rank+1] + 1
            has_lower_boundary = false
            last_local_top_vector_slice_ind = slice_points[min(group_rank+2,end)] - 1
            has_upper_boundary = false
        else
            if group_rank == 0
                first_local_top_vector_slice_ind = 1
                has_lower_boundary = slice_dim.has_lower_boundary
            else
                first_local_top_vector_slice_ind = slice_points[group_rank] + 1
                has_lower_boundary = false
            end
            if group_rank == n_groups - 1
                last_local_top_vector_slice_ind = last_slice_ind
                has_upper_boundary = slice_dim.has_upper_boundary
            else
                last_local_top_vector_slice_ind = slice_points[group_rank+1] - 1
                has_upper_boundary = false
            end
        end
        local_top_vector_indices =
            get_ind_slice(new_dimensions, slice_i,
                          first_local_top_vector_slice_ind:last_local_top_vector_slice_ind)
        all_top_vector_slice_inds = [i for i ∈ 1:last_slice_ind if i ∉ slice_points]
        top_vector_indices = get_ind_slice(new_dimensions, slice_i,
                                           all_top_vector_slice_inds)
        slice_dim = Dimension(; nelement=this_group_nelement, ngrid=slice_dim.ngrid,
                              nrank=slice_dim.nrank, irank=slice_irank, periodic=false,
                              has_lower_boundary=has_lower_boundary,
                              has_upper_boundary=has_upper_boundary,
                              remove_boundaries=false)
    end
    new_dimensions[slice_i] = slice_dim

    return LevelInfo(; new_dimensions, top_vector_indices, local_top_vector_indices,
                     bottom_vector_indices, new_comm, new_distributed_comm,
                     new_shared_comm)
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
                                 use_sparse::Bool=true,
                                 optimise_schur_complement_size::Bool=true,
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
    shared_comm_size_factors = factor(Vector, shared_comm_size_factors)

    n_levels = length(n_blocks_factors) + length(shared_comm_size_factors) + 1

    if n_levels == 1
        lowest_level_n = prod(d.n for d ∈ dimensions)
    else
        first_level_points = ind_type[]
    end

    # Create lowest level solver
    identity = Matrix{data_type}(undef, lowest_level_n, lowest_level_n)
    identity .= I
    if use_sparse
        lowest_level_solver =
            MPIStaticCondensationSerialSparse(lu(sparse(identity); check=check_lu),
                                              Vector{data_type}(undef, lowest_level_n),
                                              timer, check_lu)
    else
        lowest_level_solver =
            MPIStaticCondensationSerialDense(lu(identity; check=check_lu), timer,
                                             check_lu)
    end

    this_level_solver = lowest_level_solver
    for level ∈ 2:n_levels
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
        this_level_solver =
            MPIStaticCondensationParallel(this_level_sc, level_top_vector_entries,
                                          level_bottom_vector_entries, timer)
    end

    return this_level_solver
end

function lu!(block_diagonal_solver::BlockDiagonalSolver, A::AbstractMatrix)
    solver = block_diagonal_solver.local_block_solver
    inds = block_diagonal_solver.local_block_indices
    lu!(solver, @view(A[inds,inds]))
    return nothing
end

function ldiv!(x::AbstractVector, block_diagonal_solver::BlockDiagonalSolver, u::AbstractVector)
    solver = block_diagonal_solver.local_block_solver
    inds = block_diagonal_solver.local_block_indices
    @views ldiv!(x[inds], solver, u[inds])
    return nothing
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolver, u::AbstractVector)
    solver = block_diagonal_solver.local_block_solver
    inds = block_diagonal_solver.local_block_indices
    @views ldiv!(solver, u[inds])
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialSparse, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        # For simplicity assume non-zero pattern might change, so pass reuse_symbolic=false.
        lu!(solver.local_block_solver, sparse(A); reuse_symbolic=false, check=solver.check_lu)
    end
    return nothing
end

function ldiv!(X::AbstractVector, solver::MPIStaticCondensationSerialSparse, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        ldiv!(X, solver.local_block_solver, U)
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationSerialSparse, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        buffer = solver.buffer
        buffer .= U
        ldiv!(U, solver.local_block_solver, buffer)
    end
    return nothing
end

function lu!(solver::MPIStaticCondensationSerialDense, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        # Re-use the arrays to avoid allocating.
        mat_storage = solver.local_block_solver.factors
        ipiv = solver.local_block_solver.ipiv
        check = solver.check
        mat_storage .= A
        LAPACK.getrf!(mat_storage, ipiv; check=check)
    end
    return nothing
end

function ldiv!(solver::MPIStaticCondensationSerialDense, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        ldiv!(solver.local_block_solver, U)
    end
    return nothing
end
function ldiv!(X::AbstractVector, solver::MPIStaticCondensationSerialDense, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        ldiv!(X, solver.local_block_solver, U)
    end
    return nothing
end

function lu!(solver::MPIStaticCondensationParallel, A::AbstractMatrix)
    @sc_timeit solver.timer "Static condensation lu! $(size(A))" begin
        update_schur_complement(solver.local_block_solver, a, b, c, d)
    end
    return nothing
end

function ldiv!(X::AbstractVector, solver::MPIStaticCondensationParallel, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        top_vector_indices = solver.top_vector_indices
        bottom_vector_indices = solver.bottom_vector_indices
        x = @view X[top_vector_indices]
        u = @view U[top_vector_indices]
        y = @view X[bottom_vector_indices]
        v = @view U[bottom_vector_indices]
        ldiv!(x, y, solver.local_block_solver, u, v)
    end
    return nothing
end
function ldiv!(solver::MPIStaticCondensationParallel, U::AbstractVector)
    @sc_timeit solver.timer "Static condensation ldiv! $(size(U))" begin
        # MPISchurComplement allows the RHS and solution vectors to be the same array.
        top_vector_indices = solver.top_vector_indices
        bottom_vector_indices = solver.bottom_vector_indices
        u = @view U[top_vector_indices]
        v = @view U[bottom_vector_indices]
        ldiv!(u, v, solver, u, v)
    end
    return nothing
end

end
