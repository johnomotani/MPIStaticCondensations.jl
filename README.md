# MPIStaticCondensations

[![Build Status](https://github.com/johnomotani/MPIStaticCondensations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/johnomotani/MPIStaticCondensations.jl/actions/workflows/CI.yml?query=branch%3Amain)

Direct matrix solve, using a recursively applied static condensation approach,
also known as 'nested dissection' [\[George "Nested dissection of a regular
finite element mesh", SIAM journal on numerical analysis 10.2
(1973)\]](https://doi.org/10.1137/0710032).

Assumes a structured finite-element-like grid, motivated by the discretisation
used by https://github.com/KfT/KineticsforTransport.

Hybrid parallelism using shared- and distributed-memory MPI parallelism.

Usage example
-------------

Note total number of MPI ranks must be divisible by `shared_nproc`.

```julia
using MPIStaticCondensations
using LinearAlgebra
using MPI
using Random


# MPI setup.
############

MPI.Init()

shared_nproc = 1 #4

comm = MPI.COMM_WORLD
nproc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
distributed_nproc, rem = divrem(nproc, shared_nproc)
if rem != 0
    error("shared_nproc=$shared_nproc does not divide nproc=$nproc")
end
distributed_rank, shared_rank = divrem(rank, shared_nproc)
shared_comm = MPI.Comm_split(MPI.COMM_WORLD, distributed_rank, shared_rank)
if shared_rank == 0
    distributed_color = 0
else
    distributed_color = nothing
end
distributed_comm = MPI.Comm_split(MPI.COMM_WORLD, distributed_color,
                                  distributed_rank)

local_win_store_float = MPI.Win[]
allocate_shared_float = (dims...; comm=shared_comm)->begin
    this_shared_rank = MPI.Comm_rank(comm)
    if this_shared_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local, comm)
    array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
    push!(local_win_store_float, win)
    if this_shared_rank == 0
        array .= NaN
    end
    MPI.Barrier(comm)
    return array
end

local_win_store_int = MPI.Win[]
allocate_shared_int = (dims...; comm=shared_comm)->begin
    this_shared_rank = MPI.Comm_rank(comm)
    if this_shared_rank == 0
        dims_local = dims
    else
        dims_local = Tuple(0 for _ ∈ dims)
    end
    win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
    array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
    push!(local_win_store_int, win)
    if this_shared_rank == 0
        array .= typemin(Int64)
    end
    MPI.Barrier(comm)
    return array
end


# Create Dimension objects that define the grid.
################################################
x = create_dimension(; name="x", nelement=8, ngrid=5, nrank=distributed_nproc,
                     irank=distributed_rank)
y = create_dimension(; name="y", nelement=4, ngrid=5)
dimensions = [y, x]


# Create solver.
################

mpisc = mpi_static_condensation(dimensions; variable_dimensions=(nothing, [1]),
                                comm, distributed_comm, shared_comm,
                                allocate_shared_float, allocate_shared_int)


# Matrix and RHS.
#################
matrix =
    ((get_shared_sparse_matrix_csc_buffer(dimensions, shared_comm, allocate_shared_float, allocate_shared_int; row_dimensions=nothing, column_dimensions=nothing),
      get_shared_sparse_matrix_csc_buffer(dimensions, shared_comm, allocate_shared_float, allocate_shared_int; row_dimensions=nothing, column_dimensions=[2])),
     (get_shared_sparse_matrix_csc_buffer(dimensions, shared_comm, allocate_shared_float, allocate_shared_int; row_dimensions=[2], column_dimensions=nothing),
      get_shared_sparse_matrix_csc_buffer(dimensions, shared_comm, allocate_shared_float, allocate_shared_int; row_dimensions=[2], column_dimensions=[2])))
for (i,row) ∈ enumerate(matrix), (j,block) ∈ enumerate(row)
    rand!(block.nzval)
    if i == j
        for ind ∈ 1:size(block, 1)
            # Add to diagonal to ensure that matrix is invertible.
            block[ind,ind] += 1.0
        end
    end
end
rhs = allocate_shared_float(x.n_local * y.n_local + x.n_local)
rand!(rhs)

# Ensure overlapping points are consistent.
if shared_rank == 0
    req = [MPI.Request(), MPI.Request()]
    if distributed_rank > 0
        MPI.Irecv!(@view(rhs[1:y.n_local]), distributed_comm, req[1]; tag=1)
        MPI.Irecv!(@view(rhs[x.n_local*y.n_local+1]), distributed_comm, req[1]; tag=2)
    end
    if distributed_rank < distributed_nproc - 1
        MPI.Send!(@view(rhs[(x.n_local)*y.n_local+1:x.n_local*y.n_local]), distributed_comm, req[1]; tag=1)
        MPI.Send!(@view(rhs[end]), distributed_comm, req[1]; tag=2)
    end
    MPI.Waitall(req)
end


# Compute solver and solve rhs.
###############################
lu!(mpisc, matrix)
ldiv!(mpisc, rhs)


# Free shared-memory arrays.
############################
for w ∈ local_win_store_float
    MPI.free(w)
end
for w ∈ local_win_store_int
    MPI.free(w)
end
MPI.Finalize()
```
