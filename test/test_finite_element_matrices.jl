using MPIStaticCondensations
using MPIStaticCondensations: Dimension
using MPI
using Primes
using Test

include("utils.jl")

function get_flattened_index(n_list, ngrid_list, ielement, igrid)
    combined_inds = Tuple((iel - 1) * (ng - 1) + igr for (ng, iel, igr) ∈ zip(ngrid_list, Tuple(ielement), Tuple(igrid)))
    i_flat = 0
    for (i, n) ∈ zip(reverse(combined_inds), reverse(n_list))
        i_flat = n * i_flat + i - 1
    end
    return i_flat + 1
end

function construct_sparse_finite_element_matrix(dimensions::Vector{<:Dimension}, rng,
                                                sparse_stencils::Bool)

    data = Float64[]
    global_inds = Tuple{Int64,Int64}[]
    element_indices = CartesianIndices(Tuple(d.ngrid for d ∈ dimensions))
    n_tuple = Tuple(d.n for d ∈ dimensions)
    ngrid_tuple = Tuple(d.ngrid for d ∈ dimensions)
    counter = 0
    if sparse_stencils
        nd = length(dimensions)
        for ielement ∈ CartesianIndices(Tuple(d.nelement for d ∈ dimensions))
            istart = counter
            for igrid ∈ element_indices, d ∈ nd, this_jgrid ∈ 1:ngrid_tuple[d]
                jgrid = CartesianIndex(Tuple(this_d == d ? this_jgrid : igrid[this_d] for this_d ∈ 1:nd))
                global_i = get_flattened_index(n_list, nelement_list, ngrid_list, ielement, igrid)
                global_j = get_flattened_index(n_list, nelement_list, ngrid_list, ielement, jgrid)
                i = (global_i, global_j)
                if i ∉ global_inds
                    # Search global_inds to avoid appending repeats.
                    push!(global_inds, i)
                    if igrid == jgrid
                        # Add 1 to diagonal to ensure matrix is invertible.
                        push!(data, 1.0 + rand(rng))
                        counter += 1
                    else
                        push!(data, rand(rng))
                        counter += 1
                    end
                end
            end
            iend = counter
        end
    else
        for ielement ∈ CartesianIndices(Tuple(d.nelement for d ∈ dimensions))
            for igrid ∈ element_indices, jgrid ∈ element_indices
                global_i = get_flattened_index(n_list, nelement_list, ngrid_list, ielement, igrid)
                global_j = get_flattened_index(n_list, nelement_list, ngrid_list, ielement, jgrid)
                i = (global_i, global_j)
                if i ∉ global_inds
                    # Search global_inds to avoid appending repeats.
                    push!(global_inds, i)
                    if igrid == jgrid
                        # Add 1 to diagonal to ensure matrix is invertible.
                        push!(data, 1.0 + rand(rng))
                    else
                        push!(data, rand(rng))
                    end
                end
            end
        end
    end

    global_i = [i[1] for i ∈ global_inds]
    global_j = [i[2] for i ∈ global_inds]

    return data, global_i, global_j
end

function get_sparse_indices_for_local_block(dimensions, irank_list)
    local_dimensions = [
        create_dimension(; nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank, irank=irank,
                         periodic=d.periodic, remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(dimensions, irank_list)
    ]
    # Probably need to take in global index lists and loop through them.
    # Need function that tests whether a certain (i_global,j_global) exists on a certain
    # block, and use that to make a list of the indices that belong to each block.
end

function get_sparse_indices_for_all_local_blocks(dimensions, irank_list)
    local_block_sparse_indices = Vector{Int64}[]
    local_i_list = Vector{Int64}[]
    local_j_list = Vector{Int64}[]
    for irl ∈ local_block_irank_lists
        local_sparse_inds, local_i, local_j =
            get_sparse_indices_for_local_block(dimensions, irl)
        push!(local_block_sparse_indices, local_sparse_inds)
        push!(local_i_list, local_i)
        push!(local_j_list, local_j)
    end
    return local_block_sparse_indices, local_i_list, local_j_list
end

function get_irank_list(irank, dimensions)
    irank_list = Int64[]
    for d ∈ dimensions
        irank, this_irank = divrem(irank, d.nrank)
        push!(irank_list, this_irank)
    end
    return irank_list
end

function assemble_and_scatter_global_matrix(dimensions::Vector{<:Dimension},
                                            comm::MPI.Comm,
                                            distributed_comm::Union{MPI.Comm,Nothing},
                                            shared_comm::MPI.Comm, n_shared::Int64,
                                            allocate_shared_float, rng)
    rank = MPI.Comm_rank(comm)
    distributed_comm_size = MPI.Comm_size(distributed_comm)
    shared_comm_rank = MPI.Comm_rank(shared_comm)

    local_n = prod(d.n_local for d ∈ dimensions)
    local_matrix = allocate_shared_float(local_n, local_n)
    global_matrix = nothing
    if rank == 0
        n = prod(d.n for d ∈ dimensions)
        global_matrix = zeros(n, n)

        data, global_i, global_j = construct_sparse_finite_element_matrix(dimensions, rng)

        local_block_irank_lists = [get_irank_list(irank, dimensions)
                                   for irank ∈ 1:distributed_comm_size-1]
        local_block_sparse_indices, local_i_list, local_j_list =
            get_sparse_indices_for_all_local_blocks(dimensions, irl)

        # Count overlaps so that the corresponding points can be decreased so that when
        # overlaps are added together from all overlapping blocks, they give the original
        # value.
        overlap_count = zeros(Int64, length(global_i))
        for sparse_inds ∈ local_block_sparse_indices
            @views overlap_count[sparse_inds] .+= 1
        end
        data_to_distribute = copy(data)
        data_to_distribute ./= overlap_count

        for irank ∈ 1:distributed_comm_size-1
            local_sparse_inds = local_block_sparse_indices[irank+1]
            local_i = local_i_list[irank+1]
            local_j = local_j_list[irank+1]

            local_matrix .= 0
            local_matrix[local_i,local_j] .= @view data_to_distribute[local_sparse_inds]
            MPI.Send(local_matrix, distributed_comm; dest=irank)
        end

        local_sparse_inds = local_block_sparse_indices[1]
        local_i = local_i_list[1]
        local_j = local_j_list[1]
        local_matrix .= 0
        local_matrix[local_i,local_j] .= @view data_to_distribute[local_sparse_inds]
    elseif shared_comm_rank == 0
        MPI.Recv!(local_matrix, distributed_comm; source=0)
    end

    return global_matrix, local_matrix
end

function test_finite_element_matrices()
end
