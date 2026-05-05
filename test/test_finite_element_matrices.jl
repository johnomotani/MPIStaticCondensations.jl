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

function imin(dim)
    elements_per_block = dim.nelement ÷ dim.nrank
    irank = dim.irank
    ngrid_minus_one = dim.ngrid - 1
    return irank * ngrid_minus_one + 1
end

function imax(dim)
    elements_per_block = dim.nelement ÷ dim.nrank
    irank = dim.irank
    ngrid_minus_one = dim.ngrid - 1
    return (irank + 1) * ngrid_minus_one + 1
end

function is_global_index_in_block(inds, dimensions, global_cartinds)
    # Cannot use dimensions[i].global_inds because periodicity has not been taken into
    # account when this function is called.
    return all(imin(dimensions[i]) ≤ inds[i] ≤ imax(dimensions[i]) for i ∈ 1:length(dimensions))
end

function global_to_local(inds, dimensions)
    function global_to_local_1d(i, dim)
        # Cannot use dim.global_inds because periodicity has not been taken into account
        # when this function is called.
        i1 = imin(dim)
        i2 = imax(dim)
        if !(i1 ≤ i ≤ i2)
            error("i=$i not found in dimension's global indices $i1:$i2.")
        end
        return iloc
    end
    i = 0
    for d ∈ length(dimensions):-1:1
        iglob = inds[d]
        dim = dimensions[d]
        i = i * dim.n_local + global_to_local_1d(iglob, dim) - 1
    end
    i += 1
    return i
end

function get_sparse_indices_for_local_block(global_i, global_j, dimensions, irank_list)
    local_dimensions = [
        create_dimension(; nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank, irank=irank,
                         periodic=d.periodic, remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(dimensions, irank_list)
    ]
    global_cartinds = CartesianIndices(Tuple(d.n for d ∈ dimensions))
    local_sparse_inds = Int64[]
    local_i = Int64[]
    local_j = Int64[]
    for (sparse_i, (i, j)) ∈ enumerate(zip(global_i, global_j))
        i_inds = global_cartinds[i]
        j_inds = global_cartinds[j]
        if (is_global_index_in_block(i_inds, dimensions, global_cartinds)
                && is_global_index_in_block(j_inds, dimensions, global_cartinds))
            push!(local_sparse_inds, sparse_i)
            push!(local_i, global_to_local(i_inds, dimensions))
            push!(local_j, global_to_local(j_inds, dimensions))
        end
    end
    return local_sparse_inds, local_i, local_j
end

function get_sparse_indices_for_all_local_blocks(global_i, global_j, dimensions,
                                                 local_block_irank_lists)
    local_block_sparse_indices = Vector{Int64}[]
    local_i_list = Vector{Int64}[]
    local_j_list = Vector{Int64}[]
    for irl ∈ local_block_irank_lists
        local_sparse_inds, local_i, local_j =
            get_sparse_indices_for_local_block(global_i, global_j, dimensions, irl)
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

function apply_periodicity_to_indices!(global_inds, dimensions)
    if !any(d.periodic for d ∈ dimensions)
        # Nothing to do.
        return nothing
    end

    global_cartinds = CartesianIndices(Tuple(d.n for d ∈ dimensions))
    for (sparse_i, flat_i) ∈ enumerate(global_inds)
        inds = global_cartinds[flat_i]
        new_flat_i = 0
        for (i, d) ∈ zip(reverse(Tuple(inds)), reverse(dimensions))
            if d.periodic && i == d.n
                i = 1
            end
            new_flat_i = new_flat_i * d.n + i - 1
        end
        new_flat_i += 1
        global_inds[sparse_i] = new_flat_i
    end

    return nothing
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
        data, global_i, global_j = construct_sparse_finite_element_matrix(dimensions, rng)

        local_block_irank_lists = [get_irank_list(irank, dimensions)
                                   for irank ∈ 1:distributed_comm_size-1]
        local_block_sparse_indices, local_i_list, local_j_list =
            get_sparse_indices_for_all_local_blocks(global_i, global_j, dimensions,
                                                    local_block_irank_lists)

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

        apply_periodicity_to_indices!(global_i, dimensions)
        apply_periodicity_to_indices!(global_j, dimensions)

        # Assemble global matrix
        n = prod(d.periodic ? d.n - 1 : d.n for d ∈ dimensions)
        global_matrix = zeros(n, n)
        # Cannot broadcast data into global_matrix using global_i and global_j because
        # once periodicity is accounted for there will be duplicate indices, whose
        # corresponding entries must be summed.
        for (entry, i, j) ∈ zip(data, global_i, global_j)
            global_matrix[i,j] += data
        end
    elseif shared_comm_rank == 0
        MPI.Recv!(local_matrix, distributed_comm; source=0)
    end

    return global_matrix, local_matrix
end

function test_finite_element_matrices()
end
