using BlockArrays
using MPIStaticCondensations
using MPIStaticCondensations: Dimension
using MPI
using SparseArrays
using SparseArrays: FixedSparseCSC

function get_flattened_index(n_tuple, ngrid_tuple, ielement, igrid)
    combined_inds = [(iel - 1) * (ng - 1) + igr for (ng, iel, igr) ∈ zip(ngrid_tuple, ielement, igrid)]
    i_flat = 0
    for (i, n) ∈ zip(reverse(combined_inds), reverse(n_tuple))
        i_flat = n * i_flat + i - 1
    end
    return i_flat + 1
end
function get_flattened_index(n_tuple, ngrid_tuple, ielement, igrid::CartesianIndex)
    return get_flattened_index(n_tuple, ngrid_tuple, ielement, Tuple(igrid))
end

function construct_sparse_finite_element_matrix(common_dimensions::Tuple, rng,
                                                sparse_stencils::Bool,
                                                handle_periodicity::Bool=true;
                                                extra_row_dimensions=[],
                                                extra_column_dimensions=[],
                                                stencil="element")

    if stencil == "empty"
        data = Float64[]
        global_i = Int64[]
        global_j = Int64[]
        return data, global_i, global_j
    end

    nd = length(common_dimensions)
    data = Float64[]
    global_inds = Tuple{Int64,Int64}[]
    n_tuple = map(d->d.n, common_dimensions)
    ngrid_tuple = map(d->d.ngrid, common_dimensions)
    nelement_tuple = map(d->d.nelement, common_dimensions)
    element_indices = CartesianIndices(ngrid_tuple)
    extra_m = prod(d.n for d ∈ extra_row_dimensions; init=1)
    extra_n = prod(d.n for d ∈ extra_column_dimensions; init=1)

    last_dim_dense_boundaries = common_dimensions[end].dense_boundaries
    dense_boundary_nelement_tuple_first = ntuple((d) -> d == nd ? (1:1) : nelement_tuple[d], nd)
    dense_boundary_ngrid_tuple_first = ntuple((d) -> d == nd ? (1:1) : ngrid_tuple[d], nd)
    dense_boundary_nelement_tuple_last = ntuple((d) -> d == nd ? (common_dimensions[end].nelement:common_dimensions[end].nelement) : nelement_tuple[d], nd)
    dense_boundary_ngrid_tuple_last = ntuple((d) -> d == nd ? (common_dimensions[end].ngrid:common_dimensions[end].ngrid) : ngrid_tuple[d], nd)

    counter = 0
    function add_point!(ielement, igrid, jelement, jgrid)
        common_global_i = get_flattened_index(n_tuple, ngrid_tuple, Tuple(ielement), Tuple(igrid))
        common_global_j = get_flattened_index(n_tuple, ngrid_tuple, Tuple(jelement), jgrid)
        for extra_j ∈ 1:extra_n, extra_i ∈ 1:extra_m
            global_i = (common_global_i - 1) * extra_m + extra_i
            global_j = (common_global_j - 1) * extra_n + extra_j
            i = (global_i, global_j)
            push!(global_inds, i)
            if igrid == Tuple(jgrid)
                # Add 1 to diagonal to ensure matrix is invertible.
                push!(data, 1.0 + rand(rng))
                counter += 1
            else
                push!(data, rand(rng))
                counter += 1
            end
        end
        return nothing
    end
    if stencil == "point"
        for ielement ∈ CartesianIndices(nelement_tuple)
            for igrid ∈ element_indices
                if last_dim_dense_boundaries && ielement[nd] == 1 && igrid[nd] == 1
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_first), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_first)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                    skip_last_dim_first = true
                else
                    skip_last_dim_first = false
                end
                if last_dim_dense_boundaries && ielement[nd] == nelement_tuple[nd] && igrid[nd] == ngrid_tuple[nd]
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    skip_last_dim_last = true
                else
                    skip_last_dim_last = false
                end
                add_point!(ielement, igrid, ielement, igrid)
                if skip_last_dim_last
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_last), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_last)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                end
            end
        end
    elseif stencil != "element"
        error("Unrecognised stencil=\"$stencil\".")
    elseif sparse_stencils
        for ielement ∈ CartesianIndices(nelement_tuple)
            for igrid ∈ element_indices
                if last_dim_dense_boundaries && ielement[nd] == 1 && igrid[nd] == 1
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_first), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_first)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                    skip_last_dim_first = true
                else
                    skip_last_dim_first = false
                end
                if last_dim_dense_boundaries && ielement[nd] == nelement_tuple[nd] && igrid[nd] == ngrid_tuple[nd]
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    skip_last_dim_last = true
                else
                    skip_last_dim_last = false
                end
                for d ∈ 1:nd, this_jgrid ∈ 1:ngrid_tuple[d]
                    if d > 1 && this_jgrid == igrid[d]
                        # This repeats the diagonal entry that was already included.
                        continue
                    end
                    if skip_last_dim_first && d == nd && this_jgrid == 1
                        # Already included these points in the 'dense boundaries' branch.
                        continue
                    end
                    if skip_last_dim_last && d == nd && this_jgrid == ngrid_tuple[end]
                        # Will include these points in the following 'dense boundaries' branch.
                        continue
                    end
                    jgrid = [this_d == d ? this_jgrid : igrid[this_d] for this_d ∈ 1:nd]
                    if (any(ig == 1 && ie > 1 for (ig, ie) ∈ zip(Tuple(igrid), Tuple(ielement)))
                            && any(jg == 1 && je > 1 for (jg, je) ∈ zip(jgrid, Tuple(ielement))))
                        # Avoid repeated global index pairs.
                        continue
                    end
                    add_point!(ielement, igrid, ielement, jgrid)
                end
                if skip_last_dim_last
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_last), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_last)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                end
            end
        end
    else
        for ielement ∈ CartesianIndices(nelement_tuple)
            for igrid ∈ element_indices
                if last_dim_dense_boundaries && ielement[nd] == 1 && igrid[nd] == 1
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_first), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_first)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                    skip_last_dim_first = true
                else
                    skip_last_dim_first = false
                end
                if last_dim_dense_boundaries && ielement[nd] == nelement_tuple[nd] && igrid[nd] == ngrid_tuple[nd]
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    skip_last_dim_last = true
                else
                    skip_last_dim_last = false
                end
                for jgrid ∈ element_indices
                    if (any(ig == 1 && ie > 1 for (ig, ie) ∈ zip(Tuple(igrid), Tuple(ielement)))
                            && any(jg == 1 && je > 1 for (jg, je) ∈ zip(Tuple(jgrid), Tuple(ielement))))
                        # Avoid repeated global index pairs.
                        continue
                    end
                    add_point!(ielement, igrid, ielement, jgrid)
                end
                if skip_last_dim_last
                    # Fill all the boundary entries that are allowed to be non-zero when
                    # the last dimension has dense_boundaries=true.
                    for jelement ∈ CartesianIndices(dense_boundary_nelement_tuple_last), jgrid ∈ CartesianIndices(dense_boundary_ngrid_tuple_last)
                        add_point!(ielement, igrid, jelement, Tuple(jgrid))
                    end
                end
            end
        end
    end

    global_i = [i[1] for i ∈ global_inds]
    global_j = [i[2] for i ∈ global_inds]

    if handle_periodicity
        apply_periodicity_to_indices!(global_i, vcat(common_dimensions, extra_row_dimensions))
        apply_periodicity_to_indices!(global_j, vcat(common_dimensions, extra_column_dimensions))
    end

    return data, global_i, global_j
end

function imin(dim)
    elements_per_block = dim.nelement ÷ dim.nrank
    irank = dim.irank
    ngrid_minus_one = dim.ngrid - 1
    return irank * elements_per_block * ngrid_minus_one + 1
end

function imax(dim)
    elements_per_block = dim.nelement ÷ dim.nrank
    irank = dim.irank
    ngrid_minus_one = dim.ngrid - 1
    return (irank + 1) * elements_per_block * ngrid_minus_one + 1
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
        return i - i1 + 1
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

function get_sparse_indices_for_local_block(global_i, global_j, dimensions, irank_list;
                                            extra_row_dimensions=[],
                                            extra_column_dimensions=[])
    local_dimensions = [
        create_dimension(; name=d.name, nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                         irank=irank, periodic=d.periodic,
                         remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(dimensions, irank_list[1:length(dimensions)])
    ]
    local_extra_row_dimensions = [
        create_dimension(; name=d.name, nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                         irank=irank, periodic=d.periodic,
                         remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(extra_row_dimensions, irank_list[length(dimensions)+1:end])
    ]
    local_extra_column_dimensions = [
        create_dimension(; name=d.name, nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                         irank=irank, periodic=d.periodic,
                         remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(extra_column_dimensions, irank_list[length(dimensions)+1:end])
    ]
    local_row_dimensions = vcat(local_dimensions, local_extra_row_dimensions)
    local_column_dimensions = vcat(local_dimensions, local_extra_column_dimensions)
    row_n_tuple = Tuple(d.n for d ∈ local_row_dimensions)
    column_n_tuple = Tuple(d.n for d ∈ local_column_dimensions)
    return get_sparse_indices_for_local_block(global_i, global_j, local_row_dimensions,
                                              local_column_dimensions, irank_list,
                                              row_n_tuple, column_n_tuple;
                                              local_extra_row_dimensions,
                                              local_extra_column_dimensions)
end
function get_sparse_indices_for_local_block(global_i, global_j, local_row_dimensions,
                                            local_column_dimensions, irank_list,
                                            row_n_tuple::Tuple, column_n_tuple::Tuple;
                                            local_extra_row_dimensions=[],
                                            local_extra_column_dimensions=[])
    row_global_cartinds = CartesianIndices(row_n_tuple)
    column_global_cartinds = CartesianIndices(column_n_tuple)
    local_sparse_inds = Int64[]
    local_i = Int64[]
    local_j = Int64[]
    for (sparse_i, (i, j)) ∈ enumerate(zip(global_i, global_j))
        i_inds = row_global_cartinds[i]
        j_inds = column_global_cartinds[j]
        if (is_global_index_in_block(i_inds, local_row_dimensions, row_global_cartinds)
                && is_global_index_in_block(j_inds, local_column_dimensions, column_global_cartinds))
            push!(local_sparse_inds, sparse_i)
            push!(local_i, global_to_local(i_inds, local_row_dimensions))
            push!(local_j, global_to_local(j_inds, local_column_dimensions))
        end
    end
    return local_sparse_inds, local_i, local_j
end

function get_sparse_indices_for_all_local_blocks(global_i, global_j, dimensions,
                                                 local_block_irank_lists;
                                                 extra_row_dimensions=[],
                                                 extra_column_dimensions=[])
    local_block_sparse_indices = Vector{Int64}[]
    local_i_list = Vector{Int64}[]
    local_j_list = Vector{Int64}[]
    for irl ∈ local_block_irank_lists
        local_sparse_inds, local_i, local_j =
            get_sparse_indices_for_local_block(global_i, global_j, dimensions, irl;
                                               extra_row_dimensions,
                                               extra_column_dimensions)
        push!(local_block_sparse_indices, local_sparse_inds)
        push!(local_i_list, local_i)
        push!(local_j_list, local_j)
    end
    return local_block_sparse_indices, local_i_list, local_j_list
end

function get_rhs_indices_for_local_block(dimensions, irank_list::AbstractVector)
    local_dimensions = [
        create_dimension(; name=d.name, nelement=d.nelement, ngrid=d.ngrid, nrank=d.nrank,
                         irank=irank, periodic=d.periodic,
                         remove_boundaries=d.remove_boundaries)
        for (d, irank) ∈ zip(dimensions, irank_list)
    ]
    function get_dim_range(dim)
        irank = dim.irank
        ngrid_minus_one = dim.ngrid - 1
        nelement_local = dim.nelement ÷ dim.nrank
        return irank*nelement_local*ngrid_minus_one+1:(irank+1)*nelement_local*ngrid_minus_one+1
    end
    dim_ranges = Tuple(get_dim_range(d) for d in local_dimensions)
    return get_rhs_indices_for_local_block(dimensions, dim_ranges)
end
function get_rhs_indices_for_local_block(dimensions, dim_ranges::Tuple)
    local_inds = zeros(Int64, prod(length(r) for r ∈ dim_ranges))
    for (local_i, inds) ∈ enumerate(CartesianIndices(dim_ranges))
        flat_i = 0
        for (i, d) ∈ zip(reverse(Tuple(inds)), reverse(dimensions))
            flat_i = flat_i * d.n + i - 1
        end
        flat_i += 1
        local_inds[local_i] = flat_i
    end
    return local_inds
end

function get_rhs_indices_for_all_local_blocks(dimensions, local_block_irank_lists)
    local_block_indices_list = Vector{Int64}[]
    for irl ∈ local_block_irank_lists
        push!(local_block_indices_list, get_rhs_indices_for_local_block(dimensions, irl))
    end
    return local_block_indices_list
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
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return apply_periodicity_to_indices!(global_inds, dimensions, n_tuple)
end
function apply_periodicity_to_indices!(global_inds, dimensions, n_tuple::Tuple)
    global_cartinds = CartesianIndices(n_tuple)
    for (sparse_i, flat_i) ∈ enumerate(global_inds)
        inds = global_cartinds[flat_i]
        new_flat_i = 0
        for (i, d) ∈ zip(reverse(Tuple(inds)), reverse(dimensions))
            if d.periodic && i == d.n
                i = 1
            end
            n = d.periodic ? d.n - 1 : d.n
            new_flat_i = new_flat_i * n + i - 1
        end
        new_flat_i += 1
        global_inds[sparse_i] = new_flat_i
    end

    return nothing
end

function assemble_and_scatter_global_matrix(dimensions::Vector{<:Dimension},
                                            comm::MPI.Comm,
                                            distributed_comm::Union{MPI.Comm,Nothing},
                                            shared_comm::MPI.Comm, allocate_shared_float,
                                            allocate_shared_int, rng,
                                            sparse_stencils::Bool; return_separate=false,
                                            row_dimensions=nothing,
                                            column_dimensions=nothing, stencil="element")
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    distributed_comm_size = comm_size ÷ shared_comm_size
    distributed_comm_rank = rank ÷ shared_comm_size
    shared_comm_rank = MPI.Comm_rank(shared_comm)

    if row_dimensions === nothing
        row_dimensions = dimensions
    end
    if column_dimensions === nothing
        column_dimensions = dimensions
    end
    common_dimension_inds = [r.name == c.name
                             for (r, c) ∈ zip(row_dimensions, column_dimensions)]
    n_common_dimensions = 0
    for (r, c) ∈ zip(reverse(row_dimensions), reverse(column_dimensions))
        n_common_dimensions += 1
        if r.name != c.name
            break
        end
    end
    common_dimensions = row_dimensions[end-n_common_dimensions+1:end]
    extra_row_dimensions = row_dimensions[1:end-n_common_dimensions]
    extra_column_dimensions = column_dimensions[1:end-n_common_dimensions]

    local_m = prod(d.n_local for d ∈ row_dimensions)
    local_n = prod(d.n_local for d ∈ column_dimensions)
    if return_separate
        local_data = nothing
        global_data = nothing
        global_i = nothing
        global_j = nothing
        this_block_global_i = nothing
        this_block_global_j = nothing
        local_i = nothing
        local_j = nothing
        n_local = allocate_shared_int(1)
    else
        global_matrix = nothing
    end
    if rank == 0
        if extra_row_dimensions == [] && extra_column_dimensions == []
            all_dimensions = common_dimensions
        elseif extra_row_dimensions == []
            all_dimensions = column_dimensions
        elseif extra_column_dimensions == []
            all_dimensions = row_dimensions
        else
            error("At least one of extra_row_dimensions and extra_column_dimensions "
                  * "should be empty.")
        end
        data, global_i, global_j =
            construct_sparse_finite_element_matrix(Tuple(common_dimensions), rng,
                                                   sparse_stencils, false;
                                                   extra_row_dimensions,
                                                   extra_column_dimensions, stencil)

        local_block_irank_lists = [get_irank_list(irank, all_dimensions)
                                   for irank ∈ 0:distributed_comm_size-1]
        local_block_sparse_indices, local_i_list, local_j_list =
            get_sparse_indices_for_all_local_blocks(global_i, global_j, common_dimensions,
                                                    local_block_irank_lists;
                                                    extra_row_dimensions,
                                                    extra_column_dimensions)

        # Count overlaps so that the corresponding points can be decreased so that when
        # overlaps are added together from all overlapping blocks, they give the original
        # value.
        overlap_count = zeros(Int64, length(global_i))
        for sparse_inds ∈ local_block_sparse_indices
            @views overlap_count[sparse_inds] .+= 1
        end
        data_to_distribute = copy(data)
        data_to_distribute ./= overlap_count

        if return_separate
            for irank ∈ 1:distributed_comm_size-1
                local_sparse_inds = local_block_sparse_indices[irank+1]

                n_local[] = length(local_sparse_inds)

                MPI.Send(n_local, distributed_comm; dest=irank)
                MPI.Send(data_to_distribute[local_sparse_inds], distributed_comm;
                         dest=irank)
                MPI.Send(global_i[local_sparse_inds], distributed_comm; dest=irank)
                MPI.Send(global_j[local_sparse_inds], distributed_comm; dest=irank)
                MPI.Send(local_i_list[irank+1], distributed_comm; dest=irank)
                MPI.Send(local_j_list[irank+1], distributed_comm; dest=irank)
            end
        else
            for irank ∈ 1:distributed_comm_size-1
                local_sparse_inds = local_block_sparse_indices[irank+1]
                local_i = local_i_list[irank+1]
                local_j = local_j_list[irank+1]

                this_local_matrix = sparse(local_i, local_j,
                                           data_to_distribute[local_sparse_inds], local_m,
                                           local_n)
                this_local_matrix_nnz = Ref(nnz(this_local_matrix))
                MPI.Send(this_local_matrix_nnz, distributed_comm; dest=irank)
                MPI.Send(this_local_matrix.colptr, distributed_comm; dest=irank)
                MPI.Send(this_local_matrix.rowval, distributed_comm; dest=irank)
                MPI.Send(this_local_matrix.nzval, distributed_comm; dest=irank)
            end
        end

        apply_periodicity_to_indices!(global_i, row_dimensions)
        apply_periodicity_to_indices!(global_j, column_dimensions)

        local_sparse_inds = local_block_sparse_indices[1]

        if return_separate
            global_data = data
            n_local[] = length(local_sparse_inds)
            MPI.Barrier(shared_comm)
            local_data = allocate_shared_float(n_local[])
            this_block_global_i = allocate_shared_int(n_local[])
            this_block_global_j = allocate_shared_int(n_local[])
            local_i = allocate_shared_int(n_local[])
            local_j = allocate_shared_int(n_local[])
            local_data .= data[local_sparse_inds]
            this_block_global_i .= global_i[local_sparse_inds]
            this_block_global_j .= global_j[local_sparse_inds]
            local_i .= local_i_list[1]
            local_j .= local_j_list[1]
        else
            local_i = local_i_list[1]
            local_j = local_j_list[1]
            this_local_matrix = sparse(local_i, local_j,
                                       data_to_distribute[local_sparse_inds], local_m,
                                       local_n)
            local_matrix_nnz = Ref(nnz(this_local_matrix))
            MPI.Bcast!(local_matrix_nnz, shared_comm; root=0)
            local_matrix_colptr = allocate_shared_int(local_n + 1)
            local_matrix_rowval = allocate_shared_int(local_matrix_nnz[])
            local_matrix_nzval = allocate_shared_float(local_matrix_nnz[])

            local_matrix_colptr .= this_local_matrix.colptr
            local_matrix_rowval .= this_local_matrix.rowval
            local_matrix_nzval .= this_local_matrix.nzval

            MPI.Barrier(shared_comm)

            # Assemble global matrix
            m = prod(d.periodic ? d.n - 1 : d.n for d ∈ row_dimensions)
            n = prod(d.periodic ? d.n - 1 : d.n for d ∈ column_dimensions)
            local_matrix = FixedSparseCSC(local_m, local_n, local_matrix_colptr,
                                          local_matrix_rowval, local_matrix_nzval)
            global_matrix = sparse(global_i, global_j, data, m, n)
        end
    elseif return_separate && distributed_comm_rank == 0
        MPI.Barrier(shared_comm)
        local_data = allocate_shared_float(n_local[])
        this_block_global_i = allocate_shared_int(n_local[])
        this_block_global_j = allocate_shared_int(n_local[])
        local_i = allocate_shared_int(n_local[])
        local_j = allocate_shared_int(n_local[])
    elseif distributed_comm_rank == 0
        local_matrix_nnz = Ref(0)
        MPI.Bcast!(local_matrix_nnz, shared_comm; root=0)
        local_matrix_colptr = allocate_shared_int(local_n + 1)
        local_matrix_rowval = allocate_shared_int(local_matrix_nnz[])
        local_matrix_nzval = allocate_shared_float(local_matrix_nnz[])

        MPI.Barrier(shared_comm)
        local_matrix = FixedSparseCSC(local_m, local_n, local_matrix_colptr,
                                      local_matrix_rowval, local_matrix_nzval)
    else
        if return_separate
            n_local = allocate_shared_int(1)
            if shared_comm_rank == 0
                MPI.Recv!(n_local, distributed_comm; source=0)
            end
            MPI.Barrier(shared_comm)
            local_data = allocate_shared_float(n_local[])
            this_block_global_i = allocate_shared_int(n_local[])
            this_block_global_j = allocate_shared_int(n_local[])
            local_i = allocate_shared_int(n_local[])
            local_j = allocate_shared_int(n_local[])
            if shared_comm_rank == 0
                MPI.Recv!(local_data, distributed_comm; source=0)
                MPI.Recv!(this_block_global_i, distributed_comm; source=0)
                MPI.Recv!(this_block_global_j, distributed_comm; source=0)
                MPI.Recv!(local_i, distributed_comm; source=0)
                MPI.Recv!(local_j, distributed_comm; source=0)
            end
        else
            local_matrix_nnz = Ref(0)
            if shared_comm_rank == 0
                MPI.Recv!(local_matrix_nnz, distributed_comm; source=0)
                MPI.Bcast!(local_matrix_nnz, shared_comm; root=0)
            end
            local_matrix_colptr = allocate_shared_int(local_n + 1)
            local_matrix_rowval = allocate_shared_int(local_matrix_nnz[])
            local_matrix_nzval = allocate_shared_float(local_matrix_nnz[])
            if shared_comm_rank == 0
                MPI.Recv!(local_matrix_colptr, distributed_comm; source=0)
                MPI.Recv!(local_matrix_rowval, distributed_comm; source=0)
                MPI.Recv!(local_matrix_nzval, distributed_comm; source=0)
            end

            MPI.Barrier(shared_comm)
            local_matrix = FixedSparseCSC(local_m, local_n, local_matrix_colptr,
                                          local_matrix_rowval, local_matrix_nzval)
        end
    end

    if return_separate
        return global_data, global_i, global_j, local_data, this_block_global_i,
               this_block_global_j, local_i, local_j
    else
        return global_matrix, local_matrix
    end
end

function assemble_and_scatter_global_multi_variable_matrix(
             dimensions::Vector{<:Dimension}, variable_dimensions, comm::MPI.Comm,
             distributed_comm::Union{MPI.Comm,Nothing}, shared_comm::MPI.Comm,
             allocate_shared_float, allocate_shared_int, rng, sparse_stencils::Bool;
             return_separate=false, stencil_matrix=nothing, combine_blocks=false)

    if combine_blocks && !return_separate
        error("Cannot use combine_blocks=true unless return_separate=true")
    end

    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    distributed_comm_size = comm_size ÷ shared_comm_size
    distributed_comm_rank = rank ÷ shared_comm_size
    shared_comm_rank = MPI.Comm_rank(shared_comm)

    nd = length(dimensions)
    n_variables = length(variable_dimensions)
    if stencil_matrix === nothing
        stencil_matrix = fill("element", n_variables, n_variables)
    end

    variable_dimensions = Tuple(vdims === nothing ? (1:nd) : vdims
                                for vdims ∈ variable_dimensions)

    # Get results in a tuple of block-rows, each of which is a Tuple of matrix block
    # results.
    result_variable_blocks = Tuple(Tuple(assemble_and_scatter_global_matrix(
                                             dimensions, comm, distributed_comm,
                                             shared_comm, allocate_shared_float,
                                             allocate_shared_int, rng, sparse_stencils;
                                             return_separate,
                                             row_dimensions=dimensions[variable_dimensions[ivar]],
                                             column_dimensions=dimensions[variable_dimensions[jvar]],
                                             stencil=stencil_matrix[ivar,jvar])
                                         for jvar ∈ 1:n_variables)
                                   for ivar ∈ 1:n_variables)

    function var_tuple_from_result(i)
        return Tuple(Tuple(result_variable_blocks[ivar][jvar][i]
                           for jvar ∈ 1:n_variables)
                     for ivar ∈ 1:n_variables)
    end

    if return_separate
        if combine_blocks
            Tf = eltype(result_variable_blocks[1][1][4])
            Ti = eltype(result_variable_blocks[1][1][5])
            if rank == 0
                global_data = Tf[]
                global_i = Ti[]
                global_j = Ti[]
            else
                global_data = nothing
                global_i = nothing
                global_j = nothing
            end
            local_data = Tf[]
            this_block_global_i = Ti[]
            this_block_global_j = Ti[]
            local_i = Ti[]
            local_j = Ti[]

            row_global_offset = 0
            row_local_offset = 0
            for (variable_row, row_dims) ∈ zip(result_variable_blocks, variable_dimensions)
                column_global_offset = 0
                column_local_offset = 0
                for (variable_block, column_dims) ∈ zip(variable_row, variable_dimensions)
                    this_global_data, this_global_i, this_global_j, this_local_data,
                        this_this_block_global_i, this_this_block_global_j, this_local_i,
                        this_local_j = variable_block
                    if global_data !== nothing
                        for x ∈ this_global_data
                            push!(global_data, x)
                        end
                        for i ∈ this_global_i
                            push!(global_i, i + row_global_offset)
                        end
                        for j ∈ this_global_j
                            push!(global_j, j + column_global_offset)
                        end
                    end
                    for x ∈ this_local_data
                        push!(local_data, x)
                    end
                    for i ∈ this_this_block_global_i
                        push!(this_block_global_i, i + row_global_offset)
                    end
                    for j ∈ this_this_block_global_j
                        push!(this_block_global_j, j + column_global_offset)
                    end
                    for i ∈ this_local_i
                        push!(local_i, i + row_local_offset)
                    end
                    for j ∈ this_local_j
                        push!(local_j, j + column_local_offset)
                    end

                    column_global_offset += prod(d.n for d ∈ dimensions[column_dims])
                    column_local_offset += prod(d.n_local for d ∈ dimensions[column_dims])
                end
                row_global_offset += prod(d.n for d ∈ dimensions[row_dims])
                row_local_offset += prod(d.n_local for d ∈ dimensions[row_dims])
            end

            return global_data, global_i, global_j, local_data, this_block_global_i,
                   this_block_global_j, local_i, local_j
        else
            if rank == 0
                global_data = var_tuple_from_result(1)
            else
                global_data = nothing
            end
        end
        return global_data, Tuple(var_tuple_from_result(i) for i ∈ 2:5)...
    else
        if rank == 0
            global_matrix_tuple = var_tuple_from_result(1)
            global_matrix = mortar(reshape([global_matrix_tuple[flat_i%n_variables+1][flat_i÷n_variables+1]
                                            for flat_i ∈ 0:n_variables^2-1],
                                           n_variables, n_variables))
        else
            global_matrix = nothing
        end
        local_matrix = var_tuple_from_result(2)
        return global_matrix, local_matrix
    end
end

function remove_duplicates_from_global_vector(x_global_with_dups,
                                              dimensions::Vector{<:Dimension})
    n_tuple = Tuple(d.n for d ∈ dimensions)
    return remove_duplicates_from_global_vector(x_global_with_dups, dimensions, n_tuple)
end
function remove_duplicates_from_global_vector(x_global_with_dups,
                                              dimensions::Vector{<:Dimension},
                                              n_tuple::Tuple)
    if any(d.periodic for d ∈ dimensions)
        n = prod(d.periodic ? d.n - 1 : d.n for d ∈ dimensions)
        x_global = fill(NaN, n)
        counter = 0
        global_cartinds = CartesianIndices(n_tuple)
        for i_global ∈ 1:length(x_global_with_dups)
            inds = global_cartinds[i_global]
            if any(d.periodic && i == d.n for (d, i)
                   ∈ zip(reverse(dimensions), reverse(Tuple(inds))))
                i_dup = 0
                for (d, i) ∈ zip(reverse(dimensions), reverse(Tuple(inds)))
                    n = d.periodic ? d.n - 1 : d.n
                    if d.periodic && i == d.n
                        i = 1
                    end
                    i_dup = n * i_dup + i - 1
                end
                i_dup += 1
                x_global_with_dups[i_global] = x_global[i_dup]
            else
                counter += 1
                x_global[counter] = x_global_with_dups[i_global]
            end
        end
        return x_global
    else
        return x_global_with_dups
    end
end

function assemble_and_scatter_global_rhs(dimensions::Vector{<:Dimension}, comm::MPI.Comm,
                                         distributed_comm::Union{MPI.Comm,Nothing},
                                         shared_comm::MPI.Comm, allocate_shared_float,
                                         rng)
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    distributed_comm_size = comm_size ÷ shared_comm_size
    shared_comm_rank = MPI.Comm_rank(shared_comm)
    n_total = prod(d.n for d ∈ dimensions)
    n_local = prod(d.n_local for d ∈ dimensions)

    rhs_global = nothing
    rhs_local = allocate_shared_float(n_local)

    if rank == 0
        rhs_global_with_dups = rand(rng, n_total)
        rhs_global = remove_duplicates_from_global_vector(rhs_global_with_dups, dimensions)

        local_block_irank_lists = [get_irank_list(irank, dimensions)
                                   for irank ∈ 0:distributed_comm_size-1]
        local_block_indices_list =
            get_rhs_indices_for_all_local_blocks(dimensions, local_block_irank_lists)

        for rank ∈ 1:distributed_comm_size-1
            local_inds = local_block_indices_list[rank+1]
            rhs_local .= rhs_global_with_dups[local_inds]
            MPI.Send(rhs_local, distributed_comm; dest=rank)
        end

        local_inds = local_block_indices_list[1]
        rhs_local .= rhs_global_with_dups[local_inds]
    elseif shared_comm_rank == 0
        MPI.Recv!(rhs_local, distributed_comm; source=0)
    end

    return rhs_global, rhs_local
end

function assemble_and_scatter_global_multi_variable_rhs(
             dimensions::Vector{<:Dimension}, variable_dimensions, comm::MPI.Comm,
             distributed_comm::Union{MPI.Comm,Nothing}, shared_comm::MPI.Comm,
             allocate_shared_float, rng)

    nd = length(dimensions)
    variable_dimensions = Tuple(vdims === nothing ? (1:nd) : vdims
                                for vdims ∈ variable_dimensions)

    rhs_global, rhs_local =
        assemble_and_scatter_global_rhs(dimensions[variable_dimensions[1]], comm,
                                        distributed_comm, shared_comm,
                                        allocate_shared_float, rng)
    for vdims ∈ variable_dimensions[2:end]
        new_rhs_global, new_rhs_local =
            assemble_and_scatter_global_rhs(dimensions[vdims], comm, distributed_comm,
                                            shared_comm, allocate_shared_float, rng)
        if rhs_global !== nothing
            rhs_global = vcat(rhs_global, new_rhs_global)
        end
        rhs_local = vcat(rhs_local, new_rhs_local)
    end

    return rhs_global, rhs_local
end

function gather_vector(x_local::AbstractVector, dimensions::Vector{<:Dimension},
                       variable_dimensions, comm::Union{MPI.Comm},
                       distributed_comm::Union{MPI.Comm,Nothing}, shared_comm::MPI.Comm)
    if MPI.Comm_rank(shared_comm) > 0
        return nothing
    end
    nd = length(dimensions)
    variable_dimensions = Tuple(vdim === nothing ? (1:nd) : vdim
                                for vdim ∈ variable_dimensions)
    x_global_list = Vector{eltype(x_local)}[]
    offset = 0
    for vdims ∈ variable_dimensions
        this_dimensions = dimensions[vdims]
        var_length_local = prod(d.n_local for d ∈ this_dimensions)
        this_x_local = @view x_local[offset+1:offset+var_length_local]
        this_x_global = gather_single_variable_vector(this_x_local, this_dimensions, comm,
                                                      distributed_comm, shared_comm)
        if MPI.Comm_rank(distributed_comm) == 0
            push!(x_global_list, this_x_global)
        end
        offset += var_length_local
    end

    if MPI.Comm_rank(distributed_comm) > 0
        return nothing
    end

    x_global = vcat(x_global_list...)

    return x_global
end

function gather_single_variable_vector(x_local::AbstractVector,
                                       dimensions::Vector{<:Dimension},
                                       comm::Union{MPI.Comm},
                                       distributed_comm::Union{MPI.Comm,Nothing},
                                       shared_comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)
    shared_comm_rank = MPI.Comm_rank(shared_comm)
    shared_comm_size = MPI.Comm_size(shared_comm)
    distributed_comm_size = comm_size ÷ shared_comm_size

    x_global = nothing
    if rank == 0
        n_total = prod(d.n for d ∈ dimensions)
        x_global_with_dups = fill(NaN, n_total)

        local_block_irank_lists = [get_irank_list(irank, dimensions)
                                   for irank ∈ 0:distributed_comm_size-1]
        local_block_indices_list =
            get_rhs_indices_for_all_local_blocks(dimensions, local_block_irank_lists)

        # First add root's contributions to x_global.
        @views x_global_with_dups[local_block_indices_list[1]] .= x_local

        # Collect contributions from all other ranks. Overlapping points are overwritten,
        # but this should be OK because the overlapping points should be identical on all
        # processes anyway.
        for rank ∈ 1:distributed_comm_size-1
            MPI.Recv!(x_local, distributed_comm; source=rank)
            @views x_global_with_dups[local_block_indices_list[rank+1]] .= x_local
        end

        x_global = remove_duplicates_from_global_vector(x_global_with_dups, dimensions)
    elseif shared_comm_rank == 0
        MPI.Send(x_local, distributed_comm; dest=0)
    end

    return x_global
end

function generate_bool_permutations(n::Integer)
    return generate_bool_permutations(Val(n))
end
function generate_bool_permutations(N::Val)
    perms = Vector{Bool}[]
    for inds ∈ CartesianIndices(ntuple(i->2, N))
        this_perm = [Bool(i-1) for i ∈ Tuple(inds)]
        push!(perms, this_perm)
    end
    return perms
end

function get_nrank_permutations(nelement_list, nrank)
    nrank_list = Vector{Int64}[]
    ndim = length(nelement_list)
    nrank_factors = factor(Vector, nrank)
    function recursive_push_nrank!(remaining_nrank_factors, this_nrank_list, dim)
        if dim == 1
            remaining_nrank = prod(remaining_nrank_factors; init=1)
            if nelement_list[1] % remaining_nrank == 0
                this_nrank_list[1] = remaining_nrank
                push!(nrank_list, this_nrank_list)
            end
            return nothing
        end
        for this_factors ∈ unique(collect(combinations(remaining_nrank_factors)))
            this_nrank = prod(this_factors; init=1)
            if nelement_list[dim] % this_nrank == 0
                new_nrank_list = copy(this_nrank_list)
                new_nrank_list[dim] = this_nrank
                new_remaining_nrank_factors = copy(remaining_nrank_factors)
                for f ∈ this_factors
                    i = searchsortedfirst(new_remaining_nrank_factors, f)
                    popat!(new_remaining_nrank_factors, i)
                end
                recursive_push_nrank!(new_remaining_nrank_factors, new_nrank_list, dim - 1)
            end
        end
        return nothing
    end
    recursive_push_nrank!(nrank_factors, zeros(ndim), ndim)
    return nrank_list
end

function get_iranks(nrank_list, rank)
    irank_list = similar(nrank_list)
    for (i, nrank) ∈ enumerate(nrank_list)
        rank, this_irank = divrem(rank, nrank)
        irank_list[i] = this_irank
    end
    return irank_list
end

function get_flat_global_indices(dimensions_for_variables)
    if isa(dimensions_for_variables, Vector{<:Dimension})
        dimensions_for_variables = [dimensions_for_variables]
    end

    function getglob(dims, current_inds)
        if isempty(dims)
            return current_inds
        end
        new_inds = Int64[]
        lastdim = dims[end]
        n = lastdim.n
        ginds = lastdim.global_inds
        for i ∈ current_inds
            i = (i - 1) * n
            for g ∈ ginds
                push!(new_inds, i + g)
            end
        end
        return getglob(dims[1:end-1], new_inds)
    end

    offset = 0
    globinds = Int64[]
    for dimensions ∈ dimensions_for_variables
        globinds = vcat(globinds, offset .+ getglob(dimensions, Int64[1]))
        offset += prod(d.n for d ∈ dimensions)
    end

    return globinds
end
