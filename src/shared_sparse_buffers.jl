function get_dim_indices!(dimensions, block_sizes, flat_i)
    @inbounds begin
        block_inds = zeros(Int64, length(dimensions))
        inner_inds = zeros(Int64, length(dimensions))
        for (i, d) ∈ enumerate(dimensions)
            flat_i, dim_i = divrem(flat_i, d.n_local)
            this_block_npoints = block_sizes[i] * (d.ngrid - 1)
            block_inds[i], inner_inds[i] = divrem(dim_i, this_block_npoints) .+ 1
        end
        return block_inds, inner_inds
    end
end

function add_all_row_inds!(rv, idim, dimensions, row_indices, rowind, row_count)
    @inbounds begin
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
                row_count[] += 1
            end
            return nothing
        end
        d = dimensions[idim]
        rowind *= d.n_local
        for i ∈ 1:d.n_local
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + i - 1,
                              row_count)
        end
        return nothing
    end
end

function add_row_inds!(rv, idim, dimensions, block_sizes, nblock_list, row_indices,
                       block_inds, inner_inds, rowind, row_count)
    @inbounds begin
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
                row_count[] += 1
            end
            return nothing
        end
        d = dimensions[idim]
        dn = d.n_local
        block_npoints = block_sizes[idim] * (d.ngrid - 1)
        iblock = block_inds[idim]
        iinner = inner_inds[idim]
        rowind *= dn
        is_first_point = (iinner == 1 && iblock == 1)
        is_last_point = (((iblock - 1) * block_npoints + iinner) == dn)
        if iinner == 1 && iblock > 1
            # Is a block boundary, so include points from previous block.
            row_offset = (iblock - 2) * block_npoints
            for row_inner ∈ 1:block_npoints
                add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                              row_indices, block_inds, inner_inds,
                              rowind + row_offset + row_inner - 1, row_count)
            end
        end
        row_offset = (iblock - 1) * block_npoints
        if d.dense_boundaries && is_first_point
            # This is the first or last point in a dimension that should have 'dense
            # boundaries'.
            block_start = 2
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + row_offset,
                              row_count)
        else
            block_start = 1
        end
        if d.dense_boundaries && is_last_point
            row_end = dn - 1
        else
            row_end = dn
        end
        if iblock > nblock_list[idim] && d.dense_boundaries
            # Have added all points already.
        elseif iblock > nblock_list[idim]
            # Creating entries for the last grid point, this is 'really'
            # iel=nelement_local-1, col_igr=ngrid, so only need to add the row_igr=1
            # point.
            add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list, row_indices,
                          block_inds, inner_inds, rowind + row_offset, row_count)
        else
            for row_inner ∈ block_start:block_npoints+1
                if row_offset + row_inner > row_end
                    # Do not want to advance past the end of the dimension. The last block in
                    # any dimension may not be full-sized.
                    break
                end
                add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                              row_indices, block_inds, inner_inds,
                              rowind + row_offset + row_inner - 1, row_count)
            end
        end
        if d.dense_boundaries && is_last_point
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + dn - 1,
                              row_count)
        end
        return nothing
    end
end

function get_shared_sparse_matrix_csc_buffer(dimensions::Vector{<:Dimension},
                                             shared_comm, allocate_shared_float::F1,
                                             allocate_shared_int::F2,
                                             block_sizes::Union{Vector{<:Integer},Nothing}=nothing,
                                             row_indices::Union{Vector{<:Integer},Nothing}=nothing,
                                             column_indices::Union{Vector{<:Integer},Nothing}=nothing;
                                             ind_type::Type=Int64) where {F1, F2}
    @inbounds begin
        n_local_list = [d.n_local for d ∈ dimensions]
        n_total = prod(n_local_list; init=1)
        if block_sizes === nothing
            block_sizes = ones(ind_type, length(dimensions))
        end
        if row_indices === nothing
            row_indices = 1:n_total
        end
        if column_indices === nothing
            column_indices = 1:n_total
        end
        nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
        nblock_list = [(nel + bs - 1) ÷ bs for (nel, bs) ∈ zip(nelement_local_list,
                                                               block_sizes)]
        m = length(row_indices)
        n = length(column_indices)
        if m == 0 || n == 0
            # FixedSparseCSC constructor errors when one of the matrix sizes is zero, and
            # there are no entries anyway so do not need shared-memory allocation.
            return spzeros(m, n)
        end

        for (idim, d) ∈ enumerate(dimensions)
            if d.dense_boundaries && any(x.nrank > 1 for x ∈ dimensions[1:idim-1])
                error("Dimensions to the left of a dimension with dense_boundaries=true "
                      * "(dimension $idim) cannot be distributed across multiple MPI "
                      * "shared-memory blocks")
            end
        end

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        n_colptr = Ref(-1)
        n_rowval = Ref(-1)
        if shared_comm_rank == 0
            # Columns can be 'equivalent' in the sense that they have exactly the same
            # non-empty row indices. In a single dimension, two points are equivalent if
            # they are in the interior of the same element, or if they are the point (in
            # case they are boundary points). Columns are equivalent if their positions in
            # every dimension are equivalent.
            cartinds_list = CartesianIndices(Tuple(d.n_local for d ∈ dimensions))
            function get_equivalents_lookup(d, nblock, block_size)
                eq_list = zeros(ind_type, d.n_local)
                eq = ind_type(1)
                count = ind_type(0)
                interior_range = 2:block_size*(d.ngrid-1)
                d_n_local = d.n_local
                if d.remove_boundaries
                    for _ ∈ 1:nblock
                        # Lower boundary
                        eq_list[count+=1] = eq
                        eq += 1
                        if count ≥ d_n_local - 1
                            # Last block may not be full size, so make sure not to go
                            # outside the bounds of the dimension.
                            break
                        end
                        for _ ∈ interior_range
                            eq_list[count+=1] = eq
                            if count == d_n_local - 1
                                # Last block may not be full size, so make sure not to go
                                # outside the bounds of the dimension.
                                break
                            end
                        end
                        eq += 1
                    end
                    # Upper boundary
                    if count == d_n_local - 1
                        eq_list[count+=1] = eq
                    end
                else
                    for _ ∈ 1:block_size*(d.ngrid-1)
                        eq_list[count+=1] = eq
                    end
                    eq += 1
                    for _ ∈ 2:nblock
                        # Lower boundary
                        eq_list[count+=1] = eq
                        eq += 1
                        if count ≥ d_n_local - 1
                            # Last block may not be full size, so make sure not to go
                            # outside the bounds of the dimension.
                            break
                        end
                        for _ ∈ interior_range
                            eq_list[count+=1] = eq
                            if count == d_n_local - 1
                                # Last block may not be full size, so make sure not to go
                                # outside the bounds of the dimension.
                                break
                            end
                        end
                        eq += 1
                    end
                    # Upper boundary
                    if count == d_n_local - 1
                        eq_list[count+=1] = eq - 1
                    end
                end
                return eq_list
            end
            equivalents_lookup_lists = [get_equivalents_lookup(d, nb, bs)
                                        for (d, nb, bs)
                                        ∈ zip(dimensions, nblock_list, block_sizes)]
            n_equivalents_list = [eq[end] for eq ∈ equivalents_lookup_lists]
            function equivalence_ind(flat_i)
                cartind = Tuple(cartinds_list[flat_i])
                eq_ind = ind_type(0)
                for (i, neq, eq_lookup) ∈ zip(reverse(cartind),
                                              reverse(n_equivalents_list),
                                              reverse(equivalents_lookup_lists))
                    eq_ind *= neq
                    eq_ind += eq_lookup[i] - 1
                end
                # eq_ind was constructed as 0-based index. Convert back to 1-based.
                eq_ind += 1
                return eq_ind
            end

            cp = ind_type[]
            rv = ind_type[]
            row_count = Ref(1)
            rv_lookup = Dict{ind_type,Vector{ind_type}}()
            for col ∈ column_indices
                push!(cp, length(rv) + 1)
                col_eq = equivalence_ind(col)
                if col_eq ∈ keys(rv_lookup)
                    col_rv = rv_lookup[col_eq]
                else
                    col_rv = ind_type[]
                    block_inds, inner_inds = get_dim_indices!(dimensions, block_sizes,
                                                              col - 1)
                    row_count[] = 1
                    add_row_inds!(col_rv, length(dimensions), dimensions, block_sizes,
                                  nblock_list, row_indices, block_inds, inner_inds, 0,
                                  row_count)
                    rv_lookup[col_eq] = col_rv
                end
                new_n = length(col_rv)
                resize!(rv, length(rv) + new_n)
                rv[end-new_n+1:end] .= col_rv
            end
            push!(cp, length(rv) + 1)

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

        # Use the 'experimental' FixedSparseCSC instead of SparseMatrixCSC to ensure that
        # the Vectors are not resized, reallocated, etc.
        return FixedSparseCSC(m, n, colptr, rowval, nzval)
    end
end
