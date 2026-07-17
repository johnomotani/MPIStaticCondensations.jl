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

function add_all_row_inds!(rv, idim, dimensions, row_indices, rowind, count, row_count)
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
                count[] += 1
                row_count[] += 1
            end
            return nothing
        end
        d = dimensions[idim]
        rowind *= d.n_local
        for i ∈ 1:d.n_local
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + i - 1, count,
                              row_count)
        end
        return nothing
    end
end

function add_row_inds!(rv, idim, dimensions, block_sizes, nblock_list, row_indices,
                       block_inds, inner_inds, rowind, count, row_count)
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
                count[] += 1
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
                              rowind + row_offset + row_inner - 1, count, row_count)
            end
        end
        row_offset = (iblock - 1) * block_npoints
        if d.dense_boundaries && is_first_point
            # This is the first or last point in a dimension that should have 'dense
            # boundaries'.
            block_start = 2
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + row_offset,
                              count, row_count)
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
                          block_inds, inner_inds, rowind + row_offset, count, row_count)
        else
            for row_inner ∈ block_start:block_npoints+1
                if row_offset + row_inner > row_end
                    # Do not want to advance past the end of the dimension. The last block in
                    # any dimension may not be full-sized.
                    break
                end
                add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                              row_indices, block_inds, inner_inds,
                              rowind + row_offset + row_inner - 1, count, row_count)
            end
        end
        if d.dense_boundaries && is_last_point
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + dn - 1,
                              count, row_count)
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
            cp = Int64[]
            rv = Int64[]
            count = Ref(1)
            row_count = Ref(1)

            for col ∈ column_indices
                push!(cp, count[])
                block_inds, inner_inds = get_dim_indices!(dimensions, block_sizes,
                                                          col - 1)
                row_count[] = 1
                add_row_inds!(rv, length(dimensions), dimensions, block_sizes,
                              nblock_list, row_indices, block_inds, inner_inds, 0, count,
                              row_count)
            end
            push!(cp, count[])

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
