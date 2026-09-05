struct BlockS{Nvar,Ti,Tm,Trange,Tdbr,Tsync}
    matrix::NTuple{Nvar,NTuple{Nvar,Tm}}
    indices::NTuple{Nvar,Trange}
    column_ranges_partial::NTuple{Nvar,UnitRange{Ti}}
    dense_boundaries_ranges::Tdbr
    dense_boundaries_partial_ranges::Tdbr
    synchronize_shared::Tsync

    function BlockS(matrix::NTuple{Nvar,NTuple{Nvar,Tm}},
                    local_bottom_vector_indices::NTuple{Nvar,Tind}, shared_comm,
                    full_dense_boundaries_ranges, full_dense_boundaries_partial_ranges,
                    allocate_shared_float::F,
                    synchronize_shared::Tsync) where {Nvar,Tm,Tind,F,Tsync}
        Ti = eltype(local_bottom_vector_indices[1])
        shared_comm_size = MPI.Comm_size(shared_comm)
        shared_comm_rank = MPI.Comm_rank(shared_comm)

        ncol = Tuple(length(bvi) for bvi ∈ local_bottom_vector_indices)
        cols_per_proc = Tuple((nc + shared_comm_size - 1) ÷ shared_comm_size for nc ∈ ncol)
        column_ranges_partial = Tuple(shared_comm_rank*cpp+1:min((shared_comm_rank+1)*cpp,nc)
                                      for (cpp, nc) ∈ zip(cols_per_proc, ncol))

        if Tm <: SharedSparseBuffer
            n_flat = Tuple(Tuple(length(m.nzval) for m ∈ row) for row ∈ matrix)
            entries_per_proc = Tuple(Tuple((n + shared_comm_size - 1) ÷ shared_comm_size
                                           for n ∈ nrow)
                                     for nrow ∈ n_flat)
        end

        if full_dense_boundaries_ranges !== nothing
            dense_boundaries_ranges =
                [[searchsortedfirst(li,first(r)):searchsortedfirst(li,last(r)) for r ∈ dbr]
                 for (dbr, li) ∈ zip(full_dense_boundaries_ranges, local_bottom_vector_indices)]
            dense_boundaries_partial_ranges =
                [[searchsortedfirst(li,first(r)):searchsortedfirst(li,last(r)) for r ∈ dbr]
                 for (dbr, li) ∈ zip(full_dense_boundaries_partial_ranges, local_bottom_vector_indices)]
        else
            dense_boundaries_ranges = nothing
            dense_boundaries_partial_ranges = nothing
        end

        return new{Nvar,Ti,Tm,Tind,typeof(dense_boundaries_ranges),Tsync}(
                   matrix, local_bottom_vector_indices, column_ranges_partial,
                   dense_boundaries_ranges, dense_boundaries_partial_ranges,
                   synchronize_shared)
    end
end

struct BlockDenseS{Nvar,Ti,Tm,Tind,Tdbr,Tsync}
    matrix::Tm
    indices::NTuple{Nvar,Tind}
    ranges::NTuple{Nvar,UnitRange{Ti}}
    partial_indices::NTuple{Nvar,Tind}
    partial_ranges::NTuple{Nvar,UnitRange{Ti}}
    dense_boundaries_ranges::Tdbr
    dense_boundaries_partial_ranges::Tdbr
    synchronize_shared::Tsync

    function BlockDenseS(matrix::Tm, local_bottom_vector_indices::NTuple{Nvar,Tind},
                         shared_comm, full_dense_boundaries_ranges,
                         full_dense_boundaries_partial_ranges, allocate_shared_float::F,
                         synchronize_shared::Tsync) where {Nvar,Tm<:AbstractMatrix,Tind,F,Tsync}
        Ti = eltype(local_bottom_vector_indices[1])
        shared_comm_size = MPI.Comm_size(shared_comm)
        shared_comm_rank = MPI.Comm_rank(shared_comm)

        block_n = Tuple(length(bvi) for bvi ∈ local_bottom_vector_indices)
        block_n_per_proc = Tuple((nc + shared_comm_size - 1) ÷ shared_comm_size for nc ∈ block_n)
        block_range_offsets = vcat(0, cumsum(block_n[1:Nvar-1])...)
        ranges = Tuple(offset .+ (1:n) for (offset, n) ∈ zip(block_range_offsets,
                                                             block_n))
        partial_ranges_without_offset = Tuple(shared_comm_rank*cpp+1:min((shared_comm_rank+1)*cpp,nc)
                                              for (cpp, nc) ∈ zip(block_n_per_proc, block_n))
        partial_ranges = Tuple(offset .+ pr for (offset, pr) ∈ zip(block_range_offsets,
                                                                   partial_ranges_without_offset))
        partial_indices = Tuple(inds[pr] for (inds, pr) ∈ zip(local_bottom_vector_indices,
                                                              partial_ranges_without_offset))

        if full_dense_boundaries_ranges !== nothing
            dense_boundaries_ranges =
                [[searchsortedfirst(li,first(r))+offset:searchsortedfirst(li,last(r))+offset for r ∈ dbr]
                 for (dbr, li, offset) ∈ zip(full_dense_boundaries_ranges,
                                           local_bottom_vector_indices, block_range_offsets)]
            dense_boundaries_partial_ranges =
                [[searchsortedfirst(li,first(r))+offset:searchsortedfirst(li,last(r))+offset for r ∈ dbr]
                 for (dbr, li, offset) ∈ zip(full_dense_boundaries_partial_ranges,
                                           local_bottom_vector_indices, block_range_offsets)]
        else
            dense_boundaries_ranges = nothing
            dense_boundaries_partial_ranges = nothing
        end

        return new{Nvar,Ti,Tm,Tind,typeof(dense_boundaries_ranges),Tsync}(
                   matrix, local_bottom_vector_indices, ranges, partial_indices,
                   partial_ranges, dense_boundaries_ranges,
                   dense_boundaries_partial_ranges, synchronize_shared)
    end
end

function add_D_to_schur_complement!(schur_complement::BlockS{Nvar},
                                    full_A::NTuple{Nvar,NTuple{Nvar,T}}) where {Nvar,T}
    @inbounds begin
        # Only get the local rows for D, so just add these to the local rows of
        # `schur_complement`.
        sc_matrix = schur_complement.matrix
        indices = schur_complement.indices
        column_ranges_partial = schur_complement.column_ranges_partial
        for (vcol, ci, cr) ∈ zip(1:Nvar, indices, column_ranges_partial), (vrow, ri) ∈ zip(1:Nvar, indices)
            sc_matrix_variable_block = sc_matrix[vrow][vcol]
            A_variable_block = full_A[vrow][vcol]
            if isa(sc_matrix_variable_block, SharedSparseBuffer)
                if isa(A_variable_block, AbstractSparseMatrixCSC)
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval
                    sc_colptr = sc_matrix_variable_block.colptr
                    sc_rowval_list = sc_matrix_variable_block.rowval_list
                    sc_nzval = sc_matrix_variable_block.nzval

                    for j ∈ cr
                        first_i = sc_colptr[j]
                        last_i = sc_colptr[j+1] - 1
                        if last_i < first_i
                            continue
                        end
                        # Assume D and schur_complement have a similar pattern of non-zeros, so no
                        # significant gain from using searchsortedlast() to find the first row_i that
                        # will be within the non-zeros of D.
                        row_i = 1

                        full_j = ci[j]
                        full_first_i = full_A_colptr[full_j]
                        full_last_i = full_A_colptr[full_j+1]-1
                        if full_last_i < full_first_i
                            sc_nzval[first_i:last_i] .= 0.0
                            continue
                        end

                        sc_col_rv = sc_rowval_list[j]
                        first_row = sc_col_rv[1]
                        last_row_i = length(sc_col_rv)
                        full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_row) - 1, 1) + full_first_i - 1
                        while row_i ≤ last_row_i
                            row = ri[sc_col_rv[row_i]]
                            full_row = full_A_rowval[full_flat_i]
                            if row == full_row
                                sc_nzval[row_i+first_i-1] = full_A_nzval[full_flat_i]
                                row_i += 1
                                full_flat_i += 1
                            elseif row < full_row
                                sc_nzval[row_i+first_i-1] = 0.0
                                row_i += 1
                            else
                                full_flat_i += 1
                            end
                            if full_flat_i > full_last_i
                                sc_nzval[row_i+first_i-1:last_row_i+first_i-1] .= 0.0
                                break
                            end
                        end
                    end
                elseif isa(A_variable_block, SharedSparseBuffer)
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval_list = A_variable_block.rowval_list
                    full_A_nzval = A_variable_block.nzval
                    sc_colptr = sc_matrix_variable_block.colptr
                    sc_rowval_list = sc_matrix_variable_block.rowval_list
                    sc_nzval = sc_matrix_variable_block.nzval

                    for j ∈ cr
                        first_i = sc_colptr[j]
                        last_i = sc_colptr[j+1] - 1
                        if last_i < first_i
                            continue
                        end
                        # Assume D and schur_complement have a similar pattern of non-zeros, so no
                        # significant gain from using searchsortedlast() to find the first row_i that
                        # will be within the non-zeros of D.
                        row_i = 1

                        full_j = ci[j]
                        full_first_i = full_A_colptr[full_j]
                        full_last_i = full_A_colptr[full_j+1]-1
                        if full_last_i < full_first_i
                            sc_nzval[first_i:last_i] .= 0.0
                            continue
                        end

                        sc_col_rv = sc_rowval_list[j]
                        first_row = sc_col_rv[1]
                        last_row_i = length(sc_col_rv)

                        full_col_rv = full_A_rowval_list[full_j]
                        full_last_row_i = length(full_col_rv)
                        full_row_i = max(searchsortedlast(full_col_rv, first_row) - 1, 1)
                        while row_i ≤ last_row_i
                            row = ri[sc_col_rv[row_i]]
                            full_row = full_col_rv[full_row_i]
                            if row == full_row
                                sc_nzval[row_i+first_i-1] = full_A_nzval[full_row_i+full_first_i-1]
                                row_i += 1
                                full_row_i += 1
                            elseif row < full_row
                                sc_nzval[row_i+first_i-1] = 0.0
                                row_i += 1
                            else
                                full_row_i += 1
                            end
                            if full_row_i > full_last_row_i
                                sc_nzval[row_i+first_i-1:last_row_i+first_i-1] .= 0.0
                                break
                            end
                        end
                    end
                else
                    error("Unsupported type '$(typeof(A_variable_block))' for `A_variable_block`.")
                end
            else
                error("Unexpected type for sc_matrix_variable_block "
                      * "($(typeof(sc_matrix_variable_block))).")
            end
        end

        dense_boundaries_ranges = schur_complement.dense_boundaries_ranges
        if dense_boundaries_ranges !== nothing
            # 'Dense boundaries' entries are already stored in another buffer, so
            # need to zero them out here. As the entries to be handled are those
            # where both row and column are within the 'dense boundary' (and those
            # are not all of the entries in a given row/column) it is not possible
            # to skip the entries by modifying the `indices` (modifying `indices`
            # would skip entries where either row or column is within the skipped
            # range). It is simpler (possibly even more efficient?) to zero out
            # the 'dense boundaries' entries here.
            schur_complement.synchronize_shared()
            for (ranges, partial_ranges) ∈
                    zip(eachcol(dense_boundaries_ranges),
                        eachcol(schur_complement.dense_boundaries_partial_ranges))
                for (vcol, col_ranges) ∈ zip(1:Nvar, partial_ranges)
                    for (vrow, row_ranges) ∈ zip(1:Nvar, ranges)
                        sc_matrix_variable_block = sc_matrix[vrow][vcol]
                        colptr = sc_matrix_variable_block.colptr
                        rowval_list = sc_matrix_variable_block.rowval_list
                        nzval = sc_matrix_variable_block.nzval
                        for cr ∈ col_ranges, rr ∈ row_ranges
                            row_start = first(rr)
                            row_end = last(rr)
                            for j ∈ cr
                                col_start = colptr[j]
                                rv = rowval_list[j]
                                first_row_i = searchsortedfirst(rv, row_start)
                                for row_i ∈ first_row_i:length(rv)
                                    i = rv[row_i]
                                    if i > row_end
                                        break
                                    end
                                    nzval[row_i+col_start-1] = 0.0
                                end
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end

function add_D_to_schur_complement!(schur_complement::BlockDenseS{Nvar},
                                    full_A::NTuple{Nvar,NTuple{Nvar,T}}) where {Nvar,T}
    @inbounds begin
        # Only get the local rows for D, so just add these to the local rows of
        # `schur_complement`.
        sc_matrix = schur_complement.matrix
        indices = schur_complement.indices
        ranges = schur_complement.ranges
        partial_indices = schur_complement.partial_indices
        partial_ranges = schur_complement.partial_ranges
        for (vcol, ci, cr) ∈ zip(1:Nvar, partial_indices, partial_ranges),
                (vrow, ri, rr) ∈ zip(1:Nvar, indices, ranges)
            A_variable_block = full_A[vrow][vcol]
            first_row = first(ri)
            if isa(A_variable_block, AbstractSparseMatrixCSC)
                full_A_colptr = A_variable_block.colptr
                full_A_rowval = A_variable_block.rowval
                full_A_nzval = A_variable_block.nzval

                for (j, full_j) ∈ zip(cr, ci)
                    full_first_i = full_A_colptr[full_j]
                    full_last_i = full_A_colptr[full_j+1]-1
                    if full_last_i < full_first_i
                        sc_matrix[rr,j] .= 0.0
                        continue
                    end

                    last_full_row = full_A_rowval[full_last_i]
                    full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_row) - 1, 1) + full_first_i - 1
                    for (i, full_i) ∈ zip(rr, ri)
                        while full_flat_i < full_last_i && full_A_rowval[full_flat_i] < full_i
                            full_flat_i += 1
                        end
                        if full_i == full_A_rowval[full_flat_i]
                            sc_matrix[i,j] = full_A_nzval[full_flat_i]
                            full_flat_i += 1
                            if full_flat_i > full_last_i
                                sc_matrix[i+1:rr[end],j] .= 0.0
                                break
                            end
                        else
                            sc_matrix[i,j] = 0.0
                        end
                        if full_i > last_full_row
                            sc_matrix[i+1:rr[end],j] .= 0.0
                            break
                        end
                    end
                end
            elseif isa(A_variable_block, SharedSparseBuffer)
                full_A_colptr = A_variable_block.colptr
                full_A_rowval_list = A_variable_block.rowval_list
                full_A_nzval = A_variable_block.nzval

                for (j, full_j) ∈ zip(cr, ci)
                    full_first_i = full_A_colptr[full_j]
                    full_last_i = full_A_colptr[full_j+1]-1
                    if full_last_i < full_first_i
                        sc_matrix[rr,j] .= 0.0
                        continue
                    end

                    full_col_rv = full_A_rowval_list[full_j]
                    last_full_row = full_col_rv[end]
                    last_full_row_i = length(full_col_rv)
                    full_row_i = max(searchsortedlast(full_col_rv, first_row) - 1, 1)
                    for (i, full_i) ∈ zip(rr, ri)
                        while full_row_i < last_full_row_i && full_col_rv[full_row_i] < full_i
                            full_row_i += 1
                        end
                        if full_i == full_col_rv[full_row_i]
                            sc_matrix[i,j] = full_A_nzval[full_row_i+full_first_i-1]
                            full_row_i += 1
                            if full_row_i > last_full_row_i
                                sc_matrix[i+1:rr[end],j] .= 0.0
                                break
                            end
                        else
                            sc_matrix[i,j] = 0.0
                        end
                        if full_i > last_full_row
                                sc_matrix[i+1:rr[end],j] .= 0.0
                            break
                        end
                    end
                end
            else
                error("Unsupported type '$(typeof(A_variable_block))' for `A_variable_block`.")
            end
        end

        dense_boundaries_ranges = schur_complement.dense_boundaries_ranges
        if dense_boundaries_ranges !== nothing
            # 'Dense boundaries' entries are already stored in another buffer, so
            # need to zero them out here. As the entries to be handled are those
            # where both row and column are within the 'dense boundary' (and those
            # are not all of the entries in a given row/column) it is not possible
            # to skip the entries by modifying the `indices` (modifying `indices`
            # would skip entries where either row or column is within the skipped
            # range). It is simpler (possibly even more efficient?) to zero out
            # the 'dense boundaries' entries here.
            schur_complement.synchronize_shared()
            for (ranges, partial_ranges) ∈
                    zip(eachcol(dense_boundaries_ranges),
                        eachcol(schur_complement.dense_boundaries_partial_ranges))
                for col_ranges ∈ partial_ranges, row_ranges ∈ ranges
                    for cr ∈ col_ranges, rr ∈ row_ranges
                        sc_matrix[rr,cr] .= 0.0
                    end
                end
            end
        end
        return nothing
    end
end
