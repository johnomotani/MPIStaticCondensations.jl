struct BlockS{Nvar,Ti,Tm,Trange}
    matrix::NTuple{Nvar,NTuple{Nvar,Tm}}
    indices::NTuple{Nvar,Trange}
    column_ranges_partial::NTuple{Nvar,UnitRange{Ti}}
    flat_ranges_partial::NTuple{Nvar,NTuple{Nvar,UnitRange{Ti}}}

    function BlockS(matrix::NTuple{Nvar,NTuple{Nvar,Tm}},
                    local_bottom_vector_indices::NTuple{Nvar,Tind}, shared_comm,
                    allocate_shared_float::F) where {Nvar,Tm,Tind,F}
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
            flat_ranges_partial = Tuple(Tuple(shared_comm_rank*entries_per_proc[ivar][jvar]+1:min((shared_comm_rank+1)*entries_per_proc[ivar][jvar],n_flat[ivar][jvar])
                                              for jvar ∈ 1:Nvar)
                                        for ivar ∈ 1:Nvar)
        else
            flat_ranges_partial = ntuple(i->column_ranges_partial, Nvar)
        end

        return new{Nvar,Ti,Tm,Tind}(
                   matrix, local_bottom_vector_indices, column_ranges_partial,
                   flat_ranges_partial)
    end
end

struct BlockDenseS{Nvar,Ti,Tm,Tind}
    matrix::Tm
    indices::NTuple{Nvar,Tind}
    partial_indices::NTuple{Nvar,Tind}
    partial_ranges::NTuple{Nvar,UnitRange{Ti}}
    column_range_partial::UnitRange{Ti}

    function BlockDenseS(matrix::Tm, local_bottom_vector_indices::NTuple{Nvar,Tind},
                         shared_comm,
                         allocate_shared_float::F) where {Nvar,Tm<:AbstractMatrix,Tind,F}
        Ti = eltype(local_bottom_vector_indices[1])
        shared_comm_size = MPI.Comm_size(shared_comm)
        shared_comm_rank = MPI.Comm_rank(shared_comm)

        ncol = size(matrix, 2)
        cols_per_proc = (ncol + shared_comm_size - 1) ÷ shared_comm_size
        column_range_partial = shared_comm_rank*cols_per_proc+1:min((shared_comm_rank+1)*cols_per_proc,ncol)

        block_n = Tuple(length(bvi) for bvi ∈ local_bottom_vector_indices)
        block_n_per_proc = Tuple((nc + shared_comm_size - 1) ÷ shared_comm_size for nc ∈ block_n)
        block_range_offsets = vcat(0, cumsum(block_n[1:Nvar-1]))
        partial_ranges = Tuple(offset .+ shared_comm_rank*cpp+1:min((shared_comm_rank+1)*cpp,nc)
                                      for (offset, cpp, nc) ∈ zip(block_range_offsets,
                                                                  block_n_per_proc,
                                                                  block_n))
        partial_indices = Tuple(inds[pr] for (inds, pr) ∈ zip(local_bottom_vector_indices,
                                                              partial_ranges))

        return new{Nvar,Ti,Tm,Tind}(
                   matrix, local_bottom_vector_indices, partial_indices,
                   partial_ranges, column_range_partial)
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
                            continue
                        end

                        sc_col_rv = sc_rowval_list[j]
                        first_row = sc_col_rv[1]
                        last_row_i = length(sc_col_rv)
                        full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_row) - 1, 1) + full_first_i - 1
                        while row_i ≤ last_row_i && full_flat_i ≤ full_last_i
                            row = ri[sc_col_rv[row_i]]
                            full_row = full_A_rowval[full_flat_i]
                            if row == full_row
                                sc_nzval[row_i+first_i-1] += full_A_nzval[full_flat_i]
                                row_i += 1
                                full_flat_i += 1
                            elseif row < full_row
                                row_i += 1
                            else
                                full_flat_i += 1
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
                            continue
                        end

                        sc_col_rv = sc_rowval_list[j]
                        first_row = sc_col_rv[1]
                        last_row_i = length(sc_col_rv)

                        full_col_rv = full_A_rowval_list[full_j]
                        full_last_row_i = length(full_col_rv)
                        full_row_i = max(searchsortedlast(full_col_rv, first_row) - 1, 1)
                        while row_i ≤ last_row_i && full_row_i ≤ full_last_row_i
                            row = ri[sc_col_rv[row_i]]
                            full_row = full_col_rv[full_row_i]
                            if row == full_row
                                sc_nzval[row_i+first_i-1] += full_A_nzval[full_row_i+full_first_i-1]
                                row_i += 1
                                full_row_i += 1
                            elseif row < full_row
                                row_i += 1
                            else
                                full_row_i += 1
                            end
                        end
                    end
                else
                    error("Unsupported type '$(typeof(A_variable_block))' for `A_variable_block`.")
                end
            else
                if isa(A_variable_block, AbstractSparseMatrixCSC)
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval

                    nrow = length(ri)
                    first_full_row = sc_indices[1]
                    for j ∈ cr
                        full_j = sc_indices[j]
                        full_first_i = full_A_colptr[full_j]
                        full_last_i = full_A_colptr[full_j+1]-1
                        if full_last_i < full_first_i
                            continue
                        end

                        full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_full_row) - 1, 1) + full_first_i - 1
                        irow = 1
                        while irow ≤ nrow && full_flat_i ≤ full_last_i
                            row = sc_indices[irow]
                            full_row = full_A_rowval[full_flat_i]
                            if row == full_row
                                sc_matrix_variable_block[irow,j] += full_A_nzval[full_flat_i]
                                irow += 1
                                full_flat_i += 1
                            elseif row < full_row
                                irow += 1
                            else
                                full_flat_i += 1
                            end
                        end
                    end
                elseif isa(A_variable_block, SharedSparseBuffer)
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval_list = A_variable_block.rowval_list
                    full_A_nzval = A_variable_block.nzval
                    sc_indices = schur_complement.indices

                    nrow = length(sc_indices)
                    first_full_row = sc_indices[1]
                    for j ∈ cr
                        full_j = sc_indices[j]
                        full_first_i = full_A_colptr[full_j]
                        full_last_i = full_A_colptr[full_j+1]-1
                        if full_last_i < full_first_i
                            continue
                        end

                        full_col_rv = full_A_rowval_list[full_j]
                        full_last_row_i = length(full_col_rv)
                        full_row_i = max(searchsortedlast(full_col_rv, first_full_row) - 1, 1)
                        irow = 1
                        while irow ≤ nrow && full_row_i ≤ full_last_row_i
                            row = sc_indices[irow]
                            full_row = full_col_rv[full_row_i]
                            if row == full_row
                                sc_matrix_variable_block[irow,j] += full_A_nzval[full_row_i+full_first_i-1]
                                irow += 1
                                full_row_i += 1
                            elseif row < full_row
                                irow += 1
                            else
                                full_row_i += 1
                            end
                        end
                    end
                else
                    error("Unsupported type '$(typeof(A_variable_block))' for `A_variable_block`.")
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
        partial_indices = schur_complement.partial_indices
        partial_ranges = schur_complement.partial_ranges
        for (vcol, ci, cr) ∈ zip(1:Nvar, partial_indices, partial_ranges),
                (vrow, ri) ∈ zip(1:Nvar, indices)
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
                        continue
                    end

                    last_full_row = full_A_rowval[full_last_i]
                    full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_row) - 1, 1) + full_first_i - 1
                    for (i, full_i) ∈ enumerate(ri)
                        while full_flat_i < full_last_i && full_A_rowval[full_flat_i] < full_i
                            full_flat_i += 1
                        end
                        if full_i == full_A_rowval[full_flat_i]
                            sc_matrix[i,j] += full_A_nzval[full_flat_i]
                            full_flat_i += 1
                            if full_flat_i > full_last_i
                                break
                            end
                        end
                        if full_i > last_full_row
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
                        continue
                    end

                    full_col_rv = full_A_rowval_list[full_j]
                    last_full_row = full_col_rv[end]
                    last_full_row_i = length(full_col_rv)
                    full_row_i = max(searchsortedlast(full_col_rv, first_row) - 1, 1)
                    for (i, full_i) ∈ enumerate(ri)
                        while full_row_i < last_full_row_i && full_col_rv[full_row_i] < full_i
                            full_row_i += 1
                        end
                        if full_i == full_col_rv[full_row_i]
                            sc_matrix[i,j] += full_A_nzval[full_row_i+full_first_i-1]
                            full_row_i += 1
                            if full_row_i > last_full_row_i
                                break
                            end
                        end
                        if full_i > last_full_row
                            break
                        end
                    end
                end
            else
                error("Unsupported type '$(typeof(A_variable_block))' for `A_variable_block`.")
            end
        end
        return nothing
    end
end
