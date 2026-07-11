struct BlockS{Ti,Tm,TCAiB}
    matrix::Tm
    C_dot_Ainv_dot_B::TCAiB
    flat_range::UnitRange{Ti}

    function BlockS(C_buffer_ncopies, allocate_shared_float::F) where F
        matrix = nothing

        C_dot_Ainv_dot_B = allocate_shared_float(C_buffer_ncopies, nnz(matrix))

        return new{Ti,typeof(matrix),typeof(C_dot_Ainv_dot_B)}(matrix, C_dot_Ainv_dot_B)
    end
end

function add_D_to_schur_complement!(schur_complement::BlockS, A)
    @inbounds begin
        # Only get the local rows for D, so just add these to the local rows of
        # `schur_complement`.
        A_colptr = A.colptr
        A_rowval = A.rowval
        A_nzval = A.nzval
        sc_matrix = schur_complement.matrix
        sc_colptr = sc_matrix.colptr
        sc_rowval = sc_matrix.rowval
        sc_nzval = sc_matrix.nzval
        sc_column_range_partial = schur_complement.column_range_partial
        sc_indices = schur_complement.indices

        nrow = length(sc_indices)
        last_full_row = sc_indices[end]
        for j ∈ sc_column_range_partial
            first_i = sc_colptr[j1]
            last_i = sc_colptr[j1+1] - 1
            if last_i < first_i
                continue
            end
            # Assume D and schur_complement have a similar pattern of non-zeros, so no
            # significant gain from using searchsortedlast() to find the first flat_i that
            # will be within the non-zeros of D.
            flat_i = first_i

            full_j = sc_indices[j]
            full_first_i = A_colptr[full_j]
            full_last_i = A_colptr[full_j+1]-1
            if full_last_i < full_first_i
                continue
            end

            first_row = sc_rowval[first_i]
            full_flat_i = max(searchsortedlast(A_rowval, first_row) - 1, 1)
            while flat_i ≤ last_i && full_flat_i ≤ full_last_i
                row = sc_indices[sc_rowval[flat_i]]
                full_row = A_rowval[full_flat_i]
                if row == full_row
                    sc_nzval[flat_i] += A_nzval[full_flat_i]
                    flat_i += 1
                    full_flat_i += 1
                elseif row < full_row
                    flat_i += 1
                else
                    full_flat_i += 1
                end
            end
        end
        return nothing
    end
end
