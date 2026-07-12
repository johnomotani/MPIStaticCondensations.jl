struct BlockS{Ti,Tm,TCAiB,Trange}
    matrix::Tm
    C_dot_Ainv_dot_B::TCAiB
    indices::Trange
    column_range_partial::UnitRange{Ti}
    flat_range_partial::UnitRange{Ti}

    function BlockS(dimensions::Vector{<:Dimension}, indices,
                    block_sizes::Union{Vector{<:Integer},Nothing}, C_buffer_ncopies,
                    shared_comm, allocate_shared_float::F1,
                    allocate_shared_int::F2) where {F1,F2}
        Ti = eltype(indices)
        shared_comm_size = MPI.Comm_size(shared_comm)
        shared_comm_rank = MPI.Comm_rank(shared_comm)

        matrix = get_shared_sparse_matrix_csc_buffer(dimensions, shared_comm,
                                                     allocate_shared_float,
                                                     allocate_shared_int, block_sizes,
                                                     indices, indices; ind_type=Ti)

        n_flat = nnz(matrix)
        C_dot_Ainv_dot_B = allocate_shared_float(C_buffer_ncopies, n_flat)
        if shared_comm_rank == 0
            C_dot_Ainv_dot_B .= 0.0
        end

        ncol = length(indices)
        cols_per_proc = (ncol + shared_comm_size - 1) ÷ shared_comm_size
        column_range_partial = shared_comm_rank*cols_per_proc+1:min((shared_comm_rank+1)*cols_per_proc,ncol)

        entries_per_proc = (n_flat + shared_comm_size - 1) ÷ shared_comm_size
        flat_range_partial = shared_comm_rank*entries_per_proc+1:min((shared_comm_rank+1)*entries_per_proc,n_flat)

        return new{Ti,typeof(matrix),typeof(C_dot_Ainv_dot_B),typeof(indices)}(
                   matrix, C_dot_Ainv_dot_B, indices, column_range_partial,
                   flat_range_partial)
    end
end

function add_D_to_schur_complement!(schur_complement::BlockS, full_A)
    @inbounds begin
        # Only get the local rows for D, so just add these to the local rows of
        # `schur_complement`.
        full_A_colptr = full_A.colptr
        full_A_rowval = full_A.rowval
        full_A_nzval = full_A.nzval
        sc_matrix = schur_complement.matrix
        sc_colptr = sc_matrix.colptr
        sc_rowval = sc_matrix.rowval
        sc_nzval = sc_matrix.nzval
        sc_column_range_partial = schur_complement.column_range_partial
        sc_indices = schur_complement.indices

        nrow = length(sc_indices)
        last_full_row = sc_indices[end]
        for j ∈ sc_column_range_partial
            first_i = sc_colptr[j]
            last_i = sc_colptr[j+1] - 1
            if last_i < first_i
                continue
            end
            # Assume D and schur_complement have a similar pattern of non-zeros, so no
            # significant gain from using searchsortedlast() to find the first flat_i that
            # will be within the non-zeros of D.
            flat_i = first_i

            full_j = sc_indices[j]
            full_first_i = full_A_colptr[full_j]
            full_last_i = full_A_colptr[full_j+1]-1
            if full_last_i < full_first_i
                continue
            end

            first_row = sc_rowval[first_i]
            full_flat_i = max(searchsortedlast(@view(full_A_rowval[full_first_i:full_last_i]), first_row) - 1, 1) + full_first_i - 1
            while flat_i ≤ last_i && full_flat_i ≤ full_last_i
                row = sc_indices[sc_rowval[flat_i]]
                full_row = full_A_rowval[full_flat_i]
                if row == full_row
                    sc_nzval[flat_i] += full_A_nzval[full_flat_i]
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
