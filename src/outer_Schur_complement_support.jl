using BlockArrays
using LinearAlgebra.LAPACK: getrs!

function get_upper_right_block_shared_sparse_matrix_buffer(
        row_dimensions::Vector{<:Dimension}, col_dimensions::Vector{<:Dimension},
        shared_comm::MPI.Comm, allocate_shared_float::F1, allocate_shared_int::F2;
        stencil="point", ind_type::Type=Int64,
    ) where {F1, F2}

    shared_comm_rank = MPI.Comm_rank(shared_comm)
    m = prod(d.n_local for d ∈ row_dimensions; init=1)
    n = prod(d.n_local for d ∈ col_dimensions; init=1)

    n_colptr = Ref(-1)
    n_rowval = Ref(-1)
    if shared_comm_rank == 0
        common_dimensions = typejoin(eltype(row_dimensions), eltype(col_dimensions))[]
        first_different_dim = min(length(row_dimensions), length(col_dimensions)) + 1
        for (i, (dr, dc)) ∈ enumerate(zip(row_dimensions, col_dimensions))
            if dr.name == dc.name
                push!(common_dimensions, dc)
            else
                first_different_dim = i
                break
            end
        end
        extra_row_dimensions = row_dimensions[first_different_dim:end]
        extra_col_dimensions = col_dimensions[first_different_dim:end]
        extra_m = prod(d.n_local for d ∈ extra_row_dimensions; init=1)
        extra_n = prod(d.n_local for d ∈ extra_col_dimensions; init=1)

        cp = ind_type[1]
        rv = ind_type[]
        if stencil == "empty"
            # This block is empty, so nothing to do.
            for col ∈ 1:n
                push!(cp, 1)
            end
        elseif stencil == "point"
            # There are entries on the 'diagonal' positions where both row and column have
            # the same index in the common dimensions. This includes entries for every
            # point in 'extra' dimensions (in either row or column dimensions).
            for col ∈ 1:n
                common_i = (col - 1) ÷ extra_n + 1
                offset = (common_i - 1) * extra_m
                for row_count ∈ 1:extra_m
                    push!(rv, offset + row_count)
                end

                push!(cp, length(rv) + 1)
            end
        elseif stencil == "element"
            # There are entries in positions where the row and column indices
            # are in the same element in the common dimensions (this includes the elements
            # on either side of the boundary if the index is an element boundary). There
            # are entries for every point in 'extra' dimensions (in either row or column
            # dimensions).
            n_other_dims = [prod(d.n_local for d ∈ 1:idim-1; init=1)
                            for idim ∈ 1:length(common_dimensions)]
            function add_rows!(idim, common_i, row_offset)
                if idim == 0
                    row_offset *= extra_m
                    for row_count ∈ 1:extra_m
                        push!(rv, row_offset + row_count)
                    end
                else
                    d = common_dimensions[idim]
                    row_offset *= d.n_local
                    dim_i, common_i = divrem(common_i - 1, n_other_dims[idim]) .+ 1
                    dim_ielement, dim_igrid = divrem(dim_i - 1, (d.ngrid - 1)) .+ 1
                    if dim_ielement == d.nelement_local + 1
                        # Last local point in this dimension.
                        dim_range = (dim_ielement-2)*(d.ngrid-1)+1:(dim_ielement-1)*(d.ngrid-1)+1
                    elseif dim_ielement > 1 && dim_igrid == 1
                        # Element boundary point, so include elements on either side.
                        dim_range = (dim_ielement-2)*(d.ngrid-1)+1:dim_ielement*(d.ngrid-1)+1
                    else
                        # Include all points in this element.
                        dim_range = (dim_ielement-1)*(d.ngrid-1)+1:dim_ielement*(d.ngrid-1)+1
                    end
                    for i ∈ dim_range
                        add_rows!(idim - 1, common_i, row_offset + i - 1)
                    end
                end
                return nothing
            end
            for col ∈ 1:n
                add_rows!(length(common_dimensions), col, 0)
                push!(cp, length(rv) + 1)
            end
        else
            error("Unsupported stencil=$stencil. Expected \"empty\", \"point\" or "
                  * "\"element\".")
        end

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

    return FixedSparseCSC(m, n, colptr, rowval, nzval)
end
function get_lower_left_block_shared_sparse_matrix_buffer(
        row_dimensions::Vector{<:Dimension}, col_dimensions::Vector{<:Dimension},
        shared_comm::MPI.Comm, allocate_shared_float::F1, allocate_shared_int::F2;
        stencil="point", ind_type::Type=Int64,
    ) where {F1, F2}

    # Use transpose to turn FixedSparseCSC into a compressed-sparse-row matrix (note in
    # the following call we have transposed the row and col dimensions so the returned
    # buffer has the 'correct' un-transposed row and column structure).
    return transpose(get_upper_right_block_shared_sparse_matrix_buffer(
                         col_dimensions, row_dimensions, shared_comm,
                         allocate_shared_float, allocate_shared_int; stencil, ind_type))
end

struct OuterBSubmatrix{Tf,Ti,Tp,Tnext,Tnpib,TAAiBbl,Tmpc} <: MPISchurComplementBlockB
    parts_list::Vector{Tp}
    next_level_B::Tnext
    next_part_intermediate_buffers_list::Tnpib
    next_block_buffer::Vector{Tf}
    AAinv_dot_B_blocks_list::TAAiBbl
    block_columns::Vector{Vector{Vector{Ti}}}
    next_partial_intermediate_ranges::Vector{UnitRange{Ti}}
    bottom_copy_partial_col_ranges::Vector{UnitRange{Ti}}
    matmul_partial_copy::Tmpc
    matmul_partial_row_range::UnitRange{Ti}

    function OuterBSubmatrix(A_factorization, row_dimensions::Vector{<:Dimension},
                             column_dimensions::Vector{<:Dimension},
                             shared_comm::MPI.Comm, allocate_shared_float::F1,
                             allocate_shared_int::F2,
                             block_matrix::Union{AbstractMatrix,Vector{<:AbstractMatrix},NTuple},
                             is_top_level::Bool=true; ind_type::Type=Int64) where {F1,F2}

        if isa(block_matrix, Vector{<:AbstractMatrix})
            parts_list = block_matrix
        elseif isa(block_matrix, Tuple)
            parts_list = [block_matrix...]
        elseif isa(block_matrix, BlockMatrix)
            matrix_blocks = blocks(block_matrix)
            block_nrows, block_ncols = size(matrix_blocks)
            if block_nrows > 1
                error("OuterBSubmatrix expects a BlockMatrix with only one block row.")
            end
            parts_list = reshape(matrix_blocks, block_ncols)
        else
            parts_list = [block_matrix]
        end

        Tf = eltype(parts_list[1])
        Ti = ind_type

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)

        # Don't use parts_list to get the number of rows, because a part might have zero
        # rows if it is empty.
        m = prod(d.n_local for d ∈ row_dimensions; init=1)
        n = sum(size(p, 2) for p ∈ parts_list)

        if isa(A_factorization, MPIStaticCondensationParallel)
            bottom_vector_indices = A_factorization.bottom_vector_indices
            next_m = length(bottom_vector_indices)
            next_A_factorization = A_factorization.local_block_solver.schur_complement_factorization
            n_hypercube_positions = 2^length(row_dimensions)
            if isa(next_A_factorization, MPIStaticCondensationParallel)
                next_parts_list =
                    [get_sparse_next_upper_right_block(p, row_dimensions,
                                                       column_dimensions,
                                                       next_A_factorization.block_sizes,
                                                       bottom_vector_indices, shared_comm,
                                                       allocate_shared_float,
                                                       allocate_shared_int, Ti)
                     for p ∈ parts_list]

                next_part_intermediate_buffers_list =
                    [allocate_shared_float(n_hypercube_positions, nnz(np))
                     for np ∈ next_parts_list]
            else
                next_parts_list =
                    [allocate_shared_float(isempty(p) ? 0 : next_m, size(p, 2))
                     for p ∈ parts_list]

                next_part_intermediate_buffers_list =
                    [allocate_shared_float(n_hypercube_positions, size(np)...)
                     for np ∈ next_parts_list]
            end
            if shared_comm_rank == 0
                for inter ∈ next_part_intermediate_buffers_list
                    inter .= 0.0
                end
            end

            next_level_B = OuterBSubmatrix(next_A_factorization, row_dimensions,
                                           column_dimensions, shared_comm,
                                           allocate_shared_float, allocate_shared_int,
                                           next_parts_list, false; ind_type=Ti)

            # Could share memory between levels and blocks?
            AA_block_diagonal_solver = A_factorization.local_block_solver.A_factorization
            block_indices = A_factorization.local_top_vector_a_block_indices
            if isa(A_factorization.local_block_solver.C, BlockCSerial)
                block_buffer_size = maximum(length(inds) for inds ∈
                                            A_factorization.local_block_solver.C.block_rowinds)
            elseif isa(A_factorization.local_block_solver.C, BlockCShared)
                block_buffer_size = length(A_factorization.local_block_solver.C.block_rowinds)
            else
                block_buffer_size = next_m
            end
            next_block_buffer = fill(Tf(NaN), block_buffer_size)

            block_columns = [[[j for j ∈ 1:size(p, 2) if sparse_column_has_overlap(bi, p, j)]
                              for bi ∈ block_indices]
                             for p ∈ parts_list]

            # Use single array for storage of AAinv_dot_B_blocks_list, divided into views,
            # because allocating individual shared-memory arrays for each entry would use
            # up too many MPI communicators (one MPI communicator is associated with each
            # MPI 'Window' that is required for a shared-memory array, and the maximum
            # number of MPI communicators is a small number, e.g. 2048).
            if isempty(block_indices)
                total_AAinv_dot_B_blocks_buffer_size = 0
            else
                total_AAinv_dot_B_blocks_buffer_size =
                    sum(length(bi) for (ip, p) ∈ enumerate(parts_list)
                        for (ib, bi) ∈ enumerate(block_indices) for _ ∈ block_columns[ip][ib])
            end
            AAinv_dot_B_blocks_storage = allocate_shared_float(total_AAinv_dot_B_blocks_buffer_size)
            AAinv_dot_B_blocks_list = Vector{Vector{SubArray{Tf, 1, Vector{Tf}, Tuple{UnitRange{Ti}}, true}}}[]
            offset = 0
            for (ip, p) ∈ enumerate(parts_list)
                AAinv_dot_B_blocks_part = Vector{SubArray{Tf, 1, Vector{Tf}, Tuple{UnitRange{Ti}}, true}}[]
                push!(AAinv_dot_B_blocks_list, AAinv_dot_B_blocks_part)
                for (ib, bi) ∈ enumerate(block_indices)
                    AAinv_dot_B_block = SubArray{Tf, 1, Vector{Tf}, Tuple{UnitRange{Ti}}, true}[]
                    push!(AAinv_dot_B_blocks_part, AAinv_dot_B_block)
                    for _ ∈ block_columns[ip][ib]
                        this_range = Ti(offset) .+ (Ti(1):Ti(length(bi)))
                        push!(AAinv_dot_B_block,
                              @view(AAinv_dot_B_blocks_storage[this_range]))
                        offset += length(bi)
                    end
                end
            end

            bottom_copy_partial_col_ranges = UnitRange{Ti}[]
            for p ∈ parts_list
                pn = size(p, 2)
                cols_per_proc = (pn + shared_comm_size - 1) ÷ shared_comm_size
                r = Ti(shared_comm_rank*cols_per_proc+1):Ti(min((shared_comm_rank+1)*cols_per_proc,pn))
                push!(bottom_copy_partial_col_ranges, r)
            end

            # Get flat indices of in sparse buffer of columns corresponding to
            # bottom_copy_partial_col_ranges.
            if isa(next_A_factorization, MPIStaticCondensationParallel)
                next_partial_intermediate_ranges = UnitRange{Ti}[]
                for (r, np) ∈ zip(bottom_copy_partial_col_ranges, next_parts_list)
                    if isempty(r)
                        push!(next_partial_intermediate_ranges, 1:0)
                    else
                        colptr = np.colptr
                        first_i = colptr[r[1]]
                        last_i = colptr[r[end]+1]-1
                        push!(next_partial_intermediate_ranges, first_i:last_i)
                    end
                end
            else
                next_partial_intermediate_ranges = bottom_copy_partial_col_ranges
            end
        else
            next_level_B = nothing
            next_part_intermediate_buffers_list = nothing
            next_block_buffer = Tf[]
            block_columns = Vector{Vector{Ti}}[]
            AAinv_dot_B_blocks_list = nothing

            # Re-use these fields to store needed ranges for the dense-matrix bottom
            # level.
            next_partial_intermediate_ranges = UnitRange{Ti}[]
            bottom_copy_partial_col_ranges = UnitRange{Ti}[]
            col_offset = 0
            for p ∈ parts_list
                pn = size(p, 2)
                cols_per_proc = (pn + shared_comm_size - 1) ÷ shared_comm_size
                r = Ti(shared_comm_rank*cols_per_proc+1):Ti(min((shared_comm_rank+1)*cols_per_proc,pn))
                push!(bottom_copy_partial_col_ranges, r)
                push!(next_partial_intermediate_ranges, r .+ col_offset)
                col_offset += pn
            end
        end

        if is_top_level
            rows_per_proc = (m + shared_comm_size - 1) ÷ shared_comm_size
            first_row = Ti(shared_comm_rank*rows_per_proc+1)
            last_row = Ti(min((shared_comm_rank+1)*rows_per_proc,m))
            matmul_partial_row_range = first_row:last_row

            if eltype(parts_list) <: AbstractSparseMatrix
                # `matmul_partial_copy` is a single FixedSparseCSC that contains rows in
                # `matmul_partial_row_range` from all 'parts'.
                cp = Ti[1]
                rv = Ti[]
                for p ∈ parts_list
                    p_colptr = p.colptr
                    p_rowval = p.rowval
                    for pcol ∈ 1:size(p, 2)
                        first_i = p_colptr[pcol]
                        last_i = p_colptr[pcol+1] - 1
                        flat_i = max(searchsortedlast(@view(p_rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
                        for row ∈ matmul_partial_row_range
                            while flat_i ≤ last_i && p_rowval[flat_i] < row
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                            if p_rowval[flat_i] == row
                                push!(rv, row - first_row + 1)
                                flat_i += 1
                            end
                        end
                        push!(cp, length(rv) + 1)
                    end
                end
                nzval = zeros(Tf, length(rv))
                matmul_partial_copy = FixedSparseCSC(length(matmul_partial_row_range), n, cp, rv, nzval)
            else
                matmul_partial_copy = zeros(Tf, length(matmul_partial_row_range), n)
            end
        else
            matmul_partial_copy = nothing
            matmul_partial_row_range = Ti(1):Ti(0)
        end

        return new{Tf,Ti,eltype(parts_list),typeof(next_level_B),typeof(next_part_intermediate_buffers_list),typeof(AAinv_dot_B_blocks_list),typeof(matmul_partial_copy)}(
                   parts_list, next_level_B, next_part_intermediate_buffers_list,
                   next_block_buffer, AAinv_dot_B_blocks_list, block_columns,
                   next_partial_intermediate_ranges, bottom_copy_partial_col_ranges,
                   matmul_partial_copy, matmul_partial_row_range)
    end
end

struct OuterCSubmatrix{Tf,Ti,Tp,Tnext,Tnpib,Tmpc} <: MPISchurComplementBlockC
    output_buffer_ncopies::Ti
    output_buffer_positions::Vector{Ti}
    output_buffer_zero_init_range::UnitRange{Ti}
    parts_list::Vector{Tp}
    next_level_C::Tnext
    next_part_intermediate_buffers_list::Tnpib
    block_buffer::Vector{Tf}
    next_block_buffer::Vector{Tf}
    block_rows::Vector{Vector{Vector{Ti}}}
    next_partial_intermediate_ranges::Vector{UnitRange{Ti}}
    bottom_copy_partial_row_ranges::Vector{UnitRange{Ti}}
    matmul_partial_copy::Tmpc
    matmul_partial_row_range::UnitRange{Ti}

    function OuterCSubmatrix(A_factorization, row_dimensions::Vector{<:Dimension},
                             column_dimensions::Vector{<:Dimension},
                             shared_comm::MPI.Comm, allocate_shared_float::F1,
                             allocate_shared_int::F2,
                             block_matrix::Union{AbstractMatrix,Vector{<:AbstractMatrix},NTuple},
                             is_top_level::Bool=true; ind_type::Type=Int64) where {F1,F2}

        if isa(block_matrix, Vector{<:AbstractMatrix})
            parts_list = block_matrix
        elseif isa(block_matrix, Tuple)
            parts_list = [block_matrix...]
        elseif isa(block_matrix, BlockMatrix)
            matrix_blocks = blocks(block_matrix)
            block_nrows, block_ncols = size(matrix_blocks)
            if block_ncols > 1
                error("OuterCSubmatrix expects a BlockMatrix with only one block column.")
            end
            parts_list = reshape(matrix_blocks, block_nrows)
        else
            parts_list = [block_matrix]
        end

        Tf = eltype(parts_list[1])
        Ti = ind_type

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        shared_comm_size = MPI.Comm_size(shared_comm)

        m = sum(size(p, 1) for p ∈ parts_list)
        # Don't use parts_list to get the number of columns, because a part might have
        # zero columns if it is empty.
        n = prod(d.n_local for d ∈ column_dimensions; init=1)

        if isa(A_factorization, MPIStaticCondensationParallel)
            bottom_vector_indices = A_factorization.bottom_vector_indices
            next_n = length(bottom_vector_indices)
            next_A_factorization = A_factorization.local_block_solver.schur_complement_factorization
            n_hypercube_positions = 2^length(column_dimensions)
            if isa(next_A_factorization, MPIStaticCondensationParallel)
                next_parts_list =
                    [get_sparse_next_lower_left_block(p, row_dimensions,
                                                      column_dimensions,
                                                      next_A_factorization.block_sizes,
                                                      bottom_vector_indices, shared_comm,
                                                      allocate_shared_float,
                                                      allocate_shared_int, Ti)
                     for p ∈ parts_list]

                next_part_intermediate_buffers_list =
                    [allocate_shared_float(n_hypercube_positions, nnz(np))
                     for np ∈ next_parts_list]
            else
                next_parts_list =
                    [allocate_shared_float(size(p, 1), isempty(p) ? 0 : next_n)
                     for p ∈ parts_list]

                next_part_intermediate_buffers_list =
                    [allocate_shared_float(n_hypercube_positions, size(np)...)
                     for np ∈ next_parts_list]
            end
            if shared_comm_rank == 0
                for inter ∈ next_part_intermediate_buffers_list
                    inter .= 0.0
                end
            end

            next_level_C = OuterCSubmatrix(next_A_factorization, row_dimensions,
                                           column_dimensions, shared_comm,
                                           allocate_shared_float, allocate_shared_int,
                                           next_parts_list, false; ind_type=Ti)

            # Could share memory between levels and blocks?
            AA_block_diagonal_solver = A_factorization.local_block_solver.A_factorization
            block_indices = A_factorization.local_top_vector_a_block_indices
            if isa(A_factorization.local_block_solver.Ainv_dot_B, BlockAinvDotBSerial)
                next_block_buffer_size = maximum(length(inds) for inds ∈
                                                 A_factorization.local_block_solver.Ainv_dot_B.block_colinds)
            elseif isa(A_factorization.local_block_solver.Ainv_dot_B, BlockAinvDotBShared)
                next_block_buffer_size = length(A_factorization.local_block_solver.Ainv_dot_B.block_colinds)
            else
                next_block_buffer_size = next_n
            end
            next_block_buffer = fill(Tf(NaN), next_block_buffer_size)

            block_rows = [[[i for i ∈ 1:size(p, 1) if sparse_row_has_overlap(bi, p, i)]
                           for bi ∈ block_indices]
                          for p ∈ parts_list]

            block_buffer_size = maximum(length(bi) for bi ∈ block_indices)
            block_buffer = allocate_shared_float(block_buffer_size)

            bottom_copy_partial_row_ranges = UnitRange{Ti}[]
            for p ∈ parts_list
                pm = size(p, 1)
                rows_per_proc = (pm + shared_comm_size - 1) ÷ shared_comm_size
                r = Ti(shared_comm_rank*rows_per_proc+1):Ti(min((shared_comm_rank+1)*rows_per_proc,pm))
                push!(bottom_copy_partial_row_ranges, r)
            end

            # Get flat indices of in sparse buffer of columns corresponding to
            # bottom_copy_partial_row_ranges.
            if isa(next_A_factorization, MPIStaticCondensationParallel)
                next_partial_intermediate_ranges = UnitRange{Ti}[]
                for (r, np) ∈ zip(bottom_copy_partial_row_ranges, next_parts_list)
                    if isempty(r)
                        push!(next_partial_intermediate_ranges, 1:0)
                    else
                        rowptr = transpose(np).colptr
                        first_i = rowptr[r[1]]
                        last_i = rowptr[r[end]+1]-1
                        push!(next_partial_intermediate_ranges, first_i:last_i)
                    end
                end
            else
                next_partial_intermediate_ranges = bottom_copy_partial_row_ranges
            end

            # This is probably not the optimal choice, but it should be robust so use for a
            # first test before we try to reduce output_buffer_ncopies to the minimum.
            output_buffer_ncopies = shared_comm_size
            output_buffer_positions = [Ti(shared_comm_rank) for _ ∈ 1:length(block_indices)]
        else
            next_level_C = nothing
            next_part_intermediate_buffers_list = nothing
            block_buffer = Tf[]
            next_block_buffer = Tf[]
            block_rows = Vector{Vector{Ti}}[]
            next_partial_intermediate_ranges = UnitRange{Ti}[]

            # Re-use this field to store needed ranges for the dense-matrix bottom level.
            bottom_copy_partial_row_ranges = UnitRange{Ti}[]
            row_offset = 0
            for p ∈ parts_list
                pm = size(p, 1)
                push!(bottom_copy_partial_row_ranges, row_offset+1:row_offset+pm)
                row_offset += pm
            end

            output_buffer_ncopies = 0
            output_buffer_positions = Ti[]
        end

        cols_per_proc = (n + shared_comm_size - 1) ÷ shared_comm_size
        output_buffer_zero_init_range = Ti(shared_comm_rank*cols_per_proc+1):Ti(min((shared_comm_rank+1)*cols_per_proc,n))

        if is_top_level
            rows_per_proc = (m + shared_comm_size - 1) ÷ shared_comm_size
            first_row = Ti(shared_comm_rank*rows_per_proc+1)
            last_row = Ti(min((shared_comm_rank+1)*rows_per_proc,m))
            matmul_partial_row_range = first_row:last_row

            if eltype(parts_list) <: AbstractSparseMatrix
                # `matmul_partial_copy` is a single Transpose{Tf,FixedSparseCSC{Tf,Ti}} that
                # contains rows in `matmul_partial_row_range` from all 'parts'. As
                # `matmul_partial_copy` and `parts_list` are transposed array buffers, this
                # means we need to collect columns in `matmul_partial_row_range` from the
                # transposes.
                cp = Ti[1]
                rv = Ti[]
                row_offset = 0
                for p ∈ parts_list
                    p_rows = max(first_row - row_offset, 1):min(last_row - row_offset, size(p, 1))
                    p_colptr = transpose(p).colptr
                    p_rowval = transpose(p).rowval
                    for pcol ∈ p_rows
                        first_i = p_colptr[pcol]
                        last_i = p_colptr[pcol+1] - 1
                        for flat_i ∈ first_i:last_i
                            push!(rv, p_rowval[flat_i])
                        end
                        push!(cp, length(rv) + 1)
                    end
                    row_offset += size(p, 1)
                end
                nzval = zeros(Tf, length(rv))
                matmul_partial_copy = transpose(FixedSparseCSC(m, length(matmul_partial_row_range), cp, rv, nzval))
            else
                matmul_partial_copy = transpose(zeros(Tf, n, length(matmul_partial_row_range)))
            end
        else
            matmul_partial_copy = nothing
            matmul_partial_row_range = Ti(1):Ti(0)
        end

        return new{Tf,Ti,eltype(parts_list),typeof(next_level_C),typeof(next_part_intermediate_buffers_list),typeof(matmul_partial_copy)}(
                   output_buffer_ncopies, output_buffer_positions,
                   output_buffer_zero_init_range, parts_list, next_level_C,
                   next_part_intermediate_buffers_list, block_buffer, next_block_buffer,
                   block_rows, next_partial_intermediate_ranges,
                   bottom_copy_partial_row_ranges, matmul_partial_copy,
                   matmul_partial_row_range)
    end
end

function get_sparse_next_upper_right_block(upper_right_block::FixedSparseCSC,
                                           row_dimensions::Vector{<:Dimension},
                                           column_dimensions::Vector{<:Dimension},
                                           block_sizes::Vector{<:Integer},
                                           local_bottom_vector_indices::Vector{Int64},
                                           shared_comm::MPI.Comm,
                                           allocate_shared_float::F1,
                                           allocate_shared_int::F2,
                                           ind_type::Type) where {F1,F2}
    m = length(local_bottom_vector_indices)
    n = size(upper_right_block, 2)

    if length(upper_right_block.nzval) == 0
        # Empty block.
        Tf = eltype(upper_right_block)
        return FixedSparseCSC(spzeros(Tf, 0, n))
    end

    return get_shared_sparse_matrix_csc_buffer(row_dimensions, shared_comm,
                                               allocate_shared_float, allocate_shared_int;
                                               column_dimensions=column_dimensions,
                                               block_sizes=block_sizes[length(block_sizes)-length(column_dimensions)+1:end],
                                               row_indices=local_bottom_vector_indices,
                                               ind_type=ind_type)
end
@inline function get_sparse_next_lower_left_block(lower_left_block::Transpose{Tf, FixedSparseCSC{Tf,Ti}},
                                                  row_dimensions::Vector{<:Dimension},
                                                  column_dimensions::Vector{<:Dimension},
                                                  block_sizes::Vector{<:Integer},
                                                  local_bottom_vector_indices::Vector{Int64},
                                                  shared_comm::MPI.Comm,
                                                  allocate_shared_float::F1,
                                                  allocate_shared_int::F2,
                                                  ind_type::Type) where {Tf,Ti,F1,F2}
    return transpose(get_sparse_next_upper_right_block(transpose(lower_left_block),
                                                       column_dimensions, row_dimensions,
                                                       block_sizes,
                                                       local_bottom_vector_indices,
                                                       shared_comm, allocate_shared_float,
                                                       allocate_shared_int, ind_type))
end

function sparse_column_has_overlap(bi::Vector{<:Integer}, p::FixedSparseCSC, j::Integer)
    if nnz(p) == 0 || length(bi) == 0
        return false
    end
    colptr = p.colptr
    rowval = p.rowval
    first_i = colptr[j]
    last_i = colptr[j+1] - 1
    column_rowvals = @view rowval[first_i:last_i]
    last_row = bi[end]
    return any(searchsortedfirst(bi, x) ≤ last_row for x ∈ column_rowvals)
end
function sparse_column_has_overlap(bi::Vector{<:Integer}, p::AbstractMatrix, j::Integer)
    return true
end

function sparse_row_has_overlap(bi::Vector{<:Integer}, p::Transpose{Tf,Tp},
                                i::Integer) where {Tf,Tp}
    return sparse_column_has_overlap(bi, transpose(p), i)
end
function sparse_row_has_overlap(bi::Vector{<:Integer}, p::AbstractMatrix, i::Integer)
    return true
end

"""
    get_column_block_from_sparse_matrix!(
        A::Vector{Tf}, new_A::AbstractSparseMatrixCSC{Tf,Ti}, rowinds,
        colind) where {Tf,Ti}

Update the values of `A` in-place to the values from `new_A`.

`colind` gives the columns in `new_A` that should be copied into `A`.

`rowinds` gives the subset of rows in `new_A` that should be copied into `A`.
"""
function get_column_block_from_sparse_matrix!(
             A::AbstractVector{Tf}, new_A::AbstractSparseMatrixCSC{Tf,Ti}, rowinds,
             colind) where {Tf,Ti}
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    A_nrows = length(rowinds)

    new_firsti = new_colptr[colind]
    new_lasti = new_colptr[colind+1] - 1
    new_firstrow = new_rowval[new_firsti]
    # Expect than usually the sparsity patterns of A and new_A will match, so the
    # rowval entries for this column will be the same in both. Therefore no need to
    # use `searchsortedlast()` to speed up finding the first matching entry for `i`.
    A_row = max(searchsortedlast(rowinds, new_firstrow) - 1, 1)
    A[1:A_row-1] .= 0.0
    for new_i ∈ new_firsti:new_lasti
        new_row = new_rowval[new_i]
        while A_row ≤ A_nrows && rowinds[A_row] < new_row
            A[A_row] = 0.0
            A_row += 1
        end
        if A_row > A_nrows
            break
        end
        if rowinds[A_row] == new_row
            A[A_row] = new_nzval[new_i]
            A_row += 1
        end
    end
    A[A_row:end] .= 0.0

    return nothing
end
@inline function get_row_block_from_sparse_matrix!(
                     A::AbstractVector{Tf}, new_A::Transpose{Tf, FixedSparseCSC{Tf,Ti}},
                     colinds, rowind) where {Tf,Ti}
    return get_column_block_from_sparse_matrix!(A, new_A.parent, colinds, rowind)
end

function outer_CAiB_fill_intermediate_B_buffer!(buffer::FixedSparseCSC{Tf},
                                                intermediate::AbstractMatrix{Tf},
                                                hypercube_position::Integer,
                                                block::AbstractVector{Tf}, col,
                                                rowinds) where Tf
    colptr = buffer.colptr
    rowval = buffer.rowval
    first_i = colptr[col]
    last_i = colptr[col+1]-1
    rv = @view rowval[first_i:last_i]
    flat_i = max(searchsortedlast(rv, rowinds[1]) - 1, 1) + first_i - 1
    block_i = 1
    block_n = length(rowinds)
    while flat_i ≤ last_i && block_i ≤ block_n
        row = rowval[flat_i]
        block_row = rowinds[block_i]
        if row == block_row
            intermediate[hypercube_position,flat_i] = block[block_i]
            flat_i += 1
            block_i += 1
        elseif row < block_row
            flat_i += 1
        else # block_row < row
            block_i += 1
        end
    end
    return nothing
end
function outer_CAiB_fill_intermediate_B_buffer!(buffer::AbstractMatrix{Tf},
                                                intermediate::AbstractArray{Tf,3},
                                                hypercube_position::Integer,
                                                block::AbstractVector{Tf}, col,
                                                rowinds) where Tf
    for (i, row) ∈ enumerate(rowinds)
        intermediate[hypercube_position,row,col] = block[i]
    end
    return nothing
end
@inline function outer_CAiB_fill_intermediate_C_buffer!(buffer::Transpose{Tf,FixedSparseCSC{Tf,Ti}},
                                                        intermediate::AbstractMatrix{Tf},
                                                        hypercube_position::Integer,
                                                        block::AbstractVector{Tf}, row,
                                                        colinds) where {Tf,Ti}
    return outer_CAiB_fill_intermediate_B_buffer!(transpose(buffer), intermediate,
                                                  hypercube_position, block, row, colinds)
end
function outer_CAiB_fill_intermediate_C_buffer!(buffer::AbstractMatrix{Tf},
                                                intermediate::AbstractArray{Tf,3},
                                                hypercube_position::Integer,
                                                block::AbstractVector{Tf}, row,
                                                colinds) where Tf
    for (i, col) ∈ enumerate(colinds)
        intermediate[hypercube_position,row,col] = block[i]
    end
    return nothing
end

function outer_CAiB_sum_B_intermediates!(next_buffer::FixedSparseCSC{Tf},
                                         intermediate::AbstractMatrix{Tf},
                                         next_partial_range, buffer::FixedSparseCSC{Tf},
                                         partial_col_range, row_range) where Tf
    next_nzval = next_buffer.nzval
    @views sum!(next_nzval[next_partial_range]', intermediate[:,next_partial_range])

    next_colptr = next_buffer.colptr
    next_rowval = next_buffer.rowval
    colptr = buffer.colptr
    rowval = buffer.rowval
    nzval = buffer.nzval
    first_row = row_range[1]
    for col ∈ partial_col_range
        next_flat_i = next_colptr[col]
        next_last_i = next_colptr[col+1]-1
        first_i = colptr[col]
        last_i = colptr[col+1]-1
        flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
        if next_flat_i > next_last_i || flat_i > last_i
            continue
        end
        for (next_row, row) ∈ enumerate(row_range)
            while next_flat_i < next_last_i && next_rowval[next_flat_i] < next_row
                next_flat_i += 1
            end
            while flat_i < last_i && rowval[flat_i] < row
                flat_i += 1
            end
            if next_rowval[next_flat_i] == next_row && rowval[flat_i] == row
                next_nzval[next_flat_i] += nzval[flat_i]
                next_flat_i += 1
                flat_i += 1
            end
            if next_flat_i > next_last_i || flat_i > last_i
                break
            end
        end
    end
    return nothing
end
@inline function outer_CAiB_sum_C_intermediates!(next_buffer::Transpose{Tf,FixedSparseCSC{Tf,Ti}},
                                                 intermediate::AbstractMatrix{Tf},
                                                 next_partial_range,
                                                 buffer::Transpose{Tf,FixedSparseCSC{Tf,Ti}},
                                                 partial_row_range, col_range) where {Tf,Ti}
    return outer_CAiB_sum_B_intermediates!(transpose(next_buffer), intermediate,
                                           next_partial_range, transpose(buffer),
                                           partial_row_range, col_range)
    return nothing
end
function outer_CAiB_sum_B_intermediates!(next_buffer::AbstractMatrix{Tf},
                                         intermediate::AbstractArray{Tf,3},
                                         next_partial_range, buffer::FixedSparseCSC{Tf},
                                         partial_col_range, row_range) where Tf
    if isempty(row_range)
        return nothing
    end
    @views sum!(reshape(next_buffer[:,next_partial_range], 1, size(next_buffer, 1),
                        length(next_partial_range)),
                intermediate[:,:,next_partial_range])
    colptr = buffer.colptr
    rowval = buffer.rowval
    nzval = buffer.nzval
    first_row = row_range[1]
    last_row = row_range[end]
    for col ∈ partial_col_range
        first_i = colptr[col]
        last_i = colptr[col+1]-1
        flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
        if flat_i > last_i
            continue
        end
        for (next_row, row) ∈ enumerate(row_range)
            while rowval[flat_i] < row && flat_i < last_i
                flat_i += 1
            end
            if rowval[flat_i] == row
                next_buffer[next_row,col] += nzval[flat_i]
                flat_i += 1
            end
            if flat_i > last_i || rowval[flat_i] > last_row
                break
            end
        end
    end
    return nothing
end
function outer_CAiB_sum_C_intermediates!(next_buffer::AbstractMatrix{Tf},
                                         intermediate::AbstractArray{Tf,3},
                                         next_partial_range,
                                         buffer::Transpose{Tf,FixedSparseCSC{Tf,Ti}},
                                         next_partial_row_range, colinds) where {Tf,Ti}
    if isempty(colinds)
        return nothing
    end
    @views sum!(reshape(next_buffer[next_partial_row_range,:], 1,
                        length(next_partial_row_range), size(next_buffer, 2)),
                intermediate[:,next_partial_row_range,:])
    transpose_buffer = transpose(buffer)
    colptr = transpose_buffer.colptr
    rowval = transpose_buffer.rowval
    nzval = transpose_buffer.nzval
    first_col = colinds[1]
    last_col = colinds[end]
    for row ∈ next_partial_row_range
        first_i = colptr[row]
        last_i = colptr[row+1]-1
        flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_col), 1) + first_i - 1
        if flat_i > last_i
            continue
        end
        for (next_col, col) ∈ enumerate(colinds)
            while rowval[flat_i] < col && flat_i < last_i
                flat_i += 1
            end
            if rowval[flat_i] == col
                next_buffer[row,next_col] += nzval[flat_i]
                flat_i += 1
            end
            if flat_i > last_i || rowval[flat_i] > last_col
                break
            end
        end
    end
    return nothing
end

function mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractArray{T,N}, C::OuterCSubmatrix,
                           A_factorization::MPIStaticCondensationParallel,
                           B::OuterBSubmatrix) where {T,N}
    # Need to zero the C_dot_Ainv_dot_B buffer first.
    synchronize_shared = A_factorization.synchronize_shared
    output_buffer_zero_init_range = C.output_buffer_zero_init_range

    if N == 2
        # Using schur_complement directly as the output buffer, and that was already
        # zero'ed.
    elseif N == 3
        C_dot_Ainv_dot_B[:,:,output_buffer_zero_init_range] .= 0.0
        synchronize_shared()
    else
        error("Unexpected number of dimensions ($(N)) for C_dot_Ainv_dot_B.")
    end

    return _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B, C, A_factorization, B,
                                       synchronize_shared)
end

function _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractArray{Tf,N},
                                     C::OuterCSubmatrix,
                                     A_factorization::MPIStaticCondensationParallel{Tf,Ti},
                                     B::OuterBSubmatrix,
                                     previous_synchronize_shared::F) where {Tf,N,Ti,F}
    B_parts = B.parts_list
    next_B = B.next_level_B
    next_B_parts = next_B.parts_list
    next_B_intermediate_buffers = B.next_part_intermediate_buffers_list
    next_B_block_buffer = B.next_block_buffer
    AAinv_dot_B_blocks = B.AAinv_dot_B_blocks_list
    B_block_columns = B.block_columns
    next_B_partial_intermediate_ranges = B.next_partial_intermediate_ranges
    B_bottom_copy_partial_col_ranges = B.bottom_copy_partial_col_ranges

    C_parts = C.parts_list
    next_C = C.next_level_C
    next_C_parts = next_C.parts_list
    next_C_intermediate_buffers = C.next_part_intermediate_buffers_list
    next_C_block_buffer = C.next_block_buffer
    C_block_buffer = C.block_buffer
    C_block_rows = C.block_rows
    next_C_partial_intermediate_ranges = C.next_partial_intermediate_ranges
    C_bottom_copy_partial_row_ranges = C.bottom_copy_partial_row_ranges
    output_buffer_positions = C.output_buffer_positions

    AA_solver = A_factorization.local_block_solver.A_factorization
    AC = A_factorization.local_block_solver.C
    AB = A_factorization.local_block_solver.Ainv_dot_B
    block_inds = A_factorization.local_top_vector_a_block_indices
    local_bottom_vector_indices = A_factorization.local_bottom_vector_indices
    synchronize_shared = A_factorization.synchronize_shared

    if isa(AA_solver, BlockDiagonalSolverSerial)
        # This process (potentially) owns several blocks.
        AA_solver_block_factorizations = AA_solver.local_block_solver
        if isa(AC, MPISchurComplementBlockC)
            AC_blocks = AC.blocks
            hypercube_positions = AC.block_hypercube_positions
            AC_block_rowinds = AC.block_rowinds
        else
            AC_blocks = (AC,)
            hypercube_positions = (Ti(1),)
            AC_block_rowinds = (1:length(local_bottom_vector_indices),)
        end
        col_offset = 0
        for (B_part, next_B_part, next_B_inter, AAinv_dot_B_part_blocks,
             B_part_block_columns) ∈
                zip(B_parts, next_B_parts, next_B_intermediate_buffers,
                    AAinv_dot_B_blocks, B_block_columns)
            if !isempty(B_part) && !(issparse(B_part) && nnz(B_part) == 0)
                for (bi, block_cols, AAiB_block_for_cols, AAfac, AC_block, AC_hp,
                     next_bi) ∈
                        zip(block_inds, B_part_block_columns, AAinv_dot_B_part_blocks,
                            AA_solver_block_factorizations, AC_blocks,
                            hypercube_positions, AC_block_rowinds)
                    for (overlap_i, part_col) ∈ enumerate(block_cols)
                        col = col_offset + part_col
                        AAiB_block_this_col = AAiB_block_for_cols[overlap_i]
                        get_column_block_from_sparse_matrix!(AAiB_block_this_col, B_part,
                                                             bi, part_col)
                        ldiv!(AAfac, AAiB_block_this_col)
                        this_buffer = @view next_B_block_buffer[1:size(AC_block, 1)]
                        mul!(this_buffer, AC_block, AAiB_block_this_col, -1.0, 0.0)
                        outer_CAiB_fill_intermediate_B_buffer!(next_B_part, next_B_inter,
                                                               AC_hp, this_buffer,
                                                               part_col, next_bi)
                    end

                end
            end
            col_offset += size(B_part, 2)
        end
    else
        # This process owns a single block, that is shared with other processes.
        block_rank = AA_solver.block_comm_rank
        bi = block_inds
        AA_block_factorization = AA_solver.local_block_serial_solver
        if isa(AC, MPISchurComplementBlockC)
            AC_block = AC.block
            hypercube_position = AC.block_hypercube_position
            AC_block_rowinds = AC.block_rowinds
        else
            AC_block = AC
            hypercube_position = Ti(1)
            AC_block_rowinds = 1:length(local_bottom_vector_indices)
        end
        col_offset = 0
        for (B_part, next_B_part, next_B_inter, AAinv_dot_B_part_blocks, B_part_block_columns) ∈
                zip(B_parts, next_B_parts, next_B_intermediate_buffers,
                    AAinv_dot_B_blocks, B_block_columns)
            if !isempty(B_part)
                AAiB_block_for_cols = AAinv_dot_B_part_blocks[1]
                for (overlap_i, part_col) ∈ enumerate(B_part_block_columns[1])
                    col = col_offset + part_col
                    AAiB_block_this_col = AAiB_block_for_cols[overlap_i]

                    get_column_block_from_sparse_matrix!(AAiB_block_this_col, B_part,
                                                         bi, part_col)
                    ldiv!(AA_block_factorization, AAiB_block_this_col)
                    mul!(next_B_block_buffer, AC_block, AAiB_block_this_col, -1.0, 0.0)
                    outer_CAiB_fill_intermediate_B_buffer!(next_B_part, next_B_inter,
                                                           hypercube_position,
                                                           next_B_block_buffer, part_col,
                                                           AC_block_rowinds)
                end
            end
            col_offset += size(B_part, 2)
        end
    end

    synchronize_shared()

    for (B_part, next_B_part, next_B_inter, next_partial_range, copy_partial_col_range) ∈
            zip(B_parts, next_B_parts, next_B_intermediate_buffers,
                next_B_partial_intermediate_ranges, B_bottom_copy_partial_col_ranges)

        if !isempty(B_part) && !(issparse(next_B_part) && nnz(next_B_part) == 0)
            outer_CAiB_sum_B_intermediates!(next_B_part, next_B_inter, next_partial_range,
                                            B_part, copy_partial_col_range,
                                            local_bottom_vector_indices)
        end
    end

    if isa(AA_solver, BlockDiagonalSolverSerial)
        # This process (potentially) owns several blocks.
        if isa(AB, Union{MPISchurComplementBlockAinvDotB,MPISchurComplementBlockB})
            AB_blocks = AB.blocks
            AB_block_colinds = AB.block_colinds
            Afac_separate_Ainv_B = isa(AB, MPISchurComplementBlockB)
        else
            AB_blocks = (AB,)
            AB_block_colinds = (1:length(local_bottom_vector_indices),)
            Afac_separate_Ainv_B = A_factorization.local_block_solver.separate_Ainv_B
        end
        row_offset = 0
        for (C_part, next_C_part, next_C_inter, C_part_block_rows) ∈
                zip(C_parts, next_C_parts, next_C_intermediate_buffers, C_block_rows)
            if !isempty(C_part) && !(issparse(C_part) && nnz(transpose(C_part)) == 0)
                for (iblock, (bi, block_rows, AAfac, AB_block, AB_hp,
                              C_block_output_buffer_position, next_bi)) ∈
                        enumerate(zip(block_inds, C_part_block_rows,
                                      AA_solver_block_factorizations, AB_blocks,
                                      hypercube_positions, output_buffer_positions,
                                      AB_block_colinds))
                    for (overlap_i, part_row) ∈ enumerate(block_rows)
                        row = row_offset + part_row
                        C_block_this_row = @view C_block_buffer[1:length(bi)]
                        get_row_block_from_sparse_matrix!(C_block_this_row, C_part, bi,
                                                          part_row)

                        # Add contributions from this level to the output buffer.
                        col_offset = 0
                        for (B_part, AAinv_dot_B_part_blocks, B_part_block_columns) ∈
                                zip(B_parts, AAinv_dot_B_blocks, B_block_columns)
                            block_cols = B_part_block_columns[iblock]
                            AAiB_block = AAinv_dot_B_part_blocks[iblock]
                            for part_col ∈ block_cols
                                col = col_offset + part_col
                                AAiB_block_this_col = AAiB_block[part_col]
                                if N == 2
                                    C_dot_Ainv_dot_B[row,col] -=
                                        dot(C_block_this_row, AAiB_block_this_col)
                                elseif N == 3
                                    C_dot_Ainv_dot_B[C_block_output_buffer_position,row,col] -=
                                        dot(C_block_this_row, AAiB_block_this_col)
                                else
                                    error("Unexpected number of dimensions ($(N)) for "
                                          * "C_dot_Ainv_dot_B.")
                                end
                            end
                            col_offset += size(B_part, 2)
                        end

                        if Afac_separate_Ainv_B
                            # rdiv!() doesn't pass through to the LAPACK function, but ldiv!()
                            # of the transpose does.
                            ldiv!((transpose(AAfac)), C_block_this_row)
                        end

                        this_buffer = @view next_C_block_buffer[1:size(AB_block, 2)]
                        mul!(transpose(this_buffer), transpose(C_block_this_row),
                             AB_block, -1.0, 0.0)
                        outer_CAiB_fill_intermediate_C_buffer!(next_C_part, next_C_inter,
                                                               AB_hp, this_buffer,
                                                               part_row, next_bi)
                    end

                end
            end
            row_offset += size(C_part, 1)
        end
    else
        # This process owns a single block, that is shared with other processes.
        block_rank = AA_solver.block_comm_rank
        bi = block_inds
        AA_block_factorization = AA_solver.local_block_serial_solver
        if isa(AB, Union{MPISchurComplementBlockAinvDotB,MPISchurComplementBlockB})
            AB_block = AB.block
            AB_block_colinds = Ab.block_colinds
        else
            AB_block = AB
            AB_block_colinds = 1:length(local_bottom_vector_indices)
        end
        row_offset = 0
        for (C_part, next_C_part, next_C_inter, C_part_block_rows) ∈
                zip(C_parts, next_C_parts, next_C_intermediate_buffers, C_block_rows)
            if isempty(C_part)
                for (overlap_i, part_row) ∈ enumerate(C_part_block_rows[1])
                    row = row_offset + part_row

                    get_row_block_from_sparse_matrix!(C_block_buffer, C_part, bi, part_row)

                    # Add contributions from this level to the output buffer.
                    col_offset = 0
                    for (B_part, AAinv_dot_B_part_blocks, B_part_block_columns,
                         B_part_output_buffer_positions) ∈
                            zip(B_parts, AAinv_dot_B_blocks, B_block_columns,
                                C_block_output_buffer_positions)
                        block_cols = B_part_block_columns[1]
                        AAiB_block = AAinv_dot_B_part_blocks[1]
                        for (buffer_position, part_col) ∈
                                zip(B_part_output_buffer_positions, block_cols)
                            col = col_offset + part_col
                            AAiB_block_this_col = AAiB_block[part_col]
                            if N == 2
                                C_dot_Ainv_dot_B[row,col] -=
                                    dot(C_block_buffer, AAiB_block_this_col)
                            elseif N == 3
                                C_dot_Ainv_dot_B[buffer_position,row,col] -=
                                    dot(C_block_buffer, AAiB_block_this_col)
                            else
                                error("Unexpected number of dimensions ($(N)) for "
                                      * "C_dot_Ainv_dot_B.")
                            end
                        end
                        col_offset += size(B_part, 2)
                    end

                    # rdiv!() doesn't pass through to the LAPACK function, but ldiv!()
                    # of the transpose does.
                    ldiv!((transpose(AAfac)), C_block_buffer)

                    mul!(transpose(next_C_block_buffer), transpose(C_block_buffer),
                         AB_block, -1.0, 0.0)
                    outer_CAiB_fill_intermediate_C_buffer!(next_C_part, next_C_inter,
                                                           hypercube_position,
                                                           next_C_block_buffer, part_row,
                                                           AB_block_colinds)
                end
            end
            row_offset += size(C_part, 1)
        end
    end

    synchronize_shared()

    for (C_part, next_C_part, next_C_inter, next_partial_range, next_copy_row_range) ∈
            zip(C_parts, next_C_parts, next_C_intermediate_buffers,
                next_C_partial_intermediate_ranges, C_bottom_copy_partial_row_ranges)

        if !isempty(C_part) && !(issparse(next_C_part) && nnz(next_C_part) == 0)
            outer_CAiB_sum_C_intermediates!(next_C_part, next_C_inter, next_partial_range,
                                            C_part, next_copy_row_range,
                                            local_bottom_vector_indices)
        end
    end

    return _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B, next_C,
                                       A_factorization.local_block_solver.schur_complement_factorization,
                                       next_B, synchronize_shared)
end

function _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractArray{T,N},
                                     C::OuterCSubmatrix,
                                     A_factorization::MPIStaticCondensationNull,
                                     B::OuterBSubmatrix,
                                     synchronize_shared::F) where {T,N,F}
    error("does this actually happen???")
    return nothing
end

function _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractArray{T,N},
                                     C::OuterCSubmatrix, A_factorization::MPIDenseLU,
                                     B::OuterBSubmatrix,
                                     previous_synchronize_shared::F) where {T,N,F}
    B_parts_list = B.parts_list
    output_partial_col_ranges_list = B.next_partial_intermediate_ranges
    B_partial_col_ranges_list = B.bottom_copy_partial_col_ranges
    C_parts_list = C.parts_list
    C_row_ranges_list = C.bottom_copy_partial_row_ranges
    synchronize_shared = A_factorization.synchronize_shared

    if N == 2
        output_buffer = C_dot_Ainv_dot_B
    elseif N == 3
        output_buffer = @view C_dot_Ainv_dot_B[1,:,:]
    else
        error("Unexpected number of dimensions N=$N for C_dot_Ainv_dot_B.")
    end

    for (B_block, output_partial_col_range, B_partial_col_range) ∈
            zip(B_parts_list, output_partial_col_ranges_list, B_partial_col_ranges_list)

        ldiv_no_distributed!(A_factorization, B_buffer)
        synchronize_shared()

        for (C_block, C_row_range) ∈ zip(C_parts_list, C_row_ranges_list)
            @views mul!(output_buffer[C_row_range,output_partial_col_range], C_buffer,
                        B_buffer[:,B_partial_col_range], -1.0, 1.0)
        end
    end

    return nothing
end

function _internal_mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractArray{T,N},
                                     C::OuterCSubmatrix,
                                     A_factorization::Union{LU,Nothing},
                                     B::OuterBSubmatrix,
                                     synchronize_shared::F) where {T,N,F}
    B_parts_list = B.parts_list
    output_partial_col_ranges_list = B.next_partial_intermediate_ranges
    B_partial_col_ranges_list = B.bottom_copy_partial_col_ranges
    C_parts_list = C.parts_list
    C_row_ranges_list = C.bottom_copy_partial_row_ranges

    if N == 2
        output_buffer = C_dot_Ainv_dot_B
    elseif N == 3
        output_buffer = @view C_dot_Ainv_dot_B[1,:,:]
    else
        error("Unexpected number of dimensions N=$N for C_dot_Ainv_dot_B.")
    end

    for (B_block, output_partial_col_range, B_partial_col_range) ∈
            zip(B_parts_list, output_partial_col_ranges_list, B_partial_col_ranges_list)

        if isempty(B_block)
            continue
        end

        if A_factorization !== nothing
            ldiv!(A_factorization, B_block)
        end
        synchronize_shared()

        for (C_block, C_row_range) ∈ zip(C_parts_list, C_row_ranges_list)
            if !isempty(C_block)
                @views mul!(output_buffer[C_row_range,output_partial_col_range], C_block,
                            B_block[:,B_partial_col_range], -1.0, 1.0)
            end
        end
    end

    return nothing
end

function copy_B_submatrix!(B::OuterBSubmatrix,
                           B_input::Union{AbstractMatrix,Vector{<:AbstractMatrix},NTuple})
    if isa(B_input, Union{Vector{<:AbstractMatrix},NTuple})
        input_parts = block_matrix
    elseif isa(B_input, BlockMatrix)
        matrix_blocks = blocks(B_input)
        block_nrows, block_ncols = size(matrix_blocks)
        if block_nrows > 1
            error("OuterBSubmatrix expects a BlockMatrix with only one block row.")
        end
        input_parts = reshape(matrix_blocks, block_ncols)
    else
        input_parts = [B_input]
    end

    if !all(p === p_in for (p, p_in) ∈ zip(B.parts_list, input_parts))
        error("OuterBSubmatrix requires the matrix to be updated in-place in the same "
              * "array that was passed to its constructor, to save the cost of copying "
              * "the matrix an extra time.")
    end

    # Copy the locally-owned part into matmul_partial_copy for use in matrix-vector
    # operations. Note that the pattern of non-zeros within `matmul_partial_row_range` is
    # identical for `matmul_partial_copy` and `B_input`.
    matmul_partial_copy = B.matmul_partial_copy
    matmul_partial_row_range = B.matmul_partial_row_range
    if isempty(matmul_partial_row_range)
        return nothing
    end
    if issparse(matmul_partial_copy)
        first_row = matmul_partial_row_range[1]
        last_row = matmul_partial_row_range[end]
        col = 1
        colptr = matmul_partial_copy.colptr
        rowval = matmul_partial_copy.rowval
        nzval = matmul_partial_copy.nzval
        for p ∈ input_parts
            p_colptr = p.colptr
            p_rowval = p.rowval
            p_nzval = p.nzval
            if length(p_nzval) == 0
                col += size(p, 2)
                continue
            end
            for pcol ∈ 1:size(p, 2)
                first_i = colptr[col]
                last_i = colptr[col+1] - 1
                p_col_first_i = p_colptr[pcol]
                p_col_last_i = p_colptr[pcol+1] - 1
                if p_col_last_i < p_col_first_i
                    continue
                end
                p_first_row = p_rowval[p_col_first_i]
                p_last_row = p_rowval[p_col_last_i]
                if p_first_row > last_row || p_last_row < first_row
                    continue
                end
                p_col_rv = @view p_rowval[p_col_first_i:p_col_last_i]
                p_first_i = searchsortedfirst(p_col_rv, first_row) + p_col_first_i - 1
                p_last_i = searchsortedlast(p_col_rv, last_row) + p_col_first_i - 1
                @views nzval[first_i:last_i] .= p_nzval[p_first_i:p_last_i]
                col += 1
            end
        end
    else
        matmul_partial_copy .= @view B_input[matmul_partial_row_range,:]
    end

    return nothing
end

function copy_C_submatrix!(C::OuterCSubmatrix,
                           C_input::Union{AbstractMatrix,Vector{<:AbstractMatrix},NTuple})
    if isa(C_input, Union{Vector{<:AbstractMatrix},NTuple})
        input_parts = C_input
    elseif isa(C_input, BlockMatrix)
        matrix_blocks = blocks(C_input)
        block_nrows, block_ncols = size(matrix_blocks)
        if block_ncols > 1
            error("OuterCSubmatrix expects a BlockMatrix with only one block column.")
        end
        input_parts = reshape(matrix_blocks, block_nrows)
    else
        input_parts = [C_input]
    end

    if !all(p === p_in for (p, p_in) ∈ zip(C.parts_list, input_parts))
        error("OuterCSubmatrix requires the matrix to be updated in-place in the same "
              * "array that was passed to its constructor, to save the cost of copying "
              * "the matrix an extra time.")
    end

    # Copy the locally-owned part into matmul_partial_copy for use in matrix-vector
    # operations. Note that the pattern of non-zeros within `matmul_partial_row_range` is
    # identical for `matmul_partial_copy` and `C_input`.
    matmul_partial_copy = C.matmul_partial_copy
    matmul_partial_row_range = C.matmul_partial_row_range
    if issparse(matmul_partial_copy)
        matmul_partial_copy_transpose = transpose(matmul_partial_copy)
        if isempty(matmul_partial_row_range)
            return nothing
        end
        first_row = matmul_partial_row_range[1]
        last_row = matmul_partial_row_range[end]
        row = 1
        colptr = matmul_partial_copy_transpose.colptr
        nzval = matmul_partial_copy_transpose.nzval
        row_offset = 0
        for p ∈ input_parts
            p_transpose = transpose(p)
            p_colptr = p_transpose.colptr
            p_nzval = p_transpose.nzval
            if row_offset + size(p, 1) < first_row || length(p_nzval) == 0
                row_offset += size(p, 1)
                continue
            end
            if row_offset ≥ last_row
                break
            end
            if row_offset < first_row
                first_partial_copy_row = 1
                first_p_row = first_row - row_offset
            else
                first_partial_copy_row = row_offset + 2 - first_row
                first_p_row = 1
            end
            if row_offset + size(p, 1) ≤ last_row
                last_partial_copy_row = row_offset + size(p, 1) - first_row + 1
                last_p_row = size(p, 1)
            else
                last_partial_copy_row = length(matmul_partial_row_range)
                last_p_row = last_row - row_offset
            end
            first_i = colptr[first_partial_copy_row]
            last_i = colptr[last_partial_copy_row+1] - 1
            p_first_i = p_colptr[first_p_row]
            p_last_i = p_colptr[last_p_row+1] - 1
            @views nzval[first_i:last_i] .= p_nzval[p_first_i:p_last_i]
            row_offset += size(p, 1)
        end
    else
        matmul_partial_copy .= @view C_input[matmul_partial_row_range,:]
    end

    return nothing
end

function Ainv_dot_B_dot_y!(top_vec_buffer::AbstractVector,
                           A_factorization::MPIStaticCondensationParallel,
                           B::OuterBSubmatrix, global_y::AbstractVector)
    B_partial = B.matmul_partial_copy
    B_partial_range = B.matmul_partial_row_range
    synchronize_shared = A_factorization.synchronize_shared

    @views mul!(top_vec_buffer[B_partial_range], B_partial, global_y)

    synchronize_shared()

    ldiv!(A_factorization, top_vec_buffer)

    return nothing
end

function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::OuterCSubmatrix,
                               Ainv_dot_u::AbstractVector)
    C_partial = C.matmul_partial_copy
    C_partial_range = C.matmul_partial_row_range

    @views mul!(C_dot_Ainv_dot_u[C_partial_range], C_partial, Ainv_dot_u, -1.0, 0.0)

    return nothing
end
