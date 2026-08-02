struct BlockAinvDotBSerial{Nvar,Tf,Ti,Tb,Trange}
    blocks::Vector{Tb}
    block_rowinds::Vector{NTuple{Nvar,Trange}}
    block_vector_rowinds::Vector{Trange}
    block_row_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    block_colinds::Vector{NTuple{Nvar,Trange}}
    block_col_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    bottom_block_vector_colinds::Vector{Trange}
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}

    function BlockAinvDotBSerial{Tf}(block_rowinds::Vector{NTuple{Nvar,Tind}},
                                     block_vector_rowinds::Vector{NTuple{Nvar,Tind}},
                                     block_colinds::Vector{NTuple{Nvar,Tind}},
                                     bottom_block_vector_colinds::Vector{NTuple{Nvar,Tind}}) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti}}
        nblock_unfiltered = length(block_rowinds)
        non_empty_blocks = [!all(isempty(vbi) for vbi ∈ block_rowinds[ib]) &&
                            !all(isempty(vbi) for vbi ∈ block_colinds[ib])
                            for ib ∈ 1:nblock_unfiltered]
        block_rowinds = block_rowinds[non_empty_blocks]
        block_row_range_offsets = [vcat(0, cumsum(length(vri) for vri ∈ ri[1:end-1]))
                                   for ri ∈ block_rowinds]
        block_row_ranges = [Tuple(voffset .+ (1:length(vri))
                                  for (vri, voffset) ∈ zip(ri, offsets))
                            for (ri, offsets) ∈ zip(block_rowinds, block_row_range_offsets)]
        block_vector_rowinds = [vcat(bvri...) for bvri ∈ block_vector_rowinds[non_empty_blocks]]
        bottom_block_vector_colinds = [vcat(bbvci...) for bbvci ∈ bottom_block_vector_colinds[non_empty_blocks]]
        block_colinds = block_colinds[non_empty_blocks]
        block_col_range_offsets = [vcat(0, cumsum(length(vci) for vci ∈ ci[1:end-1]))
                                   for ci ∈ block_colinds]
        block_col_ranges = [Tuple(voffset .+ (1:length(vci))
                                  for (vci, voffset) ∈ zip(ci, offsets))
                            for (ci, offsets) ∈ zip(block_colinds, block_col_range_offsets)]
        blocks = Matrix{Tf}[]
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out = Vector{Tf}[]
        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = sum(length(inds) for inds ∈ ri)
            ncol = sum(length(inds) for inds ∈ ci)
            push!(blocks, zeros(Tf, nrow, ncol))
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
        end
        return new{Nvar,Tf,Ti,eltype(blocks),Tind}(
                   blocks, block_rowinds, block_vector_rowinds, block_row_ranges,
                   block_colinds, block_col_ranges, bottom_block_vector_colinds,
                   vector_buffer_blocks_in, vector_buffer_blocks_out)
    end
end

# This version has a single block, and operations are parallelised using shared-memory
# MPI.
struct BlockAinvDotBShared{Nvar,Tf,Ti,Tb,Tind,Tsync}
    block::Tb
    partial_block::Matrix{Tf}
    block_rowinds::NTuple{Nvar,Tind}
    block_partial_vector_rowinds::Vector{Ti}
    block_colinds::NTuple{Nvar,Tind}
    block_partial_colinds::NTuple{Nvar,Vector{Ti}}
    bottom_block_vector_colinds::Tind
    bottom_block_partial_vector_colinds::Vector{Ti}
    partial_col_ranges::NTuple{Nvar,UnitRange{Ti}}
    partial_vector_col_range::UnitRange{Ti}
    partial_vector_row_range::UnitRange{Ti}
    vector_buffer_block_in::Vector{Tf}
    vector_buffer_block_out::Vector{Tf}
    synchronize_shared::Tsync

    function BlockAinvDotBShared{Tf}(block_rowinds::NTuple{Nvar,Tind},
                                     block_vector_rowinds::NTuple{Nvar,Tind},
                                     block_colinds::NTuple{Nvar,Tind},
                                     bottom_block_vector_colinds::NTuple{Nvar,Tind},
                                     block_comm_rank::Integer, block_comm_size::Integer,
                                     allocate_shared_float::Fa,
                                     synchronize_shared::Fs) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti},Fa,Fs}
        if isempty(block_rowinds) || isempty(block_colinds)
            return new{Nvar,Tf,Ti,Matrix{Tf},Tind,Fs}(
                       zeros(Tf, 0, 0), zeros(Tf, 0, 0), ntuple(i->zeros(Ti, 0), Nvar),
                       zeros(Ti, 0), ntuple(i->zeros(Ti, 0), Nvar),
                       ntuple(i->zeros(Ti, 0), Nvar), zeros(Ti, 0), zeros(Ti, 0),
                       ntuple(i->1:0, Nvar), 1:0, 1:0, zeros(Tf, 0), zeros(Tf, 0),
                       synchronize_shared)
        end

        nrow = sum(length(bi) for bi ∈ block_rowinds)
        ncol = sum(length(bi) for bi ∈ block_colinds)
        block = allocate_shared_float(nrow, ncol)
        cols_per_proc = Tuple((length(ci) + block_comm_size - 1) ÷ block_comm_size for ci ∈ block_colinds)
        partial_col_ranges = Tuple(block_comm_rank*cpp+1:min((block_comm_rank+1)*cpp,length(ci))
                                   for (cpp, ci) ∈ zip(cols_per_proc, block_colinds))
        block_partial_colinds = Tuple(ci[pcr] for (pcr, ci) ∈ zip(partial_col_ranges, block_colinds))
        bottom_block_vector_colinds = vcat(bottom_block_vector_colinds...)
        vector_cols_per_proc = (ncol + block_comm_size - 1) ÷ block_comm_size
        partial_vector_col_range = block_comm_rank*vector_cols_per_proc+1:min((block_comm_rank+1)*vector_cols_per_proc,ncol)
        bottom_block_partial_vector_colinds = bottom_block_vector_colinds[partial_vector_col_range]
        block_vector_rowinds = vcat(block_vector_rowinds...)
        vector_rows_per_proc = (nrow + block_comm_size - 1) ÷ block_comm_size
        partial_vector_row_range = block_comm_rank*vector_rows_per_proc+1:min((block_comm_rank+1)*vector_rows_per_proc,nrow)
        partial_nrow = length(partial_vector_row_range)
        block_partial_vector_rowinds = block_vector_rowinds[partial_vector_row_range]
        vector_buffer_block_in = allocate_shared_float(ncol)
        vector_buffer_block_out = zeros(Tf, partial_nrow)
        partial_block = zeros(Tf, partial_nrow, ncol)

        block[:,partial_vector_col_range] .= 0.0
        vector_buffer_block_in[partial_vector_col_range] .= 0.0
        vector_buffer_block_out .= 0.0

        return new{Nvar,Tf,Ti,typeof(block),Tind,Fs}(
                   block, partial_block, block_rowinds, block_partial_vector_rowinds,
                   block_colinds, block_partial_colinds, bottom_block_vector_colinds,
                   bottom_block_partial_vector_colinds, partial_col_ranges,
                   partial_vector_col_range, partial_vector_row_range,
                   vector_buffer_block_in, vector_buffer_block_out, synchronize_shared)
    end
end

function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractMatrix}}) where Nvar
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = Ainv_dot_B.block_rowinds
        block_colinds = Ainv_dot_B.block_colinds
        for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                    blocks)
            for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                    (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                A_variable_block = full_A[vrow][vcol]
                for (j1, j2) ∈ zip(cr, ci), (i1, i2) ∈ zip(rr, ri)
                    block[i1,j1] = A_variable_block[i2,j2]
                end
            end
        end
        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}}) where Nvar
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = Ainv_dot_B.block_rowinds
        block_row_ranges = Ainv_dot_B.block_row_ranges
        block_colinds = Ainv_dot_B.block_colinds
        block_col_ranges = Ainv_dot_B.block_col_ranges
        for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                    blocks)
            for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                    (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval = A_variable_block.rowval
                full_A_nzval = A_variable_block.nzval
                if isempty(ci) || isempty(ri)
                    continue
                end
                if isempty(full_A_nzval)
                    block[rr,cr] .= 0.0
                    continue
                end
                last_irow = last(rr)
                if isempty(ri)
                    first_row = 1
                else
                    first_row = first(ri)
                end
                nrow = length(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    last_i = full_A_colptr[j2+1] - 1
                    col_rv = @view full_A_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                    i1 = 1
                    while i1 ≤ nrow
                        full_A_row = full_A_rowval[flat_i]
                        block_global_row = ri[i1]
                        if full_A_row == block_global_row
                            block[rr[i1],j1] = full_A_nzval[flat_i]
                            i1 += 1
                            flat_i += 1
                        elseif full_A_row > block_global_row
                            block[rr[i1],j1] = 0.0
                            i1 += 1
                        else
                            flat_i += 1
                        end
                        if flat_i > last_i && i1 ≤ nrow
                            block[rr[i1]:last_irow,j1] .= 0.0
                            break
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}}) where Nvar
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = Ainv_dot_B.block_rowinds
        block_row_ranges = Ainv_dot_B.block_row_ranges
        block_colinds = Ainv_dot_B.block_colinds
        block_col_ranges = Ainv_dot_B.block_col_ranges
        for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                    blocks)
            for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                    (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval_list = A_variable_block.rowval_list
                full_A_nzval = A_variable_block.nzval
                if isempty(ci) || isempty(ri)
                    continue
                end
                if isempty(full_A_nzval)
                    block[rr,cr] .= 0.0
                    continue
                end
                last_irow = last(rr)
                first_row = first(ri)
                nrow = length(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    col_rv = full_A_rowval_list[j2]
                    if isempty(col_rv)
                        block[rr,j1] .= 0.0
                        continue
                    end
                    last_row_i = length(col_rv)
                    row_i = max(searchsortedlast(col_rv, first_row)-1,1)
                    i1 = 1
                    while i1 ≤ nrow
                        full_A_row = col_rv[row_i]
                        block_global_row = ri[i1]
                        if full_A_row == block_global_row
                            block[rr[i1],j1] = full_A_nzval[row_i+first_i-1]
                            i1 += 1
                            row_i += 1
                        elseif full_A_row > block_global_row
                            block[rr[i1],j1] = 0.0
                            i1 += 1
                        else
                            row_i += 1
                        end
                        if row_i > last_row_i && i1 ≤ nrow
                            block[rr[i1]:last_irow,j1] .= 0.0
                            break
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBShared,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}}) where Nvar
    @inbounds begin
        block = Ainv_dot_B.block
        if length(block) == 0
            # Nothing to do.
            return nothing
        end
        block_rowinds = Ainv_dot_B.block_rowinds
        block_row_ranges = Ainv_dot_B.block_row_ranges
        block_colinds = Ainv_dot_B.block_colinds
        partial_col_ranges = Ainv_dot_B.partial_col_ranges
        partial_col_ranges = Ainv_dot_B.partial_col_ranges

        for (vcol, ci, pcr) ∈ zip(1:Nvar, block_colinds, partial_col_ranges),
                (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
            A_variable_block = full_A[vrow][vcol]
            full_A_colptr = A_variable_block.colptr
            full_A_rowval = A_variable_block.rowval
            full_A_nzval = A_variable_block.nzval
            if isempty(ci) || isempty(ri)
                continue
            end
            if isempty(full_A_nzval)
                block[rr,cr] .= 0.0
                continue
            end
            last_irow = last(rr)
            first_row = first(ri)
            nrow = length(ri)
            for j1 ∈ pcr
                j2 = ci[j1]
                first_i = full_A_colptr[j2]
                last_i = full_A_colptr[j2+1] - 1
                col_rv = @view full_A_rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                i1 = 1
                while i1 ≤ nrow
                    full_A_row = full_A_rowval[flat_i]
                    block_global_row = block_rowinds[i1]
                    if full_A_row == block_global_row
                        block[rr[i1],j1] = full_A_nzval[flat_i]
                        i1 += 1
                        flat_i += 1
                    elseif full_A_row > block_global_row
                        block[rr[i1],j1] = 0.0
                        i1 += 1
                    else
                        flat_i += 1
                    end
                    if flat_i > last_i && i1 ≤ nrow
                        block[rr[i1]:last_irow,j1] .= 0.0
                        break
                    end
                end
            end
        end

        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBShared,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}}) where Nvar
    @inbounds begin
        block = Ainv_dot_B.block
        if length(block) == 0
            # Nothing to do.
            return nothing
        end
        block_rowinds = Ainv_dot_B.block_rowinds
        block_colinds = Ainv_dot_B.block_colinds
        partial_col_ranges = Ainv_dot_B.partial_col_ranges
        partial_col_ranges = Ainv_dot_B.partial_col_ranges

        for (vcol, ci, pcr) ∈ zip(1:Nvar, block_colinds, partial_col_ranges),
                (vrow, ri) ∈ zip(1:Nvar, block_rowinds)
            A_variable_block = full_A[vrow][vcol]
            full_A_colptr = A_variable_block.colptr
            full_A_rowval_list = A_variable_block.rowval_list
            full_A_nzval = A_variable_block.nzval
            if isempty(ci) || isempty(ri)
                continue
            end
            if isempty(full_A_nzval)
                block[rr,cr] .= 0.0
                continue
            end
            last_irow = length(ri)
            first_row = first(ri)
            for j1 ∈ pcr
                j2 = ci[j1]
                first_i = full_A_colptr[j2]
                col_rv = full_A_rowval_list[j2]
                last_row_i = length(col_rv)
                row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                i1 = 1
                while i1 ≤ last_irow
                    full_A_row = col_rv[row_i]
                    block_global_row = ri[i1]
                    if full_A_row == block_global_row
                        block[i1,j1] = full_A_nzval[row_i+first_i-1]
                        i1 += 1
                        row_i += 1
                    elseif full_A_row > block_global_row
                        block[i1,j1] = 0.0
                        i1 += 1
                    else
                        row_i += 1
                    end
                    if row_i > last_row_i && i1 ≤ last_irow
                        block[i1:last_irow,j1] .= 0.0
                        break
                    end
                end
            end
        end

        return nothing
    end
end

function mul_C_Ainv_dot_B!(schur_complement::BlockS{Nvar}, C::BlockCSerial{Nvar},
                           Ainv_dot_B::BlockAinvDotBSerial) where Nvar
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        C_blocks = C.blocks
        sc_matrix = schur_complement.matrix
        synchronize_shared = C.synchronize_shared
        n_hypercube_positions = C.n_hypercube_positions
        dense_buffer_storage = C.dense_buffer_storage

        mul_blocks = C.right_multiplication_buffer_blocks
        Ainv_dot_B_blocks = Ainv_dot_B.blocks
        block_output_inds = C.bottom_block_rowinds
        block_output_ranges = C.block_row_ranges

        if isa(sc_matrix[1][1], FixedSparseCSC)
            flat_ranges_partial = schur_complement.flat_ranges_partial
            for (flat_ranges_row, matrix_row) ∈ zip(flat_ranges_partial, sc_matrix)
                for (fr, matrix_block) ∈ zip(flat_ranges_row, matrix_row)
                    nzval = matrix_block.nzval
                    if !isempty(fr)
                        # Need to zero this buffer as other levels might put non-zeros in places that
                        # will not be filled (by any process) in the following loop.
                        nzval[fr] .= 0.0
                    end
                end
            end

            # The rows are labelled by block_hypercube_position, so there are no overlaps,
            # and we can directly set entries, instead of adding to them, and so do not
            # need to zero-initialise the output buffer.
            block_hypercube_positions = C.block_hypercube_positions
            if dense_buffer_storage === nothing
                for (mb, Cb, AiBb, bhp) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks,
                                              block_hypercube_positions)
                    mul!(mb, Cb, AiBb, -1.0, 0.0)
                end
            else
                for (mb, Cb, AiBb, bhp) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks,
                                              block_output_inds,
                                              block_hypercube_positions)
                    nrow, ncol = size(Cb)
                    dense_buffer = reshape(@view(dense_buffer_storage[1:nrow*ncol]), nrow,
                                           ncol)
                    C_colptr = Cb.colptr
                    C_rowval = Cb.rowval
                    C_nzval = Cb.nzval
                    dense_buffer .= 0.0
                    for j ∈ 1:ncol
                        col_start = C_colptr[j]
                        col_end = C_colptr[j+1]-1
                        for flat_i ∈ col_start:col_end
                            i = C_rowval[flat_i]
                            dense_buffer[i,j] = C_nzval[flat_i]
                        end
                    end
                    mul!(mb, dense_buffer, AiBb, -1.0, 0.0)
                end
            end

            synchronize_shared()

            current_hypercube_position = 1
            for (mb, output_ranges, output_inds, bhp) ∈
                    zip(mul_blocks, block_output_ranges, block_output_inds,
                        block_hypercube_positions)
                for _ ∈ current_hypercube_position:bhp-1
                    # Synchronize in between copying different 'hypercube positions',
                    # as blocks in different hypercube positions can overlap.
                    # Note that the blocks are sorted by hypercube position, so this loop
                    # will include every block owned by this process.
                    synchronize_shared()
                end
                current_hypercube_position = bhp

                # Add result from mb into schur_complement.
                for (jvar, col_range, colinds) ∈ zip(1:Nvar, output_ranges, output_inds),
                        (ivar, row_range, rowinds) ∈ zip(1:Nvar, output_ranges, output_inds)
                    sc_matrix_variable_block = sc_matrix[ivar][jvar]
                    colptr = sc_matrix_variable_block.colptr
                    rowval = sc_matrix_variable_block.rowval
                    nzval = sc_matrix_variable_block.nzval
                    first_row = first(rowinds)
                    first_i = first(row_range)
                    last_i = last(row_range)
                    for (j, col) ∈ zip(col_range, colinds)
                        first_flat_i = colptr[col]
                        last_flat_i = colptr[col+1] - 1
                        col_rv = @view rowval[first_flat_i:last_flat_i]
                        flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_flat_i - 1
                        i = first_i
                        while flat_i ≤ last_flat_i && i ≤ last_i
                            if rowval[flat_i] == ri[i]
                                nzval[flat_i] += mb[i,j]
                                flat_i += 1
                                i += 1
                            else
                                # rowval[flat_i] must be less than ri[i]
                                flat_i += 1
                            end
                        end
                    end
                end
            end

            for _ ∈ current_hypercube_position:n_hypercube_positions-1
                # Synchronize in between copying different 'hypercube positions',
                # as blocks in different hypercube positions can overlap.
                synchronize_shared()
            end
        elseif isa(sc_matrix[1][1], SharedSparseBuffer)
            flat_ranges_partial = schur_complement.flat_ranges_partial
            for (flat_ranges_row, matrix_row) ∈ zip(flat_ranges_partial, sc_matrix)
                for (fr, matrix_block) ∈ zip(flat_ranges_row, matrix_row)
                    nzval = matrix_block.nzval
                    if !isempty(fr)
                        # Need to zero this buffer as other levels might put non-zeros in places that
                        # will not be filled (by any process) in the following loop.
                        nzval[fr] .= 0.0
                    end
                end
            end

            # The rows are labelled by block_hypercube_position, so there are no overlaps,
            # and we can directly set entries, instead of adding to them, and so do not
            # need to zero-initialise the output buffer.
            block_hypercube_positions = C.block_hypercube_positions
            if dense_buffer_storage === nothing
                for (mb, Cb, AiBb, bhp) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks,
                                              block_hypercube_positions)
                    mul!(mb, Cb, AiBb, -1.0, 0.0)
                end
            else
                for (mb, Cb, AiBb, bhp) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks,
                                              block_hypercube_positions)
                    nrow, ncol = size(Cb)
                    dense_buffer = reshape(@view(dense_buffer_storage[1:nrow*ncol]), nrow,
                                           ncol)
                    C_colptr = Cb.colptr
                    C_rowval = Cb.rowval
                    C_nzval = Cb.nzval
                    dense_buffer .= 0.0
                    for j ∈ 1:ncol
                        col_start = C_colptr[j]
                        col_end = C_colptr[j+1]-1
                        for flat_i ∈ col_start:col_end
                            i = C_rowval[flat_i]
                            dense_buffer[i,j] = C_nzval[flat_i]
                        end
                    end
                    mul!(mb, dense_buffer, AiBb, -1.0, 0.0)
                end
            end

            synchronize_shared()

            current_hypercube_position = 1
            for (mb, output_ranges, output_inds, bhp) ∈
                    zip(mul_blocks, block_output_ranges, block_output_inds,
                        block_hypercube_positions)
                for _ ∈ current_hypercube_position:bhp-1
                    # Synchronize in between copying different 'hypercube positions',
                    # as blocks in different hypercube positions can overlap.
                    # Note that the blocks are sorted by hypercube position, so this loop
                    # will include every block owned by this process.
                    synchronize_shared()
                end
                current_hypercube_position = bhp

                # Add result from mb into schur_complement.
                for (jvar, col_range, colinds) ∈ zip(1:Nvar, output_ranges, output_inds),
                        (ivar, row_range, rowinds) ∈ zip(1:Nvar, output_ranges, output_inds)
                    sc_matrix_variable_block = sc_matrix[ivar][jvar]
                    colptr = sc_matrix_variable_block.colptr
                    rowval_list = sc_matrix_variable_block.rowval_list
                    nzval = sc_matrix_variable_block.nzval
                    first_row = first(rowinds)
                    nrow = length(rowinds)
                    for (j, col) ∈ zip(col_range, colinds)
                        first_flat_i = colptr[col]
                        col_rv = rowval_list[col]
                        last_row_i = length(col_rv)
                        row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                        i = 1
                        while row_i ≤ last_row_i && i ≤ nrow
                            if col_rv[row_i] == rowinds[i]
                                nzval[row_i+first_flat_i-1] += mb[row_range[i],j]
                                row_i += 1
                                i += 1
                            else
                                # col_rv[row_i] must be less than rowinds[i]
                                row_i += 1
                            end
                        end
                    end
                end
            end

            for _ ∈ current_hypercube_position:n_hypercube_positions-1
                # Synchronize in between copying different 'hypercube positions',
                # as blocks in different hypercube positions can overlap.
                synchronize_shared()
            end
        else
            error("Unexpected type for sc_matrix[1][1] ($(typeof(sc_matrix[1][1]))).")
        end

        return nothing
    end
end
function mul_C_Ainv_dot_B!(schur_complement::BlockS{Nvar}, C::BlockCShared{Nvar},
                           Ainv_dot_B::BlockAinvDotBShared{Nvar}) where {Nvar}
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        sc_matrix = schur_complement.matrix
        synchronize_shared = C.synchronize_shared
        C_block = C.block
        mul_block = C.right_multiplication_buffer_block
        dense_C = C.dense_buffer
        block_output_rowinds = C.bottom_block_rowinds
        block_output_colinds = C.bottom_block_colinds
        block_hypercube_position = C.block_hypercube_position
        n_hypercube_positions = C.n_hypercube_positions
        Ainv_dot_B_block = Ainv_dot_B.block

        if isa(sc_matrix[1][1], FixedSparseCSC)
            flat_ranges_partial = schur_complement.flat_ranges_partial
            for (flat_ranges_row, matrix_row) ∈ zip(flat_ranges_partial, sc_matrix)
                for (fr, matrix_block) ∈ zip(flat_ranges_row, matrix_row)
                    nzval = matrix_block.nzval
                    if !isempty(fr)
                        # Need to zero this buffer as other levels might put non-zeros in places that
                        # will not be filled (by any process) in the following loop.
                        nzval[fr] .= 0.0
                    end
                end
            end

            if length(mul_block) != 0
                if dense_C === nothing
                    mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)
                else
                    ncol = size(C_block, 2)
                    C_colptr = C_block.colptr
                    C_rowval = C_block.rowval
                    C_nzval = C_block.nzval
                    dense_C .= 0.0
                    for j ∈ 1:ncol
                        col_start = C_colptr[j]
                        col_end = C_colptr[j+1]-1
                        for flat_i ∈ col_start:col_end
                            i = C_rowval[flat_i]
                            dense_C[i,j] = C_nzval[flat_i]
                        end
                    end
                    mul!(mul_block, dense_C, Ainv_dot_B_block, -1.0, 0.0)
                end
            end

            for hp ∈ 1:n_hypercube_positions
                synchronize_shared()
                if hp == block_hypercube_position && length(mul_block) != 0
                    # Add result from mul_block into schur_complement matrix.
                    for (jvar, colinds) ∈ zip(1:Nvar, block_output_colinds),
                            (ivar, rowinds) ∈ zip(1:Nvar, block_output_rowinds)
                        sc_matrix_variable_block = sc_matrix[ivar][jvar]
                        colptr = sc_matrix_variable_block.colptr
                        rowval = sc_matrix_variable_block.rowval
                        nzval = sc_matrix_variable_block.nzval
                        first_row = first(rowinds)
                        nrow = length(rowinds)
                        for (j, col) ∈ enumerate(colinds)
                            first_flat_i = colptr[col]
                            last_flat_i = colptr[col+1] - 1
                            col_rv = @view rowval[first_flat_i:last_flat_i]
                            flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_flat_i - 1
                            i = first_i
                            while flat_i ≤ last_flat_i && i ≤ nrow
                                if rowval[flat_i] == rowinds[i]
                                    nzval[flat_i] += mul_block[i,j]
                                    flat_i += 1
                                    i += 1
                                else
                                    # rowval[flat_i] must be less than rowinds[i].
                                    flat_i += 1
                                end
                            end
                        end
                    end
                end
            end
        elseif isa(sc_matrix[1][1], SharedSparseBuffer)
            flat_ranges_partial = schur_complement.flat_ranges_partial
            for (flat_ranges_row, matrix_row) ∈ zip(flat_ranges_partial, sc_matrix)
                for (fr, matrix_block) ∈ zip(flat_ranges_row, matrix_row)
                    nzval = matrix_block.nzval
                    if !isempty(fr)
                        # Need to zero this buffer as other levels might put non-zeros in places that
                        # will not be filled (by any process) in the following loop.
                        nzval[fr] .= 0.0
                    end
                end
            end

            if length(mul_block) != 0
                if dense_C === nothing
                    mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)
                else
                    ncol = size(C_block, 2)
                    C_colptr = C_block.colptr
                    C_rowval = C_block.rowval
                    C_nzval = C_block.nzval
                    dense_C .= 0.0
                    for j ∈ 1:ncol
                        col_start = C_colptr[j]
                        col_end = C_colptr[j+1]-1
                        for flat_i ∈ col_start:col_end
                            i = C_rowval[flat_i]
                            dense_C[i,j] = C_nzval[flat_i]
                        end
                    end
                    mul!(mul_block, dense_C, Ainv_dot_B_block, -1.0, 0.0)
                end
            end

            for hp ∈ 1:n_hypercube_positions
                synchronize_shared()
                if hp == block_hypercube_position && length(mul_block) != 0
                    # Add result from mul_block into schur_complement matrix.
                    for (jvar, colinds) ∈ zip(1:Nvar, block_output_colinds),
                            (ivar, rowinds) ∈ zip(1:Nvar, block_output_rowinds)
                        sc_matrix_variable_block = sc_matrix[ivar][jvar]
                        colptr = sc_matrix_variable_block.colptr
                        rowval_list = sc_matrix_variable_block.rowval_list
                        nzval = sc_matrix_variable_block.nzval
                        first_row = first(rowinds)
                        nrow = length(rowinds)
                        for (j, col) ∈ enumerate(colinds)
                            first_flat_i = colptr[col]
                            col_rv = rowval_list[col]
                            last_row_i = length(col_rv)
                            row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                            i = 1
                            while row_i ≤ last_row_i && i ≤ nrow
                                if col_rv[row_i] == rowinds[i]
                                    nzval[row_i+first_flat_i-1] += mul_block[i,j]
                                    row_i += 1
                                    i += 1
                                else
                                    # col_rv[row_i] must be less than rowinds[i].
                                    row_i += 1
                                end
                            end
                        end
                    end
                end
            end
        else
            error("Unexpected type for sc_matrix[1][1] ($(typeof(sc_matrix[1][1]))).")
        end

        return nothing
    end
end
function mul_C_Ainv_dot_B!(schur_complement::BlockDenseS{Nvar}, C::BlockCSerial{Nvar},
                           Ainv_dot_B::BlockAinvDotBSerial) where Nvar
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        C_blocks = C.blocks
        sc_matrix = schur_complement.matrix
        synchronize_shared = C.synchronize_shared
        n_hypercube_positions = C.n_hypercube_positions
        dense_buffer_storage = C.dense_buffer_storage

        mul_blocks = C.right_multiplication_buffer_blocks
        Ainv_dot_B_blocks = Ainv_dot_B.blocks
        block_output_inds = C.bottom_block_vector_rowinds

        column_range_partial = schur_complement.column_range_partial
        sc_matrix[:,column_range_partial] .= 0.0

        # The rows are labelled by block_hypercube_position, so there are no overlaps,
        # and we can directly set entries, instead of adding to them, and so do not
        # need to zero-initialise the output buffer.
        block_hypercube_positions = C.block_hypercube_positions
        if dense_buffer_storage === nothing
            for (mb, Cb, AiBb) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks)
                mul!(mb, Cb, AiBb, -1.0, 0.0)
            end
        else
            for (mb, Cb, AiBb) ∈ zip(mul_blocks, C_blocks, Ainv_dot_B_blocks)
                nrow, ncol = size(Cb)
                dense_buffer = reshape(@view(dense_buffer_storage[1:nrow*ncol]), nrow,
                                       ncol)
                C_colptr = Cb.colptr
                C_rowval = Cb.rowval
                C_nzval = Cb.nzval
                dense_buffer .= 0.0
                for j ∈ 1:ncol
                    col_start = C_colptr[j]
                    col_end = C_colptr[j+1]-1
                    for flat_i ∈ col_start:col_end
                        i = C_rowval[flat_i]
                        dense_buffer[i,j] = C_nzval[flat_i]
                    end
                end
                mul!(mb, dense_buffer, AiBb, -1.0, 0.0)
            end
        end

        synchronize_shared()

        current_hypercube_position = 1
        for (mb, output_inds, bhp) ∈ zip(mul_blocks, block_output_inds,
                                         block_hypercube_positions)
            for _ ∈ current_hypercube_position:bhp-1
                # Synchronize in between copying different 'hypercube positions',
                # as blocks in different hypercube positions can overlap.
                # Note that the blocks are sorted by hypercube position, so this loop
                # will include every block owned by this process.
                synchronize_shared()
            end
            current_hypercube_position = bhp

            # Add result from mb into schur_complement.
            for (j1, j2) ∈ enumerate(output_inds), (i1, i2) ∈ enumerate(output_inds)
                sc_matrix[i2,j2] += mb[i1,j1]
            end
        end

        for _ ∈ current_hypercube_position:n_hypercube_positions-1
            # Synchronize in between copying different 'hypercube positions',
            # as blocks in different hypercube positions can overlap.
            synchronize_shared()
        end

        return nothing
    end
end
function mul_C_Ainv_dot_B!(schur_complement::BlockDenseS, C::BlockCShared,
                           Ainv_dot_B::BlockAinvDotBShared)
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        sc_matrix = schur_complement.matrix
        synchronize_shared = C.synchronize_shared
        C_block = C.block
        mul_block = C.right_multiplication_buffer_block
        dense_C = C.dense_buffer
        block_output_rowinds = C.bottom_block_vector_rowinds
        block_output_colinds = C.bottom_block_vector_colinds
        block_hypercube_position = C.block_hypercube_position
        n_hypercube_positions = C.n_hypercube_positions
        Ainv_dot_B_block = Ainv_dot_B.block

        column_range_partial = schur_complement.column_range_partial
        sc_matrix[:,column_range_partial] .= 0.0

        if length(mul_block) != 0
            if dense_C === nothing
                mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)
            else
                ncol = size(C_block, 2)
                C_colptr = C_block.colptr
                C_rowval = C_block.rowval
                C_nzval = C_block.nzval
                dense_C .= 0.0
                for j ∈ 1:ncol
                    col_start = C_colptr[j]
                    col_end = C_colptr[j+1]-1
                    for flat_i ∈ col_start:col_end
                        i = C_rowval[flat_i]
                        dense_C[i,j] = C_nzval[flat_i]
                    end
                end
                mul!(mul_block, dense_C, Ainv_dot_B_block, -1.0, 0.0)
            end
        end

        for hp ∈ 1:n_hypercube_positions
            synchronize_shared()
            if hp == block_hypercube_position && length(mul_block) != 0
                # Add result from mul_block into schur_complement matrix.
                for (j1, j2) ∈ enumerate(block_output_colinds), (i1, i2) ∈ enumerate(block_output_rowinds)
                    sc_matrix[i2,j2] += mul_block[i1,j1]
                end
            end
        end

        return nothing
    end
end

function Ainv_dot_u_minus_Ainv_dot_B_dot_y!(x::AbstractVector, Ainv_dot_u,
                                            Ainv_dot_B::BlockAinvDotBSerial,
                                            y::AbstractVector)
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        for (vec_buffer_in, vec_buffer_out, rowinds, colinds, block, Aiu_block) ∈
                zip(Ainv_dot_B.vector_buffer_blocks_in,
                    Ainv_dot_B.vector_buffer_blocks_out, Ainv_dot_B.block_vector_rowinds,
                    Ainv_dot_B.bottom_block_vector_colinds, blocks, Ainv_dot_u)
            for (i1, i2) ∈ enumerate(colinds)
                vec_buffer_in[i1] = y[i2]
            end
            mul!(vec_buffer_out, block, vec_buffer_in)
            for (i2, i1) ∈ enumerate(rowinds)
                x[i1] = Aiu_block[i2] - vec_buffer_out[i2]
            end
        end
        return nothing
    end
end
function Ainv_dot_u_minus_Ainv_dot_B_dot_y!(x::AbstractVector, Ainv_dot_u,
                                            Ainv_dot_B::BlockAinvDotBShared,
                                            y::AbstractVector)
    @inbounds begin
        partial_block = Ainv_dot_B.partial_block
        vector_buffer_block_in = Ainv_dot_B.vector_buffer_block_in
        vector_buffer_block_out = Ainv_dot_B.vector_buffer_block_out
        block_partial_vector_rowinds = Ainv_dot_B.block_partial_vector_rowinds
        partial_vector_row_range = Ainv_dot_B.partial_vector_row_range
        bottom_block_partial_vector_colinds = Ainv_dot_B.bottom_block_partial_vector_colinds
        partial_vector_col_range = Ainv_dot_B.partial_vector_col_range
        synchronize_shared = Ainv_dot_B.synchronize_shared

        for (i1, i2) ∈ zip(partial_vector_col_range, bottom_block_partial_vector_colinds)
            vector_buffer_block_in[i1] = y[i2]
        end
        synchronize_shared()

        mul!(vector_buffer_block_out, partial_block, vector_buffer_block_in)
        for (i3, (i1, i2)) ∈ enumerate(zip(block_partial_vector_rowinds, partial_vector_row_range))
            x[i1] = Ainv_dot_u[i2] - vector_buffer_block_out[i3]
        end
        return nothing
    end
end
