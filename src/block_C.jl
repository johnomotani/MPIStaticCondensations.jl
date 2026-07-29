struct BlockCSerial{Nvar,Tf,Ti,Tb,Trange,Trmbb,Tdbs,Tib,Fsb<:Function,Fs<:Function}
    blocks::Vector{Tb}
    block_rowinds::Vector{NTuple{Nvar,Trange}}
    block_row_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    bottom_block_rowinds::Vector{NTuple{Nvar,Trange}}
    bottom_block_vector_rowinds::Vector{Trange}
    block_colinds::Vector{NTuple{Nvar,Trange}}
    block_col_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    block_hypercube_positions::Vector{Ti}
    n_hypercube_positions::Ti
    right_multiplication_buffer_blocks::Trmbb
    dense_buffer_storage::Tdbs
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}
    vector_intermediate_buffer::Tib
    vector_range::UnitRange{Ti}
    block_synchronize_shared::Fsb
    synchronize_shared::Fs

    function BlockCSerial{Tf}(block_rowinds::Vector{NTuple{Nvar,Tind}},
                              bottom_block_rowinds::Vector{NTuple{Nvar,Tind}},
                              bottom_block_vector_rowinds::Vector{NTuple{Nvar,Tind}},
                              block_colinds::Vector{NTuple{Nvar,Tind}},
                              matrix_template::Union{<:NTuple{Nvar,<:NTuple{Nvar,<:Union{AbstractSparseMatrixCSC,SharedSparseBuffer}}},Nothing},
                              block_hypercube_positions::Vector{Ti},
                              n_hypercube_positions::Ti,
                              right_multiplication_buffer_storage::Vector{Tf},
                              dense_buffer_storage::Vector{Tf},
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti},
                              block_synchronize_shared::Fsb,
                              synchronize_shared::Fs) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti},Fsb<:Function,Fs<:Function}
        nblock_unfiltered = length(block_rowinds)
        non_empty_blocks = [!all(isempty(vbi) for vbi ∈ block_rowinds[ib]) &&
                            !all(isempty(vbi) for vbi ∈ block_colinds[ib])
                            for ib ∈ 1:nblock_unfiltered]
        block_rowinds = block_rowinds[non_empty_blocks]
        block_row_range_offsets = [vcat(0, cumsum(length(vri) for vri ∈ ri[1:end-1]))
                                   for ri ∈ block_rowinds]
        block_row_ranges = [Tuple(voffset .+ 1:length(vri)
                                  for (vri, voffset) ∈ zip(ri, offsets))
                            for (ri, offsets) ∈ zip(block_rowinds, block_row_range_offsets)]
        bottom_block_rowinds = bottom_block_rowinds[non_empty_blocks]
        bottom_block_vector_rowinds = [vcat(bbvri...) for bbvri ∈ bottom_block_vector_rowinds[non_empty_blocks]]
        block_colinds = block_colinds[non_empty_blocks]
        block_col_range_offsets = [vcat(0, cumsum(length(vci) for vci ∈ ci[1:end-1]))
                                   for ci ∈ block_colinds]
        block_col_ranges = [Tuple(voffset .+ 1:length(vci)
                                  for (vci, voffset) ∈ zip(ci, offsets))
                            for (ci, offsets) ∈ zip(block_colinds, block_col_range_offsets)]
        if matrix_template === nothing
            blocks = Matrix{Tf}[]
        else
            blocks = FixedSparseCSC{Tf,Ti}[]
        end
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out = Vector{Tf}[]

        # Using Vector{Any} here, we convert to a concretely typed Vector after collecting
        # the buffer blocks.
        right_multiplication_buffer_blocks = []

        offset = 0
        max_length = 0
        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = sum(length(vri) for vri ∈ ri)
            ncol = sum(length(vci) for vci ∈ ci)
            if matrix_template === nothing
                push!(blocks, zeros(Tf, nrow, ncol))
            else
                b = get_partial_FixedSparseCSC_buffer(ri, ci, matrix_template, Tf)
                b.nzval .= 0.0
                push!(blocks, b)
            end
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
            right_multiplication_block_size = nrow^2
            if length(right_multiplication_buffer_storage) < offset + right_multiplication_block_size
                resize!(right_multiplication_buffer_storage,
                        offset + right_multiplication_block_size)
            end
            push!(right_multiplication_buffer_blocks,
                  reshape(@view(right_multiplication_buffer_storage[offset+1:offset+right_multiplication_block_size]),
                          nrow, nrow))
            offset += right_multiplication_block_size
            max_length = max(max_length, nrow * ncol)
        end

        if matrix_template === nothing
            dense_buffer_storage = nothing
        else
            if length(dense_buffer_storage) < max_length
                resize!(dense_buffer_storage, max_length)
            end
        end

        # Convert from Vector{Any} to concretely-typed vector of reshaped views.
        right_multiplication_buffer_blocks = [right_multiplication_buffer_blocks...]

        return new{Nvar,Tf,Ti,eltype(blocks),Tind,typeof(right_multiplication_buffer_blocks),typeof(dense_buffer_storage),typeof(vector_intermediate_buffer),Fsb,Fs}(
                   blocks, block_rowinds, block_row_ranges, bottom_block_rowinds,
                   bottom_block_vector_rowinds, block_colinds, block_col_ranges,
                   block_hypercube_positions, n_hypercube_positions,
                   right_multiplication_buffer_blocks, dense_buffer_storage,
                   vector_buffer_blocks_in, vector_buffer_blocks_out,
                   vector_intermediate_buffer, vector_range, block_synchronize_shared,
                   synchronize_shared)
    end
end

struct BlockCShared{Nvar,Tf,Ti,Tb,Tind,Trmbb,Tdb,Tbi,Tbuff,Tib,Fbs<:Function,Fs<:Function}
    block::Tb
    block_rowinds::NTuple{Nvar,Tind}
    block_row_ranges::NTuple{Nvar,UnitRange{Ti}}
    bottom_block_rowinds::NTuple{Nvar,Tind}
    bottom_block_vector_rowinds::Tind
    bottom_block_vector_colinds::Tind
    block_colinds::NTuple{Nvar,Tind}
    block_col_ranges::NTuple{Nvar,UnitRange{Ti}}
    block_hypercube_position::Ti
    n_hypercube_positions::Ti
    right_multiplication_buffer_block::Trmbb
    dense_buffer::Tdb
    vector_buffer_block_in::Tbi
    vector_buffer_block_out::Vector{Tf}
    vector_intermediate_buffer_local::Tbuff
    vector_intermediate_buffer::Tib
    vector_range::UnitRange{Ti}
    block_synchronize_shared::Fbs
    synchronize_shared::Fs

    # When multiplying a vector or a BlockAinvDotBShared matrix by a BlockCShared
    # block-structured C matrix, the output from each block can overlap as the outputs are
    # on the 'boundary points' of the grid, not the decoupled 'interior points'. To deal
    # with this, we first write the results from each block into an intermediate buffer
    # (`vector_intermediate_buffer`, or a buffer provided by MPISchurComplements), which
    # has several columns that collect different contributions to the result (where the
    # blocks written to a single column do not have overlapping results, unless they come
    # from the same process and therefore cannot conflict). The columns are summed to give
    # the final result.
    # To minimise memory bandwidth and computational time, we would like to minimise the
    # number of columns in the intermediate buffer.
    # When the number of processes is less than 2^d, where d is the number of dimensions,
    # we use one column per process.
    # When the number of processes is ≥2^d, we restrict the buffer to 2^d columns by
    # choosing the output column for each block in such a way that the blocks in a single
    # column never overlap. At any level of the solver, the grid is divided into blocks.
    # Blocks that are adjacent in any dimension share a face/edge/corner/etc. and
    # therefore have an overlap in 'C'. We group the blocks into 2x2x...
    # squares/cubes/hypercubes. Use 3d language for simplicity in the rest of this note,
    # but the same argument applies in any number of dimensions. A block in a certain
    # position within a cube cannot overlap with the blocks in the same position in
    # adjacent cubes, because they are fully separated by another block (cannot share even
    # an edge or a corner). Therefore if we put the outputs from all blocks in one
    # position in the cubes in one column, there are no overlaps (and so no conflicts
    # between outputs from different processes). The number of positions in a cube is 2^3
    # (or 2^d in d dimensions), so we need 2^d columns. We also need to keep track for
    # each block of which position it has in its cube, which translates to the column its
    # output should be written to in the intermediate buffer.
    # To find the block's position within its cube, get the block index in un-flattened
    # form. Transforming the index in each dimension to 0 for even values and 1 for odd
    # values, the binary number formed by the string of 0s and 1s (ordered in the same way
    # as the dimensions) is translated back to an integer to give the intermediate buffer
    # column.
    function BlockCShared{Tf}(block_rowinds_full::NTuple{Nvar,Tind},
                              bottom_block_rowinds_full::NTuple{Nvar,Tind},
                              bottom_block_vector_rowinds_full::NTuple{Nvar,Tind},
                              block_colinds::NTuple{Nvar,Tind},
                              matrix_template::Union{<:NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}},<:NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}},Nothing},
                              block_hypercube_position::Ti, n_hypercube_positions::Ti,
                              right_multiplication_buffer_storage::Vector{Tf},
                              dense_buffer_storage::Vector{Tf},
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti}, subgroup_i::Ti,
                              block_allocate_shared_float::Fa,
                              block_synchronize_shared::Fbs, block_comm_rank::Integer,
                              block_comm_size::Integer,
                              synchronize_shared::Fs) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti},Fa<:Function,Fbs<:Function,Fs<:Function}
        rows_per_proc = Tuple((length(ri) + block_comm_size - 1) ÷ block_comm_size
                              for ri ∈ block_rowinds_full)
        partial_row_ranges = Tuple(block_comm_rank*rpp+1:min((block_comm_rank+1)*rpp,length(ri))
                                   for (rpp, ri) ∈ zip(rows_per_proc, block_rowinds_full))
        block_rowinds = Tuple(ri[prr] for (prr, ri) ∈ zip(partial_row_ranges, block_rowinds_full))
        bottom_block_rowinds = Tuple(ri[prr] for (prr, ri) ∈ zip(partial_row_ranges, bottom_block_rowinds_full))
        block_row_range_offsets = vcat(0, cumsum(length(ri) for ri ∈ block_rowinds[1:end-1]))
        block_row_ranges = Tuple(offset .+ prr
                                 for (offset, prr) ∈ zip(block_row_range_offsets, partial_row_ranges))

        all_bottom_block_vector_rowinds_full = vcat((ri for ri ∈ bottom_block_vector_rowinds_full)...)
        vector_rows_per_proc = (length(all_bottom_block_vector_rowinds_full) + block_comm_size - 1) ÷ block_comm_size
        vector_partial_row_range = block_comm_rank*vector_rows_per_proc+1:min((block_comm_rank+1)*vector_rows_per_proc,length(all_bottom_block_vector_rowinds_full))
        bottom_block_vector_rowinds = all_bottom_block_vector_rowinds_full[vector_partial_row_range]
        bottom_block_vector_colinds = all_bottom_block_vector_rowinds_full

        col_ranges = Tuple(1:length(ci) for ci ∈ block_colinds)
        block_col_range_offsets = vcat(0, cumsum(length(ci) for ci ∈ block_colinds[1:end-1]))
        block_col_ranges = Tuple(offset .+ cr
                                 for (offset, cr) ∈ zip(block_col_range_offsets, col_ranges))

        nrow_full = sum(length(bi) for bi ∈ block_rowinds_full)
        nrow = sum(length(bi) for bi ∈ block_rowinds)
        ncol = sum(length(bi) for bi ∈ block_colinds)
        if matrix_template === nothing
            block = zeros(Tf, nrow, ncol)
            dense_buffer = nothing
        else
            block = get_partial_FixedSparseCSC_buffer(block_rowinds, block_colinds,
                                                      matrix_template, Tf)
            block.nzval .= 0.0
            if length(dense_buffer_storage) < nrow * ncol
                resize!(dense_buffer_storage, nrow * ncol)
            end
            dense_buffer = reshape(@view(dense_buffer_storage[1:nrow*ncol]), nrow, ncol)
        end
        right_multiplication_buffer_block_size = nrow * nrow_full
        if length(right_multiplication_buffer_storage) < right_multiplication_buffer_block_size
            resize!(right_multiplication_buffer_storage,
                    right_multiplication_buffer_block_size)
        end
        right_multiplication_buffer_block =
            reshape(@view(right_multiplication_buffer_storage[1:right_multiplication_buffer_block_size]),
                    nrow, nrow_full)
        vector_buffer_block_in = block_allocate_shared_float(ncol)
        vector_buffer_block_out = zeros(Tf, nrow)
        if subgroup_i < 0
            vector_intermediate_buffer_local = zeros(Tf, 0)
        else
            vector_intermediate_buffer_local = @view vector_intermediate_buffer[block_hypercube_position,:]
        end
        return new{Nvar,Tf,Ti,typeof(block),Tind,typeof(right_multiplication_buffer_block),typeof(dense_buffer),typeof(vector_buffer_block_in),typeof(vector_intermediate_buffer_local),typeof(vector_intermediate_buffer),Fbs,Fs}(
                   block, block_rowinds, block_row_ranges, bottom_block_rowinds,
                   bottom_block_vector_rowinds, bottom_block_vector_colinds,
                   block_colinds, block_col_ranges, block_hypercube_position,
                   n_hypercube_positions, right_multiplication_buffer_block, dense_buffer,
                   vector_buffer_block_in, vector_buffer_block_out,
                   vector_intermediate_buffer_local, vector_intermediate_buffer,
                   vector_range, block_synchronize_shared, synchronize_shared)
    end
end

# copy_C_submatrix!() is identical to copy_B_submatrix!(), but keep as a separate function
# instead of having a single implementation for both in case we want to experiment with
# using a transposed representation of the C blocks at some point.
function copy_C_submatrix!(block_C::BlockCSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractMatrix}}) where Nvar
    @inbounds begin
        blocks = block_C.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = block_C.block_rowinds
        block_row_ranges = block_C.block_row_ranges
        block_colinds = block_C.block_colinds
        block_col_ranges = block_C.block_col_ranges
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
function copy_C_submatrix!(block_C::BlockCSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}}) where Nvar
    @inbounds begin
        blocks = block_C.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = block_C.block_rowinds
        block_row_ranges = block_C.block_row_ranges
        block_colinds = block_C.block_colinds
        block_col_ranges = block_C.block_col_ranges
        if eltype(blocks) <: Matrix
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                        (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval
                    first_irow = first(rr)
                    last_irow = last(rr)
                    first_row = first(ri)
                    for (j1, j2) ∈ zip(cr, ci)
                        first_i = full_A_colptr[j2]
                        last_i = full_A_colptr[j2+1] - 1
                        col_rv = @view full_A_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                        i1 = first_irow
                        while i1 ≤ last_irow
                            full_A_row = full_A_rowval[flat_i]
                            block_global_row = ri[i1]
                            if full_A_row == block_global_row
                                block[i1,j1] = full_A_nzval[flat_i]
                                i1 += 1
                                flat_i += 1
                            elseif full_A_row > block_global_row
                                i1 += 1
                            else
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                        end
                    end
                end
            end
        else
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                        (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval
                    first_row = first(ri)
                    block_colptr = block.colptr
                    block_rowval = block.rowval
                    block_nzval = block.nzval
                    first_row_i = first(rr)
                    last_row_i = last(rr)
                    for (j1, j2) ∈ zip(cr, ci)
                        first_i = full_A_colptr[j2]
                        last_i = full_A_colptr[j2+1] - 1
                        col_rv = @view full_A_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                        first_block_i = block_colptr[j1]
                        last_block_i = block_colptr[j1+1] - 1
                        block_i = max(searchsortedlast(@view(block_rowval[first_block_i:last_block_i]), first_row_i) - 1, 1) + first_block_i - 1
                        while block_i ≤ last_block_i
                            full_A_row = full_A_rowval[flat_i]
                            block_global_row = ri[block_rowval[block_i]]
                            if block_global_row > last_row_i
                                break
                            end
                            if full_A_row == block_global_row
                                block_nzval[block_i] = full_A_nzval[flat_i]
                                block_i += 1
                                flat_i += 1
                            elseif full_A_row > block_global_row
                                block_i += 1
                            else
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::BlockCSerial,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}}) where Nvar
    @inbounds begin
        blocks = block_C.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = block_C.block_rowinds
        block_row_ranges = block_C.block_row_ranges
        block_colinds = block_C.block_colinds
        block_col_ranges = block_C.block_col_ranges
        if eltype(blocks) <: Matrix
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                        (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval_list = A_variable_block.rowval_list
                    full_A_nzval = A_variable_block.nzval
                    first_irow = first(rr)
                    last_irow = last(rr)
                    first_row = first(ri)
                    for (j1, j2) ∈ zip(cr, ci)
                        first_i = full_A_colptr[j2]
                        last_i = full_A_colptr[j2+1] - 1
                        col_rv = full_A_rowval_list[j2]
                        last_row_i = length(col_rv)
                        row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                        i1 = first_irow
                        while i1 ≤ last_irow
                            full_A_row = col_rv[row_i]
                            block_global_row = ri[i1]
                            if full_A_row == block_global_row
                                block[i1,j1] = full_A_nzval[row_i+first_i-1]
                                i1 += 1
                                row_i += 1
                            elseif full_A_row > block_global_row
                                i1 += 1
                            else
                                row_i += 1
                            end
                            if row_i > last_row_i
                                break
                            end
                        end
                    end
                end
            end
        else
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                block_colptr = block.colptr
                block_rowval = block.rowval
                block_nzval = block.nzval
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges),
                        (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval_list = A_variable_block.rowval_list
                    full_A_nzval = A_variable_block.nzval
                    first_irow = first(rr)
                    first_row = first(ri)
                    last_row = last(ri)
                    for (j1, j2) ∈ zip(cr, ci)
                        first_i = full_A_colptr[j2]
                        col_rv = full_A_rowval_list[j2]
                        last_row_i = length(col_rv)
                        row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                        first_block_i = block_colptr[j1]
                        last_block_i = block_colptr[j1+1] - 1
                        block_i = max(searchsortedlast(@view(block_rowval[first_block_i:last_block_i]), first_irow) - 1, 1) + first_block_i - 1
                        while block_i ≤ last_block_i
                            full_A_row = col_rv[row_i]
                            block_global_row = ri[block_rowval[block_i]]
                            if block_global_row > last_row
                                break
                            end
                            if full_A_row == block_global_row
                                block_nzval[block_i] = full_A_nzval[row_i+first_i-1]
                                block_i += 1
                                row_i += 1
                            elseif full_A_row > block_global_row
                                block_i += 1
                            else
                                row_i += 1
                            end
                            if row_i > last_row_i
                                break
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::BlockCShared,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}}) where Nvar
    @inbounds begin
        block = block_C.block
        if length(block) == 0
            # Nothing to do.
            return nothing
        end
        block_rowinds = block_C.block_rowinds
        block_row_ranges = block_C.block_row_ranges
        block_colinds = block_C.block_colinds
        block_col_ranges = block_C.block_col_ranges

        if isa(block, Matrix)
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges),
                    (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval = A_variable_block.rowval
                full_A_nzval = A_variable_block.nzval
                first_irow = first(rr)
                last_irow = last(rr)
                first_row = first(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    last_i = full_A_colptr[j2+1] - 1
                    col_rv = @view full_A_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                    i1 = first_irow
                    while i1 ≤ last_irow
                        full_A_row = full_A_rowval[flat_i]
                        block_global_row = block_rowinds[i1]
                        if full_A_row == block_global_row
                            block[i1,j1] = full_A_nzval[flat_i]
                            i1 += 1
                            flat_i += 1
                        elseif full_A_row > block_global_row
                            i1 += 1
                        else
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                    end
                end
            end
        else
            block_colptr = block.colptr
            block_rowval = block.rowval
            block_nzval = block.nzval
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges),
                    (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval = A_variable_block.rowval
                full_A_nzval = A_variable_block.nzval
                first_irow = first(rr)
                first_row = first(ri)
                last_row = last(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    last_i = full_A_colptr[j2+1] - 1
                    col_rv = @view full_A_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                    first_block_i = block_colptr[j1]
                    last_block_i = block_colptr[j1+1] - 1
                    block_i = max(searchsortedlast(@view(block_rowval[first_block_i:last_block_i]), first_row_i) - 1, 1) + first_block_i - 1
                    while block_i ≤ last_block_i
                        full_A_row = full_A_rowval[flat_i]
                        block_global_row = block_rowinds[block_rowval[block_i]]
                        if block_global_row > last_row
                            break
                        end
                        if full_A_row == block_global_row
                            block_nzval[block_i] = full_A_nzval[flat_i]
                            block_i += 1
                            flat_i += 1
                        elseif full_A_row > block_global_row
                            block_i += 1
                        else
                            flat_i += 1
                        end
                        if flat_i > last_i
                            break
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::BlockCShared,
                           full_A::NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}}) where Nvar
    @inbounds begin
        block = block_C.block
        if length(block) == 0
            # Nothing to do.
            return nothing
        end
        block_rowinds = block_C.block_rowinds
        block_colinds = block_C.block_colinds
        block_col_ranges = block_C.block_col_ranges
        if isa(block, Matrix)
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges),
                    (vrow, ri) ∈ zip(1:Nvar, block_rowinds)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval_list = A_variable_block.rowval_list
                full_A_nzval = A_variable_block.nzval
                first_row = first(ri)
                nrow = length(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    col_rv = full_A_rowval_list[j2]
                    last_row = length(col_rv)
                    row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                    i1 = 1
                    while i1 ≤ nrow
                        full_A_row = col_rv[row_i]
                        block_global_row = ri[i1]
                        if full_A_row == block_global_row
                            block[i1,j1] = full_A_nzval[row_i+first_i-1]
                            i1 += 1
                            row_i += 1
                        elseif full_A_row > block_global_row
                            i1 += 1
                        else
                            row_i += 1
                        end
                        if row_i > last_row
                            break
                        end
                    end
                end
            end
        else
            block_colptr = block.colptr
            block_rowval = block.rowval
            block_nzval = block.nzval
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges),
                    (vrow, ri) ∈ zip(1:Nvar, block_rowinds)
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval_list = A_variable_block.rowval_list
                full_A_nzval = A_variable_block.nzval
                first_row = first(ri)
                last_row = last(ri)
                for (j1, j2) ∈ zip(cr, ci)
                    first_i = full_A_colptr[j2]
                    col_rv = full_A_rowval_list[j2]
                    last_row_i = length(col_rv)
                    row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                    first_block_i = block_colptr[j1]
                    last_block_i = block_colptr[j1+1] - 1
                    block_i = first_block_i
                    while block_i ≤ last_block_i
                        full_A_row = col_rv[row_i]
                        block_global_row = ri[block_rowval[block_i]]
                        if block_global_row > last_row
                            break
                        end
                        if full_A_row == block_global_row
                            block_nzval[block_i] = full_A_nzval[row_i+first_i-1]
                            block_i += 1
                            row_i += 1
                        elseif full_A_row > block_global_row
                            block_i += 1
                        else
                            row_i += 1
                        end
                        if row_i > last_row_i
                            break
                        end
                    end
                end
            end
        end
        return nothing
    end
end

function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCSerial,
                               Ainv_dot_u)

    @inbounds begin
        blocks = C.blocks
        vector_range = C.vector_range
        vector_intermediate_buffer = C.vector_intermediate_buffer
        synchronize_shared = C.synchronize_shared

        # The rows are labelled by block_hypercube_position, so there are no overlaps, and
        # we can directly set entries, instead of adding to them, and so do not need to
        # zero-initialise the intermediate buffer.
        block_hypercube_positions = C.block_hypercube_positions
        if length(blocks) > 0
            for (vec_buffer_out, rowinds, block, bhp, Aiu_block) ∈
                    zip(C.vector_buffer_blocks_out, C.bottom_block_vector_rowinds, blocks,
                        block_hypercube_positions, Ainv_dot_u)
                vector_intermediate_buffer_local = @view vector_intermediate_buffer[bhp,:]
                mul!(vec_buffer_out, block, Aiu_block)
                for (i2, i1) ∈ enumerate(rowinds)
                    vector_intermediate_buffer_local[i1] = -vec_buffer_out[i2]
                end
            end
        end

        synchronize_shared()

        # Sum contributions from all processes into the output.
        if !isempty(vector_range)
            @views sum!(C_dot_Ainv_dot_u[vector_range]',
                        vector_intermediate_buffer[:,vector_range])
        end

        return nothing
    end
end
function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCShared,
                               Ainv_dot_u)

    @inbounds begin
        block = C.block
        vector_range = C.vector_range
        vector_intermediate_buffer = C.vector_intermediate_buffer
        vector_intermediate_buffer_local = C.vector_intermediate_buffer_local
        vec_buffer_block_out = C.vector_buffer_block_out
        bottom_block_vector_rowinds = C.bottom_block_vector_rowinds
        synchronize_shared = C.synchronize_shared

        # The rows are labelled by block_hypercube_position, so there are no overlaps, and
        # we can directly set entries, instead of adding to them, and so do not need to
        # zero-initialise the output buffer.
        mul!(vec_buffer_block_out, block, Ainv_dot_u)
        for (i2, i1) ∈ enumerate(bottom_block_vector_rowinds)
            vector_intermediate_buffer_local[i1] = -vec_buffer_block_out[i2]
        end

        synchronize_shared()

        # Sum contributions from all processes into the output.
        if !isempty(vector_range)
            @views sum!(C_dot_Ainv_dot_u[vector_range]',
                        vector_intermediate_buffer[:,vector_range])
        end

        return nothing
    end
end
