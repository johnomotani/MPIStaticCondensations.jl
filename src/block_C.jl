struct BlockCSerial{Tb,Trange,Tf,Ti,Trmbb,Tib,Fsb<:Function,Fs<:Function}
    blocks::Vector{Tb}
    block_rowinds::Vector{Trange}
    bottom_block_rowinds::Vector{Trange}
    block_colinds::Vector{Trange}
    block_hypercube_positions::Vector{Ti}
    output_buffer_ncopies::Ti
    right_multiplication_buffer_blocks::Trmbb
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}
    vector_intermediate_buffer::Tib
    vector_range::UnitRange{Ti}
    block_synchronize_shared::Fsb
    synchronize_shared::Fs

    function BlockCSerial{Tf}(block_rowinds::Vector{<:AbstractVector{Ti}},
                              bottom_block_rowinds::Vector{<:AbstractVector{Ti}},
                              block_colinds::Vector{<:AbstractVector{Ti}},
                              local_top_vector_indices::Vector{Ti},
                              local_bottom_vector_indices::Vector{Ti},
                              matrix_template::Union{AbstractSparseMatrixCSC,Nothing},
                              block_hypercube_positions::Vector{Ti},
                              output_buffer_ncopies::Ti,
                              right_multiplication_buffer_storage::Vector{Tf},
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti},
                              block_synchronize_shared::Fsb,
                              synchronize_shared::Fs) where {Tf,Ti,Fsb<:Function,Fs<:Function}
        non_empty_blocks = [!isempty(ri) && !isempty(ci)
                            for (ri, ci) ∈ zip(block_rowinds, block_colinds)]
        block_rowinds = block_rowinds[non_empty_blocks]
        bottom_block_rowinds = bottom_block_rowinds[non_empty_blocks]
        block_colinds = block_colinds[non_empty_blocks]
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

        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = length(ri)
            ncol = length(ci)
            if matrix_template === nothing
                push!(blocks, zeros(Tf, nrow, ncol))
            else
                b = get_partial_FixedSparseCSC_buffer(local_bottom_vector_indices[ri],
                                                      local_top_vector_indices[ci],
                                                      matrix_template, Tf)
                push!(blocks, b)
            end
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
            right_multiplication_buffer_size = nrow^2
            if length(right_multiplication_buffer_storage) < right_multiplication_buffer_size
                resize!(right_multiplication_buffer_storage,
                        right_multiplication_buffer_size)
            end
            push!(right_multiplication_buffer_blocks,
                  reshape(@view(right_multiplication_buffer_storage[1:right_multiplication_buffer_size]),
                          nrow, nrow))
        end

        # Convert from Vector{Any} to concretely-typed vector of reshaped views.
        right_multiplication_buffer_blocks = [right_multiplication_buffer_blocks...]

        return new{eltype(blocks),eltype(block_rowinds),Tf,Ti,typeof(right_multiplication_buffer_blocks),typeof(vector_intermediate_buffer),Fsb,Fs}(
                   blocks, block_rowinds, bottom_block_rowinds, block_colinds,
                   block_hypercube_positions, output_buffer_ncopies,
                   right_multiplication_buffer_blocks, vector_buffer_blocks_in,
                   vector_buffer_blocks_out, vector_intermediate_buffer, vector_range,
                   block_synchronize_shared, synchronize_shared)
    end
end

struct BlockCShared{Tb,Trange,Tf,Ti,Trmbb,Tbi,Tbuff,Tib,Fbs<:Function,Fs<:Function}
    block::Tb
    block_rowinds::Trange
    bottom_block_rowinds::Trange
    block_colinds::Trange
    partial_block_colinds::Vector{Ti}
    partial_col_range::UnitRange{Ti}
    block_hypercube_position::Ti
    output_buffer_ncopies::Ti
    right_multiplication_buffer_block::Trmbb
    block_right_multiplication_output_colinds::Vector{Ti}
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
    function BlockCShared{Tf}(block_rowinds_full::AbstractVector{Ti},
                              bottom_block_rowinds_full::AbstractVector{Ti},
                              partial_row_range::UnitRange{Ti}, block_colinds::AbstractVector{Ti},
                              local_top_vector_indices::AbstractVector{Ti},
                              local_bottom_vector_indices::AbstractVector{Ti},
                              matrix_template::Union{AbstractSparseMatrixCSC,Nothing},
                              block_hypercube_position::Ti,
                              output_buffer_ncopies::Ti,
                              right_multiplication_buffer_storage::Vector{Tf},
                              vector_intermediate_buffer::AbstractMatrix{Tf},
                              vector_range::UnitRange{Ti},
                              subgroup_i::Ti, block_allocate_shared_float::Fa,
                              block_synchronize_shared::Fbs, block_comm_rank::Integer,
                              block_comm_size::Integer,
                              synchronize_shared::Fs) where {Tf,Ti,Fa<:Function,Fbs<:Function,Fs<:Function}
        block_rowinds = block_rowinds_full[partial_row_range]
        bottom_block_rowinds = bottom_block_rowinds_full[partial_row_range]
        nrow_full = length(block_rowinds_full)
        nrow = length(block_rowinds)
        ncol = length(block_colinds)
        if matrix_template === nothing
            block = zeros(Tf, nrow, ncol)
        else
            block = get_partial_FixedSparseCSC_buffer(local_bottom_vector_indices[block_rowinds],
                                                      local_top_vector_indices[block_colinds],
                                                      matrix_template, Tf)
        end
        cols_per_proc = (ncol + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,ncol)
        partial_block_colinds = block_colinds[partial_col_range]
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
        return new{typeof(block),typeof(block_rowinds),Tf,Ti,typeof(right_multiplication_buffer_block),typeof(vector_buffer_block_in),typeof(vector_intermediate_buffer_local),typeof(vector_intermediate_buffer),Fbs,Fs}(
                   block, block_rowinds, bottom_block_rowinds, block_colinds,
                   partial_block_colinds, partial_col_range, block_hypercube_position,
                   output_buffer_ncopies, right_multiplication_buffer_block,
                   bottom_block_rowinds_full, vector_buffer_block_in,
                   vector_buffer_block_out, vector_intermediate_buffer_local,
                   vector_intermediate_buffer, vector_range, block_synchronize_shared,
                   synchronize_shared)
    end
end

function get_C_hypercube_position(iblock)
    return sum(((i - 1) % 2) * 2^(d-1) for (d, i) ∈ enumerate(iblock)) + 1
end

# copy_C_submatrix!() is identical to copy_B_submatrix!(), but keep as a separate function
# instead of having a single implementation for both in case we want to experiment with
# using a transposed representation of the C blocks at some point.
function copy_C_submatrix!(block_C::BlockCSerial, full_A::AbstractMatrix)
    @inbounds begin
        blocks = block_C.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = block_C.block_rowinds
        block_colinds = block_C.block_colinds
        for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
            for (j1, j2) ∈ enumerate(colinds), (i1, i2) ∈ enumerate(rowinds)
                block[i1,j1] = full_A[i2,j2]
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::BlockCSerial, full_A::AbstractSparseMatrixCSC)
    @inbounds begin
        blocks = block_C.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = block_C.block_rowinds
        block_colinds = block_C.block_colinds
        full_A_colptr = full_A.colptr
        full_A_rowval = full_A.rowval
        full_A_nzval = full_A.nzval
        if eltype(blocks) <: Matrix
            for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
                block_nrow = length(rowinds)
                first_row = first(rowinds)
                for (j1, j2) ∈ enumerate(colinds)
                    first_i = full_A_colptr[j2]
                    last_i = full_A_colptr[j2+1] - 1
                    col_rv = @view full_A_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                    i1 = 1
                    while i1 ≤ block_nrow
                        full_A_row = full_A_rowval[flat_i]
                        block_global_row = rowinds[i1]
                        if full_A_row == block_global_row
                            block[i1,j1] = full_A_nzval[flat_i]
                            i1 += 1
                            flat_i += 1
                        elseif full_A_row > block_global_row
                            block[i1,j1] = 0.0
                            i1 += 1
                        else
                            flat_i += 1
                        end
                        if flat_i > last_i
                            block[i1:end,j1] .= 0.0
                            break
                        end
                    end
                end
            end
        else
            for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
                block_nrow = length(rowinds)
                first_row = first(rowinds)
                block_colptr = block.colptr
                block_rowval = block.rowval
                block_nzval = block.nzval
                for (j1, j2) ∈ enumerate(colinds)
                    first_i = full_A_colptr[j2]
                    last_i = full_A_colptr[j2+1] - 1
                    col_rv = @view full_A_rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                    block_i = block_colptr[j1]
                    block_last_i = block_colptr[j1+1] - 1
                    while block_i ≤ block_last_i
                        full_A_row = full_A_rowval[flat_i]
                        block_global_row = rowinds[block_rowval[block_i]]
                        if full_A_row == block_global_row
                            block_nzval[block_i] = full_A_nzval[flat_i]
                            block_i += 1
                            flat_i += 1
                        elseif full_A_row > block_global_row
                            block_nzval[block_i] = 0.0
                            block_i += 1
                        else
                            flat_i += 1
                        end
                        if flat_i > last_i
                            block_nzval[block_i:block_last_i] .= 0.0
                            break
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::BlockCShared, full_A::AbstractSparseMatrixCSC)
    @inbounds begin
        block_rowinds = block_C.block_rowinds
        block_colinds = block_C.block_colinds
        if isempty(block_rowinds) || isempty(block_colinds)
            # Nothing to do.
            return nothing
        end
        block = block_C.block
        full_A_colptr = full_A.colptr
        full_A_rowval = full_A.rowval
        full_A_nzval = full_A.nzval

        block_nrow = length(block_rowinds)
        first_row = first(block_rowinds)
        if isa(block, Matrix)
            for (j1, j2) ∈ enumerate(block_colinds)
                first_i = full_A_colptr[j2]
                last_i = full_A_colptr[j2+1] - 1
                col_rv = @view full_A_rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                i1 = 1
                while i1 ≤ block_nrow
                    full_A_row = full_A_rowval[flat_i]
                    block_global_row = block_rowinds[i1]
                    if full_A_row == block_global_row
                        block[i1,j1] = full_A_nzval[flat_i]
                        i1 += 1
                        flat_i += 1
                    elseif full_A_row > block_global_row
                        block[i1,j1] = 0.0
                        i1 += 1
                    else
                        flat_i += 1
                    end
                    if flat_i > last_i
                        block[i1:end,j1] .= 0.0
                        break
                    end
                end
            end
        else
            block_colptr = block.colptr
            block_rowval = block.rowval
            block_nzval = block.nzval
            for (j1, j2) ∈ enumerate(block_colinds)
                first_i = full_A_colptr[j2]
                last_i = full_A_colptr[j2+1] - 1
                col_rv = @view full_A_rowval[first_i:last_i]
                flat_i = max(searchsortedlast(col_rv, first_row)-1,1) + first_i - 1
                block_i = block_colptr[j1]
                block_last_i = block_colptr[j1+1] - 1
                while block_i ≤ block_last_i
                    full_A_row = full_A_rowval[flat_i]
                    block_global_row = block_rowinds[block_rowval[block_i]]
                    if full_A_row == block_global_row
                        block_nzval[block_i] = full_A_nzval[flat_i]
                        block_i += 1
                        flat_i += 1
                    elseif full_A_row > block_global_row
                        block_nzval[block_i] = 0.0
                        block_i += 1
                    else
                        flat_i += 1
                    end
                    if flat_i > last_i
                        block_nzval[block_i:block_last_i] .= 0.0
                        break
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
                    zip(C.vector_buffer_blocks_out, C.bottom_block_rowinds, blocks,
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
        block_rowinds = C.block_rowinds
        synchronize_shared = C.synchronize_shared

        # The rows are labelled by block_hypercube_position, so there are no overlaps, and
        # we can directly set entries, instead of adding to them, and so do not need to
        # zero-initialise the output buffer.
        mul!(vec_buffer_block_out, block, Ainv_dot_u)
        for (i2, i1) ∈ enumerate(block_rowinds)
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
