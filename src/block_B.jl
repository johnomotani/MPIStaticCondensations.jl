struct BlockAinvDotBSerial{Tf,Ti,Tb,Trange}
    blocks::Vector{Tb}
    block_rowinds::Vector{Trange}
    block_colinds::Vector{Trange}
    bottom_block_colinds::Vector{Trange}
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Vector{Tf}}

    function BlockAinvDotBSerial{Tf}(block_rowinds::Vector{<:AbstractVector{Ti}},
                                     block_colinds::Vector{<:AbstractVector{Ti}},
                                     bottom_block_colinds::Vector{<:AbstractVector{Ti}}) where {Tf,Ti}
        non_empty_blocks = [!isempty(ri) && !isempty(ci)
                            for (ri, ci) ∈ zip(block_rowinds, block_colinds)]
        block_rowinds = block_rowinds[non_empty_blocks]
        block_colinds = block_colinds[non_empty_blocks]
        bottom_block_colinds = bottom_block_colinds[non_empty_blocks]
        blocks = Matrix{Tf}[]
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out = Vector{Tf}[]
        for (ri, ci) ∈ zip(block_rowinds, block_colinds)
            nrow = length(ri)
            ncol = length(ci)
            push!(blocks, zeros(Tf, nrow, ncol))
            push!(vector_buffer_blocks_in, zeros(Tf, ncol))
            push!(vector_buffer_blocks_out, zeros(Tf, nrow))
        end
        return new{Tf,Ti,eltype(blocks),eltype(block_rowinds)}(
                   blocks, block_rowinds, block_colinds, bottom_block_colinds,
                   vector_buffer_blocks_in, vector_buffer_blocks_out)
    end
end

# This version has a single block, and operations are parallelised using shared-memory
# MPI.
struct BlockAinvDotBShared{Tf,Ti,Tb,Trange,Tsync}
    block::Tb
    partial_block::Matrix{Tf}
    block_rowinds::Trange
    block_partial_rowinds::Vector{Ti}
    block_colinds::Trange
    block_partial_colinds::Vector{Ti}
    bottom_block_colinds::Trange
    bottom_block_partial_colinds::Vector{Ti}
    buffer::Tb
    partial_col_range::UnitRange{Ti}
    partial_row_range::UnitRange{Ti}
    vector_buffer_block_in::Vector{Tf}
    vector_buffer_block_out::Vector{Tf}
    synchronize_shared::Tsync

    function BlockAinvDotBShared{Tf}(block_rowinds::AbstractVector{Ti},
                                     block_colinds::AbstractVector{Ti},
                                     bottom_block_colinds::AbstractVector{Ti},
                                     block_comm_rank::Integer, block_comm_size::Integer,
                                     allocate_shared_float::Fa,
                                     synchronize_shared::Fs) where {Tf,Ti,Fa,Fs}
        if isempty(block_rowinds) || isempty(block_colinds)
            return new{Tf,Ti,Matrix{Tf},typeof(block_colinds),Fs}(
                       zeros(Tf, 0, 0), zeros(Tf, 0, 0), block_rowinds, zeros(Ti, 0),
                       block_colinds, zeros(Ti, 0), bottom_block_colinds, zeros(Ti, 0),
                       zeros(Tf, 0, 0), 1:0, 1:0, zeros(Tf, 0), zeros(Tf, 0),
                       synchronize_shared)
        end

        nrow = length(block_rowinds)
        ncol = length(block_colinds)
        block = allocate_shared_float(length(block_rowinds), length(block_colinds))
        buffer = allocate_shared_float(length(block_rowinds), length(block_colinds))
        cols_per_proc = (ncol + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,ncol)
        block_partial_colinds = block_colinds[partial_col_range]
        bottom_block_partial_colinds = bottom_block_colinds[partial_col_range]
        rows_per_proc = (nrow + block_comm_size - 1) ÷ block_comm_size
        partial_row_range = block_comm_rank*rows_per_proc+1:min((block_comm_rank+1)*rows_per_proc,nrow)
        partial_nrow = length(partial_row_range)
        block_partial_rowinds = block_rowinds[partial_row_range]
        vector_buffer_block_in = allocate_shared_float(ncol)
        vector_buffer_block_out = zeros(Tf, partial_nrow)
        partial_block = zeros(Tf, partial_nrow, ncol)

        block[:,partial_col_range] .= 0.0
        vector_buffer_block_in[partial_col_range] .= 0.0
        vector_buffer_block_out .= 0.0

        return new{Tf,Ti,typeof(block),typeof(block_rowinds),Fs}(
                   block, partial_block, block_rowinds, block_partial_rowinds,
                   block_colinds, block_partial_colinds, bottom_block_partial_colinds,
                   buffer, partial_col_range, partial_row_range, vector_buffer_block_in,
                   vector_buffer_block_out, synchronize_shared)
    end
end

function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial, full_A::AbstractMatrix)
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = Ainv_dot_B.block_rowinds
        block_colinds = Ainv_dot_B.block_colinds
        for (rowinds, colinds, block) ∈ zip(block_rowinds, block_colinds, blocks)
            for (j1, j2) ∈ enumerate(colinds), (i1, i2) ∈ enumerate(rowinds)
                block[i1,j1] = full_A[i2,j2]
            end
        end
        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBSerial,
                           full_A::AbstractSparseMatrixCSC)
    @inbounds begin
        blocks = Ainv_dot_B.blocks
        if length(blocks) == 0
            # Nothing to do.
            return nothing
        end

        block_rowinds = Ainv_dot_B.block_rowinds
        block_colinds = Ainv_dot_B.block_colinds
        full_A_colptr = full_A.colptr
        full_A_rowval = full_A.rowval
        full_A_nzval = full_A.nzval
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
        return nothing
    end
end
function copy_B_submatrix!(Ainv_dot_B::BlockAinvDotBShared,
                           full_A::AbstractSparseMatrixCSC)
    @inbounds begin
        block_rowinds = Ainv_dot_B.block_rowinds
        block_colinds = Ainv_dot_B.block_colinds
        if isempty(block_rowinds) || isempty(block_colinds)
            # Nothing to do.
            return nothing
        end
        block = Ainv_dot_B.block
        partial_col_range = Ainv_dot_B.partial_col_range
        full_A_colptr = full_A.colptr
        full_A_rowval = full_A.rowval
        full_A_nzval = full_A.nzval

        block_nrow = length(block_rowinds)
        first_row = first(block_rowinds)
        for j1 ∈ partial_col_range
            j2 = block_colinds[j1]
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

        return nothing
    end
end

# Note that combining all the contributions to C_dot_Ainv_dot_B from different processes
# is taken care of by MPISchurComplements.
function mul_C_Ainv_dot_B!(schur_complement::BlockS, C::BlockCSerial,
                           Ainv_dot_B::BlockAinvDotBSerial)
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        C_blocks = C.blocks
        sc_matrix = schur_complement.matrix
        C_dot_Ainv_dot_B = schur_complement.C_dot_Ainv_dot_B
        synchronize_shared = C.synchronize_shared

        if length(C_blocks) > 0
            mul_blocks = C.right_multiplication_buffer_blocks
            Ainv_dot_B_blocks = Ainv_dot_B.blocks
            block_output_inds = C.bottom_block_rowinds

            colptr = sc_matrix.colptr
            rowval = sc_matrix.rowval

            # The rows are labelled by block_hypercube_position, so there are no overlaps,
            # and we can directly set entries, instead of adding to them, and so do not
            # need to zero-initialise the output buffer.
            block_hypercube_positions = C.block_hypercube_positions
            for (mb, Cb, AiBb, output_inds, bhp) ∈ zip(mul_blocks, C_blocks,
                                                       Ainv_dot_B_blocks,
                                                       block_output_inds,
                                                       block_hypercube_positions)
                nzval = @view C_dot_Ainv_dot_B[bhp,:]

                mul!(mb, Cb, AiBb, -1.0, 0.0)

                # Copy result from mb into the sparse output buffer C_dot_Ainv_dot_B.
                first_row = first(output_inds)
                nrows = length(output_inds)
                for (j, col) ∈ enumerate(output_inds)
                    first_i = colptr[col]
                    last_i = colptr[col+1] - 1
                    col_rv = @view rowval[first_i:last_i]
                    flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
                    i = 1
                    while flat_i ≤ last_i && i ≤ nrows
                        if rowval[flat_i] == output_inds[i]
                            nzval[flat_i] = mb[i,j]
                            flat_i += 1
                            i += 1
                        else
                            # rowval[flat_i] must be less than output_inds[i]
                            flat_i += 1
                        end
                    end
                end
            end
        end

        synchronize_shared()

        flat_range_partial = schur_complement.flat_range_partial
        if !isempty(flat_range_partial)
            @views sum!(sc_matrix.nzval[flat_range_partial]', C_dot_Ainv_dot_B[:,flat_range_partial])
        end

        return nothing
    end
end
function mul_C_Ainv_dot_B!(schur_complement::BlockS, C::BlockCShared,
                           Ainv_dot_B::BlockAinvDotBShared)
    # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
    # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
    # the local chunks, then do a sum-reduce to get the final result. The
    # `schur_complement.matrix` buffer is full size on every rank.
    @inbounds begin
        sc_matrix = schur_complement.matrix
        C_dot_Ainv_dot_B = schur_complement.C_dot_Ainv_dot_B
        synchronize_shared = C.synchronize_shared
        C_block = C.block
        mul_block = C.right_multiplication_buffer_block
        block_output_inds = C.block_rowinds
        block_output_colinds = C.block_right_multiplication_output_colinds
        Ainv_dot_B_block = Ainv_dot_B.block

        if isempty(block_output_inds) || isempty(block_output_colinds)
            return nothing
        end

        colptr = sc_matrix.colptr
        rowval = sc_matrix.rowval
        nzval = @view C_dot_Ainv_dot_B[C.block_hypercube_position,:]

        # Output buffer columns are divided by 'hypercube position' so there are no
        # overlaps, and we can directly set entries, instead of adding to them, and so do
        # not need to zero-initialise the output buffer.
        mul!(mul_block, C_block, Ainv_dot_B_block, -1.0, 0.0)

        # Copy result from mul_block into the sparse output buffer C_dot_Ainv_dot_B.
        first_row = first(block_output_inds)
        nrows = length(block_output_inds)
        for (j, col) ∈ enumerate(block_output_colinds)
            first_i = colptr[col]
            last_i = colptr[col+1] - 1
            col_rv = @view rowval[first_i:last_i]
            flat_i = max(searchsortedlast(col_rv, first_row) - 1, 1) + first_i - 1
            i = 1
            while flat_i ≤ last_i && i ≤ nrows
                if rowval[flat_i] == block_output_inds[i]
                    nzval[flat_i] = mul_block[i,j]
                    flat_i += 1
                    i += 1
                else
                    # rowval[flat_i] must be less than block_output_inds[i].
                    flat_i += 1
                end
            end
        end

        synchronize_shared()

        flat_range = schur_complement.flat_range
        if !isempty(flat_range)
            @views sum!(sc_matrix.nzval[flat_range]', C_dot_Ainv_dot_B[:,flat_range])
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
                    Ainv_dot_B.vector_buffer_blocks_out, Ainv_dot_B.block_rowinds,
                    Ainv_dot_B.bottom_block_colinds, blocks, Ainv_dot_u)
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
        block_partial_rowinds = Ainv_dot_B.block_partial_rowinds
        bbottom_lock_partial_colinds = Ainv_dot_B.bbottom_lock_partial_colinds
        partial_col_range = Ainv_dot_B.partial_bottom_col_range
        synchronize_shared = Ainv_dot_B.synchronize_shared

        for (i1, i2) ∈ zip(partial_col_range, block_partial_bottom_colinds)
            vector_buffer_block_in[i1] = y[i2]
        end
        synchronize_shared()

        mul!(vector_buffer_block_out, partial_block, vector_buffer_block_in)
        for (i2, i1) ∈ enumerate(block_partial_rowinds)
            x[i1] = Ainv_dot_u[i2] - vector_buffer_block_out[i2]
        end
        return nothing
    end
end
