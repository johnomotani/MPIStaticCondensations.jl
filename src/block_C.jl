function get_this_proc_bottom_vector_entry_contribution_indices(
             bottom_block_vector_rowinds, vector_buffer_blocks_out_offsets::Vector{Ti},
             bottom_block_size::Ti, shared_comm, shared_comm_rank,
             shared_comm_size) where Ti
    n_per_proc = (bottom_block_size + shared_comm_size - 1) ÷ shared_comm_size
    all_procs_bottom_vector_entries = [r*n_per_proc+1:min((r+1)*n_per_proc,bottom_block_size)
                                       for r ∈ 0:shared_comm_size-1]
    this_proc_bottom_vector_entries = all_procs_bottom_vector_entries[shared_comm_rank+1]
    this_proc_bottom_vector_entry_contribution_indices = Ti[]
    for (r, proc_range) ∈ zip(0:shared_comm_size-1, all_procs_bottom_vector_entries)
        inds = Vector{Ti}[]
        for iout ∈ proc_range
            this_i_inds = Ti[]
            for (bi, offset) ∈ zip(bottom_block_vector_rowinds,
                                   vector_buffer_blocks_out_offsets)
                block_i_range = searchsorted(bi, iout)
                if !isempty(block_i_range)
                    push!(this_i_inds, block_i_range[1] + offset)
                end
            end
            push!(inds, this_i_inds)
        end
        counts = [length(is) for is ∈ inds]
        if isempty(counts)
            gathered_counts = Ti[]
        else
            gathered_counts = MPI.Gather(counts, shared_comm; root=r)
        end
        if shared_comm_rank == r
            gathered_counts = reshape(gathered_counts,
                                      length(proc_range), shared_comm_size)
            n_inds = sum(gathered_counts; dims=1)[1,:]
            gathered_inds = zeros(Ti, sum(n_inds))
            MPI.Gatherv!(MPI.Buffer(reduce(vcat, inds; init=Ti[])), MPI.VBuffer(gathered_inds, n_inds),
                         shared_comm; root=r)
            gathered_inds_offsets = cumsum(vcat(0, n_inds[1:end-1]))
            proc_counters = zeros(Ti, shared_comm_size)
            for (i, n_contrib) ∈ enumerate(sum(gathered_counts; dims=2))
                push!(this_proc_bottom_vector_entry_contribution_indices, n_contrib)
                this_i_count = 0
                for (s, offset) ∈ zip(1:shared_comm_size, gathered_inds_offsets)
                    n = gathered_counts[i,s]
                    for _ ∈ 1:n
                        i_out = gathered_inds[offset+(proc_counters[s]+=1)]
                        push!(this_proc_bottom_vector_entry_contribution_indices, i_out)
                        this_i_count += 1
                    end
                end
                if this_i_count != n_contrib
                    error("Did not add the same number of indices as expected. "
                          * "this_i_count($this_i_count)!=n_contrib($n_contrib).")
                end
            end
        else
            MPI.Gatherv!(reduce(vcat, inds; init=Ti[]), nothing, shared_comm; root=r)
        end
    end

    return this_proc_bottom_vector_entries, this_proc_bottom_vector_entry_contribution_indices
end

struct BlockCSerial{Nvar,Tf,Ti,Tb,Trange,Trmbb,Tdbs,Tbuff,Tstorage,Fs<:Function}
    blocks::Vector{Tb}
    block_rowinds::Vector{NTuple{Nvar,Trange}}
    block_row_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    bottom_block_rowinds::Vector{NTuple{Nvar,Trange}}
    bottom_block_vector_rowinds::Vector{Trange}
    block_colinds::Vector{NTuple{Nvar,Trange}}
    block_col_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    bottom_block_output_colinds::Array{Vector{Ti},3}
    block_output_ranges::Array{UnitRange{Ti},3}
    bottom_block_output_vector_indices::Matrix{Vector{Ti}}
    block_output_vector_ranges::Matrix{UnitRange{Ti}}
    right_multiplication_buffer_blocks::Trmbb
    dense_buffer_storage::Tdbs
    vector_buffer_blocks_in::Vector{Vector{Tf}}
    vector_buffer_blocks_out::Vector{Tbuff}
    vector_buffer_blocks_out_storage::Tstorage
    this_proc_bottom_vector_entries::UnitRange{Ti}
    this_proc_bottom_vector_entry_contribution_indices::Vector{Ti}
    synchronize_shared::Fs

    function BlockCSerial{Tf}(block_rowinds::Vector{NTuple{Nvar,Tind}},
                              bottom_block_rowinds::Vector{NTuple{Nvar,Tind}},
                              bottom_block_offset_rowinds::Vector{NTuple{Nvar,Tind}},
                              block_colinds::Vector{NTuple{Nvar,Tind}},
                              bottom_block_size::Ti,
                              matrix_template::Union{<:NTuple{Nvar,<:NTuple{Nvar,<:Union{AbstractSparseMatrixCSC,SharedSparseBuffer}}},Nothing},
                              right_multiplication_buffer_storage::Vector{Tf},
                              dense_buffer_storage::Vector{Tf},
                              shared_comm, shared_comm_rank::Ti, shared_comm_size::Ti,
                              synchronize_shared::Fs,
                              allocate_shared_float::Fa) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti},Fs<:Function,Fa<:Function}
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
        bottom_block_rowinds = bottom_block_rowinds[non_empty_blocks]
        bottom_block_offset_rowinds = bottom_block_offset_rowinds[non_empty_blocks]
        bottom_block_vector_rowinds = [vcat(bbvri...) for bbvri ∈ bottom_block_offset_rowinds]
        block_colinds = block_colinds[non_empty_blocks]
        block_col_range_offsets = [vcat(0, cumsum(length(vci) for vci ∈ ci[1:end-1]))
                                   for ci ∈ block_colinds]
        block_col_ranges = [Tuple(voffset .+ (1:length(vci))
                                  for (vci, voffset) ∈ zip(ci, offsets))
                            for (ci, offsets) ∈ zip(block_colinds, block_col_range_offsets)]
        if matrix_template === nothing
            blocks = Matrix{Tf}[]
        else
            blocks = FixedSparseCSC{Tf,Ti}[]
        end
        vector_buffer_blocks_in = Vector{Tf}[]
        vector_buffer_blocks_out_sizes = Ti[]

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
            push!(vector_buffer_blocks_out_sizes, nrow)
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

        vector_buffer_blocks_out_this_proc_size = sum(vector_buffer_blocks_out_sizes)
        vector_buffer_blocks_out_proc_sizes = MPI.Allgather(vector_buffer_blocks_out_this_proc_size, shared_comm)
        vector_buffer_blocks_out_storage = allocate_shared_float(sum(vector_buffer_blocks_out_proc_sizes))
        this_proc_offset = sum(vector_buffer_blocks_out_proc_sizes[1:shared_comm_rank])
        vector_buffer_blocks_out_offsets = cumsum(vcat(0, vector_buffer_blocks_out_sizes[1:end-1])) .+ this_proc_offset
        vector_buffer_blocks_out = [@view vector_buffer_blocks_out_storage[offset+1:offset+n]
                                    for (offset, n) ∈ zip(vector_buffer_blocks_out_offsets,
                                                          vector_buffer_blocks_out_sizes)]
        this_proc_bottom_vector_entries, this_proc_bottom_vector_entry_contribution_indices =
            get_this_proc_bottom_vector_entry_contribution_indices(
                bottom_block_vector_rowinds, vector_buffer_blocks_out_offsets,
                bottom_block_size, shared_comm, shared_comm_rank, shared_comm_size)

        # To avoid shared-memory errors, different processes must write to non-overlapping
        # entries in the output vector. However, when using multiple variables as the
        # off-diagonal variable blocks are dense in 'other' dimensions (not shared by both
        # row and column variables) the output entries of all the blocks are likely to
        # overlap. To deal with this we divide the output vector into non-overlapping
        # segments (as many as there are processes), have each process write its output to
        # one segment, then synchronize and have each process move on to a different
        # segment, continuing until all output has been written.
        n_per_proc = (bottom_block_size + shared_comm_size - 1) ÷ shared_comm_size
        output_ranges = [((shared_comm_rank+segment)%shared_comm_size)*n_per_proc+1:min(((shared_comm_rank+segment)%shared_comm_size+1)*n_per_proc,bottom_block_size)
                         for segment ∈ 0:shared_comm_size-1]
        block_output_vector_ranges = [searchsortedfirst(bi,first(or)):searchsortedlast(bi,last(or))
                                      for bi ∈ bottom_block_vector_rowinds, or ∈ output_ranges]
        bottom_block_output_vector_indices = [bi[block_output_vector_ranges[iblock,isegment]]
                                              for (iblock, bi) ∈ enumerate(bottom_block_vector_rowinds),
                                              isegment ∈ 1:shared_comm_size]
        block_output_ranges = [searchsortedfirst(rio[ivar],first(or))+block_row_range_offsets[iblock][ivar]:searchsortedlast(rio[ivar],last(or))+block_row_range_offsets[iblock][ivar]
                               for ivar ∈ 1:Nvar,
                               (iblock, rio) ∈ enumerate(bottom_block_offset_rowinds),
                               or ∈ output_ranges]
        bottom_block_output_colinds = [bbri[ivar][searchsortedfirst(rio[ivar],first(or)):searchsortedlast(rio[ivar],last(or))]
                                       for ivar ∈ 1:Nvar,
                                       (iblock, (rio, bbri)) ∈ enumerate(zip(bottom_block_offset_rowinds, bottom_block_rowinds)),
                                       or ∈ output_ranges]

        # Convert from Vector{Any} to concretely-typed vector of reshaped views.
        right_multiplication_buffer_blocks = [right_multiplication_buffer_blocks...]

        return new{Nvar,Tf,Ti,eltype(blocks),Tind,typeof(right_multiplication_buffer_blocks),typeof(dense_buffer_storage),eltype(vector_buffer_blocks_out),typeof(vector_buffer_blocks_out_storage),Fs}(
                   blocks, block_rowinds, block_row_ranges, bottom_block_rowinds,
                   bottom_block_vector_rowinds, block_colinds, block_col_ranges,
                   bottom_block_output_colinds, block_output_ranges,
                   bottom_block_output_vector_indices, block_output_vector_ranges,
                   right_multiplication_buffer_blocks, dense_buffer_storage,
                   vector_buffer_blocks_in, vector_buffer_blocks_out,
                   vector_buffer_blocks_out_storage, this_proc_bottom_vector_entries,
                   this_proc_bottom_vector_entry_contribution_indices, synchronize_shared)
    end
end

struct BlockCShared{Nvar,Tf,Ti,Tb,Tind,Tbboci,Tbor,Trmbb,Tdb,Tbi,Tbuff,Tstorage,Fs<:Function}
    block::Tb
    block_rowinds::NTuple{Nvar,Tind}
    block_row_ranges::NTuple{Nvar,UnitRange{Ti}}
    bottom_block_rowinds::NTuple{Nvar,Tind}
    bottom_block_vector_rowinds::Tind
    block_colinds::NTuple{Nvar,Tind}
    block_col_ranges::NTuple{Nvar,UnitRange{Ti}}
    bottom_block_output_colinds::Tbboci
    block_output_ranges::Tbor
    bottom_block_output_vector_indices::Vector{Vector{Ti}}
    block_output_vector_ranges::Vector{UnitRange{Ti}}
    right_multiplication_buffer_block::Trmbb
    dense_buffer::Tdb
    vector_buffer_block_in::Tbi
    vector_buffer_block_out::Tbuff
    vector_buffer_blocks_out_storage::Tstorage
    this_proc_bottom_vector_entries::UnitRange{Ti}
    this_proc_bottom_vector_entry_contribution_indices::Vector{Ti}
    synchronize_shared::Fs

    function BlockCShared{Tf}(block_rowinds_full::NTuple{Nvar,Tind},
                              bottom_block_rowinds_full::NTuple{Nvar,Tind},
                              bottom_block_offset_rowinds_full::NTuple{Nvar,Tind},
                              block_colinds::NTuple{Nvar,Tind},
                              bottom_block_size::Ti,
                              matrix_template::Union{<:NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC}},<:NTuple{Nvar,<:NTuple{Nvar,<:SharedSparseBuffer}},Nothing},
                              right_multiplication_buffer_storage::Vector{Tf},
                              dense_buffer_storage::Vector{Tf}, subgroup_i::Ti,
                              n_subgroups::Ti, block_allocate_shared_float::Fba,
                              schur_complement_is_dense::Bool, block_comm_rank::Integer,
                              block_comm_size::Integer, shared_comm,
                              shared_comm_rank::Integer, shared_comm_size::Integer,
                              synchronize_shared::Fs,
                              allocate_shared_float::Fa) where {Nvar,Tf,Ti,Tind<:AbstractVector{Ti},Fba<:Function,Fs<:Function,Fa<:Function}
        all_bottom_block_vector_rowinds_full = vcat((ri for ri ∈ bottom_block_offset_rowinds_full)...)
        vector_rows_per_proc = (length(all_bottom_block_vector_rowinds_full) + block_comm_size - 1) ÷ block_comm_size
        vector_partial_row_range = block_comm_rank*vector_rows_per_proc+1:min((block_comm_rank+1)*vector_rows_per_proc,length(all_bottom_block_vector_rowinds_full))
        block_m_full = Tuple(length(ri) for ri ∈ block_rowinds_full)
        block_row_range_offsets_full = vcat(0, cumsum(m for m ∈ block_m_full[1:end-1]))
        partial_row_ranges = Tuple(max(1,first(vector_partial_row_range)-offset):min(m,last(vector_partial_row_range)-offset)
                                   for (m, offset) ∈ zip(block_m_full, block_row_range_offsets_full))
        block_rowinds = Tuple(ri[prr] for (prr, ri) ∈ zip(partial_row_ranges, block_rowinds_full))
        block_m = Tuple(length(ri) for ri ∈ block_rowinds)
        block_row_range_offsets = vcat(0, cumsum(m for m ∈ block_m[1:end-1]))
        bottom_block_rowinds = Tuple(ri[prr] for (prr, ri) ∈ zip(partial_row_ranges, bottom_block_rowinds_full))
        bottom_block_offset_rowinds = Tuple(ri[prr] for (prr, ri) ∈ zip(partial_row_ranges, bottom_block_offset_rowinds_full))
        block_row_ranges = Tuple(offset .+ (1:length(bbi))
                                 for (offset, bbi) ∈ zip(block_row_range_offsets, bottom_block_rowinds))

        bottom_block_vector_rowinds = all_bottom_block_vector_rowinds_full[vector_partial_row_range]

        block_n_full = Tuple(length(ci) for ci ∈ block_colinds)
        col_ranges = Tuple(1:n for n ∈ block_n_full)
        block_col_range_offsets = vcat(0, cumsum(n for n ∈ block_n_full[1:end-1]))
        block_col_ranges = Tuple(offset .+ cr
                                 for (offset, cr) ∈ zip(block_col_range_offsets, col_ranges))

        nrow_full = sum(m for m ∈ block_m_full)
        nrow = sum(length(ri) for ri ∈ block_rowinds)
        ncol = sum(n for n ∈ block_n_full)
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
        vector_buffer_blocks_out_proc_sizes = MPI.Allgather(nrow, shared_comm)
        vector_buffer_blocks_out_storage = allocate_shared_float(sum(vector_buffer_blocks_out_proc_sizes))
        vector_buffer_blocks_out_offset = sum(vector_buffer_blocks_out_proc_sizes[1:shared_comm_rank])
        vector_buffer_block_out = @view vector_buffer_blocks_out_storage[vector_buffer_blocks_out_offset+1:vector_buffer_blocks_out_offset+nrow]
        this_proc_bottom_vector_entries, this_proc_bottom_vector_entry_contribution_indices =
            get_this_proc_bottom_vector_entry_contribution_indices(
                bottom_block_vector_rowinds, Ti[0], bottom_block_size, shared_comm,
                shared_comm_rank, shared_comm_size)

        # To avoid shared-memory errors, different subgroups must write to non-overlapping
        # entries in the output vector. However, when using multiple variables as the
        # off-diagonal variable blocks are dense in 'other' dimensions (not shared by both
        # row and column variables) the output entries of all the blocks are likely to
        # overlap. To deal with this we divide the output vector into non-overlapping
        # segments (as many as there are subgroups), have each subgroup write its output
        # to one segment, then synchronize and have each subgroup move on to a different
        # segment, continuing until all output has been written.
        n_per_subgroup = (bottom_block_size + n_subgroups - 1) ÷ n_subgroups
        output_ranges = [((subgroup_i+segment)%n_subgroups)*n_per_subgroup+1:min(((subgroup_i+segment)%n_subgroups+1)*n_per_subgroup,bottom_block_size)
                         for segment ∈ 0:n_subgroups-1]
        block_output_vector_ranges = [searchsortedfirst(bottom_block_vector_rowinds,first(or)):searchsortedlast(bottom_block_vector_rowinds,last(or))
                                      for or ∈ output_ranges]
        bottom_block_output_vector_indices = [bottom_block_vector_rowinds[or]
                                              for or ∈ block_output_vector_ranges]
        if schur_complement_is_dense
            # When using BlockDenseS for schur_complement, the matrix that we write the
            # result of C.Ainv.B to is dense, and not divided into variable-blocks, so we
            # need indices into the full matrix size, and ranges that are not divided by
            # variable.
            block_output_ranges = [searchsortedfirst(all_bottom_block_vector_rowinds_full,first(or)):searchsortedlast(all_bottom_block_vector_rowinds_full,last(or))
                                   for or ∈ output_ranges]
            bottom_block_output_colinds = [all_bottom_block_vector_rowinds_full[or]
                                           for or ∈ block_output_ranges]
        else
            block_output_ranges = [searchsortedfirst(vri,first(or))+voffset:searchsortedlast(vri,last(or))+voffset
                                   for (vri, voffset) ∈ zip(bottom_block_offset_rowinds_full, block_row_range_offsets_full),
                                   or ∈ output_ranges]
            bottom_block_output_colinds = [bottom_block_rowinds_full[ivar][searchsortedfirst(vri,first(or)):searchsortedlast(vri,last(or))]
                                           for (ivar, vri) ∈ enumerate(bottom_block_offset_rowinds_full),
                                           or ∈ output_ranges]
        end

        noutput = length(output_ranges[1])
        n_per_proc = (noutput + block_comm_size - 1) ÷ block_comm_size
        partial_output_range = block_comm_rank*n_per_proc+1:min((block_comm_rank+1)*n_per_proc,noutput)

        return new{Nvar,Tf,Ti,typeof(block),Tind,typeof(bottom_block_output_colinds),typeof(block_output_ranges),typeof(right_multiplication_buffer_block),typeof(dense_buffer),typeof(vector_buffer_block_in),typeof(vector_buffer_block_out),typeof(vector_buffer_blocks_out_storage),Fs}(
                   block, block_rowinds, block_row_ranges, bottom_block_rowinds,
                   bottom_block_vector_rowinds, block_colinds, block_col_ranges,
                   bottom_block_output_colinds, block_output_ranges,
                   bottom_block_output_vector_indices, block_output_vector_ranges,
                   right_multiplication_buffer_block, dense_buffer,
                   vector_buffer_block_in, vector_buffer_block_out,
                   vector_buffer_blocks_out_storage, this_proc_bottom_vector_entries,
                   this_proc_bottom_vector_entry_contribution_indices, synchronize_shared)
    end
end

# A Null type that needs to exist explicitly because when using BlockBShared and
# BlockCShared, the `mul_C_Ainv_dot_B!()` and `mul_C_dot_Ainv_dot_u!()` operations need to
# call `synchronize_shared()` `n_subgroups` times, and it would be inconvenient to create
# a separate MPI communicator including only the processes that are part of the subgroup
# for some block with a corresponding `synchronize_shared()` function.
struct NullBlockCShared{Ti,Tstorage,Fs<:Function}
    n_subgroups::Ti
    vector_buffer_blocks_out_storage::Tstorage
    this_proc_bottom_vector_entries::UnitRange{Ti}
    this_proc_bottom_vector_entry_contribution_indices::Vector{Ti}
    synchronize_shared::Fs

    function NullBlockCShared(n_subgroups::Ti, bottom_block_size::Ti, shared_comm,
                              shared_comm_rank, shared_comm_size, synchronize_shared::Fs,
                              allocate_shared_float::Fa) where {Ti,Fs<:Function,Fa<:Function}
        vector_buffer_blocks_out_proc_sizes = MPI.Allgather(Ti(0), shared_comm)
        vector_buffer_blocks_out_storage = allocate_shared_float(sum(vector_buffer_blocks_out_proc_sizes))
        this_proc_bottom_vector_entries, this_proc_bottom_vector_entry_contribution_indices =
            get_this_proc_bottom_vector_entry_contribution_indices(
                Ti[], Ti[0], bottom_block_size, shared_comm, shared_comm_rank,
                shared_comm_size)
        return new{Ti,typeof(vector_buffer_blocks_out_storage),Fs}(
                   n_subgroups, vector_buffer_blocks_out_storage,
                   this_proc_bottom_vector_entries,
                   this_proc_bottom_vector_entry_contribution_indices, synchronize_shared)
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
            for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges)
                if isempty(ci)
                    continue
                end
                for (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                    if isempty(ri)
                        continue
                    end
                    A_variable_block = full_A[vrow][vcol]
                    for (j1, j2) ∈ zip(cr, ci), (i1, i2) ∈ zip(rr, ri)
                        block[i1,j1] = A_variable_block[i2,j2]
                    end
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
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges)
                    if isempty(ci)
                        continue
                    end
                    for (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                        if isempty(ri)
                            continue
                        end
                        A_variable_block = full_A[vrow][vcol]
                        full_A_colptr = A_variable_block.colptr
                        full_A_rowval = A_variable_block.rowval
                        full_A_nzval = A_variable_block.nzval
                        if isempty(full_A_nzval)
                            continue
                        end
                        first_row = first(ri)
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
            end
        else
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges)
                    if isempty(ci)
                        continue
                    end
                    for (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                        if isempty(ri)
                            continue
                        end
                        A_variable_block = full_A[vrow][vcol]
                        full_A_colptr = A_variable_block.colptr
                        full_A_rowval = A_variable_block.rowval
                        full_A_nzval = A_variable_block.nzval
                        if isempty(full_A_nzval)
                            continue
                        end
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
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges)
                    if isempty(ci)
                        continue
                    end
                    for (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                        if isempty(ri)
                            continue
                        end
                        A_variable_block = full_A[vrow][vcol]
                        full_A_colptr = A_variable_block.colptr
                        full_A_rowval_list = A_variable_block.rowval_list
                        full_A_nzval = A_variable_block.nzval
                        if length(full_A_nzval) == 0
                            continue
                        end
                        first_row = first(ri)
                        nrow = length(ri)
                        for (j1, j2) ∈ zip(cr, ci)
                            first_i = full_A_colptr[j2]
                            last_i = full_A_colptr[j2+1] - 1
                            col_rv = full_A_rowval_list[j2]
                            last_row_i = length(col_rv)
                            row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                            i1 = 1
                            while i1 ≤ nrow
                                full_A_row = col_rv[row_i]
                                block_global_row = ri[i1]
                                if full_A_row == block_global_row
                                    block[rr[i1],j1] = full_A_nzval[row_i+first_i-1]
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
            end
        else
            for (rowinds, row_ranges, colinds, col_ranges, block) ∈
                    zip(block_rowinds, block_row_ranges, block_colinds, block_col_ranges,
                        blocks)
                block_colptr = block.colptr
                block_rowval = block.rowval
                block_nzval = block.nzval
                for (vcol, ci, cr) ∈ zip(1:Nvar, colinds, col_ranges)
                    if isempty(ci)
                        continue
                    end
                    row_offset = 0
                    for (vrow, ri, rr) ∈ zip(1:Nvar, rowinds, row_ranges)
                        if isempty(ri)
                            continue
                        end
                        A_variable_block = full_A[vrow][vcol]
                        full_A_colptr = A_variable_block.colptr
                        full_A_rowval_list = A_variable_block.rowval_list
                        full_A_nzval = A_variable_block.nzval
                        if isempty(full_A_nzval)
                            continue
                        end
                        first_irow = first(rr)
                        last_irow = last(rr)
                        first_row = first(ri)
                        variable_block_nrows = length(ri)
                        for (j1, j2) ∈ zip(cr, ci)
                            first_i = full_A_colptr[j2]
                            col_rv = full_A_rowval_list[j2]
                            last_row_i = length(col_rv)
                            row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                            first_block_i = block_colptr[j1]
                            last_block_i = block_colptr[j1+1] - 1
                            block_i = max(searchsortedlast(@view(block_rowval[first_block_i:last_block_i]), first_irow) - 1, 1) + first_block_i - 1
                            if block_i > last_block_i
                                continue
                            end
                            variable_block_i = searchsortedfirst(rr, block_rowval[block_i])
                            while block_i ≤ last_block_i
                                full_A_row = col_rv[row_i]
                                block_row = block_rowval[block_i]
                                if block_row > last_irow
                                    break
                                end
                                while rr[variable_block_i] < block_row
                                    variable_block_i += 1
                                end
                                if full_A_row == ri[variable_block_i] && block_row == rr[variable_block_i]
                                    block_nzval[block_i] = full_A_nzval[row_i+first_i-1]
                                    block_i += 1
                                    row_i += 1
                                    variable_block_i += 1
                                elseif block_row < rr[variable_block_i]
                                    block_i += 1
                                else
                                    row_i += 1
                                end
                                if row_i > last_row_i || variable_block_i > variable_block_nrows
                                    break
                                end
                            end
                        end
                        row_offset += length(ri)
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
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges)
                if isempty(ci)
                    continue
                end
                for (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
                    if isempty(ri)
                        continue
                    end
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval
                    if isempty(full_A_nzval)
                        continue
                    end
                    first_row = first(ri)
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
            block_colptr = block.colptr
            block_rowval = block.rowval
            block_nzval = block.nzval
            for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges)
                if isempty(ci)
                    continue
                end
                row_offset = 0
                for (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
                    if isempty(ri)
                        continue
                    end
                    A_variable_block = full_A[vrow][vcol]
                    full_A_colptr = A_variable_block.colptr
                    full_A_rowval = A_variable_block.rowval
                    full_A_nzval = A_variable_block.nzval
                    if isempty(full_A_nzval)
                        continue
                    end
                    first_irow = first(rr)
                    last_irow = last(rr)
                    first_row = first(ri)
                    variable_block_nrows = length(ri)
                    for (j1, j2) ∈ zip(cr, ci)
                        first_i = full_A_colptr[j2]
                        col_rv = full_A_rowval_list[j2]
                        last_row_i = length(col_rv)
                        row_i = max(searchsortedlast(col_rv, first_row) - 1, 1)
                        first_block_i = block_colptr[j1]
                        last_block_i = block_colptr[j1+1] - 1
                        block_i = max(searchsortedlast(@view(block_rowval[first_block_i:last_block_i]), first_irow) - 1, 1) + first_block_i - 1
                        if block_i > last_block_i
                            continue
                        end
                        variable_block_i = searchsortedfirst(rr, block_rowval[block_i])
                        while block_i ≤ last_block_i
                            full_A_row = col_rv[row_i]
                            block_row = block_rowval[block_i]
                            if block_row > last_irow
                                break
                            end
                            while rr[variable_block_i] < block_row
                                variable_block_i += 1
                            end
                            if full_A_row == ri[variable_block_i] && block_row == rr[variable_block_i]
                                block_nzval[block_i] = full_A_nzval[row_i+first_i-1]
                                block_i += 1
                                row_i += 1
                                variable_block_i += 1
                            elseif block_row < rr[variable_block_i]
                                block_i += 1
                            else
                                row_i += 1
                            end
                            if row_i > last_row_i || variable_block_i > variable_block_nrows
                                break
                            end
                        end
                    end
                    row_offset += length(ri)
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
            dense_buffer = block
        else
            # dense_buffer is needed in mul_C_Ainv_dot_B!(), as matrix-multiplication
            # seems to be faster with a dense array (for the amount of sparsity that we
            # typically have in the C blocks). Since we need dense_buffer anyway, we
            # simplify copying the entries from full_A by copying to the dense buffer
            # first, then from there to the sparse buffer. If we did not need the dense
            # buffer, we would save memory (and probably be slightly faster) if we copied
            # directly from sparse buffer to sparse buffer, but since full_A is split into
            # variable blocks, while block_C.block is not, the indexing would get rather
            # convoluted.
            dense_buffer = block_C.dense_buffer
            # The memory behind dense_buffer is shared by the C blocks at every level, so
            # we need to zero all the entries that will not be copied from full_A. It is
            # simpler to just zero the buffer first (it might even be more efficient -
            # have not checked as this is not likely a large difference either way).
            dense_buffer .= 0.0
        end
        block_row_ranges = block_C.block_row_ranges
        for (vcol, ci, cr) ∈ zip(1:Nvar, block_colinds, block_col_ranges)
            if isempty(ci)
                continue
            end
            for (vrow, ri, rr) ∈ zip(1:Nvar, block_rowinds, block_row_ranges)
                if isempty(ri)
                    continue
                end
                A_variable_block = full_A[vrow][vcol]
                full_A_colptr = A_variable_block.colptr
                full_A_rowval_list = A_variable_block.rowval_list
                full_A_nzval = A_variable_block.nzval
                if isempty(full_A_nzval)
                    continue
                end
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
                            dense_buffer[rr[i1],j1] = full_A_nzval[row_i+first_i-1]
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
        end

        if isa(block, AbstractSparseMatrixCSC)
            # Also need to copy the block values into the sparse `block`.
            block_colptr = block.colptr
            block_rowval = block.rowval
            block_nzval = block.nzval
            for j ∈ 1:size(dense_buffer, 2)
                col_start = block_colptr[j]
                col_end = block_colptr[j+1] - 1
                for flat_i ∈ col_start:col_end
                    block_nzval[flat_i] = dense_buffer[block_rowval[flat_i],j]
                end
            end
        end
        return nothing
    end
end
function copy_C_submatrix!(block_C::NullBlockCShared, full_A)
    return nothing
end

function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCSerial,
                               Ainv_dot_u)

    @inbounds begin
        blocks = C.blocks
        vector_buffer_blocks_out = C.vector_buffer_blocks_out
        vector_buffer_blocks_out_storage = C.vector_buffer_blocks_out_storage
        this_proc_bottom_vector_entries = C.this_proc_bottom_vector_entries
        this_proc_bottom_vector_entry_contribution_indices = C.this_proc_bottom_vector_entry_contribution_indices
        synchronize_shared = C.synchronize_shared

        if length(blocks) > 0
            for (vec_buffer_out, block, Aiu_block) ∈
                    zip(vector_buffer_blocks_out, blocks, Ainv_dot_u)
                mul!(vec_buffer_out, block, Aiu_block)
            end
        end

        synchronize_shared()

        # Add contributions from all blocks to the output.
        count = 0
        for i1 ∈ this_proc_bottom_vector_entries
            n_contributions = this_proc_bottom_vector_entry_contribution_indices[count+=1]
            for _ ∈ 1:n_contributions
                i2 = this_proc_bottom_vector_entry_contribution_indices[count+=1]
                C_dot_Ainv_dot_u[i1] -= vector_buffer_blocks_out_storage[i2]
            end
        end

        return nothing
    end
end
function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::BlockCShared,
                               Ainv_dot_u)

    @inbounds begin
        block = C.block
        vector_buffer_block_out = C.vector_buffer_block_out
        this_proc_bottom_vector_entries = C.this_proc_bottom_vector_entries
        this_proc_bottom_vector_entry_contribution_indices = C.this_proc_bottom_vector_entry_contribution_indices
        vector_buffer_blocks_out_storage = C.vector_buffer_blocks_out_storage
        synchronize_shared = C.synchronize_shared

        mul!(vector_buffer_block_out, block, Ainv_dot_u)

        synchronize_shared()

        # Add contributions from all blocks to the output.
        count = 0
        for i1 ∈ this_proc_bottom_vector_entries
            n_contributions = this_proc_bottom_vector_entry_contribution_indices[count+=1]
            for _ ∈ 1:n_contributions
                i2 = this_proc_bottom_vector_entry_contribution_indices[count+=1]
                C_dot_Ainv_dot_u[i1] -= vector_buffer_blocks_out_storage[i2]
            end
        end

        return nothing
    end
end
function mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::NullBlockCShared,
                               Ainv_dot_u)

    @inbounds begin
        # This process does not own any blocks, but still participates in adding the
        # contriutions to the output vector.
        this_proc_bottom_vector_entries = C.this_proc_bottom_vector_entries
        this_proc_bottom_vector_entry_contribution_indices = C.this_proc_bottom_vector_entry_contribution_indices
        vector_buffer_blocks_out_storage = C.vector_buffer_blocks_out_storage
        synchronize_shared = C.synchronize_shared

        synchronize_shared()

        # Add contributions from all blocks to the output.
        count = 0
        for i1 ∈ this_proc_bottom_vector_entries
            n_contributions = this_proc_bottom_vector_entry_contribution_indices[count+=1]
            for _ ∈ 1:n_contributions
                i2 = this_proc_bottom_vector_entry_contribution_indices[count+=1]
                C_dot_Ainv_dot_u[i1] -= vector_buffer_blocks_out_storage[i2]
            end
        end

        return nothing
    end
end
