struct SharedSparseBuffer{Tf,Ti,Tcp<:AbstractVector{Ti},Trv<:AbstractVector{Ti},Tnz<:AbstractVector{Tf}}
    m::Ti
    n::Ti
    colptr::Tcp
    rowval_list::Vector{Trv}
    nzval::Tnz
end
Base.eltype(::SharedSparseBuffer{Tf}) where Tf = Tf
Base.size(b::SharedSparseBuffer) = (b.m, b.n)
Base.size(b::SharedSparseBuffer, dim::Integer) = size(b)[dim]
SparseArrays.nnz(b::SharedSparseBuffer) = length(b.nzval)

function get_shared_sparse_buffer(buffer_info::Matrix{<:NamedTuple},
                                  storage::AbstractVector{<:AbstractFloat})
    flat_n = sum(bi.nzval_length for bi ∈ buffer_info)
    if length(storage) < flat_n
        error("Construction of SharedSparseBuffer requires a storage array of at least "
              * "length $(flat_n), but got array with length $(length(storage)).")
    end
    Tf = eltype(storage)
    Ti = typeof(buffer_info[1,1].m)
    Tcp = typeof(buffer_info[1,1].colptr)
    Trv = eltype(buffer_info[1,1].rowval_list)
    Tnz = typeof(@view storage[1:0])

    buffer = Matrix{SharedSparseBuffer{Tf,Ti,Tcp,Trv,Tnz}}(undef, size(buffer_info)...)
    offset = 0
    for (i, bi) ∈ enumerate(buffer_info)
        flat_n = bi.nzval_length
        nzval = @view storage[offset+1:offset+flat_n]

        buffer[i] = SharedSparseBuffer(bi.m, bi.n, bi.colptr, bi.rowval_list, nzval)

        offset += flat_n
    end

    Nvar = size(buffer_info, 1)

    return ntuple(ivar->ntuple(jvar->buffer[ivar,jvar], Nvar), Nvar)
end

function get_dim_indices!(dimensions, block_sizes, flat_i)
    @inbounds begin
        block_inds = zeros(Int64, length(dimensions))
        inner_inds = zeros(Int64, length(dimensions))
        for (i, d) ∈ enumerate(dimensions)
            flat_i, dim_i = divrem(flat_i, d.n_local)
            this_block_npoints = block_sizes[i] * (d.ngrid - 1)
            block_inds[i], inner_inds[i] = divrem(dim_i, this_block_npoints) .+ 1
        end
        return block_inds, inner_inds
    end
end

function add_all_row_inds!(rv, idim, dimensions, row_indices, rowind, row_count,
                           row_dimension_indices)
    @inbounds begin
        if idim == 0
            # rowind is constructed as a 0-based index for convenience. Convert to
            # 1-based before adding to `rv`.
            rowind += 1
            # Only add row indices that are contained in row_indices. For each column,
            # we iterate through the rows in order so we use row_count to avoid
            # searching row_indices here.
            while row_count[] ≤ length(row_indices) && rowind > row_indices[row_count[]]
                row_count[] += 1
            end
            if row_count[] ≤ length(row_indices) && rowind == row_indices[row_count[]]
                push!(rv, row_count[])
                row_count[] += 1
            end
            return nothing
        end
        if idim ∈ row_dimension_indices
            dn = dimensions[idim].n_local
        else
            dn = 1
        end
        rowind *= dn
        for i ∈ 1:dn
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + i - 1,
                              row_count, row_dimension_indices)
        end
        return nothing
    end
end

function add_row_inds!(rv, idim, dimensions, block_sizes, nblock_list, row_indices,
                       block_inds, inner_inds, rowind, row_count, stencil,
                       row_dimension_indices, column_dimension_indices,
                       include_dense_boundaries)
    @inbounds begin
        if stencil ∉ ("point", "element")
            # The main stencil type is "element". "point" is handled by a special case.
            error("Unsupported stencil type \"$stencil\".")
        end
        if idim == 0
            # rowind is constructed as a 0-based index for convenience. Convert to
            # 1-based before adding to `rv`.
            rowind += 1
            # Only add row indices that are contained in row_indices. For each column,
            # we iterate through the rows in order so we use row_count to avoid
            # searching row_indices here.
            while row_count[] ≤ length(row_indices) && rowind > row_indices[row_count[]]
                row_count[] += 1
            end
            if row_count[] ≤ length(row_indices) && rowind == row_indices[row_count[]]
                push!(rv, row_count[])
                row_count[] += 1
            end
            return nothing
        end
        d = dimensions[idim]
        if idim ∈ row_dimension_indices && idim ∉ column_dimension_indices
            dn = d.n_local
            block_npoints = dn - 1
            iblock = 1
            iinner = 1
            nblock = 1
            # Dimension is present in the row variable but not the column variable, so we
            # include all points in the dimension in the rows, and as far as the column is
            # concerned this is always effectively both 'first point' and 'last point'.
            # If dn=1, only active one special handling of boundary points, to avoid
            # double-counting.
            is_first_point = true
            is_last_point = true && dn > 1
        elseif idim ∈ row_dimension_indices
            dn = d.n_local
            block_npoints = block_sizes[idim] * (d.ngrid - 1)
            iblock = block_inds[idim]
            iinner = inner_inds[idim]
            nblock = nblock_list[idim]
            is_first_point = (iinner == 1 && iblock == 1)
            is_last_point = (((iblock - 1) * block_npoints + iinner) == dn)
        else
            dn = 1
            block_npoints = 1
            iblock = 1
            iinner = 1
            nblock = 1
            # Note ensure we only active the special handling for is_last_point=true when
            # is_first_point=false, to avoid double-counting.
            is_first_point = (inner_inds[idim] == 1 && block_inds[idim] == 1)
            is_last_point = (((block_inds[idim] - 1) * block_sizes[idim] * (d.ngrid - 1) + inner_inds[idim]) == d.n_local) && !is_first_point
        end
        rowind *= dn

        if stencil == "point"
            row_offset = (iblock - 1) * block_npoints
            if idim ∈ row_dimension_indices && idim ∉ column_dimension_indices
                # This dimension is present in the row variable but not the column
                # variable, so include all points.
                for row_inner ∈ 1:block_npoints + 1
                    add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                                  row_indices, block_inds, inner_inds,
                                  rowind + row_offset + row_inner - 1, row_count, stencil,
                                  row_dimension_indices, column_dimension_indices)
                end
                return nothing
            else
                row_inner = iinner
                return add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                                     row_indices, block_inds, inner_inds,
                                     rowind + row_offset + row_inner - 1, row_count,
                                     stencil, row_dimension_indices,
                                     column_dimension_indices)
            end
        end

        if iinner == 1 && iblock > 1
            # Is a block boundary, so include points from previous block.
            row_offset = (iblock - 2) * block_npoints
            for row_inner ∈ 1:block_npoints
                add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                              row_indices, block_inds, inner_inds,
                              rowind + row_offset + row_inner - 1, row_count, stencil,
                              row_dimension_indices, column_dimension_indices)
            end
        end
        row_offset = (iblock - 1) * block_npoints
        if include_dense_boundaries && d.dense_boundaries && is_first_point
            # This is the first or last point in a dimension that should have 'dense
            # boundaries'.
            block_start = 2
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + row_offset,
                              row_count, row_dimension_indices)
        else
            block_start = 1
        end
        if include_dense_boundaries && d.dense_boundaries && is_last_point
            row_end = dn - 1
        else
            row_end = dn
        end
        if include_dense_boundaries && iblock > nblock && d.dense_boundaries
            # Have added all points already.
        elseif iblock > nblock
            # Creating entries for the last grid point, this is 'really'
            # iel=nelement_local-1, col_igr=ngrid, so only need to add the row_igr=1
            # point.
            add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list, row_indices,
                          block_inds, inner_inds, rowind + row_offset, row_count, stencil,
                          row_dimension_indices, column_dimension_indices)
        else
            for row_inner ∈ block_start:block_npoints+1
                if row_offset + row_inner > row_end
                    # Do not want to advance past the end of the dimension. The last block in
                    # any dimension may not be full-sized.
                    break
                end
                add_row_inds!(rv, idim - 1, dimensions, block_sizes, nblock_list,
                              row_indices, block_inds, inner_inds,
                              rowind + row_offset + row_inner - 1, row_count, stencil,
                              row_dimension_indices, column_dimension_indices)
            end
        end
        if include_dense_boundaries && d.dense_boundaries && is_last_point
            add_all_row_inds!(rv, idim - 1, dimensions, row_indices, rowind + dn - 1,
                              row_count, row_dimension_indices)
        end
        return nothing
    end
end

function get_shared_sparse_matrix_info(dimensions::Vector{<:Dimension}, shared_comm,
                                       allocate_shared_int::F,
                                       block_sizes::Union{Vector{<:Integer},Nothing}=nothing,
                                       row_indices::Union{Vector{<:Integer},Nothing}=nothing,
                                       column_indices::Union{Vector{<:Integer},Nothing}=nothing,
                                       row_dimensions::Union{UnitRange{<:Integer},Vector{<:Integer},Nothing}=nothing,
                                       column_dimensions::Union{UnitRange{<:Integer},Vector{<:Integer},Nothing}=nothing;
                                       include_dense_boundaries::Bool=true, stencil::String="element",
                                       ind_type::Type=Int64) where F
    @inbounds begin
        point_stencil = (stencil == "point")
        n_local_list = [d.n_local for d ∈ dimensions]
        n_total = prod(n_local_list; init=1)
        if block_sizes === nothing
            block_sizes = ones(ind_type, length(dimensions))
        end
        if row_dimensions === nothing
            row_dimensions = 1:length(dimensions)
        end
        if column_dimensions === nothing
            column_dimensions = 1:length(dimensions)
        end
        if row_indices === nothing
            row_indices = 1:prod(d.n_local for d ∈ dimensions[row_dimensions])
        end
        if column_indices === nothing
            column_indices = 1:prod(d.n_local for d ∈ dimensions[column_dimensions])
        end

        # For any dimensions that are not shared between the row and column variables, all
        # points are coupled by the matrix. We can enforce that effect here by setting the
        # block size equal to the number of elements in those dimensions.
        common_dimensions = intersect(row_dimensions, column_dimensions)
        block_sizes = copy(block_sizes)
        for other_dim ∈ setdiff(1:length(dimensions), common_dimensions)
            d = dimensions[other_dim]
            block_sizes[other_dim] = d.nelement ÷ d.nrank
        end

        nelement_local_list = [d.nelement ÷ d.nrank for d ∈ dimensions]
        nblock_list = [(nel + bs - 1) ÷ bs for (nel, bs) ∈ zip(nelement_local_list,
                                                               block_sizes)]
        m = length(row_indices)
        n = length(column_indices)
        if m == 0 || n == 0
            # No entries so do not need shared-memory allocation.
            return (; m, n, colptr=ind_type[1], rowval_list=Vector{ind_type}[],
                    nzval_length=0)
        end

        for (idim, d) ∈ enumerate(dimensions)
            if d.dense_boundaries && any(x.nrank > 1 for x ∈ dimensions[1:idim-1])
                error("Dimensions to the left of a dimension with dense_boundaries=true "
                      * "(dimension $idim) cannot be distributed across multiple MPI "
                      * "shared-memory blocks")
            end
        end

        shared_comm_rank = MPI.Comm_rank(shared_comm)
        n_rowval = Ref(-1)
        colptr = allocate_shared_int(n + 1)
        if shared_comm_rank == 0
            # Columns can be 'equivalent' in the sense that they have exactly the same
            # non-empty row indices. In a single dimension, two points are equivalent if
            # they are in the interior of the same element, or if they are the same point
            # (in case they are boundary points). Columns are equivalent if their
            # positions in every dimension are equivalent.
            column_cartinds_list = CartesianIndices(Tuple(d.n_local for d ∈ dimensions[column_dimensions]))
            function get_equivalents_lookup(d, nblock, block_size)
                eq_list = zeros(ind_type, d.n_local)
                eq = ind_type(1)
                count = ind_type(0)
                interior_range = 2:block_size*(d.ngrid-1)
                d_n_local = d.n_local
                if d.remove_boundaries
                    for _ ∈ 1:nblock
                        # Lower boundary
                        eq_list[count+=1] = eq
                        eq += 1
                        if count ≥ d_n_local - 1
                            # Last block may not be full size, so make sure not to go
                            # outside the bounds of the dimension.
                            break
                        end
                        for _ ∈ interior_range
                            eq_list[count+=1] = eq
                            if count == d_n_local - 1
                                # Last block may not be full size, so make sure not to go
                                # outside the bounds of the dimension.
                                break
                            end
                        end
                        eq += 1
                    end
                    # Upper boundary
                    if count == d_n_local - 1
                        eq_list[count+=1] = eq
                    end
                else
                    first_block_n = block_size*(d.ngrid-1)
                    for _ ∈ 1:first_block_n
                        eq_list[count+=1] = eq
                    end
                    if first_block_n > 0
                        eq += 1
                    end
                    for _ ∈ 2:nblock
                        # Lower boundary
                        eq_list[count+=1] = eq
                        eq += 1
                        if count ≥ d_n_local - 1
                            # Last block may not be full size, so make sure not to go
                            # outside the bounds of the dimension.
                            break
                        end
                        for _ ∈ interior_range
                            eq_list[count+=1] = eq
                            if count == d_n_local - 1
                                # Last block may not be full size, so make sure not to go
                                # outside the bounds of the dimension.
                                break
                            end
                        end
                        eq += 1
                    end
                    # Upper boundary
                    if count == d_n_local - 1
                        if d.ngrid == 2
                            # There are no 'interior' points, so `eq-1` would actually be
                            # the last boundary between elements, which is not equivalent
                            # to the last grid point.
                            eq_list[count+=1] = eq
                        else
                            eq_list[count+=1] = eq - 1
                        end
                    end
                end
                return eq_list
            end
            equivalents_lookup_lists = [get_equivalents_lookup(d, nb, bs)
                                        for (d, nb, bs)
                                        ∈ zip(dimensions[column_dimensions],
                                              nblock_list[column_dimensions],
                                              block_sizes[column_dimensions])]
            n_equivalents_list = [eq[end] for eq ∈ equivalents_lookup_lists]
            function equivalence_ind(flat_i)
                cartind = Tuple(column_cartinds_list[flat_i])
                eq_ind = ind_type(0)
                for (i, neq, eq_lookup) ∈ zip(reverse(cartind),
                                              reverse(n_equivalents_list),
                                              reverse(equivalents_lookup_lists))
                    eq_ind *= neq
                    eq_ind += eq_lookup[i] - 1
                end
                # eq_ind was constructed as 0-based index. Convert back to 1-based.
                eq_ind += 1
                return eq_ind
            end

            # We temporarily use colptr as a buffer to store the sizes of rowval vectors,
            # or the positions of the first occurence of repeated rowval vectors.
            cdim = dimensions[column_dimensions]
            cbs = block_sizes[column_dimensions]
            cp = ind_type[]
            row_count = Ref(1)
            rv_lookup = Dict{ind_type,Tuple{Vector{ind_type},ind_type}}()
            rv_list = Vector{ind_type}[]
            n_nonzero = 0
            for (icol, col) ∈ enumerate(column_indices)
                push!(cp, n_nonzero + 1)
                col_eq = point_stencil ? nothing : equivalence_ind(col)
                if !point_stencil && col_eq ∈ keys(rv_lookup)
                    col_rv, origin_icol = rv_lookup[col_eq]
                    push!(rv_list, col_rv)
                    colptr[icol] = -origin_icol
                else
                    col_rv = ind_type[]
                    column_block_inds, column_inner_inds = get_dim_indices!(cdim, cbs, col - 1)
                    block_inds = ind_type[]
                    inner_inds = ind_type[]
                    cdim_count = 1
                    dim_count = 1
                    for d ∈ 1:length(dimensions)
                        if cdim_count ≤ length(column_dimensions) && column_dimensions[cdim_count] == d
                            push!(block_inds, column_block_inds[cdim_count])
                            push!(inner_inds, column_inner_inds[cdim_count])
                            dim_count += 1
                            cdim_count += 1
                        else
                            # Column variable does not include this dimension, so we are
                            # treating the whole dimension as a single block. The block
                            # index is therefore 1, and the inner index does not matter.
                            push!(block_inds, 1)
                            push!(inner_inds, 1)
                            dim_count += 1
                        end
                    end
                    row_count[] = 1
                    add_row_inds!(col_rv, length(dimensions), dimensions, block_sizes,
                                  nblock_list, row_indices, block_inds, inner_inds, 0,
                                  row_count, stencil, row_dimensions, column_dimensions,
                                  include_dense_boundaries)
                    if !point_stencil
                        rv_lookup[col_eq] = (col_rv, icol)
                    end
                    push!(rv_list, col_rv)
                    colptr[icol] = length(col_rv)
                end
                new_n = length(col_rv)
                n_nonzero += new_n
            end
            push!(cp, n_nonzero + 1)
        end

        MPI.Barrier(shared_comm)

        nzval_length = sum(n < 0 ? colptr[-n] : n for n ∈ @view(colptr[1:end-1]))
        rowval_length = sum(n for n ∈ @view(colptr[1:end-1]) if n > 0)

        rowval_list = typeof(@view(colptr[1:1]))[]
        rowval_storage = allocate_shared_int(rowval_length)
        offset = 0
        for (icol, n) ∈ enumerate(@view(colptr[1:end-1]))
            if n ≥ 0
                this_rv = @view rowval_storage[offset+1:offset+n]
                offset += n
                if shared_comm_rank == 0
                    this_rv .= rv_list[icol]
                end
            else
                this_rv = rowval_list[-n]
            end
            push!(rowval_list, this_rv)
        end

        MPI.Barrier(shared_comm)

        if shared_comm_rank == 0
            colptr .= cp
        end

        MPI.Barrier(shared_comm)

        return (; m, n, colptr, rowval_list, nzval_length)
    end
end

function get_shared_sparse_matrix_csc_buffer(dimensions::Vector{<:Dimension},
                                             shared_comm, allocate_shared_float::F1,
                                             allocate_shared_int::F2;
                                             block_sizes::Union{Vector{<:Integer},Nothing}=nothing,
                                             row_indices::Union{Vector{<:Integer},Nothing}=nothing,
                                             column_indices::Union{Vector{<:Integer},Nothing}=nothing,
                                             row_dimensions::Union{AbstractVector{<:Integer},Nothing}=nothing,
                                             column_dimensions::Union{AbstractVector{<:Integer},Nothing}=nothing,
                                             stencil::String="element",
                                             ind_type::Type=Int64) where {F1, F2}
    buffer_info = get_shared_sparse_matrix_info(dimensions, shared_comm,
                                                allocate_shared_int, block_sizes,
                                                row_indices, column_indices,
                                                row_dimensions, column_dimensions;
                                                stencil, ind_type)
    if isempty(buffer_info.rowval_list)
        rowval = ind_type[]
    else
        rowval = vcat(buffer_info.rowval_list...)
    end
    nzval = allocate_shared_float(buffer_info.nzval_length)

    return FixedSparseCSC(buffer_info.m, buffer_info.n, buffer_info.colptr, rowval, nzval)
end
