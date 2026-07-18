import MPISchurComplements: ldiv_Bmatrix!

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolverSerial{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},Nothing},Trange,Tsparse} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Vector{Tsolver}
    block_indices::Trange
    sparse_buffers::Vector{Tsparse}
    x_buffer::Vector{Tf}
    u_buffer::Vector{Tf}
    B_column_indices::Trange
    B_buffers_out::Vector{Matrix{Tf}}
    B_buffers_in::Vector{Matrix{Tf}}
    check_lu::Bool
    function BlockDiagonalSolverSerial{Tf}(n::Ti, block_indices, B_column_indices,
                                           use_sparse, timer,
                                           check_lu) where {Tf, Ti <: Integer}
        # Don't need a solver for any empty entries in block_indices, as these blocks have
        # no interior points.
        block_indices = [bi for bi ∈ block_indices if !isempty(bi)]
        B_column_indices = [Bc for (Bc, bi) ∈ zip(B_column_indices, block_indices)
                            if !isempty(bi)]
        block_sizes = [length(bi) for bi ∈ block_indices]
        block_size = maximum(block_sizes; init=0)
        function get_identity(bs)
            if use_sparse
                identity = spzeros(Tf, bs, bs)
            else
                identity = zeros(Tf, bs, bs)
            end
            copyto!(identity, I)
            return identity
        end
        if block_size > 0
            local_block_solver = [lu(get_identity(length(bi))) for bi ∈ block_indices]
            if use_sparse
                sparse_buffers = [spzeros(Tf, bs, bs) for bs ∈ block_sizes]
            else
                sparse_buffers = [nothing for _ ∈ block_indices]
            end
        else
            local_block_solver = [nothing]
            sparse_buffers = [nothing]
        end
        x_buffer = fill(NaN, block_size)
        u_buffer = fill(NaN, block_size)
        B_buffers_out = [zeros(length(bi), length(Bc))
                         for (bi, Bc) ∈ zip(block_indices, B_column_indices)]
        if use_sparse
            B_buffers_in = deepcopy(B_buffers_out)
        else
            B_buffers_in = Matrix{Tf}[]
        end
        return new{Tf,Ti,eltype(local_block_solver),typeof(block_indices),eltype(sparse_buffers)}(
                   n, local_block_solver, block_indices, sparse_buffers, x_buffer,
                   u_buffer, B_column_indices, B_buffers_out, B_buffers_in, check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverSerial) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverSerial, d::Integer) = size(Alu)[d]

# When this solver is used there are more processes than blocks, so we use multiple
# processes to solve each block, with shared-memory parallelism.
struct BlockDiagonalSolverShared{Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},MPIDenseLU{Tf},Nothing},Tserialsolver<:Union{Factorization{Tf},Nothing},Tm,Trange,Tsync} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_block_serial_solver::Tserialsolver
    factors::Tm
    block_indices::Trange
    partial_block_indices::Trange
    partial_col_range::UnitRange{Ti}
    x_buffer::Vector{Tf}
    u_buffer::Vector{Tf}
    B_column_indices::Trange
    block_comm_rank::Ti
    synchronize_shared::Tsync
    check_lu::Bool
    function BlockDiagonalSolverShared{Tf}(n::Ti, block_indices, B_column_indices,
                                           block_comm, allocate_shared_float,
                                           allocate_shared_int, synchronize_shared::F,
                                           timer, check_lu) where {Tf, Ti <: Integer, F}
        block_size = length(block_indices)
        block_comm_rank = MPI.Comm_rank(block_comm)
        block_comm_size = MPI.Comm_size(block_comm)

        if block_size == 0
            local_block_solver = nothing
            local_block_serial_solver = nothing
            factors = nothing
            x_buffer = fill(NaN, block_size)
            u_buffer = fill(NaN, block_size)
        elseif block_comm_size > 1 && block_size > 1024
            # Have multiple processes working on this block, and the block size is
            # big enough to be worth using a parallel dense-matrix LU solver.
            factors = allocate_shared_float(block_size, block_size)
            if MPI.Comm_rank(block_comm) == 0
                copyto!(factors, I)
            end

            local_block_solver =
                mpi_dense_lu(factors, 128, block_comm, block_comm, MPI.COMM_SELF,
                             allocate_shared_float, allocate_shared_int;
                             synchronize_shared=synchronize_shared,
                             skip_factorization=true, check_lu=check_lu, timer=timer)
            local_block_serial_solver = LU(factors,
                                           local_block_solver.factorization_shared_lu.ipiv,
                                           block_size)
            x_buffer = allocate_shared_float(block_size)
            u_buffer = allocate_shared_float(block_size)
            if block_comm_rank == 0
                x_buffer .= NaN
                u_buffer .= NaN
            end
        else
            factors = allocate_shared_float(block_size, block_size)
            ipiv = allocate_shared_int(block_size)
            if MPI.Comm_rank(block_comm) == 0
                copyto!(factors, I)
                local_block_solver = LU(factors, ipiv, block_size)
                local_block_serial_solver = local_block_solver
            else
                local_block_solver = nothing
                local_block_serial_solver = LU(factors, ipiv, block_size)
            end
            x_buffer = fill(NaN, block_size)
            u_buffer = fill(NaN, block_size)
        end

        cols_per_proc = (block_size + block_comm_size - 1) ÷ block_comm_size
        partial_col_range = block_comm_rank*cols_per_proc+1:min((block_comm_rank+1)*cols_per_proc,block_size)
        partial_block_indices = block_indices[partial_col_range]

        return new{Tf,Ti,typeof(local_block_solver),typeof(local_block_serial_solver),typeof(factors),typeof(block_indices),F}(
                   n, local_block_solver, local_block_serial_solver, factors,
                   block_indices, partial_block_indices, partial_col_range, x_buffer,
                   u_buffer, B_column_indices, block_comm_rank, synchronize_shared,
                   check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverShared) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverShared, d::Integer) = size(Alu)[d]

function get_block_diagonal_solver(level_info, data_type, is_top_level, use_shared_blocks,
                                   timer, check_lu, block_allocate_shared_float=nothing,
                                   block_allocate_shared_int=nothing,
                                   block_synchronize_shared=nothing)
    # The A blocks may be sparse at the top level, but will generally be dense on lower
    # levels, so only use a sparse LU solver when is_top_level=true.
    if isempty(level_info.local_top_vector_a_block_indices)
        return MPIStaticCondensationNull{data_type}()
    elseif use_shared_blocks
        return BlockDiagonalSolverShared{data_type}(level_info.global_size - level_info.global_bottom_vector_size,
                                                    level_info.local_top_vector_a_block_indices[1],
                                                    level_info.a_block_off_diagonal_indices[1],
                                                    level_info.block_comm,
                                                    block_allocate_shared_float,
                                                    block_allocate_shared_int,
                                                    block_synchronize_shared, timer,
                                                    check_lu)
    else
        return BlockDiagonalSolverSerial{data_type}(level_info.global_size - level_info.global_bottom_vector_size,
                                                    level_info.local_top_vector_a_block_indices,
                                                    level_info.a_block_off_diagonal_indices,
                                                    is_top_level, timer, check_lu)
    end
end

function lu!(block_diagonal_solver::BlockDiagonalSolverSerial,
             full_A::Union{AbstractMatrix,SharedSparseBuffer})
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        check_lu = block_diagonal_solver.check_lu
        if solver != [nothing]
            for (s, inds, buffer) ∈ zip(solver, block_diagonal_solver.block_indices,
                                        block_diagonal_solver.sparse_buffers)
                if isa(s, UmfpackLU)
                    update_sparse_matrix!(buffer, full_A, inds, inds)
                    lu!(s, buffer; reuse_symbolic=false, check=check_lu)
                else
                    factors = s.factors
                    if isa(full_A, AbstractSparseMatrixCSC)
                        colptr = full_A.colptr
                        rowval = full_A.rowval
                        nzval = full_A.nzval
                        first_row = inds[1]
                        for (j1, j2) ∈ enumerate(inds)
                            first_i = colptr[j2]
                            last_i = colptr[j2+1]-1
                            flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
                            for (i1, i2) ∈ enumerate(inds)
                                while flat_i ≤ last_i && rowval[flat_i] < i2
                                    flat_i += 1
                                end
                                if rowval[flat_i] == i2
                                    factors[i1,j1] = nzval[flat_i]
                                else
                                    factors[i1,j1] = 0.0
                                end
                            end
                        end
                    elseif isa(full_A, SharedSparseBuffer)
                        colptr = full_A.colptr
                        rowval_list = full_A.rowval_list
                        nzval = full_A.nzval
                        first_row = inds[1]
                        for (j1, j2) ∈ enumerate(inds)
                            first_i = colptr[j2]
                            col_rowval = rowval_list[j2]
                            row_i = max(searchsortedlast(col_rowval, first_row) - 1, 1)
                            last_row = length(col_rowval)
                            flat_i = row_i + first_i - 1
                            for (i1, i2) ∈ enumerate(inds)
                                while row_i ≤ last_row && col_rowval[row_i] < i2
                                    row_i += 1
                                end
                                if col_rowval[row_i] == i2
                                    factors[i1,j1] = nzval[row_i+first_i-1]
                                else
                                    factors[i1,j1] = 0.0
                                end
                            end
                        end
                    else
                        for (j1, j2) ∈ enumerate(inds), (i1, i2) ∈ enumerate(inds)
                            factors[i1,j1] = full_A[i2,j2]
                        end
                    end
                    getrf!(factors, s.ipiv; check=check_lu)
                end
            end
        end
        return nothing
    end
end
function lu!(block_diagonal_solver::BlockDiagonalSolverShared,
             full_A::Union{AbstractMatrix,SharedSparseBuffer})
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        factors = block_diagonal_solver.factors
        block_indices = block_diagonal_solver.block_indices
        partial_block_indices = block_diagonal_solver.partial_block_indices
        partial_col_range = block_diagonal_solver.partial_col_range
        synchronize_shared = block_diagonal_solver.synchronize_shared

        if isa(full_A, AbstractSparseMatrixCSC)
            colptr = full_A.colptr
            rowval = full_A.rowval
            nzval = full_A.nzval
            first_row = block_indices[1]
            for (j1, j2) ∈ zip(partial_col_range, partial_block_indices)
                first_i = colptr[j2]
                last_i = colptr[j2+1]-1
                flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
                for (i1, i2) ∈ enumerate(block_indices)
                    while flat_i ≤ last_i && rowval[flat_i] < i2
                        flat_i += 1
                    end
                    if rowval[flat_i] == i2
                        factors[i1,j1] = nzval[flat_i]
                    else
                        factors[i1,j1] = 0.0
                    end
                end
            end
        elseif isa(full_A, SharedSparseBuffer)
            colptr = full_A.colptr
            rowval_list = full_A.rowval_list
            nzval = full_A.nzval
            first_row = block_indices[1]
            for (j1, j2) ∈ zip(partial_col_range, partial_block_indices)
                first_i = colptr[j2]
                col_rowval = rowval_list[j2]
                row_i = max(searchsortedlast(col_rowval, first_row) - 1, 1)
                last_row = length(col_rowval)
                flat_i = row_i + first_i - 1
                for (i1, i2) ∈ enumerate(block_indices)
                    while row_i ≤ last_row && col_rowval[row_i] < i2
                        row_i += 1
                    end
                    if col_rowval[row_i] == i2
                        factors[i1,j1] = nzval[row_i+first_i-1]
                    else
                        factors[i1,j1] = 0.0
                    end
                end
            end
        else
            for (j1, j2) ∈ zip(partial_col_range, partial_block_indices), (i1, i2) ∈ enumerate(block_indices)
                factors[i1,j1] = full_A[i2,j2]
            end
        end

        synchronize_shared()

        if isa(solver, MPIDenseLU)
            # Note that this would not work if we were using distributed MPI in the
            # MPIDenseLU `solver`, as in the distributed-MPI case, `factors` is not
            # factorised directly, and we require that it is for
            # `local_block_serial_solver` to work.
            lu!(solver, factors)
        elseif isa(solver, LU)
            getrf!(factors, solver.ipiv; check=block_diagonal_solver.check_lu)
        end
        return nothing
    end
end

function ldiv!(buffers::AbstractVector,
               block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::AbstractVector{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if solvers != [nothing]
            x_buffer = block_diagonal_solver.x_buffer
            u_buffer = block_diagonal_solver.u_buffer
            for (bi, s, buff) ∈ zip(block_diagonal_solver.block_indices, solvers, buffers)
                n = length(bi)
                this_u_buffer = @view u_buffer[1:n]
                for (i1, i2) ∈ enumerate(bi)
                    this_u_buffer[i1] = u[i2]
                end
                ldiv!(buff, s, this_u_buffer)
            end
        end
        return nothing
    end
end
function ldiv!(buffer::AbstractVector{T},
               block_diagonal_solver::BlockDiagonalSolverShared{T},
               u::AbstractVector{T}) where T
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        block_comm_rank = block_diagonal_solver.block_comm_rank
        block_indices = block_diagonal_solver.block_indices
        synchronize_shared = block_diagonal_solver.synchronize_shared

        # Need to synchronize here as `u_buffer` is filled only on block_comm_rank==0, but
        # `u` was filled in parallel. Maybe it would be worth filling `u_buffer` in
        # parallel? Then would need to synchronize before `ldiv!()` call.
        synchronize_shared()

        if solver === nothing
            # Nothing to do.
        elseif isa(solver, MPIDenseLU)
            if length(block_indices) == length(u)
                # There is only one block which includes the whole rhs/solution vector, so
                # do not need to select range out of u.
                ldiv!(buffer, solver, u)
            else
                u_buffer = block_diagonal_solver.u_buffer
                if block_comm_rank == 0
                    for (i1, i2) ∈ enumerate(block_indices)
                        u_buffer[i1] = u[i2]
                    end
                end
                ldiv!(buffer, solver, u_buffer)
            end
        else
            if block_comm_rank == 0
                for (i1, i2) ∈ enumerate(block_indices)
                    buffer[i1] = u[i2]
                end
                ldiv!(solver, buffer)
            end
        end
        return nothing
    end
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractVector{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(x::AbstractMatrix{T},
               block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractMatrix{T}) where T
    if block_diagonal_solver.local_block_solver !== nothing
        for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
            ldiv!(this_x, block_diagonal_solver, this_u)
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::Matrix{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if length(solvers) == 1 && length(block_diagonal_solver.block_indices[1]) == size(u, 1)
            # There is only one block which includes the whole rhs/solution vector, so do
            # not need to select range out of x/u.
            ldiv!(x, solvers[1], u)
        else
            if block_diagonal_solver.local_block_solver !== nothing
                for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
                    ldiv!(this_x, block_diagonal_solver, this_u)
                end
            end
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverShared{T},
               u::Matrix{T}) where T
    for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
        ldiv!(this_x, block_diagonal_solver, this_u)
    end
    return nothing
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{T},BlockDiagonalSolverShared{T}},
               u::AbstractMatrix{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverSerial{T}, u::Matrix{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if length(solvers) == 1 && length(block_diagonal_solver.block_indices[1]) == size(u, 1)
            # There is only one block, so do not need to select range out of u.
            ldiv!(solvers[1], u)
            return nothing
        else
            return ldiv!(u, block_diagonal_solver, u)
        end
    end
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverShared{T}, u::Matrix{T}) where T
    return ldiv!(u, block_diagonal_solver, u)
end
function sparse_column_has_overlap(rowval, bi)
    @inbounds begin
        r_count = 1
        b_count = 1
        while r_count ≤ length(rowval) && b_count ≤ length(bi)
            if rowval[r_count] == bi[b_count]
                return true
            elseif rowval[r_count] < bi[b_count]
                r_count += 1
            else
                b_count += 1
            end
        end
        return false
    end
end
function ldiv!(x::AbstractSparseMatrixCSC{T},
               block_diagonal_solver::BlockDiagonalSolverSerial{T},
               u::AbstractSparseMatrixCSC{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if solvers != [nothing]
            m = size(u, 2)
            u_colptr = u.colptr
            u_rowval = u.rowval
            x_colptr = x.colptr
            x_rowval = x.rowval
            x_nzval = x.nzval
            u_buffer = block_diagonal_solver.u_buffer
            x_buffer = block_diagonal_solver.x_buffer
            for (bi, s) ∈ zip(block_diagonal_solver.block_indices, solvers)
                block_start = first(bi)
                block_end = last(bi)
                block_size = length(bi)
                this_u_buffer = @view u_buffer[1:block_size]
                if eltype(solvers) <: LU
                    this_x_buffer = this_u_buffer
                else
                    this_x_buffer = @view x_buffer[1:block_size]
                end
                for col ∈ 1:m
                    u_flat_start = u_colptr[col]
                    u_flat_end = u_colptr[col+1] - 1
                    if u_flat_end < u_flat_start
                        # Column is empty.
                        continue
                    end
                    if sparse_column_has_overlap(@view(u_rowval[u_flat_start:u_flat_end]), bi)
                        # Column has non-zero row entries for this block.
                        u_column = @view u[:,col]
                        for (i1, i2) ∈ enumerate(bi)
                            this_u_buffer[i1] = u_column[i2]
                        end
                        if eltype(solvers) <: LU
                            # Dense-matrix LU solver, most efficient to solve in-place
                            ldiv!(s, this_u_buffer)
                        else
                            ldiv!(this_x_buffer, s, this_u_buffer)
                        end
                        x_flat_start = x_colptr[col]
                        x_flat_end = x_colptr[col+1] - 1
                        x_col_rowval = @view x_rowval[x_flat_start:x_flat_end]
                        nxr = x_flat_end - x_flat_start + 1
                        count = max(searchsortedlast(x_col_rowval, first(bi)) - 1, 1)
                        for (i2, i1) ∈ enumerate(bi)
                            # Assume that the structural non-zero entries of `x` are
                            # enough to contain all the non-zero entries of the solve.
                            # Note that the entries in this_x_buffer that should be
                            # structurally zero might only be zero up to floating-point
                            # precision.
                            while count ≤ nxr && x_col_rowval[count] < i1
                                count += 1
                            end
                            if count > nxr
                                break
                            end
                            if i1 == x_col_rowval[count]
                                x_nzval[x_flat_start+count-1] = this_x_buffer[i2]
                                count += 1
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end

# Specialized implementations to be used for A^{-1}.B
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                       B::AbstractMatrix{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if solvers != [nothing]
            if eltype(solvers) <: LU
                for (bi, s, Bbuff, Bcols) ∈ zip(block_diagonal_solver.block_indices, solvers,
                                                block_diagonal_solver.B_buffers_out,
                                                block_diagonal_solver.B_column_indices)
                    for (j1, j2) ∈ enumerate(Bcols), (i1, i2) ∈ enumerate(bi)
                        Bbuff[i1,j1] = B[i2,j2]
                    end
                    ldiv!(s, Bbuff)
                    for (j2, j1) ∈ enumerate(Bcols), (i2, i1) ∈ enumerate(bi)
                        B[i1,j1] = Bbuff[i2,j2]
                    end
                end
            else
                for (bi, s, Bbuff_out, Bbuff_in, Bcols) ∈
                        zip(block_diagonal_solver.block_indices, solvers,
                            block_diagonal_solver.B_buffers_out,
                            block_diagonal_solver.B_buffers_in,
                            block_diagonal_solver.B_column_indices)
                    for (j1, j2) ∈ enumerate(Bcols), (i1, i2) ∈ enumerate(bi)
                        Bbuff_in[i1,j1] = B[i2,j2]
                    end
                    ldiv!(Bbuff_out, s, Bbuff_in)
                    for (j2, j1) ∈ enumerate(Bcols), (i2, i1) ∈ enumerate(bi)
                        B[i1,j1] = Bbuff_out[i2,j2]
                    end
                end
            end
        end
        return nothing
    end
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                       B::AbstractSparseMatrixCSC{T}) where T
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if solvers != [nothing]
            if eltype(solvers) <: LU
                for (bi, s, Bbuff, Bcols) ∈ zip(block_diagonal_solver.block_indices, solvers,
                                                block_diagonal_solver.B_buffers_out,
                                                block_diagonal_solver.B_column_indices)
                    B_colptr = B.colptr
                    B_rowval = B.rowval
                    B_nzval = B.nzval
                    firstrow = first(bi)
                    for (j1, j2) ∈ enumerate(Bcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ enumerate(bi)
                            while flat_i ≤ last_i && B_rowval[flat_i] < i2
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                            if B_rowval[flat_i] == i2
                                Bbuff[i1,j1] = B_nzval[flat_i]
                            end
                        end
                    end
                    ldiv!(s, Bbuff)
                    for (j1, j2) ∈ enumerate(Bcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ enumerate(bi)
                            while flat_i ≤ last_i && B_rowval[flat_i] < i2
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                            if B_rowval[flat_i] == i2
                                B_nzval[flat_i] = Bbuff[i1,j1]
                            end
                        end
                    end
                end
            else
                for (bi, s, Bbuff_out, Bbuff_in, Bcols) ∈
                        zip(block_diagonal_solver.block_indices, solvers,
                            block_diagonal_solver.B_buffers_out,
                            block_diagonal_solver.B_buffers_in,
                            block_diagonal_solver.B_column_indices)
                    B_colptr = B.colptr
                    B_rowval = B.rowval
                    B_nzval = B.nzval
                    firstrow = first(bi)
                    for (j1, j2) ∈ enumerate(Bcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ enumerate(bi)
                            while flat_i ≤ last_i && B_rowval[flat_i] < i2
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                            if B_rowval[flat_i] == i2
                                Bbuff_in[i1,j1] = B_nzval[flat_i]
                            end
                        end
                    end
                    ldiv!(Bbuff_out, s, Bbuff_in)
                    for (j1, j2) ∈ enumerate(Bcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ enumerate(bi)
                            while flat_i ≤ last_i && B_rowval[flat_i] < i2
                                flat_i += 1
                            end
                            if flat_i > last_i
                                break
                            end
                            if B_rowval[flat_i] == i2
                                B_nzval[flat_i] = Bbuff_out[i1,j1]
                            end
                        end
                    end
                end
            end
        end
        return nothing
    end
end
function ldiv_block_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{T},
                             B::BlockAinvDotBSerial{T}) where T
    for (solver, block) ∈ zip(block_diagonal_solver.local_block_solver, B.blocks)
        ldiv!(solver, block)
    end
    return nothing
end
function ldiv_block_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{T},
                             B::BlockAinvDotBShared{T}) where T
    @inbounds begin
        solver = block_diagonal_solver.local_block_serial_solver
        if solver !== nothing
            block = B.block
            partial_block = B.partial_block
            partial_col_range = B.partial_col_range
            partial_row_range = B.partial_row_range
            synchronize_shared = B.synchronize_shared

            # Probably more efficient to parallelise over columns in `block` than to use a
            # parallelised `ldiv!()` on the full block.
            ldiv!(solver, @view(block[:,partial_col_range]))

            synchronize_shared()

            partial_block .= @view block[partial_row_range,:]
        end
        return nothing
    end
end
function ldiv_block_Bmatrix!(block_diagonal_solver::MPIStaticCondensationNull{T},
                             B) where T
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{T},
                       B::Matrix{T}) where T
    # When not using BlockAinvDotBShared, this function will use a different
    # parallelisation than copy_B_submatrix!(), so need to synchronize.
    block_diagonal_solver.synchronize_shared()
    return ldiv!(block_diagonal_solver, B)
end
