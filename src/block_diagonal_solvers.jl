import MPISchurComplements: ldiv_Bmatrix!

# Each process participates in the solution of only one of the blocks in the
# block-diagonal solve, so only need to hold the solver and indices for that block.
struct BlockDiagonalSolverSerial{Nvar,Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},Nothing},Tinds,Tvecinds} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Vector{Tsolver}
    block_indices::Tinds
    block_vector_indices::Tvecinds
    block_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    u_buffer::Vector{Tf}
    B_column_indices::Tinds
    B_column_ranges::Vector{NTuple{Nvar,UnitRange{Ti}}}
    B_buffers_out::Vector{Matrix{Tf}}
    check_lu::Bool
    function BlockDiagonalSolverSerial{Tf}(n::Ti,
                                           block_indices::Vector{<:NTuple{Nvar,AbstractVector{Ti}}},
                                           block_vector_indices::Vector{<:NTuple{Nvar,AbstractVector{Ti}}},
                                           B_column_indices::Vector{<:NTuple{Nvar,AbstractVector{Ti}}},
                                           timer,
                                           check_lu) where {Nvar, Tf<:AbstractFloat, Ti<:Integer}
        # Don't need a solver for any empty entries in block_indices, as these blocks have
        # no interior points.
        non_empty_blocks = [!all(isempty(vbi) for vbi ∈ bi) for bi ∈ block_indices]
        block_indices = block_indices[non_empty_blocks]
        block_range_offsets = [vcat(0, cumsum(length(vbi) for vbi ∈ bi[1:end-1]))
                               for bi ∈ block_indices]
        block_ranges = [Tuple(voffset .+ (1:length(vbi))
                              for (vbi, voffset) ∈ zip(bi, offsets))
                        for (bi, offsets) ∈ zip(block_indices, block_range_offsets)]
        block_vector_indices = [vcat(bvbi...) for bvbi ∈ block_vector_indices[non_empty_blocks]]
        B_column_indices = B_column_indices[non_empty_blocks]
        B_column_range_offsets = [vcat(0, cumsum(length(vBci) for vBci ∈ Bci[1:end-1]))
                                  for Bci ∈ B_column_indices]
        B_column_ranges = [Tuple(voffset .+ (1:length(vBci))
                                 for (vBci, voffset) ∈ zip(Bci, offsets))
                           for (Bci, offsets) ∈ zip(B_column_indices, B_column_range_offsets)]
        block_sizes = [sum(length(vbi) for vbi ∈ bi) for bi ∈ block_indices]
        block_size = maximum(block_sizes; init=0)
        function get_identity(bs)
            identity = zeros(Tf, bs, bs)
            copyto!(identity, I)
            return identity
        end
        if block_size > 0
            local_block_solver = [lu(get_identity(length(bi))) for bi ∈ block_vector_indices]
        else
            local_block_solver = [nothing]
        end
        u_buffer = fill(NaN, block_size)
        B_buffers_out = [zeros(sum(length(vbi) for vbi ∈ bi), sum(length(vBc) for vBc ∈ Bc))
                         for (bi, Bc) ∈ zip(block_indices, B_column_indices)]
        return new{Nvar,Tf,Ti,eltype(local_block_solver),typeof(block_indices),typeof(block_vector_indices)}(
                   n, local_block_solver, block_indices, block_vector_indices,
                   block_ranges, u_buffer, B_column_indices, B_column_ranges,
                   B_buffers_out, check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverSerial) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverSerial, d::Integer) = size(Alu)[d]

# When this solver is used there are more processes than blocks, so we use multiple
# processes to solve each block, with shared-memory parallelism.
struct BlockDiagonalSolverShared{Nvar,Tf<:AbstractFloat,Ti<:Integer,Tsolver<:Union{Factorization{Tf},MPIDenseLU{Tf},Nothing,Missing},Tserialsolver<:Union{Factorization{Tf},Nothing},Tm,Tinds,Tsync} <: MPISchurComplementAFactorization{Tf}
    n::Ti
    local_block_solver::Tsolver
    local_block_serial_solver::Tserialsolver
    factors::Tm
    block_indices::NTuple{Nvar,Tinds}
    block_vector_indices::Vector{Ti}
    block_ranges::NTuple{Nvar,UnitRange{Ti}}
    partial_block_indices::NTuple{Nvar,Tinds}
    partial_col_ranges::NTuple{Nvar,UnitRange{Ti}}
    u_buffer::Vector{Tf}
    B_column_indices::NTuple{Nvar,Tinds}
    block_comm_rank::Ti
    synchronize_shared::Tsync
    check_lu::Bool
    function BlockDiagonalSolverShared{Tf}(n::Ti, block_indices, block_vector_indices,
                                           B_column_indices, block_comm,
                                           allocate_shared_float, allocate_shared_int,
                                           synchronize_shared::F, timer,
                                           check_lu) where {Tf, Ti <: Integer, F}
        block_size = sum(length(bi) for bi ∈ block_indices)
        block_comm_rank = MPI.Comm_rank(block_comm)
        block_comm_size = MPI.Comm_size(block_comm)

        block_offsets = vcat(Ti(0), cumsum(length(bi) for bi ∈ block_indices[1:end-1]))
        block_ranges = Tuple(block_offsets[i] .+ (1:length(bi))
                             for (i, bi) ∈ enumerate(block_indices))

        block_vector_indices = vcat(block_vector_indices...)

        if block_size == 0
            local_block_solver = nothing
            local_block_serial_solver = nothing
            factors = nothing
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
                             distributed_block_rows=1, skip_factorization=true,
                             check_lu=check_lu)
            local_block_serial_solver = LU(factors,
                                           local_block_solver.factorization_shared_lu.ipiv,
                                           block_size)
            u_buffer = allocate_shared_float(block_size)
            if block_comm_rank == 0
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
                # Use `missing` here to distinguish from the `block_size == 0` case above.
                local_block_solver = missing
                local_block_serial_solver = LU(factors, ipiv, block_size)
            end
            u_buffer = fill(NaN, block_size)
        end

        cols_per_proc = Tuple((length(bi) + block_comm_size - 1) ÷ block_comm_size for bi ∈ block_indices)
        partial_col_ranges = Tuple(block_comm_rank*nc+1:min((block_comm_rank+1)*nc,length(bi))
                                   for (nc, bi) ∈ zip(cols_per_proc, block_indices))
        partial_block_indices = Tuple(bi[pcr] for (bi, pcr) ∈ zip(block_indices, partial_col_ranges))

        return new{length(block_indices),Tf,Ti,typeof(local_block_solver),typeof(local_block_serial_solver),typeof(factors),eltype(block_indices),F}(
                   n, local_block_solver, local_block_serial_solver, factors,
                   block_indices, block_vector_indices, block_ranges,
                   partial_block_indices, partial_col_ranges, u_buffer, B_column_indices,
                   block_comm_rank, synchronize_shared, check_lu)
    end
end
Base.size(Alu::BlockDiagonalSolverShared) = (Alu.n, Alu.n)
Base.size(Alu::BlockDiagonalSolverShared, d::Integer) = size(Alu)[d]

function get_block_diagonal_solver(level_info, data_type, use_shared_blocks, timer,
                                   check_lu, block_allocate_shared_float=nothing,
                                   block_allocate_shared_int=nothing,
                                   block_synchronize_shared=nothing)
    n = sum(li.global_size - li.global_bottom_vector_size for li ∈ level_info)
    if all(isempty(li.local_top_vector_a_block_indices) for li ∈ level_info)
        return MPIStaticCondensationNull{data_type}()
    elseif use_shared_blocks
        return BlockDiagonalSolverShared{data_type}(
                   n, Tuple(li.local_top_vector_a_block_indices[1] for li ∈ level_info),
                   Tuple(li.local_top_vector_a_block_offset_indices[1] for li ∈ level_info),
                   Tuple(li.a_block_off_diagonal_indices[1] for li ∈ level_info),
                   level_info[1].block_comm, block_allocate_shared_float,
                   block_allocate_shared_int, block_synchronize_shared, timer, check_lu)
    else
        return BlockDiagonalSolverSerial{data_type}(
                   n,
                   extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_indices),
                   extract_block_field_from_Tuple(level_info, :local_top_vector_a_block_offset_indices),
                   extract_block_field_from_Tuple(level_info, :a_block_off_diagonal_indices),
                   timer, check_lu)
    end
end

function lu!(block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
             full_A::NTuple{Nvar,<:NTuple{Nvar,<:Union{AbstractMatrix{T},SharedSparseBuffer{T}}}}) where {Nvar,T}
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        check_lu = block_diagonal_solver.check_lu
        if !(eltype(solver) <: Nothing)
            for (s, ranges, inds) ∈ zip(solver, block_diagonal_solver.block_ranges,
                                        block_diagonal_solver.block_indices)
                factors = s.factors
                for (vcol, colrange, colinds) ∈ zip(1:Nvar, ranges, inds),
                        (vrow, rowrange, rowinds) ∈ zip(1:Nvar, ranges, inds)
                    A_variable_block = full_A[vrow][vcol]
                    if isempty(rowinds)
                        first_row = 1
                    else
                        first_row = rowinds[1]
                    end
                    if isa(A_variable_block, AbstractSparseMatrixCSC)
                        colptr = A_variable_block.colptr
                        rowval = A_variable_block.rowval
                        nzval = A_variable_block.nzval
                        for (j1, j2) ∈ zip(colrange, colinds)
                            first_i = colptr[j2]
                            last_i = colptr[j2+1]-1
                            flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
                            for (i1, i2) ∈ zip(rowrange, rowinds)
                                while flat_i < last_i && rowval[flat_i] < i2
                                    flat_i += 1
                                end
                                if flat_i ≤ last_i && rowval[flat_i] == i2
                                    factors[i1,j1] = nzval[flat_i]
                                    flat_i += 1
                                else
                                    factors[i1,j1] = 0.0
                                end
                            end
                        end
                    elseif isa(A_variable_block, SharedSparseBuffer)
                        colptr = A_variable_block.colptr
                        rowval_list = A_variable_block.rowval_list
                        nzval = A_variable_block.nzval
                        for (j1, j2) ∈ zip(colrange, colinds)
                            first_i = colptr[j2]
                            col_rowval = rowval_list[j2]
                            row_i = max(searchsortedlast(col_rowval, first_row) - 1, 1)
                            last_row = length(col_rowval)
                            for (i1, i2) ∈ zip(rowrange, rowinds)
                                while row_i < last_row && col_rowval[row_i] < i2
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
                        for (j1, j2) ∈ zip(colrange, colinds), (i1, i2) ∈ zip(rowrange, rowinds)
                            factors[i1,j1] = A_variable_block[i2,j2]
                        end
                    end
                end
                getrf!(factors, s.ipiv; check=check_lu)
            end
        end
        return nothing
    end
end
function lu!(block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T},
             full_A::NTuple{Nvar,<:NTuple{Nvar,<:Union{AbstractMatrix{T},AbstractSparseMatrixCSC{T},SharedSparseBuffer{T}}}}) where {Nvar, T}
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        factors = block_diagonal_solver.factors
        block_indices = block_diagonal_solver.block_indices
        block_ranges = block_diagonal_solver.block_ranges
        partial_block_indices = block_diagonal_solver.partial_block_indices
        partial_col_ranges = block_diagonal_solver.partial_col_ranges
        synchronize_shared = block_diagonal_solver.synchronize_shared

        if solver === nothing
            # Nothing to do.
        else
            for (vcol, partial_colrange, partial_colinds) ∈
                        zip(1:Nvar, partial_col_ranges, partial_block_indices),
                    (vrow, rowrange, rowinds) ∈ zip(1:Nvar, block_ranges, block_indices)
                A_variable_block = full_A[vrow][vcol]
                if isa(A_variable_block, AbstractSparseMatrixCSC)
                    colptr = A_variable_block.colptr
                    rowval = A_variable_block.rowval
                    nzval = A_variable_block.nzval
                    first_row = rowinds[1]
                    for (j1, j2) ∈ zip(partial_colrange, partial_colinds)
                        first_i = colptr[j2]
                        last_i = colptr[j2+1]-1
                        flat_i = max(searchsortedlast(@view(rowval[first_i:last_i]), first_row) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ zip(rowrange, rowinds)
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
                elseif isa(A_variable_block, SharedSparseBuffer)
                    colptr = A_variable_block.colptr
                    rowval_list = A_variable_block.rowval_list
                    nzval = A_variable_block.nzval
                    first_row = rowinds[1]
                    for (j1, j2) ∈ zip(partial_colrange, partial_colinds)
                        first_i = colptr[j2]
                        col_rowval = rowval_list[j2]
                        row_i = max(searchsortedlast(col_rowval, first_row) - 1, 1)
                        last_row = length(col_rowval)
                        for (i1, i2) ∈ zip(rowrange, rowinds)
                            while row_i < last_row && col_rowval[row_i] < i2
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
                    for (j1, j2) ∈ zip(partial_colrange, partial_colinds),
                            (i1, i2) ∈ zip(rowrange, rowinds)
                        factors[i1,j1] = A_variable_block[i2,j2]
                    end
                end
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
               block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
               u::AbstractVector{T}) where {Nvar, T}
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if (eltype(solvers) <: SparseArrays.UMFPACK.UmfpackLU)
            u_buffer = block_diagonal_solver.u_buffer
            for (bi, s, buff) ∈ zip(block_diagonal_solver.block_vector_indices, solvers, buffers)
                n = length(bi)
                this_u_buffer = @view u_buffer[1:n]
                for (i1, i2) ∈ enumerate(bi)
                    this_u_buffer[i1] = u[i2]
                end
                ldiv!(buff, s, this_u_buffer)
            end
        elseif !(eltype(solvers) <: Nothing)
            for (bi, s, buff) ∈ zip(block_diagonal_solver.block_vector_indices, solvers, buffers)
                for (i1, i2) ∈ enumerate(bi)
                    buff[i1] = u[i2]
                end
                ldiv!(s, buff)
            end
        end
        return nothing
    end
end
function ldiv!(buffer::AbstractVector{T},
               block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T},
               u::AbstractVector{T}) where {Nvar, T}
    @inbounds begin
        solver = block_diagonal_solver.local_block_solver
        block_comm_rank = block_diagonal_solver.block_comm_rank
        block_vector_indices = block_diagonal_solver.block_vector_indices
        synchronize_shared = block_diagonal_solver.synchronize_shared

        # Need to synchronize here as `u_buffer` is filled only on block_comm_rank==0, but
        # `u` was filled in parallel. Maybe it would be worth filling `u_buffer` in
        # parallel? Then would need to synchronize before `ldiv!()` call.
        synchronize_shared()

        if solver === nothing
            # Nothing to do.
        elseif isa(solver, MPIDenseLU)
            if length(block_vector_indices) == length(u)
                # There is only one block which includes the whole rhs/solution vector, so
                # do not need to select range out of u.
                ldiv!(buffer, solver, u)
            else
                u_buffer = block_diagonal_solver.u_buffer
                if block_comm_rank == 0
                    for (i1, i2) ∈ enumerate(block_vector_indices)
                        u_buffer[i1] = u[i2]
                    end
                end
                ldiv!(buffer, solver, u_buffer)
            end
        else
            if block_comm_rank == 0
                for (i1, i2) ∈ enumerate(block_vector_indices)
                    buffer[i1] = u[i2]
                end
                ldiv!(solver, buffer)
            end
        end
        return nothing
    end
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{Nvar,T},BlockDiagonalSolverShared{Nvar,T}},
               u::AbstractVector{T}) where {Nvar,T}
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(x::AbstractMatrix{T},
               block_diagonal_solver::Union{BlockDiagonalSolverSerial{Nvar,T},BlockDiagonalSolverShared{Nvar,T}},
               u::AbstractMatrix{T}) where {Nvar, T}
    if block_diagonal_solver.local_block_solver !== nothing
        for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
            ldiv!(this_x, block_diagonal_solver, this_u)
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
               u::Matrix{T}) where {Nvar, T}
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if length(solvers) == 1 && length(block_diagonal_solver.block_vector_indices[1]) == size(u, 1)
            # There is only one block which includes the whole rhs/solution vector, so do
            # not need to select range out of x/u.
            ldiv!(x, solvers[1], u)
        else
            if !(eltype(solvers) <: Nothing)
                for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
                    ldiv!(this_x, block_diagonal_solver, this_u)
                end
            end
        end
    end
    return nothing
end
function ldiv!(x::Matrix{T}, block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T},
               u::Matrix{T}) where {Nvar, T}
    for (this_x, this_u) ∈ zip(eachcol(x), eachcol(u))
        ldiv!(this_x, block_diagonal_solver, this_u)
    end
    return nothing
end
function ldiv!(block_diagonal_solver::Union{BlockDiagonalSolverSerial{Nvar,T},BlockDiagonalSolverShared{Nvar,T}},
               u::AbstractMatrix{T}) where {Nvar, T}
    return ldiv!(u, block_diagonal_solver, u)
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T}, u::Matrix{T}) where {Nvar, T}
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if length(solvers) == 1 && length(block_diagonal_solver.block_vector_indices[1]) == size(u, 1)
            # There is only one block, so do not need to select range out of u.
            ldiv!(solvers[1], u)
            return nothing
        else
            return ldiv!(u, block_diagonal_solver, u)
        end
    end
end
function ldiv!(block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T}, u::Matrix{T}) where {Nvar, T}
    return ldiv!(u, block_diagonal_solver, u)
end

# Specialized implementations to be used for A^{-1}.B
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
                       B::NTuple{Nvar,<:NTuple{Nvar,<:AbstractMatrix{T}}}) where {Nvar, T}
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if !(eltype(solvers <: Nothing))
            for (branges, bi, s, Bbuff, Bcolranges, Bcols) ∈
                    zip(block_diagonal_solver.block_ranges,
                        block_diagonal_solver.block_indices, solvers,
                        block_diagonal_solver.B_buffers_out,
                        block_diagonal_solver.B_column_indices)
                for (vcol, vBcolrange, vBcols) ∈ zip(1:Nvar, Bcolranges, Bcols),
                        (vrow, vrowrange, vrowinds) ∈ zip(1:Nvar, branges, bi)
                    B_variable_block = B[vrow][vcol]
                    for (j1, j2) ∈ zip(vBcolrange, vBcols), (i1, i2) ∈ zip(vrowrange,
                                                                           vrowinds)
                        Bbuff[i1,j1] = B_variable_block[i2,j2]
                    end
                end
                ldiv!(s, Bbuff)
                for (vcol, vBcolrange, vBcols) ∈ zip(1:Nvar, Bcolranges, Bcols),
                        (vrow, vrowrange, vrowinds) ∈ zip(1:Nvar, branges, bi)
                    B_variable_block = B[vrow][vcol]
                    for (j1, j2) ∈ zip(vBcolrange, vBcols), (i1, i2) ∈ zip(vrowrange,
                                                                           vrowinds)
                        B_variable_block[i2,j2] = Bbuff[i1,j1]
                    end
                end
            end
        end
        return nothing
    end
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
                       B::NTuple{Nvar,<:NTuple{Nvar,<:AbstractSparseMatrixCSC{T}}}) where {Nvar, T}
    @inbounds begin
        solvers = block_diagonal_solver.local_block_solver
        if !(eltype(solvers <: Nothing))
            for (branges, bi, s, Bbuff, Bcolranges, Bcols) ∈
                    zip(block_diagonal_solver.block_ranges,
                        block_diagonal_solver.block_indices, solvers,
                        block_diagonal_solver.B_buffers_out,
                        block_diagonal_solver.B_column_indices)
                for (vcol, vBcolrange, vBcols) ∈ zip(1:Nvar, Bcolranges, Bcols),
                        (vrow, vrowrange, vrowinds) ∈ zip(1:Nvar, branges, bi)
                    B_variable_block = B[vrow][vcol]
                    B_colptr = B_variable_block.colptr
                    B_rowval = B_variable_block.rowval
                    B_nzval = B_variable_block.nzval
                    firstrow = first(vrowinds)
                    for (j1, j2) ∈ zip(vBcolrange, vBcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ zip(vrowrange, vrowinds)
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
                end
                ldiv!(s, Bbuff)
                for (vcol, vBcolrange, vBcols) ∈ zip(1:Nvar, Bcolranges, Bcols),
                        (vrow, vrowrange, vrowinds) ∈ zip(1:Nvar, branges, bi)
                    B_variable_block = B[vrow][vcol]
                    B_colptr = B_variable_block.colptr
                    B_rowval = B_variable_block.rowval
                    B_nzval = B_variable_block.nzval
                    firstrow = first(vrowinds)
                    for (j1, j2) ∈ zip(vBcolrange, vBcols)
                        first_i = B_colptr[j2]
                        last_i = B_colptr[j2+1] - 1
                        col_rv = @view B_rowval[first_i:last_i]
                        flat_i = max(searchsortedlast(col_rv, firstrow) - 1, 1) + first_i - 1
                        for (i1, i2) ∈ zip(vrowrange, vrowinds)
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
            end
        end
        return nothing
    end
end
function ldiv_block_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverSerial{Nvar,T},
                             B::BlockAinvDotBSerial{Nvar,T}) where {Nvar,T}
    for (solver, block) ∈ zip(block_diagonal_solver.local_block_solver, B.blocks)
        ldiv!(solver, block)
    end
    return nothing
end
function ldiv_block_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T},
                             B::BlockAinvDotBShared{Nvar,T}) where {Nvar,T}
    @inbounds begin
        solver = block_diagonal_solver.local_block_serial_solver
        if solver !== nothing
            block = B.block
            partial_block = B.partial_block
            partial_vector_col_range = B.partial_vector_col_range
            partial_vector_row_range = B.partial_vector_row_range
            synchronize_shared = B.synchronize_shared

            # Probably more efficient to parallelise over columns in `block` than to use a
            # parallelised `ldiv!()` on the full block.
            ldiv!(solver, @view(block[:,partial_vector_col_range]))

            synchronize_shared()

            partial_block .= @view block[partial_vector_row_range,:]
        end
        return nothing
    end
end
function ldiv_block_Bmatrix!(block_diagonal_solver::MPIStaticCondensationNull{T},
                             B) where T
    return nothing
end
function ldiv_Bmatrix!(block_diagonal_solver::BlockDiagonalSolverShared{Nvar,T},
                       B::Matrix{T}) where {Nvar,T}
    # When not using BlockAinvDotBShared, this function will use a different
    # parallelisation than copy_B_submatrix!(), so need to synchronize.
    block_diagonal_solver.synchronize_shared()
    return ldiv!(block_diagonal_solver, B)
end
