function gather_matrix!(solver::MPIStaticCondensationParallel, A_in::AbstractMatrix)
    distributed_comm = solver.distributed_comm
    distributed_comm_rank = solver.distributed_comm_rank
    distributed_comm_size = solver.distributed_comm_size
    vector_gather_from_ranges = solver.vector_gather_from_ranges

    if distributed_comm_rank == 0
        A = solver.matrix_buffer
        gather_no_multiple_overlap = solver.gather_no_multiple_overlap
        matrix_gather_to_row_ranges = solver.matrix_gather_to_row_ranges
        vector_reduce_to_ranges = solver.vector_reduce_to_ranges
        vector_reduce_from_ranges = solver.vector_reduce_from_ranges
        matrix_reduce_from_row_ranges = solver.matrix_reduce_from_row_ranges

        if gather_no_multiple_overlap
            # Special case that can use a more optimised communication pattern.
            req_count = 0
            for r ∈ 1:distributed_comm_size-1
                this_column_range = vector_gather_to_ranges[r+1]
                this_row_range = matrix_gather_to_row_ranges[r+1]
                @views MPI.Irecv!(A[this_row_range, this_column_range], distributed_comm,
                                  comm_reqs[req_count+=1]; source=r)
            end
            gather_to_column_range = vector_gather_to_ranges[1]
            gather_from_column_range = vector_gather_from_ranges[1]
            gather_to_row_range = matrix_gather_to_row_ranges[1]
            for (j1, j2) ∈ zip(gather_to_column_range, gather_from_column_range),
                    (i2, i1) ∈ enumerate(gather_to_row_range)
                A[i1,j1] = A_in[i2,j2]
            end
            MPI.Waitall(comm_reqs)
            if !isempty(vector_reduce_to_ranges)
                reduce_to_column_range = vector_reduce_to_ranges[1]
                reduce_from_column_range = vector_reduce_from_ranges[1]
                reduce_from_row_range = matrix_reduce_from_row_ranges[1]
                for (j1, j2) ∈ zip(reduce_to_column_range, reduce_from_column_range),
                        (i2, i1) ∈ enumerate(reduce_from_row_range)
                    A[i1,j1] += A_in[i2,j2]
                end
            end
        else
            # Don't want to allocate a buffer to receive all the remotely-owned parts of
            # the matrix at once, to reduce memory usage, so receive and copy in the
            # remotely-owned parts one after the other. Assume that `A_in` is big enough
            # to store any of the remotely-owned parts - its values are not needed any
            # more after they have been copied into `A`, and `A_in` is at least as big on
            # the root process as it is on any other.
            #
            # If we wanted to optimise this branch a bit, in principle we could have two
            # buffers so that we could overlap communication (receiving into one buffer)
            # with 'computation' (copying/adding out of the other buffer).
            root_gather_to_column_range = vector_gather_to_ranges[1]
            root_gather_from_column_range = vector_gather_from_ranges[1]
            root_gather_to_row_range = matrix_gather_to_row_ranges[1]
            for (j1, j2) ∈ zip(root_gather_to_column_range, root_gather_from_column_range),
                    (i2, i1) ∈ enumerate(root_gather_to_row_range)
                A[i1,j1] = A_in[i2,j2]
            end

            receive_column_offset = root_gather_from_column_range[1] - 1
            for r ∈ 1:distributed_comm_size-1
                gather_to_column_range = vector_gather_to_ranges[r+1]
                gather_from_column_range = vector_gather_from_ranges[r+1]
                reduce_to_column_range = reudce_gather_to_ranges[r+1]
                receive_from_column_range = receive_column_offset+1:receive_column_offset+length(gather_to_column_range)+length(reduce_to_column_range)
                gather_to_row_range = matrix_gather_to_row_ranges[r+1]
                gather_from_row_range = matrix_gather_from_row_ranges[r+1]
                reduce_to_row_range = matrix_reduce_to_row_ranges[r+1]
                reduce_from_row_range = matrix_reduce_from_row_ranges[r+1]
                receive_from_row_range = 1:length(gather_to_row_range)+length(reduce_to_row_range)

                receive_buffer = @view(A_in[receive_from_row_range,receive_from_column_range])
                MPI.Recv!(receive_buffer, distributed_comm; source=r)
                for (j1, j2) ∈ zip(gather_to_column_range, gather_from_column_range),
                        (i1, i2) ∈ zip(gather_to_row_range, gather_from_row_range)
                    A[i1,j1] = receive_buffer[i2,j2]
                end
                for (j1, j2) ∈ zip(reduce_to_column_range, reduce_from_column_range),
                        (i1, i2) ∈ zip(reduce_to_row_range, reduce_from_row_range)
                    A[i1,j1] += receive_buffer[i2,j2]
                end
            end
        end
    else
        gather_from_column_range = vector_gather_from_ranges[1]
        MPI.Send(@view(A_in[:,gather_from_column_range]), distributed_comm; dest=0)
    end

    return nothing
end

function gather_rhs_vector!(solver::MPIStaticCondensationParallel, U_in::AbstractVector)
    distributed_comm = solver.distributed_comm
    distributed_comm_rank = solver.distributed_comm_rank
    distributed_comm_size = solver.distributed_comm_size
    gather_no_multiple_overlap = solver.gather_no_multiple_overlap
    vector_gather_from_ranges = solver.vector_gather_from_ranges

    if distributed_comm_rank == 0
        U = solver.vector_buffer
        vector_gather_to_ranges = solver.vector_gather_to_ranges
        vector_reduce_to_ranges = solver.vector_reduce_to_ranges
        vector_reduce_from_ranges = solver.vector_reduce_from_ranges
        comm_reqs = solver.comm_reqs
        if gather_no_multiple_overlap
            # Special case that can use a more optimised communication pattern.
            req_count = 0
            for r ∈ 1:distributed_comm_size-1
                @views MPI.Irecv!(U[vector_gather_to_ranges[r+1]], distributed_comm,
                                  comm_reqs[req_count+=1]; source=r)
            end
            for (i1, i2) ∈ zip(vector_gather_to_ranges[1], vector_gather_from_ranges[1])
                U[i1] = U_in[i2]
            end
            MPI.Waitall(comm_reqs)
            if !isempty(vector_reduce_to_ranges)
                for (i1, i2) ∈ zip(vector_reduce_to_ranges[1], vector_reduce_from_ranges[1])
                    U[i1] += U_in[i2]
                end
            end
        else
            vector_gather_buffer = solver.vector_gather_buffer
            if !isa(vector_gather_buffer, MPI.VBuffer)
                error("wrong type!!!!")
            end
            temp_Igatherv!(MPI.IN_PLACE, vector_gather_buffer, distributed_comm,
                           comm_reqs[1]; root=0)
            for (i1, i2) ∈ zip(vector_gather_to_ranges[1], vector_gather_from_ranges[1])
                U[i1] = U_in[i2]
            end
            MPI.Waitall(comm_reqs)
            gathered_data = vector_gather_buffer.data
            for r_plus_one ∈ 2:distributed_comm_size
                gather_to_range = vector_gather_to_ranges[r_plus_one]
                gather_from_range = vector_gather_from_ranges[r_plus_one]
                reduce_to_range = vector_reduce_to_ranges[r_plus_one]
                reduce_from_range = vector_reduce_from_ranges[r_plus_one]
                for (i1, i2) ∈ zip(gather_to_range, gather_from_range)
                    U[i1] = gathered_data[i2]
                end
                for (i1, i2) ∈ zip(reduce_to_range, reduce_from_range)
                    U[i1] += gathered_data[i2]
                end
            end
        end
    else
        if gather_no_multiple_overlap
            # Special case that can use a more optimised communication pattern.
            MPI.Send(@view(U_in[vector_gather_from_ranges[1]]), distributed_comm; dest=0)
        else
            MPI.Gatherv!(@view(U_in[vector_gather_from_ranges[1]]), nothing, distributed_comm; root=0)
        end
    end

    return nothing
end

function scatter_solution_vector!(solver::MPIStaticCondensationParallel,
                                  X_out::AbstractVector)
    distributed_comm = solver.distributed_comm
    distributed_comm_rank = solver.distributed_comm_rank
    distributed_comm_size = solver.distributed_comm_size

    if distributed_comm_rank == 0
        X = solver.vector_buffer
        vector_scatter_from_ranges = solver.vector_scatter_from_ranges
        comm_reqs = solver.comm_reqs
        req_count = 0
        if eltype(vector_scatter_from_ranges) <: UnitRange
            # Special case that can use a more optimised communication pattern.
            for (r, scatter_range) ∈ zip(1:distributed_comm_size-1,
                                         @view(vector_scatter_from_ranges[2:end]))
                MPI.Isend(@view(X[scatter_range]), distributed_comm,
                          comm_reqs[req_count+=1]; dest=r)
            end
        else
            vector_scatter_to_ranges = solver.vector_scatter_to_ranges
            vector_gather_buffer = solver.vector_gather_buffer
            for (r, scatter_to, scatter_from) ∈
                    zip(1:distributed_comm_size-1, @view(vector_scatter_to_ranges[2:end]),
                        @view(vector_scatter_from_ranges[2:end]))
                for (i1, i2) ∈ zip(scatter_to, scatter_from)
                    vector_gather_buffer[i1] = X[i2]
                end
                MPI.Isend(@view(vector_gather_buffer[scatter_to]), distributed_comm,
                          comm_reqs[req_count+=1]; dest=r)
            end
        end
        for (i1, i2) ∈ enumerate(vector_scatter_from_ranges[1])
            X_out[i1] = X[i2]
        end
        MPI.Waitall(comm_reqs)
    else
        MPI.Recv(X_out, distributed_comm; source=0)
    end
    return nothing
end

temp_Ireduce!(sendrecvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendrecvbuf, op, root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendbuf, recvbuf, op, root, comm, req)
function temp_Ireduce!(rbuf::MPI.RBuffer, op::Union{MPI.Op,MPI.MPI_Op}, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    # int MPI_Ireduce(const void* sendbuf, void* recvbuf, int count,
    #                 MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
    #                 MPI_Request* req)
    MPI.API.MPI_Ireduce(rbuf.senddata, rbuf.recvdata, rbuf.count, rbuf.datatype, op, root, comm, req)
    MPI.setbuffer!(req, rbuf)
    return req
end
temp_Ireduce!(rbuf::MPI.RBuffer, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(rbuf, MPI.Op(op, eltype(rbuf)), root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(MPI.RBuffer(sendbuf, recvbuf), op, root, comm, req)
# inplace
function temp_Ireduce!(buf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    if MPI.Comm_rank(comm) == root
        temp_Ireduce!(MPI.IN_PLACE, buf, op, root, comm, req)
    else
        temp_Ireduce!(buf, nothing, op, root, comm, req)
    end
end

# Ireduce!() interface is defined in https://github.com/JuliaParallel/MPI.jl/pull/827,
# which is not yet merged.
# Copy a similar pattern to provide IGatherv!()
temp_Igatherv!(sendbuf, recvbuf, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Igatherv!(sendbuf, recvbuf, root, comm, req)
temp_Igatherv!(sendbuf, recvbuf::Nothing, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Igatherv!(sendbuf, VBuffer(nothing), root, comm, req)
function temp_Igatherv!(sendbuf::MPI.Buffer, recvbuf::MPI.VBuffer, root::Integer,
                        comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    # int MPI_Igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
    #                 void* recvbuf, const int recvcounts[], const int displs[],
    #                 MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* req)
    MPI.API.MPI_Igatherv(sendbuf.data, sendbuf.count, sendbuf.datatype, recvbuf.data,
                         recvbuf.counts, recvbuf.displs, recvbuf.datatype, root, comm,
                         req)
    MPI.setbuffer!(req, recvbuf)
    return req
end
