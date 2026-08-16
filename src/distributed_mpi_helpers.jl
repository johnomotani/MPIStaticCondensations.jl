function gather_matrix!(solver::MPIStaticCondensationParallel, A_in::AbstractMatrix)
end

function gather_rhs_vector!(solver::MPIStaticCondensationParallel, U_in::AbstractVector)
    distributed_comm = solver.distributed_comm
    distributed_comm_rank = solver.distributed_comm_rank
    distributed_comm_size = solver.distributed_comm_size
    gather_no_multiple_overlap = solver.gather_no_multiple_overlap

    if distributed_comm_rank == 0
        U = solver.vector_buffer
        vector_gather_to_ranges = solver.vector_gather_to_ranges
        vector_gather_from_ranges = solver.vector_gather_from_ranges
        vector_reduce_to_ranges = solver.vector_reduce_to_ranges
        vector_reduce_from_ranges = solver.vector_reduce_from_ranges
        comm_reqs = solver.comm_reqs
        if gather_no_multiple_overlap
            # Special case that can use a more optimised communication pattern.
            req_count = 0
            for r ∈ 1:distributed_comm_size-1
                @views MPI.Irecv!(U[vector_gather_to_ranges[r+1]], distributed_comm,
                                  comm_reqs[req_count+=1]; source=r, tag=1)
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
            for (gather_to_range, gather_from_range, reduce_to_range, reduce_from_range)
                    ∈ zip(@view(vector_gather_to_ranges[2:end]),
                          @view(vector_gather_from_ranges[2:end]),
                          @view(vector_reduce_to_ranges[2:end]),
                          @view(vector_reduce_from_ranges[2:end]))
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
            MPI.Send(U_in, distributed_comm; dest=0, tag=1)
        else
            MPI.Gatherv!(U_in, nothing, distributed_comm; root=0)
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
