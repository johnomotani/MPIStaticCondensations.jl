function get_comms(shared_nproc, with_comm=false)
    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    distributed_nproc, rem = divrem(nproc, shared_nproc)
    if rem != 0
        error("shared_nproc=$shared_nproc does not divide nproc=$nproc")
    end
    distributed_rank, shared_rank = divrem(rank, shared_nproc)
    shared_comm = MPI.Comm_split(MPI.COMM_WORLD, distributed_rank, shared_rank)
    distributed_comm = MPI.Comm_split(MPI.COMM_WORLD, shared_rank, distributed_rank)

    local_win_store_float = MPI.Win[]
    if shared_comm == MPI.COMM_SELF && !with_comm
        allocate_array_float = (args...)->zeros(Float64, args...)
    else
        allocate_array_float = (dims...; comm=shared_comm)->begin
            this_shared_rank = MPI.Comm_rank(comm)
            if this_shared_rank == 0
                dims_local = dims
            else
                dims_local = Tuple(0 for _ ∈ dims)
            end
            win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local, comm)
            array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
            push!(local_win_store_float, win)
            if this_shared_rank == 0
                array .= NaN
            end
            MPI.Barrier(comm)
            return array
        end
    end

    local_win_store_int = MPI.Win[]
    if shared_comm == MPI.COMM_SELF && !with_comm
        allocate_array_int = (args...)->zeros(Float64, args...)
    else
        allocate_array_int = (dims...; comm=shared_comm)->begin
            this_shared_rank = MPI.Comm_rank(comm)
            if this_shared_rank == 0
                dims_local = dims
            else
                dims_local = Tuple(0 for _ ∈ dims)
            end
            win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local, comm)
            array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
            push!(local_win_store_int, win)
            if this_shared_rank == 0
                array .= typemin(Int64)
            end
            MPI.Barrier(comm)
            return array
        end
    end

    return comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
           shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
           local_win_store_float, local_win_store_int
end

function cleanup_shared_arrays!(local_win_store_float, local_win_store_int)
    if local_win_store_float !== nothing
        # Free the MPI.Win objects, because if they are free'd by the garbage collector
        # it may cause an MPI error or hang.
        for w ∈ local_win_store_float
            MPI.free(w)
        end
        resize!(local_win_store_float, 0)
    end
    if local_win_store_int !== nothing
        # Free the MPI.Win objects, because if they are free'd by the garbage collector
        # it may cause an MPI error or hang.
        for w ∈ local_win_store_int
            MPI.free(w)
        end
        resize!(local_win_store_int, 0)
    end
    return nothing
end
