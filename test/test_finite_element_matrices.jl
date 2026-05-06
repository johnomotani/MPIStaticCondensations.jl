using MPIStaticCondensations
using MPIStaticCondensations: Dimension
using MPI
using Primes
using StableRNGs
using Test

include("generate_finite_element_matrices.jl")
include("utils.jl")

function test_matrix(dimensions::Vector{<:Dimension}, n_shared::Integer,
                     random_seed::Integer, use_sparse::Bool,
                     optimize_schur_complement_size::Bool)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    rng = StableRNG(random_seed)

    global_matrix, local_matrix =
        assemble_and_scatter_global_matrix(dimensions, comm, distributed_comm,
                                           shared_comm, allocate_shared_float, rng)
    rhs_global, rhs_local =
        assemble_and_scatter_global_rhs(dimensions, comm, distributed_comm, shared_comm,
                                        allocate_shared_float, rng)
    if distributed_rank == 0 && shared_rank == 0
        x_global = similar(rhs_global)
    else
        x_global = nothing
    end

    Alu = mpi_static_condensation(dimensions; comm, distributed_comm, shared_comm,
                                  allocate_shared_float, allocate_shared_int, use_sparse,
                                  optimize_schur_complement_size, check_lu=true)

    lu!(Alu, local_matrix)

    @testset "solve" begin
        ldiv!(Alu, rhs_local)
        x_global = gather_vector!(x_global, rhs_local, dimensions, distributed_comm,
                                  shared_comm)
        if distributed_rank == 0 && shared_rank == 0
            check_soluion = global_matrix \ rhs_global
            @test isapprox(x_global, check_solution;
                           norm=(x)->NaN, rtol=1.0e-13, atol=1.0e-13)
            @test isapprox(global_matrix * x_global, rhs_global;
                           norm=(x)->NaN, rtol=1.0e-13, atol=1.0e-13)
        end
    end

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

function test_finite_element_matrices()
    if !MPI.Initialized()
        MPI.Init()
    end
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    comm_size = MPI.Comm_size(MPI.COMM_WORLD)
    dimensions = [create_dimension(; nelement=4, ngrid=3, nrank=comm_size, irank=rank, periodic=false, remove_boundaries=false)]
    test_matrix(dimensions, 1, 987, true, true)
end
