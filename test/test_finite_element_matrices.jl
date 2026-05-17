using MPIStaticCondensations
using MPIStaticCondensations: Dimension
using Combinatorics
using LinearAlgebra
using MPI
using Primes
using StableRNGs
using Test

include("generate_finite_element_matrices.jl")
include("utils.jl")

function test_matrix(dimensions::Vector{<:Dimension}, n_shared::Integer,
                     random_seed::Integer, use_sparse::Bool, sparse_stencils::Bool,
                     tol::AbstractFloat)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    rng = StableRNG(random_seed)

    global_matrix, local_matrix =
        assemble_and_scatter_global_matrix(dimensions, comm, distributed_comm,
                                           shared_comm, allocate_shared_float,
                                           allocate_shared_int, rng, sparse_stencils)
    rhs_global, rhs_local =
        assemble_and_scatter_global_rhs(dimensions, comm, distributed_comm, shared_comm,
                                        allocate_shared_float, rng)

    Alu = mpi_static_condensation(dimensions; comm, distributed_comm, shared_comm,
                                  allocate_shared_float, allocate_shared_int, use_sparse,
                                  check_lu=true)

    lu!(Alu, local_matrix)

    @testset "solve" begin
        ldiv!(Alu, rhs_local)
        x_global = gather_vector(rhs_local, dimensions, comm, distributed_comm,
                                 shared_comm)
        if distributed_rank == 0 && shared_rank == 0
            check_solution = global_matrix \ rhs_global
            @test isapprox(x_global, check_solution;
                           norm=(x)->NaN, rtol=tol, atol=tol)
            @test isapprox(global_matrix * x_global, rhs_global;
                           norm=(x)->NaN, rtol=tol, atol=tol)
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
    MPI.Barrier(shared_comm)
    return nothing
end

function test_dimension_combinations(nelement_list, ngrid_list, max_nproc, rank,
                                     comm_size, n_shared, tol, this_seed;
                                     all_use_sparse=true,
                                     all_sparse_stencils=true, all_periodic=true,
                                     all_remove_boundaries=true)
    if length(nelement_list) != length(ngrid_list)
        error("nelement_list and ngrid_list must have the same length")
    end

    if comm_size > max_nproc
        # It may not be possible to split the grids defined by these parameters into
        # `comm_size` pieces, so skip.
        return nothing
    end

    distributed_comm_size = comm_size ÷ n_shared
    distributed_comm_rank = rank ÷ n_shared

    bool_perms = generate_bool_permutations(length(nelement_list))
    if all_use_sparse
        use_sparse_list = (true, false)
    else
        use_sparse_list = (true,)
    end
    if all_sparse_stencils
        sparse_stencils_list = (true, false)
    else
        sparse_stencils_list = (true,)
    end
    @testset "nelement_list=$nelement_list, ngrid_list=$ngrid_list, use_sparse=$use_sparse, sparse_stencils=$sparse_stencils" for
            use_sparse ∈ use_sparse_list,
            sparse_stencils ∈ sparse_stencils_list
        if rank == 0
            println("* n_shared=$n_shared, nelement_list=$nelement_list, ngrid_list=$ngrid_list, use_sparse=$use_sparse, sparse_stencils=$sparse_stencils")
        end

        @testset "this_nelement_list=$this_nelement_list, this_ngrid_list=$this_ngrid_list, this_nrank_list=$this_nrank_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" for
                this_nelement_list ∈ multiset_permutations(nelement_list),
                this_ngrid_list ∈ multiset_permutations(ngrid_list),
                this_nrank_list ∈ get_nrank_permutations(this_nelement_list, distributed_comm_size),
                periodic_list ∈ (all_periodic ? bool_perms : (fill(false, length(this_nelement_list)),)),
                remove_boundaries_list ∈ (all_remove_boundaries ? bool_perms : (fill(false, length(this_nelement_list)),))
            if rank == 0
                println("  - n_shared=$n_shared, ($use_sparse, $sparse_stencils), this_nelement_list=$this_nelement_list, this_ngrid_list=$this_ngrid_list, this_nrank_list=$this_nrank_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list")
            end

            this_irank_list = get_iranks(this_nrank_list, distributed_comm_rank)
            dimensions = [create_dimension(; nelement, ngrid, nrank, irank, periodic, remove_boundaries)
                          for (nelement, ngrid, irank, nrank, periodic, remove_boundaries)
                          ∈ zip(this_nelement_list, this_ngrid_list, this_irank_list, this_nrank_list, periodic_list, remove_boundaries_list)]

            test_matrix(dimensions, n_shared, this_seed, use_sparse, sparse_stencils, tol)
            this_seed += 1
        end
    end
end

function test_finite_element_matrices()
    if !MPI.Initialized()
        MPI.Init()
    end
    BLAS.set_num_threads(1)
    @testset "finite element matrices" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        comm_size = MPI.Comm_size(MPI.COMM_WORLD)
        @testset "n_shared=$n_shared" for n_shared ∈ [prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
            @testset "1D" begin
                tol = 1.0e-12
                test_dimension_combinations([1], [3], 1, rank, comm_size, n_shared, tol, 1000)
                test_dimension_combinations([2], [3], 2, rank, comm_size, n_shared, tol, 1001)
                test_dimension_combinations([2], [4], 2, rank, comm_size, n_shared, tol, 1002)
                test_dimension_combinations([2], [5], 2, rank, comm_size, n_shared, tol, 1003)
                test_dimension_combinations([3], [3], 3, rank, comm_size, n_shared, tol, 1004)
                test_dimension_combinations([4], [3], 4, rank, comm_size, n_shared, tol, 1005)
                test_dimension_combinations([5], [3], 5, rank, comm_size, n_shared, tol, 1006)
                test_dimension_combinations([6], [3], 6, rank, comm_size, n_shared, tol, 1007)
                test_dimension_combinations([7], [3], 5, rank, comm_size, n_shared, tol, 1008)
                test_dimension_combinations([8], [3], 8, rank, comm_size, n_shared, tol, 1009)
                test_dimension_combinations([16], [3], 16, rank, comm_size, n_shared, tol, 1010)
                test_dimension_combinations([32], [3], 32, rank, comm_size, n_shared, tol, 1011)
            end
            @testset "2D" begin
                tol = 2.0e-7
                test_dimension_combinations([1, 1], [3, 3], 1, rank, comm_size, n_shared, tol, 2000)
                test_dimension_combinations([1, 2], [3, 3], 2, rank, comm_size, n_shared, tol, 2001)
                test_dimension_combinations([1, 2], [3, 5], 2, rank, comm_size, n_shared, tol, 2002)
                test_dimension_combinations([1, 3], [3, 5], 3, rank, comm_size, n_shared, tol, 2003)
                test_dimension_combinations([2, 2], [3, 5], 4, rank, comm_size, n_shared, tol, 2004)
                test_dimension_combinations([2, 3], [3, 5], 4, rank, comm_size, n_shared, tol, 2005)
                test_dimension_combinations([2, 4], [3, 5], 8, rank, comm_size, n_shared, tol, 2006)
                test_dimension_combinations([1, 8], [3, 5], 8, rank, comm_size, n_shared, tol, 2007)
                test_dimension_combinations([1, 16], [3, 5], 16, rank, comm_size, n_shared, tol, 2008)
                test_dimension_combinations([2, 8], [3, 5], 16, rank, comm_size, n_shared, tol, 2009)
                test_dimension_combinations([4, 4], [3, 5], 16, rank, comm_size, n_shared, tol, 2010)
                test_dimension_combinations([4, 4], [5, 5], 16, rank, comm_size, n_shared, tol, 2011)
                test_dimension_combinations([1, 32], [3, 5], 32, rank, comm_size, n_shared, tol, 2012)
                test_dimension_combinations([2, 16], [3, 5], 32, rank, comm_size, n_shared, tol, 2013)
                test_dimension_combinations([4, 8], [3, 5], 32, rank, comm_size, n_shared, tol, 2014)
                test_dimension_combinations([4, 8], [5, 5], 32, rank, comm_size, n_shared, tol, 2015)
            end
            @testset "3D" begin
                tol = 1.0e-7
                test_dimension_combinations([1, 1, 1], [3, 3, 3], 1, rank, comm_size, n_shared, tol, 3000; all_use_sparse=false, all_sparse_stencils=false)
                test_dimension_combinations([2, 2, 2], [3, 4, 5], 8, rank, comm_size, n_shared, tol, 3001; all_use_sparse=false, all_sparse_stencils=false)
                test_dimension_combinations([2, 3, 4], [3, 4, 5], 24, rank, comm_size, n_shared, tol, 3002; all_use_sparse=false, all_sparse_stencils=false)
                if comm_size ≥ 8
                    test_dimension_combinations([8, 8, 8], [3, 4, 5], 512, rank, comm_size, n_shared, tol, 3003; all_use_sparse=false, all_sparse_stencils=false, all_periodic=false, all_remove_boundaries=false)
                end
                if comm_size ≥ 16
                    test_dimension_combinations([9, 9, 32], [3, 3, 3], 2592, rank, comm_size, n_shared, tol, 3004; all_use_sparse=false, all_sparse_stencils=false, all_periodic=false, all_remove_boundaries=false)
                end
            end
        end
    end
end
