using MPIStaticCondensations
using MPIStaticCondensations: Dimension, FastSlow, LevelMultiplier
using Combinatorics
using LinearAlgebra
using MPI
using MUMPS
using Primes
using StableRNGs
using Test

include("generate_finite_element_matrices.jl")
include("utils.jl")

const stencil_matrix = ["element" "empty"   "point"   "element";
                        "element" "point"   "element" "empty";
                        "empty"   "element" "element" "point";
                        "point"   "element" "empty"   "element"]

function test_multivariable_matrix(
             dimensions::Vector{<:Dimension}, variable_dimensions::Tuple,
             n_shared::Integer, random_seed::Integer, sparse_stencils::Bool,
             block_sizes_heuristic, reduce_proc_count_with_blocks::Bool,
             sparse_C_blocks::Bool, mumps_fill_in_threshold::AbstractFloat,
             tol::AbstractFloat)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    rng = StableRNG(random_seed)

    global_matrix, local_matrix =
        assemble_and_scatter_global_multi_variable_matrix(
            dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
            allocate_shared_float, allocate_shared_int, rng, sparse_stencils;
            stencil_matrix=stencil_matrix)
    rhs_global, rhs_local =
        assemble_and_scatter_global_multi_variable_rhs(
            dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
            allocate_shared_float, rng)
    x_local = allocate_shared_float(size(rhs_local)...)

    max_nelement = maximum(d.nelement for d ∈ dimensions)
    if mumps_fill_in_threshold < 1.0 && reduce_proc_count_with_blocks
        @test_throws "reduce_proc_count_with_blocks=true is not compatible with using a MUMPS solver for the lowest level." begin
            mpi_static_condensation(dimensions; variable_dimensions,
                                    block_sizes_heuristic, reduce_proc_count_with_blocks,
                                    sparse_C_blocks, mumps_fill_in_threshold, comm,
                                    distributed_comm, shared_comm, allocate_shared_float,
                                    allocate_shared_int, check_lu=true)
        end
        cleanup_shared_arrays!(local_win_store_float, local_win_store_int)
        return nothing
    end
    if mumps_fill_in_threshold < 1.0 && any(d.periodic for d ∈ dimensions)
        @test_throws "MPIStaticCondensationMUMPS does not currently support periodicity." begin
            mpi_static_condensation(dimensions; variable_dimensions,
                                    block_sizes_heuristic, reduce_proc_count_with_blocks,
                                    sparse_C_blocks, mumps_fill_in_threshold, comm,
                                    distributed_comm, shared_comm, allocate_shared_float,
                                    allocate_shared_int, check_lu=true)
        end
        cleanup_shared_arrays!(local_win_store_float, local_win_store_int)
        return nothing
    end
    Alu = mpi_static_condensation(dimensions; variable_dimensions, block_sizes_heuristic,
                                  reduce_proc_count_with_blocks, sparse_C_blocks,
                                  mumps_fill_in_threshold, comm, distributed_comm,
                                  shared_comm, allocate_shared_float, allocate_shared_int,
                                  check_lu=true)

    lu!(Alu, local_matrix)

    function test_once(two_term::Bool)
        if two_term
            ldiv!(Alu, rhs_local)
            solution = rhs_local
        else
            ldiv!(x_local, Alu, rhs_local)
            solution = x_local
        end
        MPI.Barrier(shared_comm)
        x_global = gather_vector(solution, dimensions, variable_dimensions, comm,
                                 distributed_comm, shared_comm)
        if distributed_rank == 0 && shared_rank == 0
            check_solution = global_matrix \ rhs_global
            @test isapprox(x_global, check_solution;
                           norm=(x)->NaN, rtol=tol, atol=tol)
            @test isapprox(global_matrix * x_global, rhs_global;
                           norm=(x)->NaN, rtol=tol, atol=tol)
        end
    end

    @testset "solve" begin
        test_once(true)
    end

    @testset "change b" begin
        rhs_global, rhs_local =
            assemble_and_scatter_global_multi_variable_rhs(
                dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                allocate_shared_float, rng)
        MPI.Barrier(shared_comm)

        test_once(false)
    end

    @testset "change M" begin
        global_matrix, local_matrix =
            assemble_and_scatter_global_multi_variable_matrix(
                dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                allocate_shared_float, allocate_shared_int, rng, sparse_stencils;
                stencil_matrix=stencil_matrix)
        rhs_global, rhs_local =
            assemble_and_scatter_global_multi_variable_rhs(
                dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                allocate_shared_float, rng)
        MPI.Barrier(shared_comm)

        lu!(Alu, local_matrix)

        test_once(false)
    end

    @testset "change M, change b" begin
        rhs_global, rhs_local =
            assemble_and_scatter_global_multi_variable_rhs(
                dimensions, variable_dimensions, comm, distributed_comm, shared_comm,
                allocate_shared_float, rng)
        MPI.Barrier(shared_comm)

        test_once(true)
    end

    cleanup_shared_arrays!(local_win_store_float, local_win_store_int)
    finalize_mpi_static_condensation!(Alu)
    MPI.Barrier(shared_comm)
    return nothing
end

function test_multivariable_dimension_combinations(
             nelement_list, ngrid_list, variable_dimensions, rank, comm_size, n_shared,
             tol, this_seed; all_sparse_stencils=true, all_block_sizes_heuristics=true,
             all_periodic=true, all_dense_boundaries=true, both_remove_procs=true)
    if length(nelement_list) != length(ngrid_list)
        error("nelement_list and ngrid_list must have the same length")
    end

    distributed_comm_size = comm_size ÷ n_shared
    distributed_comm_rank = rank ÷ n_shared

    bool_perms = generate_bool_permutations(length(nelement_list))
    if all_sparse_stencils
        sparse_stencils_list = (true, false)
    else
        sparse_stencils_list = (true,)
    end
    if all_block_sizes_heuristics
        block_sizes_heuristic_list = (FastSlow(), LevelMultiplier())
    else
        block_sizes_heuristic_list = (FastSlow(), )
    end
    if both_remove_procs
        reduce_proc_count_with_blocks_list = (false, true)
    else
        reduce_proc_count_with_blocks_list = (false,)
    end
    @testset "ne=$nelement_list, ngr=$ngrid_list, sp_sten=$sparse_stencils, red_proc=$reduce_proc_count_with_blocks" for
            sparse_stencils ∈ sparse_stencils_list,
            reduce_proc_count_with_blocks ∈ reduce_proc_count_with_blocks_list
        if rank == 0
            println("* n_sh=$n_shared, ne=$nelement_list, ngr=$ngrid_list, sp_sten=$sparse_stencils, red_proc=$reduce_proc_count_with_blocks")
        end

        @testset "ne=$this_nelement_list, ngr=$this_ngrid_list, nrank=$this_nrank_list, periodic=$periodic_list, dense_bndry=$dense_boundaries_list, bs=$block_sizes_heuristic, spC=$sparse_C_blocks, mumps=$mumps_fill_in_threshold" for
                this_nelement_list ∈ multiset_permutations(nelement_list),
                this_ngrid_list ∈ multiset_permutations(ngrid_list),
                this_nrank_list ∈ get_nrank_permutations(this_nelement_list, distributed_comm_size),
                #periodic_list ∈ (all_periodic ? bool_perms : (fill(false, length(this_nelement_list)),)),
                periodic_list ∈ (fill(false, length(this_nelement_list)),),
                dense_boundaries_list ∈ (all_dense_boundaries ? bool_perms : (fill(false, length(this_nelement_list)),)),
                block_sizes_heuristic ∈ block_sizes_heuristic_list,
                sparse_C_blocks ∈ (false, true),
                mumps_fill_in_threshold ∈ (1.0, 0.1)
            if rank == 0
                println("  - n_sh=$n_shared, sp_sten=$sparse_stencils, ne=$this_nelement_list, ngr=$this_ngrid_list, nrank=$this_nrank_list, periodic=$periodic_list, dense_bndry=$dense_boundaries_list, bs=$block_sizes_heuristic, spC=$sparse_C_blocks, mumps=$mumps_fill_in_threshold")
            end

            this_irank_list = get_iranks(this_nrank_list, distributed_comm_rank)
            dimensions = [create_dimension(; name="d$i", nelement, ngrid, nrank, irank, periodic, dense_boundaries)
                          for (i, (nelement, ngrid, irank, nrank, periodic, dense_boundaries))
                          ∈ enumerate(zip(this_nelement_list, this_ngrid_list,
                                          this_irank_list, this_nrank_list, periodic_list,
                                          dense_boundaries_list))]

            test_multivariable_matrix(dimensions, variable_dimensions, n_shared,
                                      this_seed, sparse_stencils, block_sizes_heuristic,
                                      reduce_proc_count_with_blocks, sparse_C_blocks,
                                      mumps_fill_in_threshold, tol)
            this_seed += 1
        end
    end
end

function test_multivariable_finite_element_matrices()
    if !MPI.Initialized()
        MPI.Init()
    end
    BLAS.set_num_threads(1)
    @testset "finite element matrices" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        comm_size = MPI.Comm_size(MPI.COMM_WORLD)
        # Temporarily disable distributed-memory MPI, until we re-enable support.
        @testset "n_shared=$n_shared" for n_shared ∈ comm_size #[prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
            @testset "1D" begin
                tol = 4.0e-11
                variable_dimensions_1d = (nothing, [1], [1], [1])
                test_multivariable_dimension_combinations([1], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1000)
                test_multivariable_dimension_combinations([2], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1001)
                test_multivariable_dimension_combinations([2], [4], (nothing, nothing, nothing, nothing), rank, comm_size, n_shared, tol, 1002)
                test_multivariable_dimension_combinations([2], [5], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1003; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([3], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1004; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([4], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1005; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([5], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1006)
                test_multivariable_dimension_combinations([6], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1007; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([7], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1008; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([8], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1009; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([16], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1010; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([32], [3], variable_dimensions_1d, rank, comm_size, n_shared, tol, 1011; all_block_sizes_heuristics=false)
            end
            @testset "2D" begin
                tol = 1.0e-6
                variable_dimensions_2d = (nothing, [2], [2], [2])
                test_multivariable_dimension_combinations([1, 1], [3, 3], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2000)
                test_multivariable_dimension_combinations([1, 2], [3, 3], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2001)
                test_multivariable_dimension_combinations([1, 2], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2002; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([1, 3], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2003; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([2, 2], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2004; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([2, 3], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2005; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([2, 4], [3, 5], ([1], nothing, [2], nothing), rank, comm_size, n_shared, tol, 2006; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([1, 8], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2007; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([1, 16], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2008; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([2, 8], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2009; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([4, 4], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2010; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([4, 4], [5, 5], (nothing, nothing, nothing, [1, 2]), rank, comm_size, n_shared, tol, 2011)
                test_multivariable_dimension_combinations([1, 32], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2012; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([2, 16], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2013; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([4, 8], [3, 5], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2014; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([4, 8], [5, 5], ([2], nothing, [1], [2]), rank, comm_size, n_shared, tol, 2015; all_block_sizes_heuristics=false)
                test_multivariable_dimension_combinations([16, 15], [3, 3], variable_dimensions_2d, rank, comm_size, n_shared, tol, 2016)
            end
            @testset "3D" begin
                tol = 2.0e-5
                variable_dimensions_3d = (nothing, [3], [3], [3])
                test_dimension_combinations([1, 1, 1], [3, 3, 3], variable_dimensions_3d, rank, comm_size, n_shared, tol, 3000; all_sparse_stencils=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([2, 2, 2], [3, 4, 5], variable_dimensions_3d, rank, comm_size, n_shared, tol, 3001; all_sparse_stencils=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([2, 2, 3], [3, 3, 4], variable_dimensions_3d, rank, comm_size, n_shared, tol, 3002; all_sparse_stencils=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([4, 4, 4], [3, 3, 3], (nothing, nothing, nothing, nothing), rank, comm_size, n_shared, tol, 3003; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([4, 4, 4], [3, 3, 3], ([1], [2], [3], nothing), rank, comm_size, n_shared, tol, 3003; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([4, 4, 4], [3, 3, 3], ([1, 2], [2, 3], [1, 3], nothing), rank, comm_size, n_shared, tol, 3003; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([4, 4, 4], [3, 3, 3], ([1, 3], nothing, [2], [3]), rank, comm_size, n_shared, tol, 3003; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                test_dimension_combinations([8, 8, 8], [3, 3, 3], variable_dimensions_3d, rank, comm_size, n_shared, tol, 3003; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, all_block_sizes_heuristics=false, both_remove_procs=false)
                if comm_size ≥ 16
                    test_dimension_combinations([9, 9, 32], [3, 3, 3], variable_dimensions_3d, rank, comm_size, n_shared, tol, 3004; all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, both_remove_procs=false)
                end
            end
        end
    end
end
