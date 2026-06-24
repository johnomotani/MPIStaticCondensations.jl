using MPISchurComplements
using MPIStaticCondensations
using MPIStaticCondensations: Dimension, OuterBSubmatrix, OuterCSubmatrix,
                              get_upper_right_block_shared_sparse_matrix_buffer,
                              get_lower_left_block_shared_sparse_matrix_buffer
using BlockArrays
using Combinatorics
using LinearAlgebra
using MPI
using Primes
using Random
using StableRNGs
using Test

include("generate_finite_element_matrices.jl")
include("utils.jl")

using Debugger
function get_flat_global_indices(dimensions_for_variables)
    if isa(dimensions_for_variables, Vector{<:Dimension})
        dimensions_for_variables = [dimensions_for_variables]
    end

    function getglob(dims, current_inds)
        if isempty(dims)
            return current_inds
        end
        new_inds = Int64[]
        lastdim = dims[end]
        n = lastdim.n
        ginds = lastdim.global_inds
        for i ∈ current_inds
            i = (i - 1) * n
            for g ∈ ginds
                push!(new_inds, i + g)
            end
        end
        return getglob(dims[1:end-1], new_inds)
    end

    offset = 0
    globinds = Int64[]
    for dimensions ∈ dimensions_for_variables
        globinds = vcat(globinds, offset .+ getglob(dimensions, Int64[1]))
        offset += prod(d.n for d ∈ dimensions)
    end

    return globinds
end

function test_outer_schur_complement_matrix(dimensions::Vector{<:Dimension},
                                            n_shared::Integer, random_seed::Integer,
                                            use_sparse::Bool, sparse_stencils::Bool,
                                            reduce_proc_count_with_blocks::Bool,
                                            tol::AbstractFloat)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared)

    rng = StableRNG(random_seed)

    # Dimensions to use for the 'bottom vector' parts of the outer Schur-complement
    # system.
    bottom_dimension = dimensions[end]
    bottom_block_size = bottom_dimension.n

    function get_matrix(local_B=nothing, local_C=nothing)
        global_A, local_A =
            assemble_and_scatter_global_matrix(dimensions, comm, distributed_comm,
                                               shared_comm, allocate_shared_float,
                                               allocate_shared_int, rng, sparse_stencils)
        new_local_B = FixedSparseCSC{Float64,Int64}[]
        new_local_C = Transpose{Float64, FixedSparseCSC{Float64,Int64}}[]
        if shared_rank == 0
            global_B = FixedSparseCSC{Float64,Int64}[]
            global_C = Transpose{Float64, FixedSparseCSC{Float64,Int64}}[]
        end
        for s ∈ ("empty", "point", "element")
            glob_B, loc_B = assemble_and_scatter_global_matrix([bottom_dimension], comm,
                                                               distributed_comm,
                                                               shared_comm,
                                                               allocate_shared_float,
                                                               allocate_shared_int, rng,
                                                               false;
                                                               extra_row_dimensions=dimensions[1:end-1],
                                                               stencil=s)
            push!(new_local_B, loc_B)
            if shared_rank == 0
                push!(global_B, glob_B)
            end
        end
        for s ∈ ("element", "empty", "point")
            glob_C, loc_C = assemble_and_scatter_global_matrix([bottom_dimension], comm,
                                                               distributed_comm,
                                                               shared_comm,
                                                               allocate_shared_float,
                                                               allocate_shared_int, rng,
                                                               false;
                                                               extra_column_dimensions=dimensions[1:end-1],
                                                               stencil=s,
                                                               transpose_result=true)
            push!(new_local_C, loc_C)
            if shared_rank == 0
                push!(global_C, glob_C)
            end
        end
        new_local_B = mortar(reshape(new_local_B, 1, 3))
        new_local_C = mortar(reshape(new_local_C, 3, 1))
        if shared_rank == 0
            global_B = mortar(reshape(global_B, 1, 3))
            global_C = mortar(reshape(global_C, 3, 1))
        end
        if local_B === nothing && local_C === nothing
            local_B = new_local_B
            local_C = new_local_C
        else
            if shared_rank == 0
                for (b, newb) ∈ zip(blocks(local_B), blocks(new_local_B))
                    b.nzval .= newb.nzval
                end
                for (c, newc) ∈ zip(blocks(local_C), blocks(new_local_C))
                    transpose(c).nzval .= transpose(newc).nzval
                end
            end
        end
        local_D = FixedSparseCSC{Float64,Int64}[]
        if shared_rank == 0
            global_D = FixedSparseCSC{Float64,Int64}[]
        end
        for _ ∈ 1:9
            glob_D, loc_D = assemble_and_scatter_global_matrix([bottom_dimension], comm,
                                                               distributed_comm,
                                                               shared_comm,
                                                               allocate_shared_float,
                                                               allocate_shared_int, rng,
                                                               false)
            push!(local_D, loc_D)
            if shared_rank == 0
                push!(global_D, glob_D)
            end
        end
        local_D = mortar(reshape(local_D, 3, 3))
        if shared_rank == 0
            global_D = mortar(reshape(global_D, 3, 3))
        end

        if shared_rank == 0
            global_matrix = mortar(reshape([global_A, global_C[Block(1,1)],
                                            global_C[Block(2,1)], global_C[Block(3,1)],
                                            global_B[Block(1,1)], global_D[Block(1,1)],
                                            global_D[Block(2,1)], global_D[Block(3,1)],
                                            global_B[Block(1,2)], global_D[Block(1,2)],
                                            global_D[Block(2,2)], global_D[Block(3,2)],
                                            global_B[Block(1,3)], global_D[Block(1,3)],
                                            global_D[Block(2,3)], global_D[Block(3,3)]],
                                           4, 4))
        else
            global_matrix = nothing
        end

        if !use_sparse
            # Convert to dense matrices.
            dense_local_A = allocate_shared_float(size(local_A)...)
            dense_local_B = allocate_shared_float(size(local_B)...)
            dense_local_C = allocate_shared_float(size(local_C)...)
            dense_local_D = allocate_shared_float(size(local_D)...)
            if shared_rank == 0
                dense_local_A .= local_A
                dense_local_B .= local_B
                dense_local_C .= local_C
                dense_local_D .= local_D
            end
            MPI.Barrier(shared_comm)
            local_A = dense_local_A
            local_B = dense_local_B
            local_C = dense_local_C
            local_D = dense_local_D
        end

        return global_matrix, local_A, local_B, local_C, local_D
    end

    function get_rhs()
        U_global, U_local =
            assemble_and_scatter_global_rhs(dimensions, comm, distributed_comm, shared_comm,
                                            allocate_shared_float, rng)

        V_local = allocate_shared_float(3 * bottom_block_size)
        if shared_rank == 0
            rand!(rng, V_local)

            # This needs updating to support distributed-memory MPI...
            rhs_global = vcat(U_global, V_local)
        else
            rhs_global = nothing
        end

        return rhs_global, U_local, V_local
    end

    global_matrix, local_A, local_B, local_C, local_D = get_matrix()
    rhs_global, U_local, V_local = get_rhs()

    top_vector_indices = get_flat_global_indices(dimensions)
    bottom_vector_indices = get_flat_global_indices([[bottom_dimension], [bottom_dimension], [bottom_dimension]])

    Alu = mpi_static_condensation(dimensions; reduce_proc_count_with_blocks, comm,
                                  distributed_comm, shared_comm, allocate_shared_float,
                                  allocate_shared_int, use_sparse, check_lu=true)

    outer_B = OuterBSubmatrix(Alu, dimensions, [bottom_dimension], shared_comm,
                              allocate_shared_float, allocate_shared_int, local_B)
    outer_C = OuterCSubmatrix(Alu, [bottom_dimension], dimensions, shared_comm,
                              allocate_shared_float, allocate_shared_int, local_C)
    # When using distributed MPI, do we need to store global_top_vector_indices and pass
    # them in here?
    full_solver = mpi_schur_complement(Alu, nothing, nothing, nothing,
                                       top_vector_indices, bottom_vector_indices; comm,
                                       shared_comm, distributed_comm,
                                       allocate_shared_float, allocate_shared_int,
                                       use_sparse, sparse_Ainv_B=use_sparse,
                                       Ainv_dot_B_buffer=outer_B, C_buffer=outer_C,
                                       skip_factorization=true, check_lu=true)

    update_schur_complement!(full_solver, local_A, local_B, local_C, local_D)

    function test_once()
        ldiv!(full_solver, U_local, V_local)
        MPI.Barrier(shared_comm)
        x_global = gather_vector(U_local, dimensions, comm, distributed_comm, shared_comm)
        # This needs updating to support distributed-memory MPI...
        y_global = V_local
        if distributed_rank == 0 && shared_rank == 0
            solution = vcat(x_global, y_global)
            check_solution = global_matrix \ rhs_global
            @test isapprox(solution, check_solution;
                           norm=(x)->NaN, rtol=tol, atol=tol)
            @test isapprox(global_matrix * solution, rhs_global;
                           norm=(x)->NaN, rtol=tol, atol=tol)
        end
    end

    @testset "solve" begin
        test_once()
    end

    @testset "change b" begin
        rhs_global, U_local, V_local = get_rhs()
        MPI.Barrier(shared_comm)

        test_once()
    end

    @testset "change M" begin
        global_matrix, local_A, local_B, local_C, local_D = get_matrix(local_B, local_C)
        rhs_global, U_local, V_local = get_rhs()
        MPI.Barrier(shared_comm)
        if !use_sparse
            # Convert to dense matrix.
            if shared_rank == 0
                dense_local_matrix .= local_matrix
            end
            MPI.Barrier(shared_comm)
            local_matrix = dense_local_matrix
        end

        update_schur_complement!(full_solver, local_A, local_B, local_C, local_D)

        test_once()
    end

    @testset "change M, change b" begin
        rhs_global, U_local, V_local = get_rhs()
        MPI.Barrier(shared_comm)

        test_once()
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

function test_outer_schur_complement_dimension_combinations(
             nelement_list, ngrid_list, rank, comm_size, n_shared, tol, this_seed;
             all_use_sparse=true, all_sparse_stencils=true, all_periodic=true,
             all_dense_boundaries=true, both_remove_procs=true)
    if length(nelement_list) != length(ngrid_list)
        error("nelement_list and ngrid_list must have the same length")
    end

    distributed_comm_size = comm_size ÷ n_shared
    distributed_comm_rank = rank ÷ n_shared

    bool_perms = generate_bool_permutations(length(nelement_list))
    # Only support use_sparse=true, for now.
    #if all_use_sparse
    #    use_sparse_list = (true, false)
    #else
    #    use_sparse_list = (true,)
    #end
    use_sparse_list = (true,)
    if all_sparse_stencils
        sparse_stencils_list = (true, false)
    else
        sparse_stencils_list = (true,)
    end
    if both_remove_procs
        reduce_proc_count_with_blocks_list = (false, true)
    else
        reduce_proc_count_with_blocks_list = (false,)
    end
    @testset "nelement_list=$nelement_list, ngrid_list=$ngrid_list, use_sparse=$use_sparse, sparse_stencils=$sparse_stencils, reduce_proc_count_with_blocks=$reduce_proc_count_with_blocks" for
            use_sparse ∈ use_sparse_list,
            sparse_stencils ∈ sparse_stencils_list,
            reduce_proc_count_with_blocks ∈ reduce_proc_count_with_blocks_list
        if rank == 0
            println("* n_shared=$n_shared, nelement_list=$nelement_list, ngrid_list=$ngrid_list, use_sparse=$use_sparse, sparse_stencils=$sparse_stencils, reduce_proc_count_with_blocks=$reduce_proc_count_with_blocks")
        end

        @testset "this_nelement_list=$this_nelement_list, this_ngrid_list=$this_ngrid_list, this_nrank_list=$this_nrank_list, periodic_list=$periodic_list, dense_boundaries_list=$dense_boundaries_list" for
                this_nelement_list ∈ multiset_permutations(nelement_list),
                this_ngrid_list ∈ multiset_permutations(ngrid_list),
                this_nrank_list ∈ get_nrank_permutations(this_nelement_list, distributed_comm_size),
                #periodic_list ∈ (all_periodic ? bool_perms : (fill(false, length(this_nelement_list)),)),
                periodic_list ∈ (fill(false, length(this_nelement_list)),), # For now, periodic bc not supported.
                dense_boundaries_list ∈ (all_dense_boundaries ? bool_perms : (fill(false, length(this_nelement_list)),))
                #dense_boundaries_list ∈ ([true, false],)
            if rank == 0
                println("  - n_shared=$n_shared, ($use_sparse, $sparse_stencils), this_nelement_list=$this_nelement_list, this_ngrid_list=$this_ngrid_list, this_nrank_list=$this_nrank_list, periodic_list=$periodic_list, dense_boundaries_list=$dense_boundaries_list")
            end

            this_irank_list = get_iranks(this_nrank_list, distributed_comm_rank)
            dimensions = [create_dimension(; name=Symbol("d$i"), nelement, ngrid, nrank,
                                           irank, periodic, dense_boundaries)
                          for (i, (nelement, ngrid, irank, nrank, periodic, dense_boundaries))
                          ∈ enumerate(zip(this_nelement_list, this_ngrid_list, this_irank_list, this_nrank_list, periodic_list, dense_boundaries_list))]

            test_outer_schur_complement_matrix(dimensions, n_shared, this_seed, use_sparse, sparse_stencils, reduce_proc_count_with_blocks, tol)
            this_seed += 1
        end
    end
end

function test_outer_schur_complement_matrices()
    if !MPI.Initialized()
        MPI.Init()
    end
    BLAS.set_num_threads(1)
    @testset "finite element matrices" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        comm_size = MPI.Comm_size(MPI.COMM_WORLD)
        # Temporarily disable distributed-memory MPI, until we re-enable support.
        @testset "n_shared=$n_shared" for n_shared ∈ comm_size #[prod(x) for x ∈ unique(combinations(factor(Vector, comm_size)))]
            @testset "2D" begin
                tol = 2.0e-7
                test_outer_schur_complement_dimension_combinations([1, 1], [3, 3], rank, comm_size, n_shared, tol, 4000)
                test_outer_schur_complement_dimension_combinations([1, 2], [3, 3], rank, comm_size, n_shared, tol, 4001)
                test_outer_schur_complement_dimension_combinations([1, 3], [3, 3], rank, comm_size, n_shared, tol, 4003)
                test_outer_schur_complement_dimension_combinations([2, 2], [3, 3], rank, comm_size, n_shared, tol, 4004)
                test_outer_schur_complement_dimension_combinations([1, 16], [3, 3], rank, comm_size, n_shared, tol, 4008)
                test_outer_schur_complement_dimension_combinations([2, 8], [3, 3], rank, comm_size, n_shared, tol, 4009)
                test_outer_schur_complement_dimension_combinations([4, 4], [3, 3], rank, comm_size, n_shared, tol, 4011)
                test_outer_schur_complement_dimension_combinations([4, 8], [3, 3], rank, comm_size, n_shared, tol, 4014)
            end
            @testset "3D" begin
                tol = 5.0e-7
                test_outer_schur_complement_dimension_combinations([1, 1, 1], [3, 3, 3], rank, comm_size, n_shared, tol, 5000; all_use_sparse=false, all_sparse_stencils=false, both_remove_procs=false)
                test_outer_schur_complement_dimension_combinations([2, 2, 2], [3, 3, 3], rank, comm_size, n_shared, tol, 5001; all_use_sparse=false, all_sparse_stencils=false, both_remove_procs=false)
                test_outer_schur_complement_dimension_combinations([8, 8, 8], [3, 3, 3], rank, comm_size, n_shared, tol, 5003; all_use_sparse=false, all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, both_remove_procs=false)
                if comm_size ≥ 16
                    test_outer_schur_complement_dimension_combinations([9, 9, 32], [3, 3, 3], rank, comm_size, n_shared, tol, 3004; all_use_sparse=false, all_sparse_stencils=false, all_periodic=false, all_dense_boundaries=false, both_remove_procs=false)
                end
            end
        end
    end
end
