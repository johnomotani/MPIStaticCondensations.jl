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

    Alu = mpi_static_condensation(dimensions; comm, distributed_comm, shared_comm,
                                  allocate_shared_float, allocate_shared_int, use_sparse,
                                  optimize_schur_complement_size, check_lu=true)

    lu!(Alu, local_matrix)

    @testset "solve" begin
        ldiv!(Alu, rhs_local)
        x_global = gather_vector(rhs_local, dimensions, distributed_comm, shared_comm)
        if distributed_rank == 0 && shared_rank == 0
            check_solution = global_matrix \ rhs_global
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

function generate_bool_permutations(n)
    perms = Vector{Bool}[]
    for inds ∈ CartesianIndices(Tuple(2 for _ ∈ 1:n))
        this_perm = [Bool(i-1) for i ∈ Tuple(inds)]
        push!(perms, this_perm)
    end
    return perms
end

function get_nrank_permutations(nelement_list, nrank)
    nrank_list = Vector{Int64}[]
    ndim = length(nelement_list)
    nrank_factors = factor(Vector, nrank)
    function recursive_push_nrank!(remaining_nrank_factors, this_nrank_list, dim)
        if dim == 1
            remaining_nrank = prod(remaining_nrank_factors; init=1)
            if nelement_list[1] % remaining_nrank == 0
                this_nrank_list[1] = remaining_nrank
                push!(nrank_list, this_nrank_list)
            end
            return nothing
        end
        for this_factors ∈ unique(collect(combinations(remaining_nrank_factors)))
            this_nrank = prod(this_factors; init=1)
            if nelement_list[dim] % this_nrank == 0
                new_nrank_list = copy(this_nrank_list)
                new_nrank_list[dim] = this_nrank
                new_remaining_nrank_factors = copy(remaining_nrank_factors)
                for f ∈ this_factors
                    i = searchsortedfirst(new_remaining_nrank_factors, f)
                    popat!(new_remaining_nrank_factors, i)
                end
                recursive_push_nrank!(new_remaining_nrank_factors, new_nrank_list, dim - 1)
            end
        end
        return nothing
    end
    recursive_push_nrank!(nrank_factors, zeros(ndim), ndim)
    return nrank_list
end

function get_iranks(nrank_list, rank)
    irank_list = similar(nrank_list)
    for (i, nrank) ∈ reverse(collect(enumerate(nrank_list)))
        this_irank, rank = divrem(rank, nrank)
        irank_list[i] = this_irank
    end
    return irank_list
end

function test_dimension_combinations(nelement_list, ngrid_list, max_nproc, rank,
                                     comm_size, n_shared, this_seed)
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
    @testset "nelement_list=$nelement_list, ngrid_list=$ngrid_list, use_sparse=$use_sparse, optimize_schur_complement_size=$optimize_schur_complement_size" for
            use_sparse ∈ (true, false),
            optimize_schur_complement_size ∈ (true, false)

        @testset "$this_nelement_list, $this_ngrid_list, $this_nrank_list, $periodic_list, $remove_boundaries_list" for
                this_nelement_list ∈ multiset_permutations(nelement_list),
                this_ngrid_list ∈ multiset_permutations(ngrid_list),
                this_nrank_list ∈ get_nrank_permutations(this_nelement_list, distributed_comm_size),
                periodic_list ∈ bool_perms,
                remove_boundaries_list ∈ bool_perms

            this_irank_list = get_iranks(this_nrank_list, distributed_comm_rank)
            dimensions = [create_dimension(; nelement, ngrid, nrank, irank, periodic, remove_boundaries)
                          for (nelement, ngrid, irank, nrank, periodic, remove_boundaries)
                          ∈ zip(this_nelement_list, this_ngrid_list, this_irank_list, this_nrank_list, periodic_list, remove_boundaries_list)]

            test_matrix(dimensions, n_shared, this_seed, use_sparse, optimize_schur_complement_size)
            this_seed += 1
        end
    end
end

function test_finite_element_matrices()
    if !MPI.Initialized()
        MPI.Init()
    end
    @testset "finite element matrices" begin
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        comm_size = MPI.Comm_size(MPI.COMM_WORLD)
        n_shared = 1
        test_dimension_combinations([4], [3], 4, rank, comm_size, n_shared, 987)
    end
end
