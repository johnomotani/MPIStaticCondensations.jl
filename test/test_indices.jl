using MPIStaticCondensations
using MPIStaticCondensations: FakeComm, split_matrix, pick_dimension_to_split
using Test

# Notes
# =====
#
# We do not test global_size or global_bottom_vector_size because they cannot be
# calculated correctly without actual MPI communication.

const ngrid = 3

using Debugger
#function get_level_info(ngrid_list, nelement_list, periodic_list, remove_boundaries_list,
#                        nrank_list, irank_list, n_shared, n_groups, irank;
#                        optimize_schur_complement_size=true)
function get_level_info(ngrid_list, nelement_list, block_sizes_list, periodic_list,
                        remove_boundaries_list, nrank_list, irank_list, n_shared, irank)
    total_nrank = prod(nrank_list) * n_shared

    if !isa(ngrid_list, AbstractVector)
        ngrid_list = fill(ngrid_list, length(nelement_list))
    end

    comm = FakeComm(irank, total_nrank)
    shared_comm = FakeComm(irank % n_shared, n_shared)
    if shared_comm.rank == 0
        distributed_comm = FakeComm(irank ÷ n_shared, total_nrank ÷ n_shared)
    else
        distributed_comm = nothing
    end
    distributed_size = comm.size ÷ shared_comm.size
    distributed_rank = comm.rank ÷ shared_comm.size

    dimensions = [create_dimension(; nelement, ngrid, nrank, irank=dim_irank, periodic,
                                   remove_boundaries)
                  for (nelement, ngrid, periodic, remove_boundaries, nrank, dim_irank) ∈
                      zip(nelement_list, ngrid_list, periodic_list,
                          remove_boundaries_list, nrank_list, irank_list)]

    dimensions_without_periodic = [Dimension(; nelement=d.nelement, ngrid=d.ngrid,
                                             nrank=d.nrank, irank=d.irank, periodic=false,
                                             remove_boundaries=(d.periodic || d.remove_boundaries))
                                   for d ∈ dimensions]
    #level_info = first(split_dimension(dimensions, n_groups, optimize_schur_complement_size,
    #                                   comm, distributed_comm, shared_comm))
    this_global_size = prod(d.n for d ∈ dimensions)
    local_size = prod(d.n_local for d ∈ dimensions)
    local_indices = collect(1:local_size)
    n_levels = length(block_sizes_list)
    level_info = Any[]
    for (level, bs) ∈ enumerate(block_sizes_list)
        if level == n_levels
            # Only handle periodicity on the final level
            dims = dimensions
        else
            dims = dimensions_without_periodic
        end
        li = split_matrix(dims, local_indices, bs, this_global_size, distributed_comm,
                          shared_comm)
        push!(level_info, li)
        this_global_size = li.global_bottom_vector_size
        local_indices = li.local_bottom_vector_indices
    end

    return level_info
end

function test_split_indices_1d_2proc()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # -----+++-----===-----+++-----
    # 1:2 | 3 | 4 ∥ 5 ∥ 6 | 7 | 8:9
    # -----+++-----===-----+++-----
    nrank = 2
    n_shared = 1
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_a_block_indices == vcat(1:2, 4)
            @test li[1].a_block_sub_selection_indices == 1:3
            @test li[1].bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_indices == [3, 5]
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [1]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(6, 8:9)
            @test li[1].local_top_vector_indices == vcat(2, 4:5)
            @test li[1].local_top_vector_a_block_indices == vcat(2, 4:5)
            @test li[1].a_block_sub_selection_indices == 1:3
            @test li[1].bottom_vector_indices == [5, 7]
            @test li[1].local_bottom_vector_indices == [1, 3]
            @test li[2].top_vector_indices == [7]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [2]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [1]
        end
    end

    nrank = 2
    n_shared = 2
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_a_block_indices == vcat(1:2, 4)
            @test li[1].a_block_sub_selection_indices == 1:3
            @test li[1].bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_indices == [3, 5, 7]
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [1, 3]
            @test li[2].local_top_vector_a_block_indices == [1]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_a_block_indices == vcat(6, 8:9)
            @test li[1].a_block_sub_selection_indices == 4:6
            @test li[1].bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_indices == [3, 5, 7]
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [1, 3]
            @test li[2].local_top_vector_a_block_indices == [3]
            @test li[2].a_block_sub_selection_indices == [2]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
        end
    end

    return nothing
end

function test_split_indices_1d_2proc_periodic()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # -----+++-----===-----+++-----
    # 1:2 | 3 | 4 ∥ 5 ∥ 6 | 7 | 8:1
    # -----+++-----===-----+++-----
    nrank = 2
    n_shared = 1
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4]
            @test li[1].local_top_vector_indices == [2, 4]
            @test li[1].local_top_vector_a_block_indices == [2, 4]
            @test li[1].a_block_sub_selection_indices == 1:2
            @test li[1].bottom_vector_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_indices == [1, 3, 5]
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [2]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [1, 5]
            @test li[2].local_bottom_vector_indices == [1, 3]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [6, 8]
            @test li[1].local_top_vector_indices == [2, 4]
            @test li[1].local_top_vector_a_block_indices == [2, 4]
            @test li[1].a_block_sub_selection_indices == 1:2
            @test li[1].bottom_vector_indices == [5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5]
            @test li[2].top_vector_indices == [7]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [2]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3]
        end
    end

    nrank = 2
    n_shared = 2
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_a_block_indices == [2, 4]
            @test li[1].a_block_sub_selection_indices == 1:2
            @test li[1].bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [2, 4]
            @test li[2].local_top_vector_a_block_indices == [2]
            @test li[2].a_block_sub_selection_indices == [1]
            @test li[2].bottom_vector_indices == [1, 5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3, 5]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_a_block_indices == [6, 8]
            @test li[1].a_block_sub_selection_indices == 3:4
            @test li[1].bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [2, 4]
            @test li[2].local_top_vector_a_block_indices == [4]
            @test li[2].a_block_sub_selection_indices == [2]
            @test li[2].bottom_vector_indices == [1, 5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3, 5]
        end
    end

    return nothing
end

function test_split_indices_1d_4group()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [false]

    # With 4 groups, the global index division is:
    # -------===-----===-----===-------
    # | 1:2 ∥ 3 ∥ 4 ∥ 5 ∥ 6 ∥ 7 ∥ 8:9 |
    # -------===-----===-----===-------
    n_groups = 4
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 1:2
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 8:9
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 4
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_a_block_indices == 4:4
            @test li.a_block_sub_selection_indices == 3:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_a_block_indices == 8:9
            @test li.a_block_sub_selection_indices == 5:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_3group()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [false]

    # With 3 groups, the global index division is:
    # -------===-------===-
    # | 1:4 ∥ 5 ∥ 6:8 ∥ 9 |
    # -------===-------===-
    n_groups = 3
    nrank = 3
    n_shared = 3
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.global_top_vector_size == 8
            @test li.local_top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_a_block_indices == 1:4
            @test li.a_block_sub_selection_indices == 1:4
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.global_top_vector_size == 8
            @test li.local_top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_a_block_indices == 6:9
            @test li.a_block_sub_selection_indices == 5:8
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.global_top_vector_size == 8
            @test li.local_top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_periodic()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [false]

    n_groups = 2
    # With 2 groups, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6:9 |
    # -------===-------
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 2:3
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_4group_periodic()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [false]

    # With 4 groups, the global index division is:
    # -------===-----===-----===-------
    # | 1:2 ∥ 3 ∥ 4 ∥ 5 ∥ 6 ∥ 7 ∥ 8:9 |
    # -------===-----===-----===-------
    n_groups = 4
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 2:2
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 4
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 4:4
            @test li.a_block_sub_selection_indices == 2:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 3:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 8:8
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_3group_periodic()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [false]

    # With 3 groups, the global index division is:
    # -------===-------===-
    # | 1:4 ∥ 5 ∥ 6:8 ∥ 9 |
    # -------===-------===-
    n_groups = 3
    nrank = 3
    n_shared = 3
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_remove_boundaries()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [true]

    # With 2 groups, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6:9 |
    # -------===-------
    n_groups = 2
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 2:3
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [9]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_4group_remove_boundaries()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [true]

    # With 4 groups, the global index division is:
    # -------===-----===-----===-------
    # | 1:2 ∥ 3 ∥ 4 ∥ 5 ∥ 6 ∥ 7 ∥ 8:9 |
    # -------===-----===-----===-------
    n_groups = 4
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 2:2
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,9]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 4
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 4:4
            @test li.a_block_sub_selection_indices == 2:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 3:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 8:8
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_3group_remove_boundaries()
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [true]

    # With 3 groups, the global index division is:
    # -------===-------===-
    # | 1:4 ∥ 5 ∥ 6:8 ∥ 9 |
    # -------===-------===-
    n_groups = 3
    nrank = 3
    n_shared = 3
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_periodic_remove_boundaries()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [true]

    # With 2 groups, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6:9 |
    # -------===-------
    n_groups = 2
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 2:3
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:3
            @test li.local_top_vector_a_block_indices == 2:3
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 1:2
            @test li.local_top_vector_a_block_indices == 1:2
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 2:4
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 2
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_4group_periodic_remove_boundaries()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [true]

    # With 4 groups, the global index division is:
    # -------===-----===-----===-------
    # | 1:2 ∥ 3 ∥ 4 ∥ 5 ∥ 6 ∥ 7 ∥ 8:9 |
    # -------===-----===-----===-------
    n_groups = 4
    nrank = 4
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 2:2
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == 2:2
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    n_groups = 4
    nrank = 4
    n_shared = 4
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 2:2
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 4:4
            @test li.a_block_sub_selection_indices == 2:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 3:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_a_block_indices == 8:8
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_3group_periodic_remove_boundaries()
    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [true]

    # With 3 groups, the global index division is:
    # -------===-------===-
    # | 1:4 ∥ 5 ∥ 6:8 ∥ 9 |
    # -------===-------===-
    n_groups = 3
    nrank = 3
    n_shared = 3
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 6:8
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_oddnelement()
    # With 2 groups, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6:7 |
    # -------===-------
    nelement_list = [3]
    periodic_list = [false]
    remove_boundaries_list = [false]

    n_groups = 2
    nrank = 2
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == vcat(1:4, 6:7)
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == vcat(1:4, 6:7)
            @test li.local_top_vector_a_block_indices == 1:4
            @test li.a_block_sub_selection_indices == 1:4
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == vcat(1:4, 6:7)
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == vcat(1:4, 6:7)
            @test li.local_top_vector_a_block_indices == 6:7
            @test li.a_block_sub_selection_indices == 5:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    nelement_list = [3]
    periodic_list = [false]
    remove_boundaries_list = [true]

    n_groups = 2
    nrank = 2
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 7]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 7]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_oddnelement_periodic()
    # With 2 groups for a periodic dimension, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6,1 |
    # -------===-------
    nelement_list = [3]
    periodic_list = [true]
    remove_boundaries_list = [false]

    n_groups = 2
    nrank = 2
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 1]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 1]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_1d_2group_oddnelement_periodic_remove_boundaries()
    # With 2 groups for a periodic dimension, the global index division is:
    # -------===-------
    # | 1:4 ∥ 5 ∥ 6,1 |
    # -------===-------
    nelement_list = [3]
    periodic_list = [true]
    remove_boundaries_list = [true]

    n_groups = 2
    nrank = 2
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 1]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 2:4
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1, 5, 1]
            @test li.local_bottom_vector_indices == [1, 5, 7]
            @test li.top_vector_indices == vcat(2:4, 6)
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == vcat(2:4, 6)
            @test li.local_top_vector_a_block_indices == 6:6
            @test li.a_block_sub_selection_indices == 4:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_112()
    # With the 3rd dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # ---------------=======-----------------
    # | 1:3 | 10:12 ∥ 19:21 ∥ 28:30 | 37:39 |
    # ---------------=======-----------------
    # | 4:6 | 13:15 ∥ 22:24 ∥ 31:33 | 40:42 |
    # ---------------=======-----------------
    # | 7:9 | 16:18 ∥ 25:27 ∥ 34:36 | 43:45 |
    # ---------------=======-----------------
    nelement_list = [1, 1, 2]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 1, 2]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 19:27
            @test li.top_vector_indices == 1:18
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == 1:18
            @test li.local_top_vector_a_block_indices == 1:18
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 2
            @test li.level_dimensions[3].n_local == 2
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 1:9
            @test li.top_vector_indices == 28:45
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == 10:27
            @test li.local_top_vector_a_block_indices == 10:27
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 2
            @test li.level_dimensions[3].n_local == 2
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 19:27
            @test li.top_vector_indices == [(1:18)...,(28:45)...]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == [(1:18)...,(28:45)...]
            @test li.local_top_vector_a_block_indices == 1:18
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 2
            @test li.level_dimensions[3].n_local == 2
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 19:27
            @test li.top_vector_indices == [(1:18)...,(28:45)...]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == [(1:18)...,(28:45)...]
            @test li.local_top_vector_a_block_indices == 28:45
            @test li.a_block_sub_selection_indices == 19:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 2
            @test li.level_dimensions[3].n_local == 2
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_fff()
    # With the 2nd dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # -------------------------
    # | 1:3   | 16:18 | 31:33 |
    # -------------------------
    # | 4:6   | 19:21 | 34:36 |
    # =========================
    # ∥ 7:9   ∥ 22:24 ∥ 37:39 ∥
    # =========================
    # | 10:12 | 25:27 | 40:42 |
    # -------------------------
    # | 13:15 | 28:30 | 43:45 |
    # -------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(7:9, 16:18, 25:27)
            @test li.top_vector_indices == vcat(1:6, 16:21, 31:36)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 10:15, 19:24)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(1:3, 10:12, 19:21)
            @test li.top_vector_indices == vcat(10:15, 25:30, 40:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
            @test li.local_top_vector_a_block_indices == vcat(4:9, 13:18, 22:27)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 16:21, 31:36)
            @test li.a_block_sub_selection_indices == vcat(1:6, 13:18, 25:30)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(10:15, 25:30, 40:45)
            @test li.a_block_sub_selection_indices == vcat(7:12, 19:24, 31:36)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_fft()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, true]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:9, 22:24, 31:39)
            @test li.local_bottom_vector_indices == vcat(1:9, 16:27)
            @test li.top_vector_indices == 16:21
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == 10:15
            @test li.local_top_vector_a_block_indices == 10:15
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:15, 22:24, 37:45)
            @test li.local_bottom_vector_indices == vcat(1:9, 10:12, 19:27)
            @test li.top_vector_indices == 25:30
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == 13:18
            @test li.local_top_vector_a_block_indices == 13:18
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.local_bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.top_vector_indices == vcat(16:21, 25:30)
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == vcat(16:21, 25:30)
            @test li.local_top_vector_a_block_indices == 16:21
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.local_bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.top_vector_indices == vcat(16:21, 25:30)
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == vcat(16:21, 25:30)
            @test li.local_top_vector_a_block_indices == 25:30
            @test li.a_block_sub_selection_indices == 7:12
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_ftf()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, true, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 16:18, 22:24, 31:33, 37:39)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(4:6, 19:21, 34:36)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 13:15, 22:24, 28:30, 37:39, 43:45)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(10:12, 25:27, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_ftt()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, true, true]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:9, 16:18, 22:24, 31:39)
            @test li.local_bottom_vector_indices == vcat(1:12, 16:27)
            @test li.top_vector_indices == 19:21
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 13:15
            @test li.local_top_vector_a_block_indices == 13:15
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:15, 22:24, 28:30, 37:45)
            @test li.local_bottom_vector_indices == vcat(1:12, 16:27)
            @test li.top_vector_indices == 25:27
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == 13:15
            @test li.local_top_vector_a_block_indices == 13:15
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:18, 22:24, 28:45)
            @test li.local_bottom_vector_indices == vcat(1:18, 22:24, 28:45)
            @test li.top_vector_indices == vcat(19:21, 25:27)
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == vcat(19:21, 25:27)
            @test li.local_top_vector_a_block_indices == 19:21
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:18, 22:24, 28:45)
            @test li.local_bottom_vector_indices == vcat(1:18, 22:24, 28:45)
            @test li.top_vector_indices == vcat(19:21, 25:27)
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == vcat(19:21, 25:27)
            @test li.local_top_vector_a_block_indices == 25:27
            @test li.a_block_sub_selection_indices == 4:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_tff()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [true, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1, 3:4, 6:9, 16, 18:19, 21:24, 31, 33:34, 36:39)
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:19, 21:22, 24:27)
            @test li.top_vector_indices == [2, 5, 17, 20, 32, 35]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 20, 23]
            @test li.local_top_vector_a_block_indices == [2, 5, 11, 14, 20, 23]
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:10, 12:13, 15, 22:25, 27:28, 30, 37:40, 42:43, 45)
            @test li.local_bottom_vector_indices == vcat(1:4, 6:7, 9:13, 15:16, 18:22, 24:25, 27)
            @test li.top_vector_indices == [11, 14, 26, 29, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [5, 8, 14, 17, 23, 26]
            @test li.local_top_vector_a_block_indices == [5, 8, 14, 17, 23, 26]
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.local_top_vector_a_block_indices == [2, 5, 17, 20, 32, 35]
            @test li.a_block_sub_selection_indices == vcat(1:2, 5:6, 9:10)
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.local_top_vector_a_block_indices == [11, 14, 26, 29, 41, 44]
            @test li.a_block_sub_selection_indices == vcat(3:4, 7:8, 11:12)
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_tft()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [true, false, true]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:9, 16, 18:19, 21:24, 31:39)
            @test li.local_bottom_vector_indices == vcat(1:10, 12:13, 15:27)
            @test li.top_vector_indices == [17, 20]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [11, 14]
            @test li.local_top_vector_a_block_indices == [11, 14]
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:15, 22:25, 27:28, 30, 37:45)
            @test li.local_bottom_vector_indices == vcat(1:13, 15:16, 18:27)
            @test li.top_vector_indices == [26, 29]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [14, 17]
            @test li.local_top_vector_a_block_indices == [14, 17]
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:16, 18:19, 21:25, 27:28, 30:45)
            @test li.local_bottom_vector_indices == vcat(1:16, 18:19, 21:25, 27:28, 30:45)
            @test li.top_vector_indices == [17, 20, 26, 29]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [17, 20, 26, 29]
            @test li.local_top_vector_a_block_indices == [17, 20]
            @test li.a_block_sub_selection_indices == 1:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:16, 18:19, 21:25, 27:28, 30:45)
            @test li.local_bottom_vector_indices == vcat(1:16, 18:19, 21:25, 27:28, 30:45)
            @test li.top_vector_indices == [17, 20, 26, 29]
            @test li.global_top_vector_size == 4
            @test li.local_top_vector_indices == [17, 20, 26, 29]
            @test li.local_top_vector_a_block_indices == [26, 29]
            @test li.a_block_sub_selection_indices == 3:4
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_ttf()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [true, true, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:4, 6:9, 16:19, 21:24, 31:34, 36:39)
            @test li.local_bottom_vector_indices == vcat(1:4, 6:13, 15:22, 24:27)
            @test li.top_vector_indices == [5, 20, 35]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [5, 14, 23]
            @test li.local_top_vector_a_block_indices == [5, 14, 23]
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:10, 12:15, 22:25, 27:30, 37:40, 42:45)
            @test li.local_bottom_vector_indices == vcat(1:4, 6:13, 15:22, 24:27)
            @test li.top_vector_indices == [11, 26, 41]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [5, 14, 23]
            @test li.local_top_vector_a_block_indices == [5, 14, 23]
            @test li.a_block_sub_selection_indices == 1:3
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:4, 6:10, 12:19, 21:25, 27:34, 36:40, 42:45)
            @test li.local_bottom_vector_indices == vcat(1:4, 6:10, 12:19, 21:25, 27:34, 36:40, 42:45)
            @test li.top_vector_indices == [5, 11, 20, 26, 35, 41]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [5, 11, 20, 26, 35, 41]
            @test li.local_top_vector_a_block_indices == [5, 20, 35]
            @test li.a_block_sub_selection_indices == [1, 3, 5]
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:4, 6:10, 12:19, 21:25, 27:34, 36:40, 42:45)
            @test li.local_bottom_vector_indices == vcat(1:4, 6:10, 12:19, 21:25, 27:34, 36:40, 42:45)
            @test li.top_vector_indices == [5, 11, 20, 26, 35, 41]
            @test li.global_top_vector_size == 6
            @test li.local_top_vector_indices == [5, 11, 20, 26, 35, 41]
            @test li.local_top_vector_a_block_indices == [11, 26, 41]
            @test li.a_block_sub_selection_indices == [2, 4, 6]
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fff_ttt()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [true, true, true]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:9, 16:19, 21:24, 31:39)
            @test li.local_bottom_vector_indices == vcat(1:13, 15:27)
            @test li.top_vector_indices == [20]
            @test li.global_top_vector_size == 2
            @test li.local_top_vector_indices == [14]
            @test li.local_top_vector_a_block_indices == [14]
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:15, 22:25, 27:30, 37:45)
            @test li.local_bottom_vector_indices == vcat(1:13, 15:27)
            @test li.top_vector_indices == [26]
            @test li.global_top_vector_size == 2
            @test li.local_top_vector_indices == [14]
            @test li.local_top_vector_a_block_indices == [14]
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:19, 21:25, 27:45)
            @test li.local_bottom_vector_indices == vcat(1:19, 21:25, 27:45)
            @test li.top_vector_indices == [20, 26]
            @test li.global_top_vector_size == 2
            @test li.local_top_vector_indices == [20, 26]
            @test li.local_top_vector_a_block_indices == [20]
            @test li.a_block_sub_selection_indices == 1:1
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:19, 21:25, 27:45)
            @test li.local_bottom_vector_indices == vcat(1:19, 21:25, 27:45)
            @test li.top_vector_indices == [20, 26]
            @test li.global_top_vector_size == 2
            @test li.local_top_vector_indices == [20, 26]
            @test li.local_top_vector_a_block_indices == [26]
            @test li.a_block_sub_selection_indices == 2:2
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fft_fff()
    # With the 2nd dimension divided in two and the 3rd dimension periodic, the global
    # index division is (where columns are the right-most index, rows are the centre
    # index, and indices within each cell are the left-most index):
    # -------------------------
    # | 1:3   | 16:18 | 1:3   |
    # -------------------------
    # | 4:6   | 19:21 | 4:6   |
    # =========================
    # ∥ 7:9   ∥ 22:24 ∥ 7:9   ∥
    # =========================
    # | 10:12 | 25:27 | 10:12 |
    # -------------------------
    # | 13:15 | 28:30 | 13:15 |
    # -------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, true]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 7:9)
            @test li.local_bottom_vector_indices == vcat(7:9, 16:18, 25:27)
            @test li.top_vector_indices == vcat(1:6, 16:21, 1:6)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 10:15, 19:24)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 7:9)
            @test li.local_bottom_vector_indices == vcat(1:3, 10:12, 19:21)
            @test li.top_vector_indices == vcat(10:15, 25:30, 10:15)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
            @test li.local_top_vector_a_block_indices == vcat(4:9, 13:18, 22:27)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 7:9)
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:30, 1:6, 10:15)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 16:21, 31:36)
            @test li.a_block_sub_selection_indices == vcat(1:6, 13:18, 25:30)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 7:9)
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:30, 1:6, 10:15)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(10:15, 25:30, 40:45)
            @test li.a_block_sub_selection_indices == vcat(7:12, 19:24, 31:36)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_fft_fft()
    nelement_list = [1, 2, 1]
    periodic_list = [false, false, true]
    remove_boundaries_list = [false, false, true]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:9, 22:24, 1:9)
            @test li.local_bottom_vector_indices == vcat(1:9, 16:27)
            @test li.top_vector_indices == vcat(16:21)
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == 10:15
            @test li.local_top_vector_a_block_indices == 10:15
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:15, 22:24, 7:15)
            @test li.local_bottom_vector_indices == vcat(1:12, 19:27)
            @test li.top_vector_indices == 25:30
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == 13:18
            @test li.local_top_vector_a_block_indices == 13:18
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:15, 22:24, 1:15)
            @test li.local_bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.top_vector_indices == vcat(16:21, 25:30)
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == vcat(16:21, 25:30)
            @test li.local_top_vector_a_block_indices == 16:21
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:15, 22:24, 1:15)
            @test li.local_bottom_vector_indices == vcat(1:15, 22:24, 31:45)
            @test li.top_vector_indices == vcat(16:21, 25:30)
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == vcat(16:21, 25:30)
            @test li.local_top_vector_a_block_indices == 25:30
            @test li.a_block_sub_selection_indices == 7:12
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 1
            @test li.level_dimensions[3].n_local == 1
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_ftf_fff()
    # With the 2nd dimension divided in two and the 2nd dimension periodic, the global
    # index division is (where columns are the right-most index, rows are the centre
    # index, and indices within each cell are the left-most index):
    # -------------------------
    # | 1:3   | 16:18 | 31:33 |
    # -------------------------
    # | 4:6   | 19:21 | 34:36 |
    # =========================
    # ∥ 7:9   ∥ 22:24 ∥ 37:39 ∥
    # =========================
    # | 10:12 | 25:27 | 40:42 |
    # -------------------------
    # | 1:3   | 16:18 | 31:33 |
    # -------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [false, true, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 16:18, 22:24, 31:33, 37:39)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(4:6, 19:21, 34:36)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 1:3, 22:24, 16:18, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(10:12, 25:27, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 31:33, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 31:33, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_ftf_ftf()
    nelement_list = [1, 2, 1]
    periodic_list = [false, true, false]
    remove_boundaries_list = [false, true, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 16:18, 22:24, 31:33, 37:39)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(4:6, 19:21, 34:36)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 1:3, 22:24, 16:18, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(10:12, 25:27, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 31:33, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 31:33, 37:39, 31:33)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_ftt_fff()
    # With the 2nd dimension divided in two and the 2nd and 3rd dimensions periodic, the
    # global index division is (where columns are the right-most index, rows are the
    # centre index, and indices within each cell are the left-most index):
    # -------------------------
    # | 1:3   | 16:18 | 1:3   |
    # -------------------------
    # | 4:6   | 19:21 | 4:6   |
    # =========================
    # ∥ 7:9   ∥ 22:24 ∥ 7:9   ∥
    # =========================
    # | 10:12 | 25:27 | 10:12 |
    # -------------------------
    # | 1:3   | 16:18 | 1:3   |
    # -------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [false, true, true]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 16:18, 22:24, 1:3, 7:9)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(4:6, 19:21, 4:6)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 1:3, 22:24, 16:18, 7:9, 1:3)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == vcat(10:12, 25:27, 10:12)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 1:3, 7:9, 1:3)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 4:6, 10:12)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(1:3, 7:9, 1:3, 16:18, 22:24, 16:18, 1:3, 7:9, 1:3)
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 4:6, 10:12)
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_tff_fff()
    # With the 2nd dimension divided in two and the 1st dimension periodic, the global
    # index division is (where columns are the right-most index, rows are the centre
    # index, and indices within each cell are the left-most index):
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 31,32,31 |
    # ----------------------------------
    # | 4,5,4    | 19,20,19 | 34,35,34 |
    # ==================================
    # ∥ 7,8,7    ∥ 22,23,22 ∥ 37,38,37 ∥
    # ==================================
    # | 10,11,10 | 25,26,25 | 40,41,40 |
    # ----------------------------------
    # | 13,14,13 | 28,29,28 | 43,44,43 |
    # ----------------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [true, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(7:9, 16:18, 25:27)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 16, 17, 16, 19, 20, 19, 31, 32, 31, 34, 35, 34]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 10:15, 19:24)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(1:3, 10:12, 19:21)
            @test li.top_vector_indices == [10, 11, 10, 13, 14, 13, 25, 26, 25, 28, 29, 28, 40, 41, 40, 43, 44, 43]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
            @test li.local_top_vector_a_block_indices == vcat(4:9, 13:18, 22:27)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13, 16, 17, 16, 19, 20, 19, 25, 26, 25, 28, 29, 28, 31, 32, 31, 34, 35, 34, 40, 41, 40, 43, 44, 43]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 16:21, 31:36)
            @test li.a_block_sub_selection_indices == vcat(1:6, 13:18, 25:30)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13, 16, 17, 16, 19, 20, 19, 25, 26, 25, 28, 29, 28, 31, 32, 31, 34, 35, 34, 40, 41, 40, 43, 44, 43]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(10:15, 25:30, 40:45)
            @test li.a_block_sub_selection_indices == vcat(7:12, 19:24, 31:36)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_tff_tff()
    nelement_list = [1, 2, 1]
    periodic_list = [true, false, false]
    remove_boundaries_list = [true, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 1, 4, 4, 7, 8, 7, 16, 16, 19, 19, 22, 23, 22, 31, 31, 34, 34, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:19, 21:22, 24:27)
            @test li.top_vector_indices == [2, 5, 17, 20, 32, 35]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 20, 23]
            @test li.local_top_vector_a_block_indices == [2, 5, 11, 14, 20, 23]
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 10, 10, 13, 13, 22, 23, 22, 25, 25, 28, 28, 37, 38, 37, 40, 40, 43, 43]
            @test li.local_bottom_vector_indices == vcat(1:4, 6:7, 9:13, 15:16, 18:22, 24:25, 27)
            @test li.top_vector_indices == [11, 14, 26, 29, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [5, 8, 14, 17, 23, 26]
            @test li.local_top_vector_a_block_indices == [5, 8, 14, 17, 23, 26]
            @test li.a_block_sub_selection_indices == 1:6
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 1, 4, 4, 7, 8, 7, 10, 10, 13, 13, 16, 16, 19, 19, 22, 23, 22, 25, 25, 28, 28, 31, 31, 34, 34, 37, 38, 37, 40, 40, 43, 43]
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.local_top_vector_a_block_indices == [2, 5, 17, 20, 32, 35]
            @test li.a_block_sub_selection_indices == vcat(1:2, 5:6, 9:10)
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 1, 4, 4, 7, 8, 7, 10, 10, 13, 13, 16, 16, 19, 19, 22, 23, 22, 25, 25, 28, 28, 31, 31, 34, 34, 37, 38, 37, 40, 40, 43, 43]
            @test li.local_bottom_vector_indices == vcat(1, 3:4, 6:10, 12:13, 15:16, 18:19, 21:25, 27:28, 30:31, 33:34, 36:40, 42:43, 45)
            @test li.top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.global_top_vector_size == 12
            @test li.local_top_vector_indices == [2, 5, 11, 14, 17, 20, 26, 29, 32, 35, 41, 44]
            @test li.local_top_vector_a_block_indices == [11, 14, 26, 29, 41, 44]
            @test li.a_block_sub_selection_indices == vcat(3:4, 7:8, 11:12)
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 1
            @test li.level_dimensions[1].n_local == 1
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_tft_fff()
    # With the 2nd dimension divided in two and the 1st and 3rd dimensions periodic, the
    # global index division is (where columns are the right-most index, rows are the
    # centre index, and indices within each cell are the left-most index):
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 1,2,1    |
    # ----------------------------------
    # | 4,5,4    | 19,20,19 | 4,5,4    |
    # ==================================
    # ∥ 7,8,7    ∥ 22,23,22 ∥ 7,8,7    ∥
    # ==================================
    # | 10,11,10 | 25,26,25 | 10,11,10 |
    # ----------------------------------
    # | 13,14,13 | 28,29,28 | 13,14,13 |
    # ----------------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [true, false, true]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 7, 8, 7]
            @test li.local_bottom_vector_indices == vcat(7:9, 16:18, 25:27)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 16, 17, 16, 19, 20, 19, 1, 2, 1, 4, 5, 4]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 10:15, 19:24)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 7, 8, 7]
            @test li.local_bottom_vector_indices == vcat(1:3, 10:12, 19:21)
            @test li.top_vector_indices == [10, 11, 10, 13, 14, 13, 25, 26, 25, 28, 29, 28, 10, 11, 10, 13, 14, 13]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
            @test li.local_top_vector_a_block_indices == vcat(4:9, 13:18, 22:27)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 7, 8, 7]
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13, 16, 17, 16, 19, 20, 19, 25, 26, 25, 28, 29, 28, 1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(1:6, 16:21, 31:36)
            @test li.a_block_sub_selection_indices == vcat(1:6, 13:18, 25:30)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 22, 23, 22, 7, 8, 7]
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.top_vector_indices == [1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13, 16, 17, 16, 19, 20, 19, 25, 26, 25, 28, 29, 28, 1, 2, 1, 4, 5, 4, 10, 11, 10, 13, 14, 13]
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
            @test li.local_top_vector_a_block_indices == vcat(10:15, 25:30, 40:45)
            @test li.a_block_sub_selection_indices == vcat(7:12, 19:24, 31:36)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 2
            @test li.level_dimensions[2].n_local == 2
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_ttf_fff()
    # With the 2nd dimension divided in two and the 1st and 2nd dimensions periodic, the
    # global index division is (where columns are the right-most index, rows are the
    # centre index, and indices within each cell are the left-most index):
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 31,32,31 |
    # ----------------------------------
    # | 4,5,4    | 19,20,19 | 34,35,34 |
    # ==================================
    # ∥ 7,8,7    ∥ 22,23,22 ∥ 37,38,37 ∥
    # ==================================
    # | 10,11,10 | 25,26,25 | 40,41,40 |
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 31,32,31 |
    # ----------------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [true, true, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 16, 17, 16, 22, 23, 22, 31, 32, 31, 37, 38, 37]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == [4, 5, 4, 19, 20, 19, 34, 35, 34]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 1, 2, 1, 22, 23, 22, 16, 17, 16, 37, 38, 37, 31, 32, 31]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == [10, 11, 10, 25, 26, 25, 40, 41, 40]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 1, 2, 1, 16, 17, 16, 22, 23, 22, 16, 17, 16, 31, 32, 31, 37, 38, 37, 31, 32, 31]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == [4, 5, 4, 10, 11, 10, 19, 20, 19, 25, 26, 25, 34, 35, 34, 40, 41, 40]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 1, 2, 1, 16, 17, 16, 22, 23, 22, 16, 17, 16, 31, 32, 31, 37, 38, 37, 31, 32, 31]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == [4, 5, 4, 10, 11, 10, 19, 20, 19, 25, 26, 25, 34, 35, 34, 40, 41, 40]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_121_ttt_fff()
    # With the 2nd dimension divided in two and the 1st, 2nd, and 3rd dimensions periodic,
    # the global index division is (where columns are the right-most index, rows are the
    # centre index, and indices within each cell are the left-most index):
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 1,2,1    |
    # ----------------------------------
    # | 4,5,4    | 19,20,19 | 4,5,4    |
    # ==================================
    # ∥ 7,8,7    ∥ 22,23,22 ∥ 7,8,7    ∥
    # ==================================
    # | 10,11,10 | 25,26,25 | 10,11,10 |
    # ----------------------------------
    # | 1,2,1    | 16,17,16 | 1,2,1    |
    # ----------------------------------
    nelement_list = [1, 2, 1]
    periodic_list = [true, true, true]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [1, 2, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 16, 17, 16, 22, 23, 22, 1, 2, 1, 7, 8, 7]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == [4, 5, 4, 19, 20, 19, 4, 5, 4]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [7, 8, 7, 1, 2, 1, 22, 23, 22, 16, 17, 16, 7, 8, 7, 1, 2, 1]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:12, 16:21, 25:27)
            @test li.top_vector_indices == [10, 11, 10, 25, 26, 25, 10, 11, 10]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 13:15, 22:24)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 13:15, 22:24)
            @test li.a_block_sub_selection_indices == 1:9
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 1, 2, 1, 16, 17, 16, 22, 23, 22, 16, 17, 16, 1, 2, 1, 7, 8, 7, 1, 2, 1]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == [4, 5, 4, 10, 11, 10, 19, 20, 19, 25, 26, 25, 4, 5, 4, 10, 11, 10]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(4:6, 19:21, 34:36)
            @test li.a_block_sub_selection_indices == vcat(1:3, 7:9, 13:15)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [1, 2, 1, 7, 8, 7, 1, 2, 1, 16, 17, 16, 22, 23, 22, 16, 17, 16, 1, 2, 1, 7, 8, 7, 1, 2, 1]
            @test li.local_bottom_vector_indices == vcat(1:3, 7:9, 13:18, 22:24, 28:33, 37:39, 43:45)
            @test li.top_vector_indices == [4, 5, 4, 10, 11, 10, 19, 20, 19, 25, 26, 25, 4, 5, 4, 10, 11, 10]
            @test li.global_top_vector_size == 18
            @test li.local_top_vector_indices == vcat(4:6, 10:12, 19:21, 25:27, 34:36, 40:42)
            @test li.local_top_vector_a_block_indices == vcat(10:12, 25:27, 40:42)
            @test li.a_block_sub_selection_indices == vcat(4:6, 10:12, 16:18)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 3
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 1
            @test li.level_dimensions[2].n_local == 1
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_211()
    # With the 3rd dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # --------==---------------==---------------==--------
    # | 1:2  ;3 ;4:5   | 16:17;18;19:20 | 31:32;33;34:35 |
    # --------==---------------==---------------==--------
    # | 6:7  ;8 ;9:10  | 21:22;23;24:25 | 36:37;38;39:40 |
    # --------==---------------==---------------==--------
    # | 11:12;13;14:15 | 26:27;28;29:30 | 41:42;43;44:45 |
    # --------==---------------==---------------==--------
    nelement_list = [2, 1, 1]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [2, 1, 1]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.local_bottom_vector_indices == [3, 6, 9, 12, 15, 18, 21, 24, 27]
            @test li.top_vector_indices == vcat(1:2, 6:7, 11:12, 16:17, 21:22, 26:27, 31:32, 36:37, 41:42)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:2, 4:5, 7:8, 10:11, 13:14, 16:17, 19:20, 22:23, 25:26)
            @test li.local_top_vector_a_block_indices == vcat(1:2, 4:5, 7:8, 10:11, 13:14, 16:17, 19:20, 22:23, 25:26)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [1, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.local_bottom_vector_indices == [1, 4, 7, 10, 13, 16, 19, 22, 25]
            @test li.top_vector_indices == vcat(4:5, 9:10, 14:15, 19:20, 24:25, 29:30, 34:35, 39:40, 44:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(2:3, 5:6, 8:9, 11:12, 14:15, 17:18, 20:21, 23:24, 26:27)
            @test li.local_top_vector_a_block_indices == vcat(2:3, 5:6, 8:9, 11:12, 14:15, 17:18, 20:21, 23:24, 26:27)
            @test li.a_block_sub_selection_indices == 1:18
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    n_groups = 2
    nrank_list = [1, 1, 1]
    n_shared = 2
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.local_bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.top_vector_indices == vcat(1:2, 4:7, 9:12, 14:17, 19:22, 24:27, 29:32, 34:37, 39:42, 44:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:2, 4:7, 9:12, 14:17, 19:22, 24:27, 29:32, 34:37, 39:42, 44:45)
            @test li.local_top_vector_a_block_indices == vcat(1:2, 6:7, 11:12, 16:17, 21:22, 26:27, 31:32, 36:37, 41:42)
            @test li.a_block_sub_selection_indices == vcat(1:2, 5:6, 9:10, 13:14, 17:18, 21:22, 25:26, 29:30, 33:34)
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.local_bottom_vector_indices == [3, 8, 13, 18, 23, 28, 33, 38, 43]
            @test li.top_vector_indices == vcat(1:2, 4:7, 9:12, 14:17, 19:22, 24:27, 29:32, 34:37, 39:42, 44:45)
            @test li.global_top_vector_size == 36
            @test li.local_top_vector_indices == vcat(1:2, 4:7, 9:12, 14:17, 19:22, 24:27, 29:32, 34:37, 39:42, 44:45)
            @test li.local_top_vector_a_block_indices == vcat(4:5, 9:10, 14:15, 19:20, 24:25, 29:30, 34:35, 39:40, 44:45)
            @test li.a_block_sub_selection_indices == vcat(3:4, 7:8, 11:12, 15:16, 19:20, 23:24, 27:28, 31:32, 35:36)
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 2
            @test li.level_dimensions[1].n_local == 2
            @test li.level_dimensions[1].nelement == 1
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 3
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 1
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 3
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 1
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_224()
    # With the 3rd dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # ----------------------------------=========-----------------------------------------
    # | 1:5   | 26:30 | 51:55 | 76:80  ∥ 101:105 ∥ 126:130 | 151:155 | 176:180 | 201:205 |
    # ----------------------------------=========-----------------------------------------
    # | 6:10  | 31:35 | 56:60 | 81:85  ∥ 106:110 ∥ 131:135 | 156:160 | 181:185 | 206:210 |
    # ----------------------------------=========-----------------------------------------
    # | 11:15 | 36:40 | 61:65 | 86:90  ∥ 111:115 ∥ 136:140 | 161:165 | 186:190 | 211:215 |
    # ----------------------------------=========-----------------------------------------
    # | 16:20 | 41:45 | 66:70 | 91:95  ∥ 116:120 ∥ 141:145 | 166:170 | 191:195 | 216:220 |
    # ----------------------------------=========-----------------------------------------
    # | 21:25 | 46:50 | 71:75 | 96:100 ∥ 121:125 ∥ 146:150 | 171:175 | 196:200 | 221:225 |
    # ----------------------------------=========-----------------------------------------
    nelement_list = [2, 2, 4]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [2, 2, 2]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(101:103, 106:108, 111:113)
            @test li.local_bottom_vector_indices == 37:45
            @test li.top_vector_indices == vcat(1:3, 6:8, 11:13, 26:28, 31:33, 36:38, 51:53, 56:58, 61:63, 76:78, 81:83, 86:88)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 1:36
            @test li.local_top_vector_a_block_indices == 1:36
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [1, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(103:105, 108:110, 113:115)
            @test li.local_bottom_vector_indices == 37:45
            @test li.top_vector_indices == vcat(3:5, 8:10, 13:15, 28:30, 33:35, 38:40, 53:55, 58:60, 63:65, 78:80, 83:85, 88:90)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 1:36
            @test li.local_top_vector_a_block_indices == 1:36
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 2
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(111:113, 116:118, 121:123)
            @test li.local_bottom_vector_indices == 37:45
            @test li.top_vector_indices == vcat(11:13, 16:18, 21:23, 36:38, 41:43, 46:48, 61:63, 66:68, 71:73, 86:88, 91:93, 96:98)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 1:36
            @test li.local_top_vector_a_block_indices == 1:36
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 3
        irank_list = [1, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(113:115, 118:120, 123:125)
            @test li.local_bottom_vector_indices == 37:45
            @test li.top_vector_indices == vcat(13:15, 18:20, 23:25, 38:40, 43:45, 48:50, 63:65, 68:70, 73:75, 88:90, 93:95, 98:100)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 1:36
            @test li.local_top_vector_a_block_indices == 1:36
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == false
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 4
        irank_list = [0, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(101:103, 106:108, 111:113)
            @test li.local_bottom_vector_indices == 1:9
            @test li.top_vector_indices == vcat(126:128, 131:133, 136:138, 151:153, 156:158, 161:163, 176:178, 181:183, 186:188, 201:203, 206:208, 211:213)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 10:45
            @test li.local_top_vector_a_block_indices == 10:45
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 5
        irank_list = [1, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(103:105, 108:110, 113:115)
            @test li.local_bottom_vector_indices == 1:9
            @test li.top_vector_indices == vcat(128:130, 133:135, 138:140, 153:155, 158:160, 163:165, 178:180, 183:185, 188:190, 203:205, 208:210, 213:215)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 10:45
            @test li.local_top_vector_a_block_indices == 10:45
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 6
        irank_list = [0, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(111:113, 116:118, 121:123)
            @test li.local_bottom_vector_indices == 1:9
            @test li.top_vector_indices == vcat(136:138, 141:143, 146:148, 161:163, 166:168, 171:173, 186:188, 191:193, 196:198, 211:213, 216:218, 221:223)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 10:45
            @test li.local_top_vector_a_block_indices == 10:45
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end

        irank = 7
        irank_list = [1, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(113:115, 118:120, 123:125)
            @test li.local_bottom_vector_indices == 1:9
            @test li.top_vector_indices == vcat(138:140, 143:145, 148:150, 163:165, 168:170, 173:175, 188:190, 193:195, 198:200, 213:215, 218:220, 223:225)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == 10:45
            @test li.local_top_vector_a_block_indices == 10:45
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == false
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 4
            @test li.level_dimensions[3].n_local == 4
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 1
            @test li.level_dimensions[3].irank == 0
        end
    end

    return nothing
end

function test_split_indices_3d_242()
    # With the 2nd dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # -----------------------------------------------
    # | 1:5   | 46:50 | 91:95   | 136:140 | 181:185 |
    # -----------------------------------------------
    # | 6:10  | 51:55 | 96:100  | 141:145 | 186:190 |
    # -----------------------------------------------
    # | 11:15 | 56:60 | 101:105 | 146:150 | 191:195 |
    # -----------------------------------------------
    # | 16:20 | 61:65 | 106:110 | 151:155 | 196:200 |
    # ===============================================
    # ∥ 21:25 ∥ 66:70 ∥ 111:115 ∥ 156:160 ∥ 201:205 ∥
    # ===============================================
    # | 26:30 | 71:75 | 116:120 | 161:165 | 206:210 |
    # -----------------------------------------------
    # | 31:35 | 76:80 | 121:125 | 166:170 | 211:215 |
    # -----------------------------------------------
    # | 36:40 | 81:85 | 126:130 | 171:175 | 216:220 |
    # -----------------------------------------------
    # | 41:45 | 86:90 | 131:135 | 176:180 | 221:225 |
    # -----------------------------------------------
    nelement_list = [2, 4, 2]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [2, 2, 2]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(21:23, 66:68, 111:113)
            @test li.local_bottom_vector_indices == vcat(13:15, 28:30, 43:45)
            @test li.top_vector_indices == vcat(1:3, 6:8, 11:13, 16:18, 46:48, 51:53, 56:58, 61:63, 91:93, 96:98, 101:103, 106:108)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:12, 16:27, 31:42)
            @test li.local_top_vector_a_block_indices == vcat(1:12, 16:27, 31:42)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [1, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(23:25, 68:70, 113:115)
            @test li.local_bottom_vector_indices == vcat(13:15, 28:30, 43:45)
            @test li.top_vector_indices == vcat(3:5, 8:10, 13:15, 18:20, 48:50, 53:55, 58:60, 63:65, 93:95, 98:100, 103:105, 108:110)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:12, 16:27, 31:42)
            @test li.local_top_vector_a_block_indices == vcat(1:12, 16:27, 31:42)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 2
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(21:23, 66:68, 111:113)
            @test li.local_bottom_vector_indices == vcat(1:3, 16:18, 31:33)
            @test li.top_vector_indices == vcat(26:28, 31:33, 36:38, 41:43, 71:73, 76:78, 81:83, 86:88, 116:118, 121:123, 126:128, 131:133)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(4:15, 19:30, 34:45)
            @test li.local_top_vector_a_block_indices == vcat(4:15, 19:30, 34:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 3
        irank_list = [1, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(23:25, 68:70, 113:115)
            @test li.local_bottom_vector_indices == vcat(1:3, 16:18, 31:33)
            @test li.top_vector_indices == vcat(28:30, 33:35, 38:40, 43:45, 73:75, 78:80, 83:85, 88:90, 118:120, 123:125, 128:130, 133:135)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(4:15, 19:30, 34:45)
            @test li.local_top_vector_a_block_indices == vcat(4:15, 19:30, 34:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 4
        irank_list = [0, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(111:113, 156:158, 201:203)
            @test li.local_bottom_vector_indices == vcat(13:15, 28:30, 43:45)
            @test li.top_vector_indices == vcat(91:93, 96:98, 101:103, 106:108, 136:138, 141:143, 146:148, 151:153, 181:183, 186:188, 191:193, 196:198)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:12, 16:27, 31:42)
            @test li.local_top_vector_a_block_indices == vcat(1:12, 16:27, 31:42)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 5
        irank_list = [1, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(113:115, 158:160, 203:205)
            @test li.local_bottom_vector_indices == vcat(13:15, 28:30, 43:45)
            @test li.top_vector_indices == vcat(93:95, 98:100, 103:105, 108:110, 138:140, 143:145, 148:150, 153:155, 183:185, 188:190, 193:195, 198:200)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:12, 16:27, 31:42)
            @test li.local_top_vector_a_block_indices == vcat(1:12, 16:27, 31:42)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 6
        irank_list = [0, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(111:113, 156:158, 201:203)
            @test li.local_bottom_vector_indices == vcat(1:3, 16:18, 31:33)
            @test li.top_vector_indices == vcat(116:118, 121:123, 126:128, 131:133, 161:163, 166:168, 171:173, 176:178, 206:208, 211:213, 216:218, 221:223)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(4:15, 19:30, 34:45)
            @test li.local_top_vector_a_block_indices == vcat(4:15, 19:30, 34:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 7
        irank_list = [1, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(113:115, 158:160, 203:205)
            @test li.local_bottom_vector_indices == vcat(1:3, 16:18, 31:33)
            @test li.top_vector_indices == vcat(118:120, 123:125, 128:130, 133:135, 163:165, 168:170, 173:175, 178:180, 208:210, 213:215, 218:220, 223:225)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(4:15, 19:30, 34:45)
            @test li.local_top_vector_a_block_indices == vcat(4:15, 19:30, 34:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 5
            @test li.level_dimensions[1].n_local == 3
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 2
            @test li.level_dimensions[1].irank == 1
            @test li.level_dimensions[2].n == 4
            @test li.level_dimensions[2].n_local == 4
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end
    end

    return nothing
end

function test_split_indices_3d_422()
    # With the 1st dimension divided in two, the global index division is (where columns
    # are the right-most index, rows are the centre index, and indices within each cell
    # are the left-most index):
    # --------==---------------==-----------------===-------------------===-------------------===----------
    # | 1:4  ; 5;6:9   | 46:49;50;51:54 | 91:94  ;95 ;96:99   | 136:139;140;141:144 | 181:184;185;186:189 |
    # --------==---------------==-----------------===-------------------===-------------------===----------
    # | 10:13;14;15:18 | 55:58;59;60:63 | 100:103;104;105:108 | 145:148;149;150:153 | 190:193;194;195:198 |
    # --------==---------------==-----------------===-------------------===-------------------===----------
    # | 19:22;23;24:27 | 64:67;68;69:72 | 109:112;113;114:117 | 154:157;158;159:162 | 199:202;203;204:207 |
    # --------==---------------==-----------------===-------------------===-------------------===----------
    # | 28:31;32;33:36 | 73:76;77;78:81 | 118:121;122;123:126 | 163:166;167;168:171 | 208:211;212;213:216 |
    # --------==---------------==-----------------===-------------------===-------------------===----------
    # | 37:40;41;42:45 | 82:85;86;87:90 | 127:130;131;132:135 | 172:175;176;177:180 | 217:220;221;222:225 |
    # --------==---------------==-----------------===-------------------===-------------------===----------
    nelement_list = [4, 2, 2]
    periodic_list = [false, false, false]
    remove_boundaries_list = [false, false, false]

    n_groups = 2
    nrank_list = [2, 2, 2]
    n_shared = 1
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [5, 14, 23, 50, 59, 68, 95, 104, 113]
            @test li.local_bottom_vector_indices == [5, 10, 15, 20, 25, 30, 35, 40, 45]
            @test li.top_vector_indices == vcat(1:4, 10:13, 19:22, 46:49, 55:58, 64:67, 91:94, 100:103, 109:112)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.local_top_vector_a_block_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 1
        irank_list = [1, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [5, 14, 23, 50, 59, 68, 95, 104, 113]
            @test li.local_bottom_vector_indices == [1, 6, 11, 16, 21, 26, 31, 36, 41]
            @test li.top_vector_indices == vcat(6:9, 15:18, 24:27, 51:54, 60:63, 69:72, 96:99, 105:108, 114:117)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.local_top_vector_a_block_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 2
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [23, 32, 41, 68, 77, 86, 113, 122, 131]
            @test li.local_bottom_vector_indices == [5, 10, 15, 20, 25, 30, 35, 40, 45]
            @test li.top_vector_indices == vcat(19:22, 28:31, 37:40, 64:67, 73:76, 82:85, 109:112, 118:121, 127:130)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.local_top_vector_a_block_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 3
        irank_list = [1, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [23, 32, 41, 68, 77, 86, 113, 122, 131]
            @test li.local_bottom_vector_indices == [1, 6, 11, 16, 21, 26, 31, 36, 41]
            @test li.top_vector_indices == vcat(24:27, 33:36, 42:45, 69:72, 78:81, 87:90, 114:117, 123:126, 132:135)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.local_top_vector_a_block_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 0
        end

        irank = 4
        irank_list = [0, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [95, 104, 113, 140, 149, 158, 185, 194, 203]
            @test li.local_bottom_vector_indices == [5, 10, 15, 20, 25, 30, 35, 40, 45]
            @test li.top_vector_indices == vcat(91:94, 100:103, 109:112, 136:139, 145:148, 154:157, 181:184, 190:193, 199:202)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.local_top_vector_a_block_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 5
        irank_list = [1, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [95, 104, 113, 140, 149, 158, 185, 194, 203]
            @test li.local_bottom_vector_indices == [1, 6, 11, 16, 21, 26, 31, 36, 41]
            @test li.top_vector_indices == vcat(96:99, 105:108, 114:117, 141:144, 150:153, 159:162, 186:189, 195:198, 204:207)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.local_top_vector_a_block_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 0
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 6
        irank_list = [0, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [113, 122, 131, 158, 167, 176, 203, 212, 221]
            @test li.local_bottom_vector_indices == [5, 10, 15, 20, 25, 30, 35, 40, 45]
            @test li.top_vector_indices == vcat(109:112, 118:121, 127:130, 154:157, 163:166, 172:175, 199:202, 208:211, 217:220)
            @test li.local_top_vector_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.local_top_vector_a_block_indices == vcat(1:4, 6:9, 11:14, 16:19, 21:24, 26:29, 31:34, 36:39, 41:44)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == false
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end

        irank = 7
        irank_list = [1, 1, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == [113, 122, 131, 158, 167, 176, 203, 212, 221]
            @test li.local_bottom_vector_indices == [1, 6, 11, 16, 21, 26, 31, 36, 41]
            @test li.top_vector_indices == vcat(114:117, 123:126, 132:135, 159:162, 168:171, 177:180, 204:207, 213:216, 222:225)
            @test li.global_top_vector_size == 200
            @test li.local_top_vector_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.local_top_vector_a_block_indices == vcat(2:5, 7:10, 12:15, 17:20, 22:25, 27:30, 32:35, 37:40, 42:45)
            @test li.a_block_sub_selection_indices == 1:36
            @test li.level_dimensions[1].has_lower_boundary == false
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[3].has_lower_boundary == true
            @test li.level_dimensions[3].has_upper_boundary == true
            @test li.level_dimensions[1].n == 4
            @test li.level_dimensions[1].n_local == 4
            @test li.level_dimensions[1].nelement == 2
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 5
            @test li.level_dimensions[2].n_local == 3
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 3
            @test li.level_dimensions[2].nrank == 2
            @test li.level_dimensions[2].irank == 1
            @test li.level_dimensions[3].n == 5
            @test li.level_dimensions[3].n_local == 3
            @test li.level_dimensions[3].nelement == 2
            @test li.level_dimensions[3].ngrid == 3
            @test li.level_dimensions[3].nrank == 2
            @test li.level_dimensions[3].irank == 1
        end
    end

    return nothing
end

# A collection of edge cases that have previously caused problems in
# test_finite_element_matrices.jl.
function test_split_indices_edge_cases()
    # With the 2nd dimension divided in two, the global index division is (where rows are
    # the right index, and indices within each row are the left index):
    # -----------
    # | 1:9     |
    # -----------
    # | 10:18   |
    # -----------
    # | 19:27   |
    # -----------
    # | 28:36   |
    # -----------
    # | 37:45   |
    # -----------
    # | 46:54   |
    # -----------
    # | 55:63   |
    # -----------
    # | 64:72   |
    # ===========
    # ∥ 73:81   ∥
    # ===========
    # | 82:90   |
    # -----------
    # | 91:99   |
    # -----------
    # | 100:108 |
    # -----------
    # | 109:117 |
    # -----------
    # | 118:126 |
    # -----------
    # | 127:135 |
    # -----------
    # | 136:144 |
    # -----------
    # | 145:153 |
    # -----------
    nelement_list = [4, 4]
    ngrid_list = [3, 5]
    periodic_list = [false, false]
    remove_boundaries_list = [false, false]

    n_groups = 3
    nrank_list = [1, 1]
    n_shared = 9
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 1
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 2
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 3
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 4
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 5
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 6
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 7
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 8
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 73:81
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:72, 82:153)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

    end

    nelement_list = [4, 4]
    ngrid_list = [3, 5]
    periodic_list = [true, false]
    remove_boundaries_list = [false, false]

    n_groups = 3
    nrank_list = [1, 1]
    n_shared = 9
    @testset "nelement_list=$nelement_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank_list=$nrank_list, n_shared=$n_shared, n_groups=$n_groups" begin
        irank = 0
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 1
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 2
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:72
            @test li.a_block_sub_selection_indices == 1:72
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == true
            @test li.level_dimensions[2].has_upper_boundary == false
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 3
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 4
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 5
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 82:153
            @test li.a_block_sub_selection_indices == 73:144
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 6
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 7
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end

        irank = 8
        irank_list = [0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid_list, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(73:80, 73)
            @test li.local_bottom_vector_indices == 73:81
            @test li.top_vector_indices == vcat(1:8, 1, 10:17, 10, 19:26, 19, 28:35, 28, 37:44, 37, 46:53, 46, 55:62, 55, 64:71, 64, 82:89, 82, 91:98, 91, 100:107, 100, 109:116, 109, 118:125, 118, 127:134, 127, 136:143, 136, 145:152, 145)
            @test li.global_top_vector_size == 144
            @test li.local_top_vector_indices == vcat(1:72, 82:153)
            @test li.local_top_vector_a_block_indices == 1:0
            @test li.a_block_sub_selection_indices == 1:0
            @test li.level_dimensions[1].has_lower_boundary == true
            @test li.level_dimensions[1].has_upper_boundary == true
            @test li.level_dimensions[2].has_lower_boundary == false
            @test li.level_dimensions[2].has_upper_boundary == true
            @test li.level_dimensions[1].n == 9
            @test li.level_dimensions[1].n_local == 9
            @test li.level_dimensions[1].nelement == 4
            @test li.level_dimensions[1].ngrid == 3
            @test li.level_dimensions[1].nrank == 1
            @test li.level_dimensions[1].irank == 0
            @test li.level_dimensions[2].n == 8
            @test li.level_dimensions[2].n_local == 8
            @test li.level_dimensions[2].nelement == 2
            @test li.level_dimensions[2].ngrid == 5
            @test li.level_dimensions[2].nrank == 1
            @test li.level_dimensions[2].irank == 0
        end
    end

    return nothing
end

function test_pick_dimension_to_split()
    @testset "periodic=$periodic, remove_boundaries=$remove_boundaries" for periodic ∈ (false, true), remove_boundaries ∈ (false, true)
        optimise_schur_complement_size = true
        @testset "optimise_schur_complement_size=$optimise_schur_complement_size" begin
            # No dimensions are distributed, pick largest dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # ...even when n_group does not divide nelement for the largest dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # Check that dimensions with one element are never divided.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=1, ngrid=100, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=1, ngrid=100, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=1, ngrid=100, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=1, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=1, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=1, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=1, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test_throws "All dimensions contain one element" pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size)

            # Some or all dimensions are distributed. Pick only from the distributed
            # dimensions.
            # Pick largest distributed dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # ...but only from dimensions for which n_group divides nelement.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # Extra edge cases, that previously caused problems in
            # test_finite_element_matrices.jl
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 3
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2
        end

        optimise_schur_complement_size = false
        @testset "optimise_schur_complement_size=$optimise_schur_complement_size" begin
            # No dimensions are distributed, pick largest dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # ...but only when n_group divides nelement for the largest dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            # ...and if no dimensions are exactly divisible, fall back on the largest
            # dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=4, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=9, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            # Some or all dimensions are distributed. Pick only from the distributed
            # dimensions.
            # Pick largest distributed dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # ...but only when n_group divides nelement for the largest dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=4, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            # ...and if no dimensions are exactly divisible, fall back on the largest
            # dimension.
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 3

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=3, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 2
            dimensions = [
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=3, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=2, ngrid=10, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2

            # Extra edge cases, that previously caused problems in
            # test_finite_element_matrices.jl
            n_groups = 2
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 1

            n_groups = 3
            dimensions = [
                create_dimension(; nelement=2, ngrid=3, nrank=2, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=3, ngrid=4, nrank=3, irank=0, periodic=false, remove_boundaries=false),
                create_dimension(; nelement=4, ngrid=5, nrank=1, irank=0, periodic=false, remove_boundaries=false),
            ]
            @test pick_dimension_to_split(dimensions, n_groups, optimise_schur_complement_size) == 2
        end
    end

    return nothing
end

function test_indices()
    @testset "Test index splitting" begin
        @testset "test_split_indices_1d_2proc" test_split_indices_1d_2proc()
        @testset "test_split_indices_1d_2proc_periodic" test_split_indices_1d_2proc_periodic()
        #@testset "test_split_indices_1d_4group" test_split_indices_1d_4group()
        #@testset "test_split_indices_1d_3group" test_split_indices_1d_3group()
        #@testset "test_split_indices_1d_2group_periodic" test_split_indices_1d_2group_periodic()
        #@testset "test_split_indices_1d_4group_periodic" test_split_indices_1d_4group_periodic()
        #@testset "test_split_indices_1d_3group_periodic" test_split_indices_1d_3group_periodic()
        #@testset "test_split_indices_1d_2group_remove_boundaries" test_split_indices_1d_2group_remove_boundaries()
        #@testset "test_split_indices_1d_4group_remove_boundaries" test_split_indices_1d_4group_remove_boundaries()
        #@testset "test_split_indices_1d_3group_remove_boundaries" test_split_indices_1d_3group_remove_boundaries()
        #@testset "test_split_indices_1d_2group_periodic_remove_boundaries" test_split_indices_1d_2group_periodic_remove_boundaries()
        #@testset "test_split_indices_1d_4group_periodic_remove_boundaries" test_split_indices_1d_4group_periodic_remove_boundaries()
        #@testset "test_split_indices_1d_3group_periodic_remove_boundaries" test_split_indices_1d_3group_periodic_remove_boundaries()
        #@testset "test_split_indices_1d_2group_oddnelement" test_split_indices_1d_2group_oddnelement()
        #@testset "test_split_indices_1d_2group_oddnelement_periodic" test_split_indices_1d_2group_oddnelement_periodic()
        #@testset "test_split_indices_1d_2group_oddnelement_periodic_remove_boundaries" test_split_indices_1d_2group_oddnelement_periodic_remove_boundaries()
        #@testset "test_split_indices_3d_112" test_split_indices_3d_112()
        #@testset "test_split_indices_3d_121_fff_fff" test_split_indices_3d_121_fff_fff()
        #@testset "test_split_indices_3d_121_fff_fft" test_split_indices_3d_121_fff_fft()
        #@testset "test_split_indices_3d_121_fff_ftf" test_split_indices_3d_121_fff_ftf()
        #@testset "test_split_indices_3d_121_fff_ftt" test_split_indices_3d_121_fff_ftt()
        #@testset "test_split_indices_3d_121_fff_tff" test_split_indices_3d_121_fff_tff()
        #@testset "test_split_indices_3d_121_fff_tft" test_split_indices_3d_121_fff_tft()
        #@testset "test_split_indices_3d_121_fff_ttf" test_split_indices_3d_121_fff_ttf()
        #@testset "test_split_indices_3d_121_fff_ttt" test_split_indices_3d_121_fff_ttt()
        #@testset "test_split_indices_3d_121_fft_fff" test_split_indices_3d_121_fft_fff()
        #@testset "test_split_indices_3d_121_fft_fft" test_split_indices_3d_121_fft_fft()
        #@testset "test_split_indices_3d_121_ftf_fff" test_split_indices_3d_121_ftf_fff()
        #@testset "test_split_indices_3d_121_ftf_ftf" test_split_indices_3d_121_ftf_ftf()
        #@testset "test_split_indices_3d_121_ftt_fff" test_split_indices_3d_121_ftt_fff()
        #@testset "test_split_indices_3d_121_tff_fff" test_split_indices_3d_121_tff_fff()
        #@testset "test_split_indices_3d_121_tff_tff" test_split_indices_3d_121_tff_tff()
        #@testset "test_split_indices_3d_121_tft_fff" test_split_indices_3d_121_tft_fff()
        #@testset "test_split_indices_3d_121_ttf_fff" test_split_indices_3d_121_ttf_fff()
        #@testset "test_split_indices_3d_121_ttt_fff" test_split_indices_3d_121_ttt_fff()
        #@testset "test_split_indices_3d_211" test_split_indices_3d_211()
        #@testset "test_split_indices_3d_224" test_split_indices_3d_224()
        #@testset "test_split_indices_3d_242" test_split_indices_3d_242()
        #@testset "test_split_indices_3d_422" test_split_indices_3d_422()
        #@testset "test_pick_dimension_to_split" test_pick_dimension_to_split()
    end
end
