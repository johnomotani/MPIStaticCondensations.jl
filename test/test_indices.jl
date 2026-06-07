using MPIStaticCondensations
using MPIStaticCondensations: FakeComm, split_matrix, get_global_indices,
                              get_shared_sparse_matrix_csc_buffer
using Test

# Notes
# =====
#
# We do not test global_size or global_bottom_vector_size because they cannot be
# calculated correctly without actual MPI communication.

const ngrid = 3

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
    this_global_size = prod(d.n for d ∈ dimensions)
    local_size = prod(d.n_local for d ∈ dimensions)
    level_indices = get_global_indices(dimensions_without_periodic, collect(1:local_size))
    n_levels = length(block_sizes_list)
    level_info = Any[]
    for (level, bs) ∈ enumerate(block_sizes_list)
        if level == 1 || level == n_levels
            # Only handle periodicity on the final level
            dims = dimensions
        else
            dims = dimensions_without_periodic
        end
        li = split_matrix(dims, level_indices, bs, this_global_size, level==1,
                          level==n_levels, distributed_comm, shared_comm)
        push!(level_info, li)
        this_global_size = li.global_bottom_vector_size
        level_indices = li.bottom_vector_indices
    end

    return level_info, dimensions
end

function test_split_indices_1d_1proc_remove_boundaries()
    nelement_list = [3]
    periodic_list = [false]
    remove_boundaries_list = [true]

    # The interiors and boundaries are:
    # ++-----^^^-----===-----++
    # 1 | 2 | 3 | 4 ∥ 5 ∥ 6 | 7
    # ++-----^^^-----===-----++
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1], [2], [3]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6]
            @test li[1].local_top_vector_indices == [2, 4, 6]
            @test li[1].all_local_top_vector_a_block_indices == [2, 4, 6]
            @test li[1].local_top_vector_a_block_indices == [[2], [4], [6]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1], [2], [3]]
            @test li[1].a_block_lu_selection_indices == [[1], [2], [3]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2, 3], [3, 4]]
            @test li[1].bottom_vector_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:4
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2], []]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1], []]
            @test li[2].a_block_lu_selection_indices == [[1], []]
            @test li[2].a_block_B_column_indices == [[1, 2], [2, 3]]
            @test li[2].bottom_vector_indices == [1, 5, 7]
            @test li[2].local_bottom_vector_indices == [1, 3, 4]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3, 4]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[3].top_vector_indices == [5]
            @test li[3].local_top_vector_indices == [2]
            @test li[3].all_local_top_vector_a_block_indices == [2]
            @test li[3].local_top_vector_a_block_indices == [[2]]
            @test li[3].all_a_block_sub_selection_indices == [1]
            @test li[3].a_block_sub_selection_indices == [[1]]
            @test li[3].a_block_lu_selection_indices == [[1]]
            @test li[3].a_block_B_column_indices == [[1, 2]]
            @test li[3].bottom_vector_indices == [1, 7]
            @test li[3].local_bottom_vector_indices == [1, 3]
            @test li[3].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[3].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[3].local_bottom_vector_repeat_indices == []
            @test li[3].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    return nothing
end

function test_split_indices_1d_1proc_periodic()
    nelement_list = [1]
    periodic_list = [true]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # ==-----==
    # 1 | 2 | 3
    # ==-----==
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2]
            @test li[1].local_top_vector_indices == [2]
            @test li[1].all_local_top_vector_a_block_indices == [2]
            @test li[1].local_top_vector_a_block_indices == [[2]]
            @test li[1].all_a_block_sub_selection_indices == [1]
            @test li[1].a_block_sub_selection_indices == [[1]]
            @test li[1].a_block_lu_selection_indices == [[1]]
            @test li[1].a_block_B_column_indices == [[1]]
            @test li[1].bottom_vector_indices == [1, 1]
            @test li[1].local_bottom_vector_indices == [1, 3]
            @test li[1].local_bottom_vector_no_overlap_indices == [1]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[1].local_bottom_vector_repeat_indices == [2]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    nelement_list = [3]
    periodic_list = [true]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # ++-----^^^-----===-----++
    # 1 | 2 | 3 | 4 ∥ 5 ∥ 6 | 7
    # ++-----^^^-----===-----++
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1], [2], [3]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6]
            @test li[1].local_top_vector_indices == [2, 4, 6]
            @test li[1].all_local_top_vector_a_block_indices == [2, 4, 6]
            @test li[1].local_top_vector_a_block_indices == [[2], [4], [6]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1], [2], [3]]
            @test li[1].a_block_lu_selection_indices == [[1], [2], [3]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2, 3], [3, 4]]
            @test li[1].bottom_vector_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == [4]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2], []]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1], []]
            @test li[2].a_block_lu_selection_indices == [[1], []]
            @test li[2].a_block_B_column_indices == [[1, 2], [2, 3]]
            @test li[2].bottom_vector_indices == [1, 5, 7]
            @test li[2].local_bottom_vector_indices == [1, 3, 4]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3, 4]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[3].top_vector_indices == [5]
            @test li[3].local_top_vector_indices == [2]
            @test li[3].all_local_top_vector_a_block_indices == [2]
            @test li[3].local_top_vector_a_block_indices == [[2]]
            @test li[3].all_a_block_sub_selection_indices == [1]
            @test li[3].a_block_sub_selection_indices == [[1]]
            @test li[3].a_block_lu_selection_indices == [[1]]
            @test li[3].a_block_B_column_indices == [[1]]
            @test li[3].bottom_vector_indices == [1, 1]
            @test li[3].local_bottom_vector_indices == [1, 3]
            @test li[3].local_bottom_vector_no_overlap_indices == [1]
            @test li[3].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[3].local_bottom_vector_repeat_indices == []
            @test li[3].local_bottom_vector_periodic_pairs == [1; 3;;]
        end
    end

    return nothing
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
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_indices == vcat(1:2, 4)
            @test li[1].all_local_top_vector_a_block_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_a_block_indices == [[1, 2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_lu_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_B_column_indices == [[1], [1, 2]]
            @test li[1].bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(6, 8:9)
            @test li[1].local_top_vector_indices == vcat(2, 4:5)
            @test li[1].all_local_top_vector_a_block_indices == vcat(2, 4:5)
            @test li[1].local_top_vector_a_block_indices == [[2], [4,5]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1], [2, 3]]
            @test li[1].a_block_lu_selection_indices == [[1], [2, 3]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2]]
            @test li[1].bottom_vector_indices == [5, 7]
            @test li[1].local_bottom_vector_indices == [1, 3]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [7]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [1]
            @test li[2].local_bottom_vector_no_overlap_indices == [1]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    nrank = 2
    n_shared = 2
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].all_local_top_vector_a_block_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_a_block_indices == [[1, 2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_lu_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_B_column_indices == [[1], [1, 2]]
            @test li[1].bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [1, 3]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6, 8:9)
            @test li[1].all_local_top_vector_a_block_indices == vcat(6, 8:9)
            @test li[1].local_top_vector_a_block_indices == [[6], [8, 9]]
            @test li[1].all_a_block_sub_selection_indices == 4:6
            @test li[1].a_block_sub_selection_indices == [[4], [5, 6]]
            @test li[1].a_block_lu_selection_indices == [[1], [2, 3]]
            @test li[1].a_block_B_column_indices == [[2, 3], [3]]
            @test li[1].bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [1, 3]
            @test li[2].all_local_top_vector_a_block_indices == [3]
            @test li[2].local_top_vector_a_block_indices == [[3]]
            @test li[2].all_a_block_sub_selection_indices == [2]
            @test li[2].a_block_sub_selection_indices == [[2]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    nelement_list = [3]
    periodic_list = [false]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # -----+++-----===-----
    # 1:2 | 3 | 4 ∥ 5 ∥ 6:7
    # -----+++-----===-----
    nrank = 2
    n_shared = 2
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6:7)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6:7)
            @test li[1].all_local_top_vector_a_block_indices == vcat(1:2, 4)
            @test li[1].local_top_vector_a_block_indices == [[1, 2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:3
            @test li[1].a_block_sub_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_lu_selection_indices == [[1, 2], [3]]
            @test li[1].a_block_B_column_indices == [[1], [1, 2]]
            @test li[1].bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4, 6:7)
            @test li[1].local_top_vector_indices == vcat(1:2, 4, 6:7)
            @test li[1].all_local_top_vector_a_block_indices == vcat(6:7)
            @test li[1].local_top_vector_a_block_indices == [[6, 7]]
            @test li[1].all_a_block_sub_selection_indices == 4:5
            @test li[1].a_block_sub_selection_indices == [[4, 5]]
            @test li[1].a_block_lu_selection_indices == [[1, 2]]
            @test li[1].a_block_B_column_indices == [[2]]
            @test li[1].bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 5]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == []
            @test li[2].local_top_vector_a_block_indices == [[]]
            @test li[2].all_a_block_sub_selection_indices == []
            @test li[2].a_block_sub_selection_indices == [[]]
            @test li[2].a_block_lu_selection_indices == [[]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5]
            @test li[2].local_bottom_vector_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_indices == [2]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    return nothing
end

function test_split_indices_1d_4proc()
    nelement_list = [2]
    periodic_list = [false]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # -----===-----
    # 1:2 | 3 | 4:5
    # -----===-----
    nrank = 4
    n_shared = 4
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4:5)
            @test li[1].local_top_vector_indices == vcat(1:2, 4:5)
            @test li[1].all_local_top_vector_a_block_indices == 1:2
            @test li[1].local_top_vector_a_block_indices == [[1, 2]]
            @test li[1].all_a_block_sub_selection_indices == 1:2
            @test li[1].a_block_sub_selection_indices == [[1, 2]]
            @test li[1].a_block_lu_selection_indices == [[1, 2]]
            @test li[1].a_block_B_column_indices == [[1]]
            @test li[1].bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:1
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4:5)
            @test li[1].local_top_vector_indices == vcat(1:2, 4:5)
            @test li[1].all_local_top_vector_a_block_indices == 4:5
            @test li[1].local_top_vector_a_block_indices == [[4, 5]]
            @test li[1].all_a_block_sub_selection_indices == 3:4
            @test li[1].a_block_sub_selection_indices == [[3, 4]]
            @test li[1].a_block_lu_selection_indices == [[1, 2]]
            @test li[1].a_block_B_column_indices == [[1]]
            @test li[1].bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:1
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 2
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4:5)
            @test li[1].local_top_vector_indices == vcat(1:2, 4:5)
            @test li[1].all_local_top_vector_a_block_indices == 1:0
            @test li[1].local_top_vector_a_block_indices == []
            @test li[1].all_a_block_sub_selection_indices == 1:0
            @test li[1].a_block_sub_selection_indices == []
            @test li[1].a_block_lu_selection_indices == []
            @test li[1].a_block_B_column_indices == []
            @test li[1].bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:1
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end

        irank = 3
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:2, 4:5)
            @test li[1].local_top_vector_indices == vcat(1:2, 4:5)
            @test li[1].all_local_top_vector_a_block_indices == 1:0
            @test li[1].local_top_vector_a_block_indices == []
            @test li[1].all_a_block_sub_selection_indices == 1:0
            @test li[1].a_block_sub_selection_indices == []
            @test li[1].a_block_lu_selection_indices == []
            @test li[1].a_block_B_column_indices == []
            @test li[1].bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_indices == [3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:1
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [1]
            @test li[2].all_local_top_vector_a_block_indices == [1]
            @test li[2].local_top_vector_a_block_indices == [[1]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
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
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4]
            @test li[1].local_top_vector_indices == [2, 4]
            @test li[1].all_local_top_vector_a_block_indices == [2, 4]
            @test li[1].local_top_vector_a_block_indices == [[2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:2
            @test li[1].a_block_sub_selection_indices == [[1], [2]]
            @test li[1].a_block_lu_selection_indices == [[1], [2]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2, 3]]
            @test li[1].bottom_vector_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1, 2]]
            @test li[2].bottom_vector_indices == [1, 5]
            @test li[2].local_bottom_vector_indices == [1, 3]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [-1; -1;;]
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [6, 8]
            @test li[1].local_top_vector_indices == [2, 4]
            @test li[1].all_local_top_vector_a_block_indices == [2, 4]
            @test li[1].local_top_vector_a_block_indices == [[2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:2
            @test li[1].a_block_sub_selection_indices == [[1], [2]]
            @test li[1].a_block_lu_selection_indices == [[1], [2]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2, 3]]
            @test li[1].bottom_vector_indices == [5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[1].local_bottom_vector_repeat_indices == [3]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [7]
            @test li[2].local_top_vector_indices == [2]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1]]
            @test li[2].bottom_vector_indices == [5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3]
            @test li[2].local_bottom_vector_no_overlap_indices == [1]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [-1; -1;;]
        end
    end

    nrank = 2
    n_shared = 2
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_indices == [2, 4, 6, 8]
            @test li[1].all_local_top_vector_a_block_indices == [2, 4]
            @test li[1].local_top_vector_a_block_indices == [[2], [4]]
            @test li[1].all_a_block_sub_selection_indices == 1:2
            @test li[1].a_block_sub_selection_indices == [[1], [2]]
            @test li[1].a_block_lu_selection_indices == [[1], [2]]
            @test li[1].a_block_B_column_indices == [[1, 2], [2, 3]]
            @test li[1].bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:4
            @test li[1].local_bottom_vector_repeat_indices == [5]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [2, 4]
            @test li[2].all_local_top_vector_a_block_indices == [2]
            @test li[2].local_top_vector_a_block_indices == [[2]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1, 2]]
            @test li[2].bottom_vector_indices == [1, 5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3, 5]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [1; 5;;]
        end

        irank = 1
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid, nelement_list, block_sizes_list, periodic_list,
                                   remove_boundaries_list, [nrank÷n_shared],
                                   [irank÷n_shared], n_shared, irank)
            @test li[1].top_vector_indices == [2, 4, 6, 8]
            @test li[1].local_top_vector_indices == [2, 4, 6, 8]
            @test li[1].all_local_top_vector_a_block_indices == [6, 8]
            @test li[1].local_top_vector_a_block_indices == [[6], [8]]
            @test li[1].all_a_block_sub_selection_indices == 3:4
            @test li[1].a_block_sub_selection_indices == [[3], [4]]
            @test li[1].a_block_lu_selection_indices == [[1], [2]]
            @test li[1].a_block_B_column_indices == [[3, 4], [4, 5]]
            @test li[1].bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_indices == [1, 3, 5, 7, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 3, 5, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:4
            @test li[1].local_bottom_vector_repeat_indices == [5]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3, 7]
            @test li[2].local_top_vector_indices == [2, 4]
            @test li[2].all_local_top_vector_a_block_indices == [4]
            @test li[2].local_top_vector_a_block_indices == [[4]]
            @test li[2].all_a_block_sub_selection_indices == [2]
            @test li[2].a_block_sub_selection_indices == [[2]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[2]]
            @test li[2].bottom_vector_indices == [1, 5, 1]
            @test li[2].local_bottom_vector_indices == [1, 3, 5]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == 1:2
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [1; 5;;]
        end
    end

    return nothing
end

function test_split_indices_2d_1proc()
    nelement_list = [2, 1]
    ngrid_list = [3, 3]
    periodic_list = [false, false]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # -----------
    # 1, 6,  11
    # -----------
    # 2, 7,  12
    # ===========
    # 3, 8,  13
    # ===========
    # 4, 9,  14
    # -----------
    # 5, 10, 15
    # -----------
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1], [2, 1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15]
            @test li[1].local_top_vector_indices == [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15]
            @test li[1].all_local_top_vector_a_block_indices == [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15]
            @test li[1].local_top_vector_a_block_indices == [[1, 2, 6, 7, 11, 12], [4, 5, 9, 10, 14, 15]]
            @test li[1].all_a_block_sub_selection_indices == 1:12
            @test li[1].a_block_sub_selection_indices == [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
            @test li[1].a_block_lu_selection_indices == [[1, 2, 5, 6, 9, 10], [3, 4, 7, 8, 11, 12]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3], [1, 2, 3]]
            @test li[1].bottom_vector_indices == [3, 8, 13]
            @test li[1].local_bottom_vector_indices == [3, 8, 13]
            @test li[1].local_bottom_vector_no_overlap_indices == [3, 8, 13]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [3, 8, 13]
            @test li[2].local_top_vector_indices == 1:3
            @test li[2].all_local_top_vector_a_block_indices == 1:3
            @test li[2].local_top_vector_a_block_indices == [[1, 2, 3]]
            @test li[2].all_a_block_sub_selection_indices == 1:3
            @test li[2].a_block_sub_selection_indices == [[1, 2, 3]]
            @test li[2].a_block_lu_selection_indices == [[1, 2, 3]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    nelement_list = [1, 2]
    ngrid_list = [3, 3]
    periodic_list = [false, false]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # -----------------
    # 1, 4 ∥ 7 ∥ 10, 13
    # -----------------
    # 2, 5 ∥ 8 ∥ 11, 14
    # -----------------
    # 3, 6 ∥ 9 ∥ 12, 15
    # -----------------
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1], [1, 2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == vcat(1:6, 10:15)
            @test li[1].local_top_vector_indices == vcat(1:6, 10:15)
            @test li[1].all_local_top_vector_a_block_indices == vcat(1:6, 10:15)
            @test li[1].local_top_vector_a_block_indices == [1:6, 10:15]
            @test li[1].all_a_block_sub_selection_indices == 1:12
            @test li[1].a_block_sub_selection_indices == [1:6, 7:12]
            @test li[1].a_block_lu_selection_indices == [1:6, 7:12]
            @test li[1].a_block_B_column_indices == [[1, 2, 3], [1, 2, 3]]
            @test li[1].bottom_vector_indices == 7:9
            @test li[1].local_bottom_vector_indices == 7:9
            @test li[1].local_bottom_vector_no_overlap_indices == 7:9
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == 7:9
            @test li[2].local_top_vector_indices == 1:3
            @test li[2].all_local_top_vector_a_block_indices == 1:3
            @test li[2].local_top_vector_a_block_indices == [[1, 2, 3]]
            @test li[2].all_a_block_sub_selection_indices == 1:3
            @test li[2].a_block_sub_selection_indices == [[1, 2, 3]]
            @test li[2].a_block_lu_selection_indices == [[1, 2, 3]]
            @test li[2].a_block_B_column_indices == [[]]
            @test li[2].bottom_vector_indices == []
            @test li[2].local_bottom_vector_indices == []
            @test li[2].local_bottom_vector_no_overlap_indices == []
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == []
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    return nothing
end

function test_split_indices_2d_1proc_remove_boundaries()
    nelement_list = [1, 1]
    ngrid_list = [3, 3]
    periodic_list = [false, false]
    remove_boundaries_list = [true, true]

    # The interiors and boundaries are:
    # =============
    # ∥ 1 | 4 | 7 ∥
    # =---=====---=
    # ∥ 2 ∥ 5 ∥ 8 ∥
    # =---=====---=
    # ∥ 3 | 6 | 9 ∥
    # =============
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [5]
            @test li[1].local_top_vector_indices == [5]
            @test li[1].all_local_top_vector_a_block_indices == [5]
            @test li[1].local_top_vector_a_block_indices == [[5]]
            @test li[1].all_a_block_sub_selection_indices == [1]
            @test li[1].a_block_sub_selection_indices == [[1]]
            @test li[1].a_block_lu_selection_indices == [[1]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3, 4, 5, 6, 7, 8]]
            @test li[1].bottom_vector_indices == [1, 2, 3, 4, 6, 7, 8, 9]
            @test li[1].local_bottom_vector_indices == [1, 2, 3, 4, 6, 7, 8, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 2, 3, 4, 6, 7, 8, 9]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:8
            @test li[1].local_bottom_vector_repeat_indices == []
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    return nothing
end

function test_split_indices_2d_1proc_periodic()
    nelement_list = [1, 1]
    ngrid_list = [3, 3]
    periodic_list = [true, false]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # =============
    # ∥ 1 | 4 | 7 ∥
    # =============
    # | 2 | 5 | 8 |
    # =============
    # ∥ 3 | 6 | 9 ∥
    # =============
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [2, 5, 8]
            @test li[1].local_top_vector_indices == [2, 5, 8]
            @test li[1].all_local_top_vector_a_block_indices == [2, 5, 8]
            @test li[1].local_top_vector_a_block_indices == [[2, 5, 8]]
            @test li[1].all_a_block_sub_selection_indices == [1, 2, 3]
            @test li[1].a_block_sub_selection_indices == [[1, 2, 3]]
            @test li[1].a_block_lu_selection_indices == [[1, 2, 3]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3]]
            @test li[1].bottom_vector_indices == [1, 1, 4, 4, 7, 7]
            @test li[1].local_bottom_vector_indices == [1, 3, 4, 6, 7, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 4, 7]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == [1, 3, 5]
            @test li[1].local_bottom_vector_repeat_indices == [2, 4, 6]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    periodic_list = [false, true]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # ====-----====
    # ∥ 1 ∥ 4 ∥ 7 ∥
    # -------------
    # ∥ 2 ∥ 5 ∥ 8 ∥
    # -------------
    # ∥ 3 ∥ 6 ∥ 9 ∥
    # ====-----====
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [4, 5, 6]
            @test li[1].local_top_vector_indices == [4, 5, 6]
            @test li[1].all_local_top_vector_a_block_indices == [4, 5, 6]
            @test li[1].local_top_vector_a_block_indices == [[4, 5, 6]]
            @test li[1].all_a_block_sub_selection_indices == [1, 2, 3]
            @test li[1].a_block_sub_selection_indices == [[1, 2, 3]]
            @test li[1].a_block_lu_selection_indices == [[1, 2, 3]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3]]
            @test li[1].bottom_vector_indices == [1, 2, 3, 1, 2, 3]
            @test li[1].local_bottom_vector_indices == [1, 2, 3, 7, 8, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 2, 3]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == 1:3
            @test li[1].local_bottom_vector_repeat_indices == [4, 5, 6]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    periodic_list = [true, true]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # =============
    # ∥ 1 | 4 | 7 ∥
    # =---=====---=
    # ∥ 2 ∥ 5 ∥ 8 ∥
    # =---=====---=
    # ∥ 3 | 6 | 9 ∥
    # =============
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [5]
            @test li[1].local_top_vector_indices == [5]
            @test li[1].all_local_top_vector_a_block_indices == [5]
            @test li[1].local_top_vector_a_block_indices == [[5]]
            @test li[1].all_a_block_sub_selection_indices == [1]
            @test li[1].a_block_sub_selection_indices == [[1]]
            @test li[1].a_block_lu_selection_indices == [[1]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3]]
            @test li[1].bottom_vector_indices == [1, 2, 1, 4, 4, 1, 2, 1]
            @test li[1].local_bottom_vector_indices == [1, 2, 3, 4, 6, 7, 8, 9]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 2, 4]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == [1, 2, 4]
            @test li[1].local_bottom_vector_repeat_indices == [3, 5, 6, 7, 8]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
        end
    end

    nelement_list = [1, 2]
    ngrid_list = [3, 3]
    periodic_list = [true, false]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # =======================
    # ∥ 1 | 4 | 7 | 10 | 13 ∥
    # =========---===========
    # | 2 | 5 ∥ 8 ∥ 11 | 14 |
    # =========---===========
    # ∥ 1 | 4 | 7 | 10 | 13 ∥
    # =======================
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1], [1, 2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [2, 5, 11, 14]
            @test li[1].local_top_vector_indices == [2, 5, 11, 14]
            @test li[1].all_local_top_vector_a_block_indices == [2, 5, 11, 14]
            @test li[1].local_top_vector_a_block_indices == [[2, 5], [11, 14]]
            @test li[1].all_a_block_sub_selection_indices == [1, 2, 3, 4]
            @test li[1].a_block_sub_selection_indices == [[1, 2], [3, 4]]
            @test li[1].a_block_lu_selection_indices == [[1, 2], [3, 4]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 10, 11]]
            @test li[1].bottom_vector_indices == [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15]
            @test li[1].local_bottom_vector_indices == [1, 3, 4, 6, 7, 8, 9, 10, 12, 13, 15]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 4, 7, 8, 10, 13]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == [1, 3, 5, 6, 8, 10]
            @test li[1].local_bottom_vector_repeat_indices == [2, 4, 7, 9, 11]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [8]
            @test li[2].local_top_vector_indices == [6]
            @test li[2].all_local_top_vector_a_block_indices == [6]
            @test li[2].local_top_vector_a_block_indices == [[6]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1, 2, 3, 4, 5]]
            @test li[2].bottom_vector_indices == [1, 1, 4, 4, 7, 7, 10, 10, 13, 13]
            @test li[2].local_bottom_vector_indices == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 3, 5, 8, 10]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1, 3, 5, 7, 9]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [1 3 5 7 9; 2 4 7 9 11]
        end
    end

    nelement_list = [1, 2]
    ngrid_list = [3, 3]
    periodic_list = [true, true]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # ======================
    # ∥ 1 | 4 | 7 | 10 | 1 ∥
    # =---=====---======---=
    # | 2 ∥ 5 ∥ 8 ∥ 11 ∥ 2 |
    # =---=====---======---=
    # ∥ 1 | 4 | 7 | 10 | 1 ∥
    # ======================
    nrank = 1
    n_shared = 1
    block_sizes_list = [[1, 1], [1, 2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list, nrank=$nrank, n_shared=$n_shared" begin
        irank = 0
        @testset "irank=$irank" begin
            li, _ = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                   periodic_list, remove_boundaries_list,
                                   [1, 1], [0, 0], n_shared, irank)
            @test li[1].top_vector_indices == [5, 11]
            @test li[1].local_top_vector_indices == [5, 11]
            @test li[1].all_local_top_vector_a_block_indices == [5, 11]
            @test li[1].local_top_vector_a_block_indices == [[5], [11]]
            @test li[1].all_a_block_sub_selection_indices == [1, 2]
            @test li[1].a_block_sub_selection_indices == [[1], [2]]
            @test li[1].a_block_lu_selection_indices == [[1], [2]]
            @test li[1].a_block_B_column_indices == [[1, 2, 3, 4, 5, 6, 7, 8], [6, 7, 8, 9, 10, 11, 12, 13]]
            @test li[1].bottom_vector_indices == [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15]
            @test li[1].local_bottom_vector_indices == [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15]
            @test li[1].local_bottom_vector_no_overlap_indices == [1, 2, 4, 7, 8, 10]
            @test li[1].local_bottom_vector_no_overlap_sub_selection_indices == [1, 2, 4, 6, 7, 9]
            @test li[1].local_bottom_vector_repeat_indices == [3, 5, 8, 10, 11, 12, 13]
            @test li[1].local_bottom_vector_periodic_pairs == zeros(Int64, 2, 0)
            @test li[2].top_vector_indices == [8]
            @test li[2].local_top_vector_indices == [7]
            @test li[2].all_local_top_vector_a_block_indices == [7]
            @test li[2].local_top_vector_a_block_indices == [[7]]
            @test li[2].all_a_block_sub_selection_indices == [1]
            @test li[2].a_block_sub_selection_indices == [[1]]
            @test li[2].a_block_lu_selection_indices == [[1]]
            @test li[2].a_block_B_column_indices == [[1, 2, 3, 4, 5]]
            @test li[2].bottom_vector_indices == [1, 2, 1, 4, 4, 7, 7, 10, 10, 1, 2, 1]
            @test li[2].local_bottom_vector_indices == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
            @test li[2].local_bottom_vector_no_overlap_indices == [1, 2, 4, 6, 9]
            @test li[2].local_bottom_vector_no_overlap_sub_selection_indices == [1, 2, 4, 6, 8]
            @test li[2].local_bottom_vector_repeat_indices == []
            @test li[2].local_bottom_vector_periodic_pairs == [1 1 1 2 4 6 8; 3 11 13 12 5 8 10]
        end
    end

    return nothing
end

function test_get_shared_sparse_matrix_csc_buffer_1d()
    allocate_shared_float = (args...; kwargs...) -> zeros(args...)
    allocate_shared_int = (args...; kwargs...) -> zeros(Int64, args...)

    # The logic in `get_shared_sparse_matrix_csc_buffer()` is only on shared_comm_rank=0,
    # so no need to test with different n_shared.
    shared_comm = FakeComm(0, 1)

    periodic_list = [false]
    remove_boundaries_list = [false]

    # The interiors and boundaries are:
    # -----===-----
    # 1:2 | 3 | 4:5
    # -----===-----
    nelement_list = [2]
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:4
            @test buffer.colptr == [1, 5]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1]
            @test buffer.colptr == [1, 2]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]
        end
    end

    # The interiors and boundaries are:
    # -----===-----===-----
    # 1:2 | 3 | 4 | 5 | 6:7
    # -----===-----===-----
    nelement_list = [3]
    block_sizes_list = [[1], [2], [3]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 3:5)
            @test buffer.colptr == [1, 4, 7]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:2)
            @test buffer.colptr == [1, 3, 5]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1]
            @test buffer.colptr == [1, 2]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1]
            @test buffer.colptr == [1, 2]
        end

        level = 3
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]
        end
    end

    # The interiors and boundaries are:
    # -----===-----===-----===-----
    # 1:2 | 3 | 4 | 5 | 6 | 7 | 8:9
    # -----===-----===-----===-----
    nelement_list = [4]
    block_sizes_list = [[1], [2], [4]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 3:4, 4:6)
            @test buffer.colptr == [1, 4, 6, 9]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:3)
            @test buffer.colptr == [1, 3, 6, 8]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:2
            @test buffer.colptr == [1, 3]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1]
            @test buffer.colptr == [1, 2]
        end

        level = 3
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]
        end
    end

    return nothing
end

function test_get_shared_sparse_matrix_csc_buffer_1d_remove_boundaries()
    allocate_shared_float = (args...; kwargs...) -> zeros(args...)
    allocate_shared_int = (args...; kwargs...) -> zeros(Int64, args...)

    # The logic in `get_shared_sparse_matrix_csc_buffer()` is only on shared_comm_rank=0,
    # so no need to test with different n_shared.
    shared_comm = FakeComm(0, 1)

    periodic_list = [false]
    remove_boundaries_list = [true]

    # The interiors and boundaries are:
    # ==-----===-----==
    # 1 | 2 | 3 | 4 | 5
    # ==-----===-----==
    nelement_list = [2]
    block_sizes_list = [[1], [2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1, 1:2, 2)
            @test buffer.colptr == [1, 2, 4, 5]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:3)
            @test buffer.colptr == [1, 3, 6, 8]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1, 1]
            @test buffer.colptr == [1, 2, 3]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:2)
            @test buffer.colptr == [1, 3, 5]
        end
    end

    # The interiors and boundaries are:
    # ==-----===-----===-----==
    # 1 | 2 | 3 | 4 | 5 | 6 | 7
    # ==-----===-----===-----==
    nelement_list = [3]
    block_sizes_list = [[1], [2], [3]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1, 1:2, 2:3, 3)
            @test buffer.colptr == [1, 2, 4, 6, 7]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:4, 3:4)
            @test buffer.colptr == [1, 3, 6, 9, 11]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1, 1]
            @test buffer.colptr == [1, 2, 3, 3]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:3)
            @test buffer.colptr == [1, 3, 6, 8]
        end

        level = 3
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1, 1]
            @test buffer.colptr == [1, 2, 3]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:2)
            @test buffer.colptr == [1, 3, 5]
        end
    end

    # The interiors and boundaries are:
    # ==-----===-----===-----===-----==
    # 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    # ==-----===-----===-----===-----==
    nelement_list = [4]
    block_sizes_list = [[1], [2], [4]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1], [0],
                                        1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1, 1:2, 2:3, 3:4, 4)
            @test buffer.colptr == [1, 2, 4, 6, 8, 9]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:4, 3:5, 4:5)
            @test buffer.colptr == [1, 3, 6, 9, 12, 14]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1, 1:2, 2)
            @test buffer.colptr == [1, 2, 4, 5]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:3, 2:3)
            @test buffer.colptr == [1, 3, 6, 8]
        end

        level = 3
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == [1, 1]
            @test buffer.colptr == [1, 2, 3]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:2, 1:2)
            @test buffer.colptr == [1, 3, 5]
        end
    end

    return nothing
end

function test_get_shared_sparse_matrix_csc_buffer_2d()
    allocate_shared_float = (args...; kwargs...) -> zeros(args...)
    allocate_shared_int = (args...; kwargs...) -> zeros(Int64, args...)

    # The logic in `get_shared_sparse_matrix_csc_buffer()` is only on shared_comm_rank=0,
    # so no need to test with different n_shared.
    shared_comm = FakeComm(0, 1)

    periodic_list = [false, false]
    remove_boundaries_list = [false, false]

    # The interiors and boundaries are:
    # -----------------
    # 1 6  ∥ 11 ∥ 16 21
    # -----------------
    # 2 7  ∥ 12 ∥ 17 22
    # =================
    # 3 8  ∥ 13 ∥ 18 23
    # =================
    # 4 9  ∥ 14 ∥ 19 24
    # -----------------
    # 5 10 ∥ 15 ∥ 20 25
    # -----------------
    nelement_list = [2, 2]
    ngrid_list = [ngrid, ngrid]
    block_sizes_list = [[1, 1], [2, 2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1, 1],
                                        [0, 0], 1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:8, 1:8, 1:2, 5:6, 9:10, 13:14, 1:2, 5:6, 9:10, 13:14, 1:16, 3:4, 7:8, 11:12, 15:16, 3:4, 7:8, 11:12, 15:16, 9:16, 9:16)
            @test buffer.colptr == [1, 9, 17, 25, 33, 49, 57, 65, 73, 81]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:7, 1:7, 1:5, 8:9, 1:5, 8:9, 1:9, 1:2, 5:9, 1:2, 5:9, 3:9, 3:9)
            @test buffer.colptr == [1, 8, 15, 22, 29, 38, 45, 52, 59, 66]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == 1:0
            @test buffer.colptr == [1]
        end
    end

    periodic_list = [false, false]
    remove_boundaries_list = [true, true]

    # The interiors and boundaries are:
    # =========================
    # ∥ 1 ∥ 6  ∥ 11 ∥ 16 ∥ 21 ∥
    # =========================
    # ∥ 2 ∥ 7  ∥ 12 ∥ 17 ∥ 22 ∥
    # =========================
    # ∥ 3 ∥ 8  ∥ 13 ∥ 18 ∥ 23 ∥
    # =========================
    # ∥ 4 ∥ 9  ∥ 14 ∥ 19 ∥ 24 ∥
    # =========================
    # ∥ 5 ∥ 10 ∥ 15 ∥ 20 ∥ 25 ∥
    # =========================
    nelement_list = [2, 2]
    block_sizes_list = [[1, 1], [2, 2]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1, 1],
                                        [0, 0], 1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1, 1, 1:2, 2, 2, 1, 1:2, 2, 1, 3, 1, 3, 1:4, 2, 4, 2, 4, 3, 3:4, 4, 3, 3, 3:4, 4, 4)
            @test buffer.colptr == [1, 2, 3, 5, 6, 7, 8, 10, 11, 13, 15, 19, 21, 23, 24, 26, 27, 28, 29, 31, 32, 33]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 6:7, 9:11,
                                        1:3, 6:7, 9:11,
                                        1:13,
                                        3:5, 7:8, 11:13,
                                        3:5, 7:8, 11:13,
                                        1:3, 6:7, 9:11,
                                        1:13,
                                        3:5, 7:8, 11:13,
                                        1:3, 6:7, 9:11, 14:15, 17:19,
                                        1:3, 6:7, 9:11, 14:15, 17:19,
                                        1:21,
                                        3:5, 7:8, 11:13, 15:16, 19:21,
                                        3:5, 7:8, 11:13, 15:16, 19:21,
                                        9:11, 14:15, 17:19,
                                        9:21,
                                        11:13, 15:16, 19:21,
                                        9:11, 14:15, 17:19,
                                        9:11, 14:15, 17:19,
                                        9:21,
                                        11:13, 15:16, 19:21,
                                        11:13, 15:16, 19:21)
            @test buffer.colptr == [1, 9, 17, 30, 38, 46, 54, 67, 75, 88, 101, 122, 135, 148, 156, 169, 177, 185, 193, 206, 214, 222]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5, 1:5)
            @test buffer.colptr == [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16)
            @test buffer.colptr == [1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177, 193, 209, 225, 241, 257]
        end
    end

    periodic_list = [false, false]
    remove_boundaries_list = [true, true]

    # The interiors and boundaries are:
    # ===================================
    # ∥ 1 ∥ 6  ∥ 11 ∥ 16 ∥ 21 ∥ 26 ∥ 31 ∥
    # ===================================
    # ∥ 2 ∥ 7  ∥ 12 ∥ 17 ∥ 22 ∥ 27 ∥ 32 ∥
    # -===------====------====------====-
    # ∥ 3 ∥ 8  ∥ 13 ∥ 18 ∥ 23 ∥ 28 ∥ 33 ∥
    # -===------====------====------====-
    # ∥ 4 ∥ 9  ∥ 14 ∥ 19 ∥ 24 ∥ 29 ∥ 34 ∥
    # ===================================
    # ∥ 5 ∥ 10 ∥ 15 ∥ 20 ∥ 25 ∥ 30 ∥ 35 ∥
    # ===================================
    nelement_list = [1, 3]
    ngrid_list = [5, 3]
    block_sizes_list = [[1, 1], [1, 2], [1, 3]]
    @testset "nelement_list=$nelement_list, block_sizes_list=$block_sizes_list, periodic_list=$periodic_list, remove_boundaries_list=$remove_boundaries_list" begin
        li, dimensions = get_level_info(ngrid_list, nelement_list, block_sizes_list,
                                        periodic_list, remove_boundaries_list, [1, 1],
                                        [0, 0], 1, 0)
        level = 1
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3,
                                        1:6, 1:6, 1:6, 1:6, 1:6,
                                        4:6, 4:6,
                                        4:9, 4:9, 4:9, 4:9, 4:9,
                                        7:9, 7:9, 7:9, 7:9, 7:9, 7:9, 7:9)
            @test buffer.colptr == [1, 4, 7, 10, 13, 16, 19, 22,
                                    28, 34, 40, 46, 52,
                                    55, 58,
                                    64, 70, 76, 82, 88,
                                    91, 94, 97, 100, 103, 106, 109]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:12, 1:12, 1:12, 1:12, 1:12, 1:12, 1:12,
                                        1:19, 1:19, 1:19, 1:19, 1:19,
                                        8:19, 8:19,
                                        8:26, 8:26, 8:26, 8:26, 8:26,
                                        15:26, 15:26, 15:26, 15:26, 15:26, 15:26, 15:26)
            @test buffer.colptr == [1, 13, 25, 37, 49, 61, 73, 85,
                                    104, 123, 142, 161, 180,
                                    192, 204,
                                    223, 242, 261, 280, 299,
                                    311, 323, 335, 347, 359, 371, 383]
        end

        level = 2
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3)
            @test buffer.colptr == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 49, 49, 49, 49, 49, 49, 49]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16, 1:16,
                                        1:23, 1:23, 1:23, 1:23, 1:23,
                                        12:23, 12:23, 12:23, 12:23, 12:23, 12:23, 12:23)
            @test buffer.colptr == [1, 17, 33, 49, 65, 81, 97, 113, 129, 145, 161, 177,
                                    200, 223, 246, 269, 292,
                                    304, 316, 328, 340, 352, 364, 376]
        end

        level = 3
        @testset "level=$level" begin
            block_sizes = block_sizes_list[level]
            level_info = li[level]
            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.top_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3, 1:3)
            @test buffer.colptr == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61]

            buffer =
                get_shared_sparse_matrix_csc_buffer(dimensions, block_sizes,
                                                    level_info.bottom_vector_indices,
                                                    level_info.bottom_vector_indices,
                                                    shared_comm, allocate_shared_float,
                                                    allocate_shared_int)
            @test buffer.rowval == vcat(1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20, 1:20)
            @test buffer.colptr == [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261, 281, 301, 321, 341, 361, 381, 401]
        end
    end

    return nothing
end

function test_indices()
    @testset "Test index splitting" begin
        @testset "test_split_indices_1d_1proc_remove_boundaries" test_split_indices_1d_1proc_remove_boundaries()
        @testset "test_split_indices_1d_1proc_periodic" test_split_indices_1d_1proc_periodic()
        @testset "test_split_indices_1d_2proc" test_split_indices_1d_2proc()
        @testset "test_split_indices_1d_4proc" test_split_indices_1d_4proc()
        @testset "test_split_indices_1d_2proc_periodic" test_split_indices_1d_2proc_periodic()
        @testset "test_split_indices_2d_1proc" test_split_indices_2d_1proc()
        @testset "test_split_indices_2d_1proc_remove_boundaries" test_split_indices_2d_1proc_remove_boundaries()
        @testset "test_split_indices_2d_1proc_periodic" test_split_indices_2d_1proc_periodic()
        @testset "test_get_shared_sparse_matrix_csc_buffer_1d" test_get_shared_sparse_matrix_csc_buffer_1d()
        @testset "test_get_shared_sparse_matrix_csc_buffer_1d_remove_boundaries" test_get_shared_sparse_matrix_csc_buffer_1d_remove_boundaries()
        @testset "test_get_shared_sparse_matrix_csc_buffer_2d" test_get_shared_sparse_matrix_csc_buffer_2d()
    end
end
