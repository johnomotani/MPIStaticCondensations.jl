using MPIStaticCondensations
using MPIStaticCondensations: FakeComm, split_dimension
using Test

function get_level_info(ngrid, nelement_list, periodic_list, remove_boundaries_list,
                        nrank_list, irank_list, n_shared, n_groups, irank;
                        optimize_schur_complement_size=true)
    total_nrank = prod(nrank_list) * n_shared

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
                  for (nelement, periodic, remove_boundaries, nrank, dim_irank) ∈
                      zip(nelement_list, periodic_list, remove_boundaries_list,
                          nrank_list, irank_list)]

    level_info = split_dimension(dimensions, n_groups, optimize_schur_complement_size,
                                 comm, distributed_comm, shared_comm)

    return level_info
end

function test_split_indices_1d()
    ngrid = 3

    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [false]

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
            @test li.bottom_vector_indices == []
            @test li.local_bottom_vector_indices == []
            @test li.top_vector_indices == 1:3
            @test li.local_top_vector_indices == 1:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.local_top_vector_indices == 1:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.local_top_vector_indices == 2:3
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == []
            @test li.local_bottom_vector_indices == []
            @test li.top_vector_indices == 7:9
            @test li.local_top_vector_indices == 1:3
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
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == 1:4
            @test li.local_top_vector_indices == 1:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == 1:4
            @test li.local_top_vector_indices == 1:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:9
            @test li.local_top_vector_indices == 2:5
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:9
            @test li.local_top_vector_indices == 2:5
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
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_indices == 1:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_indices == 1:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_indices == 6:9
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
            @test li.local_top_vector_indices == 6:9
        end
    end

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
            @test li.local_top_vector_indices == 1:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.local_top_vector_indices == 2:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.local_top_vector_indices == 2:2
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 8:9
            @test li.local_top_vector_indices == 2:3
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
            @test li.local_top_vector_indices == 1:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.top_vector_indices == [1,2,4,6,8,9]
            @test li.local_top_vector_indices == 8:9
        end
    end

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
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [5,9]
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
            @test li.local_top_vector_indices == 1:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [5,9]
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [5,9]
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
            @test li.local_top_vector_indices == 1:0
        end
    end

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
            @test li.local_top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.local_top_vector_indices == 1:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.local_top_vector_indices == 2:3
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.local_top_vector_indices == 1:2
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end
    end

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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.local_top_vector_indices == 2:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.local_top_vector_indices == 2:2
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.local_top_vector_indices == 2:2
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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 8:8
        end
    end

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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 1:0
        end
    end

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
            @test li.local_top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.local_top_vector_indices == 1:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.local_top_vector_indices == 2:3
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [9]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.local_top_vector_indices == 1:2
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end
    end

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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.local_top_vector_indices == 2:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.local_top_vector_indices == 2:2
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,9]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.local_top_vector_indices == 2:2
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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 8:8
        end
    end

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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 1:0
        end
    end

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
            @test li.local_top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 3:4
            @test li.local_top_vector_indices == 1:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.top_vector_indices == 6:7
            @test li.local_top_vector_indices == 2:3
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.top_vector_indices == 7:8
            @test li.local_top_vector_indices == 1:2
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 2:4
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.top_vector_indices == 6:8
            @test li.local_top_vector_indices == 2:4
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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end
    end

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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 4:4
            @test li.local_top_vector_indices == 2:2
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 6:6
            @test li.local_top_vector_indices == 2:2
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.top_vector_indices == 8:8
            @test li.local_top_vector_indices == 2:2
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
            @test li.local_top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.top_vector_indices == [2,4,6,8]
            @test li.local_top_vector_indices == 8:8
        end
    end

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
            @test li.local_top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 6:8
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
            @test li.local_top_vector_indices == 1:0
        end
    end
end

function test_split_indices_3d()
    ngrid = 3

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
            @test li.local_top_vector_indices == 1:18
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
            @test li.local_top_vector_indices == 10:27
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
            @test li.local_top_vector_indices == 1:18
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
            @test li.local_top_vector_indices == 28:45
        end
    end

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
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
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
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
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
            @test li.local_top_vector_indices == vcat(1:6, 16:21, 31:36)
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
            @test li.local_top_vector_indices == vcat(10:15, 25:30, 40:45)
        end
    end

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
            @test li.local_top_vector_indices == vcat(1:2, 4:5, 7:8, 10:11, 13:14, 16:17, 19:20, 22:23, 25:26)
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
            @test li.local_top_vector_indices == vcat(2:3, 5:6, 8:9, 11:12, 14:15, 17:18, 20:21, 23:24, 26:27)
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
            @test li.local_top_vector_indices == vcat(1:2, 6:7, 11:12, 16:17, 21:22, 26:27, 31:32, 36:37, 41:42)
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
            @test li.local_top_vector_indices == vcat(4:5, 9:10, 14:15, 19:20, 24:25, 29:30, 34:35, 39:40, 44:45)
        end
    end
end

function runtests()
    @testset "MPIStaticCondensations.jl" begin
        #@testset "test_split_indices_1d" test_split_indices_1d()
        @testset "test_split_indices_3d" test_split_indices_3d()
    end
end

runtests()
