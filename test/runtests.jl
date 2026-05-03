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
            @test li.local_top_vector_indices == 1:3
            @test li.top_vector_indices == 1:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 3:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 6:7
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == []
            @test li.local_bottom_vector_indices == []
            @test li.local_top_vector_indices == 1:3
            @test li.top_vector_indices == 7:9
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
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == 1:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == 1:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:5
            @test li.top_vector_indices == 6:9
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:5
            @test li.top_vector_indices == 6:9
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
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 6:9
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 6:9
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end
    end

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
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 1:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 8:9
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
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 4:4
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 6:6
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 8:9
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end
    end

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
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [5,9]
            @test li.local_top_vector_indices == 1:0
            @test li.top_vector_indices == [(1:4)...,(6:8)...]
        end
    end

    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [false]

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
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 3:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 6:7
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 7:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end

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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 8:8
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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 4:4
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 6:6
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 8:8
            @test li.top_vector_indices == [2,4,6,8]
        end
    end

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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 1:0
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end

    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [true]

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
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 3:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 6:7
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [9]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 7:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,9]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end

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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,9]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 8:8
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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 4:4
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 6:6
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,9]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 8:8
            @test li.top_vector_indices == [2,4,6,8]
        end
    end

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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,9]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 1:0
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end

    nelement_list = [4]
    periodic_list = [true]
    remove_boundaries_list = [true]

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
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 2:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 3:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_bottom_vector_indices == [1]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 6:7
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1]
            @test li.local_bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 7:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 2:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,1]
            @test li.local_bottom_vector_indices == [1,5]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == 6:8
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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end

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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 2:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7,1]
            @test li.local_bottom_vector_indices == [1,3]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 8:8
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
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 4:4
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 6:6
            @test li.top_vector_indices == [2,4,6,8]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,3,5,7,1]
            @test li.local_bottom_vector_indices == [1,3,5,7,9]
            @test li.local_top_vector_indices == 8:8
            @test li.top_vector_indices == [2,4,6,8]
        end
    end

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
            @test li.local_top_vector_indices == 2:4
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 6:8
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, [nrank÷n_shared],
                                [irank÷n_shared], n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [1,5,1]
            @test li.local_bottom_vector_indices == [1,5,9]
            @test li.local_top_vector_indices == 1:0
            @test li.top_vector_indices == [(2:4)...,(6:8)...]
        end
    end
end

function test_split_indices_3d()
    ngrid = 3

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
            @test li.local_top_vector_indices == 1:18
            @test li.top_vector_indices == 1:18
        end

        irank = 1
        irank_list = [0, 0, 1]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 1:9
            @test li.local_top_vector_indices == 10:27
            @test li.top_vector_indices == 28:45
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
            @test li.local_top_vector_indices == 1:18
            @test li.top_vector_indices == [(1:18)...,(28:45)...]
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == 19:27
            @test li.local_bottom_vector_indices == 19:27
            @test li.local_top_vector_indices == 28:45
            @test li.top_vector_indices == [(1:18)...,(28:45)...]
        end
    end

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
            @test li.local_top_vector_indices == vcat(1:6, 10:15, 19:24)
            @test li.top_vector_indices == vcat(1:6, 16:21, 31:36)
        end

        irank = 1
        irank_list = [0, 1, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(1:3, 10:12, 19:21)
            @test li.local_top_vector_indices == vcat(4:9, 13:18, 22:27)
            @test li.top_vector_indices == vcat(10:15, 25:30, 40:45)
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
            @test li.local_top_vector_indices == vcat(1:6, 16:21, 31:36)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
        end

        irank = 1
        irank_list = [0, 0, 0]
        @testset "irank=$irank, irank_list=$irank_list" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank_list, irank_list, n_shared,
                                n_groups, irank)
            @test li.bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_bottom_vector_indices == vcat(7:9, 22:24, 37:39)
            @test li.local_top_vector_indices == vcat(10:15, 25:30, 40:45)
            @test li.top_vector_indices == vcat(1:6, 10:21, 25:36, 40:45)
        end
    end
end

function runtests()
    @testset "MPIStaticCondensations.jl" begin
        @testset "test_split_indices_1d" test_split_indices_1d()
        @testset "test_split_indices_3d" test_split_indices_3d()
    end
end

runtests()
