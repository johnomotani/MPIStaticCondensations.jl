using MPIStaticCondensations
using MPIStaticCondensations: FakeComm, split_dimension
using Test

function get_level_info(ngrid, nelement_list, periodic_list, remove_boundaries_list,
                        nrank, n_shared, n_groups, irank;
                        optimize_schur_complement_size=true)
    if nrank % n_shared != 0
        error("n_shared=$n_shared should divide nrank=$nrank")
    end

    comm = FakeComm(irank, nrank)
    shared_comm = FakeComm(irank % n_shared, n_shared)
    if shared_comm.rank == 0
        distributed_comm = FakeComm(irank ÷ n_shared, nrank ÷ n_shared)
    else
        distributed_comm = nothing
    end
    distributed_size = comm.size ÷ shared_comm.size
    distributed_rank = comm.rank ÷ shared_comm.size

    dimensions = [create_dimension(; nelement, ngrid, nrank=distributed_size,
                                   irank=distributed_rank, periodic,
                                   remove_boundaries)
                  for (nelement, periodic, remove_boundaries) ∈
                      zip(nelement_list, periodic_list, remove_boundaries_list)]

    level_info = split_dimension(dimensions, n_groups, optimize_schur_complement_size,
                                 comm, distributed_comm, shared_comm)

    return level_info
end

function test_split_indices()
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
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == []
            @test li.local_top_vector_indices == 1:3
            @test li.top_vector_indices == 1:3
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 3:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 2:3
            @test li.top_vector_indices == 6:7
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == []
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
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == 1:4
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == 1:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 2:5
            @test li.top_vector_indices == 6:9
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
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
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 1:4
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
            @test li.local_top_vector_indices == 6:9
            @test li.top_vector_indices == [(1:4)...,(6:9)...]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5]
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
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == 1:2
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 4:4
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [5,7]
            @test li.local_top_vector_indices == 2:2
            @test li.top_vector_indices == 6:6
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [7]
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
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 1:2
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 1
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 4:4
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 2
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 6:6
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end

        irank = 3
        @testset "irank=$irank" begin
            li = get_level_info(ngrid, nelement_list, periodic_list,
                                remove_boundaries_list, nrank, n_shared, n_groups, irank)
            @test li.bottom_vector_indices == [3,5,7]
            @test li.local_top_vector_indices == 8:9
            @test li.top_vector_indices == [1,2,4,6,8,9]
        end
    end
end

function runtests()
    @testset "MPIStaticCondensations.jl" begin
        @testset "test_split_indices" test_split_indices()
    end
end

runtests()
