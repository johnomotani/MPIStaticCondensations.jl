using MPIStaticCondensations
using MPIStaticCondensations: FakeComm, split_dimension
using Test

using Debugger
function test_split_indices(nrank, n_groups, optimize_schur_complement_size=true)
    n_shared = 2
    if nrank % n_shared != 0
        error("n_shared=$n_shared should divide nrank=$nrank")
    end
    ngrid = 3
    nelement_list = [4]
    periodic_list = [false]
    remove_boundaries_list = [false]

    for irank ∈ 0:nrank-1
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
#for (nelement, periodic, remove_boundaries) ∈
#  zip(nelement_list, periodic_list, remove_boundaries_list)
#  @enter create_dimension(; nelement, ngrid, nrank=distributed_size,
#                                       irank=distributed_rank, periodic,
#                                       remove_boundaries)
#end
println("dimensions=$dimensions")

        level_info = split_dimension(dimensions, n_groups, optimize_schur_complement_size,
                                     comm, distributed_comm, shared_comm)

        println("irank=$irank")
println("original global indices ", 1:dimensions[1].n)
        println(level_info)
        println()
#@enter split_dimension(dimensions, n_groups, optimize_schur_complement_size,
#                             comm, distributed_comm, shared_comm)
    end
end

function runtests()
    @testset "MPIStaticCondensations.jl" begin
        # Write your tests here.
    end
end

runtests()
