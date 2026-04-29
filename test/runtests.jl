using MPIStaticCondensations
using Test

include("test_indices.jl")
include("test_finite_element_matrices.jl")

function runtests()
    @testset "MPIStaticCondensations.jl" begin
        test_indices()
        test_finite_element_matrices()
    end
end

runtests()
