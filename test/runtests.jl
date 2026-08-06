using MPIStaticCondensations
using Test

include("test_indices.jl")
include("test_finite_element_matrices.jl")
include("test_multivariable_finite_element_matrices.jl")

function runtests()
    #@testset "MPIStaticCondensations.jl" begin
    @testset "MPIStaticCondensations.jl" failfast=true begin
        #test_indices()
        #test_finite_element_matrices()
        test_multivariable_finite_element_matrices()
    end
end

runtests()
