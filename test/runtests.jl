using Test

function runtests()
    @testset "MPIStaticCondensations.jl" failfast=true begin
        include("test_indices.jl")
        include("test_finite_element_matrices.jl")
        include("test_multivariable_finite_element_matrices.jl")
    end
end

runtests()
