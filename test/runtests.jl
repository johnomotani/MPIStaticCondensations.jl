println("using LinearAlgebra")
using LinearAlgebra
println("using MPIStaticCondensations")
using MPIStaticCondensations
println("using Test")
using Test

println("BLAS.set_num_threads(1)")
BLAS.set_num_threads(1)

println("include(\"generate_finite_element_matrices.jl\")")
include("generate_finite_element_matrices.jl")
println("include(\"utils.jl\")")
include("utils.jl")

println("include(\"test_indices.jl\")")
include("test_indices.jl")
println("include(\"test_finite_element_matrices.jl\")")
include("test_finite_element_matrices.jl")
println("include(\"test_multivariable_finite_element_matrices.jl\")")
include("test_multivariable_finite_element_matrices.jl")

function runtests()
println("runtests")
    @testset "MPIStaticCondensations.jl" failfast=true begin
println("bfore test_indices")
        test_indices()
println("before test_finite_element_matrices")
        test_finite_element_matrices()
println("before test_multivariable_finite_element_matrices")
        test_multivariable_finite_element_matrices()
    end
end

println("runtests()")
runtests()
