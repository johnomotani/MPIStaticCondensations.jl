include("common-multivariable.jl")
include("benchmark-MPIStaticCondensations-shared.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
