using CSV
using GLMakie

include("plot-functions.jl")

const multivariable_results_directory = "results-multivariable-benchmark"

plot_multivariable_comparison(args...; kwargs...) =
    plot_comparison(args...; results_directory=multivariable_results_directory, kwargs...)

for case ∈ cases
    plot_multivariable_comparison(case)
end
