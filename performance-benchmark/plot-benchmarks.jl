using CSV
using GLMakie

include("plot-functions.jl")

for case ∈ cases
    plot_comparison(case)
end
