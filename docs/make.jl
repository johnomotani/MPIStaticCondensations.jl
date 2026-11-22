using Pkg

Pkg.instantiate()

using Documenter
using MPIStaticCondensations

makedocs(
    sitename = "MPIStaticCondensations",
    format = Documenter.HTML(),
    modules = [MPIStaticCondensations]
)

if get(ENV, "CI", nothing) == "true"
    # Documenter can also automatically deploy documentation to gh-pages.
    # See "Hosting Documentation" and deploydocs() in the Documenter manual
    # for more information.
    deploydocs(
        repo = "github.com/johnomotani/MPIStaticCondensations.jl",
        push_preview = true,
    )
end
