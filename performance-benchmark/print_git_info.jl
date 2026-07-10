using LibGit2
using MPIStaticCondensations
using MPISchurComplements

function print_git_info(io=nothing)
    if io === nothing
        io = stdout
    end

    for mod ∈ (MPIStaticCondensations, MPISchurComplements)
        print_git_info_for_module(io, mod)
    end

    return nothing
end

function print_git_info_for_module(io, mod)
    modname = String(nameof(mod))

    project_dir = pkgdir(mod)
    repo = GitRepo(project_dir)
    git_commit_hash = string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))))
    if LibGit2.isdirty(repo)
        # Use a shell command to get the 'git diff' because it seems to be complicated (if
        # not impossible) to get this using LibGit2.
        # Use `setenv()` to run the command in `project_dir` without changing the current
        # working directory.
        # Use `read()` rather than `run()` so that the command returns the terminal
        # output.
        # Finally need to convert the output to String as `read()` returns a
        # Vector{UInt8}.
        git_diff = String(read(setenv(`git diff`; dir=project_dir)))
    else
        git_diff = ""
    end

    println(io, modname, ": ", git_commit_hash)
    println(io, "git diff:")
    println(io, git_diff)
    println(io)

    return nothing
end
