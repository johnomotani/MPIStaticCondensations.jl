using CSV
using GLMakie

const results_directory = "results-benchmark"
const solvers = ("UMFPACK", "MPIStaticCondensations", "MUMPS")
const cases = ("1d", "2d", "3d")

function load_data(filename)
    file = CSV.File(filename; delim=" ", header=false, comment="#")
    setup_dict = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
    lu_dict = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
    solve_dict = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
    sizes_set = String[]
    sizes_dict = Dict{String,String}()
    for row in file
        nproc, n_shared, ndim, total_size, level_multiplier, tsetup, tlu, tsolve,
            nelement_list, ngrid_list, periodic_list, remove_boundaries_list = row

        key = join([nelement_list, ngrid_list], ",")
        n_shared_blocks = nproc ÷ n_shared
        if key ∉ keys(setup_dict)
            setup_dict[key] = Dict{Int64,Dict{Int64,Dict{Int64,Float64}}}()
            lu_dict[key] = Dict{Int64,Dict{Int64,Dict{Int64,Float64}}}()
            solve_dict[key] = Dict{Int64,Dict{Int64,Dict{Int64,Float64}}}()
        end
        if level_multiplier ∉ keys(setup_dict[key])
            setup_dict[key][level_multiplier] = Dict{Int64,Dict{Int64,Float64}}()
            lu_dict[key][level_multiplier] = Dict{Int64,Dict{Int64,Float64}}()
            solve_dict[key][level_multiplier] = Dict{Int64,Dict{Int64,Float64}}()
        end
        if n_shared_blocks ∉ keys(setup_dict[key][level_multiplier])
            setup_dict[key][level_multiplier][n_shared_blocks] = Dict{Int64,Float64}()
            lu_dict[key][level_multiplier][n_shared_blocks] = Dict{Int64,Float64}()
            solve_dict[key][level_multiplier][n_shared_blocks] = Dict{Int64,Float64}()
        end
        setup_dict[key][level_multiplier][n_shared_blocks][nproc] = tsetup
        lu_dict[key][level_multiplier][n_shared_blocks][nproc] = tlu
        solve_dict[key][level_multiplier][n_shared_blocks][nproc] = tsolve

        if key ∉ keys(sizes_dict)
            base_label = "$total_size"
            size_label = base_label
            if size_label ∈ sizes_set
                for suffix ∈ collect('A':'Z')
                    size_label = base_label * suffix
                    if size_label ∉ sizes_set
                        break
                    end
                end
                if size_label ∈ sizes_set
                    error("too many entries with total_size=$total_size")
                end
            end
            push!(sizes_set, size_label)
            sizes_dict[key] = size_label
        end
    end

    return setup_dict, lu_dict, solve_dict, sizes_dict
end

function plot_scaling!(ax, params, all_results; label=nothing, linestyle=nothing)
    if params ∉ keys(all_results)
        # Don't have a result for these parameters, so skip.
        return nothing
    end
    results = all_results[params]

    level_multipliers = collect(keys(results))
    sort!(level_multipliers)

    first_plot = true
    for lm ∈ level_multipliers
        level_results = results[lm]
        n_shared_blocks_list = collect(keys(level_results))
        sort!(n_shared_blocks_list)
        for n_shared_blocks ∈ n_shared_blocks_list
            nsb_results = level_results[n_shared_blocks]
            nprocs = collect(keys(nsb_results))
            sort!(nprocs)
            times = [nsb_results[n] for n ∈ nprocs]
            if label === nothting
                label = "level_multiplier=$lm"
            end
            label *= " n_shared_blocks=$n_shared_blocks"
            kwargs = Dict{Symbol,Any}()
            kwargs[:label] = label
            if linestyle !== nothing
                kwargs[:linestyle] = linestyle
            end
            lines!(nprocs, times; kwargs...)

            if first_plot
                t0 = times[1]
                n0 = nprocs[1]
                expected_times = [t0 * n0 / n for n ∈ nprocs]
                lines!(nprocs, expected_times; linestyle=:dot, label="ideal scaling")
                first_plot = false
            end
        end
    end

    return nothing
end

function plot_serial_reference!(ax, params, results)
    if params ∉ keys(results)
        # Don't have an UMFPACK result for these parameters, so skip.
        return nothing
    end
    t = results[params][1][1]
    hlines!(ax, t; linestyle=:dash, label="UMFPACK")
    return nothing
end

function plot_comparison(case, interactive_parameter=nothing; datainspector_kwargs=Dict())
    setup_dict, lu_dict, solve_dict, sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_MPIStaticCondensations_$(case).txt"))
    MUMPS_setup_dict, MUMPS_lu_dict, MUMPS_solve_dict, MUMPS_sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_MUMPS_$(case).txt"))
    UMFPACK_setup_dict, UMFPACK_lu_dict, UMFPACK_solve_dict, UMFPACK_sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_UMFPACK_$(case).txt"))

    if interactive_parameter === nothing
        parameter_list = keys(solve_dict)
        interactive_plot = false
    else
        parameter_list = [interactive_parameter]
        interactive_plot = true
        backend = Makie.current_backend()
    end
    for p ∈ parameter_list
        label = sizes_dict[p]

        setup_fig = Figure()
        setup_ax = Axis(setup_fig[1,1]; xscale=log2, yscale=log10, title="$p setup")
        plot_scaling!(setup_ax, p, setup_dict)
        plot_scaling!(setup_ax, p, MUMPS_setup_dict; label="MUMPS", linestyle=:dash)
        plot_serial_reference!(setup_ax, p, UMFPACK_setup_dict)
        if interactive_plot
            DataInspector(setup_fig; datainspector_kwargs...)
            display(backend.Screen(), setup_fig)
        else
            Legend(setup_fig[2,1], setup_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "setup-$label.png"), setup_fig)
        end

        lu_fig = Figure()
        lu_ax = Axis(lu_fig[1,1]; xscale=log2, yscale=log10, title="$p lu")
        plot_scaling!(lu_ax, p, lu_dict)
        plot_scaling!(lu_ax, p, MUMPS_lu_dict; label="MUMPS", linestyle=:dash)
        plot_serial_reference!(lu_ax, p, UMFPACK_lu_dict)
        if interactive_plot
            DataInspector(lu_fig; datainspector_kwargs...)
            display(backend.Screen(), lu_fig)
        else
            Legend(lu_fig[2,1], lu_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "lu-$label.png"), lu_fig)
        end

        solve_fig = Figure()
        solve_ax = Axis(solve_fig[1,1]; xscale=log2, yscale=log10, title="$p solve")
        plot_scaling!(solve_ax, p, solve_dict)
        plot_scaling!(solve_ax, p, MUMPS_solve_dict; label="MUMPS", linestyle=:dash)
        plot_serial_reference!(solve_ax, p, UMFPACK_solve_dict)
        if interactive_plot
            DataInspector(solve_fig; datainspector_kwargs...)
            display(backend.Screen(), solve_fig)
        else
            Legend(solve_fig[2,1], solve_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "solve-$label.png"), solve_fig)
        end
    end
end

for case ∈ cases
    plot_comparison(case)
end
