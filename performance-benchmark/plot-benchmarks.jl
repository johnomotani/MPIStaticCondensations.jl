using CSV
using GLMakie

const default_results_directory = "results-benchmark"
const solvers = ("UMFPACK", "MPIStaticCondensations", "MUMPS")
const cases = ("1d", "2d", "3d")

function load_data(filename; count_shared_blocks=true)
    setup_dict = Dict{String,Dict{String,Dict{Int64,Dict{Int64,Float64}}}}()
    lu_dict = Dict{String,Dict{String,Dict{Int64,Dict{Int64,Float64}}}}()
    solve_dict = Dict{String,Dict{String,Dict{Int64,Dict{Int64,Float64}}}}()
    sizes_set = String[]
    sizes_dict = Dict{String,String}()

    if !isfile(filename)
        return setup_dict, lu_dict, solve_dict, sizes_dict
    end

    file = CSV.File(filename; delim=" ", header=false, comment="#")
    for row in file
        nproc, n_shared, ndim, total_size, tsetup, tlu, tsolve, nelement_list, ngrid_list,
            periodic_list, remove_boundaries_list, sparse_C_blocks,
            mumps_fill_in_threshold, block_sizes_heuristic = row

        key = join([nelement_list, ngrid_list], ",")
        other_parameters = join([sparse_C_blocks, mumps_fill_in_threshold, block_sizes_heuristic], ";")
        if count_shared_blocks
            shared_collect_label = nproc ÷ n_shared
        else
            shared_collect_label = n_shared
        end
        if key ∉ keys(setup_dict)
            setup_dict[key] = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
            lu_dict[key] = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
            solve_dict[key] = Dict{String,Dict{Int64,Dict{Int64,Float64}}}()
        end
        if other_parameters ∉ keys(setup_dict[key])
            setup_dict[key][other_parameters] = Dict{Int64,Dict{Int64,Float64}}()
            lu_dict[key][other_parameters] = Dict{Int64,Dict{Int64,Float64}}()
            solve_dict[key][other_parameters] = Dict{Int64,Dict{Int64,Float64}}()
        end
        if shared_collect_label ∉ keys(setup_dict[key][other_parameters])
            setup_dict[key][other_parameters][shared_collect_label] = Dict{Int64,Float64}()
            lu_dict[key][other_parameters][shared_collect_label] = Dict{Int64,Float64}()
            solve_dict[key][other_parameters][shared_collect_label] = Dict{Int64,Float64}()
        end
        setup_dict[key][other_parameters][shared_collect_label][nproc] = tsetup
        lu_dict[key][other_parameters][shared_collect_label][nproc] = tlu
        solve_dict[key][other_parameters][shared_collect_label][nproc] = tsolve

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

    other_parameters_list = collect(keys(results))

    first_plot = true
    for op ∈ other_parameters_list
        level_results = results[op]
        shared_collect_label_list = collect(keys(level_results))
        sort!(shared_collect_label_list)
        for shared_collect_label ∈ shared_collect_label_list
            nsb_results = level_results[shared_collect_label]
            nprocs = collect(keys(nsb_results))
            sort!(nprocs)
            times = [nsb_results[n] for n ∈ nprocs]
            if label === nothing
                this_label = "other_parameters=$op"
            else
                this_label = label
            end
            this_label *= " shared_collect_label=$shared_collect_label"
            kwargs = Dict{Symbol,Any}()
            kwargs[:label] = this_label
            kwargs[:inspector_label] = (self,i,p) -> "$(self.label[])\nnproc=$(p[1]), t=$(p[2])"
            if length(nprocs) == 1
                scatter!(nprocs, times; kwargs...)
            else
                if linestyle !== nothing
                    kwargs[:linestyle] = linestyle
                end
                lines!(nprocs, times; kwargs...)
            end

            if first_plot
                t0 = times[1]
                n0 = nprocs[1]
                expected_times = [t0 * n0 / n for n ∈ nprocs]
                lines!(nprocs, expected_times; linestyle=:dot, label="ideal scaling",
                       inspector_label=(self,i,p) -> "$(self.label[])\nnproc=$(p[1]), t=$(p[2])")
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
    t = first(values(first(values(first(values(results[params]))))))
    # Not sure if we need to use 10^p[2] here because of a bug, or this is correct
    # behaviour of hlines!()...
    hlines!(ax, t; linestyle=:dash, label="UMFPACK",
            inspector_label=(self,i,p) -> "UMFPACK, t=$(10^p[2])")
    return nothing
end

function plot_comparison(case, interactive_parameter=nothing; datainspector_kwargs=Dict(),
                         results_directory=default_results_directory)
    setup_dict, lu_dict, solve_dict, sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_MPIStaticCondensations_$(case).txt"))
    MUMPS_setup_dict, MUMPS_lu_dict, MUMPS_solve_dict, MUMPS_sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_MUMPS_$(case).txt");
                  count_shared_blocks=false)
    UMFPACK_setup_dict, UMFPACK_lu_dict, UMFPACK_solve_dict, UMFPACK_sizes_dict =
        load_data(joinpath(results_directory, "benchmarks_UMFPACK_$(case).txt"))

    merge!(sizes_dict, MUMPS_sizes_dict, UMFPACK_sizes_dict)

    if interactive_parameter === nothing
        parameter_list = keys(merge(solve_dict, MUMPS_solve_dict, UMFPACK_solve_dict))
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
        if setup_dict !== nothing
            plot_scaling!(setup_ax, p, setup_dict)
        end
        if MUMPS_setup_dict !== nothing
            plot_scaling!(setup_ax, p, MUMPS_setup_dict; label="MUMPS", linestyle=:dash)
        end
        if UMFPACK_setup_dict !== nothing
            plot_serial_reference!(setup_ax, p, UMFPACK_setup_dict)
        end
        if interactive_plot
            DataInspector(setup_fig; datainspector_kwargs...)
            display(backend.Screen(), setup_fig)
        else
            Legend(setup_fig[2,1], setup_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "setup-$label.png"), setup_fig)
        end

        lu_fig = Figure()
        lu_ax = Axis(lu_fig[1,1]; xscale=log2, yscale=log10, title="$p lu")
        if lu_dict !== nothing
            plot_scaling!(lu_ax, p, lu_dict)
        end
        if MUMPS_lu_dict !== nothing
            plot_scaling!(lu_ax, p, MUMPS_lu_dict; label="MUMPS", linestyle=:dash)
        end
        if UMFPACK_lu_dict !== nothing
            plot_serial_reference!(lu_ax, p, UMFPACK_lu_dict)
        end
        if interactive_plot
            DataInspector(lu_fig; datainspector_kwargs...)
            display(backend.Screen(), lu_fig)
        else
            Legend(lu_fig[2,1], lu_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "lu-$label.png"), lu_fig)
        end

        solve_fig = Figure()
        solve_ax = Axis(solve_fig[1,1]; xscale=log2, yscale=log10, title="$p solve")
        if solve_dict !== nothing
            plot_scaling!(solve_ax, p, solve_dict)
        end
        if MUMPS_solve_dict !== nothing
            plot_scaling!(solve_ax, p, MUMPS_solve_dict; label="MUMPS", linestyle=:dash)
        end
        if UMFPACK_solve_dict !== nothing
            plot_serial_reference!(solve_ax, p, UMFPACK_solve_dict)
        end
        if interactive_plot
            DataInspector(solve_fig; datainspector_kwargs...)
            display(backend.Screen(), solve_fig)
        else
            Legend(solve_fig[2,1], solve_ax; tellwidth=false, tellheight=true)
            save(joinpath(results_directory, "solve-$label.png"), solve_fig)
        end
    end
end

function plot_comparisons(results_directory=default_results_directory)
    for case ∈ cases
        plot_comparison(case; results_directory)
    end
    return nothing
end

plot_comparisons()
