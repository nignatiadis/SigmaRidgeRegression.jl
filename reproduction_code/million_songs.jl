using Pkg
Pkg.activate(@__DIR__)
using CSV
using DataFrames
using StatsBase
using RCall
using SigmaRidgeRegression
using LaTeXStrings
using LinearAlgebra
using MLJ
using Random
using ColorSchemes
using Plots
using StatsPlots
using PGFPlotsX

msd_filepath = joinpath(@__DIR__, "dataset", "YearPredictionMSD.txt")
msd = DataFrame(CSV.File(msd_filepath; header=false))

function feature_map(X)
    mean_features = X[:, 1:12]
    var_features = X[:, 13:24]
    sd_features = sqrt.(var_features)
    cov_features = X[:, 25:90]
    cor_features = zeros(size(cov_features))
    cnt = 0
    for offset in 1:11
        for i in 1:(12 - offset)
            cnt = cnt + 1
            cor_features[:, cnt] =
                cov_features[:, cnt] ./
                sqrt.(var_features[:, i] .* var_features[:, i + offset])
        end
    end
    grp = GroupedFeatures([12, 12, 66, 66])
    return (MLJ.table([mean_features sd_features cov_features cor_features]), grp)
end

Y = Float64.(msd[:, 1])
X, groups = feature_map(Matrix(msd[:, 2:91]));

train_idx = 1:463_715
test_idx = (1:51_630) .+ 463_715

loo_single_ridge = LooRidgeRegressor(;
    ridge=SingleGroupRidgeRegressor(; groups=groups, center=true, scale=true)
)

loo_sigmaridge = LooSigmaRidgeRegressor(; groups=groups, center=true, scale=true)

loo_multi_ridge = LooRidgeRegressor(;
    ridge=MultiGroupRidgeRegressor(; groups=groups, center=true, scale=true),
    rng=MersenneTwister(1),
)

holdout_glasso = TunedSeagull(; groups=groups, center=true, scale=true)

ns_subsample = [500; 1000; 5000]
n_montecarlo = 20
Random.seed!(10)

mse_array = Array{Float64}(undef, length(ns_subsample), n_montecarlo, 4)
time_array = Array{Float64}(undef, length(ns_subsample), n_montecarlo, 4)
λs_array = Array{Any}(undef, length(ns_subsample), n_montecarlo, 4)
#an initial run to make sure everything is precompiled followed by n_montecarlo runs
for n_montecarlo in [1; n_montecarlo]
for (k, n_subsample) in enumerate(ns_subsample)
    for j in Base.OneTo(n_montecarlo)
        train_idx_subsample = sample(train_idx, n_subsample; replace=false)
        resampling_idx = [(train_idx_subsample, test_idx)]
        for (i, mach) in
            enumerate([loo_sigmaridge, loo_single_ridge, loo_multi_ridge, holdout_glasso])
            time_array[k, j, i] = @elapsed begin
                _mach = machine(mach, X, Y)
                _eval = evaluate!(_mach; resampling=resampling_idx, measure=l2)
            end
            mse_array[k, j, i] = _eval.measurement[1]
            λs_array[k, j, i] = deepcopy(_mach.report.best_λs)
        end
    end
end
end

pgfplotsx()

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\blambda}{\bm{\lambda}}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\risk}[1]{\bm{R}(#1)}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[bbgreekl]{mathbbol}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\sigmacv}{\bbsigma}")

_orange = RGB{Float64}(0.933027, 0.665164, 0.198652)
_orange2 = RGB{Float64}(0.85004, 0.540122, 0.136212)

method_names =
    [L"\sigmacv-\textrm{Ridge}" L"\textrm{Single Ridge}" L"\textrm{Multi Ridge}" L"\textrm{Group Lasso}"]

_thickness_scaling = 1.8
mse_plot = plot(
    dotplot(
        method_names,
        mse_array[1, :, :];
        title=L"n\;=\;%$(ns_subsample[1])",
        frame=:box,
        grid=nothing,
        yguide="Mean squared error",
        label=nothing,
        markerstrokecolor=_orange2,
        markerstrokewidth=0.5,
        alpha=0.7,
        thickness_scaling=_thickness_scaling,
        ylim=(89, 110),
        color=_orange,
    ),
    dotplot(
        method_names,
        mse_array[2, :, :];
        title=L"n\;=\;%$(ns_subsample[2])",
        frame=:box,
        grid=nothing,
        label=nothing,
        markerstrokecolor=_orange2,
        markerstrokewidth=0.5,
        ylim=(89, 110),
        alpha=0.7,
        thickness_scaling=_thickness_scaling,
        color=_orange,
    ),
    dotplot(
        method_names,
        mse_array[3, :, :];
        title=L"n\;=\;%$(ns_subsample[3])",
        frame=:box,
        grid=nothing,
        label=nothing,
        markerstrokecolor=_orange2,
        markerstrokewidth=0.5,
        thickness_scaling=_thickness_scaling,
        alpha=0.7,
        ylim=(89, 110),
        color=_orange,
    );
    size=(1650, 400),
    layout=(1, 3),
)




savefig(mse_plot, "one_million_songs_mse.pdf")

time_plot = plot(
    dotplot(
        method_names,
        time_array[1, :, :];
        title=L"n\;=\;%$(ns_subsample[1])",
        frame=:box,
        grid=nothing,
        yguide="Time (seconds)",
        label=nothing,
        markerstrokecolor=_orange2,
        alpha=0.7,
        markerstrokewidth=0.5,
        yscale=:log10,
        ylim = (0.1, 100),
        yticks = ([0.1, 1,10,100], ["0.1","1","10","100"]),
        #ylim=(-0.5, 58.0),
        thickness_scaling=_thickness_scaling,
        color=_orange,
    ),
    dotplot(
        method_names,
        time_array[2, :, :];
        title=L"n\;=\;%$(ns_subsample[2])",
        frame=:box,
        grid=nothing,
        label=nothing,
        alpha=0.7,
        markerstrokecolor=_orange2,
        markerstrokewidth=0.5,
        thickness_scaling=_thickness_scaling,
        yscale=:log10,
        ylim = (0.1, 100),
        yticks = ([0.1, 1,10,100], ["0.1","1","10","100"]),
        #ylim=(-0.5, 58.0),
        color=_orange,
    ),
    dotplot(
        method_names,
        time_array[3, :, :];
        title=L"n\;=\;%$(ns_subsample[3])",
        frame=:box,
        grid=nothing,
        label=nothing,
        alpha=0.7,
        markerstrokecolor=_orange2,
        markerstrokewidth=0.5,
        thickness_scaling=_thickness_scaling,
        yscale=:log10,
        ylim = (0.1, 100),
        yticks = ([0.1, 1,10,100], ["0.1","1","10","100"]),
        #ylim=(-0.5, 58.0),
        color=_orange,
    );
    size=(1650, 400),
    layout=(1, 3),
)

savefig(time_plot, "one_million_songs_time.pdf")

_trunc = 20

λs_mean = min.(getindex.(λs_array, 1), 20)
λs_std = min.(getindex.(λs_array, 2), 20)
λs_cov = min.(getindex.(λs_array, 3), 20)
λs_cor = min.(getindex.(λs_array, 4), 20)

λs_sigma_ridge = [λs_mean[1, :, 1] λs_std[1, :, 1] λs_cov[1, :, 1] λs_cor[1, :, 1]]

feature_names = ["mean" "std" "cov" "cor"]
λs_names = [L"\hat{\lambda}_{\textrm{%$n}}" for n in feature_names]

λ_plot_params = (;
    frame=:box,
    grid=nothing,
    label="",
    markerstrokecolor=:purple,
    markerstrokewidth=0.5,
    markercolor=RGB{Float64}(205 / 256, 153 / 256, 255 / 256),
    thickness_scaling=1.7,
    ylim=(-0.9, _trunc + 0.8),
)


method_names_short =
    [L"\sigmacv-\textrm{Ridge}" L"\textrm{Single}" L"\textrm{Multi}" L"\textrm{GLasso}"]

plot(
    dotplot(
        method_names_short,
        λs_mean[1, :, :];
        yguide=L"\min\{\widehat{\lambda},20\}",
        title=λs_names[1],
    ),
    dotplot(method_names_short, λs_std[1, :, :]; title=λs_names[2]),
    dotplot(method_names_short, λs_cov[1, :, :]; title=λs_names[3]),
    dotplot(method_names_short, λs_cor[1, :, :]; title=λs_names[4]);
    size=(1500, 270),
    layout=(1, 4),
    λ_plot_params...
)
savefig("one_million_song_lambdas_n500.pdf")

plot(
    dotplot(method_names_short, λs_mean[2, :, :]; yguide=L"\min\{\widehat{\lambda},20\}"),
    dotplot(method_names_short, λs_std[2, :, :]),
    dotplot(method_names_short, λs_cov[2, :, :]),
    dotplot(method_names_short, λs_cor[2, :, :]);
    size=(1500, 260),
    layout=(1, 4),
    λ_plot_params...
)
savefig("one_million_song_lambdas_n1000.pdf")

plot(
    dotplot(method_names_short, λs_mean[3, :, :]; yguide=L"\min\{\widehat{\lambda},20\}"),
    dotplot(method_names_short, λs_std[3, :, :]),
    dotplot(method_names_short, λs_cov[3, :, :]),
    dotplot(method_names_short, λs_cor[3, :, :]);
    size=(1500, 260),
    layout=(1, 4),
    λ_plot_params...,
)

savefig("one_million_song_lambdas_n5000.pdf")
