using Pkg
Pkg.activate(@__DIR__)
using FileIO
using DataFrames
using SigmaRidgeRegression
using Distributions
using LaTeXStrings
using Plots
using StatsPlots
using PGFPlotsX

pgfplotsx()

empty!(PGFPlotsX.CUSTOM_PREAMBLE)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\risk}[1]{\bm{R}(#1)}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[bbgreekl]{mathbbol}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\sigmacv}{\bbsigma}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\bSigma}{\bm{\Sigma}}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\bw}{\bm{w}}")

sim_path = joinpath(@__DIR__, "simulation_results")

all_files = FileIO.load.(joinpath.(sim_path, readdir(sim_path)))
all_tuples = vcat((x["res"] for x in all_files)...)

df = DataFrame(all_tuples)
df.n = getfield.(df.sim, :ntrain)

summary_f = x -> mean(x) .- 25 #25 is noise variance


gdf = groupby(df, [:cov,:n ,:K]) |>
      df -> combine(df, :mse_sigma =>  summary_f => :mse_sigma,
                        :mse_single => summary_f  => :mse_single,
                        :mse_multi => summary_f  => :mse_multi,
                        :mse_glasso => summary_f  => :mse_glasso,
                        :mse_bayes => summary_f  => :mse_bayes,
                        nrow) |>
      df -> groupby(df, [:cov,:n])

function f_tbl(i)
    return [gdf[i].mse_single gdf[i].mse_glasso gdf[i].mse_multi gdf[i].mse_sigma gdf[i].mse_bayes]
end

_cols = [:steelblue :green :orange :purple :grey]
_markers = [:utriangle :dtriangle :diamond :pentagon :circle]
_labels = ["Single Ridge" "Group Lasso" "Multi Ridge" L"\sigmacv\textrm{-Ridge}" "Bayes"]
_linestyles = [:dot :dashdot :dashdotdot :solid :dash]

plot_params = (
    frame=:box,
    grid=nothing,
    color=_cols,
    background_color_legend=:transparent,
    foreground_color_legend=:transparent,
    thickness_scaling=2.3,
    markershape=_markers,
    ylabel=L"\mathbb{E}[\risk{\widehat{\bw}}] - \sigma^2",
    linestyle=_linestyles,
    xlabel=L"K",
    xscale=:log2,
    markeralpha=0.6,
    size=(550, 440),
)

idx1 = (cov=AR1Design{Int64}(800, 0.8), n=400)
pl1 = plot(
    gdf[idx1].K,
    f_tbl(idx1);
    label=_labels,
    legend=:topleft,
    title=L"\bSigma=\textrm{AR}(0.8),\;n=p/2",
    ylim=(0, 795),
    plot_params...,
)

idx2 = (cov=AR1Design{Int64}(800, 0.8), n=800)
pl2 = plot(
    gdf[idx2].K,
    f_tbl(idx2);
    label=nothing,
    title=L"\bSigma=\textrm{AR}(0.8),\;n=p",
    ylim=(0, 285),
    plot_params...,
)

idx3 = (cov=AR1Design{Int64}(800, 0.8), n=1600)
pl3 = plot(
    gdf[idx3].K,
    f_tbl(idx3);
    label=nothing,
    title=L"\bSigma=\textrm{AR}(0.8),\;n=2p",
    ylim=(0, 55),
    plot_params...,
)

idx4 = (cov=IdentityCovarianceDesign{Int64}(800), n=400)
pl4 = plot(
    gdf[idx4].K,
    f_tbl(idx4);
    label=nothing,
    title=L"\bSigma=I,\; n=p/2",
    legend=:topleft,
    ylim=(0, 795),
    plot_params...,
)

idx5 = (cov=IdentityCovarianceDesign{Int64}(800), n=800)
pl5 = plot(
    gdf[idx5].K,
    f_tbl(idx5);
    label=nothing,
    title=L"\bSigma=I,\; n=p",
    ylim=(0, 285),
    plot_params...,
)

idx6 = (cov=IdentityCovarianceDesign{Int64}(800), n=1600)
pl6 = plot(
    gdf[6].K,
    f_tbl(6);
    label=nothing,
    title=L"\bSigma=I,\; n=2p",
    ylim=(0, 55),
    plot_params...,
)

savefig(pl1, "simulations_ar_phalf.tikz")
savefig(pl2, "simulations_ar_p.tikz")
savefig(pl3, "simulations_ar_ptwice.tikz")

savefig(pl4, "simulations_id_phalf.tikz")
savefig(pl5, "simulations_id_p.tikz")
savefig(pl6, "simulations_id_ptwice.tikz")
