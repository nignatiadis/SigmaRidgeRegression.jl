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

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\risk}[1]{\bm{R}(#1)}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[bbgreekl]{mathbbol}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\sigmacv}{\bbsigma}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\bSigma}{\bm{\Sigma}}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\bw}{\bm{w}}")





all_files = load.(readdir("simulation_results"))
all_files = load("simulation_results/loaded_files.jld2")["loaded_files"]
all_tuples = vcat([x["res"] for x in all_files]...)

df = DataFrame(all_tuples)
df.n = getfield.(df.sim, :ntrain)


summary_f = x-> median(x) .- 25 #25 is noise variance

gdf = groupby(df, [:cov,:n ,:K]) |>
      df -> combine(df, :mse_sigma =>  summary_f => :mse_sigma,
                        :mse_single => summary_f  => :mse_single,
                        :mse_multi => summary_f  => :mse_multi,
                        :mse_glasso => summary_f  => :mse_glasso,
                        :mse_bayes => summary_f  => :mse_bayes,
                        nrow) |>
      df -> groupby(df, [:n,:cov])


f_tbl_norm(i) = [gdf[i].mse_sigma  gdf[i].mse_single gdf[i].mse_multi gdf[i].mse_glasso] ./ gdf[i].mse_bayes
f_tbl(i) = [gdf[i].mse_single gdf[i].mse_glasso gdf[i].mse_multi gdf[i].mse_sigma  gdf[i].mse_bayes]



_cols =  [:steelblue :green :orange :purple :grey]
_markers = [:utriangle :dtriangle :diamond   :pentagon  :circle]
_labels = ["Single Ridge" "Group Lasso" "Multi Ridge" L"\sigmacv\textrm{-Ridge}"  "Bayes"]
_linestyles =[:dot :dashdot :dashdotdot :solid :dash]

plot_params = (frame = :box, grid=nothing,
    color = _cols,
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    thickness_scaling = 2.3,
    markershape = _markers,
    ylabel = L"\risk{\widehat{\bw}} - \sigma^2",
    linestyle = _linestyles,
    xlabel=L"K",
    xscale = :log2,
    markeralpha=0.6,
    size= (550, 440))

pl1= plot(gdf[1].K,  f_tbl(1); label=_labels, legend=:topleft, title=L"\bSigma=\textrm{AR}(0.8),\;n=p/2", ylim=(0,650), plot_params...)
pl2= plot(gdf[2].K,  f_tbl(2); label=nothing, title=L"\bSigma=\textrm{AR}(0.8),\;n=p", ylim=(0,220), plot_params...)
pl3= plot(gdf[3].K,  f_tbl(3); label=nothing, title=L"\bSigma=\textrm{AR}(0.8),\;n=2p", ylim=(0,45), plot_params...)

pl4= plot(gdf[4].K,  f_tbl(4); label=nothing, title=L"\bSigma=I,\; n=p/2", legend=:topleft, ylim=(0,650), plot_params... )
pl5= plot(gdf[5].K,  f_tbl(5); label=nothing, title=L"\bSigma=I,\; n=p", ylim=(0,220), plot_params... )
pl6= plot(gdf[6].K,  f_tbl(6); label=nothing, title=L"\bSigma=I,\; n=2p", ylim=(0,45), plot_params... )

savefig(pl1, "simulations_ar_phalf.tikz")
savefig(pl2, "simulations_ar_p.tikz")
savefig(pl3, "simulations_ar_ptwice.tikz")


savefig(pl4, "simulations_id_phalf.tikz")
savefig(pl5, "simulations_id_p.tikz")
savefig(pl6, "simulations_id_ptwice.tikz")


pl_ar= plot(pl1,pl2,pl3, size=(1150,280), layout=(1,3))
savefig(pl_ar, "simulations_ar.tikz")

pl_id= plot(pl4,pl5,pl6, size=(1150,280), layout=(1,3))

pl = plot(pl4,pl5,pl6,pl1,pl2,pl3, size=(1200,800), layout=(2,3))
savefig(pl_ar, "simulations_identity.tikz")
