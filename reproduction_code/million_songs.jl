using Pkg
Pkg.activate(@__DIR__)
using Pkg.Artifacts
using CSV
using DataFrames
using StatsBase
using SigmaRidgeRegression
using LaTeXStrings
using LinearAlgebra
using MLJ
using Random
using ColorSchemes
using Plots
using StatsPlots
using PGFPlotsX

#------------------------------------------------------------------
# Code that generated the Artifact file Artifacts.toml
#------------------------------------------------------------------
#using ArtifactUtils
#add_artifact!(
#           "Artifacts.toml",
#           "YearPredictionMSD",
#           "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
#           force=true,
#       )


#--------------------------------------------------------------------------------------
# Command below may take a while since it will automatically download the Million Song
# Dataset from the UCI repository (around 400 MB).
#--------------------------------------------------------------------------------------

msd_filepath = joinpath(artifact"YearPredictionMSD", "YearPredictionMSD.txt")
msd = CSV.File(msd_filepath, header=false) |> DataFrame



function feature_map(X) #; noisegroups=10, noisefeatures=50)
	mean_features = X[:, 1:12]
	var_features = X[:, 13:24]
	sd_features = sqrt.(var_features)
	#cv_features = sd_features ./ abs.(mean_features)
	cov_features = X[:, 25:90]
	cor_features = zeros(size(cov_features))
	cnt = 0
	for offset=1:11
		for i=1:(12-offset)
			cnt = cnt + 1
			cor_features[:,cnt] = cov_features[:,cnt] ./ sqrt.(var_features[:,i] .* var_features[:,i+offset])
		end
	end
	#noise_features = randn(size(X,1)#, noisegroups*noisefeatures )
	grp = GroupedFeatures([12, 12, 66, 66]) #, fill(noisefeatures, noisegroups)))
	(MLJ.table([mean_features sd_features cov_features cor_features]), grp)
end



Y = Float64.(msd[:, 1])
X, groups = feature_map(Matrix(msd[:, 2:91]))

train_idx = 1:463_715
test_idx =  (1:51_630) .+ 463_715

single_ridge = SingleGroupRidgeRegressor(decomposition = :cholesky, groups=groups, λ=1.0, center=true, scale=true)
loo_single_ridge = LooRidgeRegressor(ridge = deepcopy(single_ridge))

sigma_ridge = SigmaRidgeRegressor(groups=groups, decomposition = :cholesky, σ=0.01, center=true, scale=true)
loo_sigmaridge = LooRidgeRegressor(ridge=deepcopy(sigma_ridge), tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))

multi_ridge = MultiGroupRidgeRegressor(groups; decomposition = :cholesky, center=true, scale=true)
loo_multi_ridge = LooRidgeRegressor(ridge = deepcopy(multi_ridge), rng=MersenneTwister(1))

glasso = GroupLassoRegressor(groups=groups, decomposition = :cholesky, center=true, scale=true)
holdout_glasso = TunedRidgeRegressor(ridge=deepcopy(glasso), resampling= Holdout(shuffle=true, rng=1), tuning=DefaultTuning(param_min_ratio=1e-5))


ns_subsample = [200; 500; 1000]
n_montecarlo = 20
Random.seed!(1)

mse_array = Array{Float64}(undef, length(ns_subsample), n_montecarlo, 4)
time_array = Array{Float64}(undef, length(ns_subsample), n_montecarlo, 4)
λs_array = Array{Any}(undef, length(ns_subsample), n_montecarlo, 4)
for (k, n_subsample) in enumerate(ns_subsample)
    for j in Base.OneTo(n_montecarlo)
        train_idx_subsample = sample(train_idx, n_subsample, replace=false)
        resampling_idx = [(train_idx_subsample, test_idx)]
        for (i,mach) in enumerate([loo_sigmaridge, loo_single_ridge, loo_multi_ridge, holdout_glasso])
            time_array[k,j,i] = @elapsed begin
                _mach = machine(mach, X, Y)
                _eval = evaluate!(_mach, resampling=resampling_idx, measure=l2)
            end
            mse_array[k,j,i] = _eval.measurement[1]
            λs_array[k,j,i] = deepcopy(_mach.report.best_λs)
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


_orange =   RGB{Float64}(0.933027,0.665164,0.198652)
_orange2 =  RGB{Float64}(0.85004,0.540122,0.136212)

method_names = [L"\sigmacv-\textrm{Ridge}" L"\textrm{Single Ridge}" L"\textrm{Multi Ridge}" L"\textrm{Group Lasso}"]

_thickness_scaling = 1.8
mse_plot = plot(
dotplot(method_names, mse_array[1,:,:],
    title=L"n\;=\;%$(ns_subsample[1])",
    frame = :box, grid=nothing,
    yguide = "Mean squared error",
    label=nothing,
    markerstrokecolor=_orange2,
    markerstrokewidth=0.5,
    thickness_scaling = _thickness_scaling,
    ylim = (88,141),
    color=_orange),
dotplot(method_names, mse_array[2,:,:],
    title=L"n\;=\;%$(ns_subsample[2])",
    frame = :box, grid=nothing,
    label=nothing,
    markerstrokecolor=_orange2,
    markerstrokewidth=0.5,
    ylim = (88,141),
    thickness_scaling = _thickness_scaling,
    color=_orange),
dotplot(method_names, mse_array[3,:,:],
    title=L"n\;=\;%$(ns_subsample[3])",
    frame = :box, grid=nothing,
    label=nothing,
    markerstrokecolor=_orange2,
    markerstrokewidth=0.5,
    thickness_scaling = _thickness_scaling,
    ylim = (88,141),
    color=_orange), size=(1650,400), layout=(1,3))


savefig(mse_plot, "one_million_songs_mse.pdf")

time_plot = plot(
        dotplot(method_names, time_array[1,:,:],
            title=L"n\;=\;%$(ns_subsample[1])",
            frame = :box, grid=nothing,
            yguide = "Time (seconds)",
            label=nothing,
            markerstrokecolor=_orange2,
            markerstrokewidth=0.5,
            ylim = (-0.5,10.5),
            thickness_scaling = _thickness_scaling,
            color=_orange),
        dotplot(method_names, time_array[2,:,:],
            title=L"n\;=\;%$(ns_subsample[2])",
            frame = :box, grid=nothing,
            label=nothing,
            markerstrokecolor=_orange2,
            markerstrokewidth=0.5,
            thickness_scaling = _thickness_scaling,
            ylim = (-0.5,10.5),
            color=_orange),
        dotplot(method_names, time_array[3,:,:],
            title=L"n\;=\;%$(ns_subsample[3])",
            frame = :box, grid=nothing,
            label=nothing,
            markerstrokecolor=_orange2,
            markerstrokewidth=0.5,
            thickness_scaling = _thickness_scaling,
            ylim = (-0.5,10.5),
            color=_orange), size=(1650,400), layout=(1,3))

savefig(time_plot, "one_million_songs_time.pdf")


_trunc = 20

λs_mean = min.(getindex.(λs_array,1), 20)
λs_std = min.(getindex.(λs_array,2), 20)
λs_cov = min.(getindex.(λs_array,3), 20)
λs_cor = min.(getindex.(λs_array,4), 20)

λs_sigma_ridge = [λs_mean[1,:,1] λs_std[1,:,1] λs_cov[1,:,1] λs_cor[1,:,1]]

feature_names = ["mean" "std"  "cov" "cor"]
λs_names = [L"\hat{\lambda}_{\textrm{%$n}}"  for n in feature_names]


λ_plot_params = (frame = :box, grid=nothing,
    label="",
    markerstrokecolor=:purple,
    markerstrokewidth=0.5,
    markercolor= RGB{Float64}(205/256,153/256,255/256),
    thickness_scaling = 1.7,
    ylim = (-0.9,_trunc + 0.8))

#λ_yguide=L"\min\{\widehat{\lambda},20\}",

method_names_short = [L"\sigmacv-\textrm{Ridge}" L"\textrm{Single}" L"\textrm{Multi}" L"\textrm{GLasso}"]


plot(
dotplot(method_names_short, λs_mean[1,:,:]; yguide=L"\min\{\widehat{\lambda},20\}", title=λs_names[1]),
dotplot(method_names_short, λs_std[1,:,:], title=λs_names[2]),
dotplot(method_names_short, λs_cov[1,:,:], title=λs_names[3]),
dotplot(method_names_short, λs_cor[1,:,:], title=λs_names[4]),
size=(1500,270), layout=(1,4); λ_plot_params...)
savefig("one_million_song_lambdas_n200.pdf")

plot(
dotplot(method_names_short, λs_mean[2,:,:]; yguide=L"\min\{\widehat{\lambda},20\}"),
dotplot(method_names_short, λs_std[2,:,:]),
dotplot(method_names_short, λs_cov[2,:,:]),
dotplot(method_names_short, λs_cor[2,:,:]),
size=(1500,260), layout=(1,4); λ_plot_params...)
savefig("one_million_song_lambdas_n1000.pdf")

plot(
dotplot(method_names_short, λs_mean[3,:,:]; yguide=L"\min\{\widehat{\lambda},20\}"),
dotplot(method_names_short, λs_std[3,:,:]),
dotplot(method_names_short, λs_cov[3,:,:]),
dotplot(method_names_short, λs_cor[3,:,:]),
size=(1500,260), layout=(1,4); λ_plot_params...)
savefig("one_million_song_lambdas_n5000.pdf")
