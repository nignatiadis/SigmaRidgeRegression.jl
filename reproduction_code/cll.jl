using Pkg
Pkg.activate(@__DIR__)
using RCall
using SigmaRidgeRegression
using StatsBase
using Plots
using Statistics
using Random
using MLJ
using Tables
using LaTeXTabulars
using LaTeXStrings
using BenchmarkTools


R"""
	data("CLL_data", package="MOFAdata")

	# use methylation data, gene expression data and drug responses as predictors
	CLL_data <- CLL_data[1:3]
	CLL_data <- lapply(CLL_data,t)
	ngr <- sapply(CLL_data,ncol)
	CLL_data <- Reduce(cbind, CLL_data)

	#only include patient samples profiles in all three omics
	CLL_data2 <- CLL_data[apply(CLL_data,1, function(p) !any(is.na(p))),]
	dim(CLL_data2)

	# prepare design matrix and response
	X <- CLL_data2[,!grepl("D_002", colnames(CLL_data))]
	y <- rowMeans(CLL_data2[,grepl("D_002", colnames(CLL_data))])
	annot <- rep(1:3, times = ngr-c(5,0,0)) # group annotations to drugs, meth and RNA
	ngr_prime <- ngr-c(5,0,0)
"""

# run with seed from Velten & Huber
R"""
set.seed(9876)
foldid <- sample(rep(seq(10), length=nrow(X)))
"""

@rget foldid
resample_ids = [ ( findall(foldid .!= k), findall(foldid .== k) )    for k in 1:10 ]

@rget X
@rget y
@rget ngr_prime



groups = GroupedFeatures(Int.(ngr_prime))

Random.seed!(1)
rand_idx = sample(1:121, 121, replace=false)
Xdrug_resample = X[rand_idx, 1:305]
Xgaussian = randn(121, 100)
Xnoise = [X Xdrug_resample Xgaussian]

groups_noise = GroupedFeatures([Int.(ngr_prime); 305; 100])


X_table = MLJ.table(X);
X_plus_noise =  MLJ.table(Xnoise);




function single_table_line(X, y, resampling, _model, model_name; tuning_name=nothing, sigdigits=3)
	ridge_machine = machine(_model, X, y)
	fit!(ridge_machine)
	best_param = ridge_machine.report.best_param
	if isnothing(tuning_name)
		tuning_string =""
	else
		best_param = round(best_param; sigdigits=sigdigits)
		tuning_string = L"%$(tuning_name) = %$(best_param)"
	end

	λs = round.(deepcopy(ridge_machine.report.best_λs), sigdigits=sigdigits)

	#if isa(_model.ridge, SingleGroupRidgeRegressor)
	#	λs = fill(λs[1], 3)
	#end

	ridge_benchmark = @benchmark fit!(machine($(_model), $(X), $(y)))

	time_loo = round(mean(ridge_benchmark.times)/(1e9),  sigdigits=sigdigits)

	eval_ridge = evaluate!(ridge_machine,  resampling=resampling, measure=rms)
	_rmse = round(eval_ridge.measurement[1], sigdigits=sigdigits)

	#[model_name, tuning_string, λs, _rmse,  time_loo]
	[model_name, tuning_string, λs..., time_loo, _rmse]
end



single_ridge = SingleGroupRidgeRegressor(decomposition = :woodbury, groups=groups, λ=1.0, center=true, scale=true)
loo_single_ridge = LooRidgeRegressor(ridge = deepcopy(single_ridge))

single_ridge_noise = SingleGroupRidgeRegressor(decomposition = :woodbury, groups=groups_noise, λ=1.0, center=true, scale=true)
loo_single_ridge_noise = LooRidgeRegressor(ridge = deepcopy(single_ridge_noise))

sigma_ridge = SigmaRidgeRegressor(groups=groups, decomposition = :woodbury, σ=0.01, center=true, scale=true)
loo_sigmaridge = LooRidgeRegressor(ridge=deepcopy(sigma_ridge), tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))

sigma_ridge_noise = SigmaRidgeRegressor(groups=groups_noise, decomposition = :woodbury, σ=0.01, center=true, scale=true)
loo_sigmaridge_noise = LooRidgeRegressor(ridge=deepcopy(sigma_ridge_noise), tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))

multi_ridge = MultiGroupRidgeRegressor(groups; decomposition = :woodbury, center=true, scale=true)
loo_multi_ridge = LooRidgeRegressor(ridge = deepcopy(multi_ridge), rng=MersenneTwister(1))

multi_ridge_noise = MultiGroupRidgeRegressor(groups_noise; decomposition = :woodbury, center=true, scale=true)
loo_multi_ridge_noise = LooRidgeRegressor(ridge = deepcopy(multi_ridge_noise), rng=MersenneTwister(1))

 #CV(nfolds=5, shuffle=true, rng=1))
glasso = GroupLassoRegressor(groups=groups, decomposition = :woodbury, center=true, scale=true)
cv_glasso = TunedRidgeRegressor(ridge=deepcopy(glasso), resampling= Holdout(shuffle=true, rng=1), tuning=DefaultTuning(param_min_ratio=1e-5))

glasso_noise = GroupLassoRegressor(groups=groups_noise, decomposition = :woodbury, center=true, scale=true)
cv_glasso_noise = TunedRidgeRegressor(ridge=deepcopy(glasso_noise), resampling= Holdout(shuffle=true, rng=1), tuning=DefaultTuning(param_min_ratio=1e-5))

line_single_ridge = single_table_line(X_table, y, resample_ids, loo_single_ridge, "\textbf{Single Ridge}")
line_single_ridge_noise = single_table_line(X_plus_noise, y, resample_ids, loo_single_ridge_noise, "\textbf{Single Ridge}")

line_sigma_ridge = single_table_line(X_table, y, resample_ids, loo_sigmaridge, L"$\sigmacv$\textbf{-Ridge}"; tuning_name=L"\sigmacv")
line_sigma_ridge_noise = single_table_line(X_plus_noise, y, resample_ids, loo_sigmaridge_noise, L"$\sigmacv$\textbf{-Ridge}"; tuning_name=L"\sigmacv")

line_multi_ridge = single_table_line(X_table, y, resample_ids, loo_multi_ridge, "\textbf{Multi Ridge}")
line_multi_ridge_noise = single_table_line(X_plus_noise, y, resample_ids, loo_multi_ridge_noise, "\textbf{Multi Ridge}")

line_glasso = single_table_line(X_table, y, resample_ids, cv_glasso, L"\textbf{Group Lasso}", tuning_name=L"$\widehat{\lambda}^{gl}")
line_glasso_noise = single_table_line(X_plus_noise, y, resample_ids, cv_glasso_noise, L"\textbf{Group Lasso}", tuning_name=L"$\widehat{\lambda}^{gl}")



tbl_spec = Tabular("lllllll")
line1 = ["", "Tuning", L"$\widehat{\lambda}_{\text{Drugs}}$",  L"\widehat{\lambda}_{\text{Methyl}}", L"\widehat{\lambda}_{\text{RNA}}",  "Time (s)", "10-fold RMSE"]
lines  = [line1, Rule(), line_sigma_ridge, line_single_ridge, line_multi_ridge, line_glasso]
latex_tabular("cll_analysis.tex", tbl_spec, lines)


tbl_spec2 = Tabular("lllllllll")
line1 = ["", "Tuning", L"$\widehat{\lambda}_{\text{Drugs}}$",
                       L"\widehat{\lambda}_{\text{Methyl}}",
                       L"\widehat{\lambda}_{\text{RNA}}",
                       L"\widehat{\lambda}_{\text{Noise1}}",
                       L"\widehat{\lambda}_{\text{Noise2}}",
                        "Time (s)", "10-fold RMSE"]
lines  = [line1, Rule(), line_sigma_ridge_noise, line_single_ridge_noise, line_multi_ridge_noise, line_glasso_noise]
latex_tabular("cll_analysis_noise.tex", tbl_spec2, lines)
