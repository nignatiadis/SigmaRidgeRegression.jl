using RCall
using SigmaRidgeRegression
using StatsBase
using Plots
using Statistics
using Random
using MLJLinearModels
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

# run with seed from...	
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




X = MLJ.table(X);




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
	
	if isa(_model.ridge, SingleGroupRidgeRegressor)
		λs = fill(λs[1], 3)
	end 

	ridge_benchmark = @benchmark fit!(machine($(_model), $(X), $(y)))
		
	time_loo = round(mean(ridge_benchmark.times)/(1e9),  sigdigits=sigdigits)
	
	eval_ridge = evaluate!(ridge_machine,  resampling=resampling, measure=rms)
	_rmse = round(eval_ridge.measurement[1], sigdigits=sigdigits)
	
	[model_name, tuning_string, λs[1], λs[2], λs[3], _rmse, time_loo]
end


line_single_ridge = single_table_line(X, y, resample_ids, loo_single_ridge, "\textbf{Single Ridge}")
line_sigma_ridge = single_table_line(X, y, resample_ids, loo_sigmaridge, L"$\sigmacv$\textbf{-Ridge}"; tuning_name=L"\sigmacv")

line_multi_ridge = single_table_line(X, y, resample_ids, loo_multi_ridge, "\textbf{Multi Ridge}")
line_glasso = single_table_line(X, y, resample_ids, loo_glasso, "\textbf{Group Lasso}"; tuning_name=L"\lambda^{\text{glasso}}")


single_ridge = SingleGroupRidgeRegressor(decomposition = :woodbury, λ=0.00001, center=true, scale=true)
loo_single_ridge = LooRidgeRegressor(ridge = deepcopy(single_ridge))


sigma_ridge = SigmaRidgeRegressor(groups=groups, decomposition = :woodbury, σ=0.01, center=true, scale=true)
loo_sigmaridge = LooRidgeRegressor(ridge=sigma_ridge, tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))





loo_single_ridge_machine = machine(loo_single_ridge, X, y)
fit!(loo_single_ridge_machine)

loo_sigma_ridge_machine = machine(loo_sigmaridge, X, y)
fit!(loo_sigma_ridge_machine)

multi_ridge = MultiGroupRidgeRegressor(groups; decomposition = :woodbury, center=true, scale=true)
loo_multi_ridge = LooRidgeRegressor(ridge = deepcopy(multi_ridge), rng=MersenneTwister(1))
loo_multi_ridge_machine = machine(loo_multi_ridge, X, y)
fit!(loo_multi_ridge_machine)

loo_multi_ridge_machine.report.best_λs



glasso = GroupLassoRegressor(groups=groups, decomposition = :woodbury, center=true, scale=true)


glasso_machine = machine(glasso, X, y)
fit!(glasso_machine)

glasso_machine.model.λ = 0.1

loo_glasso = LooRidgeRegressor(ridge=glasso)
loo_glasso_machine = machine(loo_glasso, X, y)
fit!(loo_glasso_machine)


loo_glasso_machine.report.best_λs




using Plots
evaluate!(single_ridge_machine,  resampling=CV(nfolds=121), measure=rms)

evaluate!(single_ridge_machine,  resampling=resample_ids, measure=rms)
evaluate!(single_ridge_scaled_machine,  resampling=resample_ids, measure=rms)

evaluate!(loo_single_ridge_machine,  resampling=resample_ids, measure=rms)

evaluate!(loo_multi_ridge_machine,  resampling=resample_ids, measure=rms)
evaluate!(loo_multi_ridge_scaled_machine,  resampling=resample_ids, measure=rms)

evaluate!(loo_sigma_ridge_machine,  resampling=resample_ids, measure=rms)

TunedRidgeModel()

tbl_spec = Tabular("lllllll")
line1 = ["", "Tuning", L"\lambda_{\text{Drugs}}",  L"\lambda_{\text{Methylation}}", L"\lambda_{\text{RNA}}", "10-fold RMSE",  "Time (sec)"]
lines  = [line1, Rule(), line_sigma_ridge, line_single_ridge]
lines  = [line1, Rule(), line_multi_ridge]
latex_tabular("cll_analysis.tex", tbl_spec, lines) 


