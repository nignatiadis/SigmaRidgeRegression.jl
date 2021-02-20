using Pkg
Pkg.activate(@__DIR__)
using SigmaRidgeRegression
using StatsBase
using Plots
using Random
using MLJ
using Tables
using LaTeXTabulars
using LaTeXStrings
using BenchmarkTools

# Workaround until the following is resolved
# https://github.com/alan-turing-institute/MLJBase.jl/issues/428
# Else the session will hang upon printing the X matrix
# and wrapping it as Tables.MatrixTable
function Base.show(io::IO, tbl::Tables.MatrixTable)
    return Base.show(io, typeof(tbl))
end

# Load CLL dataset
cll_data = CLLData.load();
foldid = cll_data.foldid;
X = cll_data.X;
y = cll_data.y;
ngr = cll_data.ngr

resample_ids = [(findall(foldid .!= k), findall(foldid .== k)) for k in 1:10]

groups = GroupedFeatures(ngr)

Random.seed!(1)
rand_idx = sample(1:121, 121; replace=false)
Xdrug_resample = X[rand_idx, 1:305]
Xgaussian = randn(121, 100)
Xnoise = [X Xdrug_resample Xgaussian]

groups_noise = GroupedFeatures([ngr; 305; 100])

X_table = MLJ.table(X)
X_plus_noise = MLJ.table(Xnoise)

function single_table_line(
    X, y, resampling, _model, model_name; tuning_name=nothing, sigdigits=3
)
    ridge_machine = machine(_model, X, y)
    fit!(ridge_machine)
    best_param = ridge_machine.report.best_param
    if isnothing(tuning_name)
        tuning_string = ""
    else
        best_param = round(best_param; sigdigits=sigdigits)
        tuning_string = L"%$(tuning_name) = %$(best_param)"
    end

    λs = round.(deepcopy(ridge_machine.report.best_λs), sigdigits=sigdigits)

    ridge_benchmark = @benchmark fit!(machine($(_model), $(X), $(y)))

    time_loo = round(mean(ridge_benchmark.times) / (1e9); sigdigits=sigdigits)

    eval_ridge = evaluate!(ridge_machine; resampling=resampling, measure=rms)
    _rmse = round(eval_ridge.measurement[1]; sigdigits=sigdigits)

    return [model_name, tuning_string, λs..., time_loo, _rmse]
end

loo_single_ridge = LooRidgeRegressor(;
    ridge=SingleGroupRidgeRegressor(; groups=groups, center=true, scale=true)
)
loo_single_ridge_noise = LooRidgeRegressor(;
    ridge=SingleGroupRidgeRegressor(; groups=groups_noise, center=true, scale=true)
)

loo_sigmaridge = LooSigmaRidgeRegressor(; groups=groups, center=true, scale=true)
loo_sigmaridge_noise = LooSigmaRidgeRegressor(;
    groups=groups_noise, center=true, scale=true
)

loo_multi_ridge = LooRidgeRegressor(;
    ridge=MultiGroupRidgeRegressor(; groups=groups, center=true, scale=true),
    rng=MersenneTwister(1),
)
loo_multi_ridge_noise = LooRidgeRegressor(;
    ridge=MultiGroupRidgeRegressor(; groups=groups_noise, center=true, scale=true),
    rng=MersenneTwister(1),
)

cv_glasso = TunedRidgeRegressor(;
    ridge=GroupLassoRegressor(; groups=groups, center=true, scale=true),
    resampling=Holdout(; shuffle=true, rng=1),
)
cv_glasso_noise = TunedRidgeRegressor(;
    ridge=GroupLassoRegressor(; groups=groups_noise, center=true, scale=true),
    resampling=Holdout(; shuffle=true, rng=1),
)

line_single_ridge = single_table_line(
    X_table, y, resample_ids, loo_single_ridge, "\textbf{Single Ridge}"
)
line_single_ridge_noise = single_table_line(
    X_plus_noise, y, resample_ids, loo_single_ridge_noise, "\textbf{Single Ridge}"
)

line_sigma_ridge = single_table_line(
    X_table,
    y,
    resample_ids,
    loo_sigmaridge,
    L"$\sigmacv$\textbf{-Ridge}";
    tuning_name=L"\sigmacv",
)
line_sigma_ridge_noise = single_table_line(
    X_plus_noise,
    y,
    resample_ids,
    loo_sigmaridge_noise,
    L"$\sigmacv$\textbf{-Ridge}";
    tuning_name=L"\sigmacv",
)

line_multi_ridge = single_table_line(
    X_table, y, resample_ids, loo_multi_ridge, "\textbf{Multi Ridge}"
)
line_multi_ridge_noise = single_table_line(
    X_plus_noise, y, resample_ids, loo_multi_ridge_noise, "\textbf{Multi Ridge}"
)

line_glasso = single_table_line(
    X_table,
    y,
    resample_ids,
    cv_glasso,
    L"\textbf{Group Lasso}";
    tuning_name=L"$\widehat{\lambda}^{gl}",
)
line_glasso_noise = single_table_line(
    X_plus_noise,
    y,
    resample_ids,
    cv_glasso_noise,
    L"\textbf{Group Lasso}";
    tuning_name=L"$\widehat{\lambda}^{gl}",
)

tbl_spec = Tabular("lllllll")
line1 = [
    "",
    "Tuning",
    L"$\widehat{\lambda}_{\text{Drugs}}$",
    L"\widehat{\lambda}_{\text{Methyl}}",
    L"\widehat{\lambda}_{\text{RNA}}",
    "Time (s)",
    "10-fold RMSE",
]
lines = [line1, Rule(), line_sigma_ridge, line_single_ridge, line_multi_ridge, line_glasso]
latex_tabular("cll_analysis.tex", tbl_spec, lines)

tbl_spec2 = Tabular("lllllllll")
line1 = [
    "",
    "Tuning",
    L"$\widehat{\lambda}_{\text{Drugs}}$",
    L"\widehat{\lambda}_{\text{Methyl}}",
    L"\widehat{\lambda}_{\text{RNA}}",
    L"\widehat{\lambda}_{\text{Noise1}}",
    L"\widehat{\lambda}_{\text{Noise2}}",
    "Time (s)",
    "10-fold RMSE",
]
lines = [
    line1,
    Rule(),
    line_sigma_ridge_noise,
    line_single_ridge_noise,
    line_multi_ridge_noise,
    line_glasso_noise,
]
latex_tabular("cll_analysis_noise.tex", tbl_spec2, lines)
