using RCall
using SigmaRidgeRegression
using LinearAlgebra
using StatsBase
using Random
using Test
using MLJ
using Plots



Random.seed!(1)
n = 200
p = 160
X = randn(n, p)
Xtable = MLJ.table(X);
βs = randn(p)./sqrt(p)
Y = X*βs .+ randn(n)
groups = GroupedFeatures([60;60;40])

_scale = true
_center = true
_param_min_ratio = 0.9
seagull_glasso = TunedSeagull(groups=groups, scale=_scale, center=_center,
                      param_min_ratio=_param_min_ratio)
seagull_mach = machine(seagull_glasso, X, Y)
fit!(seagull_mach)

glasso = GroupLassoRegressor(groups=groups,λ=0.2, scale=_scale, center=_center)
glasso_machine = machine(glasso, Xtable, Y)
fit!(glasso_machine)
λ_max =	SigmaRidgeRegression._default_hyperparameter_maximum(glasso, glasso_machine)

@test seagull_mach.report.λ_max ≈ λ_max atol=0.02

λs_opt = seagull_mach.report.best_λs

multiridge = MultiGroupRidgeRegressor(groups=groups, λs=λs_opt, scale=_scale, center=_center)
multiridge_mach = machine(multiridge, X, Y)
fit!(multiridge_mach)

@test multiridge_mach.fitresult.coef ≈ seagull_mach.fitresult.coef atol=1e-3

group_summary(groups, multiridge_mach.fitresult.coef, norm)
group_summary(groups, seagull_mach.fitresult.coef, norm)



@test MLJ.predict(seagull_mach, Xtable) ≈ MLJ.predict(multiridge_mach, Xtable) atol=0.0001
