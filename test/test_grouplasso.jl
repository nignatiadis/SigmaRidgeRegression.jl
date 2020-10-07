using RCall
using SigmaRidgeRegression
using LinearAlgebra
import StatsBase:fit
using Random
using Test
using MLJ
using Plots

Random.seed!(1)
n = 100
p = 80
X = randn(n, p)
Xtable = MLJ.table(X);
βs = randn(p)./sqrt(p)
Y = X*βs .+ randn(n)
groups = GroupedFeatures([30;30;20])

glasso = GroupLassoRegressor(groups=groups)


glasso_machine = machine(glasso, X, Y)
fit!(glasso_machine)
λ_max =	SigmaRidgeRegression._default_hyperparameter_maximum(glasso, glasso_machine)

glasso_machine.model.λ = λ_max
fit!(glasso_machine)
group_summary(groups, glasso_machine.fitresult.coef, norm)


group_index = group_expand(groups, Base.OneTo(ngroups(groups)))

R"library(gglasso)"
@rput X
@rput Y
@rput group_index
@rput p

R"gglasso_fit_all <- gglasso(X, Y, group=group_index,  intercept=FALSE)"
R"lambda_max <- max(gglasso_fit_all$lambda)"
@rget lambda_max
@test lambda_max .*sqrt(p) ≈ λ_max

for λ in [0.001; 0.1; 0.5]
	glasso_machine.model.λ = λ
	fit!(glasso_machine)
	R"gglasso_fit <- gglasso(X, Y, group=group_index, lambda=$λ / sqrt(p), intercept=FALSE)"
	R"beta <- gglasso_fit$beta"
	@rget beta
	beta = vec(beta)
	@test group_summary(groups, beta, norm) ≈  group_summary(groups, fitted_params(glasso_machine).fitresult.coef, norm) atol =0.005
	@test norm( beta .- fitted_params(glasso_machine).fitresult.coef, Inf) < 0.005
end


MLJ.predict(glasso_machine)

# Now check CVGGLasso code

cvgglasso = CVGGLassoRegressor(groups=groups)
cvgglasso_machine = machine(cvgglasso, X, Y)
fit!(cvgglasso_machine)

multiridge = MultiGroupRidgeRegressor(groups, cvgglasso_machine.report.best_λs)
multiridge_machine = machine(multiridge, X, Y)
fit!(multiridge_machine)

@test predict(multiridge_machine) ≈ predict(cvgglasso_machine) atol =0.01
new_X = randn(2,p)
@test predict(multiridge_machine, new_X) ≈ predict(cvgglasso_machine, new_X) atol =0.01

loo_glasso = LooRidgeRegressor(ridge = glasso)
loo_glasso_machine = machine(loo_glasso, X, Y)
fit!(loo_glasso_machine)

loo_list = loo_glasso_machine.report.loos
λ = loo_glasso_machine.report.params

using Plots
plot(λ,loo_list, xscale=:log10)

λ_path  =  vcat(loo_glasso_machine.report.λs'...)
plot(λ , λ_path, xscale=:log10, yscale=:log10)


#ps = fill(50, 10)
#n = 200
#grp = GroupedFeatures(ps)
#design = IdentityCovarianceDesign(grp.p)
#αs = vcat(fill(0.0, 5), fill(3.0,5))
#ridge_sim= GroupRidgeSimulationSettings(grp = grp,
#							 ntrain= n,
#							 Σ = design,
#							 response_model = RandomLinearResponseModel(αs = αs, grp=grp))
#sim_res = simulate(ridge_sim)
