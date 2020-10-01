using RCall
using SigmaRidgeRegression
using LinearAlgebra
import StatsBase:fit
using Random
using Test
using MLJ


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

group_index = group_expand(groups, Base.OneTo(ngroups(groups)))

R"library(gglasso)"
@rput X
@rput Y
@rput group_index
@rput p
for λ in [0.1; 1.0; 10.0]
	glasso_machine.model.λ = λ
	fit!(glasso_machine)
	R"gglasso_fit <- gglasso(X, Y, group=group_index, lambda=$λ / sqrt(p), intercept=FALSE)"
	R"beta <- gglasso_fit$beta"
	@rget beta
	beta = vec(beta)
	@test group_summary(groups, beta, norm) ≈  group_summary(groups, fitted_params(glasso_machine).fitresult, norm) atol =0.005
	@test norm( beta .- fitted_params(glasso_machine).fitresult, Inf) < 0.005
end


MLJ.predict(glasso_machine)







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




