using RCall
using SigmaRidgeRegression
using StatsBase
using LinearAlgebra
import StatsBase:fit

Base.@kwdef struct GroupLasso{T,P,S}
	decomposition::Symbol = :default
	grp::GroupedFeatures
	grp_multiplier::P = sqrt.(grp.ps)
	λ::T
 	η::S 
	η_threshold::S
end


ps = fill(50, 10)
n = 200
grp = GroupedFeatures(ps)
design = IdentityCovarianceDesign(grp.p)
αs = vcat(fill(0.0, 5), fill(3.0,5))
ridge_sim= GroupRidgeSimulationSettings(grp = grp, 
							 ntrain= n,
							 Σ = design, 
							 response_model = RandomLinearResponseModel(αs = αs, grp=grp))
sim_res = simulate(ridge_sim)



glasso = GroupLasso(grp=grp, λ=0.1, η=1e-7, η_threshold =0.0)
	
gridge = GroupRidgeRegression(tuning = glasso.grp_multiplier * glasso.λ)
	
rdg_workspace = fit(gridge, sim_res.X_train, sim_res.Y_train, sim_res.grp)
	
ηs = group_summary(grp, coef(rdg_workspace), norm)


for i=1:500
	ηs = group_summary(grp, coef(rdg_workspace), norm)
	new_λs = glasso.λ .* glasso.grp_multiplier ./ sqrt.( ηs.^2 .+ glasso.η)
	fit!(rdg_workspace, new_λs)
end 

ηs 

describe(coef(rdg_workspace))

ηs = group_summary(grp, coef(rdg_workspace), norm)



group_index = repeat(1:10, inner= 50)

R"library(gglasso)"
X = sim_res.X_train
Y = sim_res.Y_train
@rput X
@rput Y
@rput group_index

R"gglasso_fit <- gglasso(X, Y, group=group_index, lambda=0.1, intercept=FALSE)"

R"beta <- gglasso_fit$beta"
@rget beta
beta = vec(beta)

maximum(abs.( group_summary(grp, beta, norm) .-  group_summary(grp, coef(rdg_workspace), norm)))

norm(beta)

norm(beta .- coef(rdg_workspace), Inf)/norm(beta)



group_summary(grp, beta, norm)

rdg_workspace.λs
R"summary(gglasso_fit)"

R"gglasso_fit$jerr"


