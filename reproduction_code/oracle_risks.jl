using SigmaRidgeRegression
using Plots
using StatsBase
using Statistics


grp = GroupedFeatures(num_groups=2,group_size=200)





hs = [1.0;1.0] #Spectrum
γs = [0.25;0.25]
αs = [1.0; 2.0]

n = 2000
grp = GroupedFeatures(round.(γs .* n))
design = IdentityCovarianceDesign(grp.p)

ridge_sim= GroupRidgeSimulationSettings(grp = grp, 
                             ntest= n,
                             Σ = design, 
                             response_model = RandomLinearResponseModel(αs = αs, grp=grp))


λs = SigmaRidgeRegression.optimal_λs(γs, αs)  # [1.0; 2.0]

SigmaRidgeRegression.risk_formula(hs, γs, αs, λs)
SigmaRidgeRegression.optimal_risk(hs, γs, αs)

sim_res = simulate(ridge_sim)
ridge_fit = fit(GroupRidgeRegression(tuning=λs), sim_res.X_train, sim_res.Y_train, grp)
mse_ridge(ridge_fit, sim_res.X_test, sim_res.Y_test)


group_summary(grp, sim_res.βs, norm)


1 + sum(abs2, coef(ridge_fit)-sim_res.βs)

cov(sim_res.X_train)

sim_res.

optimal_ignore_second_group_risk([0.5;1.0], γs, hs)
optimal_single_λ_risk([0.5;1.0], γs, hs)
optimal_risk([0.5;1.0], γs, hs)


risk_formula(hs, γs, αs, [1.0;100_000])


optimal_r_squared([0.5;0.5], γs, hs)
 
optimal_single_λ_risk
α1_squared = range(0.0, 4.0, length=100)
α2_squared = range(0.0, 4.0, length=100)

gr()
plotly()
new_surface = Surface((α1,α2) -> optimal_risk(sqrt.([α1;α2]), γs, hs), α1_squared, α2_squared)
new_surface2 = Surface((α1,α2) -> optimal_single_λ_risk(sqrt.([α1;α2]), γs, hs), α1_squared, α2_squared)
new_surface3 = Surface((α1,α2) -> optimal_ignore_second_group_risk(sqrt.([α1;α2]), γs, hs), α1_squared, α2_squared)

plotly()
surface!(α1,α2,new_surface2, linealpha = 0.3)
surface(α1,α2,new_surface, linealpha = 0.3, color=:viridis)
surface!(α1,α2,new_surface3, linealpha = 0.3, color=:blues)

#	surface!(α1,α2,new_surface3, linealpha = 0.3, color=:plasma)
#
gr()
risk_formula(hs, γs, αs, λs)
risk_formula(hs, γs, αs, λs_opt)


f = find_zero( tmp, (0.0, 100.0))

λs_opt


myf = (x,y) -> risk_formula(hs, γs, αs, [x;y])
myf_optim = xs-> risk_formula(hs, γs, αs, xs)


sum_alpha_squared = 1.0
ratio_squared = range(0.0, 1.0, length=100)


αs_squared = ratio_squared .* sum_alpha_squared
bs_squared = reverse(ratio_squared) .* sum_alpha_squared



risk1 =  [optimal_risk(sqrt.([αs_squared[i];bs_squared[i]]), γs, hs) for i=1:length(ratio_squared)] 
risk2 =  [optimal_single_λ_risk(sqrt.([αs_squared[i];bs_squared[i]]), γs, hs) for i=1:length(ratio_squared)] 
risk3 =  [optimal_ignore_second_group_risk(sqrt.([αs_squared[i];bs_squared[i]]), γs, hs) for i=1:length(ratio_squared)] 

plot(ratio_squared, [risk1 risk2 risk3])




myf(0.001,0.001)

risk_formula(hs, γs, αs, [0.25;1])

tmp = optimize(myf_optim, [0.001;0.001], [Inf; Inf], [1.0; 1.5], Fminbox(LBFGS()))
	tmp.minimizer
	
λs_opt

myf(xs[1],ys[1])


myf(λs_opt...)
myf(1.0,1.0)
xs = range(0.001, 0.2, length=100)
ys = range(0.001, 0.4, length=100)


z = Surface(myf, xs, ys)
surface(x,y,z, linealpha = 0.3, color=:plasma)

pgfplotsx()
plotly()