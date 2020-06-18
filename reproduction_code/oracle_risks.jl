using SigmaRidgeRegression
using Plots
using StatsBase
using Statistics
using LaTeXStrings
using Random

# grp = GroupedFeatures(num_groups=2,group_size=200)

# To add to tests.jl
# SigmaRidgeRegression.fixed_point_function(hs, γs, [1.0; Inf])
# SigmaRidgeRegression.risk_formula(hs, γs, αs, [1.0; 10000])





function theoretical_and_realized_mse(γs, αs; n=400, nreps=50, ntest=10_000)
	
	hs = [1.0;1.0] # Identity Spectrum
	
	grp = GroupedFeatures(round.(γs .* n))
	design = IdentityCovarianceDesign(grp.p)



							 
	λs1 = SigmaRidgeRegression.optimal_λs(γs, αs)	
	λs2 = SigmaRidgeRegression.optimal_single_λ(γs, αs)								   
	λs3 = SigmaRidgeRegression.optimal_ignore_second_group_λs(γs, αs)								   

	all_λs = (λs1, λs2, λs3)
	opt_risk_theory = Matrix{Float64}(undef, 1, length(all_λs)) 
	risk_empirical = Matrix{Float64}(undef, nreps, length(all_λs)) 

	for (i, λs) in enumerate(all_λs)						 	
		opt_risk_theory[1,i] = SigmaRidgeRegression.risk_formula(hs, γs, αs, λs)
	end 
	
	for j in 1:nreps
		ridge_sim= GroupRidgeSimulationSettings(grp = grp, 
									 ntrain= n,
									 ntest = ntest,
									 Σ = design, 
									 response_model = RandomLinearResponseModel(αs = αs, grp=grp))
		sim_res = simulate(ridge_sim)
		
		for (i, λs) in enumerate(all_λs)			
			risk_empirical[j, i] = mse_ridge(fit(GroupRidgeRegression(tuning=λs), sim_res.X_train, sim_res.Y_train, grp), 
		                                          sim_res.X_test, sim_res.Y_test)
		end 
	end 
	risk_empirical = mean(risk_empirical; dims=1)						   				   		   
	(theoretical = opt_risk_theory, empirical = risk_empirical, all_λs=all_λs)						   
end



function oracle_risk_plot(γs, sum_alpha_squared; ylim=(0,1.5), legend=nothing, kwargs...)
	ratio_squared = range(0.0, 1.0, length=30)

	αs_squared = ratio_squared .* sum_alpha_squared
	bs_squared = reverse(ratio_squared) .* sum_alpha_squared

	risks = [theoretical_and_realized_mse(γs, sqrt.([αs_squared[i];bs_squared[i]]); n=400, kwargs...) for i=1:length(ratio_squared)] 
	theoretical_risks = vcat(map(r -> r.theoretical, risks)...) .-1 
	empirical_risks = vcat(map(r -> r.empirical, risks)...) .-1

	labels = [""]
	colors = [:black :purple :green] #[:red :blue :purple]
	ylabel = L"Risk $- \sigma^2$"
	xlabel = L"\alpha_1^2/(\alpha_1^2 + \alpha_2^2)"
	pl = plot(ratio_squared, theoretical_risks, color=colors, ylim=ylim, xguide=xlabel, 
	           yguide=ylabel, legend = legend)
	plot!(pl, ratio_squared, empirical_risks, seriestype=:scatter, color=colors,
	           markershape =:utriangle, markerstrokealpha=0.0, markersize=4, label=nothing)
	pl
end

curve_1 = oracle_risk_plot([0.25,0.25], 1.0, legend=:topright, nreps=2)

plot(curve_1, legend=nothing)

;


using Plots.Measures
plot(curve_1, deepcopy(curve_1), size=(1200,400))
tmp_plot = plot(curve_1, deepcopy(curve_1), deepcopy(curve_1), deepcopy(curve_1), 
              deepcopy(curve_1), deepcopy(curve_1), deepcopy(curve_1), deepcopy(curve_1),
			  size=(1800,400),
              title = fill(title_curve_1, 1, 8))
				
tmp_plot = plot(curve_1, deepcopy(curve_1), deepcopy(curve_1), 
                deepcopy(curve_1), deepcopy(curve_1), deepcopy(curve_1), 
				title = fill(title_curve_1, 1, 6),
				size=(1400,750), bottom_margin= 6mm)

title_curve_1 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 1"
			    #)
savefig(tmp_plot, "oracle_risks.tex")
savefig(tmp_plot, "oracle_risks.pdf")
			  				
			  	
pgfplotsx()				
					
plot(curve_1, deepcopy(curve_1))				
tmp_plot



plot(curve_1, legend=:bottomright)
plot(curve_1, title=title_curve_1)


plot(1:10,[1:10 2:11], color=[:red :blue])
plot!(1:10,[1:10 2:11], color=[:red :blue], seriestype=:scatter,



pgfplotsx()


oracle_risk_plot([0.4,0.1], 1.0)

oracle_risk_plot([0.1,0.4], 1.0)


curve_1 = oracle_risk_plot([0.25,0.25], 1.0)
curve_4 = oracle_risk_plot([1.0,1.0], 1.0)

curve_8 = oracle_risk_plot([1.0,1.0], 0.5)

pgfplotsx()
plot(x=1:10,y=1:10)
plot(curve_1)


plot(curve_1, curve_4, curve_8, ylim=(1.0, 1.6))

tmp = theoretical_and_realized_mse([0.25;0.25], [0.5;2.0])






pl_both_tree = plot(pl_left_tree,pl_right_tree, 
                title=["(a)" "(b)"], size=(550,220));


#λs = SigmaRidgeRegression.optimal_λs(γs, αs)  # [1.0; 2.0]
λs = [1.0; 2.0]

SigmaRidgeRegression.optimal_risk(hs, γs, αs)

ridge_fit = fit(GroupRidgeRegression(tuning=λs), sim_res.X_train, sim_res.Y_train, grp)
mse_ridge(ridge_fit, sim_res.X_test, sim_res.Y_test)


ws = barebones_ridge(sim_res.X_train, sim_res.Y_train, λs, grp)


ws = barebones_ridge(sim_res.X_train, sim_res.Y_train, λs, grp)

ws = barebones_ridge(sim_res.X_train, sim_res.Y_train, λs, grp)


maximum(abs.(ws .- ridge_fit.β_curr))


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