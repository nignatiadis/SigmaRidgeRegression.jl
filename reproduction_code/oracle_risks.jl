using SigmaRidgeRegression
using Plots
using StatsBase
using Statistics
using LaTeXStrings
using Random
using ColorSchemes

# grp = GroupedFeatures(num_groups=2,group_size=200)

# To add to tests.jl
# SigmaRidgeRegression.fixed_point_function(hs, γs, [1.0; Inf])
# SigmaRidgeRegression.risk_formula(hs, γs, αs, [1.0; 10000])





function theoretical_and_realized_mse(γs, αs; n=400, nreps=50, ntest=20_000)
	
	hs = [1.0;1.0] # Identity Spectrum
	
	grp = GroupedFeatures(round.(Int, γs .* n))
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
			risk_empirical[j, i] = mse_ridge(fit(MultiGroupRidgeRegressor(grp, λs), sim_res.X_train, sim_res.Y_train, grp), 
		                                          sim_res.X_test, sim_res.Y_test)
		end 
	end 
	risk_empirical = mean(risk_empirical; dims=1)						   				   		   
	(theoretical = opt_risk_theory, empirical = risk_empirical, all_λs=all_λs)						   
end



function oracle_risk_plot(γs, sum_alpha_squared; ylim=(0,2.5), n=1000, legend=nothing, kwargs...)
	ratio_squared = range(0.0, 1.0, length=30)

	αs_squared = ratio_squared .* sum_alpha_squared
	bs_squared = reverse(ratio_squared) .* sum_alpha_squared

	risks = [theoretical_and_realized_mse(γs, sqrt.([αs_squared[i];bs_squared[i]]); n=n, kwargs...) for i=1:length(ratio_squared)] 
	theoretical_risks = vcat(map(r -> r.theoretical, risks)...) .-1 
	empirical_risks = vcat(map(r -> r.empirical, risks)...) .-1

	labels = [L"Optimal $(\lambda_1, \lambda_2)$" L"Optimal $(\lambda, \lambda)$" L"Optimal $(\lambda, \infty)$"]
	colors = reshape(colorschemes[:seaborn_deep6][1:3],1,3)
	#colors = [:red :blue :purple]
	linestyles= [:solid :solid :solid]
	ylabel = L"Risk $- \sigma^2$"
	xlabel = L"\alpha_1^2/(\alpha_1^2 + \alpha_2^2)"
	pl = plot(ratio_squared, theoretical_risks, color=colors, ylim=ylim, xguide=xlabel, 
			   yguide=ylabel, legend = legend, label=labels, 
			   background_color_legend = :transparent,
			   foreground_color_legend = :transparent,
			   grid = false,
			   thickness_scaling = 1.5)
	plot!(pl, ratio_squared, empirical_risks, seriestype=:scatter, color=colors,
	           markershape =:utriangle, markerstrokealpha=0.0, markersize=4, label=nothing)
	pl
end

Random.seed!(10)

pgfplotsx()
nreps = 1
curve_1 = oracle_risk_plot([0.25,0.25], 1.0, legend=:topleft, nreps=nreps)
title_curve_1 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 1"
plot!(curve_1, title=title_curve_1)

savefig(curve_1, "oracle_risks1.tikz")


curve_2 = oracle_risk_plot([0.1,0.4], 1.0, legend=nothing, nreps=nreps)
title_curve_2 = L"\gamma_1 = \frac{1}{10},\; \gamma_2 = \frac{4}{10},\;\; \alpha_1^2 + \alpha_2^2 = 1"

curve_3 = oracle_risk_plot([1.0,1.0], 1.0, legend=nothing, nreps=nreps)
title_curve_3 = L"\gamma_1 = \gamma_2 = 1,\;\; \alpha_1^2 + \alpha_2^2 = 1"

curve_4 = oracle_risk_plot([0.25,0.25], 2.0, legend=nothing, nreps=nreps)
title_curve_4 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 2"

curve_5 = oracle_risk_plot([0.1,0.4], 2.0, legend=nothing, nreps=nreps)
title_curve_5 = L"\gamma_1 = \frac{1}{10},\; \gamma_2 = \frac{4}{10},\;\; \alpha_1^2 + \alpha_2^2 = 2"

curve_6 = oracle_risk_plot([1.0,1.0], 2.0, legend=nothing, nreps=nreps)
title_curve_6 = L"\gamma_1 = \gamma_2 = 1,\;\; \alpha_1^2 + \alpha_2^2 = 2"


risk_curves = plot(curve_1, curve_2, curve_3, 
                curve_4, curve_5, curve_6,
                title = [title_curve_1 title_curve_2 title_curve_3 title_curve_4 title_curve_5 title_curve_6],
				size=(1000,600))

savefig(risk_curves, "oracle_risks.pdf")



