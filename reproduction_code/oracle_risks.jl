using Polynomials
using Roots
using Optim

using Plots
using FiniteDifferences


tmp = fixed_point([1.0,1.0], [0.5,0.5], [2.0, 2.0])

hs = [1.0;1.0]
γs = [0.25;0.25]
αs = [2.0; 1.0]

λs_opt = γs ./ αs.^2

λs = [2.0; 1.0]


function fixed_point_function(hs, γs, λs)
	γ = sum(γs)
	fixed_point_f = f -> f - sum( γs./γ./(λs./hs .+ 1 ./(1 .+ γ*f))  )
	find_zero(fixed_point_f, (0.0, 100.0))
end


fixed_point_function(hs, γs, λs)

function risk_formula(hs, γs, αs, λs)
	γ = sum(γs)
	fixed_pt = λs_tilde -> fixed_point_function(hs, γs, λs_tilde)
	f = fixed_pt(λs)
	∇f = grad(central_fdm(5, 1), fixed_pt, λs)[1]
	1 +	γ*f + sum(γ ./ γs .* (γs .* λs - αs.^2 .* λs.^2) .* ∇f)
end

function r_squared(hs, γs, αs, λs)
	response_var = 1 + sum(abs2, αs)
	risk = risk_formula(hs, γs, αs, λs)
	1 - risk/response_var
end







function optimal_r_squared(αs, γs, hs)
	λs_opt = γs ./ αs.^2
	r_squared(hs, γs, αs, λs_opt)
end

function optimal_risk(αs, γs, hs)
	λs_opt = min.( γs ./ αs.^2, 10_000)
	risk_formula(hs, γs, αs, λs_opt)
end

function optimal_single_λ_risk(αs, γs, hs)
	λ_opt = sum(γs)/sum(abs2, αs)
	λs_opt = fill(λ_opt, 2)
	risk_formula(hs, γs, αs, λs_opt)
end

function optimal_ignore_second_group_risk(αs, γs, hs)
	λ1_opt = γs[1]*(1+αs[2]^2)/αs[1]^2
	λs_opt = [λ1_opt ; 10_000]
	risk_formula(hs, γs, αs, λs_opt)
end

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