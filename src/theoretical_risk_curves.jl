
function fixed_point_function(hs, γs, λs)
	γ = sum(γs)
	fixed_point_f = f -> f - sum( γs./γ./(λs./hs .+ 1 ./(1 .+ γ*f))  )
	find_zero(fixed_point_f, (0.0, 100.0))
end


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