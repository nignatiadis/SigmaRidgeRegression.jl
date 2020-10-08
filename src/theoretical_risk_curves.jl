function _integrate_spectrum(h::Float64, γ, λ, f) #interpreted as point mass spectrum
    denom = (λ / h + 1 / (1 + γ * f))
    1 / denom
end

#interpreted as
function _integrate_spectrum(h::Distribution, γ, λ, f)
    expectation(u->_integrate_spectrum(u, γ, λ, f), h)
end

function fixed_point_function(hs, γs, λs)
    γ = sum(γs)
    fixed_point_f = f -> f - sum(γs ./ γ .* _integrate_spectrum.(hs, γ, λs, f))
    find_zero(fixed_point_f, (0.0, 100.0))
end


function risk_formula(hs, γs, αs, λs)
    λs = min.(λs, 10_000) #hack for now until properly dealing with Infinity
    γ = sum(γs)
    fixed_pt = λs_tilde -> fixed_point_function(hs, γs, λs_tilde)
    f = fixed_pt(λs)
    ∇f = grad(central_fdm(5, 1), fixed_pt, λs)[1]
    #return ∇f
    #return γ ./ γs .* (γs .* λs - αs.^2 .* λs.^2) .* ∇f
    1 + γ * f + sum(γ ./ γs .* (γs .* λs - αs .^ 2 .* λs .^ 2) .* ∇f)
end

function r_squared(hs, γs, αs, λs)
    response_var = 1 + sum(abs2, αs)
    risk = risk_formula(hs, γs, αs, λs)
    1 - risk / response_var
end


function optimal_r_squared(αs, γs, hs)
    λs_opt = γs ./ αs .^ 2
    r_squared(hs, γs, αs, λs_opt)
end


function optimal_λs(γs, αs)
    γs ./ αs .^ 2
end

function optimal_risk(hs, γs, αs)
    λs_opt = optimal_λs(γs, αs)
    risk_formula(hs, γs, αs, λs_opt)
end

function optimal_single_λ(γs, αs)
    λ_opt = sum(γs) / sum(abs2, αs)
    λs_opt = fill(λ_opt, 2)
end

function optimal_single_λ_risk(hs, γs, αs)
    λs_opt = optimal_single_λ(γs, αs)
    risk_formula(hs, γs, αs, λs_opt)
end

function optimal_ignore_second_group_λs(γs, αs)
    λ1_opt = γs[1] * (1 + αs[2]^2) / αs[1]^2
    λs_opt = [λ1_opt; Inf]
end

function optimal_ignore_second_group_risk(hs, γs, αs)
    λs_opt = optimal_ignore_second_group_λs(γs, αs)
    risk_formula(hs, γs, αs, λs_opt)
end
