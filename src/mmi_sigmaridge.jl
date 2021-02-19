"""
    SigmaRidgeRegressor(; decomposition, groups, σ, center, scale)

A MLJ model that fits σ-Ridge Regression with `groups` and parameter `σ`.
`center` and `scale` (default `true` for both) control whether the response and
features should be centered and scaled first (make sure that `center=true` if the
model is supposed to have an intercept!). `decomposition` can be one of `:default`,
`:cholesky` or `:woodbury` and determines how the linear system is solved.
"""
Base.@kwdef mutable struct SigmaRidgeRegressor{G,T,M} <:
                           SigmaRidgeRegression.AbstractGroupRidgeRegressor
    decomposition::Symbol = :default
    groups::G
    σ::T = 1.0
    center::Bool = true
    scale::Bool = true
    init_model::M = SingleGroupRidgeRegressor(
        decomposition = decomposition,
        groups = groups,
        center = center,
        scale = scale,
    )
end

_main_hyperparameter(::SigmaRidgeRegressor) = :σ

function _default_hyperparameter_maximum(model::SigmaRidgeRegressor, fitted_machine)
    sqrt(σ_squared_max(fitted_machine.cache.mom))
end

_default_param_min_ratio(::SigmaRidgeRegressor, fitted_machine) = 1e-3
_default_scale(::SigmaRidgeRegressor, fitted_machine)  = :linear


function MMI.fit(m::SigmaRidgeRegressor, verb::Int, X, y)
    @unpack init_model, decomposition, center, scale, groups = m
    init_machine = MLJ.machine(init_model, X, y)
    fit!(init_machine; verbosity = verb)
    mom = MomentTunerSetup(init_machine.cache)

    σ = m.σ
    λs = SigmaRidgeRegression.get_λs(mom, abs2(σ))

    multiridge = MultiGroupRidgeRegressor(
        groups,
        λs;
        decomposition = decomposition,
        center = center,
        scale = scale,
    )
    multiridge_machine = MLJ.machine(multiridge, X, y)
    fit!(multiridge_machine; verbosity = verb)

    workspace = multiridge_machine.cache
    cache = (workspace = workspace, mom = mom, multiridge_machine = multiridge_machine)

    βs = StatsBase.coef(workspace)
    x_transform = multiridge_machine.fitresult.x_transform
    y_transform = multiridge_machine.fitresult.y_transform
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)

    # return
    return fitresult, cache, NamedTuple{}()
end


function MMI.update(
    model::SigmaRidgeRegressor,
    verbosity::Int,
    old_fitresult,
    old_cache,
    X,
    y,
)
    @unpack init_model, decomposition, center, scale, groups = model

    workspace = old_cache.workspace
    multiridge_machine = old_cache.multiridge_machine
    mom = old_cache.mom

    σ = model.σ
    λs = SigmaRidgeRegression.get_λs(mom, abs2(σ))
    multiridge = MultiGroupRidgeRegressor(
        groups,
        λs;
        decomposition = decomposition,
        center = center,
        scale = scale,
    )
    multiridge_machine.model = multiridge
    fit!(multiridge_machine; verbosity = verbosity)

    cache = (workspace = workspace, mom = mom, multiridge_machine = multiridge_machine)

    βs = StatsBase.coef(workspace)
    x_transform = multiridge_machine.fitresult.x_transform
    y_transform = multiridge_machine.fitresult.y_transform
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)
    return fitresult, cache, NamedTuple{}()
end

const LooSigmaRidgeRegressor =  LooRidgeRegressor{<:SigmaRidgeRegressor}

function LooSigmaRidgeRegressor(; kwargs...)
    sigma_ridge = SigmaRidgeRegressor(;kwargs...)
    LooRidgeRegressor(ridge=deepcopy(sigma_ridge))
end
