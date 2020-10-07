Base.@kwdef mutable struct SigmaRidgeRegressor{G,T,M} <:
                           SigmaRidgeRegression.AbstractGroupRidgeRegressor
    decomposition::Symbol = :default
    groups::G
    σ::T = 1.0
    center::Bool = false
    scale::Bool = false
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

    # return
    return fitresult, cache, NamedTuple{}()
end
