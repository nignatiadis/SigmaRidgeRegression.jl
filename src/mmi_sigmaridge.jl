
Base.@kwdef mutable struct SigmaRidgeRegressor{G, T, M} <: SigmaRidgeRegression.AbstractGroupRidgeRegressor
    decomposition::Symbol = :default
    groups::G
    σ::T = 1.0
    init_model::M = LooCVRidgeRegressor(ridge=SingleGroupRidgeRegressor(decomposition=decomposition, groups=groups))
end 

_main_hyperparameter(::SigmaRidgeRegressor) = :σ


function MMI.fit(m::SigmaRidgeRegressor, verb::Int, X, y)
    init_machine = machine(m.init_model, X, y)
    fit!(init_machine)
    mom = MomentTunerSetup(init_machine.cache)

    σ = m.σ
    λs = SigmaRidgeRegression.get_λs(mom, abs2(σ))

    multiridge = MultiGroupRidgeRegressor(m.groups, λs; decomposition=m.decomposition)
    multiridge_machine = machine(multiridge, X, y)
    fit!(multiridge_machine)
    
    workspace = multiridge_machine.cache
    cache = (workspace=workspace, mom=mom, multiridge_machine=multiridge_machine)
    
    βs = StatsBase.coef(workspace)
    # return
    return βs, cache, NamedTuple{}()
end


function MMI.update(model::SigmaRidgeRegressor, verbosity::Int, old_fitresult, old_cache, X, y)
    workspace = old_cache.workspace
    multiridge_machine = old_cache.multiridge_machine
    mom = old_cache.mom

    σ = model.σ
    λs = SigmaRidgeRegression.get_λs(mom, abs2(σ))

    cache = (workspace=workspace, mom=mom, multiridge_machine=multiridge_machine)
    βs = StatsBase.coef(workspace)
    return βs, cache, NamedTuple{}()
end