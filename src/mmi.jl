Base.@kwdef mutable struct SingleGroupRidgeRegressor{T} <: AbstractGroupRidgeRegressor    
    decomposition::Symbol = :default  
    λ::T = 0.0
end 

function MMI.fit(m::SingleGroupRidgeRegressor, verb::Int, X, y)
    Xmatrix = MMI.matrix(X)
    p = size(Xmatrix, 2)
    grps = GroupedFeatures([p])
    workspace = StatsBase.fit(m, Xmatrix, y, grps)
    βs = StatsBase.coef(workspace)
    # return
    return βs, workspace, NamedTuple{}()
end

function MMI.update(model::SingleGroupRidgeRegressor, verbosity::Int, old_fitresult, old_cache, X, y)
    new_λ = model.λ
    StatsBase.fit!(old_cache, new_λ)
    βs = StatsBase.coef(old_cache)
    return βs, old_cache, NamedTuple{}()
end

function MMI.predict(model::AbstractGroupRidgeRegressor, fitresult, Xnew)
    MMI.matrix(Xnew)*fitresult
end 

Base.@kwdef mutable struct LooCVRidgeRegressor{G, T} <: AbstractGroupRidgeRegressor
    ridge::G = SingleGroupRidgeRegressor(:cholesky, 1.0)
    n::Int = 100
    λ_min_ratio::Float64 = 1e-6
    λ_max::T = :default
    scale = :log10
end 

function MMI.fit(m::LooCVRidgeRegressor, verb::Int, X, y)
    ridge = m.ridge
    mach = MLJ.machine(ridge, X, y)
    fit!(mach)
    ridge_workspace = mach.cache
    if m.λ_max  == :default 
        λ_max = 100*maximum(abs.(ridge_workspace.XtY))
    end 
    λ_range = range(ridge, :λ, lower=m.λ_min_ratio*λ_max, upper=λ_max, scale=m.scale)
    model_grid = MLJTuning.grid(ridge, [:λ], [MLJ.iterator(λ_range, m.n)])
    history = map(model_grid) do newm
        λ = newm.λ
        mach.model.λ = newm.λ
        fit!(mach)
        loo = loo_error(ridge_workspace)
        (loo = loo, λ=λ, model=newm)
    end
    loos = [h.loo for h in history]
    λs = [h.λ for h in history]
    best_model_idx = argmin(loos)
    best_λ = model_grid[best_model_idx].λ
    mach.model.λ = best_λ

    report = (best_model =  model_grid[best_model_idx],
              best_λ = best_λ,
              loos = loos,
              λs = λs,
              λ_max = λ_max)
    fit!(mach)
    βs = StatsBase.coef(ridge_workspace)
    # return
    return βs, ridge_workspace, report
end