abstract type  FixedLambdaGroupRidgeRegressor <: AbstractGroupRidgeRegressor end

"""
    SingleGroupRidgeRegressor(; decomposition, λ, groups)
"""
Base.@kwdef mutable struct SingleGroupRidgeRegressor{T, G} <: FixedLambdaGroupRidgeRegressor    
    decomposition::Symbol = :default  
    λ::T = 1.0
    groups::G = nothing
    center::Bool = false
    scale::Bool = false
end 

_main_hyperparameter(::SingleGroupRidgeRegressor) = :λ
_main_hyperparameter_value(m) = getproperty(m, _main_hyperparameter(m))

function _groups(m::SingleGroupRidgeRegressor, p) 
    isnothing(m.groups) ? GroupedFeatures([p]) : m.groups
end

function MMI.fit(m::FixedLambdaGroupRidgeRegressor, verb::Int, X, y)
    @unpack center, scale, groups = m
    Xmatrix = MMI.matrix(X)
    if center || scale
        x_transform = StatsBase.fit(ZScoreTransform, Xmatrix; dims=1, center=center, scale=scale)
        y_transform = StatsBase.fit(ZScoreTransform, y; dims=1, center=center, scale=scale)
        Xmatrix = StatsBase.transform(x_transform, Xmatrix)
        y = StatsBase.transform(y_transform, y)
    else 
        x_transform = nothing
        y_transform = nothing
    end 
    p = size(Xmatrix, 2)
    groups = _groups(m, p)
    workspace = StatsBase.fit(m, Xmatrix, y, groups)  # see end_to_end.jl
    βs = StatsBase.coef(workspace)
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)
    # return
    return fitresult, workspace, NamedTuple{}()
end

function MMI.update(model::AbstractGroupRidgeRegressor, verbosity::Int, old_fitresult, old_cache, X, y)
    new_λ = model.λ
    StatsBase.fit!(old_cache, new_λ)
    βs = StatsBase.coef(old_cache)
    fitresult = (coef = βs, x_transform = old_fitresult.x_transform, y_transform = old_fitresult.y_transform)
    return fitresult, old_cache, NamedTuple{}()
end

function MMI.predict(model::AbstractGroupRegressor, fitresult, Xnew)
    Xnew = MMI.matrix(Xnew)
    @unpack coef, x_transform, y_transform = fitresult
    !isnothing(x_transform) && (Xnew = StatsBase.transform(x_transform, Xnew))
    ypred = Xnew*coef
    !isnothing(y_transform) && StatsBase.reconstruct!(y_transform, ypred)
    ypred
end 


function range_and_grid(ridge::SingleGroupRidgeRegressor, λ_min, λ_max, scale, n)
    λ_range = range(ridge, :λ, lower=λ_min, upper=λ_max, scale=scale)
    model_grid = MLJTuning.grid(ridge, [:λ], [MLJ.iterator(λ_range, n)])
    λ_range, model_grid
end



"""
    MultiGroupRidgeRegressor(; decomposition, λ, groups)
"""
mutable struct MultiGroupRidgeRegressor{T, G<:GroupedFeatures} <: FixedLambdaGroupRidgeRegressor    
    decomposition::Symbol
    λ::T   #Named tuple 
    groups::G
    center::Bool
    scale::Bool
end 

_main_hyperparameter(::MultiGroupRidgeRegressor) = :λ
_groups(m::MultiGroupRidgeRegressor, p) = m.groups

function MultiGroupRidgeRegressor(groups::GroupedFeatures; kwargs...)
    ngr = ngroups(groups)
    MultiGroupRidgeRegressor(groups, ones(ngr); kwargs...)
end

function MultiGroupRidgeRegressor(groups::GroupedFeatures, λs::AbstractVector; 
                                  decomposition=:default, center=false, scale=false)
    ngr = ngroups(groups)
    λ_expr = Tuple(Symbol.(:λ, Base.OneTo(ngr)))
    λ_tupl = MutableNamedTuple{λ_expr}(tuple(λs...))
    MultiGroupRidgeRegressor(decomposition, λ_tupl, groups)
end


function range_and_grid(ridge::MultiGroupRidgeRegressor, λ_min, λ_max, scale, n)
    λ_names = [Meta.parse("(λ.$λ)") for λ in keys(ridge.λ)]
    λ_range = [range(ridge, λ, lower=λ_min, upper=λ_max, scale=scale) for λ in λ_names]
    model_grid = MLJTuning.grid(ridge, λ_names, MLJ.iterator.(λ_range, n))
    λ_range, model_grid
end




# Autotuning code

"""
Use leave-one-out-cross-validation to choose over
group-specific Ridge-penalty matrices
of the form, i.e., over ``λ \\in (0,∞)^K``.
"""
Base.@kwdef mutable struct LooCVRidgeRegressor{G, T} <: AbstractGroupRidgeRegressor
    ridge::G = SingleGroupRidgeRegressor(decomposition=:cholesky, λ = 1.0)
    n::Int = 100
    λ_min_ratio::Float64 = 1e-6
    λ_max::T = :default
    scale = :log10
end 

function MMI.fit(m::LooCVRidgeRegressor, verb::Int, X, y)
    ridge = m.ridge
    mach = MLJ.machine(ridge, X, y)
    fit!(mach)
    x_transform = mach.fitresult.x_transform
    y_transform = mach.fitresult.y_transform

    ridge_workspace = mach.cache
    if m.λ_max  == :default 
        λ_max = 100*maximum(abs.(ridge_workspace.XtY))
    end 
    λ_min = m.λ_min_ratio*λ_max
    λ_range, model_grid = range_and_grid(ridge, λ_min, λ_max, m.scale, m.n)

    history = map(model_grid) do newm
        λ = newm.λ
        mach.model.λ = newm.λ
        fit!(mach; verbosity=0)
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
              λ_max = λ_max,
              λ_range = λ_range)
    fit!(mach)
    βs = StatsBase.coef(ridge_workspace)
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)
    # return
    return fitresult, ridge_workspace, report
end