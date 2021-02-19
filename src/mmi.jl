abstract type FixedLambdaGroupRidgeRegressor <: AbstractGroupRidgeRegressor end

"""
    SingleGroupRidgeRegressor(; λ,
                                decomposition = :default,
                                center = true,
                                scale = true)

Type representing vanilla Ridge regression with hyperparameter `λ`.
`center` and `scale` (default `true` for both) control whether the response and
features should be centered and scaled first (make sure that `center=true` if the
model is supposed to have an intercept!). `decomposition` can be one of `:default`,
`:cholesky` or `:woodbury` and determines how the linear system is solved.
"""
Base.@kwdef mutable struct SingleGroupRidgeRegressor{T,G} <: FixedLambdaGroupRidgeRegressor
    decomposition::Symbol = :default
    λ::T = 1.0
    groups::G = nothing
    center::Bool = true
    scale::Bool = true
end

_main_hyperparameter(::SingleGroupRidgeRegressor) = :λ
_main_hyperparameter_value(m) = getproperty(m, _main_hyperparameter(m))

function _default_hyperparameter_maximum(
    model::FixedLambdaGroupRidgeRegressor,
    fitted_machine,
)
    1000 * maximum(abs.(fitted_machine.cache.XtY))
end

function _default_param_min_ratio(ridge, fitted_machine)
    1e-6
end

function _default_scale(ridge, fitted_machine)
    :log10
end

function _groups(m::SingleGroupRidgeRegressor, p)
    isnothing(m.groups) ? GroupedFeatures([p]) : m.groups
end

function MMI.fit(m::FixedLambdaGroupRidgeRegressor, verb::Int, X, y)
    @unpack center, scale, groups = m
    Xmatrix = MMI.matrix(X)
    if center || scale
        x_transform = StatsBase.fit(
            ZScoreTransform,
            Xmatrix;
            dims = 1,
            center = center,
            scale = scale,
        )
        y_transform =
            StatsBase.fit(ZScoreTransform, y; dims = 1, center = center, scale = scale)
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

function MMI.update(
    model::AbstractGroupRidgeRegressor,
    verbosity::Int,
    old_fitresult,
    old_cache,
    X,
    y,
)
    new_λ = _main_hyperparameter_value(model)
    StatsBase.fit!(old_cache, new_λ)
    βs = StatsBase.coef(old_cache)
    fitresult = (
        coef = βs,
        x_transform = old_fitresult.x_transform,
        y_transform = old_fitresult.y_transform,
    )
    return fitresult, old_cache, NamedTuple{}()
end

function MMI.predict(model::AbstractGroupRegressor, fitresult, Xnew)
    Xnew = MMI.matrix(Xnew)
    @unpack coef, x_transform, y_transform = fitresult
    !isnothing(x_transform) && (Xnew = StatsBase.transform(x_transform, Xnew))
    ypred = Xnew * coef
    !isnothing(y_transform) && StatsBase.reconstruct!(y_transform, ypred)
    ypred
end


function range_and_grid(
    ridge::AbstractGroupRegressor,
    param_min,
    param_max,
    scale,
    resolution,
    n,
    rng
)
    param_symbol = _main_hyperparameter(ridge)
    param_range =
        range(ridge, param_symbol, lower = param_min, upper = param_max, scale = scale)
    model_grid =
        MLJTuning.grid(ridge, [param_symbol], [MLJ.iterator(param_range, resolution)])

    if length(model_grid) > n
        model_grid = sample(rng, model_grid, n; replace = false)
    end

    param_range, model_grid
end



"""
    MultiGroupRidgeRegressor(; decomposition, λ, groups)
"""
mutable struct MultiGroupRidgeRegressor{T,G<:GroupedFeatures} <:
               FixedLambdaGroupRidgeRegressor
    decomposition::Symbol
    λs::T   #Named tuple
    groups::G
    center::Bool
    scale::Bool
end

_main_hyperparameter(::MultiGroupRidgeRegressor) = :λs
_groups(m::MultiGroupRidgeRegressor, p) = m.groups

function MultiGroupRidgeRegressor(;
    groups::GroupedFeatures,
    λs::AbstractVector = ones(ngr),
    decomposition = :default,
    center = true,
    scale = true,
)
    ngr = ngroups(groups)
    λ_expr = Tuple(Symbol.(:λ, Base.OneTo(ngr)))
    λ_tupl = MutableNamedTuple{λ_expr}(tuple(λs...))
    MultiGroupRidgeRegressor(decomposition, λ_tupl, groups, center, scale)
end


function range_and_grid(ridge::MultiGroupRidgeRegressor, λ_min, λ_max, scale, resolution, n, rng)
    λ_names = [Meta.parse("(λs.$λ)") for λ in keys(ridge.λs)]
    nparams = length(λ_names)
    λ_range =
        [range(ridge, λ, lower = λ_min, upper = λ_max, scale = scale) for λ in λ_names]
    λ_product_grid = MLJ.iterator.(λ_range, resolution)
    if nparams*log(resolution) > log(n)
        tmp_idx = zeros(Int, nparams)
        model_grid = [deepcopy(ridge) for i in Base.OneTo(n)]
        for i in Base.OneTo(n)
            sample!(rng, 1:resolution, tmp_idx)
            clone = model_grid[i]
            for k in eachindex(λ_names)
                MLJ.recursive_setproperty!(clone, λ_names[k], λ_product_grid[k][tmp_idx[k]])
            end
        end
    else
        model_grid = MLJTuning.grid(ridge, λ_names, λ_product_grid)
    end
    λ_range, model_grid
end


"""
    DefaultTuning(resolution, n, param_min_ratio, param_max, scale)

Determines the default set of hyperparameters to loop over when tuning a
`AbstractGroupRidgeRegressor` method.  Parameters are chosen on a grid
that is equidistant in `scale` (e.g. `:log10` or `:linear` or `:default`) with number
of points given by `resolution` (default `100`) that ranges from `param_min_ratio*param_max` to
`param_max`. Both `param_min_ratio` and `param_max` can be specified as `:default`,
in which case a method specific default choice will be used.

If there are multiple hyperparameters (say `d`),
then the above rules are used componentwise. `n` (default `1000`)
is the largest number of hyperparameters to explore (if `resolution^d > n`,
then the parameters are randomly subsampled to `n` of them).
"""
Base.@kwdef struct DefaultTuning{T,M}
    resolution::Int = 100
    n::Int = 1000
    param_min_ratio::M = :default
    param_max::T = :default
    scale = :default
end

function _tuning_grid(tuning::DefaultTuning, model, fitted_machine, rng)
    @unpack resolution, n, scale = tuning
    if tuning.param_max === :default
        param_max = _default_hyperparameter_maximum(model, fitted_machine)
    elseif isa(tuning.param_max, Number)
        param_max = tuning.param_max
    else
        error("param_max can be :default or a number only.")
    end

    if tuning.param_min_ratio === :default
        param_min_ratio = _default_param_min_ratio(model, fitted_machine)
    elseif isa(tuning.param_min_ratio, Number)
        param_min_ratio = tuning.param_min_ratio
    else
        error("param_min_ratio can be :default or a number only.")
    end

    if tuning.scale === :default
        _scale = _default_scale(model, fitted_machine)
    else
        _scale = tuning.scale
    end

    param_min = param_min_ratio * param_max
    param_range, model_grid = range_and_grid(model, param_min, param_max, _scale, resolution, n, rng)

    param_range, model_grid, param_max
end



"""
    LooRidgeRegressor(;ridge,
                       tuning = DefaultTuning(),
                       rng = Random.GLOBAL_RNG)


A MLJ model that wraps a `ridge` model such as `SigmaRidgeRegressor` and tunes
its parameters by leave-one-out-cross-validation with `tuning` settings defaulting to
[`DefaultTuning`](@ref). In case there is randomness in choosing the search space of
hyperparameters, then the `rng` may be specified (defaults to `Random.GLOBAL_RNG`).
"""
Base.@kwdef mutable struct LooRidgeRegressor{G,T} <: AbstractGroupRidgeRegressor
    ridge::G = SingleGroupRidgeRegressor()
    tuning::T = DefaultTuning()
    rng = Random.GLOBAL_RNG
end

_groups(loo::LooRidgeRegressor) = _groups(loo.ridge)


_workspace(wk::BasicGroupRidgeWorkspace) = wk
_workspace(wk) = wk.workspace

function MMI.fit(m::LooRidgeRegressor, verb::Int, X, y)
    ridge = m.ridge
    mach = MLJ.machine(ridge, X, y)
    fit!(mach)
    x_transform = mach.fitresult.x_transform
    y_transform = mach.fitresult.y_transform

    ridge_workspace = _workspace(mach.cache)
    param_range, model_grid, param_max = _tuning_grid(m.tuning, ridge, mach, m.rng)
    history = map(model_grid) do newm
        param = _main_hyperparameter_value(newm)
        mach.model = newm
        fit!(mach; verbosity = 0)
        λ = deepcopy(ridge_workspace.λs)
        loo = loo_error(ridge_workspace)
        (loo = loo, param = param, model = newm, λ = λ)
    end
    loos = [h.loo for h in history]
    params = [h.param for h in history]
    λs = [h.λ for h in history]
    best_model_idx = argmin(loos)
    best_model = model_grid[best_model_idx]
    best_param = params[best_model_idx]
    best_loo = loos[best_model_idx]
    best_λs = λs[best_model_idx]

    report = (
        best_model = best_model,
        best_param = best_param,
        best_λs = best_λs,
        best_loo = best_loo,
        loos = loos,
        λs = λs,
        params = params,
        param_max = param_max,
        param_range = param_range,
    )
    mach.model = best_model
    fit!(mach)
    βs = StatsBase.coef(ridge_workspace)
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)
    # return
    return fitresult, ridge_workspace, report
end


Base.@kwdef mutable struct TunedRidgeRegressor{G,R,M,T} <: AbstractGroupRidgeRegressor
    ridge::G = SingleGroupRidgeRegressor(decomposition = :cholesky, λ = 1.0)
    tuning::T = DefaultTuning()
    resampling::R = MLJ.CV(nfolds = 5, shuffle=true)
    measure::M = MLJ.l2
end


function MMI.fit(m::TunedRidgeRegressor, verb::Int, X, y)
    ridge = m.ridge
    mach = MLJ.machine(ridge, X, y)
    fit!(mach)
    #x_transform = mach.fitresult.x_transform
    #y_transform = mach.fitresult.y_transform

    ridge_workspace = _workspace(mach.cache)
    param_range, model_grid, param_max = _tuning_grid(m.tuning, ridge, mach, m.resampling.rng)

    tuned_model = MLJ.TunedModel(
        model = ridge,
        ranges = model_grid,
        tuning = MLJ.Explicit(),
        resampling = m.resampling,
        measure = m.measure,
    )

    tuned_mach = MLJ.machine(tuned_model, X, y)
    fit!(tuned_mach)

    _fitresult = tuned_mach.fitresult.fitresult
    _cache = tuned_mach.fitresult.cache
    best_λs = deepcopy(_workspace(_cache).λs)

    tunedreport = tuned_mach.report
    best_param = _main_hyperparameter_value(tunedreport.best_model)

    tunedreport = (tunedreport..., best_param=best_param, best_λs = best_λs)
    # return
    return _fitresult, _cache, tunedreport
end
