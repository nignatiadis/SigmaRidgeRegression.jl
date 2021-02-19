Base.@kwdef mutable struct GroupLassoRegressor{G,P,T<:Number} <: AbstractGroupRegressor
    decomposition::Symbol = :default
    groups::G
    groups_multiplier::P = sqrt.(groups.ps) ./ sqrt(groups.p)
    λ::T = 1.0
    center::Bool = true
    scale::Bool = true
    maxiter::Int = 100
    η_reg::T = 1e-5
    η_threshold::T = 1e-2
    abs_tol::T = 1e-4
    truncate_to_zero::Bool = true
end

_main_hyperparameter(::GroupLassoRegressor) = :λ

function _default_hyperparameter_maximum(model::GroupLassoRegressor, fitted_machine)
    @unpack groups, groups_multiplier = model
    _norms = group_summary(groups, _workspace(fitted_machine.cache).XtY, norm)
    maximum(_norms ./ groups_multiplier)
end

_default_param_min_ratio(::GroupLassoRegressor, fitted_machine) = 1e-5

function _glasso_fit!(workspace, glasso::GroupLassoRegressor)
    @unpack η_reg, η_threshold, abs_tol, groups, maxiter, λ, groups_multiplier, truncate_to_zero = glasso

    tmp_λs = copy(workspace.λs)
    ηs_new = group_summary(groups, StatsBase.coef(workspace), norm)
    ηs_old = copy(ηs_new)

    converged = false
    iter_cnt = 0
    for i = 1:maxiter
        tmp_λs .= λ .* groups_multiplier ./ sqrt.(abs2.(ηs_new) .+ η_reg)
        fit!(workspace, tmp_λs)
        ηs_new .= group_summary(groups, StatsBase.coef(workspace), norm)
        #converged = norm(ηs_new .- ηs_old, Inf) < abs_tol
        #@show (ηs_new .- ηs_old) ./ sqrt.(abs2.(ηs_old) .+ η_reg)
        #@show (ηs_new .- ηs_old)
        #@show sqrt.( abs2.(ηs_old) .+ η_reg)
        converged =
            (norm((ηs_new .- ηs_old) ./ sqrt.(abs2.(ηs_old) .+ η_reg), Inf) < abs_tol) ||
            (norm(ηs_new .- ηs_old, Inf) < abs_tol)
        ηs_old .= ηs_new
        iter_cnt += 1
        converged && break
    end
    #@show "conv"
    ηs = group_summary(groups, StatsBase.coef(workspace), norm)
    final_λs = deepcopy(workspace.λs)
    #zero_groups = group_summary(groups, StatsBase.coef(workspace), norm) .< η_threshold .* groups_multiplier
    #final_λs[zero_groups] .= Inf
    #fit!(workspace, final_λs)

    (workspace = workspace, converged = converged, iter_count = iter_cnt)
end

function MMI.fit(m::GroupLassoRegressor, verb::Int, X, y)
    @unpack decomposition, center, scale = m
    Xmatrix = MMI.matrix(X)
    p = size(Xmatrix, 2)
    m_tmp = MultiGroupRidgeRegressor(;
        groups = m.groups,
        decomposition = decomposition,
        scale = scale,
        center = center,
    )
    multiridge_machine = MLJ.machine(m_tmp, X, y)
    fit!(multiridge_machine)
    workspace = _workspace(multiridge_machine.cache)
    glasso_workspace = _glasso_fit!(workspace, m)
    βs = StatsBase.coef(glasso_workspace.workspace)
    x_transform = multiridge_machine.fitresult.x_transform
    y_transform = multiridge_machine.fitresult.y_transform
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)

    # return
    return fitresult, glasso_workspace, NamedTuple{}()
end

function MMI.update(
    model::GroupLassoRegressor,
    verbosity::Int,
    old_fitresult,
    old_cache,
    X,
    y,
)
    glasso_workspace = _glasso_fit!(old_cache.workspace, model)
    βs = StatsBase.coef(glasso_workspace.workspace)

    x_transform = old_fitresult.x_transform
    y_transform = old_fitresult.y_transform
    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)

    return fitresult, glasso_workspace, NamedTuple{}()
end
