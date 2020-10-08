Base.@kwdef mutable struct GroupLassoRegressor{G,P,T<:Number} <: AbstractGroupRegressor
    decomposition::Symbol = :default
    groups::G
    groups_multiplier::P = sqrt.(groups.ps) ./ sqrt(groups.p)
    λ::T = 1.0
    center::Bool = false
    scale::Bool = false
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

_default_param_min_ratio(::GroupLassoRegressor, fitted_machine) = 1e-3

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
    m_tmp = MultiGroupRidgeRegressor(
        m.groups;
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



Base.@kwdef mutable struct CVGGLassoRegressor{G,S} <: AbstractGroupRegressor
    groups::G
    nfolds::S = 10
    center::Bool = false
    scale::Bool = false
    engine::Symbol = :gglasso
    eps::Float64 = 1e-8
end

function MMI.fit(m::CVGGLassoRegressor, verb::Int, X, y)
    @unpack center, scale, groups, nfolds, engine = m
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

    n, p = size(Xmatrix)
    nfolds = nfolds === :Loo ? n : nfolds
    groups = m.groups
    group_index = group_expand(groups, Base.OneTo(ngroups(groups)))

    @rput Xmatrix
    @rput y
    @rput group_index
    @rput p

    if engine === :gglasso
        R"library(gglasso)"
        R"gglasso_cv_fit <- cv.gglasso(Xmatrix, y, group=group_index, nfolds=$(nfolds), intercept=FALSE)"
        R"betas <- coef(gglasso_cv_fit, s='lambda.min')[-1]"
        R"lambda_cv <- gglasso_cv_fit$lambda.min"
        R"lambda_max <- max(gglasso_cv_fit$lambda)"
        R"tmp_intercept <- 0.0"
    elseif engine === :grpreg
        R"library(grpreg)"
        R"gglasso_cv_fit <- cv.grpreg(Xmatrix, y, group=as.factor(group_index), nfolds=$(nfolds),
                                      penalty = 'grLasso', family='gaussian')"
        R"lambda_cv <- gglasso_cv_fit$lambda.min"
        R"tmp_intercept <- coef(gglasso_cv_fit, lambda_cv)[1]"
        R"lambda_max <- max(gglasso_cv_fit$lambda)"
        R"betas <- coef(gglasso_cv_fit, lambda_cv)[-1]"
    end


    @rget betas
    @rget lambda_cv
    @rget lambda_max
    @rget tmp_intercept

    λ_opt = sqrt(p)*lambda_cv
    ηs_groupwise = group_summary(groups, betas, norm)
    λs = λ_opt .* sqrt.( groups.ps ./ p) ./ ηs_groupwise


    fitresult = (coef = betas, x_transform = x_transform, y_transform = y_transform)
    report = (best_param = λ_opt, best_λs = λs, param_max = lambda_max, tmp_intercept=tmp_intercept)
    # return
    return fitresult, nothing, report
end
