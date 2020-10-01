Base.@kwdef mutable struct GroupLassoRegressor{G, P, T<:Number} <: AbstractGroupRegressor
	decomposition::Symbol = :default
	groups::G
	groups_multiplier::P = sqrt.(groups.ps)./sqrt(groups.p)
	λ::T = 1.0
	maxiter::Int = 100
	η_reg::T = 1e-5
	η_threshold::T = 1e-4
	abs_tol::T = η_threshold
end



function _glasso_fit!(workspace, glasso::GroupLassoRegressor)
	@unpack η_reg, η_threshold, abs_tol, groups, maxiter, λ, groups_multiplier = glasso

	tmp_λs = copy(workspace.λs)
	ηs_new = group_summary(groups, StatsBase.coef(workspace), norm)
	ηs_old = copy(ηs_new)
	
	converged = false
	iter_cnt = 0
	for i=1:maxiter
		tmp_λs .= λ .* groups_multiplier ./ sqrt.( abs2.(ηs_new) .+ η_reg)
		fit!(workspace, tmp_λs)
		ηs_new .= group_summary(groups, StatsBase.coef(workspace), norm)
		converged = (@show norm(ηs_new .- ηs_old)) < abs_tol
		ηs_old .= ηs_new
		iter_cnt += 1
		converged && break
    end 
    # zero_groups = group_summary(grp, rdg_workspace.β_curr, norm).^2 .< glasso.η*1000 #TODO

	(workspace = workspace, converged = converged, iter_count = iter_cnt)
end


function MMI.fit(m::GroupLassoRegressor, verb::Int, X, y)
    Xmatrix = MMI.matrix(X)
    p = size(Xmatrix, 2)
	m_tmp = MultiGroupRidgeRegressor(m.groups; decomposition = m.decomposition)
	workspace = StatsBase.fit(m_tmp, Xmatrix, y, m.groups)
	glasso_workspace = _glasso_fit!(workspace, m)
    βs = StatsBase.coef(glasso_workspace.workspace)
    # return
    return βs, glasso_workspace, NamedTuple{}()
end


function MMI.update(model::GroupLassoRegressor, verbosity::Int, old_fitresult, old_cache, X, y)
    glasso_workspace = _glasso_fit!(old_cache.workspace, model)
    βs = StatsBase.coef(glasso_workspace.workspace)
    return βs, glasso_workspace, NamedTuple{}()
end