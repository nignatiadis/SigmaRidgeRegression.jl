abstract type AbstractGroupRidgeRegressor <: MMI.Deterministic end

Base.@kwdef struct GroupRidgeRegression{T} <: AbstractGroupRidgeRegressor
	decomposition::Symbol = :default 
	λ::T
end

function StatsBase.fit(grp_ridge::AbstractGroupRidgeRegressor, X, Y, grp::GroupedFeatures)
	decomposition = grp_ridge.decomposition
	tuning = grp_ridge.λ
	if decomposition == :default || decomposition == :cholesky 
		pred = CholeskyRidgePredictor(X)
	elseif decomposition == :woodbury
		pred = WoodburyRidgePredictor(X)
	end 
	workspace = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp, XtXpΛ_chol = pred)
	
	StatsBase.fit!(workspace, tuning)
	workspace
end 

abstract type AbstractRidgeTuning end

"""
Use leave-one-out-cross-validation to choose over Ridge-penalty matrices
of the form:

	``λ \\cdot I,  λ \\in (0,∞)``
"""
Base.@kwdef struct OneParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = GoldenSection()
end

function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, tune::OneParamCrossValRidgeTuning)
	function _tune(λ) # \lambda is 1-dimensional
	    StatsBase.fit!(rdg, λ)
	end
	λ_min = 1e-6
	λ_max = 1e3
	opt_res = optimize(_tune, λ_min, λ_max, tune.optimizer)
	λ = opt_res.minimizer
	StatsBase.fit!(rdg, λ)
	rdg.cache = (params = λ, opt_res = opt_res, tune=tune)
	rdg
end 



Base.@kwdef struct SigmaRidgeTuning{O,T} <: AbstractRidgeTuning
	noiseestimator::O = SigmaLeaveOneOut()
	initializer::T = OneParamCrossValRidgeTuning()

end 

function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, 
	          tune::SigmaRidgeTuning{<:SigmaLeaveOneOut})
	StatsBase.fit!(rdg, tune.initializer)
	mom = MomentTunerSetup(rdg)
	fit!(rdg, mom, tune)
end 

function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, 
	          mom::MomentTunerSetup,
	          tune::SigmaRidgeTuning{<:SigmaLeaveOneOut})

	function _tune(σ)
		λ_σ = get_λs(mom, σ^2)
		fit!(rdg, λ_σ)
	end
	
	opt_res = optimize(_tune, 1e-7, sqrt(σ_squared_max(mom)), 
	                   tune.noiseestimator.optimizer)
	σ = opt_res.minimizer
	λs = get_λs(mom, σ^2)
	params = (σ = σ, λs = λs)
	rdg.cache = (params = params, opt_res = opt_res, tune=tune)
	rdg
end



"""
Use leave-one-out-cross-validation to choose over
group-specific Ridge-penalty matrices
of the form, i.e., over ``λ \\in (0,∞)^K``.
"""
Base.@kwdef struct MultiParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = LBFGS()
end 

function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, tune::MultiParamCrossValRidgeTuning)
	
	function _tune(logλ)
	    StatsBase.fit!(rdg, exp.(logλ))
	end

	logλ_init = clamp.(log.(rdg.λs), -10.0, 10.0)
	opt_res = optimize(_tune, logλ_init,  tune.optimizer)
	λs = exp.(opt_res.minimizer)
	rdg.cache = (params = λs, opt_res = opt_res, tune=tune)
	rdg
end 




