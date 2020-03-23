abstract type AbstractRidgeTuning end

Base.@kwdef struct SigmaRidgeTuning{O} <: AbstractRidgeTuning
	noiseestimator::O = SigmaLeaveOneOut()
end 

function fit!(rdg::BasicGroupRidgeWorkspace, 
	          tune::SigmaRidgeTuning{<:SigmaLeaveOneOut})
	mom = MomentTunerSetup(rdg)
	fit!(rdg, mom, tune)
end 

function fit!(rdg::BasicGroupRidgeWorkspace, 
	          mom::MomentTunerSetup,
	          tune::SigmaRidgeTuning{<:SigmaLeaveOneOut})

	function _tune(σ)
		λ_σ = get_λs(mom, σ^2)
		fit!(rdg, λ_σ)
	end
	
	opt_res = optimize(_tune, 1e-10, sqrt(σ_squared_max(mom)), 
	                   tune.noiseestimator.optimizer)
	σ = opt_res.minimizer
	λs = get_λs(mom, σ^2)
	params = (σ = σ, λs = λs)
	(params = params, opt_res = opt_res)
end


"""
Use leave-one-out-cross-validation to choose over Ridge-penalty matrices
of the form:

	``λ \\cdot I,  λ \\in (0,∞)``
"""
Base.@kwdef struct OneParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = GoldenSection()
end

function fit!(rdg::BasicGroupRidgeWorkspace, tune::OneParamCrossValRidgeTuning)
	function _tune(λ) # \lambda is 1-dimensional
	    fit!(rdg, λ)
	end
	λ_min = 1e-7
	λ_max = 1e3
	opt_res = optimize(_tune, λ_min, λ_max, tune.optimizer)
	λ = opt_res.minimizer
	(params = λ, opt_res = opt_res)
end 


"""
Use leave-one-out-cross-validation to choose over
group-specific Ridge-penalty matrices
of the form, i.e., over ``λ \\in (0,∞)^K``.
"""
Base.@kwdef struct MultiParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = LBFGS()
end 

function fit!(rdg::BasicGroupRidgeWorkspace, tune::MultiParamCrossValRidgeTuning)
	
	function _tune(logλ)
	    fit!(rdg, exp.(logλ))
	end

	logλ_init = clamp.(log.(rdg.λs), -10.0, 10.0)
	opt_res = optimize(_tune, logλ_init,  tune.optimizer)

	λs = exp.(opt_res.minimizer)

	(params = λs, opt_res = opt_res)
end 


