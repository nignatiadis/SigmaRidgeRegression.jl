abstract type AbstractRidgeTuning end

Base.@kwdef struct SigmaRidgeTuning{O} <: AbstractRidgeTuning
	noiseestimator::O = SigmaLeaveOneOut()
end 

Base.@kwdef struct OneParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = GoldenSection()
end

Base.@kwdef struct MultiParamCrossValRidgeTuning{O} <: AbstractRidgeTuning
	optimizer::O = LBFGS()
end 


#function fit!(rdg::BasicGroupRidgeWorkspace, ::OneParamCrossValRidgeTuning)
#	
#end

