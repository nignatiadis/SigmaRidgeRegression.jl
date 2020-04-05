module SigmaRidgeRegression

using LinearAlgebra
using NonNegLeastSquares
using Optim 
using StatsBase
using WoodburyMatrices

import Base.\
import Base:reduce, rand
import LinearAlgebra:ldiv!
import StatsBase:fit!
import WoodburyMatrices:_ldiv!

include("utils.jl")
include("groupedfeatures.jl")
include("blockridge.jl")
include("variance_estimation.jl")
include("end_to_end.jl")
include("simulations.jl")

export GroupedFeatures,
       ngroups,
       group_idx,
	   group_summary,
	   group_expand,
	   random_betas,
	   CholeskyRidgePredictor,
	   WoodburyRidgePredictor,
	   BasicGroupRidgeWorkspace,
	   MomentTunerSetup,
	   get_αs_squared,
	   get_λs,
	   λωλας_λ,
	   loo_error,
	   mse_ridge,
	   σ_squared_max,
	   sigma_squared_path,
	   DickerMoments,
	   estimate_var,
	   whiten_covariates,
	   NoiseLevelEstimator,
	   SigmaLeaveOneOut,
	   DickerMoments,
	   AbstractRidgeTuning,
	   SigmaRidgeTuning,
	   OneParamCrossValRidgeTuning,
	   MultiParamCrossValRidgeTuning


end # module
