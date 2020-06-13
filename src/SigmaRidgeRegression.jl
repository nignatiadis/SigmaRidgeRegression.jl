module SigmaRidgeRegression

using BlockDiagonals
using Distributions
using LinearAlgebra
using NonNegLeastSquares
using Optim 
using RCall
using Setfield
using StatsBase
using WoodburyMatrices

import Base.\
import Base:reduce, rand
import LinearAlgebra:ldiv!
import StatsBase:fit!,fit, coef, leverage
import WoodburyMatrices:_ldiv!

include("utils.jl")
include("groupedfeatures.jl")
include("blockridge.jl")
include("variance_estimation.jl")
include("end_to_end.jl")
include("covariance_design.jl")
include("simulations.jl")
include("r_wrapper.jl")

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
	   GroupRidgeRegression,
	   AbstractRidgeTuning,
	   SigmaRidgeTuning,
	   OneParamCrossValRidgeTuning,
	   MultiParamCrossValRidgeTuning,
	   CovarianceDesign,
	   nfeatures,
	   get_Σ,
	   spectrum,
	   simulate_rotated_design,
	   AR1Design,
	   DiagonalCovarianceDesign,
	   ExponentialOrderStatsCovarianceDesign


end # module
