module SigmaRidgeRegression

using BlockDiagonals
using Distributions
using FillArrays #not used yet
using FiniteDifferences
using LinearAlgebra
import MLJModelInterface
const MMI = MLJModelInterface
import MLJ 
import MLJTuning
using MutableNamedTuples

using NonNegLeastSquares
using Optim 
using RCall
using Roots
using Setfield
using StatsBase
using UnPack
using WoodburyMatrices

import Base.\
import Base:reduce, rand
import LinearAlgebra:ldiv!
import StatsBase:fit!,fit, coef, islinear, leverage, modelmatrix, response, predict
import WoodburyMatrices:_ldiv!

include("utils.jl")
include("groupedfeatures.jl")
include("blockridge.jl")
include("variance_estimation.jl")
include("end_to_end.jl")
include("covariance_design.jl")
include("simulations.jl")
include("r_wrapper.jl")
include("theoretical_risk_curves.jl")
include("mmi.jl")
include("grouplasso.jl")

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
	   IdentityCovarianceDesign,
	   ExponentialOrderStatsCovarianceDesign,
	   simulate,
	   GroupRidgeSimulationSettings,
	   RandomLinearResponseModel,
	   optimal_risk,
	   optimal_single_λ_risk,
	   optimal_ignore_second_group_risk,
	   SingleGroupRidgeRegressor,
	   LooCVRidgeRegressor,
	   MultiGroupRidgeRegressor,
	   GroupLassoRegressor
	   
	   
end # module
