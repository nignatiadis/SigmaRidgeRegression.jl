module RidgeRegression

using LinearAlgebra
using NonNegLeastSquares
using StatsBase

import Base:reduce, rand
import StatsBase:fit!

include("groupedfeatures.jl")
include("blockridge.jl")

export GroupedFeatures,
       ngroups,
       group_idx,
	   group_summary,
	   group_expand,
	   random_betas,
	   BasicGroupRidgeWorkspace,
	   MomentTunerSetup,
	   get_αs_squared,
	   get_λs,
	   λωλας_λ,
	   loo_error,
	   mse_ridge,
	   σ_squared_max,
	   sigma_squared_path


end # module
