module RidgeRegression

using LinearAlgebra
using NonNegLeastSquares
using StatsBase

import Base:reduce, rand
import StatsBase:fit!

include("groupedfeatures.jl")
include("blockridge.jl")

export GroupedFeatures,
       group_idx,
	   group_summary,
	   group_expand,
	   random_betas,
	   BasicGroupRidgeWorkspace,
	   MomentTunerSetup,
	   get_αs_squared,
	   get_λs_squared,
	   λωλας_λ,
	   loo_error,
	   mse_ridge


end # module
