module SigmaRidgeRegression

using BlockDiagonals
using Distributions

using Expectations
using FillArrays #not used yet
using FiniteDifferences
using LinearAlgebra
using MLJBase
import MLJModelInterface
const MMI = MLJModelInterface
import MLJ
import MLJTuning
using MutableNamedTuples

using Random
using Requires
using Roots
using Setfield
using StatsBase
using Tables
using UnPack
using WoodburyMatrices

import Base.\
import Base: reduce, rand
import LinearAlgebra: ldiv!
import StatsBase: fit!, fit, coef, islinear, leverage, modelmatrix, response, predict
import WoodburyMatrices: _ldiv!


#---------- piracy ---------------------------------------------------------
MMI.nrows(X::Tables.MatrixTable) = size(MMI.matrix(X), 1)
MMI.selectrows(X::Tables.MatrixTable, ::Colon) = X
MMI.selectrows(X::Tables.MatrixTable, r::Integer) =
    MMI.selectrows(X::Tables.MatrixTable, r:r)
function MMI.selectrows(X::Tables.MatrixTable, r)
    new_matrix = MMI.matrix(X)[r, :]
    _names = getfield(X, :names)
    MMI.table(new_matrix; names = _names)
end

#----------------------------------------------------------------------------

include("nnls.jl")
include("utils.jl")
include("groupedfeatures.jl")
include("blockridge.jl")
include("end_to_end.jl")
include("covariance_design.jl")
include("simulations.jl")
include("theoretical_risk_curves.jl")
include("mmi.jl")
include("mmi_sigmaridge.jl")
include("grouplasso.jl")
include("mmi_metadata.jl")
include("datasets/CLLData/CLLData.jl")

function __init__()
    @require RCall="6f49c342-dc21-5d91-9882-a32aef131414" include("seagull.jl")
end

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
    loo_error,
    mse_ridge,
    σ_squared_max,
    sigma_squared_path,
    CovarianceDesign,
    nfeatures,
    get_Σ,
    spectrum,
    simulate_rotated_design,
    AR1Design,
    set_groups,
    DiagonalCovarianceDesign,
    IdentityCovarianceDesign,
    UniformScalingCovarianceDesign,
    ExponentialOrderStatsCovarianceDesign,
    BlockCovarianceDesign,
    simulate,
    GroupRidgeSimulationSettings,
    RandomLinearResponseModel,
    optimal_risk,
    optimal_single_λ_risk,
    optimal_ignore_second_group_risk,
    SingleGroupRidgeRegressor,
    MultiGroupRidgeRegressor,
    LooRidgeRegressor,
    TunedRidgeRegressor,
    SigmaRidgeRegressor,
    GroupLassoRegressor,
    DefaultTuning,
    LooSigmaRidgeRegressor


end # module
