const REG_MODELS =  (
    GroupLassoRegressor,
    SigmaRidgeRegressor,
    SingleGroupRidgeRegressor,
    MultiGroupRidgeRegressor,
    LooRidgeRegressor,
    TunedRidgeRegressor,
)

_path(M) =  "SigmaRidgeRegression.$(MMI.name(M))"
_descr(::Type{<:GroupLassoRegressor}) = "Group Lasso"
_descr(::Type{<:SigmaRidgeRegressor}) = "σ-Ridge regression"
_descr(::Type{<:SingleGroupRidgeRegressor}) = "Vanilla ridge regression"
_descr(::Type{<:MultiGroupRidgeRegressor}) = "Group-regularized ridge regression"
_descr(::Type{<:LooRidgeRegressor}) = "Ridge regressor tuned by leave-one-out cross-validation"
_descr(::Type{<:TunedRidgeRegressor}) = "Tuned ridge regressor"

MMI.metadata_pkg.(
    REG_MODELS,
    package_name = "SigmaRidgeRegression",
    package_uuid = "e566bdc2-f852-4ed0-870a-7389015b131d",
    package_url = "https://github.com/nignatiadis/SigmaRidgeRegression.jl",
    is_pure_julia = true,
    package_license = "MIT",
    is_wrapper = false,
)

for M in REG_MODELS
    MMI.metadata_model(M,
        input_scitype=MMI.Table(MMI.Continuous),
        target_scitype=AbstractVector{MMI.Continuous},
        supports_weights=false,
        docstring=_descr(M),
        load_path=_path(M))
end
