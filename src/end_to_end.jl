abstract type AbstractGroupRegressor <: MMI.Deterministic end
abstract type AbstractGroupRidgeRegressor <: AbstractGroupRegressor end


function StatsBase.fit(grp_ridge::AbstractGroupRidgeRegressor, X, Y, grp::GroupedFeatures)
    decomposition = grp_ridge.decomposition
    tuning = _main_hyperparameter_value(grp_ridge)
    nobs = length(Y)
    if decomposition === :default
        decomposition = (nfeatures(grp) <= 4*nobs) ? :cholesky : :woodbury
    end

    if decomposition === :cholesky
        pred = CholeskyRidgePredictor(X)
    elseif decomposition === :woodbury
        pred = WoodburyRidgePredictor(X)
    else
        "Only :default, :cholesky and :woodbury currently supported"
    end
    workspace = BasicGroupRidgeWorkspace(X = X, Y = Y, groups = grp, XtXpΛ_chol = pred)
    StatsBase.fit!(workspace, tuning)
    workspace
end
