abstract type AbstractGroupRegressor <: MMI.Deterministic end
abstract type AbstractGroupRidgeRegressor <: AbstractGroupRegressor end


function StatsBase.fit(grp_ridge::AbstractGroupRidgeRegressor, X, Y, grp::GroupedFeatures)
    decomposition = grp_ridge.decomposition
    tuning = grp_ridge.λ
    if decomposition == :default || decomposition == :cholesky
        pred = CholeskyRidgePredictor(X)
    elseif decomposition == :woodbury
        pred = WoodburyRidgePredictor(X)
    else
        "Only :default, :cholesky and :woodbury currently supported"
    end
    workspace = BasicGroupRidgeWorkspace(X = X, Y = Y, groups = grp, XtXpΛ_chol = pred)
    StatsBase.fit!(workspace, tuning)
    workspace
end
