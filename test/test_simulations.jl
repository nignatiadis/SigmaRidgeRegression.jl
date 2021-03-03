using Random
using MLJ
using SigmaRidgeRegression

id_design = BlockCovarianceDesign([IdentityCovarianceDesign(), IdentityCovarianceDesign()])
ps = [100; 100]
grp = GroupedFeatures(ps)
design = set_groups(id_design, grp)
αs = [1.0; 1.0]

ntest = 20_000
ntrain = 400

ridge_sim = GroupRidgeSimulationSettings(;
    groups = grp,
    ntrain = ntrain,
    ntest = ntest,
    Σ = design,
    response_model = RandomLinearResponseModel(; αs = αs, grp = grp),
)

Random.seed!(1)
sim_res = simulate(ridge_sim)

@test length.(sim_res.resampling_idx[1]) == (ntrain, ntest)

X_train = sim_res.X[sim_res.resampling_idx[1][1], :]
Y_train = sim_res.Y[sim_res.resampling_idx[1][1]]

X_test = sim_res.X[sim_res.resampling_idx[1][2], :]
Y_test = sim_res.Y[sim_res.resampling_idx[1][2]]

λs = [1.0; 2.0]
multiridge =
    MultiGroupRidgeRegressor(; groups = grp, λs = λs, center = false, scale = false)

mse = mse_ridge(
    StatsBase.fit(MultiGroupRidgeRegressor(; groups = grp, λs = λs), X_train, Y_train, grp),
    X_test,
    Y_test,
)


_mach = machine(multiridge, MLJ.table(sim_res.X), sim_res.Y)
_eval = evaluate!(_mach, resampling = sim_res.resampling_idx, measure = l2)

@test _eval.measurement[1] == mse
