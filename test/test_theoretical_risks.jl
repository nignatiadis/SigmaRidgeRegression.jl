using SigmaRidgeRegression
using Test
using Random
import StatsBase

ar1_design = BlockCovarianceDesign([
    AR1Design(;ρ=0.8),
    AR1Design(;ρ=0.5)])

id_design = BlockCovarianceDesign([
        IdentityCovarianceDesign(),
        IdentityCovarianceDesign()])

grp1 = GroupedFeatures([2000,4000])
grp2 = GroupedFeatures([800,500])

for grp in [grp1; grp2]
    for _design in [ar1_design; id_design]
        _design = set_groups(_design, grp)
        _n = 2000
        _γs = grp.ps ./ _n
        _λs = [2.0; 0.4]

        _αs = [1.0; 7.0]

        theory_risk = @show SigmaRidgeRegression.risk_formula(spectrum.(_design.blocks), _γs, _αs, _λs)


        _ridge_sim = GroupRidgeSimulationSettings(;
            groups=grp,
            ntrain=_n,
            ntest=20_000,
            Σ=_design,
            response_model=RandomLinearResponseModel(; αs=_αs, grp=grp),
        )
        Random.seed!(1)


        _sim_res = simulate(_ridge_sim)
        _X_train = _sim_res.X[_sim_res.resampling_idx[1][1],:]
        _Y_train = _sim_res.Y[_sim_res.resampling_idx[1][1]]

        _X_test = _sim_res.X[_sim_res.resampling_idx[1][2],:]
        _Y_test = _sim_res.Y[_sim_res.resampling_idx[1][2]]

        ridge_risk = mse_ridge(
                StatsBase.fit(
                    MultiGroupRidgeRegressor(; groups=grp, λs=_λs, center=false, scale=false),
                    _X_train,
                    _Y_train,
                    grp,
                ),
                _X_test,
                _Y_test,
            )
        @test ridge_risk ≈ theory_risk atol =1.0
    end
end
