using Pkg
Pkg.activate(@__DIR__)
using RCall
using SigmaRidgeRegression
using StatsBase
using Random
using MLJ
using DrWatson
using JLD2
using Distributions

opt_no = parse(Int64, ARGS[1])
@show opt_no

# helper functions
function _merge(grp::GroupedFeatures; groups_goal=2)
    ngroups = length(grp.ps)
    new_ps = Vector{Int64}(undef, groups_goal)
    mod(ngroups, groups_goal) == 0 ||
        throw("Only implemented when #groups_goal divides ngroups")
    step_length = div(ngroups, groups_goal)
    cnt = 1
    for i in 1:groups_goal
        cnt_upper = cnt + (step_length - 1)
        new_ps[i] = sum(grp.ps[cnt:cnt_upper])
        cnt = cnt_upper
    end
    return GroupedFeatures(new_ps)
end

function single_simulation(sim; Ks=Ks, save=true)
    res = []
    sim_name = randstring(16)

    groups = sim.groups

    _simulated_model = simulate(sim)
    X = MLJ.table(_simulated_model.X)
    Y = _simulated_model.Y
    resampling_idx = _simulated_model.resampling_idx

    bayes_λs =
        groups.ps .* var(sim.response_noise) ./ abs2.(sim.response_model.αs) ./ sim.ntrain
    bayes_ridge = MultiGroupRidgeRegressor(;
        groups=groups, λs=bayes_λs, center=false, scale=false
    )

    _mach = machine(bayes_ridge, X, Y)
    _eval = evaluate!(_mach; resampling=resampling_idx, measure=l2)
    mse_bayes = _eval.measurement[1]

    for K in Ks
        newgroups = _merge(groups; groups_goal=K)

        loo_single_ridge = LooRidgeRegressor(;
            ridge=SingleGroupRidgeRegressor(; groups=newgroups, center=false, scale=false)
        )

        loo_sigmaridge = LooSigmaRidgeRegressor(;
            groups=newgroups, center=false, scale=false
        )

        loo_multi_ridge = LooRidgeRegressor(;
            ridge=MultiGroupRidgeRegressor(; groups=newgroups, center=false, scale=false),
            rng=MersenneTwister(1),
        )

        holdout_glasso = TunedSeagull(; groups=newgroups, center=false, scale=false)

        models = [loo_sigmaridge, loo_single_ridge, loo_multi_ridge, holdout_glasso]

        tmp_mses = fill(Inf, length(models))

        for (model_idx, model) in enumerate(models)
            _mach = machine(model, X, Y)
            _eval = evaluate!(_mach; resampling=resampling_idx, measure=l2)
            tmp_mses[model_idx] = _eval.measurement[1]
        end

        push!(
            res,
            (
                mse_sigma=tmp_mses[1],
                mse_single=tmp_mses[2],
                mse_multi=tmp_mses[3],
                mse_glasso=tmp_mses[4],
                mse_bayes=mse_bayes,
                K=K,
                sim=sim,
                p=sim.groups.p,
                cov=sim.Σ,
                response=sim.response_model,
                sim_name=sim_name,
            ),
        )
    end
    if save
        @save "simulation_results/$(sim_name).jld2" res
    end
    return res
end

# Code that starts simulations


Ks = 2 .^ (1:5)

p = 32 * 25

ns = Int.([p / 2; p; 2p])

groups = GroupedFeatures(fill(25, 32))

ar1 = SigmaRidgeRegression.AR1Design(p, 0.8)
id = IdentityCovarianceDesign(p)

informative_response_model = RandomLinearResponseModel(; αs=(0:31) ./ 3.1, grp=groups)

all_opts = dict_list(Dict(:n => ns, :cov => [ar1, id]))
opt = all_opts[opt_no]

n = opt[:n]
@show n
Σ = opt[:cov]
@show Σ
nreps = 400
sim = GroupRidgeSimulationSettings(;
    groups=groups,
    Σ=Σ,
    response_noise=Normal(0, 5),
    response_model=informative_response_model,
    ntrain=n,
)

for i in Base.OneTo(nreps)
    @show i
    single_simulation(sim)
end
