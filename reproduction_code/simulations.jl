using Pkg
Pkg.activate(@__DIR__)
using SigmaRidgeRegression
using StatsBase
using Statistics
using Random
using MLJ
using Distributions
using DrWatson
using JLD2

opt_no = parse(Int64, ARGS[1])
@show opt_no

 # helper functions
function _merge(grp::GroupedFeatures; groups_goal = 2)
    ngroups = length(grp.ps)
    new_ps = Vector{Int64}(undef, groups_goal)
    mod(ngroups, groups_goal) == 0 || throw("Only implemented when #groups_goal divides ngroups")
    step_length = div(ngroups, groups_goal)
    cnt = 1
    for i=1:groups_goal
        cnt_upper = cnt + (step_length - 1)
        new_ps[i] = sum(grp.ps[cnt:cnt_upper])
        cnt = cnt_upper
    end
    GroupedFeatures(new_ps)
end



function single_simulation(sim; Ks=Ks, save=true)
    res = []
    sim_name = randstring(16)

    groups = sim.groups

    _simulated_model = simulate(sim)
    X = MLJ.matrix(_simulated_model.X)
    Y = _simulated_model.Y
    resampling_idx = _simulated_model.resampling_idx

    bayes_λs = groups.ps  .* var(sim.response_noise) ./ abs2.(sim.response_model.αs) ./ sim.ntrain
    bayes_ridge = MultiGroupRidgeRegressor(groups, bayes_λs; center=false, scale=false)

    _mach = machine(bayes_ridge, X, Y)
    _eval = evaluate!(_mach, resampling=resampling_idx, measure=l2)
    mse_bayes = _eval.measurement[1]

    for K in Ks
        newgroups = _merge(groups; groups_goal = K)

        single_ridge = SingleGroupRidgeRegressor(groups=newgroups, λ=1.0, center=false, scale=false)
        loo_single_ridge = LooRidgeRegressor(ridge = deepcopy(single_ridge))

        sigma_ridge = SigmaRidgeRegressor(groups=newgroups, σ=0.01, center=false, scale=false)
        loo_sigmaridge = LooRidgeRegressor(ridge=deepcopy(sigma_ridge), tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))

        multi_ridge = MultiGroupRidgeRegressor(newgroups; center=false, scale=false)
        loo_multi_ridge = LooRidgeRegressor(ridge = deepcopy(multi_ridge), rng=MersenneTwister(1))

        glasso = GroupLassoRegressor(groups=newgroups, center=false, scale=false)
        holdout_glasso = TunedRidgeRegressor(ridge=deepcopy(glasso), resampling= Holdout(shuffle=true, rng=1), tuning=DefaultTuning(param_min_ratio=1e-5))

        models = [loo_sigmaridge, loo_single_ridge, loo_multi_ridge, holdout_glasso]

        tmp_mses = fill(Inf, length(models))

        for (model_idx, model) in enumerate(models)
            _mach = machine(model, X, Y)
            _eval = evaluate!(_mach, resampling=resampling_idx, measure=l2)
            tmp_mses[model_idx] =  _eval.measurement[1]
        end
        push!(res,
             (mse_sigma =  tmp_mses[1],
              mse_single = tmp_mses[2],
              mse_multi=   tmp_mses[3],
              mse_glasso = tmp_mses[4],
              mse_bayes = mse_bayes,
              K=K,
              sim=sim,
              p = sim.groups.p,
              cov = sim.Σ,
              response = sim.response_model,
              sim_name = sim_name)
        )
    end
    if save
        @save "simulation_results/$(sim_name).jld2" save
    end
    res
end

# Code that starts simulations

#Varying K from 1 to 10How informative?n=p/2n−2p p= 1280informative vs uninformative.K= 2`,`= 0,...,Kbla

Ks = 2 .^ (1:5)

p = 32*25

ns = Int.([p/2; p; 2p])

groups = GroupedFeatures(fill(25, 32))

ar1 = SigmaRidgeRegression.AR1Design(p, 0.8)
id = IdentityCovarianceDesign(p)

#uninformative_response_model = RandomLinearResponseModel(αs = fill(1.0,32), grp = groups)
informative_response_model = RandomLinearResponseModel(αs = (0:31)./3.1, grp = groups)


all_opts = dict_list(Dict(:n => ns, :cov => [ar1, id]))
opt = all_opts[opt_no]

n = opt[:n]
@show n
Σ = opt[:cov]
@show Σ
nreps = 100
sim = GroupRidgeSimulationSettings(groups=groups, Σ=id, response_noise = Normal(0,5), response_model=informative_response_model, ntrain = n)

for i in Base.OneTo(nreps)
    @show i
    single_simulation(sim)
end
