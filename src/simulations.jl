abstract type AbstractResponseModel end


"""
    random_betas(gr::GroupedFeatures, αs)

Suppose `gr` consists of ``K`` groups with ``p_1, \\dotsc, p_K`` features each.
Then this returns a random vector of βs of length ``\\sum p_g``,
where for `j` in the `g`-th group
we draw (independent) ``β_j \\sim N(0, α_g^2/p_g)``.
``α_g`` is the `g`-th element of the vectpr `αs`.
"""
function random_betas(gr::GroupedFeatures, αs)
    ps = gr.ps
    βs = zeros(eltype(αs), gr.p)
    for i = 1:gr.num_groups
        βs[group_idx(gr, i)] .= randn(ps[i]) .* sqrt(αs[i]^2 / ps[i])
    end
    βs
end



Base.@kwdef struct RandomLinearResponseModel <: AbstractResponseModel
    αs::Vector{Float64}
    grp::GroupedFeatures
    iid_measure = Normal()
end


function (resp::RandomLinearResponseModel)(X)
    β = random_betas(resp.grp, resp.αs) #todo, allow other noise dbn.
    Xβ = X * β
    Xβ, β
end

Base.@kwdef struct GroupRidgeSimulationSettings{C,R,D}
    groups::GroupedFeatures
    Σ::C
    response_model::R
    response_noise::D = Normal()
    ntest::Int = 10000
    ntrain::Int
    iid_measure = Normal()
end

Base.@kwdef struct GroupRidgeSimulation
    groups::GroupedFeatures
    X::Matrix{Float64}
    Y::Vector{Float64}
    resampling_idx = nothing
    β = nothing
end



function simulate(group_simulation::GroupRidgeSimulationSettings)
    ntrain = group_simulation.ntrain
    ntest = group_simulation.ntest
    ntotal = ntrain + ntest
    @unpack response_model, response_noise = group_simulation

    X = simulate_rotated_design(
        group_simulation.Σ,
        ntotal;
        rotated_measure = group_simulation.iid_measure,
    )

    Xβ, β = response_model(X)
    Y = Xβ .+ rand(response_noise, ntotal)

    resampling_idx = [(1:ntrain, (ntrain+1):ntotal)]

    GroupRidgeSimulation(;
        groups = group_simulation.groups,
        X=X,
        Y=Y,
        resampling_idx = resampling_idx,
        β = β,
    )
end
