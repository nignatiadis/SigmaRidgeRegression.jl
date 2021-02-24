# Wrapper for Seagull
using .RCall

Base.@kwdef mutable struct TunedSeagull{G,R} <: AbstractGroupRegressor
    groups::G
    center::Bool = true
    scale::Bool = true
    param_min_ratio::Float64 = 1e-6
    resampling::R = MLJ.CV(nfolds = 5, shuffle=true, rng=1)
    resolution::Int = 100
end

export TunedSeagull


function MMI.fit(m::TunedSeagull, verb::Int, X, y)
    @unpack center, scale, groups, resolution, resampling, param_min_ratio = m
    Xmatrix = MMI.matrix(X)
    if center || scale
        x_transform = StatsBase.fit(
            ZScoreTransform,
            Xmatrix;
            dims = 1,
            center = center,
            scale = scale,
        )
        y_transform =
            StatsBase.fit(ZScoreTransform, y; dims = 1, center = center, scale = scale)

        Xmatrix_transformed = StatsBase.transform(x_transform, Xmatrix)
        Y_transformed = StatsBase.transform(y_transform, y)

    else
        x_transform = nothing
        y_transform = nothing
        Xmatrix_transformed = Xmatrix
        Y_transformed = y
    end
    n, p = size(Xmatrix)


    train_test = MLJBase.train_test_pairs(resampling, Base.OneTo(n))
    _cv_matrix = zeros(length(train_test), resolution)

    group_index =  group_expand(groups, 1:nfeatures(groups))
    @rput group_index
    R"library(seagull)"
    R"seagull_fit_full <- seagull(y=$Y_transformed, Z=$Xmatrix_transformed,
            groups=group_index, loops_lambda=$resolution,
            alpha=0, xi=$param_min_ratio)"
    R"seagull_lambdas_full <- seagull_fit_full$lambda"
    R"seagull_beta_full <- seagull_fit_full$random_effects"
    @rget seagull_beta_full
    @rget seagull_lambdas_full
    seagull_λ_max = maximum(seagull_lambdas_full)


    for (j,(train_idx, test_idx)) in enumerate(train_test)
        Xtrain = Xmatrix[train_idx,:]
        Ytrain = y[train_idx]
        Xtest  = Xmatrix[test_idx,:]
        Ytest  = y[test_idx]

        if center || scale
            x_transform_temp = StatsBase.fit(
                ZScoreTransform,
                Xtrain;
                dims = 1,
                center = center,
                scale = scale,
            )
            Xtrain = StatsBase.transform(x_transform_temp, Xtrain)
            Xtest  = StatsBase.transform(x_transform_temp, Xtest)

            y_transform_temp =
                StatsBase.fit(ZScoreTransform, Ytrain; dims = 1, center = center, scale = scale)

            Ytrain = StatsBase.transform(y_transform_temp, Ytrain)
            Ytest  = StatsBase.transform(y_transform_temp, Ytest)
        else
            x_transform = nothing
            y_transform = nothing
        end


        R"seagull_fit <- seagull(y=$Ytrain, Z=$Xtrain,
                max_lambda = $seagull_λ_max,
                groups=group_index, loops_lambda=$resolution,
                alpha=0, xi=$param_min_ratio)"
        R"seagull_betas <- seagull_fit$random_effects"

        @rget seagull_betas
        preds = Xtest * seagull_betas'
        _cv_matrix[j,:] = MLJ.rms.(Ref(Ytest), eachcol(preds))
    end

    best_idx = argmin(vec(mean(_cv_matrix, dims=1)))

    λ_opt = seagull_lambdas_full[best_idx]*sqrt(p)
    βs = vec(seagull_beta_full[best_idx,:])

    ηs = group_summary(groups, βs, norm)
    λs = λ_opt .* sqrt.(groups.ps ./ p) ./ ηs
   # fit!(workspace, tmp_λs)
   #

    report = (
        λ_max = seagull_λ_max * sqrt(p),
        best_param = λ_opt,
        best_λs = λs,
    )

    fitresult = (coef = βs, x_transform = x_transform, y_transform = y_transform)
    # return
    return fitresult,  NamedTuple{}(), report
end
