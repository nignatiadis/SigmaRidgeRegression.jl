using Pkg
Pkg.activate(@__DIR__)
using SigmaRidgeRegression
using Plots
using StatsBase
using Statistics
using LaTeXStrings
using Random
#using ColorSchemes

# grp = GroupedFeatures(num_groups=2,group_size=200)

# To add to tests.jl
# SigmaRidgeRegression.fixed_point_function(hs, γs, [1.0; Inf])
# SigmaRidgeRegression.risk_formula(hs, γs, αs, [1.0; 10000])
_linestyles = [ :dot :dashdot :dash]
_main_cols = [:grey :purple :green]


id_design = BlockCovarianceDesign([IdentityCovarianceDesign(), IdentityCovarianceDesign()])

function theoretical_and_realized_mse(γs, αs, design::BlockCovarianceDesign; n = 400, nreps = 50, ntest = 20_000)

    grp = GroupedFeatures(round.(Int, γs .* n))
    design = set_groups(design, grp)
    #design = IdentityCovarianceDesign(grp.p)
    #hs = [spectrum(design);spectrum(design)]
    @show "hiii"
    hs = spectrum.(design.blocks)

    λs1 = SigmaRidgeRegression.optimal_ignore_second_group_λs(γs, αs)
    λs2 = SigmaRidgeRegression.optimal_single_λ(γs, αs)
    λs3 = SigmaRidgeRegression.optimal_λs(γs, αs)

    all_λs = (λs1, λs2, λs3)
    opt_risk_theory = Matrix{Float64}(undef, 1, length(all_λs))
    risk_empirical = Matrix{Float64}(undef, nreps, length(all_λs))

    for (i, λs) in enumerate(all_λs)
        opt_risk_theory[1, i] = SigmaRidgeRegression.risk_formula(hs, γs, αs, λs)
    end

    for j = 1:nreps
        ridge_sim = GroupRidgeSimulationSettings(
            grp = grp,
            ntrain = n,
            ntest = ntest,
            Σ = design,
            response_model = RandomLinearResponseModel(αs = αs, grp = grp),
        )
        sim_res = simulate(ridge_sim)

        for (i, λs) in enumerate(all_λs)
            risk_empirical[j, i] = mse_ridge(
                StatsBase.fit(
                    MultiGroupRidgeRegressor(grp, λs),
                    sim_res.X_train,
                    sim_res.Y_train,
                    grp,
                ),
                sim_res.X_test,
                sim_res.Y_test,
            )
        end
    end
    risk_empirical = mean(risk_empirical; dims = 1)
    (theoretical = opt_risk_theory, empirical = risk_empirical, all_λs = all_λs)
end


function oracle_risk_plot(
    γs,
    sum_alpha_squared;
    design = id_design,
    ylim = (0, 2.5),
    n = 1000,
    title = nothing,
    legend = nothing,
    kwargs...,
)
    ratio_squared = range(0.0, 1.0, length = 30)

    αs_squared = ratio_squared .* sum_alpha_squared
    bs_squared = reverse(ratio_squared) .* sum_alpha_squared

    @show "hello"
    risks = [
        theoretical_and_realized_mse(
            γs,
            sqrt.([αs_squared[i]; bs_squared[i]]),
            design;
            n = n,
            kwargs...,
        ) for i = 1:length(ratio_squared)
    ]
    theoretical_risks = vcat(map(r -> r.theoretical, risks)...) .- 1
    empirical_risks = vcat(map(r -> r.empirical, risks)...) .- 1

    @show "hii"
    labels =
        [L"$\;$Optimal $\blambda = (\lambda, \infty)$" L"$\;$Optimal $\blambda = (\lambda, \lambda)$"  L"$\;$Optimal $\blambda = (\lambda_1, \lambda_2)$"]
    #colors = reshape(colorschemes[:seaborn_deep6][1:3], 1, 3)
    #colors = [:red :blue :purple]
    ylabel = L"$\risk{\blambda}- \sigma^2$"
    xlabel = L"\alpha_1^2/(\alpha_1^2 + \alpha_2^2)"
    pl = plot(
        ratio_squared,
        theoretical_risks,
        color = _main_cols,
        linestyle = _linestyles,
        ylim = ylim,
        xguide = xlabel,
        yguide = ylabel,
        legend = legend,
        label = labels,
        background_color_legend = :transparent,
        foreground_color_legend = :transparent,
        grid = false,
        title = title,
        frame = :box,
        plot_titlefontsize = 0.5,
        thickness_scaling = 2.2,
        legendfontsize = 12,
        size = (650, 500),
    )
    plot!(
        pl,
        ratio_squared,
        empirical_risks,
        seriestype = :scatter,
        color = _main_cols,
        markershape = :utriangle,
        markerstrokealpha = 0.0,
        markersize = 4,
        label = nothing,
    )
    pl
end

Random.seed!(10)
pgfplotsx()

using PGFPlotsX
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\blambda}{\bm{\lambda}}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\risk}[1]{\bm{R}(#1)}")


nreps = 1
title_curve_1 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 1"
curve_1 = oracle_risk_plot(
    [0.25, 0.25],
    1.0,
    legend = :topleft,
    nreps = nreps,
    title = title_curve_1,
)
plot!(curve_1, tex_output_standalone = true)
savefig(curve_1, "oracle_risk1.tikz")


function generate_risk_plots(base_plot_name; nreps=nreps, kwargs...)
    title_curve_1 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 1"
    curve_1 = oracle_risk_plot([0.25, 0.25], 1.0, legend = :topleft, nreps = nreps, title = title_curve_1; kwargs...)

    title_curve_2 = L"\gamma_1 = \frac{1}{10},\; \gamma_2 = \frac{4}{10},\;\; \alpha_1^2 + \alpha_2^2 = 1"
    curve_2 = oracle_risk_plot([0.1, 0.4], 1.0, legend = nothing, nreps = nreps, title=title_curve_2; kwargs...)

    title_curve_3 = L"\gamma_1 = \gamma_2 = 1,\;\; \alpha_1^2 + \alpha_2^2 = 1"
    curve_3 = oracle_risk_plot([1.0, 1.0], 1.0, legend = nothing, nreps = nreps, title=title_curve_3; kwargs...)

    title_curve_4 = L"\gamma_1 = \gamma_2 = \frac{1}{4},\;\; \alpha_1^2 + \alpha_2^2 = 2"
    curve_4 = oracle_risk_plot([0.25, 0.25], 2.0, legend = nothing, nreps = nreps, title=title_curve_4; kwargs...)

    title_curve_5 = L"\gamma_1 = \frac{1}{10},\; \gamma_2 = \frac{4}{10},\;\; \alpha_1^2 + \alpha_2^2 = 2"
    curve_5 = oracle_risk_plot([0.1, 0.4], 2.0, legend = nothing, nreps = nreps, title=title_curve_5; kwargs...)

    title_curve_6 = L"\gamma_1 = \gamma_2 = 1,\;\; \alpha_1^2 + \alpha_2^2 = 2"
    curve_6 = oracle_risk_plot([1.0, 1.0], 2.0, legend = nothing, nreps = nreps, title=title_curve_6; kwargs...)

    for (i, c) in enumerate([curve_1, curve_2, curve_3, curve_4, curve_5, curve_6])
        savefig(c, "$(base_plot_name)$i.tikz")
    end
end


generate_risk_plots("oracle_risk")

#ar1_block_design = BlockCovarianceDesign([AR1Design(ρ=0.95), AR1Design(ρ=0.95)])
exponential_design = BlockCovarianceDesign([ExponentialOrderStatsCovarianceDesign(rate=0.5),
                                            ExponentialOrderStatsCovarianceDesign(rate=0.5)])

generate_risk_plots("exponential_covariance/oracle_risk"; design=exponential_design, ylim=(0,6))
