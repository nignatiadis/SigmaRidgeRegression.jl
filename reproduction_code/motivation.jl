using Pkg
Pkg.activate(@__DIR__)

using SigmaRidgeRegression
using LinearAlgebra
using StatsBase
using Plots
using Statistics
using MLJ
using LaTeXStrings
using Random

using PGFPlotsX
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amsmath}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage[bbgreekl]{mathbbol}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\sigmacv}{\bbsigma}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\loo}{\operatorname{CV}^*}")
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\blambda}{\bm{\lambda}}")


main_cols =  [:purple :green :grey]# [:black :purple :green]


Random.seed!(1)
σ = 4.0
grp = GroupedFeatures([25;25;25])
n = 400
p = grp.p

#ρ = 0.7
#Σ = [ρ^(abs(i-j)) for i=1:p,j=1:p]
#Σ_chol = cholesky(Σ)
X = randn(n, p)# * Σ_chol.UL

αs = sqrt.([4.0;8.0;12.0])
#r#ange(2.0, 2.75, 3.5; length=3)#1.0:5.0
β = random_betas(grp, αs)
group_summary(grp, β, norm)
sum(abs2, β)
Y = X*β .+ σ .* randn(n)

loo_ridge = LooRidgeRegressor(ridge=SingleGroupRidgeRegressor(λ=1.0))
loo_ridge_machine = machine(loo_ridge, MLJ.table(X), Y)

loo_sigma_ridge = LooRidgeRegressor(ridge=SigmaRidgeRegressor(groups=grp), tuning=SigmaRidgeRegression.DefaultTuning(scale=:linear, param_min_ratio=0.001))
loo_sigmaridge_machine = machine(loo_sigma_ridge, MLJ.table(X), Y)


fit!(loo_ridge_machine)
fit!(loo_sigmaridge_machine)

σs = loo_sigmaridge_machine.report.params
loo_sigmaridge_machine.report.loos

param_subset = loo_sigmaridge_machine.report.params .<= 6.2
λs = Matrix(hcat(loo_sigmaridge_machine.report.λs...)')
λs_subset = λs[param_subset, :]

βs_list = [fit!(machine(SigmaRidgeRegressor(groups=grp, σ=σ), MLJ.table(X),Y)).fitresult.coef for σ in σs]
βs = Matrix(hcat(βs_list...)')



plot(loo_sigmaridge_machine.report.params[param_subset], λs_subset)

plot(loo_sigmaridge_machine.report.params, loo_sigmaridge_machine.report.loos)


loo_λ = loo_ridge_machine.report.best_param


#with gr
#plot(sqrt.(σs_squared1), mypath1.loos)
pl_left = plot(loo_sigmaridge_machine.report.params[param_subset], λs_subset,
            legend=:topleft, linecolor=main_cols,
            linestyle=[ :dot :dashdot :dash], xlab=L"\sigmacv",
            ylab=L"\widehat{\lambda}(\sigmacv)",
            background_color_legend = :transparent,
            foreground_color_legend = :transparent,
            grid = nothing,  frame=:box,
            label=["Group 1" "Group 2" "Group 3"],
            thickness_scaling = 1.8,
            size=(550,400))
savefig(pl_left, "intro_figure_left.tikz")

pl_right = plot(loo_sigmaridge_machine.report.params[param_subset], loo_sigmaridge_machine.report.loos[param_subset], color=:darkblue,
                grid = nothing,  frame=:box,
                background_color_legend = :transparent,
                foreground_color_legend = :transparent,
                xlab=L"\sigmacv", ylab=L"\loo(\sigmacv)", label=nothing,
                thickness_scaling = 1.8,
                size=(550,400), ylim=(18.2,19.2), yticks=[18.4;18.6;18.8;19.0])

savefig(pl_right, "intro_figure_right.tikz")



pl_left_tree = plot(loo_sigmaridge_machine.report.params, λs,
            ylim = (0,15), legend=:topleft, color=main_cols,
            linestyle=[ :dot :dashdot :dash],
            background_color_legend = :transparent,
            foreground_color_legend = :transparent,
            grid = nothing,  frame=:box,
			xlab=L"\sigmacv",
			ylab= L"\widehat{\lambda}_k(\sigmacv)",
            label=["Group 1" "Group 2" "Group 3"],
            thickness_scaling = 1.8,
            size=(550,400))


savefig(pl_left_tree, "christmas_tree_left.tikz")

cols_rep = hcat([fill(col, 1, 25) for col in  [:purple; :green; :grey]]...)
ltype_rep = hcat([fill(lt, 1, 25) for lt in  [:solid; :dash; :dot]]...)


pl_right_tree = plot(σs, βs, alpha=0.6,
                  linewidth=0.8,
                  label="",
                  grid = nothing,  frame=:box,
				  ylab= L"\widehat{w}_j(\widehat{\blambda}(\sigmacv))",
				  xlab=L"\sigmacv",
                  color=cols_rep, ylim=(-2.1,2.1),
                  thickness_scaling = 1.8,
                  size=(550,400))

savefig(pl_right_tree, "christmas_tree_right.tikz")
