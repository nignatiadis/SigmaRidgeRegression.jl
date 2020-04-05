using SigmaRidgeRegression
using LinearAlgebra
using StatsBase
using Plots
using Statistics

using LaTeXStrings
using Random

pgfplotsx()


main_cols = [:black :purple :green]

Random.seed!(100)
σ = 4.0
grp = GroupedFeatures([30;30;30])
n = 400
p = grp.p

#ρ = 0.7
#Σ = [ρ^(abs(i-j)) for i=1:p,j=1:p]
#Σ_chol = cholesky(Σ)
X = randn(n, p)# * Σ_chol.UL

αs = sqrt.([4.0;8.0;12.0])#r#ange(2.0, 2.75, 3.5; length=3)#1.0:5.0
β = random_betas(grp, αs)
group_summary(grp, β, norm)
sum(abs2, β)
Y = X*β .+ σ .* randn(n)

tmp = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp)
fit!(tmp, λωλας_λ(tmp))

mom = MomentTunerSetup(tmp)
#scatter( αs.^2, SigmaRidgeRegression.get_αs_squared(mom,1.0))
#plot!(αs.^2,αs.^2)
σs_squared1 = range(0.0, 34; length=200)
mypath1 = sigma_squared_path(tmp, mom, σs_squared1)

#with gr
#plot(sqrt.(σs_squared1), mypath1.loos)
pl_left = plot(sqrt.(σs_squared1), mypath1.λs, legend=:topleft, color=main_cols, 
            linestyle=[ :dot :dashdot :dash], xlab=L"\sigmacv",
			ylab=L"\hat{\lambda}(\sigmacv)",
			label=["Group 1" "Group 2" "Group 3"]);
pl_right = plot(sqrt.(σs_squared1), mypath1.loos, color=:darkblue, 
           xlab=L"\sigmacv", ylab=L"\loo(\sigmacv)", label=nothing);
pl_both = plot(pl_left,pl_right, title=["(a)" "(b)"],
             size=(550,250));
savefig(pl_both, "intro_figure.tex")

σs_squared2 = range(0.0, 170.0; length=100)
mypath2 = sigma_squared_path(tmp, mom, σs_squared2)
#plot(σs_squared2, mypath.loos)


pl_left_tree = plot(sqrt.(σs_squared2), mypath2.λs, 
            ylim = (0,15), legend=:topleft, color=main_cols, 
            linestyle=[ :dot :dashdot :dash], 
			xlab=L"\sigmacv",
			ylab= L"\hat{\lambda}(\sigmacv)",
			label=["Group 1" "Group 2" "Group 3"]);

			four_cols_rep = hcat([fill(col, 1, 30) for col in   [:black; :purple; :green]]...)


pl_right_tree = plot(sqrt.(σs_squared2), mypath2.βs, alpha=0.6, 
                  linewidth=0.5,
                  label="", 
				  ylab= L"\hat{w}(\hat{\lambda}(\sigmacv))",
				  xlab=L"\sigmacv",
			      color=four_cols_rep, ylim=(-2.2,2.2))

pl_both_tree = plot(pl_left_tree,pl_right_tree, 
                title=["(a)" "(b)"], size=(550,220));
savefig(pl_both_tree, "christmas_tree.tex")
