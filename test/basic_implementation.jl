using Plots
using LaTeXStrings
using CategoricalArrays
using GLM
using Distributions

using SigmaRidgeRegression
using LinearAlgebra
using StatsBase
using Plots

using Random


tmp_X = whiten_covariates(X, Σ_chol)

cov(tmp_X)
estimate_var(DickerMoments(),X,Y; Σ = Σ_chol)


using ForwardDiff

σ = 5.0
#gr = GroupedFeatures(repeat([200],5))
Random.seed!(1)
σ = 5.0
grp = GroupedFeatures([30;30;30;30;30;30])
n = 400
p = grp.p

ρ = 0.7
Σ = [ρ^(abs(i-j)) for i=1:p,j=1:p]
Σ_chol = cholesky(Σ)
X = randn(n, p) * Σ_chol.UL
Xnew = randn(10000, p) * Σ_chol.UL

using Statistics
cov(X)
#X= randn(n, p) .+ randn(n)
αs = sqrt.(range(4.0,12.0,length=ngroups(grp)))#r#ange(2.0, 2.75, 3.5; length=3)#1.0:5.0
β = random_betas(grp, αs)
group_summary(grp, β, norm)
sum(abs2, β)
Y = X*β .+ σ .* randn(n)

tmp = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp)

λωλας_λ(tmp)

λs


σ_squared_max(mom)
using StatsBase

StatsBase.fit!(tmp, λωλας_λ(tmp))
mom = MomentTunerSetup(tmp)



tune_σ(1.0)
SigmaRidgeRegression.


min1 = opt_res.minimizer
lambda_min1 = get_λs(mom, min1)
min_val1 = opt_res.minimum
β1 = copy(tmp.β_curr)


function tune_λ(λ)
    fit!(tmp, λ)
end

lower_box_constraint = fill(0.0, 6)
upper_box_constraint = fill(Inf, 6)

opt_res2 = optimize(tune_λ, 
                    lower_box_constraint, upper_box_constraint,
                    lambda_min1)
opt_res2.minimizer
opt_res2.minimizer
opt_res2.minimum
fit!(tmp, opt_res2.minimizer)
β2 = copy(tmp.β_curr)


mean(abs2, X*(β-β1))
mean(abs2, X*(β-β2))

opt_res3 = optimize(tune_λ, 
                    lower_box_constraint, upper_box_constraint,
                    fill(1.0,6))
opt_res3.minimizer
opt_res3.minimizer
#opt_res3.minimum

opt_res.minimizer

oracle_λ = σ^2 ./ αs.^2 .* 30 ./ n


using Optim

using Plots
scatter( αs.^2, SigmaRidgeRegression.get_αs_squared(mom,1.0))
plot!(αs.^2,αs.^2)
σs_squared = range(0.01, 3.0; length=100)
mypath1 = sigma_squared_path(tmp, mom, σs_squared)
plot(σs_squared, mypath1.loos)

using Plots
plot(σs_squared, mypath1.λs)

λs = vcat([get_λs(mom, s)' for s in σs_squared]...)
pl = plot(σs_squared, λs)

σs_squared = range(0.01, 50.0; length=100)
mypath = sigma_squared_path(tmp, mom, σs_squared)
plot(σs_squared, mypath.loos)

four_cols_rep = hcat([fill(col, 1, 30) for col in   ["#440154"; "#31688E"; "#35B779"]]...)
linetype_rep = hcat([fill(col, 1, 25) for col in [:dash,:dot,:dashdot, :solid]]...)

[:black,"#9818d6","#ff5151"]
#66c2a5
#fc8d62
#8da0cb
#e78ac3
four_cols_rep = hcat([fill(col, 1, 30) for col in   [:black; :purple; :green]]...)

plot(σs_squared, mypath.βs, alpha=0.8, linewidth=1.0,label="", color=four_cols_rep, ylim=(-2,2))
using Plots
using PlotThemes
theme(:default)
opt_λ_empirical = σ^2/norm(β)^2*γ
opt_λ = σ^2/20.0*p/n



p/n



max_σ_squared(tmp)
fit!(tmp, 0.050)



mom = MomentTunerSetup(tmp)
mom.M_squared


my
find_λs_squared(mom, 0.2)

find_αs_squared(mom, 1.0)

using NonNegLeastSquares


using Plots
using LaTeXStrings
pgfplotsx()
, ylab=L"\hat{\lambda}(\sigmacv)", xlab=L"\sigmacv", size=(300,200));
pl = plot(σs_squared, get_α,  size=(500,400))

savefig(pl, "pl2.tex")

mom.N_norms_squared
find_αs(mom, 2.0)


using NonNegLeastSquares


find_αs(mom, 20.0)






using RandomMatrices

myrot = rand(GaussianHermite{1},n,1)

n_test = 20_000
n = 10_000
p = 1_000
σ = 1.0


γ = p/n
X = randn(n, p) .+ randn(n) #strong positive correlations.

Z = randn(p,p)
Z_qr_Q = Matrix(qr(Z).Q)

my_eigs = [fill(5, 500);fill(1, 500)]
Σ = Z_qr_Q * Diagonal(my_eigs) * Z_qr_Q'

X = real.(Matrix((sqrt(Σ)*randn(p,n))'))

X_qr.Q'*X_qr.Q


Y_test = X_test*β .+ σ .* randn(n_test)


1.41
tmp.XtXpΛ



size(tmp.X)

size(tmp.XtXpΛ_div_Xt)
size(tmp.Y)






StatsBase.fit!(tmp, 2.0)

diag(tmp.X*inv(tmp.XtX + Diagonal(group_expand(tmp.groups, tmp.λs)))tmp.X')

tmp.leverage_store


hat_matrix = tmp.X*inv(tmp.XtX + Diagonal(group_expand(tmp.groups, tmp.λs)))tmp.X'./n
hat_matrix*tmp.Y ≈ tmp.X * tmp.β_curr

diag(hat_matrix) ≈ tmp.leverage_store
mean(diag(hat_matrix))
mean(tmp.leverage_store)


using ForwardDiff
using Zygote

Zygote.gradient(λ->fit!(tmp, λ), fill(1.0,5))

ForwardDiff.gradient(λ->fit!(tmp, λ), fill(1.0,5))

ForwardDiff.Hess

loo_error(tmp)

mse_ridge(tmp, X_test, Y_test)

λs = range(0.00, 3.0; length=50)

mses_hat = zeros(length(λs))
loos_hat = zeros(length(λs))
for (i, λ) in enumerate(λs)
    fit!(tmp, λ)
    mses_hat[i] = mse_ridge(tmp, X_test, Y_test)
    loos_hat[i] = loo_error(tmp)
end




using Plots
using LaTeXStrings

plot(λs, [mses_hat loos_hat], color=["black" "blue"], linestyle=[:solid :dot],
                    label=["MSE" "LOO"], xlabel=L"\lambda")

plot(λs, mses_hat .- loos_hata)

fits = [solve_ridge(XtX, XtY, X, Y, λ) for λ in λs]
mses_hat = [mse_ridge(X_test, Y_test, fit[:β_hat]) for fit in fits]
loos_hat = [fit[:LOO_error] for fit in fits]

using Plots
using LaTeXStrings

vline!([opt_λ_empirical opt_λ], color=[:green :red])



true_error =



tmp2 = fit!(tmp., 1.0:5.0)
tmp2.XtXpΛ_chol\(tmp.XtX + Diagonal(group_expand(tmp.groups, 1.0:5.0))) ≈ I
(tmp.XtX + Diagonal(group_expand(tmp.groups, 1.0:5.0)))\tmp.XtY ≈ tmp.β_curr

function BasicGroupRidgeWorkspace(X, Y, groups)

end
mychol = cholesky(XtX)

vs = XtX
ldiv!(XtX, mychol, I)


cholesky(XtX)
isa(vs, AbstractMatrix)



#function rand()






#n_test = 10_000



β = randn(p) .* sqrt(α^2/p)
norm(β)^2







XtY = X'*Y./n
function solve_ridge(XtX::Symmetric, XtY, X, Y, λ; compute_M_matrix=false)
    n, p = size(X)
    chol_trans_inv = inv(cholesky(XtX + λ*I(p)))

    β_hat = chol_trans_inv*XtY

    hat_matrix = X*chol_trans_inv*X' ./ n

    Y_hat = X*β_hat

    LOO_error = norm( (Y .- Y_hat)./ ( 1.0 .- diag(hat_matrix)))^2 / n

    res= Dict(:β_hat => β_hat, :LOO_error => LOO_error, :λ => λ)

    if compute_M_matrix
        M_matrix = chol_trans_inv * XtX
        N_matrix = chol_trans_inv * X'./n

        res[:M_matrix] = M_matrix
        res[:N_matrix] = N_matrix

    end
    res
end













P_mat = ridge_sol[:N_matrix]


sol_matrix
sol_rhs
β_hat_norms

α_squared_hat = sol_matrix\sol_rhs

matrix_sol =


# want a fun:

# FeatureGroups

# repeat(..., FeatureGroups)

# + iterator protocol for groups.
# groupwise(Groups(), \beta::Vector, )
# groupwise(Groups(), \beta::Matrix, )
# groupwise(Groups(), \beta::Matrix, )


a = reshape(Vector(1:16), (4,4))
@which reduce(max, a, dims=1)



Σ = [ρ^(abs(i-j)) for i=1:p,j=1:p]

myinv = inv(Σ_chol.UL)


myinv*Σ*myinv'


X = randn(n, p) * Σ_chol.L


woodbury_playi

using WoodburyMatrices

A, B, D
bla = copy(tmp.λs)
bla[6] = Inf
A_tmp = Diagonal(group_expand(tmp.groups, bla))
wd = SymWoodbury(A_tmp, X', I(n)/n)
wd.Dp
wd.B

wd.

Dp = inv(n*I + wd.B'*(A_tmp\wd.B))

wd.Dp ≈ Dp 

≈ Dp
Dp = safeinv(safeinv(D) .+ B'*(A\B))


?Woodbury




Random.seed!(100)
σ = 4.0
grp = GroupedFeatures([300;3000;5000])
n = 400
p = grp.p


X = randn(n, p)# * Σ_chol.UL

αs = sqrt.([4.0;8.0;12.0])#r#ange(2.0, 2.75, 3.5; length=3)#1.0:5.0
β = random_betas(grp, αs)
group_summary(grp, β, norm)
sum(abs2, β)
Y = X*β .+ σ .* randn(n)

tmp = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp,
                              XtXpΛ_chol = WoodburyRidgePredictor(X))
fit!(tmp, λωλας_λ(tmp))

mom = MomentTunerSetup(tmp)
#scatter( αs.^2, SigmaRidgeRegression.get_αs_squared(mom,1.0))
#plot!(αs.^2,αs.^2)
σs_squared1 = range(0.0001, 34; length=30)
mypath1 = sigma_squared_path(tmp, mom, σs_squared1)

#with gr
plot(sqrt.(σs_squared1), mypath1.loos)

get_λs(mom, 4)