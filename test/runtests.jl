using SigmaRidgeRegression
using Test
using LinearAlgebra
using Random
import StatsBase

@testset "Woodbury and Cholesky" begin
    Random.seed!(1)
    σ = 5.0
    grp = GroupedFeatures([30;30;30;30;30;30])
    n = 400
    p = grp.p
    X = randn(n, p)
    αs = sqrt.(range(4.0,12.0,length=ngroups(grp)))#r#ange(2.0, 2.75, 3.5; length=3)#1.0:5.0
    β = random_betas(grp, αs)
    Y = X*β .+ σ .* randn(n)

    tmp_chol = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp)

    tmp_woodb = BasicGroupRidgeWorkspace(X=X, Y=Y, groups=grp,
                               XtXpΛ_chol = WoodburyRidgePredictor(X))

    @test SigmaRidgeRegression.λωλας_λ(tmp_chol) ≈ SigmaRidgeRegression.λωλας_λ(tmp_woodb)
    beta_chol = tmp_chol.XtXpΛ_chol \ tmp_woodb.XtY
    beta_wdb =  tmp_woodb.XtXpΛ_chol \ tmp_woodb.XtY
    @test beta_chol ≈ beta_wdb

    tmp_chol.XtXpΛ_chol \ tmp_chol.X' ≈ tmp_woodb.XtXpΛ_chol \ tmp_woodb.X'

    ldiv_chol = ldiv!(tmp_chol.XtXpΛ_div_Xt, tmp_chol.XtXpΛ_chol, tmp_chol.X')
    ldiv_wdb = ldiv!(tmp_woodb.XtXpΛ_div_Xt, tmp_woodb.XtXpΛ_chol, tmp_woodb.X')

    @test ldiv_chol ≈ ldiv_wdb

    loo_chol = StatsBase.fit!(tmp_chol, SigmaRidgeRegression.λωλας_λ(tmp_chol))
    loo_wdb = StatsBase.fit!(tmp_woodb, SigmaRidgeRegression.λωλας_λ(tmp_chol))

    @test tmp_woodb.β_curr ≈ tmp_chol.β_curr
end



@testset "MMI" begin
    include("test_mmi.jl")
end
#include("test_grouplasso.jl")
include("test_covariance.jl")
include("test_simulations.jl")

@testset "Theoretical risks" begin
    include("test_theoretical_risks.jl")
end
# TO ADD
# Compare: - GroupLasso AGAINST R
#          - Check group lasso lambda_max
#          - test the TunedRidgeRegressor type
#          - test σ_max  (probably won't increase coverage but...)
