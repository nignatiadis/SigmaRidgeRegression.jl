"""
`AbstractRidgePredictor` is supposed to implement the interface
* `update_λs!`
* `trace_XtX`
* `XtXpΛ_ldiv_XtX`
* `LinearAlgebra.ldiv!`
* `Base.\`
Concrete subtypes available are `CholeskyRidgePredictor` and
`WoodburyRidgePredictor`.
"""
abstract type AbstractRidgePredictor end 

"""
Used typically for p < n.
"""  
struct CholeskyRidgePredictor{M<:AbstractMatrix,
                              SYM<:Symmetric,
                              C<:Cholesky} <: AbstractRidgePredictor
   X::M
   XtX::SYM
   XtXpΛ::SYM
   XtXpΛ_chol::C
end

  
function CholeskyRidgePredictor(X)
    (n,p) = size(X)
    XtX = Symmetric(X'*X ./n)
    XtXpΛ = XtX + 1.0*I
    XtXpΛ_chol = cholesky!(XtXpΛ)
    CholeskyRidgePredictor(X, XtX, XtXpΛ, XtXpΛ_chol)
end 

function update_λs!(chol::CholeskyRidgePredictor, groups, λs)
    chol.XtXpΛ .= Symmetric(chol.XtX + Diagonal(group_expand(groups, λs)))
    cholesky!(chol.XtXpΛ)
end

function ldiv!(A, chol::CholeskyRidgePredictor, B)
    ldiv!(A, chol.XtXpΛ_chol, B)
end

function \(chol::CholeskyRidgePredictor, B)
   chol.XtXpΛ_chol \ B
end

function XtXpΛ_ldiv_XtX(chol::CholeskyRidgePredictor)
    chol.XtXpΛ_chol\chol.XtX
end 

function trace_XtX(chol::CholeskyRidgePredictor)
    tr(chol.XtX)
end 


"""
Used typically for p >> n and n reasonably small
"""
struct WoodburyRidgePredictor{M<:AbstractMatrix,
                              S<:SymWoodbury} <: SigmaRidgeRegression.AbstractRidgePredictor
   X::M
   wdb::S
end
    
function WoodburyRidgePredictor(X)
    (n,p) = size(X)
    wdb = SymWoodbury(1.0*I(p), X', I(n)/n)
    WoodburyRidgePredictor(X, wdb)
end
 

#Hi Tim,

#It would be very useful if there could be an implementation of
#```julia
#ldiv!(dest::AbstracMatrix, W::AbstractWoodbury, B::AbstractMatrix)
#```
#Right now I think this only works with `AbstractVector`. Before implementing and filing a pull request, I was wondering whether you think it is an OK approach to 

#```julia
# for i=1:ncols 
#      ldiv!(view(dest, :, i), A, view(B,:,i))
#end
#```
#------------------------------------------------------------------------ 
# TODO: Fix the following two things upstream on WoodburyMatrices.jl
#------------------------------------------------------------------------
function _ldiv!(dest, W::SymWoodbury, A::Diagonal, B)
    WoodburyMatrices.myldiv!(W.tmpN1, A, B)
    mul!(W.tmpk1, W.V, W.tmpN1)
    mul!(W.tmpk2, W.Cp, W.tmpk1)
    mul!(W.tmpN2, W.U, W.tmpk2)
    WoodburyMatrices.myldiv!(A, W.tmpN2)
    for i = 1:length(W.tmpN2)
        @inbounds dest[i] = W.tmpN1[i] - W.tmpN2[i]
    end
    return dest
end

#-----------------------------------------------------------------------
function ldiv!(Y::AbstractMatrix, A::SymWoodbury, B::AbstractMatrix)
    ncols = size(B,2)
    for i=1:ncols 
        ldiv!(view(Y, :, i), A, view(B,:,i))
    end
    Y
end
#-----------------------------------------------------------------------

function update_λs!(wbpred::WoodburyRidgePredictor, groups, λs)
    wdb = wbpred.wdb
    n = size(wdb.D,1)
    A =  Diagonal(group_expand(groups, λs))
    wdb.A .= A
    wdb.Dp .= inv(n*I + wdb.B'*(A\wdb.B))
end


function ldiv!(A, wbpred::WoodburyRidgePredictor, B)
    ldiv!(A, wbpred.wdb, B)
end

function \(wbpred::WoodburyRidgePredictor, B)
   wbpred.wdb \ B
end


function XtXpΛ_ldiv_XtX(wbpred::WoodburyRidgePredictor)
    n = size(wbpred.X, 1)
    (wbpred.wdb\wbpred.X')*wbpred.X ./n
end 

function trace_XtX(wbpred::WoodburyRidgePredictor)
    n = size(wbpred.X,1)
    # recall XtX here really is XtX/n
    tr(wbpred.X'*wbpred.X)/n #make more efficient later.
end 

Base.@kwdef mutable struct BasicGroupRidgeWorkspace{CP<:AbstractRidgePredictor,
                                M<:AbstractMatrix,
                                V<:AbstractVector}
    X::M
    Y::V
    groups::GroupedFeatures
    n::Integer = size(X,1)
    p::Integer = size(X,2)
    λs::V = ones(groups.num_groups)
    XtY::V = X'*Y./n
    XtXpΛ_chol::CP = CholeskyRidgePredictor(X)
    XtXpΛ_div_Xt::M = XtXpΛ_chol\X'.\n
    β_curr::V = XtXpΛ_chol\XtY
    leverage_store::V = zeros(n)
    Y_hat::V = X*β_curr
    cache = nothing
end



ngroups(rdg::BasicGroupRidgeWorkspace) = ngroups(rdg.groups)

# StatsBase.jl interace 
coef(rdg::BasicGroupRidgeWorkspace) = rdg.β_curr
islinear(rdg::BasicGroupRidgeWorkspace) = true
leverage(rdg::BasicGroupRidgeWorkspace) = rdg.leverage_store
modelmatrix(rdg::BasicGroupRidgeWorkspace) = rdg.X
response(rdg::BasicGroupRidgeWorkspace) = rdg.Y


function loo_error(rdg::BasicGroupRidgeWorkspace)
    mean(abs2.((rdg.Y .- rdg.Y_hat)./ ( 1.0 .- rdg.leverage_store)))
end

function mse_ridge(rdg::BasicGroupRidgeWorkspace, X_test, Y_test)
    mean(abs2.(Y_test - X_test*rdg.β_curr))
end


function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, λs)
    rdg.λs .= λs
    update_λs!(rdg.XtXpΛ_chol, rdg.groups, λs)
    #rdg.XtXpΛ .= Symmetric(rdg.XtX + Diagonal(group_expand(rdg.groups, λs)))
    #cholesky!(rdg.XtXpΛ)
    ldiv!(rdg.β_curr, rdg.XtXpΛ_chol, rdg.XtY)
    mul!(rdg.Y_hat, rdg.X, rdg.β_curr)
    ldiv!(rdg.XtXpΛ_div_Xt, rdg.XtXpΛ_chol, rdg.X')
    rdg.XtXpΛ_div_Xt ./= rdg.n
    _prod_diagonals!(rdg.leverage_store, rdg.X, rdg.XtXpΛ_div_Xt)
    loo_error(rdg)
end


"""
    λωλας_λ(rdg; multiplier=0.1)
    
Implements the Panagiotis Lolas rule of thumb for picking an optimal λ.    
"""
function λωλας_λ(rdg; multiplier=0.1)
   multiplier*rdg.p^2/rdg.n/trace_XtX(rdg.XtXpΛ_chol) #TODO 2s
end

#function max_σ_squared(rdg)
#   mean(abs2, rdg.Y)
#end





# Tuning through Moment Fitting



Base.@kwdef struct MomentTunerSetup{IV<:AbstractVector,
                               FV<:AbstractVector,
                               FM<:AbstractMatrix}
    ps::IV
    n::Integer
    beta_norms_squared::FV
    N_norms_squared::FV
    M_squared::FM
end

function MomentTunerSetup(rdg::BasicGroupRidgeWorkspace)
    grps = rdg.groups
    n = rdg.n
    ps = grps.ps
    ngroups = grps.num_groups
    beta_norms_squared = group_summary(grps, rdg.β_curr, x->sum(abs2,x))
    N_matrix = rdg.XtXpΛ_div_Xt #sqrt(n)*N from paper
    M_matrix =  XtXpΛ_ldiv_XtX(rdg.XtXpΛ_chol) #TODO 1
    N_norms_squared = Vector{eltype(beta_norms_squared)}(undef, ngroups)
    M_squared = Matrix{eltype(beta_norms_squared)}(undef, ngroups, ngroups)

    for g in 1:ngroups
        N_norms_squared[g]  = sum(abs2,N_matrix[group_idx(grps,g),:])
        for h in 1:ngroups
            # Mij  is entry (j,i)  and dived by p_i
            M_squared[h,g] = sum(abs2,M_matrix[group_idx(grps,g), group_idx(grps,h)])
        end
    end
    MomentTunerSetup(ps=ps, n=n, beta_norms_squared=beta_norms_squared,
                     N_norms_squared=N_norms_squared, M_squared=M_squared)

end

function σ_squared_max(mom::MomentTunerSetup)
    u = mom.beta_norms_squared
    v = mom.N_norms_squared
    maximum( u./ v)
end

function get_αs_squared(mom::MomentTunerSetup, σ_squared)
   rhs = mom.beta_norms_squared .- σ_squared .* mom.N_norms_squared
   α_sq_by_p = vec(nonneg_lsq(mom.M_squared,rhs;alg=:fnnls)) #  mom.M_squared\rhs\
   α_sq_by_p .* mom.ps
end

function get_λs(mom::MomentTunerSetup, σ_squared)
   αs_squared = get_αs_squared(mom, σ_squared)
   γs = mom.ps ./ mom.n
   σ_squared .* γs ./ αs_squared
end

function sigma_squared_path(rdg::BasicGroupRidgeWorkspace, 
                            mom::MomentTunerSetup,
                            σs_squared)
                           
    n_σs = length(σs_squared)
    n_groups = ngroups(rdg)
    loos_hat = zeros(n_σs)
    λs = zeros(n_σs, n_groups)
    βs = zeros(n_σs, rdg.groups.p)
    for (i, σ_squared) in enumerate(σs_squared)
        λs_tmp = get_λs(mom, σ_squared)
        λs[i,:] = λs_tmp
        loos_hat[i] = fit!(rdg, λs_tmp)
        βs[i,:] = rdg.β_curr
    end
    (λs = λs, loos = loos_hat, βs=βs)
end