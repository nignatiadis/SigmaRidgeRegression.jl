"""
`AbstractRidgePredictor` is supposed to implement the interface
* `update_λs!`
* `trace_XtX`
* `LinearAlgebra.ldiv!`
* `Base.\`
Concrete subtypes available are `CholeskyRidgePredictor` and
`WoodburyRidgePredictor`.
"""
abstract type AbstractRidgePredictor end 

struct CholeskyRidgePredictor{SYM<:Symmetric,
                                C<:Cholesky} <: AbstractRidgePredictor
   XtX::SYM
   XtXpΛ::SYM
   XtXpΛ_chol::C
end

function trace_XtX(chol::CholeskyRidgePredictor)
    tr(chol.XtX)
end 
    
function CholeskyRidgePredictor(X)
    (n,p) = size(X)
    XtX = Symmetric(X'*X ./n)
    XtXpΛ = XtX + 1.0*I
    XtXpΛ_chol = cholesky!(XtXpΛ)
    CholeskyRidgePredictor(XtX, XtXpΛ, XtXpΛ_chol)
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


Base.@kwdef struct BasicGroupRidgeWorkspace{CP<:AbstractRidgePredictor,
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
end



ngroups(rdg::BasicGroupRidgeWorkspace) = ngroups(rdg.groups)

function _prod_diagonals!(Y, A, B)
    @inbounds for j ∈ 1:size(A,1)
        Y[j] = 0
        @inbounds for i ∈ 1:size(A,2)
            Y[j] += A[j,i]*B[i,j]
        end
    end
    Y
end

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

function λωλας_λ(rdg; multiplier=0.1)
   multiplier*rdg.p^2/rdg.n/tr(rdg.XtXpΛ_chol.XtX) #TODO 2
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
    M_matrix = rdg.XtXpΛ_chol\rdg.XtXpΛ_chol.XtX #TODO 1
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
        @show λs_tmp
        @show typeof(λs_tmp)
        @show typeof(λs)
        @show size(λs)

        λs[i,:] = λs_tmp
        loos_hat[i] = fit!(rdg, λs_tmp)
        βs[i,:] = rdg.β_curr
    end
    (λs = λs, loos = loos_hat, βs=βs)
end