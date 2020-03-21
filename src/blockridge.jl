Base.@kwdef struct BasicGroupRidgeWorkspace{SYM<:Symmetric,
                                C<:Cholesky,
                                M<:AbstractMatrix,
                                V<:AbstractVector}
    X::M
    Y::V
    groups::GroupedFeatures
    n::Integer = size(X,1)
    p::Integer = size(X,2)
    λs::V = ones(groups.num_groups)
    XtX::SYM = Symmetric(X'*X ./n)
    XtY::V = X'*Y./n
    XtXpΛ::SYM = XtX + 1.0*I
    XtXpΛ_chol::C = cholesky!(XtXpΛ)
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
    @show typeof(rdg.λs)
    rdg.λs .= λs
    rdg.XtXpΛ .= Symmetric(rdg.XtX + Diagonal(group_expand(rdg.groups, λs)))
    cholesky!(rdg.XtXpΛ)
    ldiv!(rdg.β_curr, rdg.XtXpΛ_chol, rdg.XtY)
    mul!(rdg.Y_hat, rdg.X, rdg.β_curr)
    ldiv!(rdg.XtXpΛ_div_Xt, rdg.XtXpΛ_chol, rdg.X')
    rdg.XtXpΛ_div_Xt ./= rdg.n
    _prod_diagonals!(rdg.leverage_store, rdg.X, rdg.XtXpΛ_div_Xt)
    loo_error(rdg)
end

function λωλας_λ(rdg; multiplier=0.1)
   multiplier*rdg.p^2/rdg.n/tr(rdg.XtX)
end

function max_σ_squared(rdg)
   mean(abs2, rdg.Y)
end





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
    M_matrix = rdg.XtXpΛ_chol\rdg.XtX
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
    max( u./ v)
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
        λs[i,:] = λs_tmp
        loos_hat[i] = fit!(rdg, λs_tmp)
        βs[i,:] = rdg.β_curr
    end
    (λs = λs, loos = loos_hat, βs=βs)
end