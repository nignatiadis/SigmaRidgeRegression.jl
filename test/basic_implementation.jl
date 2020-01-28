using StatsBase
using LinearAlgebra
using Plots
using LaTeXStrings
using CategoricalArrays

pgfplotsx()


n = 4_000
n_test = 10_000
p = 2_000
γ = p/n

X = randn(n, p)
X_test = randn(n_test, p)
σ = 1.0
α = 2.0
β = randn(p) .* sqrt(α^2/p)
norm(β)^2

Y = X*β .+ σ .* randn(n)
Y_test = X_test*β .+ σ .* randn(n_test)


opt_λ = σ^2/α^2*γ
opt_λ_empirical = σ^2/norm(β)^2*γ

λs = range(0, 3.0; length=100)

# only for p<n, rethink how to do p>n
# div by N!!!!!!!!!!!
XtX = Symmetric(X'*X ./n)
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
        res[:M_matrix] = M_matrix
    end
    res
end
function mse_ridge(X_test, Y_test, β)
    n = length(Y_test)
    norm(Y_test - X_test*β)^2 /n
end

fits = [solve_ridge(XtX, XtY, X, Y, λ) for λ in λs]
mses_hat = [mse_ridge(X_test, Y_test, fit[:β_hat]) for fit in fits]
loos_hat = [fit[:LOO_error] for fit in fits]

plot(λs, [mses_hat loos_hat], color=["black" "blue"], linestyle=[:solid :dot],
                    label=["MSE" "LOO"], xlabel=L"\lambda")
vline!([opt_λ_empirical opt_λ], color=[:green :red])



num_groups = 4
ps = fill(200, num_groups)
p = sum(ps)
feature_groups = CategoricalArray(vcat(fill.(1:num_groups, ps)...))


αs = fill(1.0, num_groups)
β = Vector{Float64}(undef, p)
for i=1:num_groups
    idx = findall(feature_groups .== i)
    β[idx] .= randn(ps[i]) .* sqrt(αs[i]^2/ps[i])
end

αs = range(1, 4, length=4)


# let us presume things have been ordered.

# generalize this.
starts = cumsum([1;ps])[1:end-1]
ends = cumsum(ps)

# sanity check
β_norms = Vector{Float64}(undef, num_groups)
for g=1:num_groups
    β_norms[g] = norm(β[starts[g]:ends[g]])^2
end
β_norms

X = randn(n, p)
XtX = Symmetric(X'*X ./n)
XtY = X'*Y./n
Y = X*β .+ σ .* randn(n)

ridge_sol = solve_ridge(XtX, XtY, X, Y, 0.1; compute_M_matrix=true)
β_hat = ridge_sol[:β_hat]

β_hat_norms = Vector{Float64}(undef, p)
for g=1:num_groups
    β_hat_norms[g] = norm(β_hat[starts[g]:ends[g]])^2
end

β_hat_norms
ridge_sol[:M_matrix]
