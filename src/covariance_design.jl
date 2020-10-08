abstract type CovarianceDesign{T} end

get_Σ(mat) = mat
nfeatures(mat) = size(mat, 1)

function spectrum(mat)
    eigs = eigvals(mat)
    probs = fill(1/length(eigs), length(eigs))
    DiscreteNonParametric(eigs, probs)
end

nfeatures(cov::CovarianceDesign) = cov.p

function (cov::CovarianceDesign)(p::Int)
    cov = @set cov.p = p
    cov
end

function simulate_rotated_design(cov, n; rotated_measure = Normal())
    Σ = get_Σ(cov)
    Σ_chol = cholesky(Σ)
    p = nfeatures(cov)
    Z = rand(rotated_measure, n, p)
    X = randn(n, p) * Σ_chol.UL
    X
end

Base.@kwdef struct AR1Design{P<:Union{Missing,Int}} <: CovarianceDesign{P}
    p::P = missing
    ρ = 0.7
end

function get_Σ(cov::AR1Design{Int})
    p = nfeatures(cov)
    ρ = cov.ρ
    Σ = [ρ^(abs(i - j)) for i = 1:p, j = 1:p]
    Σ
end

abstract type DiagonalCovarianceDesign{T} <: CovarianceDesign{T} end

Base.@kwdef struct IdentityCovarianceDesign{P<:Union{Missing,Int}} <:
                   DiagonalCovarianceDesign{P}
    p::P = missing
end

spectrum(::IdentityCovarianceDesign) = DiscreteNonParametric([1.0],[1.0])

function get_Σ(cov::IdentityCovarianceDesign{Int})
    I(cov.p)
end

Base.@kwdef struct UniformScalingCovarianceDesign{P<:Union{Missing,Int}} <:
                   DiagonalCovarianceDesign{P}
    scaling::Float64 = 1.0
    p::P = missing
end

spectrum(unif::UniformScalingCovarianceDesign) = DiscreteNonParametric([unif.scaling],[1.0])

function get_Σ(cov::UniformScalingCovarianceDesign{Int})
    (cov.scaling * I)(cov.p)
end

Base.@kwdef struct ExponentialOrderStatsCovarianceDesign{P<:Union{Missing,Int}} <:
                   DiagonalCovarianceDesign{P}
    p::P = missing
    rate::Float64
end

function spectrum(cov::ExponentialOrderStatsCovarianceDesign)
    p = cov.p
    rate = cov.rate
    tmp = range(1 / (2p); stop = 1 - 1 / (2p), length = p)
    eigs = 1 / rate .* log.(1 ./ tmp)
    DiscreteNonParametric(eigs, fill(1/p, p))
end

struct BlockCovarianceDesign{T, S <: CovarianceDesign{T}} <: CovarianceDesign{T}
    blocks::Vector{S}
    groups::GroupedFeatures
end

function get_Σ(blockdesign::BlockCovarianceDesign)
    BlockDiagonal(get_Σ.(blockdesign.blocks))
end

function spectrum(blockdesign::BlockCovarianceDesign)
    @unpack blocks, groups = blockdesign
    spectra = spectrum.(blocks)
    mixing_prop = groups.ps ./ groups.p
    MixtureModel(spectra, mixing_prop)
end
