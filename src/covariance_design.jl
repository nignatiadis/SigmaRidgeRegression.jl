abstract type CovarianceDesign{T} end

get_Σ(mat) = mat
nfeatures(mat) = size(mat, 1)

function spectrum(mat)
    eigs = eigvals(mat)
    probs = fill(1/length(eigs), length(eigs))
    DiscreteNonParametric(eigs, probs)
end

spectrum(cov::CovarianceDesign) = spectrum(get_Σ(cov))

nfeatures(cov::CovarianceDesign) = cov.p

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

get_Σ(cov::ExponentialOrderStatsCovarianceDesign) = Diagonal(support(spectrum(cov)))


struct BlockCovarianceDesign{T, S <: CovarianceDesign{T}, G} <: CovarianceDesign{T}
    blocks::Vector{S}
    groups::G
end

function BlockCovarianceDesign(blocks::Vector{S}) where S<:CovarianceDesign{Missing}
    BlockCovarianceDesign(blocks, missing)
end

nfeatures(cov::BlockCovarianceDesign) = sum(nfeatures.(cov.blocks))

function get_Σ(blockdesign::BlockCovarianceDesign)
    BlockDiagonal(get_Σ.(blockdesign.blocks))
end

function spectrum(blockdesign::BlockCovarianceDesign)
    @unpack blocks, groups = blockdesign
    spectra = spectrum.(blocks)
    mixing_prop = groups.ps ./ groups.p
    MixtureModel(spectra, mixing_prop)
end


function simulate_rotated_design(cov::BlockCovarianceDesign, n; rotated_measure = Normal())
    hcat(simulate_rotated_design.(cov.blocks, n; rotated_measure=rotated_measure)...)
end


# Set groups
function set_groups(design::CovarianceDesign, p::Integer)
    @set design.p = p
end

function set_groups(design::CovarianceDesign, groups::GroupedFeatures)
    set_groups(design, nfeatures(groups))
end

function set_groups(blockdesign::BlockCovarianceDesign, groups::GroupedFeatures)
    updated_blocks = set_groups.(blockdesign.blocks, groups.ps)
    BlockCovarianceDesign(updated_blocks, groups)
end
