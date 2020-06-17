
struct GroupedFeatures
    ps::Vector{Int}
    p::Int
    num_groups::Int
end

function GroupedFeatures(; group_size::Int, num_groups::Int)
    GroupedFeatures(fill(group_size, num_groups))
end

GroupedFeatures(ps) = GroupedFeatures(ps, sum(ps), length(ps))

ngroups(gr::GroupedFeatures) = gr.num_groups
nfeatures(gr::GroupedFeatures) = gr.p

function group_idx(gr::GroupedFeatures, i::Integer)
    starts = cumsum([1;gr.ps])[1:end-1]
    ends = cumsum(gr.ps)
    starts[i]:ends[i]
end


function group_summary(gr::GroupedFeatures, vec::AbstractVector, f)
    ps = gr.ps
    num_groups = gr.num_groups
    starts = cumsum([1;ps])[1:end-1]
    ends = cumsum(ps)
    el = eltype(f(vec))
    output = Vector{el}(undef, num_groups)
    for g=1:num_groups
        output[g] = f(vec[starts[g]:ends[g]])
    end
    output
end

function group_expand(gr::GroupedFeatures, vec::AbstractVector)
    arr = zeros(eltype(vec), gr.p)
    for i=1:gr.num_groups
        arr[group_idx(gr, i)] .= vec[i]
    end
    arr
end

function group_expand(gr::GroupedFeatures, el::Number)
    fill(el, gr.p)
end

function random_betas(gr::GroupedFeatures, αs)
    ps = gr.ps
    βs = zeros(eltype(αs), gr.p)
    for i=1:gr.num_groups
        βs[group_idx(gr, i)] .= randn(ps[i]) .* sqrt(αs[i]^2/ps[i])
    end
    βs
end