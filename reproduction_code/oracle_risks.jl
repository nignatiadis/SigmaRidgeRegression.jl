using Polynomials
using Roots
using Optim

using Plots
using FiniteDifferences


tmp = fixed_point([1.0,1.0], [0.5,0.5], [2.0, 2.0])

hs = [1.0;1.0]
γs = [1.0;1.0]
αs = [2.0; 1.0]

λs_opt = γs ./ αs.^2

λs = [2.0; 1.0]


function fixed_point_function(hs, γs, λs)
	γ = sum(γs)
	fixed_point_f = f -> f - sum( γs./γ./(λs./hs .+ 1 ./(1 .+ γ*f))  )
	find_zero(fixed_point_f, (0.0, 1000.0))
end


fixed_point_function(hs, γs, λs)

function risk_formula(hs, γs, αs, λs)
	γ = sum(γs)
	fixed_pt = λs_tilde -> fixed_point_function(hs, γs, λs_tilde)
	f = fixed_pt(λs)
	∇f = grad(central_fdm(5, 1), fixed_pt, λs)[1]
	1 +	γ*f + sum(γ ./ γs .* (γs .* λs - αs.^2 .* λs.^2) .* ∇f)
end

risk_formula(hs, γs, αs, λs)
risk_formula(hs, γs, αs, λs_opt)


f = find_zero( tmp, (0.0, 100.0))

xs = range(0.01, 1.0, length=100)
ys = range(0.01, 1.5, length=100)

λs_opt


myf = (x,y) -> risk_formula(hs, γs, αs, [x;y])
myf_optim = xs-> risk_formula(hs, γs, αs, xs)

myf(0.001,2.0)

risk_formula(hs, γs, αs, [0.25;1])

tmp = optimize(myf_optim, [0.1;0.1], [Inf; Inf], [1.0; 1.5], Fminbox(LBFGS()))
tmp.optimizer

myf(xs[1],ys[1])

pgfplotsx()
plot(xs, ys, myf, st = :surface)
gr()
function rosenbrock(x::Vector)
  return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
x, y = -1.5:0.1:1.5, -1.5:0.1:1.5
z = Surface((x,y)->myf([x,y]), x, y)
surface(x,y,z, linealpha = 0.3)

plotly()