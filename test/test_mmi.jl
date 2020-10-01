using MLJLinearModels
using MLJ
using MLJModelInterface
import StatsBase
using SigmaRidgeRegression
using Test
using Random
using Plots

const MMI = MLJModelInterface



# Mostly here to check implementation.





# Let us first test if the code works for a single predictor

single_group_ridge_reg = SingleGroupRidgeRegressor(:cholesky, 0.0)
single_group_ridge_reg_woodbury = SingleGroupRidgeRegressor(:woodbury, 0.0)
mljlm_ridge = RidgeRegressor(lambda=0.0, fit_intercept=false)


Random.seed!(1)
n = 100 
p = 80
X = randn(n, p)
Xtable = MLJ.table(X);
βs = randn(p)./sqrt(p)
Y = X*βs .+ randn(n)
grps = GroupedFeatures([p]);



single_group_ridge_machine = machine(single_group_ridge_reg, Xtable, Y)
single_group_ridge_woodbury_machine = machine(single_group_ridge_reg_woodbury, Xtable, Y)
mljlm_ridge_machine = machine(mljlm_ridge, Xtable, Y) 

fit!(single_group_ridge_machine)
@test_broken fit!(single_group_ridge_woodbury_machine) #cannot handle 0.0

fit!(mljlm_ridge_machine)

@test predict(single_group_ridge_machine) ≈ predict(mljlm_ridge_machine)

single_group_ridge_machine.model.λ = 1.0
single_group_ridge_woodbury_machine.model.λ = 1.0
mljlm_ridge_machine.model.lambda = 1.0 * n

fit!(single_group_ridge_machine)
fit!(single_group_ridge_woodbury_machine)
fit!(mljlm_ridge_machine)

@test predict(single_group_ridge_machine) ≈ predict(mljlm_ridge_machine)
@test predict(single_group_ridge_machine) ≈ predict(single_group_ridge_woodbury_machine)



# Start checking LOOCVRidgeRegressor

loocv_ridge = LooCVRidgeRegressor()
loocv_ridge_machine = machine(loocv_ridge, X, Y)
@time fit!(loocv_ridge_machine)

λ_max = loocv_ridge_machine.report.λ_max
λ_range = range(single_group_ridge_reg, :λ, lower=loocv_ridge.λ_min_ratio*λ_max, upper=λ_max, scale=loocv_ridge.scale)

## Compare against brute froce predictions
loocv_ridge_bruteforce = TunedModel(model = single_group_ridge_reg,
                                    tuning = Grid(resolution=loocv_ridge.n), 
                                    resampling=  CV(nfolds=n),
                                    measure = l2,
                                    range = λ_range)

loocv_ridge_bruteforce_machine = machine(loocv_ridge_bruteforce, X,Y)
@time fit!(loocv_ridge_bruteforce_machine)

@test predict(loocv_ridge_machine) == predict(loocv_ridge_bruteforce_machine)

Xnew = MLJ.table(randn(10, p));
@test predict(loocv_ridge_machine, Xnew) == predict(loocv_ridge_bruteforce_machine, Xnew)

## visualize
plot(loocv_ridge_machine.report.λs, loocv_ridge_machine.report.loos, scale=loocv_ridge_machine.model.scale, label="loo shortcut")
vline!([loocv_ridge_machine.report.best_λ])
single_ridge_cv_curve_loo = learning_curve(single_group_ridge_machine, range=λ_range, resampling=CV(nfolds=n), measure=l2)
plot!(single_ridge_cv_curve_loo.parameter_values,
     single_ridge_cv_curve_loo.measurements,
     xlab=single_ridge_cv_curve_loo.parameter_name,
     xscale=single_ridge_cv_curve_loo.parameter_scale,
     label = "LOO brute force")

# Let us also try with other number of folds

single_ridge_cv = TunedModel(model = single_group_ridge_reg,
                             tuning = Grid(resolution=100), 
                             resampling=  CV(nfolds=5),
                             measure = l2,
                             range = λ_range)

single_ridge_cv_machine = machine(single_ridge_cv, Xtable, Y)

single_ridge_cv_curve_5fold = learning_curve(single_group_ridge_machine, range=λ_range, resampling=CV(nfolds=5), measure=l2)


plot(single_ridge_cv_curve_5fold.parameter_values,
     single_ridge_cv_curve_5fold.measurements,
     xlab=single_ridge_cv_curve_5fold.parameter_name,
     xscale=single_ridge_cv_curve_5fold.parameter_scale,
     label = "5-fold",
     ylab = "CV estimate of RMS error")
plot!(single_ridge_cv_curve_loo.parameter_values,
     single_ridge_cv_curve_loo.measurements,
     xlab=single_ridge_cv_curve_loo.parameter_name,
     xscale=single_ridge_cv_curve_loo.parameter_scale,
     label = "LOOCV")

vline!([0.8])

tmp_eval = evaluate!(single_group_ridge_machine, resampling=CV(nfolds=n), measure=l2)
@test tmp_eval.measurement[1] ≈ loo_error(single_group_ridge_machine.cache) atol=0.02



# Check multiridge

multiridge = MultiGroupRidgeRegressor(GroupedFeatures([30;50]))
loocv_multiridge = LooCVRidgeRegressor(ridge=multiridge, n=10)

loocv_multiridge_mach = machine(loocv_multiridge, X, Y)
fit!(loocv_multiridge_mach)

multiridge_ranges = loocv_multiridge_mach.report.λ_range
multiridge_loo_bruteforce = TunedModel(model=multiridge, 
                                       resampling=CV(nfolds=n), 
                                       tuning=Grid(resolution=loocv_multiridge.n), 
                                       range=multiridge_ranges, 
                                       measure=l2)

multiridge_loo_bruteforce_machine = machine(multiridge_loo_bruteforce, X, Y)
fit!(multiridge_loo_bruteforce_machine)
@test values(multiridge_loo_bruteforce_machine.report.best_model.λ) == values(loocv_multiridge_mach.report.best_λ)
@test predict(multiridge_loo_bruteforce_machine) == predict(loocv_multiridge_mach)
