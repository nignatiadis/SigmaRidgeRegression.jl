using CSV
using DataFrames
using StatsBase
using SigmaRidgeRegression
using Splines2
using LinearAlgebra

using Random
msd = CSV.File(joinpath(@__DIR__,"dataset","YearPredictionMSD.txt"), header=false) |> DataFrame

using Plots




msd_mat = Matrix(msd[:, 2:91])

mean(msd_mat, dims=1)[1:12]

var_features =  Matrix(msd[:, 14:25])
cov_features = Matrix(msd[:, 26:91])

cor_features = zeros(size(cov_features))
cnt = 0
for offset=1:11
	for i=1:(12-offset)
		global cnt
		cnt = cnt + 1
		cor_features[:,cnt] = cov_features[:,cnt] ./ sqrt.(var_features[:,i] .* var_features[:,i+offset])
	end
end 


maximum(cor_features, dims=1)[12:21]

msd_cov_svd = svd(cov_features)
plot(msd_cov_svd.S)
vline!([12])

function subsampled_msd_analysis(n_subsample, feature_map, method; center=true,scale=true)
	train_idx = 1:463_715
	test_idx =  (1:51_630) .+ 463_715

	train_idx = sample(train_idx, n_subsample, replace=false)
	Y_train = msd[train_idx, 1]
	Y_train_bar = mean(Y_train)
	Y_train = Y_train .- Y_train_bar

	X_train = Matrix(msd[train_idx, 2:91])
	X_test = Matrix(msd[test_idx, 2:91])
	Y_test = msd[test_idx, 1]

	centering_transform = fit(ZScoreTransform, X_train; center=center, scale=scale, dims=1)
	X_train_transformed = StatsBase.transform(centering_transform, X_train)
	X_test_transformed = StatsBase.transform(centering_transform, X_test)

	X_train_transformed, grp = feature_map(X_train_transformed)
	X_test_transformed,  _ = feature_map(X_test_transformed)

	fitted_method = fit(method, X_train_transformed, Y_train, grp)
	preds = X_test_transformed*coef(fitted_method) .+ Y_train_bar
	mse = mean( abs2.(preds .- Y_test) )
	(mse=mse, fitted_method=fitted_method)
end 




function subsampled_msd_analysis(n_subsample, feature_map, methods; center=true,scale=true, seed=100)
	Random.seed!(seed)

	train_idx = 1:463_715
	test_idx =  (1:51_630) .+ 463_715

	train_idx = sample(train_idx, n_subsample, replace=false)
	Y_train = msd[train_idx, 1]
	Y_train_bar = mean(Y_train)
	Y_train = Y_train .- Y_train_bar

	X_train = Matrix(msd[train_idx, 2:91])
	X_test = Matrix(msd[test_idx, 2:91])
	
	Y_test = msd[test_idx, 1]

	X_train, grp = feature_map(X_train)
	X_test, grp = feature_map(X_test)
	
	centering_transform = fit(ZScoreTransform, X_train; center=center, scale=scale, dims=1)
	X_train_transformed = StatsBase.transform(centering_transform, X_train)
	X_test_transformed = StatsBase.transform(centering_transform, X_test)
	
	res = []
	for (method_name, method) in methods
		fitted_method = fit(method, X_train_transformed, Y_train, grp)
		preds = X_test_transformed*coef(fitted_method) .+ Y_train_bar
		mse = mean( abs2.(preds .- Y_test) )
		push!(res, (mse=mse, method_name = method_name, fitted_method=fitted_method))
	end
	res
end 

myfeatures = 

identity_map(X) = (X, GroupedFeatures([12,78]))

function correlation_map(X; noisegroups=10, noisefeatures=50)
	mean_features = X[:, 1:12]
	var_features = X[:, 13:24]
	sd_features = sqrt.(var_features)
	cv_features = mean_features ./ sd_features
	cov_features = X[:, 25:90]
	cor_features = zeros(size(cov_features))
	cnt = 0
	for offset=1:11
		for i=1:(12-offset)
			cnt = cnt + 1
			cor_features[:,cnt] = cov_features[:,cnt] ./ sqrt.(var_features[:,i] .* var_features[:,i+offset])
		end
	end
	noise_features = randn(size(X,1), noisegroups*noisefeatures )
	grp = GroupedFeatures(vcat([12, 12, 12, 12, 66, 66], fill(noisefeatures, noisegroups)))
	([mean_features var_features sd_features cv_features  cov_features cor_features noise_features], grp)
end




loo_oneparam_ridge = GroupRidgeRegression(tuning=OneParamCrossValRidgeTuning())
res1 = subsampled_msd_analysis(1000, identity_map, [("bla", loo_oneparam_ridge)], scale=false)

res2 = subsampled_msd_analysis(5000, identity_map, [("bla", sigma_ridge)], scale=false)

res3 = subsampled_msd_analysis(2000, correlation_map, [("bla", loo_oneparam_ridge)], scale=false)
res5 = subsampled_msd_analysis(1000, correlation_map, [("bla", mgcv_reml_ridge)], scale=false)
res5 = subsampled_msd_analysis(500, correlation_map, [("bla", mgcv_gcv_ridge)], scale=false)
res5 = subsampled_msd_analysis(1000, correlation_map, [("bla", mgcv_ml_ridge)], scale=false)

res4 = subsampled_msd_analysis(1000, correlation_map, [("bla", sigma_ridge)], scale=false)


tmp = res4[1].fitted_method
tmp.λs	
sigma_ridge = GroupRidgeRegression(tuning=SigmaRidgeTuning())

tmp = correlation_map(msd_mat)

function polymap(d)
	function poly(X)
		Xs = hcat( hcat([X[:, 1:9].^k for k=1:d]...),  hcat([X[:, 10:90].^k for k=1:d]...))
		grps = GroupedFeatures([fill(9,d); fill(9^2,d)])
		(Xs, grps)
	end 
end





loo_oneparam_ridge = GroupRidgeRegression(tuning=OneParamCrossValRidgeTuning())
sigma_ridge = GroupRidgeRegression(tuning=SigmaRidgeTuning())
mgcv_reml_ridge = GroupRidgeRegression(tuning=SigmaRidgeRegression.MGCVTuning())
mgcv_ml_ridge = GroupRidgeRegression(tuning=SigmaRidgeRegression.MGCVTuning(method="ML"))
mgcv_gcv_ridge = GroupRidgeRegression(tuning=SigmaRidgeRegression.MGCVTuning(method="GCV.Cp"))


Random.seed!(10)
res1 = subsampled_msd_analysis(1000, identity_map, loo_oneparam_ridge)
res1 = subsampled_msd_analysis(1000, polymap(5), loo_oneparam_ridge)

Random.seed!(10);res1 = subsampled_msd_analysis(100, polymap(2), loo_oneparam_ridge)
Random.seed!(100);res1 = subsampled_msd_analysis(1000, polymap(1), sigma_ridge)
Random.seed!(100); res1 = subsampled_msd_analysis(100, polymap(2), mgcv_reml_ridge)


res1.fitted_method.λs

Random.seed!(20)


X_train



n_subsample = 1000
train_idx = 1:463_715
test_idx =  (1:51_630) .+ 463_715

train_idx = sample(train_idx, n_subsample, replace=false)
Y_train = msd[train_idx, 1]
Y_train_bar = mean(Y_train)
Y_train = Y_train .- Y_train_bar

X_train = Matrix(msd[train_idx, 2:91])
X_test = Matrix(msd[test_idx, 2:91])
Y_test = msd[test_idx, 1]

centering_X = fit(ZScoreTransform, X_train; center=true, scale=true, dims=1)
X_transformed = StatsBase.transform(centering_X, X_train)
X_test_transformed = StatsBase.transform(centering_X, X_test)

grp = 
GroupRi


fit1= fit(, X, Y_train, grp)

predict(fit1, )



fit2= fit(GroupRidgeRegression(tuning=SigmaRidgeTuning()), X, Y_train, grp)


function predict(rdg::BasicGroupRidgeWorkspace, X)
	X*rdg.β 
end

coef(rdg::BasicGroupRidgeWorkspace) = rdg.β_curr

import StatsBase:coef




function mse_given_beta(β)
	Y_hat_sigma_ridge = X_test_transformed*β .+ Y_train_bar
	mean(abs2.(Y_hat_sigma_ridge .- Y_test))
end 


mse_given_beta(coef(fit1))
mse_given_beta(coef(fit2))
mse_given_beta(coef(fit3))




function mse_given_beta(β)
	Y_hat_sigma_ridge = myf(X_test_transformed)*β .+ Y_train_bar
	mean(abs2.(Y_hat_sigma_ridge .- Y_test))
end 

mgcv_lambdas = SigmaRidgeRegression.mgcv(X, Y_train, grp)

fit3= fit(GroupRidgeRegression(tuning=mgcv_lambdas.λs), X, Y_train, grp)

fit3.λs






multiridge_lambdas = SigmaRidgeRegression.multiridge(X, Y_train, grp)










struct GroupRidgeRegression{V, Vs<:AbstractVector{V}}
	λs::Vs 
end

function fit(GroupRidgeRegression, X, Y, grp::GroupedFeatures)
	
end 
GroupRidgeRegression(multiridge_lambdas)


init_ridge = BasicGroupRidgeWorkspace(X=X, Y=Y_train, groups=grp)








bla = GroupedRidgeRegression(tuning = OneParamCrossValRidgeTuning())

res = fit(bla, X, Y_test, grp)

res.λs

import StatsBase:fit




 

X_test_transformed = StatsBase.transform(centering_X, X_test)


struct SigmaRidgeRegression{I}
	initializer::I
end 
	




fit!(init_ridge)




SigmaRidge















init_ridge = BasicGroupRidgeWorkspace(X=X, Y=Y_train, groups=grp)
ridge_1 = fit!(init_ridge, OneParamCrossValRidgeTuning())

loo_error(ridge_1)
ridge_1.λs

mean(abs2.(Y_train_bar .- Y_test))


Y_hat = X_test_transformed*ridge_1.β_curr .+ Y_train_bar
mean(abs2.(Y_hat .- Y_test))

init_ridge2 = BasicGroupRidgeWorkspace(X=X, Y=Y_train, groups=grp)
fit!(init_ridge2, λωλας_λ(init_ridge2))
ridge_2 = fit!(init_ridge, SigmaRidgeTuning())
ridge_2.λs
ridge_2.cache


mom = MomentTunerSetup(ridge_2)

Y_hat_sigma_ridge = X_test_transformed*ridge_1.β_curr .+ Y_train_bar

mean(abs2.(Y_hat_sigma_ridge .- Y_test))










train_idx = 1:463_715
test_idx =  (1:51_630) .+ 463_715

train_idx = sample(train_idx, 10000, replace=false)

Y_train = df[train_idx, 1]
Y_train_bar = mean(Y_train)
Y_train = Y_train .- Y_train_bar

X_train = Matrix(df[train_idx, 2:91])
X_train = X_train 



# UPDATES 



X_test = Matrix(df[test_idx, 2:91])
Y_test = df[test_idx, 1]




centering_X = fit(ZScoreTransform, X_train; center=true, scale=true, dims=1)
X = StatsBase.transform(centering_X, X_train)

myf(X) = [X X.^2 X.^3]


X
grp = GroupedFeatures([9,9^2])


X = myf(X)

grp = GroupedFeatures([9,9^2, 9, 9^2, 9, 9^2])


init_ridge = BasicGroupRidgeWorkspace(X=X, Y=Y_train, groups=grp)
ridge_1 = fit!(init_ridge, OneParamCrossValRidgeTuning())

loo_error(ridge_1)

ridge_1.λs

mean(abs2.(Y_train_bar .- Y_test))


Y_hat = myf(X_test_transformed)*ridge_1.β_curr .+ Y_train_bar
mean(abs2.(Y_hat .- Y_test))

init_ridge2 = BasicGroupRidgeWorkspace(X=X, Y=Y_train, groups=grp)

ridge_1 = fit!(init_ridge, OneParamCrossValRidgeTuning())

tuner = MomentTunerSetup(ridge_1)

σs = range(0,20, length=100)
mypath = sigma_squared_path(ridge_1, tuner, σs.^2)

plot(σs, mypath.loos, label="Leave-one-out", xlabel="sigmacv" )
plot!(σs, true_mse, color="green", ylim=(85,94), label="true MSE" )

plot(σs, mypath.λs, xlim=(0, 20), ylim=(0,2))
vline!([ridge_2.cache.params.σ])
plot(σs, mypath.λs, xlim=(0, 20), ylim=(0,2))
mypath.βs




true_mse = [mse_given_beta(vec(mypath.βs[i,:])) for i=1:length(σs)]


 
ridge_2 = fit!(init_ridge,  tuner, SigmaRidgeTuning())

ridge_2.λs



Y_hat_sigma_ridge = myf(X_test_transformed)*ridge_1.β_curr .+ Y_train_bar



mean(abs2.(Y_hat_sigma_ridge .- Y_test))
