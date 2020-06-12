using CSV
using Tables
using DataFrames
using StatsBase
msd = CSV.File(joinpath(@__DIR__,"dataset","YearPredictionMSD.txt"), header=false)

df = msd |> DataFrame

train_idx = 1:463_715
test_idx =  (1:51_630) .+ 463_715

train_idx = sample(train_idx, 10000, replace=false)

Y_train = df[train_idx, 1]
Y_train_bar = mean(Y_train)
Y_train = Y_train .- Y_train_bar

X_train = Matrix(df[train_idx, 2:91])
X_train = X_train 







X_test = Matrix(df[test_idx, 2:91])
Y_test = df[test_idx, 1]




centering_X = fit(ZScoreTransform, X_train; center=true, scale=true, dims=1)
X = StatsBase.transform(centering_X, X_train)
X_test_transformed = StatsBase.transform(centering_X, X_test)


grp = GroupedFeatures([9,9^2])


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
X_test_transformed = StatsBase.transform(centering_X, X_test)

myf(X) = [X X.^2 X.^3]


X
grp = GroupedFeatures([9,9^2])


X = [X X.^2 X.^3]

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

σs = range(1,20, length=400)
mypath = sigma_squared_path(ridge_1, tuner, σs.^2)

plot(σs, mypath.loos)
plot(σs, mypath.λs, xlim=(0, 20), ylim=(0,2))


ridge_2 = fit!(init_ridge, SigmaRidgeTuning())
ridge_2.λs



Y_hat_sigma_ridge = myf(X_test_transformed)*ridge_1.β_curr .+ Y_train_bar



mean(abs2.(Y_hat_sigma_ridge .- Y_test))
