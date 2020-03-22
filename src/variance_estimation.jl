abstract type NoiseLevelEstimator end


struct DickerMoments <: NoiseLevelEstimator end 

function whiten_covariates(X, Σ::UniformScaling{Bool})
	X
end	

function whiten_covariates(X, Σ::Cholesky)
	X*inv(Σ.UL)
end	

function estimate_var(::DickerMoments, X, Y; Σ=I)
	X = whiten_covariates(X, Σ)
	n, p = size(X)
	Y_norm_squared = sum(abs2, Y)
	XtopY_norm_squared = sum(abs2, X'*Y) 
	(p+n+1)/n/(n+1)*Y_norm_squared - 1/n/(n+1)*XtopY_norm_squared
end 

struct LeaveOneOut <: NoiseLevelEstimator end 

#function dicker_moments
	