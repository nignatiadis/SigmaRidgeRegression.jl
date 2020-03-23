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

Base.@kwdef struct SigmaLeaveOneOut{O} <: NoiseLevelEstimator
	optimizer::O = GoldenSection()
end

#function dicker_moments
#mlestlin <- function(Y,X){
#  maxv <- var(Y)
#  sim2 = function(ts){
	#ts <- c(log(0.01),log(maxv));X <- highdimdata
#	tausq<-ts[1];sigmasq<-ts[2]
#	n<- nrow(X)
#	varY = X %*% t(X) * exp(tausq) + diag(rep(1,n))*exp(sigmasq)
#	mlk <- -dmvnorm(Y,mean=rep(0,n),sigma=varY,log=TRUE)
#	return(mlk)
 # }
  #op <- optim(c(log(0.01),log(maxv)),sim2)
  #mlests <- exp(op$par)
  #return(mlests)
#}	