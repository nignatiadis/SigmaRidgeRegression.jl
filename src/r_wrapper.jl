function multiridge(X, Y, grp)
  # figure out what happens to offset
  @rput X
  @rput Y
  n = size(X,1)
  grp_ps = [group_idx(grp, i) for i=1:grp.num_groups]
  @rput grp_ps
  R"Xst <- lapply(grp_ps, function(idxs) X[,idxs])"
  R"library(multiridge)"
  R"Xomics <- createXblocks(Xst)"
  R"XXomics <- createXXblocks(Xst)"
  R"cvperblock <- fastCV(Xomics,Y=Y,kfold=10, intercept=FALSE, fixedfolds = TRUE, model='linear')"
  R"lambdas <- cvperblock$lambdas"
  R"lambdas" 
  R"leftout <- CVfolds(Y=Y,kfold=10,nrepeat=2,fixedfolds = TRUE, model='linear')"
  R"jointlambdas <- optLambdas(penaltiesinit=lambdas, XXblocks=XXomics,Y=Y,folds=leftout,score='mse',model='linear')"
  R"chosen_params <- jointlambdas$optpen / $n"
  @rget chosen_params
  chosen_params
end 

function mgcv(X, Y, grp; method="REML")
  n,p = size(X)
  @rput X
  @rput Y
  @rput p
  @rput n
  @rput method
  R"library(mgcv)"
  R"""
    construct_mat <- function(p, idxs){
      mat <- matrix(0, p, p)
      diag(mat)[idxs] <- 1
      mat
    }
    """
   grp_ps = [group_idx(grp, i) for i=1:grp.num_groups]
   @rput grp_ps
   R"S_reg <- lapply(grp_ps, function(idxs) construct_mat(p, idxs))"
   R"ptm <- proc.time()"
   R"fit_mgcv <- bam(Y~X-1, method=method, paraPen=list(X=S_reg))"
   R"time_elapsed <- proc.time() - ptm"
   R"mgcv_lambdas <- fit_mgcv$sp/n"
   @rget mgcv_lambdas
   @rget time_elapsed
   (λs = mgcv_lambdas, time = time_elapsed)
end


Base.@kwdef struct MGCVTuning <: AbstractRidgeTuning
  method::String = "REML"
end 
  
function StatsBase.fit!(rdg::BasicGroupRidgeWorkspace, tune::MGCVTuning)
  r_res = mgcv(rdg.X, rdg.Y, rdg.groups; method=tune.method)
  StatsBase.fit!(rdg, r_res.λs)
  rdg.cache = r_res
  rdg
end 