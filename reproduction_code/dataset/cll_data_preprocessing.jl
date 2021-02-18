# This file provides reproducible code for the extraction of SigmaRidgeRegression.CLLData
# from the R/Bioconductor MOFA package.

# A R installation (it will be called through `RCall`) is required
# with an installation of the `MOFAdata` package.
# This package may be installed from within `R` as follows:

# ```r
# if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")
#BiocManager::install("MOFAdata")
#```

using JLD2
using RCall

R"""
	data("CLL_data", package="MOFAdata")

	# use methylation data, gene expression data and drug responses as predictors
	CLL_data <- CLL_data[1:3]
	CLL_data <- lapply(CLL_data,t)
	ngr <- sapply(CLL_data,ncol)
	CLL_data <- Reduce(cbind, CLL_data)

	#only include patient samples profiles in all three omics
	CLL_data2 <- CLL_data[apply(CLL_data,1, function(p) !any(is.na(p))),]
	dim(CLL_data2)

	# prepare design matrix and response
	X <- CLL_data2[,!grepl("D_002", colnames(CLL_data))]
	y <- rowMeans(CLL_data2[,grepl("D_002", colnames(CLL_data))])
	annot <- rep(1:3, times = ngr-c(5,0,0)) # group annotations to drugs, meth and RNA
	ngr_prime <- ngr-c(5,0,0)
"""

# run with seed from Velten & Huber
R"""
set.seed(9876)
foldid <- sample(rep(seq(10), length=nrow(X)))
"""

@rget foldid
@rget X
@rget y
@rget ngr_prime

cll_data = (X = X, y = y, ngr = Int.(ngr_prime), foldid = foldid)
JLD2.@save "cll_data.jld2" {compress=true} cll_data
