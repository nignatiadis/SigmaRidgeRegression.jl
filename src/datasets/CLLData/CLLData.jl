export CLLData
"""
    CLLData

A dataset of different omic measurements for Chronic lymphocytic leukaemia (CLL)
patient samples. The data can be loaded via:

```
cll_data = CLLData.load()
```

`cll_data` is a named tuple with fields:
* `X`: The features.
* `y`: The response, namely Ibrutinib sensitivity.
* `ngr`: The number of features in each of the three feature groupings, namely
drug sensitivity, methylation and RNAseq data.
* `foldid`: A `Vector{Int}` with values in 1,..,10 that assign each of the rows of `X` to
a fold to be used in cross-validation.

## References

The dataset was originally published in:

Dietrich, Sascha, et al. "Drug-perturbation-based stratification of blood cancer."
The Journal of clinical investigation 128.1 (2018): 427-445.

It was used in the context of side-information by:

Velten, Britta, and Wolfgang Huber.
"Adaptive penalization in high-dimensional regression and classification
 with external covariates using variational Bayes."
Biostatistics (2019).

The `foldid` assignment into folds is the same as the one used by the above publication.

The dataset was copied from the Bioconductor MOFAdata package, available at:
https://bioconductor.org/packages/release/data/experiment/html/MOFAdata.html
"""
module CLLData

using JLD2

const DATA = joinpath(@__DIR__, "cll_data.jld2")

function load()
    JLD2.@load DATA cll_data
    cll_data
end

end
