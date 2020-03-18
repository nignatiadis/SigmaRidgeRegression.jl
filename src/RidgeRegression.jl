module RidgeRegression

import Base:reduce, rand

include("groupedfeatures.jl")

export GroupedFeatures,
       group_idx,
	   group_summary,
	   group_expand,
	   random_betas
end # module
