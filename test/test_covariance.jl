using SigmaRidgeRegression
using LinearAlgebra
using Test



tmp_block = BlockCovarianceDesign([IdentityCovarianceDesign(), IdentityCovarianceDesign(missing)], missing)


id = IdentityCovarianceDesign()
groups = GroupedFeatures([200;200])

@test set_groups(id, 400) == set_groups(id, groups)

instantiated_block = set_groups(tmp_block, groups)

bla = simulate_rotated_design(instantiated_block, 20)
@test size(bla) == (20,400)

instantiated_block.blocks[1]

spectrum(instantiated_block)
cov1 = SigmaRidgeRegression.UniformScalingCovarianceDesign(p=100, scaling=2.5)
