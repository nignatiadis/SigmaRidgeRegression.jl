abstract type AbstractResponseModel end 


Base.@kwdef struct RandomLinearResponseModel <: AbstractResponseModel
	αs::Vector{Float64}
	grp::GroupedFeatures
	iid_measure = Normal()
end 


function (resp::RandomLinearResponseModel)(X_train, X_test)
	βs =  random_betas(resp.grp, resp.αs) #todo, allow other noise dbn.
	Y_train = X_train * βs
	Y_test = X_test * βs
	Y_train, Y_test, βs
end

Base.@kwdef struct GroupRidgeSimulationSettings{C, R}
	grp::GroupedFeatures
	Σ::C
	response_model::R
	response_noise = Normal()
	σ::Float64 = 1.0
	ntest::Int = 10000
	ntrain::Int 
	iid_measure = Normal()
end

Base.@kwdef struct GroupRidgeSimulation
	grp::GroupedFeatures
	X_test::Matrix{Float64}
	Y_test::Vector{Float64}
	X_train::Matrix{Float64}
	Y_train::Vector{Float64}
	βs = nothing
end



function simulate(group_simulation::GroupRidgeSimulationSettings)
	ntrain = group_simulation.ntrain
	ntest = group_simulation.ntest
	response_model = group_simulation.response_model
	response_noise = group_simulation.response_noise 
	
	X_train = simulate_rotated_design(group_simulation.Σ, ntrain; rotated_measure = group_simulation.iid_measure)
	X_test = simulate_rotated_design(group_simulation.Σ, ntest; rotated_measure = group_simulation.iid_measure)
	Y_train, Y_test, βs = response_model(X_train, X_test)
	Y_train = Y_train .+ rand(response_noise, ntrain)
	Y_test = Y_test .+ rand(response_noise, ntest)

	GroupRidgeSimulation(;grp=group_simulation.grp,
	                     X_train = X_train,
	                     X_test  = X_test,
						 Y_train = Y_train,
						 Y_test  = Y_test,
						 βs = βs)
end