# activate project environment
using Pkg
Pkg.activate()

# load packages
using DataFrames, JSON
using Plots
using Random, Distributions
using JuMP, Mosek, MosekTools
using LinearAlgebra

function unbox_data(data_import)
    n = size(data_import["x_train"],1)
    x_train = zeros(4500,n)
    y_train = zeros(4500)
    x_test = zeros(500,n)
    y_test = zeros(500)

    for i in 1:n
        x_train[:,i] = data_import["x_train"][i]
        x_test[:,i] = data_import["x_test"][i]
    end
    y_train = data_import["y_train"]
    y_test = data_import["y_test"]

    return Dict(:x_train => x_train, :x_test => x_test, :y_train => y_train, :y_test => y_test)
end

function accuracy(w,b,data)
    m, n = size(data[:x_test])
    acc = [sign(w'*data[:x_test][i,:] - b) == sign(data[:y_test][i]) ? 1 : 0 for i in 1:m]
    return acc = sum(acc)/m
end

function svm_det(set,data)
    m, n = size(data[:x_train])
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    @variable(model, b)
    @variable(model, w[1:n])
    @variable(model, ζ[1:m] >= 0)
    @objective(model, Min, set[:λ] * w'w + 1/m * sum(ζ[i] for i in 1:m))
    @constraint(model, con[i=1:m], data[:y_train][i]*(w'data[:x_train][i,:] - b) >= 1 - ζ[i])
    optimize!(model)
    return Dict(:b => JuMP.value.(b), :w => JuMP.value.(w),
                :obj => JuMP.objective_value(model))
end

Φ(x) = quantile(Normal(0,1),1-x)

function svm_sto(set,data,set_cc)
    k = size(set_cc[:Σ],1)
    m, n = size(data[:x_train])

    model = Model(optimizer_with_attributes(Mosek.Optimizer))

    @variable(model, b̅)
    @variable(model, B[1,1:k])
    @variable(model, z̅[1:m])
    @variable(model, Z[1:m,1:k])
    @variable(model, w̅[1:n])
    @variable(model, W[1:n,1:k])

    @objective(model, Min, set[:λ] * w̅'w̅ + set[:λ] * tr(W*set_cc[:Σ]*W') + 1/m * sum(z̅[i] for i in 1:m))

    @constraint(model, con_1[i=1:m], [data[:y_train][i]*(w̅' * data[:x_train][i,:] - b̅) + z̅[i] - 1 ;
                                      Φ(set_cc[:η]) * Matrix(cholesky(set_cc[:Σ]).L) * vec(data[:y_train][i] * (W'*data[:x_train][i,:])' .- data[:y_train][i]*B .+ Z[i,:]')] in SecondOrderCone())

    @constraint(model, con_2[i=1:m], [z̅[i];Φ(set_cc[:η])*Matrix(cholesky(set_cc[:Σ]).L)*Z[i,:]] in SecondOrderCone())

    @constraint(model, [W;B] .== diagm(ones(k)))

    optimize!(model)
    return Dict(:w̅ => JuMP.value.(w̅), :b̅ => JuMP.value.(b̅), :W => JuMP.value.(W), :B => JuMP.value.(B), :obj => JuMP.objective_value(model))
end

function sampled_sensitivity(data,set,p)
    @info("started computing sensitivity")

    n, m = size(data[:x_train])

    S = Int(ceil(1/(set[:η]*set[:β])))
    Δ = zeros(S)

    Random.seed!(10); d_scale_i = rand(Uniform(1-set[:α],1+set[:α]),n,m,S)
    Random.seed!(20); d_scale_j = rand(Uniform(1-set[:α],1+set[:α]),n,m,S)

    for s in 1:S
        data_i = copy(data)
        data_i[:x_train] = data[:x_train] .* d_scale_i[:,:,s]
        sol_svm_i = svm_det(set,data_i)

        data_j = copy(data)
        data_j[:x_train] = data[:x_train] .* d_scale_j[:,:,s]
        sol_svm_j = svm_det(set,data_j)

        Δ[s] = norm([sol_svm_i[:w];sol_svm_i[:b]] .- [sol_svm_j[:w];sol_svm_j[:b]],p)
    end

    @info("done computing sensitivity")
    return maximum(Δ)
end

# load OPF classification dataset
cd(dirname(@__FILE__))
caseID="pglib_opf_case14_ieee_3"
data_import = JSON.parsefile("data/"*"$(caseID)"*".json")
data = unbox_data(data_import)

# define experiment settings
set = Dict(:λ => 1e-5, :η => 0.1, :β => 0.1, :α => 0.01, :ε => 1, :δ => 1/Int(0.9*size(data[:x_train],1)), :Δ => NaN)

# solve and evaluate deterministic SVM
sol_svm = svm_det(set,data)
acc_det = accuracy(sol_svm[:w],sol_svm[:b],data)

# compute SVM sensitivity to datasets
set[:Δ] = sampled_sensitivity(data,set,2)

# output perturbation (summary for 1000 perurbation scenarios)
ξ = rand(Normal(0,sqrt(2*log(1.25/set[:δ]))*set[:Δ]/set[:ε]),1000,size(data[:x_train],2)+1)
acc_op = [accuracy(sol_svm[:w] .+ ξ[s,1:size(ξ,2)-1],sol_svm[:b] .+ ξ[s,size(ξ,2)],data) for s in 1:1000]

# program perturbation
set_cc = Dict(:Σ => diagm(ones(size(ξ,2))*(sqrt(2*log(1.25/set[:δ]))*set[:Δ]/set[:ε])^2), :η => 0.05)
sol_ = svm_sto(set,data,set_cc)
sol_sto = Dict(:w̃ => sol_[:w̅] .+ ξ[:,1:size(ξ,2)-1]', :b̃ => sol_[:b̅] .+ ξ[:,size(ξ,2)], :w => sol_[:w̅], :b => sol_[:b̅])
acc_pp = [accuracy(sol_sto[:w̃][:,i],sol_sto[:b̃][i],data) for i in 1:1000]

# print results
@info("classification dataset: $(caseID)")
@info("dataset balance: -1: $(round(sum(-data[:y_train][findall(x->x==-1,data[:y_train])])/Int(0.9*5000)*100,digits=1)) ... 1: $(round(sum( data[:y_train][findall(x->x== 1,data[:y_train])])/Int(0.9*5000)*100,digits=1))")
@info("number of features (n): $(size(data[:x_train],2))")

println("deterministic solution: mean acc: $(round(mean(acc_det)*100,digits=1))")
println("ouput perturbation:     mean acc: $(round(mean(acc_op)*100,digits=1)) ... std: $(round(std(acc_op)*100,digits=1))")
println("program perturbation:   mean acc: $(round(mean(acc_pp)*100,digits=1)) ... std: $(round(std(acc_pp)*100,digits=1))")
