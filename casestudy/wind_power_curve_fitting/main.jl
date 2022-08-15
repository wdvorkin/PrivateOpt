using Plots
using JuMP, Mosek, MosekTools
using DataFrames, CSV, JSON
using Distributions, Random
using LinearAlgebra, Calculus

function wind_power_measurements(σ,bias_x = 0,bias_y = 0)
    Random.seed!(100)
    ξ = rand(Normal(0,σ),n)

    x_ = x .+ bias_x

    y_ = y .+ ξ .+ bias_y

    x_ = max.(x_,minimum(x)); x_ = min.(x_,maximum(x))
    y_ = max.(y_,0); y_ = min.(y_,1)

    return x_, y_
end

function IQBF(x, mu, gamma = 1)
    ``` Inverse quadratic radial basis function```
    return sqrt(1+(gamma*(mu-x))^2)
end

function basis_fun_vals(x_tr, set)
    Φ = ones(length(x_tr),set[:n_base_fun])
    for m in 1:set[:n_base_fun]
        mu = LinRange(minimum(x_tr), maximum(x_tr), set[:n_base_fun])[m]
        Φ[:, m] = IQBF.(x_tr, mu)
    end
    return Φ
end

function fit_model(x_tr,y_tr)
    Φ = basis_fun_vals(x_tr, set)
    n, m = size(Φ)

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))

    @variable(model, β[1:m])

    @variable(model, t)

    @objective(model, Min, t)

    @constraint(model, [t;y_tr .- Φ * β] in SecondOrderCone())

    optimize!(model)

    @info("model status : $(termination_status(model))")

    return Dict(:β => JuMP.value.(β), :loss => JuMP.objective_value.(model))
end

function model_sensitivity(x_tr, y_tr, set, p)
    N = Int(ceil(1/(set[:η_Δ]*set[:β_Δ])))
    δ = zeros(N)
    Random.seed!(200)
    ζ_1 = rand(Uniform(0, 2*π),2,N)
    Random.seed!(300)
    ζ_2 = rand(Uniform(0, 1),2,N)
    for i in 1:N
        x_tr_, y_tr_ = x_tr .+ set[:max_speed_div]*ζ_2[1,i]*cos(ζ_1[1,i]), y_tr .+ set[:max_power_div]*ζ_2[1,i]*sin(ζ_1[1,i])
        sol_1 = fit_model(x_tr_,y_tr_)
        β_1 = sol_1[:β]

        x_tr_, y_tr_ = x_tr .+ set[:max_speed_div]*ζ_2[2,i]*cos(ζ_1[2,i]), y_tr .+ set[:max_power_div]*ζ_2[2,i]*sin(ζ_1[2,i])
        sol_2 = fit_model(x_tr_,y_tr_)
        β_2 = sol_2[:β]

        δ[i] = norm(β_1 .- β_2,p)
    end
    return maximum(δ)
end

function fit_model_sto(set)

    x_tr, y_tr = wind_power_measurements(set[:σ])

    Random.seed!(300)
    ξ = rand(Laplace(0,Δ),100)
    ξ̲, ξ̅ = minimum(ξ), maximum(ξ)

    μ = LinRange(minimum(x_tr), maximum(x_tr), set[:n_base_fun])
    φ(x) = [IQBF(x, μ[i]) for i in 1:set[:n_base_fun]]

    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))

    @variable(model, β̅[1:set[:n_base_fun]])

    @variable(model, t)
    @objective(model, Min, t)

    @constraint(model, [t; y_tr .- basis_fun_vals(x_tr, set) * β̅] in SecondOrderCone())

    @constraint(model, set[:C] * (β̅ .+ ξ̲) .>= 0)
    @constraint(model, set[:C] * (β̅ .+ ξ̅) .>= 0)

    optimize!(model)

    @info("model status : $(termination_status(model))")

    return Dict(:β => JuMP.value.(β̅), :loss => JuMP.objective_value.(model))
end

function expected_loss(set,sol)
    x_tr, y_tr = wind_power_measurements(set[:σ])
    ξ = rand(Laplace(0,Δ),set[:n_base_fun],1000)
    loss = zeros(1000)
    for s in 1:1000
        loss[s] = norm(y_tr .- basis_fun_vals(x_tr, set) * (sol[:β] + ξ[:,s]),2)
    end
    return mean(loss)
end

function feasibility(set,sol)
    x_tr, y_tr = wind_power_measurements(set[:σ])

    ξ = rand(Laplace(0,Δ),set[:n_base_fun],1000)

    flag = 0
    for i in 1:1000
        sum(set[:C] * (sol[:β] .+ ξ[:,i]) .>= 0) == p ? NaN : flag += 1
    end

    return flag/1000*100
end

# upload data
cd(dirname(@__FILE__))
wind_data = JSON.parsefile("data/dfpc_ninja_clean.json")
wind_data_keys = collect(keys(wind_data))


# pick one turbine
tubine = "GE.2.75.103"
# find non-zero indeces
non_zero_ind = findall(x -> x != 0, wind_data[tubine])
# filter out speeds byound 20 m/s
cut_off_seed_index = findall(x -> x <= 15, wind_data["speed"])
# get indeces of interest
ind = findall(in(non_zero_ind),cut_off_seed_index)

# experiment settings
set = Dict( :σ => 0.1, :n_base_fun => 4,                      # data and regression settings
            :η_Δ => 0.1, :β_Δ => 0.1,                         # sensitivity settings
            :max_speed_div => 0, :max_power_div => 0.025,     # dataset variation
            :C => [] )                                        # monotone constraints matrix)

# prepare training dataset
n = length(ind)
y = wind_data[tubine][ind]
x = wind_data["speed"][ind]
x_tr, y_tr = wind_power_measurements(set[:σ])

# optimize deterministic regression model
sol_det = fit_model(x_tr,y_tr)

# regression senstivity to datasets
Δ = model_sensitivity(x_tr, y_tr, set, 1)

## build constraint matrix and save data
p = 10; Random.seed!(400); point = rand(Uniform(3,10),p);
μ = LinRange(minimum(x_tr), maximum(x_tr), set[:n_base_fun])
set[:C] = [Calculus.gradient(x -> IQBF(x, μ[j]), point[i]) for i in 1:p, j in 1:set[:n_base_fun]]

## program perturbation
sol_sto = fit_model_sto(set)

## print summary
println("sumamry for turbine:  $(tubine)")
println("deterministic loss:   $(round.(sol_det[:loss],digits=5))")
println("expected loss for OP: $(round.(expected_loss(set,sol_det),digits=5))")
println("expected loss for PP: $(round.(expected_loss(set,sol_sto),digits=5))")
println("feasibility of OP:    $(round.(feasibility(set,sol_det),digits=5))")
println("feasibility of PP:    $(round.(feasibility(set,sol_sto),digits=5))")

## plot output perturbation
plo = plot(xlims=(minimum(x),maximum(x)),ylims=(0,1),frame=:box,legend=false,
        xlabel="wind speed (m/s)",ylabel="normalized power output", xtickfontsize=14,ytickfontsize=14, labelfontsize=14)
scatter!(x_tr,y_tr,c=:blue)
N_plots = 300
Random.seed!(3000)
ξ = rand(Laplace(0,Δ),set[:n_base_fun],N_plots)
for i in 1:N_plots
    if sum(set[:C] * (sol_det[:β] .+ ξ[:,i]) .>= 0) == p
        plot!(collect(LinRange(minimum(x),maximum(x),100)),
                basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set)
                    * (sol_det[:β] .+ ξ[:,i]) ,label=false, lc=:green, lw=1,alpha=0.5)
    end
end
for i in 1:N_plots
    if sum(set[:C] * (sol_det[:β] .+ ξ[:,i]) .>= 0) != p
        plot!(collect(LinRange(minimum(x),maximum(x),100)),
                basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set)
                    * (sol_det[:β] .+ ξ[:,i]) ,label=false, lc=:red, lw=1,alpha=0.5)
    end
end
plot!(collect(LinRange(minimum(x),maximum(x),100)),
        basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set)
            * sol_det[:β] ,label=false, lc=:blue, lw=3)
plot!(title="Mean regression loss: $(round(expected_loss(set,sol_det),digits=2)), infeasibility: $(round(feasibility(set,sol_det),digits=1))%")
display(plo)
savefig(plo, "$(tubine)_$(set[:max_power_div])_op.pdf")

## plot program perturbation
plo_ = plot(xlims=(minimum(x),maximum(x)),ylims=(0,1),frame=:box,legend=false,
        xlabel="wind speed (m/s)",ylabel="normalized power output", xtickfontsize=14,ytickfontsize=14, labelfontsize=14)
scatter!(x_tr,y_tr,c=:blue)
N_plots = 300
Random.seed!(3000)
ξ = rand(Laplace(0,Δ),set[:n_base_fun],N_plots)
for i in 1:N_plots
    if sum(set[:C] * (sol_sto[:β] .+ ξ[:,i]) .>= 0) == p
        plot!(collect(LinRange(minimum(x),maximum(x),100)),
                basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set)
                    * (sol_sto[:β] .+ ξ[:,i]) ,label=false, lc=:green, lw=1,alpha=0.5)
    end
end
for i in 1:N_plots
    if sum(set[:C] * (sol_sto[:β] .+ ξ[:,i]) .>= 0) != p
        plot!(collect(LinRange(minimum(x),maximum(x),100)),
                basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set)
                    * (sol_sto[:β] .+ ξ[:,i]) ,label=false, lc=:red, lw=1,alpha=0.5)
    end
end
plot!(collect(LinRange(minimum(x),maximum(x),100)),
        min.(1,max.(0,basis_fun_vals(collect(LinRange(minimum(x),maximum(x),100)), set) * sol_sto[:β])) ,label=false, lc=:blue, lw=3)
plot!(title="Mean regression loss: $(round(expected_loss(set,sol_sto),digits=2)), infeasibility: $(round(feasibility(set,sol_sto),digits=1))%")
display(plo_)
savefig(plo_, "$(tubine)_$(set[:max_power_div])_pp.pdf")
