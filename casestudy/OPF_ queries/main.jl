# activate project environment
using Pkg
Pkg.activate()

# load packages
using PowerModels
using Statistics, LinearAlgebra, Distributions, Random
using JuMP, Mosek, MosekTools
using Plots, DataFrames


# auxiliary functions
ns(l) = Int(net[:n_s][l])

nr(l) = Int(net[:n_r][l])

Φ(x) = quantile(Normal(0,1),1-x)

S_cc(η,β) = Int(ceil(1/η * ℯ/(ℯ -1) * (2 - 1 + log(1/β))))

function remove_col_and_row(B,refbus)
    @assert size(B,1) == size(B,2)
    n = size(B,1)
    return B[1:n .!= refbus, 1:n .!= refbus]
end

function build_B̆(B̂inv,refbus)
    Nb = size(B̂inv,1)+1
    B̆ = zeros(Nb,Nb)
    for i in 1:Nb, j in 1:Nb
        if i < refbus && j < refbus
            B̆[i,j] = B̂inv[i,j]
        end
        if i > refbus && j > refbus
            B̆[i,j] = B̂inv[i-1,j-1]
        end
        if i > refbus && j < refbus
            B̆[i,j] = B̂inv[i-1,j]
        end
        if i < refbus && j > refbus
            B̆[i,j] = B̂inv[i,j-1]
        end
    end
    return B̆
end

function load_network_data(caseID)
    data_net = PowerModels.parse_file(caseID)
    # Network size
    G = length(data_net["gen"])
    N = length(data_net["bus"])
    E = length(data_net["branch"])
    D = length(data_net["load"])

    # order bus indexing
    bus_keys=collect(keys(data_net["bus"]))
    bus_key_dict = Dict()
    for i in 1:N
        push!(bus_key_dict, i => bus_keys[i])
    end
    node(key) = [k for (k,v) in bus_key_dict if v == key][1]

    # Load generation data
    gen_key=collect(keys(data_net["gen"]))
    p̅ = zeros(G); p̲ = zeros(G); c0 = zeros(G); c1 = zeros(G); c2 = zeros(G); M_p = zeros(N,G)
    for g in gen_key
        p̅[parse(Int64,g)] = data_net["gen"][g]["pmax"]*data_net["baseMVA"]
        p̲[parse(Int64,g)] = data_net["gen"][g]["pmin"]*data_net["baseMVA"]
        if sum(data_net["gen"][g]["ncost"]) == 3
            c0[parse(Int64,g)] = data_net["gen"][g]["cost"][3]
            c1[parse(Int64,g)] = data_net["gen"][g]["cost"][2] / data_net["baseMVA"]
            c2[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]^2
        end
        if sum(data_net["gen"][g]["ncost"]) == 2
            c0[parse(Int64,g)] = data_net["gen"][g]["cost"][2]
            c1[parse(Int64,g)] = data_net["gen"][g]["cost"][1] / data_net["baseMVA"]
        end
        M_p[node(string(data_net["gen"][g]["gen_bus"])),parse(Int64,g)] = 1
    end
    # sum(c2) == 0 ? c2 = 0.05*c1 : NaN

    # Load demand data
    load_key=collect(keys(data_net["load"]))
    d = zeros(D); M_d = zeros(N,D)
    for h in load_key
        d[parse(Int64,h)] = data_net["load"][h]["pd"]*data_net["baseMVA"] + 1e-3
        M_d[node(string(data_net["load"][h]["load_bus"])),parse(Int64,h)] = 1
    end

    # Load transmission data
    line_key=collect(keys(data_net["branch"]))
    β = zeros(E); f̅ = zeros(E); n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    for l in line_key
        β[data_net["branch"][l]["index"]] = -imag(1/(data_net["branch"][l]["br_r"] + data_net["branch"][l]["br_x"]im))
        n_s[data_net["branch"][l]["index"]] = data_net["branch"][l]["f_bus"]
        n_r[data_net["branch"][l]["index"]] = data_net["branch"][l]["t_bus"]
        f̅[data_net["branch"][l]["index"]] = data_net["branch"][l]["rate_a"]*data_net["baseMVA"]
    end
    # merge parallel lines
    ff = zeros(N,N); ββ = zeros(N,N)
    for l in line_key
        ff[node(string(n_s[data_net["branch"][l]["index"]])),node(string(n_r[data_net["branch"][l]["index"]]))] += f̅[data_net["branch"][l]["index"]]
        ff[node(string(n_r[data_net["branch"][l]["index"]])),node(string(n_s[data_net["branch"][l]["index"]]))] += f̅[data_net["branch"][l]["index"]]
        ββ[node(string(n_s[data_net["branch"][l]["index"]])),node(string(n_r[data_net["branch"][l]["index"]]))]  = β[data_net["branch"][l]["index"]]
        ββ[node(string(n_r[data_net["branch"][l]["index"]])),node(string(n_s[data_net["branch"][l]["index"]]))]  = β[data_net["branch"][l]["index"]]
    end
    # find all parallel lines
    parallel_lines = []
    for l in line_key, e in line_key
        if l != e && node(string(n_s[data_net["branch"][l]["index"]])) == node(string(n_s[data_net["branch"][e]["index"]])) && node(string(n_r[data_net["branch"][l]["index"]])) == node(string(n_r[data_net["branch"][e]["index"]]))
            push!(parallel_lines,l)
        end
    end
    # for l in sort!(parallel_lines)
    #     println("$(l) ... $(data_net["branch"][l]["f_bus"]) ... $(data_net["branch"][l]["t_bus"]) ... $(f̅[data_net["branch"][l]["index"]]) ... $(β[data_net["branch"][l]["index"]])")
    # end
    # update number of edges
    E = E - Int(length(parallel_lines)/2)
    # get s and r ends of all edge
    n_s = trunc.(Int64,zeros(E)); n_r = trunc.(Int64,zeros(E))
    ff = LowerTriangular(ff)
    for l in 1:E
        n_s[l] = findall(!iszero, ff)[l][1]
        n_r[l] = findall(!iszero, ff)[l][2]
    end
    β = zeros(E); f̅ = zeros(E);
    for l in 1:E
        β[l] = ββ[n_s[l],n_r[l]]
        f̅[l] = ff[n_s[l],n_r[l]]
    end

    # Find reference node
    ref = 1
    for n in 1:N
        if sum(M_p[n,:]) == 0 &&  sum(M_d[n,:]) == 0 == 0
            ref = n
        end
    end

    # Compute PTDF matrix
    B_line = zeros(E,N); B̃_bus = zeros(N,N); B = zeros(N,N)
    for n in 1:N
        for l in 1:E
            if n_s[l] == n
                B[n,n] += β[l]
                B_line[l,n] = β[l]
            end
            if n_r[l] == n
                B[n,n] += β[l]
                B_line[l,n] = -β[l]
            end
        end
    end
    for l in 1:E
        B[Int(n_s[l]),Int(n_r[l])] = - β[l]
        B[Int(n_r[l]),Int(n_s[l])] = - β[l]
    end
    B̃_bus = remove_col_and_row(B,ref)
    B̃_bus = inv(B̃_bus)
    B̃_bus = build_B̆(B̃_bus,ref)
    PTDF = B_line*B̃_bus

    # safe network data
    net = Dict(
    # transmission data
    :f̅ => f̅, :n_s => n_s, :n_r => n_r, :T => round.(PTDF,digits=8),
    # load data
    :d => round.(d,digits=5), :M_d => M_d,
    # generation data
    :p̅ => round.(p̅,digits=5), :p̲ => round.(p̲,digits=5), :M_p => M_p,
    :c0 => round.(c0,digits=5), :c1 => round.(c1,digits=5), :c2 => round.(c2,digits=5), :c̀2 => round.(sqrt.(c2),digits=5),
    # graph data
    :N => N, :E => E, :G => G, :D => D, :ref => ref
    )
    return net
end

function solve_det_OPF(net)
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # model variables
    @variable(model, p[1:net[:G]])
    # model objective
    @objective(model, Min, net[:c1]'p)
    # OPF equations
    @constraint(model, λ, ones(net[:N])'*(net[:M_p]*p .- net[:M_d]*net[:d]) .== 0)
    @constraint(model, μ, -net[:f̅] .<= net[:T]*(net[:M_p]*p .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, net[:p̲] .<= p .<= net[:p̅])
    # solve model
    optimize!(model)
    # @info("done solving Det-OPF: $(termination_status(model))")
    sol = Dict(:status => "$(termination_status(model))",
                :obj => JuMP.objective_value(model),
                :p => JuMP.value.(p),
                :CPUtime => solve_time(model))
    return sol
end

function solve_sto_OPF(net)
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # model variables
    @variable(model, x̅[1:net[:G]])
    @variable(model, X[1:net[:G],1])
    # model objective
    @objective(model, Min, net[:c1]'x̅)
    # OPF equations
    @constraint(model, λ_mean, ones(net[:N])'*(net[:M_p]*x̅ .- net[:M_d]*net[:d]) .== 0)
    @constraint(model, λ_stoc, ones(net[:N])'*vec(net[:M_p]*X) .== 0)
    @constraint(model, μ_min, -net[:f̅] .<= net[:T]*(net[:M_p]*(x̅ + X * set[:ζ̲]) .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, μ_max, -net[:f̅] .<= net[:T]*(net[:M_p]*(x̅ + X * set[:ζ̅]) .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, φ_min, net[:p̲] .<= vec(x̅ .+ X * set[:ζ̲]) .<= net[:p̅])
    @constraint(model, φ_max, net[:p̲] .<= vec(x̅ .+ X * set[:ζ̅]) .<= net[:p̅])
    @constraint(model, recourse, net[:c1]'*vec(X) == 1)
    # solve model
    optimize!(model)
    @info("done solving CC-OPF: $(termination_status(model))")
    sol = Dict(:status => termination_status(model),
                :obj => JuMP.objective_value(model),
                :x̅ => vec(JuMP.value.(x̅)),
                :X => vec(JuMP.value.(X)),
                :CPUtime => solve_time(model))
    return sol
end

function OPF_query_feasibility_test(net,cost_fx)
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # model variables
    @variable(model, p[1:net[:G]])
    # model objective
    @objective(model, Min, net[:c1]'p)
    # OPF equations
    @constraint(model, λ, ones(net[:N])'*(net[:M_p]*p .- net[:M_d]*net[:d]) .== 0)
    @constraint(model, μ, -net[:f̅] .<= net[:T]*(net[:M_p]*p .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, net[:p̲] .<= p .<= net[:p̅])
    @constraint(model, net[:c1]'p == cost_fx)
    # solve model
    optimize!(model)
    sol = Dict(:status => "$(termination_status(model))")
    return sol
end

function posterior_analysis(sol,set,startegy,print_flag)
    cost = zeros(set[:S])
    x = zeros(net_data[:G],set[:S])
    inf_counter = 0
    if startegy == "program"
        for s in 1:set[:S]
            x[:,s] = sol[:x̅] + sol[:X] * set[:ζ̂][s]
            cost[s] = net_data[:c1]' * x[:,s]
            feas_res = OPF_query_feasibility_test(net_data,cost[s])
            feas_res[:status] != "OPTIMAL" ? inf_counter += 1 : NaN
        end
    end

    if startegy == "output"
        for s in 1:set[:S]
            cost[s] = sol[:obj] +  set[:ζ̂][s]
            feas_res = OPF_query_feasibility_test(net_data,cost[s])
            feas_res[:status] != "OPTIMAL" ? inf_counter += 1 : NaN
        end
    end

    if startegy == "input"
        ζ̂_data = rand(Laplace(0,set[:α]/set[:ε]),net_data[:D],set[:S])
        for s in 1:set[:S]
            perturbed_net_data = deepcopy(net_data)
            perturbed_net_data[:d] = net_data[:d] + ζ̂_data[:,s]
            sol_OPF_input = solve_det_OPF(perturbed_net_data)

            if sol_OPF_input[:status] == "OPTIMAL"
                cost[s] = sol_OPF_input[:obj]
                feas_res = OPF_query_feasibility_test(net_data,cost[s])
                feas_res[:status] != "OPTIMAL" ? inf_counter += 1 : NaN
            elseif sol_OPF_input[:status] != "OPTIMAL"
                inf_counter += 1
            end
        end
    end

    if print_flag == 1
        println("posterior analysis for $(startegy) perturbation strategy")
        println(" ======================== cost summary ======================== ")
        println("determenistic cost:              $(round(sol_OPF_det[:obj],digits=1))")
        println("in-sample perturbed cost:        $(round(sol[:obj],digits=1))")
        println("out-of-sample perturbed cost:    $(round(mean(cost),digits=1))")
        println("")
        println(" ======================== optimality loss ======================== ")
        println("mean value:                      $(round(norm(sol_OPF_det[:obj]-mean(cost),1)/sol_OPF_det[:obj]*100,digits=2))%")
        println("CVaR_10% value:                  $(round(norm(sol_OPF_det[:obj]-mean(sort!(cost)[901:end]),1)/sol_OPF_det[:obj]*100,digits=2))%")
        println("")
        println(" ======================== query feasibility ====================== ")
        println("$(startegy) perturbation is infeasible in $(sum(inf_counter)/set[:S]*100)% of scenarios")
    end

    return Dict(:cost => filter(!iszero, cost), :x => x)
end

# Load network data from PowerModels
cd(dirname(@__FILE__))
PowerModels.silence()
caseID="data/pglib_opf_case24_ieee_rts.m"
net_data = load_network_data(caseID)

println("")
@warn("test case: $(caseID) ... N: $(net_data[:N]) ... E: $(net_data[:E])")
## experiment settings
set = Dict(:α => 10, :ε => 1,                               # differential privacy
           :Δ => NaN,                                       # sensitivity inputs
           :ζ̲ => NaN, :ζ̅ => NaN, :η => 0.01, :β => 0.1,     # chance constraint reformulation inputs
           :S => 1000, :ζ̂ => [],                            # posterior analysis inputs
           :cvar_q => 0.01,                                 # CVaR setting
           :gen_set => []                                   # sum query generation release set
           )

## Comparison of input, output, and program perturbation strategies

# deterministic OPF solution
sol_OPF_det = solve_det_OPF(net_data)

# bound on the sum query sensitivity to α-adjacent datasets
set[:Δ] = set[:α] * maximum(net_data[:c1])

# draw 1000 random perturbation scenarios for the posterior analysis
Random.seed!(100)
set[:ζ̂] = rand(Laplace(0,set[:Δ]/set[:ε]),set[:S])

# posterior analysis of the input perturbation strategy
posterior_analysis(sol_OPF_det,set,"input",1)

# posterior analysis of the output perturbation strategy
posterior_analysis(sol_OPF_det,set,"output",1)

# program perturbation strategy implementation
Random.seed!(200);
set[:ζ̲] = minimum(rand(Laplace(0,set[:Δ]/set[:ε]),S_cc(set[:η],set[:β])))
Random.seed!(300);
set[:ζ̅] = maximum(rand(Laplace(0,set[:Δ]/set[:ε]),S_cc(set[:η],set[:β])))
sol_OPF_sto = solve_sto_OPF(net_data)

# posterior analysis of the program perturbation strategy
cost_base = posterior_analysis(sol_OPF_sto,set,"program",1)

## Visualization of privacy guarantees on 24_ieee dataset
net_data_copy = deepcopy(net_data)
net_data_copy[:d][15] = net_data[:d][15] - set[:α]
sol_OPF_sto = solve_sto_OPF(net_data_copy)
cost_minus = posterior_analysis(sol_OPF_sto,set,"program",0)

net_data_copy = deepcopy(net_data)
net_data_copy[:d][15] = net_data[:d][15] + set[:α]
sol_OPF_sto = solve_sto_OPF(net_data_copy)
cost_plus = posterior_analysis(sol_OPF_sto,set,"program",0)

plo = plot(frame=:box,
xlims=(49000,62000),
titlefontsize = 12,
xtickfont=font(12),
ytickfont=font(12),
guidefont=font(12),
xlabel = "range of OPF cost (sum query)",
title = "24_ieee dataset: α = $(set[:α]) MWh, ε = $(set[:ε])",
size = (500, 350)
        )
histogram!(cost_minus[:cost],fillalpha=0.85,label="D''", c=RGB(240/255,128/255,128/255))
histogram!(cost_plus[:cost],fillalpha=0.85,label="D'", c=RGB(32/255,178/255,170/255))
histogram!(cost_base[:cost],fillalpha=0.5,label="D", c=:blue)
plot!(legendfontsize=12)
display(plo)
savefig(plo, "privacy_guarantee.pdf")

## CVaR optimality loss control of the sum query on 24_ieee dataset

function rand_gen_set(net_data,portion)
    Random.seed!(500);
    sett=[1:net_data[:G];][rand(1:net_data[:G],max(Int(floor(net_data[:G]*portion*1.1)),1))]
    return unique!(sett)
end

function solve_CVaR_OPF(net,set)
    # DC-OPF definition
    model = Model(optimizer_with_attributes(Mosek.Optimizer, "LOG" => 0))
    # model variables
    @variable(model, γ)
    @variable(model, z[1:set[:S]] >= 0)
    @variable(model, x̅[1:net[:G]])
    @variable(model, X[1:net[:G],1])
    # model objective
    @objective(model, Min, γ + 1/(set[:cvar_q]*set[:S]) * sum(z))
    @constraint(model, cvar[s=1:set[:S]], z[s] >= net[:c1]'*vec(x̅ .+ X * set[:ζ̂][s]) - γ)
    # OPF equations
    @constraint(model, λ_mean, ones(net[:N])'*(net[:M_p]*x̅ .- net[:M_d]*net[:d]) .== 0)
    @constraint(model, λ_stoc, ones(net[:N])'*vec(net[:M_p]*X) .== 0)
    @constraint(model, μ_min, -net[:f̅] .<= net[:T]*(net[:M_p]*(x̅ + X * set[:ζ̲]) .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, μ_max, -net[:f̅] .<= net[:T]*(net[:M_p]*(x̅ + X * set[:ζ̅]) .- net[:M_d]*net[:d]) .<= net[:f̅])
    @constraint(model, φ_min, net[:p̲] .<= vec(x̅ .+ X * set[:ζ̲]) .<= net[:p̅])
    @constraint(model, φ_max, net[:p̲] .<= vec(x̅ .+ X * set[:ζ̅]) .<= net[:p̅])
    @constraint(model, recourse, sum(vec(X)[set[:gen_set]]) == 1)
    # solve model
    optimize!(model)
    @info("done solving CC-OPF: $(termination_status(model))")
    sol = Dict(:status => termination_status(model),
                :obj => JuMP.objective_value(model),
                :x̅ => vec(JuMP.value.(x̅)),
                :X => vec(JuMP.value.(X)),
                :CPUtime => solve_time(model))
    return sol
end

set[:α] = 35
set[:Δ] = set[:α]
set[:gen_set] = rand_gen_set(net_data,0.5)

# draw 1000 random perturbation scenarios for the posterior analysis
Random.seed!(500)
set[:ζ̂] = rand(Laplace(0,set[:Δ]/set[:ε]),set[:S])

# program perturbation strategy implementation
Random.seed!(200);
set[:ζ̲] = minimum(rand(Laplace(0,set[:Δ]/set[:ε]),S_cc(set[:η],set[:β])))
Random.seed!(300);
set[:ζ̅] = maximum(rand(Laplace(0,set[:Δ]/set[:ε]),S_cc(set[:η],set[:β])))

range_q = [.99 .95 .9 .85 .8 .75 .7 .65 .6 .55 .5 .45 .4 .35 .3 .25 .2 .15 .1 0.001 0.001 0.001 0.001 0.001]
anim = @animate for i ∈ range_q
    set[:cvar_q] = i

    sol_OPF_sto = solve_CVaR_OPF(net_data,set)
    posterior_results = posterior_analysis(sol_OPF_sto,set,"program",0)

    plo = plot(frame=:box,
    legend=:topleft,
    ylims=(0,200),
    titlefontsize = 12,
    xtickfont=font(12),
    ytickfont=font(12),
    guidefont=font(12),
    xlabel = "range of OPF cost (x1000)",
    title = "Distribution of optimality loss for q = $(round(set[:cvar_q],digits=3))",
    size = (500, 350)
    )

    histogram!(posterior_results[:cost]./1000, label = false, fillalpha=0.5, c=:black,
    bins = 49:0.1:55
    )

    mean_cost = mean(sort!(posterior_results[:cost]./1000)[1:end])
    cvar_cost = mean(sort!(posterior_results[:cost]./1000)[Int(ceil(set[:S]*0.95)):end])
    vline!([mean_cost], label = "mean", lw = 3, c = RGB(32/255,178/255,170/255))
    vline!([cvar_cost], label = "CVaR-5%", lw = 3, c = RGB(240/255,128/255,128/255), ls = :dot)

    annotate!([(mean_cost-0.5,150,text("$(round(mean_cost,digits=1))", 10, :top, color = RGB(32/255,178/255,170/255), rotation = 90, Plots.font(14)))])
    annotate!([(cvar_cost+0.5,150,text("$(round(cvar_cost,digits=1))", 10, :top, color = RGB(240/255,128/255,128/255), rotation = 90, Plots.font(14)))])
end
gif(anim, "anim_cvar_cost.gif", fps = 2)
