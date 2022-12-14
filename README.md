# PrivateOpt: Differentially Private Convex Optimization

This repository collects tutorials and case studies of differentially private convex optimization. The materials are based on the theory and applications from the following preprint:

[*Privacy-Preserving Convex Optimization: When Differential Privacy Meets Stochastic Programming*](https://arxiv.org/pdf/2209.14152.pdf)

by Vladimir Dvorkin, Ferdinando Fioretto, Pascal Van Hentenryck, Pierre Pinson and Jalal Kazempour. 

Animations below depict how changes in optimization datasets leak through optimization results (model parameters, allocation costs, etc.), hence causing privacy risks for data owners. In this work, we provide formal privacy guarantees to optimization datasets by internalizing the standard differential privacy mechanisms into convex optimization using linear decision rules and chance-constrained programming. 

<table align="center">
    <tr>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557633-4285460b-2437-4159-a38c-4891b268e62a.gif">
        resource allocation
        </td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557705-11c922f0-59b8-4ad9-bb97-80e31e34f8ab.gif">
        SVM classification
        </td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184562910-a5d42118-e2db-49f3-bc1f-b794787bb38e.gif">
        regression analysis
        </td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184562925-7d3a90d3-d134-4678-9ff2-b17857875233.gif">
        ellipsoid fitting
        </td>
    </tr>
</table>

***

## Installation and usage

All models are implemented in Julia Language v.1.6 using [JuMP](https://github.com/jump-dev/JuMP.jl) modeling language for mathematical optimization and commercial [Mosek](https://github.com/MOSEK/Mosek.jl) optimization solver, which needs to be licensed (free for academic use). Make sure to active project environment using ```Project.toml``` and ```Manifest.toml``` appended to each Jupiter notebook (for [tutorials](https://github.com/wdvorkin/PrivateOpt/tree/main/tutorials)) and Julia execution file ```main.jl``` (for [case studies](https://github.com/wdvorkin/PrivateOpt/tree/main/casestudy)). 

