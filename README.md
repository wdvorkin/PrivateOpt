# PrivateOpt: Differentially Private Convex Optimization

This repository collects tutorials and case studies of differentially private convex optimization. The materials are based on the theory and applications from the following preprint:

[*Privacy-Preserving Convex Optimization: When Differential Privacy Meets Stochastic Programming*]()

by Vladimir Dvorkin, Ferdinando Fioretto, Pascal Van Hentenryck, Pierre Pinson and Jalal Kazempour. 

***

## Preventing data leakage in convex optimization


<table align="center">
    <tr>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557633-4285460b-2437-4159-a38c-4891b268e62a.gif" alt="some text"></td>
        <td align="center"> The optimal resource allocation discloses <br> the boundary of the feasible set </td>
    </tr>
    <tr>
        <td align="center">
            The maximum-margin hyperplane of the support <br> vector 
            machines classifier is unique with <br> respect to   marginal 
            data points </td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557705-11c922f0-59b8-4ad9-bb97-80e31e34f8ab.gif" alt="some text"></td>
    </tr>
    <tr>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557774-5e2ca222-c164-49ca-a11d-d74c39a74126.gif" alt="some text"></td>
        <td align="center">
            The optimal regression model is unique on a training dataset, disclosing agent participation in a dataset
        </td>
    </tr>
    <tr>
        <td align="center">The optimal inscribed ellipsoid is unique on a particular polyhedral set, discussing its characteristics</td>
        <td align="center"><img src="https://user-images.githubusercontent.com/31773955/184557785-405b2ad4-675f-4ef1-aedb-4f554b9c3658.gif" alt="some text"></td>
    </tr>
</table>


Many optimization and machine learning models can be addressed by the following conic optimziation program: 
\begin{align*}
    x(\mathcal{D}) = \argmin{x}\quad& c^{\top}x\\
    \st\quad&
    b - Ax \in\mathcal{K},
\end{align*}


