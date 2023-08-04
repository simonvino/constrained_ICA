# Constrained Independent Component Anaysis

<img src="https://github.com/simonvino/constrained_ICA/blob/main/figures/example_ICA.png" width="800">


This is a python implementation for constrained ICA (or ICA with reference), based on the paper of Lu and Rajapakse:

> Lu, W., & Rajapakse, J. C.,
[ICA with reference](https://www.sciencedirect.com/science/article/abs/pii/S0925231205003176), Neurocomputing, 69(16):2244–2257, 2006.

## Demo 

A demo version is provided, which can be directly run in Google Colab with:

> https://colab.research.google.com/github/simonvino/constrained_ICA/blob/main/Demo_colab_constrained_ICA.ipynb

## Theory

Like in classical ICA, the basic goal of constrained ICA (cICA) is to estimate a set of $ N $ source components $ \boldsymbol{y} \in \mathbb{R}^{N} $ from the observed data $ \boldsymbol{x} \in \mathbb{R}^{K} $ by estimating a demixing/weight matrix $ \boldsymbol{W} \in \mathbb{R}^{N \times K} $:

$$
\begin{equation}
\boldsymbol{y} = \boldsymbol{W} \boldsymbol{x}
\end{equation}
$$

Different objective functions  have been proposed to estimate independent components $\boldsymbol{y}$ from data $ \boldsymbol{x} $. In their original paper [Lu and Rajapaske (2006)](https://www.sciencedirect.com/science/article/abs/pii/S0925231205003176) proposed to use and approximation of negentropy as objective function:

$$
\begin{equation}
    J(y) = \rho [ E\{G(y)\} - E\{G(\nu)\} ]^2
\end{equation}
$$

where $ \rho $ denotes a positive constant, $ E\{\cdot\} $ represents the expectation value and $ \nu $ is a Gaussian random variable with zero mean and unit variance. Further $ G(\cdot) $ can be any non-quadratic function, which can practically be chosen as $ G(y) = (\textrm{log} \,\, \textrm{cosh}(a_1 y)) / a_1 $ with constants $ 1 \leq a_1 \leq 2 $. Besides maximizing the objective function $ J(y) $, cICA includes the similarity to a given reference component $ r_n(t) $ as constraint into the optimization. This additional constraint can be formulated as $ g(\boldsymbol{w}) = \rho - \epsilon (y, r) \le 0 $, where $ \rho $ denotes a pre-defined similarity threshold parameter, and $ \epsilon(\cdot) $ a function that measures the closeness of the estimated source component $ y $ to a reference $ r $. The similarity can be simply defined as correlation between $ y $ and $ r $ as $ \epsilon(y, r) = E[y, r] $. Based on these definitions, the augmented Lagrangian function $ \mathcal{L}(\boldsymbol{W}, \boldsymbol{\mu}) $ for estimating $ N $ source components $ y $, given $ N $ references $ r_n $ can be defined as}: 

$$
\begin{equation}
    \mathcal{L}(\boldsymbol{W}, \boldsymbol{\mu}) = \sum_{n=1}^N \left( J(y) + \frac{\textrm{max}^2\{0, \mu_n + \gamma_n g_n(\boldsymbol{w}_n) \} - \mu_n^2}{2 \gamma_n} \right)
\end{equation}
$$

with $ \boldsymbol{\mu} = (\mu_1, \ldots, \mu_N)^{T} $ denoting a set of Lagrangian multipliers, and $ \boldsymbol{\gamma} = ( \gamma_1, \ldots, \gamma_N )^T $ representing positive learning parameters for the penalty term. The Lagrangian function can then be maximised by simply using a gradient-based learning update rule:

$$
\begin{equation}
    \boldsymbol{W}_i = \boldsymbol{W}_{i-1} + \eta \frac{\partial \mathcal{L}(\boldsymbol{W})}{\partial \boldsymbol{W}}
\end{equation}
$$

where the update step at iteration $ i $ is controlled by the learning rate $ \eta $. 

As an alternative to maximizing negentropy as introduced above, the algorithm implements the Infomax objective function (equivalent to the maximum likelihood principle): 

$$
\begin{equation}
    J(y) = E \left[ \sum_{n=1}^N \textrm{log} \, p(\boldsymbol{w}_{n}^T\boldsymbol{x}) \right] + \textrm{log} \, |\textrm{det} \, (\boldsymbol{W})|
\end{equation}
$$

whereby $ p(y) $ denotes the probability density function of $ y $. To estimate a more diverse set of source signals the extended infomax algorithm adapts this nonlinearity to both super- and sub-Gaussion distributions [Lee at al. (1999)](https://dl.acm.org/doi/10.1162/089976699300016719). The gradient of $ p(y) $ with respect to $ y $ can be chosen as $ \frac{p(\partial y)}{\partial y} = \textrm{tanh}(y) - y $ for sub-Gaussion sources and $ \frac{\partial p(y)}{\partial y} = \textrm{tanh}(y) - y $ for super-Gaussion sources.

Per default this implementation maximizes Negentropy (argument ``` obj_func='negentr' ```), but can be adapted to the Infomax (``` obj_func='infomax' ```), as well as extended Infomax (``` obj_func='ext_infomax' ```).  


## Citations

Constrained ICA can be used to resolve inherent ambiguities of ICA in the ordering of estimated source components. This allows for example to apply ICA to group fMRI studies, as proposed in:

> Wang, Z., Xia M., Jin Z., Yao L., Long Z.: [Temporally and Spatially Constrained ICA of fMRI Data Analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0094211). PLOS ONE 9(4) (2014)

> Wein, S., Tome, A. M., Goldhacker, M., Greenlee, M. W., Lang, E. W.: [A constrained ICA-EMD model for group level fMRI analysis](https://www.frontiersin.org/articles/10.3389/fnins.2020.00221/full). Frontiers in Neuroscience 14 (2020).

