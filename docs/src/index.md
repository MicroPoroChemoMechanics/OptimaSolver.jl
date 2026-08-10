```@meta
CurrentModule = OptimaSolver
```

# OptimaSolver.jl

**OptimaSolver** is a Julia-native primal-dual interior-point solver for
Gibbs-energy minimization in equilibrium chemistry.

It solves problems of the form

```math
\min_{n \in \mathbb{R}^{n_s}}\ f(n, p)
\qquad \text{subject to} \qquad
A n = b,\quad n \geq \varepsilon
```

where $f$ is the Gibbs free energy (e.g. $G(n) = \sum_i n_i(\mu_i^0/RT + \ln n_i)$
for an ideal/dilute solution), $A \in \mathbb{R}^{m \times n_s}$ is the
mass-conservation (stoichiometric) matrix, $b \in \mathbb{R}^m$ is the
element-abundance vector, and $\varepsilon$ is a small positivity floor.

## Key features

- **Schur-complement Newton step** — reduces the $(n_s+m)$-dimensional KKT system
  to an $m \times m$ system by exploiting the diagonal Hessian structure.
  For chemistry problems $m$ is typically the number of elements ($m \leq 15$), so
  this is a dramatic reduction.
- **Filter line search** — Wächter & Biegler (2006) filter method with Armijo
  sufficient decrease on the barrier objective.
- **Variable stability classification** — near-bound (absent) species receive
  reduced step sizes to avoid numerical blow-up near the positivity boundary.
- **Implicit-differentiation sensitivity** — post-solve computation of
  $\partial n^*/\partial b$ and $\partial n^*/\partial(\mu^0/RT)$ using the
  same Schur-complement factorization as the last Newton step.
- **Warm-start** — consecutive solves (e.g. temperature scans, titration curves)
  reuse the previous solution as the starting point, typically halving the iteration
  count or more.
- **ForwardDiff / AD compatibility** — no `Float64` casts; the entire solver stack
  is written in generic Julia arithmetic so thermodynamic parameters can be
  differentiated through the solver.
- **SciML drop-in** — [`OptimaOptimizer`](@ref) implements
  `SciMLBase.AbstractOptimizationAlgorithm` and is a drop-in replacement for
  `IpoptOptimizer` in ChemistryLab.jl.

## Lineage

OptimaSolver is a Julia port of the **optima** C++ library developed by
[Allan Leal](https://erdw.ethz.ch/en/people/profile.allan-leal.html) (ETH Zürich):

<https://github.com/reaktoro/optima>

The algorithmic design — Schur-complement reduction, filter line search, variable
stability classification, and implicit-differentiation sensitivity — originates from
that library and from the following reference:

> Leal, A.M.M., Blunt, M.J., LaForce, T.C. (2014).
> Efficient chemical equilibrium calculations for geochemical speciation and reactive
> transport modeling.
> *Geochimica et Cosmochimica Acta*, **131**, 301–322.
> <https://doi.org/10.1016/j.gca.2014.01.006>

The Julia port was authored by Jean-François Barthélémy (CEREMA, France) with
assistance from [Claude Code](https://claude.ai/code) (Anthropic).

## Documentation structure

| Section | Content |
|---------|---------|
| [Getting Started](@ref "Getting Started") | Installation and first solve |
| [Theory](@ref "Theory") | Interior-point algorithm, KKT, Schur complement, filter line search, sensitivity |
| [Basic Usage](@ref "Basic Usage") | Simple Gibbs problems, `Canonicalizer` reuse |
| [Warm Start](@ref "Warm Start") | Temperature scans, SciML caching |
| [Sensitivity](@ref "Sensitivity Analysis") | $\partial n^*/\partial b$, $\partial n^*/\partial(\mu^0/RT)$ |
| [SciML Interface](@ref "SciML Interface") | `OptimaOptimizer` with ChemistryLab.jl |
| [API Reference](@ref "API Reference") | Docstrings for all exported symbols |
```
