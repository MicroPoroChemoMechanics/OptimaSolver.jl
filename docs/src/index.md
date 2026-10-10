```@meta
CurrentModule = OptimaSolver
```

# OptimaSolver.jl

**OptimaSolver** minimizes a Gibbs energy under linear mass balances, the problem
every chemical equilibrium reduces to. It provides an interior-point method ported
from the Optima C++ library, and a Newton method on the optimality conditions in the
space of the conservation multipliers, whose answers a KKT certificate checks.

Both solve problems of the form

```math
\min_{n \in \mathbb{R}^{n_s}}\ f(n)
\qquad \text{subject to} \qquad
A n = b,\quad n \geq 0
```

where $f$ is the Gibbs free energy (e.g. $G(n) = \sum_i n_i(\mu_i^0/RT + \ln n_i)$
for an ideal/dilute solution), $A \in \mathbb{R}^{m \times n_s}$ is the
mass-conservation (stoichiometric) matrix and $b \in \mathbb{R}^m$ is the
element-abundance vector.

## The interior-point method, ported from Optima

[`solve`](@ref) on an [`OptimaProblem`](@ref):

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
  reuse the previous solution as the starting point. Whether that saves iterations
  depends on the problem, since the barrier parameter restarts from `barrier_init`
  on every solve; [Warm Start](@ref "Warm Start") measures a case where it does not.

## Newton on the KKT conditions, with a certificate

[`dual_newton_solve`](@ref) on a [`DualNewtonProblem`](@ref), original to this
package (see [Theory](@ref "Theory")):

- **Newton in the space of the multipliers** — the Brinkley–Karpov formulation: the
  members of a mixing phase follow from the multipliers through their mass-action
  laws, solved in log-amounts so that no fraction-to-boundary rule caps the step, and
  the pure phases are held in an active set, exactly.
- **Mixing phases** — [`SolutionPhase`](@ref): a phase with a solvent, whose solutes
  a supplied `invert` can recover in one call, or a mole-fraction phase.
- **The phase rule in the search** — an active set never holds more stationarity
  conditions than there are conservation rows ([`stationarity_capacity`](@ref)).
- **A certificate, not a stopping test** — [`kkt_certificate`](@ref) checks the KKT
  conditions at any point, whatever produced it; for a convex problem it proves the
  global minimum.
- **Phase stability** — Michelsen's tangent-plane test for an absent mixing phase
  ([`phase_tangent_trial`](@ref)) and for a present one that would split
  ([`phase_split_trial`](@ref)).
- **Starts** — the vertex of the pure-phase linear program by a verified simplex
  ([`lp_start`](@ref)), which proves an impossible budget infeasible.
- **Derivatives of the answer** — [`dual_newton_tangent`](@ref) lifts the answer to
  the dual numbers of the budget or the data, by the implicit-function theorem.

Both solvers are written in generic Julia arithmetic, so `ForwardDiff` dual numbers
pass through them, and [`OptimaOptimizer`](@ref) implements
`SciMLBase.AbstractOptimizationAlgorithm`, a drop-in replacement for
`IpoptOptimizer` in ChemistryLab.jl.

## Lineage

OptimaSolver began as a Julia port of the **optima** C++ library developed by
[Allan Leal](https://erdw.ethz.ch/en/people/profile.allan-leal.html) (ETH Zürich),
<https://github.com/reaktoro/optima>. The interior-point solver keeps that lineage:
the canonicalization of the conservation matrix, the Schur-complement reduction, the
stability classification of the variables and the sensitivity of the answer come
from that library and from Leal, Blunt and LaForce (2014),
*Geochimica et Cosmochimica Acta* **131**, 301–322,
<https://doi.org/10.1016/j.gca.2014.01.038>. Its log-barrier formulation and filter
line search follow Wächter and Biegler (2006).

The Newton solver in the space of the multipliers, the certificate, the
phase-stability tests, the linear-programming start and the derivatives of the
answer are original to OptimaSolver.jl. They implement the published methods of
Brinkley (1947), White, Johnson and Dantzig (1958), Karpov, Chudnenko and Kulik
(1997) and Michelsen (1982), referenced in [Theory](@ref "Theory").

The Julia port was authored by Jean-François Barthélémy (CEREMA, France) with
assistance from [Claude Code](https://claude.ai/code) (Anthropic).

OptimaSolver.jl is **independent** of the upstream Optima library: it is not
affiliated with, endorsed by, or supported by its authors or by the Reaktoro project.
Please report problems with this package here rather than to them. The package logo
is an original work made for this package and reproduces no mark of the upstream
project.

## Documentation structure

| Section | Content |
|---------|---------|
| [Getting Started](@ref "Getting Started") | Installation and first solve |
| [Theory](@ref "Theory") | Interior-point algorithm, Newton in multiplier space, certificate, linear program, sensitivity |
| [Basic Usage](@ref "Basic Usage") | Simple Gibbs problems, `Canonicalizer` reuse |
| [Warm Start](@ref "Warm Start") | Temperature scans, SciML caching |
| [Sensitivity](@ref "Sensitivity Analysis") | $\partial n^*/\partial b$, $\partial n^*/\partial(\mu^0/RT)$ |
| [SciML Interface](@ref "SciML Interface") | `OptimaOptimizer` with ChemistryLab.jl |
| [API Reference](@ref "API Reference") | Docstrings for all exported symbols |
```
