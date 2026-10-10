<p align="center">
  <img src="./docs/src/assets/logo.svg" alt="OptimaSolver.jl" width="100"/>
</p>

# OptimaSolver.jl

[![Docs - Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/stable/)
[![Docs - Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/dev/)

[![CI](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MicroPoroChemoMechanics/OptimaSolver.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MicroPoroChemoMechanics/OptimaSolver.jl)

[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-pink)](https://github.com/fredrikekre/Runic.jl)

[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL_v2.1+-blue.svg)](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19534031-blue)](https://doi.org/10.5281/zenodo.19534031)

Gibbs-energy minimization for equilibrium chemistry, in Julia: an interior-point
method ported from the Optima C++ library, and a Newton method on the optimality
conditions in the space of the conservation multipliers, whose answers a KKT
certificate checks.

## What it does

Both solvers minimize a Gibbs energy under linear mass balances,

```
minimize    f(n)           (e.g. G(n) = Σ nᵢ(μᵢ⁰/RT + ln nᵢ))
subject to  A n = b        (mass conservation, m equations)
            n ≥ 0
```

### `solve` — the interior-point method, ported from Optima

- **Schur-complement Newton step** — reduces the KKT system from $(n_s+m)\times(n_s+m)$
  to $m\times m$ by exploiting the diagonal Hessian structure. For typical chemistry
  problems $m$ is the number of elements ($\leq 15$), so this is a dramatic reduction.
- **Filter line search** (Wächter & Biegler 2006) with Armijo sufficient decrease on the
  barrier objective.
- **Variable stability classification** — near-bound species receive reduced steps.
- **Implicit-differentiation sensitivity** — post-solve computation of $\partial n^{\ast}/\partial b$
  and $\partial n^{\ast}/\partial(\mu^0/RT)$ at marginal cost.
- **Warm start** — consecutive solves reuse the previous solution as the starting point.

### `dual_newton_solve` — Newton on the KKT conditions, with a certificate

Original to OptimaSolver.jl. The objective is written $\sum_i x_i\,(g_i + h_i(x))$, $h$
the part of the chemical potentials that depends on the composition.

- **Newton in the space of the multipliers** — the Brinkley–Karpov formulation: the
  members of a mixing phase follow from the multipliers through their mass-action laws,
  solved in log-amounts so that no fraction-to-boundary rule caps the step, and the pure
  phases are held in an active set, exactly.
- **Mixing phases** — `SolutionPhase`: a phase with a solvent, whose solutes a supplied
  `invert` can recover in one call, or a mole-fraction phase (a solid solution).
- **The phase rule in the search** — an active set never holds more stationarity
  conditions than there are conservation rows (`stationarity_capacity`), and a component
  absent from the budget is recognized as such (`degenerate_components`).
- **A certificate, not a stopping test** — `kkt_certificate` checks the KKT conditions at
  any point, whatever produced it, each mass balance against what it holds; for a convex
  problem `optimal = true` proves the global minimum.
- **Phase stability** — Michelsen's tangent-plane test, for an absent mixing phase
  (`phase_tangent_trial`) and for a present one that would split (`phase_split_trial`),
  with the trial composition it finds.
- **Starts** — the vertex of the pure-phase linear program by a verified simplex
  (`lp_start`), which proves an impossible budget infeasible, and the multipliers of
  the concave dual of the ideal problem (Brinkley's method).
- **Unknown parameters** — a prescribed property (a pH, a temperature) or a reaction
  extent, solved for together with the composition.
- **Derivatives of the answer** — dual numbers in the budget or in the data give, through
  `dual_newton_tangent`, the derivatives of the answer by the implicit-function theorem,
  nested differentiations kept apart by their tags.

Both solvers are written in generic Julia arithmetic, so `ForwardDiff` dual numbers pass
through them.

## Installation

OptimaSolver.jl is registered in Julia's General registry.

In Pkg REPL mode (press `]` in the Julia REPL):

```julia-repl
pkg> add OptimaSolver
```

Or via the `Pkg` API:

```julia
using Pkg
Pkg.add("OptimaSolver")
```

Requires Julia ≥ 1.12.

## Quick example

```julia
using OptimaSolver

# Ideal three-species Gibbs problem: minimize Σ nᵢ(μᵢ⁰ + ln nᵢ) subject to Σ nᵢ = 1
μ⁰ = [0.0, 1.0, 2.0]

G(n, p)    = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
∇G!(g,n,p) = for i in eachindex(n); g[i] = p.μ⁰[i] + log(n[i]) + 1; end

A = ones(1, 3)
b = [1.0]
prob = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16, 3), p=(μ⁰=μ⁰,))
result = solve(prob, OptimaOptions(tol=1e-12))

println(result.n)          # ≈ [0.665241, 0.244728, 0.090031]  (exp(-μᵢ⁰)/Z)
println(result.converged)  # true
println(result.iterations) # 32 at this tolerance
```

## SciML / ChemistryLab interface

`OptimaOptimizer` is a drop-in replacement for `IpoptOptimizer` in
[ChemistryLab.jl](https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl):

```julia
using ChemistryLab, OptimaSolver
state_eq = equilibrate(state0; solver=OptimaOptimizer(tol=1e-10, verbose=false))
```

The SciML interface handles variable scaling (critical for multi-decade concentration
ranges), cold-start lifting of absent species, and transparent warm-start caching
between consecutive solves. A constraint that is not affine is linearized again at each
answer, and `Success` is reported only when the constraint is met.

## Documentation

- [**STABLE**](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/stable/) — most recently tagged version of the documentation.
- [**DEV**](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/dev/) — development version of the documentation.

## Credits and lineage

OptimaSolver.jl began as a Julia port of the **Optima** C++ library developed by
[Allan Leal](https://erdw.ethz.ch/en/people/profile.allan-leal.html) (ETH Zürich),
<https://github.com/reaktoro/optima>. The interior-point solver keeps that lineage: the
canonicalization of the conservation matrix, the Schur-complement Newton step, the
stability classification of the variables and the sensitivity of the answer come from
that library and from

> Leal, A.M.M., Blunt, M.J., LaForce, T.C. (2014).
> Efficient chemical equilibrium calculations for geochemical speciation and reactive
> transport modelling.
> *Geochimica et Cosmochimica Acta*, **131**, 301–322.
> <https://doi.org/10.1016/j.gca.2014.01.038>

Its log-barrier formulation and its filter line search follow

> Wächter, A., Biegler, L.T. (2006).
> On the implementation of an interior-point filter line-search algorithm for
> large-scale nonlinear programming.
> *Mathematical Programming*, **106**(1), 25–57.
> <https://doi.org/10.1007/s10107-004-0559-y>

The Newton solver in the space of the multipliers and what surrounds it — the KKT
certificate, the phase-stability tests, the linear-programming start and the
derivatives of the answer — are original to OptimaSolver.jl (`src/dual_newton.jl`,
`src/dual_newton_ad.jl` and `src/lp.jl`, more than half of the code). They implement
published methods:

> Brinkley, S.R. (1947). Calculation of the equilibrium composition of systems of many
> constituents. *The Journal of Chemical Physics*, **15**(2), 107–110.
> <https://doi.org/10.1063/1.1746420>
>
> White, W.B., Johnson, S.M., Dantzig, G.B. (1958). Chemical equilibrium in complex
> mixtures. *The Journal of Chemical Physics*, **28**(5), 751–755.
> <https://doi.org/10.1063/1.1744264>
>
> Karpov, I.K., Chudnenko, K.V., Kulik, D.A. (1997). Modeling chemical mass transfer in
> geochemical processes; thermodynamic relations, conditions of equilibria and numerical
> algorithms. *American Journal of Science*, **297**(8), 767–806.
> <https://doi.org/10.2475/ajs.297.8.767>
>
> Michelsen, M.L. (1982). The isothermal flash problem. Part I. Stability.
> *Fluid Phase Equilibria*, **9**(1), 1–19.
> <https://doi.org/10.1016/0378-3812(82)85001-2>

The Julia port was authored by Jean-François Barthélémy (CEREMA, France) with
assistance from [Claude Code](https://claude.ai/code) (Anthropic).

OptimaSolver.jl is **independent** of the upstream Optima library: it is not affiliated
with, endorsed by, or supported by its authors or by the Reaktoro project. Please report
problems with this package here rather than to them. The package logo is an original
work made for this package and reproduces no mark of the upstream project.

## License

OptimaSolver.jl is licensed under the **GNU Lesser General Public License,
version 2.1 or (at your option) any later version** (LGPL-2.1-or-later),
matching the license of the upstream Optima C++ library from which its
interior-point solver is derived.

- Copyright © 2020–2024 Allan Leal (original C++ Optima).
- Copyright © 2026 Jean-François Barthélémy (Julia port).
- Copyright © 2025–2026 Jean-François Barthélémy (Cerema, UMR MCD) (`src/dual_newton.jl`,
  `src/dual_newton_ad.jl`, `src/lp.jl`).

See [`LICENSE`](LICENSE) for the full notice and [`COPYING.LESSER`](COPYING.LESSER)
for the full LGPL-2.1 text.

**Practical note for downstream users.** The LGPL permits
`using OptimaSolver` from Julia code of **any** license (including MIT,
Apache-2.0, or proprietary code). The copyleft applies only to modifications
of OptimaSolver.jl itself, which must remain LGPL.

## Citation

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19534031-blue)](https://doi.org/10.5281/zenodo.19534031)

If you use OptimaSolver.jl, please cite it as below, and for the interior-point algorithm
also Leal et al. (2014). See [CITATION.cff](CITATION.cff) for citation details.

```bibtex
@software{optimasolver_jl,
  author    = {Barth{\'e}lemy, Jean-Fran{\c{c}}ois},
  title     = {{OptimaSolver.jl}: certified Gibbs-energy minimization for equilibrium chemistry},
  doi       = {10.5281/zenodo.19534031},
  url       = {https://doi.org/10.5281/zenodo.19534031},
  year      = {2026}
}
```
