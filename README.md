# OptimaSolver.jl

[![Docs - Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/stable/)
[![Docs - Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/dev/)

[![CI](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/actions/workflows/CI.yml)

[![License: LGPL v2.1+](https://img.shields.io/badge/License-LGPL_v2.1+-blue.svg)](https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl/blob/main/LICENSE)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-pink)](https://github.com/fredrikekre/Runic.jl)

A Julia-native primal-dual interior-point solver for Gibbs-energy minimisation in
equilibrium chemistry.

## What it does

OptimaSolver solves constrained optimisation problems of the form

```
minimize    f(n, p)              (e.g. Gibbs energy G(n) = Σ nᵢ(μᵢ⁰/RT + ln nᵢ))
subject to  A n = b              (mass conservation, m equations)
            n ≥ ε                (positivity bounds)
```

The algorithm is a log-barrier interior-point method with:

- **Schur-complement Newton step** — reduces the KKT system from $(n_s+m)\times(n_s+m)$
  to $m\times m$ by exploiting the diagonal Hessian structure. For typical chemistry
  problems $m$ is the number of elements ($\leq 15$), so this is a dramatic reduction.
- **Filter line search** (Wächter & Biegler 2006) with Armijo sufficient decrease on the
  barrier objective.
- **Implicit-differentiation sensitivity** — post-solve computation of $\partial n^{\ast}/\partial b$
  and $\partial n^{\ast}/\partial(\mu^0/RT)$ at marginal cost.
- **Warm-start** — consecutive solves reuse the previous solution as the starting point.
- **ForwardDiff/AD compatibility** — no `Float64` casts; the entire solver stack uses
  generic Julia arithmetic.
- **SciML drop-in** — `OptimaOptimizer` implements `SciMLBase.AbstractOptimizationAlgorithm`
  and is a drop-in replacement for `IpoptOptimizer` in ChemistryLab.jl.

## Installation

OptimaSolver.jl is hosted on the [MPCM registry](https://github.com/MicroPoroChemoMechanics/MPCM-Registry).
Add the registry once, then install as usual.

In Pkg REPL mode (press `]` in the Julia REPL):

```julia-repl
pkg> registry add https://github.com/MicroPoroChemoMechanics/MPCM-Registry
pkg> add OptimaSolver
```

Or via the `Pkg` API:

```julia
using Pkg
Pkg.Registry.add(RegistrySpec(url="https://github.com/MicroPoroChemoMechanics/MPCM-Registry"))
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
println(result.iterations) # typically 15–25
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
between consecutive solves.

## Documentation

- [**STABLE**](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/stable/) — most recently tagged version of the documentation.
- [**DEV**](https://MicroPoroChemoMechanics.github.io/OptimaSolver.jl/dev/) — development version of the documentation.

## Credits and lineage

OptimaSolver.jl is a Julia port of the **Optima** C++ library developed by
[Allan Leal](https://erdw.ethz.ch/en/people/profile.allan-leal.html) (ETH Zürich):

<https://github.com/reaktoro/optima>

The algorithmic design — Schur-complement Newton step, filter line search, variable
stability classification, and implicit-differentiation sensitivity — originates from
that library and from the following reference:

> Leal, A.M.M., Blunt, M.J., LaForce, T.C. (2014).
> Efficient chemical equilibrium calculations for geochemical speciation and reactive
> transport modelling.
> *Geochimica et Cosmochimica Acta*, **131**, 301–322.
> <https://doi.org/10.1016/j.gca.2014.01.006>

The Julia port was authored by Jean-François Barthélémy (CEREMA, France) with
assistance from [Claude Code](https://claude.ai/code) (Anthropic).

## License

OptimaSolver.jl is licensed under the **GNU Lesser General Public License,
version 2.1 or (at your option) any later version** (LGPL-2.1-or-later),
matching the licence of the upstream Optima C++ library from which it is
derived.

- Copyright © 2020–2024 Allan Leal (original C++ Optima).
- Copyright © 2026 Jean-François Barthélémy (Julia port).

See [`LICENSE`](LICENSE) for the full notice and [`COPYING.LESSER`](COPYING.LESSER)
for the full LGPL-2.1 text.

**Practical note for downstream users.** The LGPL permits
`using OptimaSolver` from Julia code of **any** licence (including MIT,
Apache-2.0, or proprietary code). The copyleft applies only to modifications
of OptimaSolver.jl itself, which must remain LGPL.

## Citation

See [CITATION.cff](CITATION.cff) for citation details.

```bibtex
@software{optimasolver_jl,
  author    = {Barth{\'e}lemy, Jean-Fran{\c{c}}ois},
  title     = {{OptimaSolver.jl}: Julia-native primal-dual interior-point solver for Gibbs-energy minimisation},
  url       = {https://github.com/MicroPoroChemoMechanics/OptimaSolver.jl},
  year      = {2026}
}
```
