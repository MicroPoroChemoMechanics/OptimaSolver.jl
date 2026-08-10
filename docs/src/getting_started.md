```@meta
CurrentModule = OptimaSolver
```

# Getting Started

## Installation

OptimaSolver.jl requires Julia ≥ 1.10. Install it from the Julia package manager:

```julia
julia> import Pkg; Pkg.add("OptimaSolver")
```

or in Pkg REPL mode (press `]`):

```
pkg> add OptimaSolver
```

Dependencies (`ForwardDiff`, `SciMLBase`) are resolved automatically.

## First solve: three-species ideal Gibbs problem

This minimal example solves a Gibbs-energy minimization for a three-species system
under a single mole-balance constraint.

**Problem.** Find mole amounts $n = (n_1, n_2, n_3)$ minimizing

```math
G(n) = \sum_{i=1}^{3} n_i\!\left(\mu_i^0 + \ln n_i\right),
\qquad \mu^0 = (0, 1, 2),
```

subject to $n_1 + n_2 + n_3 = 1$ and $n_i \geq 10^{-16}$.

**Analytical solution.** At the minimum,
$n_i^* = e^{-\mu_i^0}/Z$ where $Z = \sum_j e^{-\mu_j^0}$,
giving $n^* \approx (0.665,\; 0.245,\; 0.090)$.

```julia
using OptimaSolver

# --- Objective and gradient ---
μ⁰ = [0.0, 1.0, 2.0]

G(n, p) = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))

function ∇G!(grad, n, p)
    for i in eachindex(n)
        grad[i] = p.μ⁰[i] + log(n[i]) + 1
    end
end

# --- Build problem ---
A = ones(1, 3)    # single conservation: n₁ + n₂ + n₃ = 1
b = [1.0]

prob = OptimaProblem(A, b, G, ∇G!;
                     lb = fill(1e-16, 3),
                     p  = (μ⁰ = μ⁰,))

# --- Solve ---
result = solve(prob, OptimaOptions(tol=1e-12, verbose=false))

println("Converged:  ", result.converged)    # true
println("Iterations: ", result.iterations)   # typically 15–25
println("n* = ", round.(result.n; digits=6)) # [0.665241, 0.244728, 0.090031]

# Compare with analytical solution
n_exact = exp.(-μ⁰) ./ sum(exp.(-μ⁰))
@assert maximum(abs, result.n .- n_exact) < 1e-7
```

## Interpreting `OptimaResult`

[`OptimaResult`](@ref) carries:

| Field | Type | Description |
|-------|------|-------------|
| `n` | `Vector{T}` | equilibrium mole amounts |
| `y` | `Vector{T}` | Lagrange multipliers ($\approx \partial G^*/\partial b$) |
| `iterations` | `Int` | total Newton iterations |
| `converged` | `Bool` | `true` if KKT residual < `tol` |
| `error_opt` | `T` | final $\|e_x\|_\infty$ (optimality) |
| `error_feas` | `T` | final $\|e_w\|_\infty$ (feasibility) |

## Tuning `OptimaOptions`

Key options in [`OptimaOptions`](@ref):

| Option | Default | Effect |
|--------|---------|--------|
| `tol` | `1e-10` | KKT convergence tolerance |
| `max_iter` | `300` | maximum Newton iterations |
| `verbose` | `false` | print per-iteration log |
| `barrier_init` | `1e-4` | initial barrier weight $\mu_0$ |
| `barrier_decay` | `0.1` | $\mu \leftarrow 0.1\,\mu$ each outer step |
| `use_fd_hessian` | `false` | finite-difference Hessian diagonal |

### When to use `use_fd_hessian = true`

The default Hessian approximation $h_i = 1/n_i$ is exact for an ideal solution
(objective $\sum_i n_i \ln n_i$). For problems that include pure solid or pure gas
species, the true $\partial^2 f/\partial n_i^2 = 0$, which makes the approximation
$1/n_i$ inflate $h_i$ by factors up to $10^{16}/n_i$. This inflated value dominates
the Schur complement and the effective Newton step for such species is negligible,
causing the solver to converge only linearly (hundreds of iterations) instead of
quadratically.

Setting `use_fd_hessian = true` computes $h_i$ by a forward finite difference on
$\nabla f$ at marginal cost and yields correct quadratic convergence for all problem
types. The [`OptimaOptimizer`](@ref) SciML interface defaults to
`use_fd_hessian = true` for this reason.

## Running the documentation locally

```bash
julia --project=docs docs/make.jl
```

Then open `docs/build/index.html` in a browser.
```
