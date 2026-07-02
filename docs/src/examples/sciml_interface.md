```@meta
CurrentModule = OptimaSolver
```

# SciML Interface

[`OptimaOptimizer`](@ref) implements `SciMLBase.AbstractOptimizationAlgorithm`
and is designed as a drop-in replacement for `IpoptOptimizer` inside
[ChemistryLab.jl](https://github.com/MicroPoroChemoMechanics/ChemistryLab.jl).

## Usage with ChemistryLab.jl

```julia
using ChemistryLab, OptimaSolver

# Build a ChemicalState as usual, then equilibrate with OptimaOptimizer
state_eq = equilibrate(state0; solver=OptimaOptimizer(tol=1e-10, verbose=false))
```

`OptimaOptimizer` handles:
- **Variable scaling** — each species is scaled by its starting value so that all
  normalised variables are $O(1)$, critical for convergence when concentrations span
  multiple decades (e.g. a titration from pH 1 to pH 13).
- **Cold-start lifting** — absent species (at their lower bound) are raised to a
  rough element-balance estimate before the first Newton step, avoiding the
  log-barrier singularity.
- **Transparent warm-start caching** — each converged result is stored; the next
  call starts from it automatically.

## Constructor options

```julia
alg = OptimaOptimizer(;
    tol            = 1e-10,  # KKT convergence tolerance
    max_iter       = 300,    # maximum Newton iterations
    warm_start     = true,   # cache and reuse last converged result
    verbose        = false,  # print iteration log
    barrier_init   = 1e-4,   # initial barrier weight μ₀
    barrier_decay  = 0.1,    # μ ← 0.1 μ per outer step
    use_fd_hessian = true,   # correct for mixed solid/aqueous (default true here)
)
```

!!! note "`use_fd_hessian` differs from `OptimaOptions` default"
    In [`OptimaOptions`](@ref) the default is `false` (ideal-solution Hessian
    $h_i = 1/n_i$). In `OptimaOptimizer` the default is `true` because real
    chemical systems typically include pure solid or gas species with zero
    curvature, for which the ideal approximation would cause extremely slow
    convergence.

Alternatively, pass a pre-built [`OptimaOptions`](@ref):

```julia
opts = OptimaOptions(tol=1e-12, verbose=true, use_fd_hessian=true)
alg  = OptimaOptimizer(opts)
```

## Warm-start cache management

```julia
alg = OptimaOptimizer(tol=1e-10, warm_start=true)

# First call: cold start (no cache yet)
sol1 = SciMLBase.solve(opt_prob_pH5, alg)

# Second call: warm-starts from sol1
sol2 = SciMLBase.solve(opt_prob_pH6, alg)

# Reset cache when the chemical system changes
reset_cache!(alg)

# Next call is a cold start again
sol3 = SciMLBase.solve(opt_prob_new_system, alg)
```

Non-converged solutions are **never cached**: if a call fails, the cache is
unchanged and the next call falls back to a cold start from `opt_prob.u0`.

## Direct use with `SciMLBase.OptimizationProblem`

`OptimaOptimizer` works with any `SciMLBase.OptimizationProblem` encoding mass-balance
constraints as equalities:

```julia
using OptimaSolver

ns = 3
A = ones(1, ns)
b = [1.0]
μ⁰ = [0.0, 1.0, 2.0]

f_sci = SciMLBase.OptimizationFunction(
    (u, p) -> sum(u[i] * (p.μ⁰[i] + log(u[i])) for i in eachindex(u)),
    grad = (g, u, p) -> for i in eachindex(u); g[i] = p.μ⁰[i] + log(u[i]) + 1; end,
    cons = (res, u, p) -> (res .= A * u .- b),
)
u0  = fill(1/ns, ns)
lb  = fill(1e-16, ns)
opt_prob = SciMLBase.OptimizationProblem(f_sci, u0, (μ⁰=μ⁰,);
                                          lb     = lb,
                                          lcons  = zeros(1),
                                          ucons  = zeros(1))

sol = SciMLBase.solve(opt_prob, OptimaOptimizer())
println(sol.u)        # ≈ [0.6652, 0.2447, 0.0900]
println(sol.retcode)  # SciMLBase.ReturnCode.Success
```

The raw [`OptimaResult`](@ref) is accessible via `sol.original`:

```julia
raw = sol.original           # OptimaResult
println(raw.converged)       # true
println(raw.iterations)      # number of Newton iterations
println(raw.error_opt)       # final optimality residual
```

## Constraint extraction mechanism

`OptimaOptimizer` needs the linear constraint matrix $A$ and vector $b$ explicitly.
It extracts them via one of three paths (in order of preference):

1. **Explicit parameters**: if `p` is a `NamedTuple` with fields `:A` and `:b`,
   they are used directly. This is the most efficient path.
2. **No constraints**: if `cons` is `nothing`, the problem is treated as
   unconstrained ($A = 0 \times n_s$, $b = []$).
3. **Finite-difference extraction**: otherwise, $A$ is recovered by forward
   differencing the constraint function at `u0` (one evaluation per species).
   Accurate for linear constraints; adds $n_s$ extra function evaluations at
   problem setup time.

To use path 1 directly:

```julia
opt_prob = SciMLBase.OptimizationProblem(f_sci, u0, (μ⁰=μ⁰, A=A, b=b);
                                          lb=lb, lcons=zeros(1), ucons=zeros(1))
sol = SciMLBase.solve(opt_prob, OptimaOptimizer())
```

## Variable scaling details

Internally the SciML interface transforms the problem to scaled coordinates
$\tilde{n}_i = n_i / s_i$ where $s_i = \max(n_i^{(0)},\, 10^{-10})$.
The scaled conservation matrix $\tilde{A}_{ij} = s_j A_{ij}$ makes the Schur
complement well-conditioned across the full range of species concentrations.
The transformation is fully transparent: `sol.u` is always returned in the original
unscaled units.

This scaling is particularly important in titration problems where
$[\mathrm{H}^+]$ can span 12 orders of magnitude: without scaling, species
present at $10^{-12}$ mol effectively vanish from the Newton step and the solver
makes no progress toward the correct speciation.
```
