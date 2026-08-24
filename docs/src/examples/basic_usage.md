```@meta
CurrentModule = OptimaSolver
```

# Basic Usage

## Example 1: three-species single-element problem

The simplest non-trivial Gibbs problem: three species sharing one conserved quantity.

```@example basic
using OptimaSolver
using LinearAlgebra

μ⁰ = [0.0, 1.0, 2.0]

G(n, p)    = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
∇G!(g,n,p) = for i in eachindex(n); g[i] = p.μ⁰[i] + log(n[i]) + 1; end

A = ones(1, 3)
b = [1.0]
prob = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16, 3), p=(μ⁰=μ⁰,))
result = solve(prob)

println(result.n)   # ≈ [0.6652, 0.2447, 0.0900]
@assert result.converged
```

Analytical check: $n_i^* = e^{-\mu_i^0}\!/\sum_j e^{-\mu_j^0}$.

```@example basic
n_exact = exp.(-μ⁰) ./ sum(exp.(-μ⁰))
@assert maximum(abs, result.n .- n_exact) < 1e-7
```

## Example 2: two elements, four species

A system with two conserved elements and four species.

```@example basic
# Conservation matrix: each column = elemental composition of one species
# A[i,k] = number of atoms of element i in species k
A2 = Float64[2 1 1 2;   # element X
              1 0 1 0]  # element Y
b2 = [4.0, 1.0]        # 4 units of X, 1 unit of Y

μ⁰4 = [0.0, 0.5, 1.0, 1.5]

G4(n, p)     = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
∇G4!(g, n, p) = for i in eachindex(n); g[i] = p.μ⁰[i] + log(n[i]) + 1; end

prob4 = OptimaProblem(A2, b2, G4, ∇G4!;
                      lb = fill(1e-16, 4),
                      p  = (μ⁰ = μ⁰4,))
result4 = solve(prob4)

@assert result4.converged
@assert norm(A2 * result4.n .- b2) < 1e-8
```

## Reusing a `Canonicalizer` for temperature scans

When $A$ is fixed across a series of solves (e.g. varying temperature so only
$\mu^0$ changes), build the [`Canonicalizer`](@ref) once and pass it to `solve`.
This skips the QR decomposition on every call and reduces overhead for large
$n_s$:

```@example basic
# Build canonicalizer once (expensive QR + LU)
can = Canonicalizer(A2)

# Stand-in for whatever supplies standard chemical potentials at a temperature.
# It is a placeholder, not thermodynamics: what this example is about is reusing
# the factorisation, not where μ⁰(T) comes from.
μ⁰_at_temperature(T_K) = μ⁰4 .* (298.15 / T_K)

# Solve at many temperatures, reusing the LU factorisation
for T_K in range(298.15, 400.0; step=5.0)
    μ⁰_T = μ⁰_at_temperature(T_K)
    prob_T = OptimaProblem(A2, b2, G4, ∇G4!;
                           lb = fill(1e-16, 4),
                           p  = (μ⁰ = μ⁰_T,))
    result_T = solve(prob_T, can)     # QR not repeated
    # ... process result_T
end
```

## Verbose iteration log

Set `verbose=true` in [`OptimaOptions`](@ref) to see a per-iteration summary:

```@example basic
result = solve(prob, OptimaOptions(tol=1e-12, verbose=true))
```

Each row shows the iteration count, the current barrier weight $\mu$, the
optimality and feasibility residuals, the accepted line-search step length
$\alpha$ and the step norms. The log above is the real one: this block is
executed when the documentation is built, so what you read is what the solver
prints, not a transcription of it.

## Lower and upper bounds

Both lower and upper bounds can be set per species:

```@example basic
# Aqueous species: lb = 1e-16 (trace amount), no upper bound
# Solid species:   lb = 0, ub = total moles available
prob_mixed = OptimaProblem(A, b, G, ∇G!;
                           lb = [1e-16, 1e-16, 0.0],
                           ub = [Inf,   Inf,   2.5],
                           p  = (μ⁰ = μ⁰,))
```

For solids and gases with zero curvature ($\partial^2 G/\partial n_i^2 = 0$),
enable finite-difference Hessian computation:

```@example basic
opts = OptimaOptions(tol=1e-10, use_fd_hessian=true)
result_mixed = solve(prob_mixed, opts)
println(result_mixed.n)
```
```
