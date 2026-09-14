```@meta
CurrentModule = OptimaSolver
```

# Warm Start

When solving a sequence of related problems (varying temperature, pH, element
amounts, etc.), passing the solution of one solve as the initial guess of the
next reuses the composition and the multipliers, and avoids the cold-start
lifting heuristic.

!!! warning "It does not always pay, and on the example below it does not"
    A warm start reuses ``n`` and ``y``. It does **not** reuse the barrier
    parameter: `μ` is reset to `barrier_init` on every solve. So the iteration
    restarts from a point that sits near the solution of the *previous* problem
    at ``\mu \to 0`` — which is far from the central path of ``\mu`` = 1e-4,
    and the iteration has to travel back out before it can come in again.

    Measured on the three-species problem below, at `tol = 1e-12`:

    | | iterations |
    |:--|--:|
    | first problem, cold | 55 |
    | second problem, **cold** | **39** |
    | second problem, **warm-started** | **52** |

    Thirteen iterations *worse* than starting from scratch. And the barrier
    reset is demonstrably the cause — lowering `barrier_init` for the warm solve
    alone recovers the saving and more:

    | `barrier_init` | warm | cold |
    |--:|--:|--:|
    | 1e-4 (default) | 52 | 39 |
    | 1e-6 | 41 | 43 |
    | 1e-8 | **17** | 31 |
    | 1e-10 | 34, **not converged** | 43 |

    So on a problem that is easy to start cold, warm-starting at the default
    barrier costs rather than saves. What a warm start does buy unconditionally
    is the *starting composition*: through the SciML interface it skips the
    cold-start lifting heuristic described at the bottom of this page, which is
    the part that matters when species are absent and a cold start has to invent
    an interior point for them. That is a different saving from the iteration
    count, and it is not measured here.

    Tuning `barrier_init` down for a warm solve is a caller's lever and not a
    recommendation: the last row shows it stops converging if pushed. Carrying
    the barrier level across solves is the fix, and it is not implemented —
    `OptimaResult` does not carry `μ`.

## Direct API warm-start

Pass a previous [`OptimaResult`](@ref) as the `u0` keyword argument:

```@example warm
using OptimaSolver

μ⁰ = [0.0, 1.0, 2.0]
G(n, p)    = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
∇G!(g,n,p) = for i in eachindex(n); g[i] = p.μ⁰[i] + log(n[i]) + 1; end
A = ones(1, 3)
b = [1.0]
opts = OptimaOptions(tol=1e-12)

# Cold start (first solve)
prob1 = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16,3), p=(μ⁰=μ⁰,))
r1 = solve(prob1, opts)
println("Cold start: ", r1.iterations, " iterations")

# Warm start: pass r1 directly as u0
μ⁰2 = [0.0, 0.9, 2.1]
prob2 = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16,3), p=(μ⁰=μ⁰2,))
r2 = solve(prob2, opts; u0=r1)
println("Warm start: ", r2.iterations, " iterations")

# The same problem from scratch, for comparison — this is the number that
# decides whether the warm start was worth anything.
r2_cold = solve(prob2, opts)
println("Cold, same problem: ", r2_cold.iterations, " iterations")
```

The solver reads `r1.n` and `r1.y` as the initial ``(n, y)``, and **resets** the
barrier parameter to `barrier_init`. `iterations` counts Newton steps in total,
not barrier levels — the outer loop runs far fewer than that, but the two numbers
are not interchangeable and the one printed above is the total.

On this problem the warm start loses, for the reason given in the warning at the
top of the page. It is shown here because it is what the code does, and because
a warm start that is assumed rather than measured is how a sequence of solves
gets slower without anyone noticing.

## Temperature scan with `Canonicalizer` reuse

Combine warm-start with a pre-built [`Canonicalizer`](@ref) to minimize overhead
when both $A$ and $\mu^0$ change slowly:

```@example warm
can = Canonicalizer(A)   # fixed A, build QR + LU once

# Stand-in for whatever supplies standard chemical potentials at a temperature.
# A placeholder, not thermodynamics: the subject here is warm-starting.
μ⁰_at_temperature(T_K) = μ⁰ .* (298.15 / T_K)

results = OptimaResult[]
prev = nothing

for T_K in range(298.15, 400.0; step=5.0)
    p_T = (μ⁰ = μ⁰_at_temperature(T_K),)
    prob_T = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16, 3), p=p_T)
    r = solve(prob_T, can, opts; u0=prev)
    push!(results, r)
    global prev = r   # warm-start next step
end

println(length(results), " solves, iterations: ", [r.iterations for r in results])
```

!!! note "Non-converged results"
    If a solve does not converge (e.g. at a pathological temperature point),
    `result.converged == false`. It is safe to pass such a result as `u0` for
    the next solve, but the warm-start quality will be poor. Consider falling
    back to a cold start if `!result.converged`.

## SciML: automatic caching in `OptimaOptimizer`

[`OptimaOptimizer`](@ref) caches the last **converged** result automatically in
an internal `Ref`. Consecutive calls to `SciMLBase.solve` on related problems will
warm-start transparently:

```julia
using OptimaSolver

alg = OptimaOptimizer(tol=1e-10, warm_start=true)

# First call: cold start, result stored in alg._cache
sol1 = SciMLBase.solve(opt_prob_T1, alg)

# Second call: warm-starts from sol1 automatically
sol2 = SciMLBase.solve(opt_prob_T2, alg)

# Third call: warm-starts from sol2
sol3 = SciMLBase.solve(opt_prob_T3, alg)
```

**Reset the cache** whenever the chemical system changes (new set of species,
different $A$ matrix):

```julia
reset_cache!(alg)
# Next call will be a cold start
```

**Non-converged solutions are never cached**: if a call fails to converge, the
cache is unchanged and the next call falls back to the cold start based on
`opt_prob.u0`.

## Cold-start lifting

When starting from scratch (no warm-start or absent species), the SciML interface
performs *cold-start lifting*: species that are at their lower bound $\ell_i$
(i.e. not set by the caller) are raised to a rough element-balance estimate
$n_i \approx b_j / \sum_k A_{jk}$ times a small fraction (default $10^{-3}$).

This is necessary because starting at $n_i = \ell_i$ places that species exactly
on the log-barrier boundary: $-\mu/(n_i - \ell_i) \to -\infty$, making the
barrier gradient $\sim 10^{12}$ times larger than the chemical gradient, so the
solver can only advance the species by $\sim\varepsilon$ per Newton step. Lifting
to $O(10^{-3})$ puts the species well inside the barrier and allows normal
Newton convergence from the first iteration.
```
