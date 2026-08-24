```@meta
CurrentModule = OptimaSolver
```

# Sensitivity Analysis

After convergence, [`sensitivity`](@ref) computes two Jacobian matrices by
implicit differentiation of the KKT system (see [Theory](@ref "Theory") for
the mathematical derivation):

| Matrix | Size | Meaning |
|--------|------|---------|
| `∂n_∂b` | $n_s \times m$ | $\partial n^*/\partial b$ — response to element budgets |
| `∂n_∂μ0` | $n_s \times n_s$ | $\partial n^*/\partial(\mu^0/RT)$ — response to standard potentials |

Both matrices are computed at marginal cost ($m$ back-substitutions against the
already-factored Schur complement $S = A H^{-1} A^\top$).

## Computing sensitivity matrices

```@example sens
using OptimaSolver

μ⁰ = [0.0, 1.0, 2.0]
G(n, p)    = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
∇G!(g,n,p) = for i in eachindex(n); g[i] = p.μ⁰[i] + log(n[i]) + 1; end
A = ones(1, 3)
b = [1.0]

prob = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16,3), p=(μ⁰=μ⁰,))
result = solve(prob, OptimaOptions(tol=1e-12))

# --- Build Hessian diagonal at the converged point ---
n = result.n
hf = gibbs_hessian_diag(n)        # ideal approximation: ∂²f/∂nᵢ² ≈ 1/nᵢ
μ_conv = 1e-14                     # barrier ≈ 0 at convergence
h  = hessian_diagonal(prob, n, μ_conv, hf)

# --- Compute sensitivity ---
sens = sensitivity(prob, n, result.y, h, μ_conv)

println("∂n*/∂b  = ", sens.∂n_∂b)    # (3 × 1) matrix
println("∂n*/∂μ⁰ = ", sens.∂n_∂μ0)   # (3 × 3) matrix
```

## Interpreting `∂n_∂b`

Column $j$ of `sens.∂n_∂b` answers: "if one extra unit of element $j$ is added
to the system, how does the equilibrium composition change?"

**Sanity check** — the extra mole must be distributed among the species, so
columns sum to 1:

```@example sens
@assert sum(sens.∂n_∂b[:, 1]) ≈ 1.0 atol=1e-8
println(sum(sens.∂n_∂b[:, 1]))
```

For a single-element problem (as here), the entire column is positive and sums
to 1 — adding more of the element increases all species proportionally.

## Interpreting `∂n_∂μ0`

Column $k$ of `sens.∂n_∂μ0` answers: "if the standard potential $\mu_k^0/RT$
increases by 1 (species $k$ becomes less stable), how does the equilibrium shift?"

Expected signs:
- Diagonal: $\partial n_k^*/\partial \mu_k^0 < 0$ (species $k$ decreases)
- Off-diagonal: $\partial n_j^*/\partial \mu_k^0 > 0$ for $j \neq k$ (competitors
  pick up the released mass)

```@example sens
@assert sens.∂n_∂μ0[1, 1] < 0   # n₁ decreases when μ₁⁰ increases
@assert sens.∂n_∂μ0[2, 1] > 0   # n₂ increases (competitor gains)
```

## Finite-difference validation

A standard way to validate the sensitivity matrices is to perturb $b$ or $\mu^0$
by a small $\delta$ and compare with the finite-difference approximation:

```@example sens
δ = 1e-5

# --- Validate ∂n*/∂b ---
b_pert = copy(b); b_pert[1] += δ
prob_pert = OptimaProblem(A, b_pert, G, ∇G!; lb=fill(1e-16,3), p=(μ⁰=μ⁰,))
r_pert = solve(prob_pert, OptimaOptions(tol=1e-12); u0=result)

∂n_∂b_fd = (r_pert.n .- result.n) ./ δ
println("IFT error: ", maximum(abs, sens.∂n_∂b[:,1] .- ∂n_∂b_fd))  # ≈ O(δ)

# --- Validate ∂n*/∂μ⁰ ---
μ⁰_pert = copy(μ⁰); μ⁰_pert[1] += δ
prob_μ_pert = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16,3), p=(μ⁰=μ⁰_pert,))
r_μ_pert = solve(prob_μ_pert, OptimaOptions(tol=1e-12); u0=result)

∂n_∂μ0_fd = (r_μ_pert.n .- result.n) ./ δ
println("IFT error: ", maximum(abs, sens.∂n_∂μ0[:,1] .- ∂n_∂μ0_fd))  # ≈ O(δ)
```

## Use in coupled kinetics–thermodynamics

The sensitivity matrices appear naturally as Jacobians when coupling a kinetics
ODE (dissolution/precipitation rates feed into $\dot{b}$) with the thermodynamics
solver:

```math
\frac{dn^*}{dt} = \frac{\partial n^*}{\partial b}\,\frac{db}{dt}
```

```math
\frac{\partial G^*}{\partial T} \approx
\frac{\partial n^*}{\partial (\mu^0/RT)}\,
\frac{\partial (\mu^0/RT)}{\partial T}
```

Providing these exact Jacobians to a stiff ODE integrator (e.g.
`DifferentialEquations.jl` with `Rodas5`) avoids finite differences at each time
step and can reduce the total solve time by an order of magnitude.

## ForwardDiff through sensitivity

Because Optima uses generic arithmetic throughout, `ForwardDiff` can
differentiate the entire sensitivity computation with respect to parameters:

```@example sens
using ForwardDiff

# Jacobian of n* w.r.t. μ⁰, computed by AD rather than IFT
jac_ad = ForwardDiff.jacobian(
    μ0 -> begin
        p_ad = (μ⁰ = μ0,)
        prob_ad = OptimaProblem(A, b, G, ∇G!; lb=fill(1e-16,3), p=p_ad)
        solve(prob_ad, OptimaOptions(tol=1e-10)).n
    end,
    μ⁰,
)

# Should agree with sens.∂n_∂μ0 (modulo sign — IFT gives ∂n*/∂(μ⁰/RT), AD gives ∂n*/∂μ⁰)
println(maximum(abs, jac_ad .- sens.∂n_∂μ0))   # < 1e-5
```
```
