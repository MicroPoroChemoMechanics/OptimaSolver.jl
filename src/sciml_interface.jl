# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── sciml_interface.jl ─────────────────────────────────────────────────────────
# Drop-in replacement for IpoptOptimizer() in ChemistryLab.jl.
#
# Usage (ChemistryLab side — no internal changes required):
#
#   using OptimaSolver
#   state_eq = equilibrate(state0; solver = OptimaOptimizer())
#
# The `OptimaOptimizer` struct satisfies the SciML `AbstractOptimizationAlgorithm`
# interface. `SciMLBase.solve` is dispatched to convert the generic
# `OptimizationProblem` into an internal `OptimaProblem` and call the
# primal-dual solver.
#
# State caching for warm-start:
# The optimizer caches the last `OptimaResult` in a `Ref` so that consecutive
# calls (e.g. during a temperature scan) reuse the previous solution.

"""
    OptimaOptimizer

Drop-in SciML optimizer implementing the OptimaSolver primal-dual interior-point
algorithm for Gibbs-energy minimization.

# Constructors
```julia
OptimaOptimizer(; tol=1e-10, max_iter=300, warm_start=true, verbose=false)
OptimaOptimizer(opts::OptimaOptions)
```

# Fields
- `options`: `OptimaOptions` with all algorithm hyperparameters
- `_cache`:  `Ref{Union{Nothing, OptimaResult}}` — previous solution for warm-start
"""
struct OptimaOptimizer <: SciMLBase.AbstractOptimizationAlgorithm
    options::OptimaOptions
    _cache::Ref{Union{Nothing, OptimaResult}}
end

function OptimaOptimizer(;
        tol::Float64 = 1.0e-10,
        max_iter::Int = 300,
        warm_start::Bool = true,
        barrier_init::Float64 = 1.0e-4,
        barrier_min::Float64 = 1.0e-14,
        barrier_decay::Float64 = 0.2,
        barrier_eps_factor::Float64 = 1.0,
        barrier_stall_iters::Int = 8,
        ls_alpha::Float64 = 1.0e-4,
        ls_beta::Float64 = 0.5,
        ls_max_iter::Int = 40,
        verbose::Bool = false,
        # Kept at `true`, matching the historical behavior of this constructor,
        # but note that it disagrees with `OptimaOptions()`, whose own default is
        # `false`. The two are genuinely different regimes and neither is right
        # everywhere: on an aqueous-only system the analytic `1/nᵢ` (`false`)
        # gives water at `[H⁺]/[OH⁻] = 1.000003`, while the finite-difference
        # diagonal (`true`) gives 3.78; on a mixed solid/aqueous system the
        # verdict reverses. Building the same optimizer two ways must at least
        # not silently pick opposite regimes, so the disagreement is documented
        # here and on `OptimaOptions` until the underlying conditioning issue is
        # resolved.
        use_fd_hessian::Bool = true,
        nullspace_step::Bool = true,
    )
    opts = OptimaOptions(;
        tol, max_iter, warm_start,
        barrier_init, barrier_min, barrier_decay, barrier_eps_factor,
        barrier_stall_iters,
        ls_alpha, ls_beta, ls_max_iter, verbose, use_fd_hessian,
        nullspace_step,
    )
    return OptimaOptimizer(opts, Ref{Union{Nothing, OptimaResult}}(nothing))
end

OptimaOptimizer(opts::OptimaOptions) = OptimaOptimizer(opts, Ref{Union{Nothing, OptimaResult}}(nothing))

"""
    reset_cache!(alg::OptimaOptimizer)

Clear the warm-start cache. Call this when the chemical system changes
(new set of species, different A matrix).
"""
function reset_cache!(alg::OptimaOptimizer)
    alg._cache[] = nothing
    return alg
end

# ── SciMLBase.solve dispatch ──────────────────────────────────────────────────

"""
    SciMLBase.solve(opt_prob, alg::OptimaOptimizer; kwargs...) -> OptimizationSolution

Convert a SciML `OptimizationProblem` to an internal `OptimaProblem` and solve
with the OptimaSolver primal-dual method.

The `OptimizationProblem` is expected to carry:
- `f.f`:       objective `(u, p) -> scalar`
- `f.grad`:    in-place gradient `(g, u, p) -> nothing`  (or `nothing`)
- `prob.cons`: equality constraints `(res, u, p) -> nothing`  (A u = b encoded as residual)
- `prob.lcons`, `prob.ucons`: lower/upper constraint bounds (should be equal for equality)
- `prob.lb`:   lower bounds on u
- `prob.u0`:   initial guess
- `prob.p`:    parameter tuple

# Gradient fallback
If `f.grad` is `nothing`, a ForwardDiff gradient is constructed automatically.
"""
function SciMLBase.solve(
        opt_prob::SciMLBase.OptimizationProblem,
        alg::OptimaOptimizer;
        kwargs...,
    )
    # ── Extract components ───────────────────────────────────────────────────
    f_obj = opt_prob.f.f
    p = opt_prob.p
    u0 = opt_prob.u0
    T = eltype(u0)
    ns = length(u0)

    lb = opt_prob.lb !== nothing ? opt_prob.lb : fill(T(1.0e-16), ns)
    ub = opt_prob.ub !== nothing ? opt_prob.ub : fill(T(Inf), ns)

    # ── Build gradient function ──────────────────────────────────────────────
    g! = if opt_prob.f.grad !== nothing
        opt_prob.f.grad
    else
        # ForwardDiff fallback
        (grad, u, par) -> ForwardDiff.gradient!(grad, v -> f_obj(v, par), u)
    end

    # ── Extract linear constraints A n = b from OptimizationProblem ─────────
    # ChemistryLab encodes mass conservation as equality cons with lcons=ucons=0
    # (res = A*u - b, so b is implicit). We extract A and b via finite differences
    # on the constraint function evaluated at u0.
    #
    # If the problem was built with explicit A and b stored in p (Optima-native
    # usage), extract them directly.
    A, b = _extract_constraints(opt_prob, u0, p)

    # ── Warm-start: determine starting point ─────────────────────────────────
    prev = alg.options.warm_start ? alg._cache[] : nothing

    # The cache must never override a starting point the caller actually chose.
    #
    # `_cache` holds the previous solution of whatever problem this algorithm
    # object last solved — not necessarily this one. Reusing it unconditionally
    # means an explicit `u0` is silently discarded, and the consequences are not
    # academic: a kinetics run re-speciating at every accepted step leaves the
    # 28-day composition in the cache, and a later replay of the SAME trajectory
    # through the same algorithm object then starts every solve from the end
    # state. On an ordinary Portland cement that returned pH 14.2 with 0.31 mol
    # of ettringite and no AFm, where honoring the caller's guess gives pH 12.58
    # with the sulfate entirely in AFm. It also quietly defeated the caller's own
    # warm-start logic during the run itself.
    #
    # So the cache is now what it was meant to be: a convenience for repeated
    # solves where the caller has nothing better to offer. It is used only when
    # the problem has the same size and `u0` carries no interior information —
    # i.e. every variable still sits at its lower bound.
    if !isnothing(prev)
        same_size = length(prev.n) == length(u0)
        caller_supplied_guess = any(u0 .> lb .* 100)
        (same_size && !caller_supplied_guess) || (prev = nothing)
    end

    if isnothing(prev)
        # Cold start: use opt_prob.u0 as base, but lift absent species.
        # ChemistryLab leaves un-set species at the lower bound lb ≈ 1e-16.
        # Starting there puts them exactly on the log-barrier boundary, making
        # the barrier gradient billions of times larger than the chemical
        # gradient.  The solver can only advance them by ≈ 1 ε per Newton step
        # → 10⁵+ iterations needed.  Ipopt avoids this via "feasibility
        # restoration"; we approximate it by distributing the element mass.
        u_start = _lift_cold_start(copy(u0), A, b, lb)
    else
        # Warm start: reuse present species from the previous solution, but lift
        # absent ones.  A species that was near-zero at pH 6 (e.g. OH⁻ ≈ 3e-15 mol)
        # would otherwise get scale s = 1e-10 (floor).  Its Schur complement
        # contribution A[:,k]² × s[k]² / h[k] is then ~10⁻²³ times smaller than
        # major species, effectively excluding it from the Newton step.  Near an
        # equivalence point where such a species must become abundant, this causes
        # O(10⁸) iterations (or non-convergence).  _lift_cold_start only touches
        # species with u[k] ≤ lb[k]×100 (truly absent), leaving present species at
        # their warm-start values and preserving the warm-start benefit.
        u_start = _lift_cold_start(copy(prev.n), A, b, lb)
    end
    y_start = isnothing(prev) ? nothing : prev.y

    # ── Variable scaling ─────────────────────────────────────────────────────
    # Ipopt scales each variable by its initial value so all normalized
    # variables are O(1) at the starting point.  This is critical when
    # concentrations span many orders of magnitude (e.g. a titration with
    # pH 1–13 where [H⁺] varies over 12 decades).  Without scaling the
    # Schur complement A H⁻¹ Aᵀ is dominated by the largest entries and the
    # Newton step for minority species becomes numerically negligible.
    #
    # Mathematics: let ñ = n / s (component-wise), Ã = A * diag(s).
    #   • Constraint: Ã ñ = b  ⟺  A n = b   (unchanged b)
    #   • KKT dual:   ỹ = y                  (invariant under column scaling)
    #   • Gradient:   ∂G̃/∂ñᵢ = sᵢ * ∂G/∂nᵢ
    #   • Hessian:    ∂²G̃/∂ñᵢ² = sᵢ² * ∂²G/∂nᵢ²
    s = max.(abs.(u_start), T(1.0e-10))   # scale = starting value (floor 1e-10)
    inv_s = one(T) ./ s

    A_s = A .* s'                         # A * diag(s)
    lb_s = lb .* inv_s
    ub_s = ub .* inv_s
    u0_s = u_start .* inv_s              # scaled start (≈ 1 component-wise)

    # Scaled closures — unscale internally so f/g! see original n = s ⊙ ñ
    f_s = (ũ, par) -> f_obj(s .* ũ, par)
    g_s! = (grad, ũ, par) -> begin
        g!(grad, s .* ũ, par)
        grad .*= s                        # chain rule: ∂G̃/∂ũᵢ = sᵢ * ∂G/∂nᵢ
    end

    # ── Build OptimaProblem (in scaled space) ────────────────────────────────
    prob_s = OptimaProblem(A_s, b, f_s, g_s!; lb = lb_s, ub = ub_s, p = p)

    # ── Solve ────────────────────────────────────────────────────────────────
    result_s = solve(prob_s, alg.options; u0 = u0_s, y0 = y_start)

    # ── Unscale result ───────────────────────────────────────────────────────
    # n = s ⊙ ñ;  y is invariant under column scaling (see above)
    n_out = result_s.n .* s
    result = OptimaResult{T}(
        n_out,
        result_s.y,
        result_s.iterations,
        result_s.converged,
        result_s.error_opt,
        result_s.error_feas,
        result_s.error_feas_abs,
    )

    # Cache for next call — only cache converged solutions; a non-converged
    # result would give a bad warm-start that cascades into subsequent failures.
    # When the cache is not updated, the next call falls back to opt_prob.u0.
    if result.converged
        alg._cache[] = result
    end

    # ── Pack into SciML solution ─────────────────────────────────────────────
    retcode = result.converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters
    cache = SciMLBase.DefaultOptimizationCache(opt_prob.f, opt_prob.p)
    return SciMLBase.build_solution(
        cache, alg, result.n, f_obj(result.n, p);
        retcode = retcode,
        original = result,
    )
end

# ── Cold-start lifting helper ─────────────────────────────────────────────────

"""
    _lift_cold_start(u, A, b, lb) -> u

Lift species that start at their lower bound to a non-trivial interior point.

When a species is not set by the caller (e.g. `Ace⁻` before any dissociation),
it defaults to `lb ≈ 1e-16`.  Starting there pins it to the log-barrier boundary:
  `−μ/(n − lb) ≈ −μ/ε → −∞`
This makes the barrier gradient billions of times larger than the chemical
gradient, so the solver can advance the species by only one `ε` per Newton step.

Fix: for each absent species (`n ≤ 100·lb`) that the element balance allows to
be nonzero (some row `j` has `A[j,i] > 0` and `b[j] > 0`), set it to a rough
element-balance estimate, then project back onto `A n = b`.

Warm-start paths are not affected (converged solutions have all species nonzero).
"""
function _lift_cold_start(u::Vector{T}, A::Matrix{T}, b::Vector{T}, lb::Vector{T}) where {T}
    ns = length(u)
    m = size(A, 1)
    m == 0 && return u   # unconstrained problem

    # ── Element-balance estimate: same logic as _default_initial_n ────────────
    n_def = fill(T(1.0e-3), ns)
    for i in 1:m
        row_sum = sum(abs, @view A[i, :])
        row_sum > zero(T) || continue
        sc = b[i] / row_sum
        for k in 1:ns
            A[i, k] > zero(T) || continue
            n_def[k] = max(n_def[k], sc)
        end
    end

    # ── Lift absent species to n_def × LIFT_FRACTION ─────────────────────────
    # Threshold: 100·lb + 10·ε separates "not set by caller" (n ≈ lb + ε)
    # from "explicitly set to a small but meaningful value" (n ≫ lb).
    #
    # Why LIFT_FRACTION = 1e-3 and no explicit re-projection:
    #   • The lifted value n_def × 1e-3 is the scale s used later, so ñ ≈ 1 at
    #     the starting point — the species is well inside the barrier.
    #   • The infeasibility added is only n_def × 1e-3 per species, typically
    #     ≪ 1 % of the element budget.  The solver's own _initialise_feasible!
    #     corrects it with a negligible adjustment to present species.
    #   • Projecting to full n_def risks over-consuming present species (e.g.
    #     clinker reactants) when many products are absent simultaneously.
    LIFT_FRACTION = T(1.0e-3)
    for k in 1:ns
        u[k] <= lb[k] * T(100) + T(10) * eps(T) || continue
        # Only lift if the element balance allows this species to be nonzero.
        # E.g. Na⁺ at V=0 has b[Na⁺]=0 → must remain zero.
        can_be_present = any(i -> A[i, k] > zero(T) && b[i] > T(1.0e-12), 1:m)
        can_be_present || continue
        u[k] = n_def[k] * LIFT_FRACTION
    end

    # ── Re-clamp ──────────────────────────────────────────────────────────────
    for k in 1:ns
        u[k] = max(u[k], lb[k] + eps(T))
    end
    return u
end

# ── Constraint extraction helper ─────────────────────────────────────────────

"""
    _extract_constraints(opt_prob, u0, p) -> (A, b)

Extract the linear constraint matrix A and RHS b from a SciML
`OptimizationProblem`.

Two paths:
1. `p` is a NamedTuple with fields `A` and `b` → use directly (Optima-native).
2. Otherwise, finite-difference the constraint function at `u0` to get A, b.
"""
function _extract_constraints(opt_prob, u0::AbstractVector{T}, p) where {T}
    # Path 1: parameters carry A and b explicitly
    if p isa NamedTuple && haskey(p, :A) && haskey(p, :b)
        return convert(Matrix{T}, p.A), convert(Vector{T}, p.b)
    end

    # Path 2: nothing to extract. Returning an empty `A` here used to look like
    # support for unconstrained problems; it is not — the solver needs at least
    # one constraint (see `OptimaProblem`). Say so while the caller can still
    # act on it, rather than a `BoundsError` three calls down.
    if opt_prob.f.cons === nothing
        throw(
            ArgumentError(
                "the OptimizationProblem carries no constraints and `p` holds " *
                    "no `A`/`b`: there is nothing for this solver to work with. " *
                    "Supply `cons` (with `lcons == ucons`), or pass `A` and `b` " *
                    "in `p`."
            )
        )
    end

    # Path 3: finite-difference the constraint function
    # Number of constraints = length(lcons) when available, else call cons once
    ns = length(u0)
    m = (opt_prob.lcons !== nothing && length(opt_prob.lcons) > 0) ?
        length(opt_prob.lcons) : ns
    res0 = zeros(T, m)
    opt_prob.f.cons(res0, u0, p)
    m = length(res0)  # re-confirm

    A = zeros(T, m, ns)
    res1 = similar(res0)
    u_pert = copy(u0)

    # Step size, chosen per column and then checked.
    #
    # `cons` is documented as `A u - b`. For an affine residual the difference
    # quotient is exact whatever the step, so the only error left is
    # cancellation in `res1 - res0` — and a LARGE step is therefore right. The
    # reflex 1e-7 is the worst choice here: against residuals of order one it
    # leaves `A` wrong by ~1e-9 relative, which is a floor the solver cannot get
    # below, so a `tol = 1e-12` run stalls at 3e-12 and reports `MaxIters` on a
    # problem it has in fact solved.
    #
    # Callers do pass nonlinear residuals all the same — a log-parameterized
    # equilibrium sends `A exp(x) - b` — and there a large step is not a
    # derivative at all: on `exp` around x = -2 it comes out 72 % wrong. So
    # affinity is verified rather than assumed. Each column is differenced at
    # two scales; the large answer is kept only when the two agree, otherwise
    # the small one stands, which is the local Jacobian and what this always did.
    a_big = similar(res0)
    a_small = similar(res0)
    rtol = sqrt(eps(T))
    for k in 1:ns
        ε_big = max(one(T), abs(u0[k]))
        for (ε, dest) in ((ε_big, a_big), (ε_big * T(1.0e-7), a_small))
            u_pert[k] = u0[k] + ε
            opt_prob.f.cons(res1, u_pert, p)
            @. dest = (res1 - res0) / ε
        end
        u_pert[k] = u0[k]
        scale = max(maximum(abs, a_big), one(T))
        affine = maximum(abs, a_big .- a_small) <= rtol * scale
        A[:, k] .= affine ? a_big : a_small
    end

    # b = A*u0 - res0 (since cons encodes A*u - b = 0 → b = A*u0 - res0)
    b = A * u0 .- res0

    return A, b
end
