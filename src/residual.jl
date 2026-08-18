# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── residual.jl ────────────────────────────────────────────────────────────────
# KKT residual F(n, y; μ) and Hessian structure for Gibbs minimization.
#
# KKT conditions for the log-barrier problem:
#
#   minimize    f(n) + barrier(n; μ)     barrier = -μ Σ ln(nᵢ - lbᵢ)
#   s.t.        A n = b,   n ≥ lb
#
# Optimality (stationarity):
#   ex = ∇f(n) + Aᵀ y - μ / (n - lb)  = 0        (ns,)
#
# Feasibility:
#   ew = A n - b                         = 0        (m,)
#
# Hessian of the barrier-augmented objective w.r.t. n:
#   H = ∇²f(n) + μ * diag(1 / (n - lb)²)
#
# For Gibbs minimization, ∇²f(n) is provided by the caller (typically
# diagonal + rank-1 for ideal-solution models, or full for HKF).

"""
    KKTResidual{T}

Holds the KKT residual vectors and associated norms for one evaluation.
"""
struct KKTResidual{T <: Real}
    ex::Vector{T}    # optimality residual  (ns,)
    ew::Vector{T}    # feasibility residual (m,)
    error_opt::T     # ‖ex‖∞
    error_feas::T    # ‖ew‖∞
    error::T         # max(error_opt, error_feas)
end

"""
    kkt_residual(prob, n, y, grad_f, μ) -> KKTResidual

Compute the KKT residual at (n, y) with barrier weight μ.

- `grad_f`: gradient ∇f(n) evaluated outside (allows caching)
- `μ`:      log-barrier weight (scalar, T-compatible for AD)
"""
function kkt_residual(
        prob::OptimaProblem{T},
        n::AbstractVector,
        y::AbstractVector,
        grad_f::AbstractVector,
        μ,
    ) where {T}
    Tv = promote_type(eltype(n), eltype(y), eltype(grad_f), typeof(μ))

    s = n .- prob.lb           # slack = n - lb  (all > 0 at interior point)

    # Optimality: ∇f + Aᵀy - μ/s
    ex = grad_f .+ prob.A' * y .- μ ./ s

    # Feasibility: An - b
    ew = prob.A * n .- prob.b

    # Optimality error, measured in COMPLEMENTARITY form.
    #
    # Stationarity of the barrier problem reads (∇f + Aᵀy)ᵢ = μ/sᵢ. Measuring it
    # as `ex = ∇f + Aᵀy − μ/s` — the Newton residual, which is what `ex` above
    # must stay for the step — makes the *error* diverge as a variable
    # approaches its bound: with μ = 1e-4 and s = 1e-15, μ/s = 1e11. On a cement
    # equilibrium `err_opt` started at 4.5e11 and never fell below 2e9, so
    # `is_converged` could not fire at any tolerance, `should_reduce_barrier`
    # never let μ decrease from its initial 1e-4, and every solve ran to
    # `max_iter` and stopped wherever it happened to be — while the feasibility
    # error was meanwhile reaching 1e-14. The old guard against this, excluding
    # variables within `1e-6 × lb` of their bound, could never fire either: with
    # lb = 1e-16 that threshold is 1e-22.
    #
    # Multiplying stationarity through by sᵢ gives the equivalent condition
    # sᵢ(∇f + Aᵀy)ᵢ − μ = 0, which is bounded, vanishes at the optimum, and is
    # the complementarity measure Ipopt reports (Wächter & Biegler 2006, §3.5).
    gL = grad_f .+ prob.A' * y
    err_opt = zero(Tv)
    @inbounds for i in eachindex(gL)
        v = abs(s[i] * gL[i] - μ)
        if v > err_opt
            err_opt = v
        end
    end

    err_feas = isempty(ew) ? zero(Tv) : maximum(abs, ew)

    # Scale the optimality error by the size of the multipliers it is built from
    # — Wächter & Biegler (2006), Eq. (5), the same device Ipopt uses.
    #
    # Unscaled, `ex = ∇f + Aᵀy − μ/s` is compared against `tol`, and in a
    # chemical system its terms are chemical potentials of order 10²–10³ in RT
    # units. Asking their sum to fall below 1e-10 demands thirteen digits of
    # cancellation, which Float64 cannot deliver: on a cement run EVERY solve
    # returned `MaxIters`, including the ones whose element balance closed at
    # 1e-17. A criterion that can never be met is not a safety net — it makes the
    # retcode uninformative and leaves the iterate wherever the budget ran out.
    #
    # The feasibility error is NOT scaled: `An − b` is in moles and already
    # means something absolute.
    return KKTResidual{Tv}(ex, ew, err_opt, err_feas, max(err_opt, err_feas))
end

"""
    hessian_diagonal(prob, n, μ, hess_f_diag) -> Vector

Return the diagonal of the barrier-augmented Hessian:

    H_diag[i] = hess_f_diag[i] + μ / (n[i] - lb[i])²

where `hess_f_diag` is the diagonal of ∇²f(n) (caller-provided).
For a convex Gibbs function with positive curvature, H_diag > 0 always.
"""
function hessian_diagonal(
        prob::OptimaProblem,
        n::AbstractVector,
        μ,
        hess_f_diag::AbstractVector,
    )
    s = n .- prob.lb
    return hess_f_diag .+ μ ./ (s .* s)
end

"""
    gibbs_hessian_diag(n, p) -> Vector

Diagonal of ∇²G for the ideal/dilute Gibbs function G(n) = nᵀ μ(n,p).

For an ideal solution where μᵢ(n) = μᵢ⁰(T,P)/RT + ln(aᵢ(n)):
- Aqueous solvent (mole fraction): ∂²G/∂nᵢ² = 1/nᵢ - 1/n_aq + 1/n_aq (approximately 1/nᵢ)
- Aqueous solutes (molality): ∂²G/∂nᵢ² ≈ 1/nᵢ
- Pure solids/gases: ∂²G/∂nᵢ² = 0 (or small positive for regularization)

For the general case we use finite-difference or AD. Here we provide the
ideal approximation H_diag[i] = 1/nᵢ as a sensible default that is always
positive definite.

**When using AD via ForwardDiff**: do not call this function; instead pass
`hess_f_diag` computed analytically or via forward-mode on the gradient.
"""
function gibbs_hessian_diag(n::AbstractVector{T}, ε::T = T(1.0e-16)) where {T}
    return one(T) ./ max.(n, ε)
end

"""
    eval_gradient!(grad, prob, n)

In-place gradient evaluation: `grad .= ∇f(n, prob.p)`.
Calls `prob.g!(grad, n, prob.p)`.
"""
function eval_gradient!(grad::AbstractVector, prob::OptimaProblem, n::AbstractVector)
    prob.g!(grad, n, prob.p)
    return grad
end
