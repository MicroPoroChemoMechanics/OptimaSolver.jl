# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── residual.jl ────────────────────────────────────────────────────────────────
# KKT residual F(n, y; μ) and Hessian structure for Gibbs minimisation.
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
# For Gibbs minimisation, ∇²f(n) is provided by the caller (typically
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

    # Optimality error: exclude near-bound species that correctly sit at their
    # lower bound.  For such species (slack ≈ 0, ex ≥ 0), the barrier term
    # μ/slack → ∞ as slack → 0, making |ex| arbitrarily large even though the
    # species is at its optimal position.  Including them in the error norm
    # prevents convergence.  This mirrors C++ Optima ResidualErrors.cpp:
    #   ex(ju).fill(0.0)  — zero the error for lower-unstable variables.
    #
    # Rule: a variable is excluded when
    #   slack[i] ≤ 1e-8 · max_slack   (near its bound)   AND
    #   ex[i]   ≥ 0                    (gradient pushes toward the bound)
    # Interior variables and near-bound variables with ex < 0 (species should
    # come off the bound) are always included.
    max_slack = maximum(s)
    slack_tol = Tv(1.0e-8) * max(one(Tv), max_slack)

    err_opt = zero(Tv)
    @inbounds for i in eachindex(ex)
        if s[i] > slack_tol || ex[i] < zero(Tv)
            v = abs(ex[i])
            if v > err_opt
                err_opt = v
            end
        end
    end

    err_feas = isempty(ew) ? zero(Tv) : maximum(abs, ew)

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
- Pure solids/gases: ∂²G/∂nᵢ² = 0 (or small positive for regularisation)

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
