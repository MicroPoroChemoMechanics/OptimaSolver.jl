# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── sensitivity.jl ─────────────────────────────────────────────────────────────
# Sensitivity analysis: ∂n*/∂c by implicit differentiation of the KKT system.
#
# At the solution u* = (n*, y*) the KKT conditions are:
#
#   F(u*, c) = 0    with c = (b, μ⁰/RT)
#
# By the implicit function theorem:
#
#   ∂F/∂u · ∂u*/∂c + ∂F/∂c = 0
#   ⟹  ∂u*/∂c = -J⁻¹ ∂F/∂c
#
# where J = ∂F/∂u is the KKT Jacobian (already factorised during the last
# Newton step — we reconstruct it here at marginal cost).
#
# For the two sensitivity parameters of interest:
#
#   ∂F/∂b  = [ 0; -I ]    (only feasibility residual changes)
#   ∂F/∂μ⁰ = [ Aᵀ; 0 ]   (gradient of Gibbs via Δ(μ⁰/RT))
#
# Both right-hand sides are cheap (m-column matrices) → m back-substitutions.

"""
    SensitivityResult{T}

Sensitivity of the equilibrium composition n* with respect to:
- `∂n_∂b`:   ∂n*/∂b  (ns × m) — response to changes in mass-balance RHS
- `∂n_∂μ0`:  ∂n*/∂(μ⁰/RT)  (ns × ns) — response to standard potentials

These Jacobians can be used directly as the Jacobian of the RHS in an ODE
coupling kinetics (rates→ b) and thermodynamics (T-dependent μ⁰ → potentials).
"""
struct SensitivityResult{T <: Real}
    ∂n_∂b::Matrix{T}     # (ns × m)
    ∂n_∂μ0::Matrix{T}    # (ns × ns)
end

"""
    sensitivity(prob, n, y, h; μ) -> SensitivityResult

Compute the sensitivity matrices ∂n*/∂b and ∂n*/∂(μ⁰/RT) at the converged
point (n, y) with Hessian diagonal h.

The KKT Jacobian is:

    J = [ H   Aᵀ ]
        [ A    0  ]

We solve J [dn/dc; dy/dc] = -dF/dc for each perturbation direction.

Using the Schur complement (same decomposition as the Newton step):

    S dy = rhs_dual         S = A H⁻¹ Aᵀ
    dn   = -H⁻¹ (ex_rhs + Aᵀ dy)

# Arguments
- `prob`: `OptimaProblem`
- `n`:    converged primal iterate (ns,)
- `y`:    converged dual iterate (m,)
- `h`:    Hessian diagonal at (n, y) (ns,)
- `μ`:    barrier parameter at convergence
"""
function sensitivity(
        prob::OptimaProblem{T},
        n::AbstractVector,
        y::AbstractVector,
        h::AbstractVector,
        μ,
    ) where {T}
    m = prob.m
    A = prob.A

    Tv = promote_type(eltype(n), eltype(y), eltype(h), typeof(μ))

    # ── Schur complement S = A H⁻¹ Aᵀ ────────────────────────────────────────
    AoverH = A ./ h'    # m × ns
    S = AoverH * A'     # m × m

    S_lu = LinearAlgebra.lu(S)

    # ── ∂n*/∂b ────────────────────────────────────────────────────────────────
    # dF/db: ∂ex/∂b = 0, ∂ew/∂b_j = -e_j  (since ew = An - b)
    # IFT: J [∂n/∂b_j; ∂y/∂b_j] = -∂F/∂b_j = [0; e_j]
    # Primal block:  H ∂n/∂b_j = -Aᵀ ∂y/∂b_j  → ∂n/∂b_j = -H⁻¹ Aᵀ ∂y/∂b_j
    # Dual block:    A ∂n/∂b_j = e_j
    # Substituting:  -S ∂y/∂b_j = e_j  → S ∂y/∂b_j = -e_j
    # Then:          ∂n/∂b_j = -H⁻¹ Aᵀ ∂y/∂b_j = H⁻¹ Aᵀ S⁻¹ e_j
    # ∂y/∂b : solve all m RHS simultaneously (BLAS TRSM)
    ∂y_∂b = S_lu \ Matrix{Tv}(-I, m, m)  # m × m : each column = ∂y/∂b_j = -S⁻¹ e_j
    # ∂n/∂b = -H⁻¹ Aᵀ ∂y/∂b  (BLAS GEMM)
    ∂n_∂b = -(A' * ∂y_∂b) ./ h           # ns × m

    # ── ∂n*/∂(μ⁰/RT) ─────────────────────────────────────────────────────────
    # Perturbing μ⁰ₖ/RT shifts ∇f by eₖ (one unit in direction k).
    # dF/d(μ⁰ₖ) = [ eₖ; 0 ]  (primal rhs = eₖ, dual rhs = 0)
    # Schur: S DY = -(A H⁻¹)   [m × ns, all ns RHS at once]
    #        dn_k = -H⁻¹ (eₖ + Aᵀ DY[:,k])
    #
    # Vectorised form (single BLAS TRSM + GEMM):
    #   RHS_μ0 = -(A ./ h')           m × ns
    #   DY_μ0  = S⁻¹ RHS_μ0           m × ns
    #   ∂n_∂μ0 = -(A' * DY_μ0) ./ h   ns × ns
    #   diagonal correction: subtract δ_{lk}/h[l]
    RHS_μ0 = Matrix{Tv}(-(A ./ h'))             # m × ns
    DY_μ0  = S_lu \ RHS_μ0                      # m × ns  (BLAS TRSM)
    ∂n_∂μ0 = -(A' * DY_μ0) ./ h                # ns × ns (BLAS GEMM)
    ∂n_∂μ0[LinearAlgebra.diagind(∂n_∂μ0)] .-= one(Tv) ./ h  # δ_{lk}/h[l] term

    return SensitivityResult{Tv}(∂n_∂b, ∂n_∂μ0)
end
