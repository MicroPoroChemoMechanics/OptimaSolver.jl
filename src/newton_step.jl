# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── newton_step.jl ─────────────────────────────────────────────────────────────
# Newton step computation for the KKT system.
#
# The KKT system to solve is:
#
#   [ H    Aᵀ ] [ dn ]   [ -ex ]
#   [ A    0  ] [ dy ] = [ -ew ]
#
# where H = diag(h) is the barrier-augmented Hessian diagonal.
#
# Exploiting the diagonal structure of H via Schur complement:
#
#   S dy = ew - A H⁻¹ ex        S = A H⁻¹ Aᵀ   (m × m)
#   dn   = -H⁻¹ (ex + Aᵀ dy)
#
# Derivation: substituting dn = -H⁻¹(ex + Aᵀ dy) into A dn = -ew gives
#   -A H⁻¹ ex - S dy = -ew  →  S dy = ew - A H⁻¹ ex
#
# S is dense (m × m) but m is small (number of elements, typically ≤ 15).
# LU of S costs O(m³) — negligible. This avoids the full (ns+m)³ solve.

"""
    NewtonStep{T}

Workspace for the Schur-complement Newton solver. All matrices and vectors
are pre-allocated once and reused across Newton iterations.

Fields:
- `S`:      Schur complement `A * diag(1/h) * A'`  (m × m)
- `rhs`:    RHS `ew - A * diag(1/h) * ex` for the Schur system (m,)
- `dn`:     primal Newton step δn (ns,)
- `dy`:     dual Newton step δy (m,)
- `d`:      equilibration diagonal `sqrt.(diag(S))` (m,)
- `AoverH`: buffer for `A ./ h'` (m × ns), shared between the Schur build
            (`S = AoverH * A'`) and the RHS computation (`rhs = ew - AoverH * ex`)
"""
mutable struct NewtonStep{T <: Real}
    S::Matrix{T}          # Schur complement A H⁻¹ Aᵀ (m × m)
    rhs::Vector{T}        # RHS for Schur system (m,)
    dn::Vector{T}         # primal step (ns,)
    dy::Vector{T}         # dual step (m,)
    d::Vector{T}          # equilibration diagonal sqrt(diag(S)) (m,)
    AoverH::Matrix{T}     # A ./ h'  (m × ns) — reusable buffer
end

function NewtonStep(ns::Int, m::Int, T::Type = Float64)
    return NewtonStep{T}(
        zeros(T, m, m),
        zeros(T, m),
        zeros(T, ns),
        zeros(T, m),
        zeros(T, m),
        zeros(T, m, ns),
    )
end

"""
    compute_step!(ws, can, h, ex, ew) -> (dn, dy)

Compute the Newton step (dn, dy) by Schur complement elimination.

# Arguments
- `ws`:  `NewtonStep` workspace (mutated in-place)
- `can`: `Canonicalizer` (provides A)
- `h`:   Hessian diagonal (ns,), all positive
- `ex`:  optimality residual (ns,)
- `ew`:  feasibility residual (m,)

# Returns
`(dn, dy)` — views into the workspace vectors.
"""
function compute_step!(
        ws::NewtonStep,
        can::Canonicalizer,
        h::AbstractVector,
        ex::AbstractVector,
        ew::AbstractVector,
    )
    A = can.A
    m = size(A, 1)

    # Schur complement: S = A H⁻¹ Aᵀ  (BLAS GEMM)
    ws.AoverH .= A ./ h'          # m × ns : AoverH[i,k] = A[i,k] / h[k]
    mul!(ws.S, ws.AoverH, A')     # S = AoverH * A'

    # RHS: ew - A H⁻¹ ex  (BLAS GEMV)
    mul!(ws.rhs, ws.AoverH, ex)   # rhs = AoverH * ex
    ws.rhs .= ew .- ws.rhs        # rhs = ew - A H⁻¹ ex

    # Solve S dy = rhs  (LU factorisation, m × m).
    #
    # Tikhonov regularisation: add δ·I where δ = diag_max × 1e-14.
    # Without this, S is nearly singular when some conservation rows involve
    # only absent species (e.g. Na⁺ row at V=0 in a titration: all sodium
    # species are absent, so S[Na⁺,Na⁺] ≈ (scale)²/h ≈ 10⁻²⁷ → zero pivot).
    # δ ≪ well-conditioned diagonals, so it does not affect accurate rows.
    T_s = eltype(ws.S)
    diag_max = one(T_s)
    @inbounds for i in 1:m
        diag_max = max(diag_max, ws.S[i, i])
    end
    δ_reg = diag_max * T_s(1.0e-14)
    @inbounds for i in 1:m
        ws.S[i, i] += δ_reg
    end
    # Equilibration scaling: scale row/col i by 1/sqrt(S[i,i]) so all
    # diagonal entries become 1.  This reduces the condition number from
    # O(1e7) near equivalence points (where, e.g., S[H+,H+]≈2e-7 while
    # S[H2O@,H2O@]≈11) to O(1), allowing tol=1e-12 to be reached.
    # S is approximately diagonal because off-diagonal terms are products
    # of A[:,i]·A[:,j] weighted by small 1/h[k], which are negligible when
    # species span many orders of magnitude.
    # Mathematics: let D = diag(sqrt(diag(S))), solve
    #   (D⁻¹ S D⁻¹)(D dy) = D⁻¹ rhs  →  unscale  dy = D⁻¹ dy'.
    @inbounds for i in 1:m
        ws.d[i] = sqrt(ws.S[i, i])
    end
    @inbounds for i in 1:m
        ws.rhs[i] /= ws.d[i]
        for j in 1:m
            ws.S[i, j] /= ws.d[i] * ws.d[j]
        end
    end
    S_lu = LinearAlgebra.lu!(ws.S)
    ws.dy .= S_lu \ ws.rhs
    ws.dy ./= ws.d    # unscale: dy = D⁻¹ dy'

    # Recover dn = -H⁻¹ (ex + Aᵀ dy)  (BLAS GEMV)
    mul!(ws.dn, A', ws.dy)          # dn = A' * dy
    ws.dn .= -(ex .+ ws.dn) ./ h   # dn = -(ex + A' dy) / h

    return ws.dn, ws.dy
end

"""
    clamp_step(n, lb, dn; τ=0.995)

Scale the primal step so that n + α dn stays strictly above lb,
using the fraction-to-boundary rule with safety factor τ ∈ (0,1).

Returns α ∈ (0, 1].
"""
function clamp_step(
        n::AbstractVector,
        lb::AbstractVector,
        dn::AbstractVector;
        τ::Float64 = 0.995,
    )
    T = promote_type(eltype(n), eltype(dn))
    α = one(T)
    @inbounds for i in eachindex(n)
        if dn[i] < zero(T)
            slack = n[i] - lb[i]
            α_i = -τ * slack / dn[i]
            if α_i < α
                α = α_i
            end
        end
    end
    return α
end
