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

    # Solve S dy = rhs  (LU factorization, m × m).
    #
    # Tikhonov regularization: add δ·I where δ = diag_max × 1e-14.
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
        τ = 0.995,
    )
    T = promote_type(eltype(n), eltype(dn), typeof(τ))
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

"""
    binding_variable(n, lb, dn; τ = 0.995) -> (i, α_i)

Index of the variable that sets the fraction-to-boundary limit, and the limit it
imposes. Diagnostic only.
"""
function binding_variable(n, lb, dn; τ = 0.995)
    T = promote_type(eltype(n), eltype(dn), typeof(τ))
    α = one(T); k = 0
    @inbounds for i in eachindex(n)
        if dn[i] < zero(T)
            α_i = -τ * (n[i] - lb[i]) / dn[i]
            if α_i < α
                α = α_i; k = i
            end
        end
    end
    return k, α
end

# ── Nullspace step ────────────────────────────────────────────────────────────

"""
    compute_step_nullspace!(ws, can, h, ex, ew) -> (dn, dy)

Newton step by the **nullspace** method, which never inverts `H`.

`compute_step!` above forms the Schur complement `S = A H⁻¹ Aᵀ`. That is the
`Rangespace` method of the C++ Optima this package is ported from, and Optima
documents it as suitable only for a Hessian that is diagonal *and* invertible.
It is neither harmless nor hypothetical to ignore the second condition: in a
chemical equilibrium a **pure phase** has unit activity, so `∂²G/∂nᵢ² = 0`
exactly, and `H⁻¹` is then unbounded along that variable. The step degenerates
and the solve stalls essentially at its starting point.

The nullspace method avoids the inverse entirely. With `Z` a basis of `null(A)`,
write `dn = dnₚ + Z dz` where `A dnₚ = −ew`. Since `Zᵀ Aᵀ = 0`, the dual drops
out of the projected stationarity condition and

```
(Zᵀ H Z) dz = −Zᵀ (ex + H dnₚ)
```

is an `(ns − m) × (ns − m)` system in which `H` appears only as a *product*.

The canonicalizer already supplies the basis for free. With `A[:, jb] = B`,
`A[:, jn] = N` and `R = B⁻¹N`, the nullspace basis is `Z[jb, :] = −R`,
`Z[jn, :] = I`, so

```
Zᵀ H Z = Rᵀ diag(h[jb]) R + diag(h[jn]),
```

symmetric and positive definite at any interior point, since `h > 0` there
whatever the curvature — the barrier term `μ/s²` guarantees it.

The dual step follows from the basic rows of `H dn + Aᵀ dy = −ex`, that is
`Bᵀ dy = −ex[jb] − h[jb] ⊙ dn[jb]`.
"""
function compute_step_nullspace!(
        ws::NewtonStep,
        can::Canonicalizer,
        h::AbstractVector,
        ex::AbstractVector,
        ew::AbstractVector,
    )
    T = eltype(ws.dn)
    jb, jn, R = can.jb, can.jn, can.R
    nb, nn = length(jb), length(jn)

    # Particular solution: dnₚ[jn] = 0, dnₚ[jb] = B⁻¹(−ew).
    dnp_b = can.BLU \ collect(-ew)

    # Reduced Hessian Zᵀ H Z = Rᵀ diag(h_b) R + diag(h_n).
    hb = @view h[jb]
    K = Matrix{T}(undef, nn, nn)
    hR = R .* hb                      # diag(h_b) * R
    mul!(K, R', hR)
    @inbounds for j in 1:nn
        K[j, j] += h[jn[j]]
    end

    # RHS: −Zᵀ(ex + H dnₚ). H dnₚ is nonzero only on the basic block.
    r_b = @view ex[jb]
    w_b = r_b .+ hb .* dnp_b
    rhs = Vector{T}(undef, nn)
    mul!(rhs, R', w_b)                # (−R)ᵀ w_b, sign folded below
    @inbounds for j in 1:nn
        rhs[j] = rhs[j] - ex[jn[j]]
    end

    # `Zᵀ H Z` is positive definite at any interior point in exact arithmetic,
    # since `h > 0` there whatever the curvature. It can still lose definiteness
    # numerically when the amounts span ten orders of magnitude, so the
    # factorization is attempted and fallen back on rather than assumed —
    # a `PosDefException` from a default code path is not acceptable.
    Ksym = LinearAlgebra.Symmetric(K)
    chol = LinearAlgebra.cholesky(Ksym; check = false)
    dz = LinearAlgebra.issuccess(chol) ? chol \ rhs :
        LinearAlgebra.bunchkaufman(Ksym; check = false) \ rhs

    # Assemble dn = dnₚ + Z dz.
    fill!(ws.dn, zero(T))
    @inbounds for j in 1:nn
        ws.dn[jn[j]] = dz[j]
    end
    Rdz = R * dz
    @inbounds for i in 1:nb
        ws.dn[jb[i]] = dnp_b[i] - Rdz[i]
    end

    # Dual step from the basic rows: Bᵀ dy = −ex[jb] − h[jb] ⊙ dn[jb].
    rhs_y = Vector{T}(undef, nb)
    @inbounds for i in 1:nb
        rhs_y[i] = -ex[jb[i]] - h[jb[i]] * ws.dn[jb[i]]
    end
    ws.dy .= can.BLU' \ rhs_y

    return ws.dn, ws.dy
end
