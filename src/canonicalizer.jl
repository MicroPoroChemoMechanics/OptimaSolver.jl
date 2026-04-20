# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── canonicalizer.jl ───────────────────────────────────────────────────────────
# Transforms A (m × ns) into canonical form [I S] via QR with column pivoting.
# Exposes the LU factorisation of the basic block for reuse across Newton steps.

"""
    Canonicalizer{T}

Decomposes the conservation matrix A (m × ns) into canonical form:

    A Q = [B  N]    with B = basic block (m × m, full rank)

where Q is a column permutation such that the first m columns are linearly
independent. The LU factorisation of B is cached and reused for every
Newton step without re-factorisation.

# Fields
- `A`:    original conservation matrix (m × ns)
- `Q`:    column permutation (pivot indices into 1:ns)
- `Qinv`: inverse permutation
- `jb`:   basic variable indices (length m)
- `jn`:   non-basic variable indices (length ns-m)
- `B`:    basic block A[:, jb] (m × m)
- `BLU`:  LU factorisation of B (used for Newton and sensitivity)
- `R`:    R = B⁻¹ N  (m × (ns-m)), the reduced-cost matrix
- `ns`:   number of species
- `m`:    number of constraints
- `rank_A`: effective rank of A
"""
struct Canonicalizer{T <: Real}
    A::Matrix{T}
    Q::Vector{Int}        # column permutation (pivot first)
    Qinv::Vector{Int}     # inverse permutation
    jb::Vector{Int}       # basic variable indices
    jn::Vector{Int}       # non-basic variable indices
    B::Matrix{T}          # A[:, jb]
    BLU::Any              # lu(B)
    R::Matrix{T}          # B⁻¹ N
    ns::Int
    m::Int
    rank_A::Int
end

"""
    Canonicalizer(A; tol=1e-12)

Build the canonicalizer for conservation matrix `A` (m × ns).

Uses QR with column pivoting to identify m linearly independent columns
(basic variables). Stores LU of the basic block B for O(m²) solves.
"""
function Canonicalizer(A::AbstractMatrix{T}; tol::Float64 = 1.0e-12) where {T <: Real}
    m, ns = size(A)
    @assert m <= ns "A must have more columns than rows (m=$m, ns=$ns)"

    # QR with column pivoting to identify linearly independent columns
    F = LinearAlgebra.qr(A, LinearAlgebra.ColumnNorm())
    Q = F.p                           # column permutation
    R_qr = F.R

    # Determine rank from diagonal of R
    diag_R = abs.(diag(R_qr))
    rank_A = count(d -> d > tol * diag_R[1], diag_R)
    rank_A = max(rank_A, 1)

    jb = sort(Q[1:rank_A])           # basic variable indices (sorted for stability)
    jn = sort(setdiff(1:ns, jb))     # non-basic variable indices

    B = A[:, jb]
    BLU = LinearAlgebra.lu(B)

    # Reduced-cost matrix R = B⁻¹ N
    N = A[:, jn]
    R_mat = BLU \ Matrix{T}(N)

    Qvec = vcat(jb, jn)              # full permutation: basics first, then non-basics
    Qinv = invperm(Qvec)

    return Canonicalizer{T}(
        Matrix{T}(A), Qvec, Qinv, jb, jn, Matrix{T}(B), BLU, R_mat, ns, m, rank_A,
    )
end

"""
    refactorize!(c::Canonicalizer)

Re-compute the LU factorisation of B in-place. Call this if A changes
(e.g. after variable repartitioning). Returns a new `Canonicalizer`.
"""
function refactorize(c::Canonicalizer{T}) where {T}
    BLU = LinearAlgebra.lu(c.B)
    N = c.A[:, c.jn]
    R_mat = BLU \ Matrix{T}(N)
    return Canonicalizer{T}(c.A, c.Q, c.Qinv, c.jb, c.jn, c.B, BLU, R_mat, c.ns, c.m, c.rank_A)
end

"""
    solve_B(c::Canonicalizer, rhs)

Solve B x = rhs using the cached LU factorisation. O(m²).
"""
solve_B(c::Canonicalizer, rhs) = c.BLU \ rhs

"""
    solve_Bt(c::Canonicalizer, rhs)

Solve Bᵀ x = rhs using the cached LU factorisation. O(m²).
"""
solve_Bt(c::Canonicalizer, rhs) = c.BLU' \ rhs

"""
    schur_complement(c::Canonicalizer, h::AbstractVector)

Compute the Schur complement matrix S = A * diag(1/h) * Aᵀ (m × m),
where h is the diagonal of the Hessian (length ns, all positive).

Exploits the canonical form: S = B diag(1/h_b)⁻¹ Bᵀ + N diag(1/h_n)⁻¹ Nᵀ
computed directly as A * Diagonal(1 ./ h) * Aᵀ.
"""
function schur_complement(c::Canonicalizer{T}, h::AbstractVector) where {T}
    # S = A * H⁻¹ * Aᵀ  where H = diag(h)
    # Efficient: S = (A ./ h') * Aᵀ
    AoverH = c.A ./ h'    # m × ns, each column j scaled by 1/h[j]
    return AoverH * c.A'
end
