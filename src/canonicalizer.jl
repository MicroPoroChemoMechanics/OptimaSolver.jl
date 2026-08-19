# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── canonicalizer.jl ───────────────────────────────────────────────────────────
# Transforms A (m × ns) into canonical form [I S] via QR with column pivoting.
# Exposes the LU factorization of the basic block for reuse across Newton steps.

"""
    Canonicalizer{T}

Decomposes the conservation matrix A (m × ns) into canonical form:

    A Q = [B  N]    with B = basic block (m × m, full rank)

where Q is a column permutation such that the first m columns are linearly
independent. The LU factorization of B is cached and reused for every
Newton step without re-factorization.

# Fields
- `A`:    original conservation matrix (m × ns)
- `Q`:    column permutation (pivot indices into 1:ns)
- `Qinv`: inverse permutation
- `jb`:   basic variable indices (length m)
- `jn`:   non-basic variable indices (length ns-m)
- `B`:    basic block A[:, jb] (m × m)
- `BLU`:  LU factorization of B (used for Newton and sensitivity)
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

    # The basis is chosen on a COLUMN-EQUILIBRATED copy, and this is not a
    # refinement — without it the factorization is wrong on exactly the problems
    # it is needed for.
    #
    # `rank(A * diag(s)) = rank(A)` for any positive `s`, so scaling cannot
    # change which columns are independent. The pivoted-QR rank test, however,
    # compares each pivot to the LARGEST one, and the caller scales columns by
    # the starting value of each variable. Warm-starting from a converged
    # equilibrium makes that span some ten orders of magnitude — a trace ion sits
    # at its bound near 1e-16 while the solvent is O(1) — and the smallest honest
    # pivot then falls below `tol` times the largest. The rank came out one short,
    # `B = A[:, jb]` was built with m-1 columns, and the LU solve failed with a
    # "matrix is not square" deep inside LAPACK.
    #
    # Equilibrating first makes the test scale-invariant, which is what it was
    # always meant to be.
    colnorm = [LinearAlgebra.norm(@view A[:, j]) for j in 1:ns]
    Aeq = similar(Matrix{T}(A))
    @inbounds for j in 1:ns
        c = colnorm[j] > 0 ? colnorm[j] : one(T)
        Aeq[:, j] .= @view(A[:, j]) ./ c
    end

    F = LinearAlgebra.qr(Aeq, LinearAlgebra.ColumnNorm())
    Q = F.p
    diag_R = abs.(diag(F.R))
    rank_A = count(d -> d > tol * diag_R[1], diag_R)
    rank_A = max(rank_A, 1)

    # A genuinely rank-deficient A means a conservation law that is a combination
    # of the others. It is not an error in itself — the primal solution is
    # unaffected and the multipliers are simply not unique — but every downstream
    # step here assumes a square basic block, so say so plainly instead of
    # failing later inside a factorization.
    rank_A < m && throw(
        ArgumentError(
            "the conservation matrix has rank $rank_A for $m rows: $(m - rank_A) " *
                "constraint(s) are linear combinations of the others. Remove the " *
                "redundant rows (and the matching entries of b) before solving.",
        ),
    )

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

Re-compute the LU factorization of B in-place. Call this if A changes
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
