# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

using LinearAlgebra

# ── The linear program over pure phases ──────────────────────────────────────
#
#     minimize  gᵀx   subject to   A x = b,   x ≥ 0
#
# is the Gibbs energy with every species treated as a pure phase: the mixing
# terms dropped, `g` the standard chemical potentials in units of RT. Its answer
# serves two purposes.
#
# A proof. When the program is infeasible, a vector `z` with `Aᵀz ≥ 0` and
# `bᵀz < 0` exists (Farkas's lemma), and it names the components that no
# nonnegative combination of the species can supply. "This budget is
# impossible" is then a statement about `b`, not a failure of an iteration.
#
# A start. At the optimum, the basic columns form an assemblage of at most `m`
# species, the multipliers `y` of `A x = b` satisfy `A_Bᵀ y = −g_B` (the sign of
# `u = −Aᵀy` in the dual Newton), and the reduced costs `d = g + Aᵀy ≥ 0` are
# minus the saturation indices of every species taken as a pure phase: zero for
# the basic ones, positive for the undersaturated ones.
#
# Every status returned is verified on the original data after the tableau has
# finished, since a dense tableau accumulates rounding: `:optimal` requires the
# balance, the sign of `x` and of the reduced costs; `:infeasible` requires the
# Farkas inequalities. What cannot be verified is reported `:undecided`.
#
# The arithmetic is generic. The pivots are chosen on values, so a `Dual` input
# gives the vertex, the multipliers and the reduced costs with their derivatives
# for the basis found — which is what they are, since a vertex is a smooth
# function of `b` as long as its basis does not change. A symbolic input cannot
# work: choosing a pivot needs a comparison between numbers.

"""
    LPStart{T}

The answer of [`lp_start`](@ref).

  - `status`: `:optimal`, `:infeasible`, `:undecided` or `:iteration_limit`.
  - `x`: the vertex (zeros unless `:optimal`).
  - `y`: the multipliers of `A x = b`, with `A_Bᵀ y = −g_B` on the basis.
  - `basis`: the basic column of each row, `0` for a redundant row.
  - `reduced_costs`: `g + Aᵀy`, nonnegative at the optimum; `NaN` for a column
    excluded with `dead`.
  - `redundant_rows`: rows that are linear combinations of the others.
  - `farkas`: when `:infeasible`, a `z` with `Aᵀz ≥ 0` and `bᵀz < 0`; empty
    otherwise.
  - `balance`: `‖A x − b‖∞ / max(1, ‖b‖∞)`.
  - `iterations`: simplex pivots, both phases.

`y` is unique only where the basis is; on a redundant row it is the
minimum-norm solution.
"""
struct LPStart{T <: Real}
    status::Symbol
    x::Vector{T}
    y::Vector{T}
    basis::Vector{Int}
    reduced_costs::Vector{T}
    redundant_rows::Vector{Int}
    farkas::Vector{T}
    balance::T
    iterations::Int
end

# One pivot on (r, c), column by column so that the inner loop runs down a
# column of the column-major tableau. `col` is a buffer for the pivot column.
function _lp_pivot!(T::AbstractMatrix, basis, r, c, col::AbstractVector)
    p = T[r, c]
    @inbounds for j in axes(T, 2)
        T[r, j] /= p
    end
    @inbounds for i in axes(T, 1)
        col[i] = T[i, c]
    end
    @inbounds for j in axes(T, 2)
        trj = T[r, j]
        iszero(trj) && continue
        for i in axes(T, 1)
            i == r && continue
            T[i, j] -= col[i] * trj
        end
    end
    basis[r] = c
    return nothing
end

# Bland's rule: the lowest-index column of negative reduced cost enters, and on
# a tie of the ratio test the lowest basic index leaves. It cannot cycle, which
# matters on a chemical polytope, where most species are absent at the vertex
# and degeneracy is the rule.
function _lp_run!(T::AbstractMatrix, basis, ncols, itmax, tol, col)
    nrows = size(T, 1) - 1
    for it in 1:itmax
        c = 0
        @inbounds for j in 1:ncols
            if T[end, j] < -tol
                c = j
                break
            end
        end
        c == 0 && return (:optimal, it - 1)
        r, best = 0, Inf
        @inbounds for i in 1:nrows
            T[i, c] > tol || continue
            ratio = T[i, end] / T[i, c]
            if ratio < best - tol || (abs(ratio - best) <= tol && r != 0 && basis[i] < basis[r])
                r, best = i, ratio
            end
        end
        r == 0 && return (:unbounded, it - 1)
        _lp_pivot!(T, basis, r, c, col)
    end
    return (:iteration_limit, itmax)
end

_inf_norm(v) = maximum(abs, v; init = zero(eltype(v)))

"""
    lp_start(A, g, b; dead = (), tol = 1e-9, maxit = 0) -> LPStart
    lp_start(prob::DualNewtonProblem, b; tol = 1e-9, maxit = 0) -> LPStart

Solve `minimize gᵀx subject to A x = b, x ≥ 0` and return the vertex, the
multipliers, the basis and the reduced costs, or a certificate of
infeasibility; see [`LPStart`](@ref).

Columns in `dead` are excluded (held at zero). Rows are equilibrated before the
tableau is built, every tolerance is relative to the scale of its data, and the
status is verified on the original data at the end. The element type is that of
the inputs, promoted: `ForwardDiff.Dual` numbers pass through, with the pivots
chosen on their values.

The second form takes `A` and the standard potentials from `prob`, and holds at
zero the species of the rows whose budget forces them to vanish
([`degenerate_components`](@ref)); the multipliers of those rows are set to
`DEGENERATE_POTENTIAL`, as the dual Newton sets them.

# What it is worth as a start, measured

On two cold cement pastes (the `cement107` benchmark case, 107 species, and a
CEM I with the CNASH gel of Myers et al.), every time the median of three runs
after a discarded warm-up:

| start | cement107 | CEM I + CNASH |
|:--|--:|--:|
| the caller's certified cascade from the recipe | 4.7 s | 4.2 s |
| `dual_newton_solve` alone, vertex floored at 1e-16 | not converged | not converged |
| alone, absent species raised to `exp(uⱼ − gⱼ)`, first `y` the program's | not converged | not converged |
| alone, the same with the basis as initial active set | not converged | not converged |
| the cascade from the raised vertex | 0.37 s | 0.26 s |

A vertex holds at most `m` species and a log-domain Newton cannot start from
the others at zero; raised to the amounts the multipliers give them, the vertex
is a start the interior-point back ends then take the rest of the way, to the
same composition (1e-12 relative). So what this returns is a state, and nothing
of it is passed to `dual_newton_solve` directly.

A staged start of this kind, a linear program followed by an interior method, is
common in Gibbs-energy minimizers; see Kulik et al. (2013), *Comput. Geosci.*
17, 1–24, doi:10.1007/s10596-012-9310-6.
"""
function lp_start(
        A::AbstractMatrix, g::AbstractVector, b::AbstractVector;
        dead = (), tol::Real = 1.0e-9, maxit::Integer = 0,
    )
    m, n = size(A)
    length(b) == m || throw(DimensionMismatch("`b` has $(length(b)) rows, `A` has $m."))
    length(g) == n || throw(DimensionMismatch("`g` has $(length(g)) entries, `A` has $n columns."))
    F = float(promote_type(eltype(A), eltype(g), eltype(b)))
    Af = convert(Matrix{F}, A)
    gf = convert(Vector{F}, g)
    bf = convert(Vector{F}, b)
    alive = [j for j in 1:n if !(j in dead)]
    itmax = maxit > 0 ? Int(maxit) : 50 * (m + n) + 100
    na = length(alive)

    # Row equilibration. A row with no alive entry keeps a unit scale: if its
    # budget is not zero it is, on its own, a proof of infeasibility.
    scale = [(s = _inf_norm(@view Af[i, alive]); iszero(s) ? one(F) : inv(s)) for i in 1:m]
    As = scale .* Af
    bs = scale .* bf
    bnorm = max(one(F), _inf_norm(bs))
    ptol = 1.0e-12

    # Phase I: [S A_alive  I | S b] with S making the right-hand side
    # nonnegative, the artificials as the starting basis, and the cost row
    # reduced so that the basic columns carry a zero reduced cost.
    sgn = [bs[i] < 0 ? -one(F) : one(F) for i in 1:m]
    T = zeros(F, m + 1, na + m + 1)
    @views T[1:m, 1:na] .= sgn .* As[:, alive]
    for i in 1:m
        T[i, na + i] = one(F)
    end
    @views T[1:m, end] .= sgn .* bs
    @views T[end, (na + 1):(na + m)] .= one(F)
    @views T[end, :] .-= vec(sum(T[1:m, :]; dims = 1))
    basis = collect((na + 1):(na + m))
    col = zeros(F, m + 1)
    st1, it1 = _lp_run!(T, basis, na + m, itmax, ptol, col)
    st1 === :iteration_limit && return _lp_answer(F, :iteration_limit, m, n, it1)

    # The phase-I value is compared with the rounding level of the tableau, not
    # with a fraction of the budget: a trace row of 1e-9 mol that no species can
    # supply, next to a row of 55 mol, is infeasible, and a tolerance relative
    # to the largest budget would call it feasible.
    phase1 = -T[end, end]
    # The phase-I multipliers π solve Bᵀπ = c_B for the final basis; z = −S π
    # then satisfies Aᵀz ≥ 0 and bᵀz < 0 when the program is infeasible.
    z = phase1 > 1.0e-13 * bnorm ? scale .* _farkas(As, sgn, basis, alive, na, m) : F[]
    verdict = _phase1_verdict(phase1, bnorm, Af, bf, z, alive, tol)
    verdict === :infeasible &&
        return LPStart{F}(:infeasible, zeros(F, n), zeros(F, m), zeros(Int, m), fill(F(NaN), n), Int[], z, F(NaN), it1)
    verdict === :undecided && return _lp_answer(F, :undecided, m, n, it1)

    # Drive the artificials out of the basis; a row where no alive column can
    # replace its artificial is redundant.
    redundant = Int[]
    for i in 1:m
        basis[i] > na || continue
        c, best = 0, ptol
        for j in 1:na
            v = abs(T[i, j])
            v > best && ((c, best) = (j, v))
        end
        c == 0 ? push!(redundant, i) : _lp_pivot!(T, basis, i, c, col)
    end

    # Phase II on the alive columns.
    T2 = T[:, vcat(1:na, na + m + 1)]
    @views T2[end, :] .= zero(F)
    @views T2[end, 1:na] .= gf[alive]
    for (i, bi) in pairs(basis)
        bi <= na || continue
        f = T2[end, bi]
        iszero(f) || (@views T2[end, :] .-= f .* T2[i, :])
    end
    st2, it2 = _lp_run!(T2, basis, na, itmax, ptol * max(1.0, _inf_norm(gf)), col)
    st2 === :iteration_limit && return _lp_answer(F, :iteration_limit, m, n, it1 + it2)
    st2 === :unbounded && return _lp_answer(F, :undecided, m, n, it1 + it2)

    # Verification on the original data, from the final basis alone.
    cols = [basis[i] <= na ? alive[basis[i]] : 0 for i in 1:m]
    basic = filter(>(0), cols)
    x = zeros(F, n)
    y = zeros(F, m)
    if !isempty(basic)
        # One thin QR of the basic columns, A_B = Q₁R, gives both answers: x_B
        # from R x_B = Q₁ᵀb, and the minimum-norm y of A_Bᵀ y = −g_B as
        # y = −Q₁ R⁻ᵀ g_B. Householder QR is generic, so a Dual passes through.
        fac = qr(Af[:, basic])
        k = length(basic)
        Q1 = Matrix(fac.Q)[:, 1:k]
        R = UpperTriangular(fac.R[1:k, 1:k])
        x[basic] .= R \ (transpose(Q1) * bf)
        y .= -(Q1 * (transpose(R) \ gf[basic]))
    end
    balance = _inf_norm(Af * x .- bf) / max(one(F), _inf_norm(bf))
    d = gf .+ transpose(Af) * y
    for j in dead
        d[j] = F(NaN)
    end
    gscale = max(one(F), _inf_norm(gf))
    xscale = max(one(F), _inf_norm(x))
    ok = balance <= tol && all(>=(-tol * xscale), x) && all(j -> d[j] >= -tol * gscale, alive)
    return LPStart{F}(ok ? :optimal : :undecided, max.(x, zero(F)), y, cols, d, redundant, F[], balance, it1 + it2)
end

function lp_start(prob::DualNewtonProblem, b::AbstractVector; tol::Real = 1.0e-9, maxit::Integer = 0)
    degenerate = _degenerate_conservation_rows(prob, b)
    dead = _dead_variables(prob.A, degenerate)
    lp = lp_start(prob.A, current_g(prob, prob.q0), b; dead, tol, maxit)
    lp.status === :optimal || return lp
    y = copy(lp.y)
    y[degenerate] .= DEGENERATE_POTENTIAL
    return LPStart(lp.status, lp.x, y, lp.basis, lp.reduced_costs, lp.redundant_rows, lp.farkas, lp.balance, lp.iterations)
end

# What phase I proves. `:infeasible` only on a Farkas vector that verifies on
# the data; `:feasible` when the phase-I value is at the rounding level of the
# tableau, or within the tolerance with no such vector; `:undecided` when the
# value says infeasible and no vector proves it, which is never reported as
# either answer.
function _phase1_verdict(phase1, bnorm, Af, bf, z, alive, tol)
    phase1 <= 1.0e-13 * bnorm && return :feasible
    _is_farkas(Af, bf, z, alive, tol) && return :infeasible
    return phase1 > tol * bnorm ? :undecided : :feasible
end

_lp_answer(::Type{F}, status, m, n, it) where {F} =
    LPStart{F}(status, zeros(F, n), zeros(F, m), zeros(Int, m), fill(F(NaN), n), Int[], F[], F(NaN), it)

# The phase-I multipliers: the columns of [S A_alive I] in the final basis and
# their phase-I costs (0 for a species, 1 for an artificial) give Bᵀπ = c_B.
function _farkas(As::AbstractMatrix{F}, sgn, basis, alive, na, m) where {F}
    B = zeros(F, m, m)
    cB = zeros(F, m)
    for (i, bi) in pairs(basis)
        if bi <= na
            @views B[:, i] .= sgn .* As[:, alive[bi]]
        else
            B[bi - na, i] = one(F)
            cB[i] = one(F)
        end
    end
    π = transpose(B) \ cB
    return -(sgn .* π)
end

function _is_farkas(A, b, z, alive, tol)
    zn = _inf_norm(z)
    zn > 0 || return false
    Aᵀz = transpose(A) * z
    ascale = _inf_norm(A) * zn
    bscale = max(_inf_norm(b) * zn, eps())
    return all(j -> Aᵀz[j] >= -tol * ascale, alive) && dot(b, z) < -1.0e-12 * bscale
end
