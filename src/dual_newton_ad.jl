# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# ── differentiating through the dual Newton ──────────────────────────────────
#
# A budget or a problem that carries dual numbers is solved on its values, and
# the derivatives of the answer come from the implicit-function theorem at that
# answer: with `Φ(z; θ) = 0` the system of the active set (`_implicit_system`),
#
#     Φ_z ż = −Φ_θ θ̇ ,
#
# where `Φ_z` is its Jacobian in the unknowns `z = (v, w)`, from forward mode
# under a tag of its own, and `Φ_θ θ̇` is the dual part of `Φ` evaluated at the
# primal `z` with the dual data. The answer comes back in duals of the caller's
# tag. A caller's duals that are themselves nested are stripped one level at a
# time: the primal solve is again a solve on duals, one level down, so second
# derivatives are exact as well.

# Whether `x` carries a dual number anywhere: itself, an element, a field of a
# tuple.
function _is_dual_data(x)
    x isa ForwardDiff.Dual && return true
    x isa AbstractArray && return eltype(x) <: ForwardDiff.Dual || any(_is_dual_data, x)
    x isa Union{Tuple, NamedTuple} && return any(_is_dual_data, values(x))
    return false
end

# The values of `x`, one level of duals down.
_values(x) = x
_values(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_values(x::AbstractArray{<:ForwardDiff.Dual}) = ForwardDiff.value.(x)
# An array of an abstract number type may mix dual numbers with plain ones: each
# element is taken down, and the result is of their common type.
_values(x::AbstractArray) = (eltype(x) <: Number && isconcretetype(eltype(x))) ? x : map(_values, x)
_values(x::Union{Tuple, NamedTuple}) = map(_values, x)

# The dual type the answer is returned in: that of the data, of the budget, or of
# what the callbacks return at the answer when they capture the duals themselves.
function _dual_type(prob, b, x = nothing, q = prob.q0)
    for v in (b, prob.g, prob.A)
        eltype(v) <: ForwardDiff.Dual && return eltype(v)
    end
    D = _param_dual_type(prob.params)
    D === nothing || return D
    if x !== nothing
        for v in (current_h(prob, x, q), current_g(prob, q), prob.nq == 0 ? Float64[] : prob.cq(x, q, prob.params))
            eltype(v) <: ForwardDiff.Dual && return eltype(v)
        end
    end
    throw(ArgumentError("no dual numbers to differentiate"))
end
_param_dual_type(x) = nothing
_param_dual_type(x::ForwardDiff.Dual) = typeof(x)
_param_dual_type(x::AbstractArray) = eltype(x) <: ForwardDiff.Dual ? eltype(x) : _first_dual(map(_param_dual_type, x))
_param_dual_type(x::Union{Tuple, NamedTuple}) = _first_dual(map(_param_dual_type, values(x)))
_first_dual(xs) = (
    for x in xs
        x === nothing || return x
    end; nothing
)

# The same problem on the values of its data, one level of duals down.
function _primal_problem(prob::DualNewtonProblem)
    A, g, Aq = _values(prob.A), _values(prob.g), _values(prob.Aq)
    q0, qs = _values(prob.q0), _values(prob.qscale)
    T = promote_type(eltype(A), eltype(g), eltype(Aq), eltype(q0), eltype(qs))
    return DualNewtonProblem(
        Matrix{T}(A), Vector{T}(g), prob.h, prob.phases, prob.idx_bounded,
        _values(prob.params), prob.nq, prob.gq, prob.cq, prob.hq, Matrix{T}(Aq),
        prob.always_active, prob.conservation_rows, Vector{T}(q0), Vector{T}(qs),
    )
end

# The same problem on the values of its data, every level of duals down.
function _full_primal_problem(prob::DualNewtonProblem)
    while _is_dual_data((prob.A, prob.g, prob.Aq, prob.q0, prob.qscale, prob.params))
        prob = _primal_problem(prob)
    end
    return prob
end

"""
    _dual_newton_solve_ad(prob, b, x0; opts) -> NamedTuple

`dual_newton_solve` where the budget `b` or the data of `prob` (`A`, `g`,
`params`) carry dual numbers: the answer of the values, with `x`, `y` and `q` in
duals of the caller's tag (`dual_newton_tangent`).
"""
function _dual_newton_solve_ad(prob, b, x0; opts)
    res = dual_newton_solve(_primal_problem(prob), _values(b), _values(x0); opts)
    t = dual_newton_tangent(
        prob, b, res.x;
        y = res.y, q = res.q, active_phases = res.active_phases, active = res.active,
    )
    return merge(res, t)
end

"""
    dual_newton_tangent(prob, b, x; y = nothing, q = nothing,
                        active_phases = nothing, active = nothing,
                        floor = 1e-25, primal = nothing) -> (; x, y, q)

The answer `x` of `prob` on the values of its data, lifted to the dual numbers
that `b` or the data of `prob` (`A`, `g`, `params`) carry: `x`, `y` and `q` are
returned in duals of the caller's tag, their partials the derivatives of the
answer given by the implicit-function theorem with the active set frozen.

`x` (and `y`, `q` when given) is the answer one level of duals down, as
`dual_newton_solve` returns it on the values of the data, whatever route
produced it. What is not given is recovered from `x`:

- the active phases are those holding more than `floor`, and the active bounded
  variables those above it: an interior-point answer leaves an absent pure phase
  at a negligible amount rather than at zero;
- `y` solves the stationarity of the species above `floor` by least squares, as
  `kkt_certificate` does.

The answer must be one: the derivative of a point that is not an equilibrium is
not the derivative of anything. Audit it with `kkt_certificate` first.

A member the inversion does not determine (a model flooring a vanishing
activity, below its floor) is held: its amount has no derivative.

`primal` is the same problem on the values of its data. It is built here from
`A`, `g` and `params`, and must be given when the duals are captured by `h`,
`gq`, `hq` or `cq` themselves (the parameters of an activity model, say), which
nothing here can strip.
"""
function dual_newton_tangent(
        prob::DualNewtonProblem, b::AbstractVector, x::AbstractVector;
        y = nothing, q = nothing, active_phases = nothing, active = nothing,
        floor::Real = 1.0e-25, primal = nothing,
    )
    q = q === nothing ? _values(prob.q0) : q
    D = _dual_type(prob, b, x, q)
    Tg = ForwardDiff.tagtype(D)
    P = ForwardDiff.npartials(D)
    pprob = primal === nothing ? _primal_problem(prob) : primal
    bv = _values(b)
    # The number type of the answer: `Float64`, or the duals of a differentiation
    # enclosing this one, which the whole computation below carries.
    V = promote_type(eltype(x), eltype(q), eltype(pprob.g), eltype(pprob.A))

    m = size(pprob.A, 1)
    nq = pprob.nq
    degenerate = _degenerate_conservation_rows(pprob, bv)
    dead = _dead_variables(pprob.A, degenerate)
    act_ph = active_phases !== nothing ? active_phases : [
            k for (k, ph) in pairs(pprob.phases)
            if ph.always_present || sum(x[i] for i in ph.members) > floor
        ]
    active = active !== nothing ? active : [
            i for i in pprob.idx_bounded if !(i in dead) && (x[i] > floor || i in pprob.always_active)
        ]
    if y === nothing
        ∇f = current_g(pprob, q) .+ current_h(pprob, x, q)
        interior = vcat(
            [i for k in act_ph for i in pprob.phases[k].members if !(i in dead) && x[i] > floor],
            active,
        )
        rows = [k for k in 1:m if !(k in degenerate)]
        y = fill(V(DEGENERATE_POTENTIAL), m)
        y[rows] .= qr(transpose(pprob.A[rows, interior]), ColumnNorm()) \ (-∇f[interior])
    end

    W = [
        V[
            (i in dead) ? W_FLOOR : clamp(log(max(x[i], exp(W_FLOOR))), W_FLOOR, W_CEIL)
                for i in ph.members
        ] for ph in pprob.phases
    ]
    refs = [
        let ph = pprob.phases[k]
            ph.mole_fraction ? sum(x[i] for i in ph.members) : x[ph.members[ph.j_ref]]
        end for k in act_ph
    ]
    v = vcat(log.(refs), y, x[active], q)
    unk = _inner_unknowns(pprob, W, v, act_ph, dead)
    z0 = vcat(v, V[W[k][j] for (_, k, j, _, _) in unk])

    # `Φ_z` by forward mode under a tag of its own; `Φ_θ θ̇`, the dual part of `Φ`
    # at the answer with the caller's data.
    Φz = ForwardDiff.jacobian(
        z -> _implicit_system(pprob, z, unk, W, act_ph, active, bv; dead, degenerate), z0,
    )
    Φd = _implicit_system(prob, z0, unk, W, act_ph, active, b; dead, degenerate)
    rhs = [ForwardDiff.partials(Φd[r], k) for r in eachindex(Φd), k in 1:P]
    # A member the inversion does not determine is held: its equation and its
    # unknown leave the system, and its amount has no derivative.
    Nv = length(v)
    nG = length(unk)
    keep = _determined(@view Φz[1:nG, (Nv + 1):end])
    rows = vcat(keep, (nG + 1):size(Φz, 1))
    cols = vcat(1:Nv, Nv .+ keep)
    F = lu(Φz[rows, cols]; check = false)
    ż_k = -(issuccess(F) ? F \ rhs[rows, :] : qr(Φz[rows, cols], ColumnNorm()) \ rhs[rows, :])
    ż = zeros(eltype(ż_k), size(Φz, 2), P)
    ż[cols, :] .= ż_k

    nph, na = length(act_ph), length(active)
    ẋ = zeros(eltype(ż), length(x), P)
    for (a, k) in enumerate(act_ph)
        ph = pprob.phases[k]
        ph.mole_fraction && continue
        ir = ph.members[ph.j_ref]
        ẋ[ir, :] .= x[ir] .* ż[a, :]
    end
    for (p, (_, _, _, i, _)) in enumerate(unk)
        ẋ[i, :] .= x[i] .* ż[Nv + p, :]
    end
    for (j, i) in enumerate(active)
        ẋ[i, :] .= ż[nph + m + j, :]
    end
    DT = ForwardDiff.Dual{Tg, V, P}
    dual(val, row) = DT(V(val), ForwardDiff.Partials(ntuple(k -> V(row[k]), P)))
    x_d = DT[dual(x[i], @view ẋ[i, :]) for i in eachindex(x)]
    y_d = DT[dual(y[c], @view ż[nph + c, :]) for c in 1:m]
    q_d = DT[dual(q[c], @view ż[nph + m + na + c, :]) for c in 1:nq]
    return (; x = x_d, y = y_d, q = q_d)
end
