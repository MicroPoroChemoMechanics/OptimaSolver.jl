# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# ── dual_newton.jl ────────────────────────────────────────────────────────────
#
# Exact Newton on the KKT system, in the space of the equality multipliers.
#
# THE PROBLEM. Minimize `f(x) = Σᵢ xᵢ(gᵢ + hᵢ(x))` subject to `A x = b`, `x ≥ 0`,
# where `h` is a callback supplying the state-dependent part of the gradient and
# `∇f = g + h(x)` (this identity holds whenever `Σⱼ xⱼ ∂hⱼ/∂xᵢ = 0`, which is
# the Gibbs–Duhem relation of a chemical system and the Euler relation of any
# first-order homogeneous mixing term).
#
# It is convex whenever `h` is the gradient of a convex mixing term and the
# remaining variables enter linearly, so the KKT conditions are necessary AND
# sufficient: a point satisfying them is proved globally optimal.
#
# THE FORMULATION. With `y` the multipliers of `A x = b` and `u := −Aᵀy`, the
# stationarity conditions split by the nature of the variable:
#
#   interior variable   hᵢ(x) = uᵢ − gᵢ           (invertible: a mass-action law)
#   bounded variable    gᵢ = uᵢ if present, gᵢ ≥ uᵢ if at its bound
#
# The second line is a phase-stability criterion: a bounded variable is positive
# exactly when its "saturation index" `uᵢ − gᵢ` vanishes, and zero when negative.
#
# WHY IT IS WELL CONDITIONED. Parameterizing the interior variables by `w = ln x`
# makes their positivity automatic, so no fraction-to-boundary rule applies to
# them — and that rule is what caps the step of the interior-point method in
# `solve!` at every single iteration. Only the bounded variables carry a bound,
# and they are handled by an active set, exactly and finitely.
#
# This is NOT the log reparameterization of the objective that
# `variable_space = Val(:log)` performs. `f ∘ exp` is not convex — its second
# derivative `xᵢ(∇fᵢ + 1)` is negative wherever `∇fᵢ < −1`, which for a chemical
# potential of order −200 is everywhere. Here the logarithm is applied to the KKT
# EQUATIONS, solved as a square nonlinear system; convexity of the original
# problem is what makes that system's solution unique.

using LinearAlgebra

"""
    DEGENERATE_POTENTIAL

Multiplier assigned to a component whose right-hand side has vanished. Any value
large enough to drive `exp(uᵢ − gᵢ)` below the floor for every variable carrying
that component will do; the solution does not depend on it.
"""
const DEGENERATE_POTENTIAL = 500.0

"""
    degenerate_components(A, b) -> Vector{Int}

Rows `k` of `A x = b` whose right-hand side forces every variable carrying
component `k` to zero.

The criterion is **not** simply `b_k ≈ 0`. With `x ≥ 0`, the row
`Σᵢ A_{ki} xᵢ = 0` forces `xᵢ = 0` for all `i` with `A_{ki} ≠ 0` only when the
non-zero entries of the row share a sign: a sum of non-negative terms vanishes
only if each vanishes. A row with entries of both signs permits cancellation and
forces nothing.

That distinction is not academic. In a chemical system the `H+` row carries `−1`
for `OH-` and `+1` for `H+`, so `b = 0` there is the ordinary state of pure
water — declaring it degenerate kills the whole acid–base system and returns
pH 7.000 with the solid undissolved.
"""
function degenerate_components(A::AbstractMatrix, b::AbstractVector)
    m = size(A, 1)
    scale = max(maximum(abs, b; init = 0.0), 1.0)
    out = Int[]
    @inbounds for k in 1:m
        abs(b[k]) <= 1.0e-12 * scale || continue
        pos = false
        neg = false
        for j in axes(A, 2)
            a = A[k, j]
            a > 0 && (pos = true)
            a < 0 && (neg = true)
            pos && neg && break
        end
        (pos && neg) || push!(out, k)
    end
    return out
end

"""
    SolutionPhase(members, j_ref; always_present = false)

A set of variables whose `hᵢ` depends on the composition of the set — a mixing
phase — together with the position **within `members`** of its reference.

Every member of such a phase satisfies `hᵢ = uᵢ − gᵢ`, and for all but the
reference that condition is inverted directly. The reference is exempt for a
structural reason: `hᵢ` is a logarithm of a mole fraction, hence **bounded
above**, so `hᵢ = uᵢ − gᵢ` has no solution for an arbitrary `y`. Inverting it is
not merely slow, it can be infeasible, and an inner loop that included it can
never report convergence — which invalidates the outer Jacobian, that Jacobian
being derived on the assumption the inner conditions hold exactly. Its
stationarity is carried by the outer system, where it fixes the phase's total.

`always_present = true` exempts the phase from the stability test: the aqueous
phase of a wet system is there by hypothesis, a solid solution is not.
"""
struct SolutionPhase
    members::Vector{Int}
    j_ref::Int
    always_present::Bool
    mole_fraction::Bool
end

"""
    SolutionPhase(members, j_ref; always_present=false, mole_fraction=false)

One mixing phase: the variable indices it holds, which of them is the reference,
whether it may leave, and whether EVERY member's activity is a mole fraction.

That last flag decides how the members are recovered, and the two cases are
genuinely different rather than a matter of taste.

In an aqueous solution the solutes carry molalities, unbounded above, so `hᵢ` can
meet `uᵢ − gᵢ` whatever its value and each member is recovered from its own
stationarity. Only the solvent is a mole fraction, which is exactly why it is the
reference and why its equation is carried by the outer system instead.

In a solid solution there is no solvent: every member is a mole fraction and
every `hᵢ` is bounded above by zero. Asking any of them to meet a positive
`uᵢ − gᵢ` is asking for the impossible, and the iteration answers by growing the
member without bound. Such a phase is recovered from the RATIOS between its
members, which are always attainable, with the absolute level left to the
reference's own equation — where it belongs.
"""
SolutionPhase(
    members, j_ref; always_present::Bool = false, mole_fraction::Bool = false,
) = SolutionPhase(collect(Int, members), j_ref, always_present, mole_fraction)

"""
    DualNewtonProblem(A, g, h; phases, idx_bounded, params)

A convex program in the form solved by [`dual_newton_solve`](@ref):

```math
\\min_x \\sum_i x_i\\bigl(g_i + h_i(x)\\bigr)
\\quad\\text{s.t.}\\quad A x = b, \\; x \\ge 0 .
```

# Arguments

  - `A`: the `m × n` equality matrix.
  - `g`: the constant part of the gradient, `∇f = g + h(x)`.
  - `h`: callback `h(x, params) -> Vector`, the state-dependent part.

# Keywords

  - `phases`: the mixing phases, as [`SolutionPhase`](@ref) values. Their members
    are strictly positive while the phase exists, hence parameterized by `ln x`
    and recovered from their own stationarity.
  - `idx_bounded`: variables outside every phase. Their `hᵢ` does not depend on
    the composition — a pure phase, of unit activity — so they are either at a
    stationarity of their own or at zero, and an active set decides which.
  - `params`: passed through to `h`.
  - `always_active`: bounded variables that are never dropped from the active set.
    A variable whose amount is fixed by a linear row is not deciding anything by a
    sign test: it is determined, and its extra multiplier makes its stationarity
    satisfiable at whatever amount the row demands. Without this it can never
    enter, because it starts at zero and the drop rule removes anything below
    `si_tol` — which is exactly what happens to the products of a solid-to-solid
    reaction whose extents pin them.

# The two kinds of variable

The distinction is not cosmetic. A pure phase satisfies `gᵢ + hᵢ = uᵢ` when
present and `≥` when absent, and it can be **exactly** zero. A member of a mixing
phase cannot: its activity goes to `−∞` as its fraction goes to zero, so it is
never exactly absent while the phase exists. The active set for a mixing phase is
therefore over the PHASE, and the criterion is a tangent-plane test rather than a
sign of a saturation index.
"""
struct DualNewtonProblem{T <: Real, H, G, C, HQ}
    A::Matrix{T}
    g::Vector{T}
    h::H
    phases::Vector{SolutionPhase}
    idx_bounded::Vector{Int}
    params::Any
    # ── the q block: prescribed properties, solved for simultaneously ────────
    nq::Int              # number of unknown parameters (0 for a plain T, P solve)
    gq::G                # gq(q, params) -> Vector, the standard part at those q
    cq::C                # cq(x, q, params) -> Vector of length nq, the residuals
    hq::HQ               # hq(x, q, params) -> Vector, `h` when it depends on q
    Aq::Matrix{T}        # m × nq, so the linear rows read `A x + Aq q = b`
    always_active::Vector{Int}   # bounded variables pinned by a linear row
    q0::Vector{T}        # starting guess
    qscale::Vector{T}    # difference-step scale, one per entry
end

function DualNewtonProblem(
        A::AbstractMatrix, g::AbstractVector, h;
        phases::AbstractVector{SolutionPhase},
        idx_bounded::AbstractVector{Int} = Int[],
        params = nothing,
        gq = nothing,
        cq = nothing,
        hq = nothing,
        q0::AbstractVector = Float64[],
        qscale::AbstractVector = Float64[],
        Aq::AbstractMatrix = zeros(Float64, size(A, 1), length(q0)),
        always_active::AbstractVector{Int} = Int[],
    )
    isempty(phases) && throw(
        ArgumentError(
            "`phases` is empty: the multipliers are determined by the stationarity " *
                "of the variables inside a mixing phase, and without one there is " *
                "nothing to determine them."
        )
    )
    for (k, ph) in pairs(phases)
        1 <= ph.j_ref <= length(ph.members) || throw(
            ArgumentError("phase $k: `j_ref` = $(ph.j_ref) is not a position in `members`.")
        )
    end
    seen = Set{Int}()
    for ph in phases, i in ph.members
        i in seen && throw(ArgumentError("variable $i belongs to more than one phase."))
        push!(seen, i)
    end
    for i in idx_bounded
        i in seen && throw(
            ArgumentError("variable $i is both in a phase and bound-constrained.")
        )
    end

    nq = length(q0)
    if nq > 0
        # `gq` is optional. A prescribed temperature moves the standard
        # potentials, so it needs one; a reaction extent does not — it enters
        # through `Aq` and `cq` only, `g` staying put. Requiring `gq` refused a
        # perfectly well-posed kinetic step.
        cq === nothing && throw(
            ArgumentError("`q0` has $nq entries but no `cq`: each unknown parameter " *
                          "needs one residual equation.")
        )
        length(qscale) == nq || throw(
            ArgumentError("`qscale` must have one entry per unknown parameter " *
                          "($nq), got $(length(qscale)). It sets the difference " *
                          "step, and a temperature in kelvin and a reaction extent " *
                          "in moles do not share a scale.")
        )
        all(>(0), qscale) || throw(ArgumentError("`qscale` entries must be positive."))
        size(Aq) == (size(A, 1), nq) || throw(
            ArgumentError("`Aq` must be $(size(A, 1))×$nq to sit beside `A` in the " *
                          "linear rows, got $(size(Aq)).")
        )
    end

    return DualNewtonProblem(
        Matrix{Float64}(A), Vector{Float64}(g), h,
        collect(SolutionPhase, phases), collect(Int, idx_bounded), params,
        nq, gq, cq, hq, Matrix{Float64}(Aq),
        collect(Int, always_active),
        Vector{Float64}(q0), Vector{Float64}(qscale),
    )
end

"""
    current_g(prob, q) -> Vector

The standard part of the gradient at the unknown parameters `q`: `prob.g` when
there are none, `prob.gq(q, prob.params)` otherwise.
"""
current_g(prob::DualNewtonProblem, q) =
    (prob.nq == 0 || prob.gq === nothing) ? prob.g : prob.gq(q, prob.params)

"""
    current_h(prob, x, q) -> Vector

The state-dependent part of the gradient. `prob.h(x, params)` unless the problem
declares `hq`, in which case `prob.hq(x, q, params)`.

An activity model can depend on the unknown parameters as well: the Debye-Hückel
coefficients are functions of temperature, so an adiabatic solve that left `h` at
the starting temperature would be minimizing the wrong Gibbs energy. Declaring it
explicitly is what keeps that dependence visible instead of routing it through a
mutated `params`, where the difference-quotient loop would evaluate it in an order
nothing guarantees.
"""
current_h(prob::DualNewtonProblem, x, q) =
    (prob.nq == 0 || prob.hq === nothing) ? prob.h(x, prob.params) :
    prob.hq(x, q, prob.params)

"""
    stationarity_capacity(prob) -> Int

The number of conservation rows `m`, which is the largest number of simultaneous
stationarity conditions the element potentials can carry.

This is Gibbs' phase rule, in the form the dual system takes. A bound-constrained
variable held ACTIVE contributes `uᵢ = gᵢ`, i.e. `aᵢᵀ y = −gᵢ`, one **linear**
equation in `y`; a mole-fraction mixing phase contributes
`logsumexp(uᵢ − gᵢ) = 0`, one more, nonlinear but still a condition on `y` alone.
A phase with a solvent does not, because its reference equation involves the
composition. With `y ∈ ℝᵐ` there is no `y` satisfying more than `m` of them, so an
active set carrying more cannot support a solution — the Newton residual cannot
reach zero for any iterate, and the least-squares step merely spreads the
violation over the rows.

That is not a slow case, it is an unsolvable one, and it has to be excluded by
construction rather than discovered. Measured on an LC³ equilibrium the active set
grew to 15 pure phases and 5 solid solutions — **19 conditions on 12 components** —
and the solve came back with a stationarity residual of 18 and an element balance
of 800 having never had a solution to find.
"""
stationarity_capacity(prob::DualNewtonProblem) = size(prob.A, 1)

"""
    _n_stationarity_conditions(prob, active, act_ph) -> Int

How many conditions on `y` alone the active set imposes: one per active
bound-constrained variable, one per active mole-fraction phase.
"""
function _n_stationarity_conditions(prob, active, act_ph)
    n = length(active)
    for k in act_ph
        prob.phases[k].mole_fraction && (n += 1)
    end
    return n
end

"""
    _active_set_supports_a_solution(prob, active, act_ph) -> Bool

Whether the stationarity block of this active set can be satisfied at some `y`.

Two conditions, both necessary. The count must not exceed
[`stationarity_capacity`](@ref). And the active bound-constrained variables'
composition vectors must be linearly INDEPENDENT: `A[:, active]ᵀ y = −g[active]`
is solvable for some `y` only then, since two dependent columns demand a fixed
relation between their `gᵢ` that the database will not happen to satisfy. That
second test is what catches two polymorphs of one composition — `Gbs` and
`AlOHmic` are both Al(OH)₃ — which impose `uᵢ = gᵢ` twice on the same vector.
"""
function _active_set_supports_a_solution(prob, active, act_ph)
    _n_stationarity_conditions(prob, active, act_ph) <= stationarity_capacity(prob) ||
        return false
    isempty(active) && return true
    # The rank is taken on the VALUE part, so that a `ForwardDiff.Dual` matrix — a
    # conservation matrix carrying a stoichiometric parameter, say the Mg/Al ratio
    # of a hydrotalcite or the Si substitution of a katoite — gives the same
    # answer as its primal. A rank IS a property of the value part: letting an
    # infinitesimal perturbation change which phases the algorithm considers
    # admissible would make the derivative of the answer discontinuous, which is
    # not a subtlety one wants to discover downstream.
    return LinearAlgebra.rank(ForwardDiff.value.(prob.A[:, active])) == length(active)
end

"""
    DualNewtonOptions(; tol, maxit, max_active_updates, si_tol, verbose)

  - `tol`: tolerance on the KKT residual.
  - `maxit`: Newton iterations per active set.
  - `max_active_updates`: how many times the active set may change.
  - `si_tol`: saturation index above which a variable at its bound is admitted.
"""
Base.@kwdef struct DualNewtonOptions
    tol::Float64 = 1.0e-10
    maxit::Int = 200
    max_active_updates::Int = 200
    si_tol::Float64 = 1.0e-8
    verbose::Bool = false
end

# ── inner level: invert the stationarity of the log variables ─────────────────

"""
    _invert_phases!(prob, W, y, refs, act_ph, active, xB, x_buf; dead) -> x_buf

Recover the members of every ACTIVE mixing phase from their own stationarity,
`hᵢ(x) = uᵢ − gᵢ`, at fixed multipliers and fixed phase references.

For a variable whose `hᵢ` behaves like `ln xᵢ` plus a slowly varying part — which
is what a logarithm of a mole fraction is — `∂hᵢ/∂wᵢ = 1`, so `w += r` is an
exact Newton step; the coupling through the phase total and the activity
coefficients is what makes the loop iterative rather than one-shot.

The floor is `exp(-700)`: a variable whose stationarity demands `hᵢ = −200` can
never reach it against a floor of `−80`, its residual stays at 120 for ever, and
the loop can never report convergence — which then makes the OUTER Jacobian
meaningless.
"""
# Composition of a mole-fraction phase from the potentials alone.
#
# `x_i = N · softmax(u_i − g_i − lnγ_i)` and the phase is consistent when
# `Σ_i exp(u_i − g_i − lnγ_i) = 1`, i.e. when the log-sum-exp vanishes. Both are
# written with the maximum factored out, which is the only form that survives the
# range of `u − g` a cement produces.
function _mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead, g = prob.g)
    d = Vector{Float64}(undef, length(ph.members))
    for (j, i) in enumerate(ph.members)
        if i in dead
            d[j] = -Inf
            continue
        end
        xi = x_buf[i]
        # `lnγ` is read at the current composition; identically zero for ideal
        # mixing, so the loop below converges in one pass there.
        lnγ = (xi > 0 && N > 0) ? hv[i] - log(xi / N) : 0.0
        d[j] = u[i] - g[i] - lnγ
    end
    return d
end

function _logsumexp(d)
    M = maximum(d)
    isfinite(M) || return -Inf
    return M + log(sum(exp(dj - M) for dj in d))
end

function _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
    fill!(x_buf, 0.0)
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        for (j, i) in enumerate(ph.members)
            x_buf[i] = (!ph.mole_fraction && j == ph.j_ref) ? refs[a] : exp(W[k][j])
        end
    end
    @inbounds for (j, i) in enumerate(active)
        x_buf[i] = xB[j]
    end
    return x_buf
end

function _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf;
        dead = Set{Int}(), g = prob.g, q = prob.q0,
    )
    u = -(transpose(prob.A) * y)

    for _ in 1:200
        _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
        hv = current_h(prob, x_buf, q)
        worst = 0.0

        for (a, k) in enumerate(act_ph)
            ph = prob.phases[k]

            if ph.mole_fraction
                # No member of a solid solution can be inverted from its own
                # stationarity: `hᵢ` is the logarithm of a mole fraction, bounded
                # above by zero, so a positive `uᵢ − gᵢ` is unreachable and the
                # iteration answers by growing the member without bound. What the
                # potentials DO determine, always, is the composition: the
                # fractions are the softmax of `u − g − lnγ`, and the phase's
                # total is the outer unknown. Nothing here can overflow.
                N = refs[a]
                d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead, g)
                M = maximum(d)
                isfinite(M) || continue          # every member of the phase is dead
                lZ = M + log(sum(exp(dj - M) for dj in d))
                for (j, _) in enumerate(ph.members)
                    w = clamp(log(N) + d[j] - lZ, -700.0, 700.0)
                    worst = max(worst, abs(w - W[k][j]))
                    W[k][j] = w
                end
            else
                # An aqueous solution has a solvent, and the solutes carry
                # molalities that are unbounded above, so each of them IS
                # recoverable from its own stationarity — and must be, because
                # that is where their dependence on the potentials lives.
                for (j, i) in enumerate(ph.members)
                    (j == ph.j_ref || i in dead) && continue
                    r = (u[i] - g[i]) - hv[i]
                    worst = max(worst, abs(r))
                    W[k][j] = clamp(W[k][j] + clamp(r, -30.0, 30.0), -700.0, 20.0)
                end
            end
        end

        worst <= 1.0e-14 && break
    end

    return _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
end

"""
    _outer_residual(prob, v, W, act_ph, active, b, x_buf; dead, degenerate) -> R

The outer residual at `v = [ln x_ref(φ) for each active phase; y; x_B; q]`:

  - stationarity of each active phase's reference — one equation per phase, which
    is what fixes that phase's total amount;
  - the equality constraints `A x − b` — `m` equations;
  - stationarity of the active bound-constrained variables — `|P|` equations;
  - `prob.cq(x, q, params)` — one equation per unknown parameter.

`|Φ| + m + |P| + nq` equations in as many unknowns: some fifteen for a cement,
against forty-seven variables in the interior-point route.

The `q` block is how a prescribed property is imposed: an adiabatic solve has `T`
in `q` and `H(x, T) − H₀` in `cq`; a fixed-volume solve has `P` in `q` and
`V(x, P) − V₀`; a kinetic step has the reaction extents in `q` and
`Δξ − Δt·M·r(x)`. They are unknowns of the SAME system as the amounts and the
multipliers, not an outer loop around it, which is the structure Reaktoro uses
and the reason a kinetic step costs `nq` extra equations rather than a second
solve.
"""
function _outer_residual(
        prob, v, W, act_ph, active, b, x_buf;
        dead = Set{Int}(), degenerate = Int[],
    )
    m = size(prob.A, 1)
    nph = length(act_ph)
    na = length(active)
    nq = prob.nq

    refs = [exp(v[a]) for a in 1:nph]
    y = v[(nph + 1):(nph + m)]
    xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
    q = nq == 0 ? Float64[] : v[(nph + m + na + 1):(nph + m + na + nq)]
    g = current_g(prob, q)

    _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf; dead = dead, g = g, q = q,
    )

    hv = current_h(prob, x_buf, q)
    u = -(transpose(prob.A) * y)

    # One equation per active phase, fixing its absolute level.
    #
    # With a solvent that is the reference member's own stationarity. Without one
    # it is the statement that the mole fractions sum to unity, written as a
    # log-sum-exp — which is the same quantity the tangent-plane test uses to
    # decide whether the phase may form at all, so admission and stationarity are
    # measured by one expression rather than two that could disagree.
    R_ref = Float64[]
    for (a, k) in enumerate(act_ph)
        ph = prob.phases[k]
        if ph.mole_fraction
            d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, refs[a], dead, g)
            push!(R_ref, _logsumexp(d))
        else
            ir = ph.members[ph.j_ref]
            push!(R_ref, g[ir] + hv[ir] - u[ir])
        end
    end

    # `A x + Aq q − b`: a linear relation may involve the unknown parameters as
    # well. That is how a reaction extent enters — the reactivity constraint
    # `Kᵀn − Δξ = ξ₀` is LINEAR, so it belongs in the conservation block beside
    # the elements and the charge, and the algebraic cost of kinetics is the
    # number of reactions rather than the number of species.
    Rb = nq == 0 ? (prob.A * x_buf .- b) : (prob.A * x_buf .+ prob.Aq * q .- b)

    # The stationarity of an active bound-constrained variable is `gᵢ + hᵢ = uᵢ`,
    # not `gᵢ = uᵢ`. The two coincide only when `hᵢ = 0`, the case of a pure phase;
    # writing the general form costs nothing there and is the only correct one
    # otherwise.
    Rs = na == 0 ? Float64[] : g[active] .+ hv[active] .- u[active]

    # For a vanished component the balance row carries no information — every
    # variable containing it is pinned at the floor, so the row reads 0 = 0 and
    # leaves `y_k` free, making the Jacobian singular. Replace it by the equation
    # that fixes the multiplier, which keeps the system square.
    for k in degenerate
        Rb[k] = y[k] - DEGENERATE_POTENTIAL
    end

    Rq = nq == 0 ? Float64[] : prob.cq(x_buf, q, prob.params)
    length(Rq) == nq || throw(
        DimensionMismatch("`cq` returned $(length(Rq)) residuals for $nq unknown " *
                          "parameters; the system would not be square.")
    )

    return vcat(R_ref, Rb, Rs, Rq)
end

"""
    _phase_tangent(prob, k, u, x_buf) -> Float64

Michelsen's stability measure for an absent mixing phase: `Σᵢ exp(uᵢ − gᵢ) − 1`,
evaluated over its members.

At equilibrium a present phase has `hᵢ = uᵢ − gᵢ` with `hᵢ = ln xᵢ + ln γᵢ`, so
its fractions satisfy `Σᵢ exp(uᵢ − gᵢ)/γᵢ = 1`. A value above one means a trial
composition of that phase lies below the tangent plane of the current state — the
phase can form and lower `G`. Below one it cannot.

This is the criterion a **mixing** phase needs, and it is not the sign of a
saturation index: a solution phase has no single index, and its members are never
exactly absent while it exists.
"""
function _phase_tangent(prob, k, u, x_buf, g = prob.g)
    ph = prob.phases[k]
    s = 0.0
    for (j, i) in enumerate(ph.members)
        # IDEAL test: `lnγ` is taken as zero, not read from the activity model.
        # This is the SEARCH heuristic — which candidate phase to try next — and a
        # first-order test is the right cost there. The comment here used to claim
        # γ was read at the current state, which the code never did.
        #
        # Nothing in the proof depends on it: `kkt_certificate` uses the true
        # `∇f = g + h(x)` for the variables it tests, and
        # `phase_tangent_measure` for the mixing phases held absent, which
        # refines the trial composition against the phase's own activity model.
        s += exp(clamp(u[i] - g[i], -700.0, 50.0))
    end
    return s - 1.0
end

"""
    phase_tangent_measure(prob, k, u, x; maxit = 50, tol = 1e-12, total = 1e-6)

Michelsen's tangent-plane measure for mixing phase `k` at the multipliers `u` and
composition `x`: the log-sum-exp of `uᵢ − gᵢ − lnγᵢ` over the phase's members,
with the trial composition refined against the phase's OWN activity model by
successive substitution.

Positive means a trial composition of the phase lies below the tangent plane, so
the phase can form and a composition without it is not optimal. Zero means the
phase is exactly at its stability limit.

This is what an absent **mixing** phase requires, and it is not the sign of a
saturation index: a solution phase has no single index, and its members are never
exactly zero while it exists — which is why `kkt_certificate` cannot test them
one by one and needs this instead.

`total` is the trial phase amount. The measure is independent of it for an ideal
phase, and for a non-ideal one it sets the composition at which `γ` is read; a
small value is the right choice, since the question is whether an *infinitesimal*
amount of the phase is stable.
"""
function phase_tangent_measure(
        prob::DualNewtonProblem, k::Int, u::AbstractVector, x::AbstractVector;
        maxit::Int = 50, tol::Float64 = 1.0e-12, total::Float64 = 1.0e-6,
        g = prob.g, q = prob.q0,
    )
    ph = prob.phases[k]
    nm = length(ph.members)
    nm == 0 && return -Inf
    xt = Vector{Float64}(x)
    frac = fill(1.0 / nm, nm)
    lnZ = -Inf
    d = Vector{Float64}(undef, nm)
    for _ in 1:maxit
        for (j, i) in enumerate(ph.members)
            xt[i] = total * frac[j]
        end
        hv = current_h(prob, xt, q)
        for (j, i) in enumerate(ph.members)
            # `hᵢ` is `ln aᵢ = ln(xᵢ/N) + lnγᵢ`, so `lnγ` is what remains once the
            # ideal part is removed.
            lnγ = xt[i] > 0 ? hv[i] - log(xt[i] / total) : 0.0
            d[j] = u[i] - g[i] - lnγ
        end
        M = maximum(d)
        isfinite(M) || return -Inf
        lnZ = M + log(sum(exp(dj - M) for dj in d))
        newfrac = [exp(dj - lnZ) for dj in d]
        Δ = maximum(abs, newfrac .- frac)
        frac = newfrac
        Δ < tol && break
    end
    return lnZ
end

# ── the solve ─────────────────────────────────────────────────────────────────

"""
    _element_potential_start(A, g, b, y0; dead, maxit, tol) -> y

Element potentials from the CONCAVE dual of the ideal problem — Brinkley's method,
after White, Johnson & Dantzig (1958).

Treat every species as an ideal one whose activity is its own amount. The
Lagrangian then minimizes in closed form, `xᵢ = exp(uᵢ − gᵢ)` with `u = −Aᵀy`, and
the dual becomes

```
φ(y) = −bᵀ y − Σᵢ exp(uᵢ − gᵢ),
∇φ  = A x − b,
∇²φ = −A diag(x) Aᵀ ≺ 0 .
```

`φ` is smooth and **strictly concave**, so Newton with a backtracking line search
converges from any starting point — there is no active set, no combinatorics, and
no way to stall in a wrong one. Its solution is the `y` for which the ideal
composition conserves matter exactly.

That is not the model this solver goes on to solve — the real phases carry mole
fractions and molalities, not bare amounts — but it is the right place to start
from, and it is what the three least-squares fits were standing in for. Those fits
minimize `‖Aᵀy + g + h‖` over a subset of species, which says nothing about the
element budget; the potentials they produce are consistent with no composition in
particular. Measured on an LC³ equilibrium at a quarter of full reaction, the inner
Newton failed to converge on EVERY active set the search visited, with residuals
between 8 and 19, and the same problem reached by continuation from a nearby
solution converged to 1e-11.
"""
function _element_potential_start(
        A::AbstractMatrix{T}, g::AbstractVector, b::AbstractVector, y0::AbstractVector;
        dead = Set{Int}(), maxit::Int = 200, tol::Float64 = 1.0e-10,
    ) where {T <: Real}
    m = size(A, 1)
    y = collect(float.(y0))
    alive = [i for i in eachindex(g) if !(i in dead)]
    Aa = Matrix(@view A[:, alive])
    ga = collect(float.(g[alive]))
    bscale = max(1.0, maximum(abs, b))

    xa = similar(ga)
    φ(yv) = begin
        u = -(transpose(Aa) * yv)
        # `exp` is clamped only against overflow; the line search below is what
        # keeps the iterates in a range where that clamp is never reached.
        -dot(b, yv) - sum(exp(clamp(u[j] - ga[j], -700.0, 300.0)) for j in eachindex(ga))
    end

    φ_cur = φ(y)
    for _ in 1:maxit
        u = -(transpose(Aa) * y)
        @inbounds for j in eachindex(ga)
            xa[j] = exp(clamp(u[j] - ga[j], -700.0, 300.0))
        end
        r = Aa * xa .- b
        maximum(abs, r) <= tol * bscale && break

        H = Aa * Diagonal(xa) * transpose(Aa)
        dmax = maximum(abs, diag(H))
        @inbounds for k in 1:m
            H[k, k] += max(dmax, 1.0) * 1.0e-12
        end
        δ = qr(H, ColumnNorm()) \ r

        # Backtracking on a concave function: an ascent step exists for small
        # enough `α`, so this cannot fail to make progress unless `∇φ = 0`.
        α = 1.0
        improved = false
        for _ in 1:60
            φ_try = φ(y .+ α .* δ)
            if isfinite(φ_try) && φ_try > φ_cur
                y .+= α .* δ
                φ_cur = φ_try
                improved = true
                break
            end
            α /= 2
        end
        improved || break
    end
    return y
end

"""
    dual_newton_solve(prob, b, x0; opts) -> (; x, y, q, active_phases, active, converged)

Solve `prob` for the right-hand side `b`, starting from `x0`.

`x0` supplies a neighborhood, not a feasible point: this is a Newton method, and
the intended use is to hand it the answer of an interior-point solve. Verify the
result with [`kkt_certificate`](@ref); for a convex problem that certificate is a
proof of **global** optimality.

# Two active sets

Over the **bound-constrained** variables, on the sign of `uᵢ − (gᵢ + hᵢ)`: a pure
phase is present exactly when that index vanishes, absent when it is negative.

Over the **mixing phases**, on Michelsen's tangent-plane measure
`Σᵢ exp(uᵢ − gᵢ) − 1`: a solution phase forms when a trial composition of it lies
below the tangent plane of the current state. That test is what a mixing phase
requires, since its members are never exactly absent while it exists and it has
no single saturation index.

Both admit one candidate per round, the most violated, and the sets visited are
recorded, which bounds the loop by the number of subsets and therefore
terminates. Admitting a batch feeds a cycle in which a variable is admitted,
driven negative, dropped and readmitted.
"""
function dual_newton_solve(
        prob::DualNewtonProblem, b::AbstractVector, x0::AbstractVector;
        opts::DualNewtonOptions = DualNewtonOptions(),
    )
    bv = Vector{Float64}(b)
    n0 = Vector{Float64}(x0)
    m = size(prob.A, 1)
    x_buf = zeros(Float64, length(prob.g))

    degenerate = degenerate_components(prob.A, bv)
    dead = isempty(degenerate) ? Set{Int}() :
        Set(j for j in eachindex(prob.g) if any(abs(prob.A[k, j]) > 0 for k in degenerate))


    # One log-vector per phase, whether or not the phase is active: an inactive
    # phase keeps its last state, so readmitting it costs nothing.
    W = [Float64[log(max(n0[i], 1.0e-30)) for i in ph.members] for ph in prob.phases]
    for (k, ph) in pairs(prob.phases), (j, i) in pairs(ph.members)
        i in dead && (W[k][j] = -700.0)
    end

    # A phase starts active if it is always present or if the guess holds it — and
    # never if every one of its members carries a vanished component.
    #
    # That last clause is not defensive tidying. A phase whose components are all
    # absent from the budget cannot exist, and its stationarity condition
    # `logsumexp(uᵢ − gᵢ) = 0` is evaluated over an empty set: every exponent is
    # `-Inf`, the log-sum-exp is `-Inf`, and the outer residual is `Inf` from the
    # first evaluation onwards. The admission test below already excludes such a
    # phase; the seeding did not, and against an interior-point warm start that
    # matters, because a barrier point holds even a dead species near `μ` rather
    # than at zero. On an LC³ budget, which carries no magnesium at all, that
    # seeded the M-S-H or hydrotalcite solution as present and the solve never
    # produced a finite residual.
    act_ph = Int[
        k for (k, ph) in pairs(prob.phases)
            if !all(i in dead for i in ph.members) &&
            (ph.always_present || sum(n0[i] for i in ph.members) > 1.0e-9)
    ]
    isempty(act_ph) && (act_ph = Int[1])
    # `refs[a]` is the reference member's amount for a phase with a solvent, and
    # the phase TOTAL for a mole-fraction phase — in both cases the quantity whose
    # logarithm the outer system carries.
    refs = Float64[
        let ph = prob.phases[k]
                ph.mole_fraction ? max(sum(n0[i] for i in ph.members), 1.0e-6) :
                max(n0[ph.members[ph.j_ref]], 1.0e-6)
        end for k in act_ph
    ]

    # The initial active set must already satisfy the phase rule, and reading it
    # off a threshold does not.
    #
    # `n0[i] > 1e-6` was the test, and against an interior-point warm start it is
    # no test at all: a barrier point at `μ` holds every ABSENT phase at
    # `sᵢ = μ/gᵢ`, which for `μ = 1e-6` is exactly the threshold. On an LC³ budget
    # that seeded 15 pure phases, 5 solid solutions on top, and 19 stationarity
    # conditions on 12 components — an active set with no solution, from the first
    # iteration, and no way back since the loop only ever rejects the variable it
    # just added.
    #
    # Candidates are therefore taken in order of decreasing amount — the phase
    # rule says at most `m` phases are present, and the abundant ones are the best
    # guess as to which — and admitted only while the set still supports a
    # solution. What the seeding leaves out is not lost: the saturation index
    # brings it back in below, by exchange.
    active = Int[]
    xB = Float64[]
    # Pinned variables go in first and unconditionally: they are determined by a
    # linear row, not admitted by a test, and the phase-rule check below would
    # reject a species sitting at zero.
    for i in prob.always_active
        i in active && continue
        push!(active, i)
        push!(xB, max(n0[i], 1.0e-12))
    end
    let cand0 = sort(
            [i for i in prob.idx_bounded
                if n0[i] > 1.0e-6 && !(i in dead) && !(i in prob.always_active)];
            by = i -> -n0[i],
        )
        for i in cand0
            push!(active, i)
            if _active_set_supports_a_solution(prob, active, act_ph)
                push!(xB, n0[i])
            else
                pop!(active)
            end
        end
    end

    # `y` from a WEIGHTED least-squares fit of the phase stationarity at the
    # guess. It holds for every phase member at the solution, but as a starting
    # point the fit must be driven by the variables that carry weight: an
    # interior-point answer is accurate on the major ones and can be a factor 1e5
    # out on a 1e-9 trace, and an unweighted fit lets those traces set `y`.
    fill!(x_buf, 0.0)
    for (a, k) in enumerate(act_ph), (j, i) in pairs(prob.phases[k].members)
        x_buf[i] = j == prob.phases[k].j_ref ? refs[a] : exp(W[k][j])
    end
    for (j, i) in enumerate(active)
        x_buf[i] = xB[j]
    end
    # The starting guess is built at the starting parameters, deliberately: the
    # Newton loop moves both together from there.
    h0 = current_h(prob, x_buf, prob.q0)
    # `y` from the stationarity of the phase members at the guess. At the
    # solution that condition holds for EVERY member, so any `m` independent
    # equations fix `y`; as a STARTING POINT the question is which of them the
    # guess gets right, and no single answer serves every problem.
    #
    #   * weighting by `√xᵢ` lets the solvent outweigh a trace by fifteen decades,
    #     so the fit becomes a statement about the solvent alone;
    #   * weighting equally lets a trace the guess got wrong by a factor 1e5 set
    #     the multipliers;
    #   * fitting on the largest members only is well conditioned but discards
    #     information the other two keep.
    #
    # Each is right somewhere and wrong elsewhere: measured on a cement replay,
    # the first certifies 36 instants of 40 and the third certifies the 4 it
    # misses while losing 34 of the others. So the solver TRIES them in turn and
    # keeps the first that converges. That is not a tuning knob — the acceptance
    # test is the KKT system itself, so a poor start can only cost time, never
    # correctness.
    all_members = vcat([prob.phases[k].members for k in act_ph]...)
    gh_fit = -(prob.g[all_members] .+ h0[all_members])
    A_all = Matrix(@view prob.A[:, all_members])

    # Candidates are ordered from the most informed by the caller's guess to the
    # least, and the loop stops at the first that converges. A caller replaying a
    # trajectory hands over a composition that is nearly the answer, and the fits
    # built from it succeed immediately; the element-potential solve at the end is
    # the cold-start device and is then never even run. Putting it first cost a
    # warm-started replay three digits of element balance — 5.7e-10 against
    # 1.4e-11 — for no gain, because the fits it displaced were the better guess.
    y_starts = Vector{Vector{Float64}}()

    let wgt = [sqrt(max(x_buf[i], 1.0e-30)) for i in all_members]
        push!(y_starts, qr(transpose(A_all * Diagonal(wgt)), ColumnNorm()) \ (gh_fit .* wgt))
    end
    let order = sortperm([x_buf[i] for i in all_members]; rev = true),
            nfit = min(length(all_members), max(3 * m, m + 4))
        sel = order[1:nfit]
        push!(y_starts, qr(transpose(A_all[:, sel]), ColumnNorm()) \ gh_fit[sel])
    end
    push!(y_starts, qr(transpose(A_all), ColumnNorm()) \ gh_fit)

    # Last candidate: the concave dual of the ideal problem, solved globally. See
    # `_element_potential_start` — it is the only one of these that knows the
    # element budget exists, and the only one that does not depend on the guess.
    let y_ls = copy(y_starts[end])
        for k in degenerate
            y_ls[k] = DEGENERATE_POTENTIAL
        end
        push!(
            y_starts,
            _element_potential_start(prob.A, prob.g, bv, y_ls; dead = dead),
        )
    end

    for yk in y_starts
        for k in degenerate
            yk[k] = DEGENERATE_POTENTIAL
        end
    end

    # Attempts are ranked by the KKT error they reach, not by the order they were
    # tried in. Keeping the first unless a later one CONVERGES throws away a better
    # answer whenever none converges: a start that lands at 1e8 was returned in
    # preference to one at 20, because neither had crossed the tolerance.
    best = nothing
    for y0 in y_starts
        out = _dual_newton_attempt(
            prob, bv, y0, W, act_ph, refs, active, xB, x_buf, dead, degenerate, m, opts,
        )
        if best === nothing || out.kkt_error < best.kkt_error
            best = out
        end
        out.converged && break
    end
    return best
end

"""
    _dual_newton_attempt(...) -> (; x, y, q, active_phases, active, converged)

One run of the two-level Newton from a given set of multipliers. Called once per
starting point by [`dual_newton_solve`](@ref).
"""
function _dual_newton_attempt(
        prob, bv, y_init, W0, act_ph0, refs0, active0, xB0, x_buf, dead, degenerate, m, opts,
    )
    y = copy(y_init)
    W = [copy(w) for w in W0]
    act_ph = copy(act_ph0)
    refs = copy(refs0)
    active = copy(active0)
    xB = copy(xB0)
    nq = prob.nq
    q = copy(prob.q0)

    converged = false
    seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
    # Candidates whose admission was tried and left the stationarity block
    # unsatisfiable. Two bound-constrained variables both declared stationary
    # over-determine `y` whenever their formulas are dependent modulo the span of
    # the mixing phases: the least-squares step is then the best available and the
    # residual simply cannot reach zero. Observed on a cement at six hours as a
    # residual falling to 1.05, jumping to 18.9 on the admission, and stalling
    # there. Rejecting the candidate permanently keeps the outer loop finite and
    # lets the next one be tried.
    rejected = Set{Int}()
    drop_ph_prev = Int[]
    last_added = 0

    # The active-set search is a DESCENT method on the outer residual.
    #
    # Each move — admit the most violated candidate, release the most violated
    # incumbent — is a guess, and a guess that makes the residual worse has to be
    # undone, not built upon. Without that the search wanders: on an LC³
    # equilibrium the residual went 16.04 → 8.05 → 16.04 → 514 and stalled there,
    # having passed through and abandoned its best state. Accepting only
    # improvements makes the sequence of visited sets strictly decreasing in
    # residual, hence finite, and the answer is the best set found rather than the
    # last one tried.
    best_res = Inf
    best_state = nothing

    for _ in 1:(opts.max_active_updates)
        nph = length(act_ph)
        na = length(active)
        v = vcat([log(r) for r in refs], y, xB, q)
        inner_ok = false

        for _ in 1:(opts.maxit)
            R = _outer_residual(prob, v, W, act_ph, active, bv, x_buf; dead, degenerate)
            res = maximum(abs, R)
            if opts.verbose
                # Split by block: the three carry different units — log-activities
                # for the phase and stationarity rows, moles for the balance — and
                # a single maximum says nothing about which of them is stuck.
                rp = nph == 0 ? 0.0 : maximum(abs, @view R[1:nph])
                rb = maximum(abs, @view R[(nph + 1):(nph + m)])
                rs = na == 0 ? 0.0 : maximum(abs, @view R[(nph + m + 1):(nph + m + na)])
                @info "dual-newton" res res_phase = rp res_balance = rb res_stat = rs nph na
            end
            if res <= opts.tol
                inner_ok = true
                break
            end

            N = length(v)
            J = zeros(N, N)
            W_ref = [copy(w) for w in W]
            for kk in 1:N
                # The difference step is scaled to the RESIDUAL's sensitivity, not
                # to the size of the unknown, and for the multipliers those are not
                # the same thing at all.
                #
                # The residual depends on `y` only through `u = −Aᵀy`, and
                # exponentially so. A relative step `1e-5·|y_k|` looks harmless
                # until one notices what `|y_k|` is: the `gᵢ` are Gibbs energies of
                # formation FROM THE ELEMENTS, of order 10²–10³ in RT units, and
                # `y` carries that offset — an offset that says nothing about the
                # problem. The step came out near `5e-3`, which moves `u` by
                # `|A|·5e-3 ≈ 0.15` and every `exp(uᵢ − gᵢ)` by some sixteen
                # percent. That is a secant across a wide interval, not a
                # derivative: on an LC³ equilibrium the resulting direction reduced
                # the residual at no step length, all forty backtracks were
                # refused, and the residual sat at 30.94 while moving by 1e-11 per
                # iteration.
                #
                # The scale over which the residual varies with `y_k` is
                # `1/maxᵢ|A[k,i]|`, so that is what sets the step; `√eps` in `u`
                # is the usual forward-difference compromise between truncation and
                # roundoff. `ln x_ref` and the bounded amounts are O(1) in their own
                # right and keep the relative rule.
                hk = if kk > nph && kk <= nph + m
                    krow = kk - nph
                    1.0e-8 / max(1.0, maximum(abs, @view prob.A[krow, :]))
                elseif nq > 0 && kk > nph + m + na
                    # A temperature in kelvin and a reaction extent in moles do
                    # not share a scale, and the caller is the only one who knows
                    # theirs — hence `qscale`. The relative rule below would give
                    # 3e-3 K on a 298 K unknown, which moves every `exp(uᵢ − gᵢ)`
                    # by a visible amount and secants across it.
                    1.0e-6 * prob.qscale[kk - (nph + m + na)]
                else
                    1.0e-5 * max(abs(v[kk]), 1.0)
                end
                vp = copy(v)
                vp[kk] += hk
                W_t = [copy(w) for w in W_ref]
                Rp = _outer_residual(prob, vp, W_t, act_ph, active, bv, x_buf; dead, degenerate)
                J[:, kk] .= (Rp .- R) ./ hk
            end
            all(isfinite, J) || break

            δ = qr(J, ColumnNorm()) \ (-R)

            α = 1.0
            for a in 1:nph
                abs(δ[a]) > 1.0 && (α = min(α, 1.0 / abs(δ[a])))
            end
            for j in 1:na
                dj = δ[nph + m + j]
                if dj < 0 && xB[j] + α * dj < 0
                    α = min(α, -0.9 * xB[j] / dj)
                end
            end

            accepted = false
            for _ in 1:40
                v_t = v .+ α .* δ
                W_t = [copy(w) for w in W_ref]
                R_t = _outer_residual(prob, v_t, W_t, act_ph, active, bv, x_buf; dead, degenerate)
                if maximum(abs, R_t) < res
                    v = v_t
                    for kk in eachindex(W)
                        W[kk] .= W_t[kk]
                    end
                    accepted = true
                    break
                end
                α /= 2
            end
            accepted || break

            xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
            # A pinned variable may legitimately pass through small values, so the
            # early break looks only at the ones an active set actually decides.
            let free = [j for j in 1:na if !(active[j] in prob.always_active)]
                !isempty(free) && minimum(@view xB[free]) < opts.si_tol && break
            end
            nph > 0 && minimum(@view v[1:nph]) < log(opts.si_tol) && break
        end

        refs = [exp(v[a]) for a in 1:nph]
        y = v[(nph + 1):(nph + m)]
        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
        q = nq == 0 ? Float64[] : v[(nph + m + na + 1):(nph + m + na + nq)]

        u = -(transpose(prob.A) * y)
        hv = current_h(prob, x_buf, q)
        si = u .- (current_g(prob, q) .+ hv)

        # Record this set if it is the best seen, measured by the KKT error of the
        # WHOLE problem — not by the residual of the subproblem this set defines.
        #
        # The distinction decides the search. An active set that omits a phase the
        # solution needs still solves its own equations exactly: the outer residual
        # goes to 1e-12 while the omitted phase sits absent and supersaturated by
        # ten RT. Ranking states on that residual therefore rewards leaving phases
        # out. The measure below is the one the certificate applies — stationarity,
        # element balance, and the worst violation among the phases held absent —
        # so descending it descends the distance to a KKT point.
        let res_outer = maximum(
                abs, _outer_residual(prob, v, W, act_ph, active, bv, x_buf; dead, degenerate),
            )
            viol = 0.0
            for i in prob.idx_bounded
                (i in active || i in dead) && continue
                viol = max(viol, si[i])
            end
            for k in eachindex(prob.phases)
                k in act_ph && continue
                all(i in dead for i in prob.phases[k].members) && continue
                viol = max(viol, _phase_tangent(prob, k, u, x_buf))
            end
            kkt_err = max(res_outer, viol)
            opts.verbose && @info "active-set round" kkt_err res_outer viol nph na inner_ok
            if kkt_err < best_res * (1 - 1.0e-9)
                best_res = kkt_err
                # `v` carries refs, `y` and the bounded amounts together, so the
                # state is that vector plus the sets it is indexed by.
                best_state = (copy(act_ph), copy(active), copy(v), [copy(w) for w in W])
            end
        end

        # ── bound-constrained variables ──
        # Complementarity, not just the amount.
        #
        # For a bound-constrained variable the KKT conditions are `xᵢ ≥ 0`,
        # `sᵢ ≤ 0` and `xᵢ sᵢ = 0` with `sᵢ = uᵢ − gᵢ − hᵢ`. Testing only
        # `xᵢ → 0` catches one half: a variable held ACTIVE while it is
        # UNDERSATURATED — `sᵢ` strictly negative — violates complementarity just as
        # plainly, and no amount of Newton iteration will repair it, because its own
        # equation `sᵢ = 0` is the one that cannot hold. It has to leave.
        #
        # Measured on an LC³ equilibrium at a quarter of full reaction, the search
        # settled with nothing supersaturated, the element balance at 4.5e-2 and a
        # stationarity residual of 9.86 carried entirely by such a phase: present,
        # and undersaturated by ten RT.
        # …and only on a point that solves the current subproblem. While the inner
        # Newton is still working, `sᵢ` on an active variable is a transient, not a
        # violation, and dropping on it removes phases that were on their way to
        # stationarity.
        drop = [
            j for j in eachindex(xB)
                if !(active[j] in prob.always_active) &&
                (xB[j] < opts.si_tol ||
                 (inner_ok && si[active[j]] < -max(opts.si_tol, opts.tol)))
        ]

        # An admission that does not converge used to be undone by rejecting the
        # entrant for good. That reads the failure backwards.
        #
        # The inner Newton breaks out the moment an active variable falls below
        # the bound (`minimum(xB) < si_tol`), and that is not a failed admission
        # — it is the departing variable announcing itself. The `drop` list above
        # already holds it, and letting the normal path remove it while the
        # entrant stays IS the active-set exchange. Treating it as the entrant's
        # fault is what left a cement with ettringite rejected and monosulphate
        # in its place: the two compete for the same sulfate, so admitting one
        # necessarily drives the other out, and the solver then converged —
        # beautifully, to 1e-12 — onto an assemblage in which ettringite was
        # absent and supersaturated by 14.8 RT.
        #
        # So the entrant is only reconsidered when the Newton failed with NOTHING
        # leaving, which is the genuine over-determination this guard was written
        # for: two bound variables declared stationary whose formulas are
        # dependent modulo the mixing phases, where the residual cannot reach
        # zero at all.
        # A stalled Newton with nothing leaving and nothing newly admitted means the
        # active set itself cannot be satisfied, and until now the loop had no way
        # out of that: `drop` tests only the AMOUNTS, so a phase that is held
        # active while its stationarity `uᵢ = gᵢ` is unreachable stays for ever.
        # Measured on an LC³ equilibrium the stationarity residual sat at 12.5 with
        # the element balance at 0.02 — the least-squares step sacrificing the one
        # to hold the other, which is what an inconsistent system looks like.
        #
        # The variable to release is the one whose own equation carries the
        # residual: removing that equation is precisely what lets the rest be
        # satisfied, and `si[i] = uᵢ − gᵢ − hᵢ` IS the residual of active variable
        # `i`. This is the leaving rule of an active-set method — the entering rule
        # is the most violated candidate, the leaving rule the most violated
        # incumbent — and `seen` still bounds the search.
        if !inner_ok && isempty(drop) && last_added == 0 && !isempty(active)
            releasable = [j for j in eachindex(active)
                              if !(active[j] in prob.always_active)]
            worst = isempty(releasable) ? 0 :
                releasable[argmax([abs(si[active[j]]) for j in releasable])]
            if worst != 0 && abs(si[active[worst]]) > opts.tol
                push!(rejected, active[worst])
                active = active[setdiff(eachindex(active), [worst])]
                xB = xB[setdiff(eachindex(xB), [worst])]
            end
        end

        if !inner_ok && isempty(drop) && last_added != 0 && last_added in active
            push!(rejected, last_added)
            j = findfirst(==(last_added), active)
            if j !== nothing
                active = active[setdiff(eachindex(active), [j])]
                xB = xB[setdiff(eachindex(xB), [j])]
            end
            last_added = 0
        end

        # A veto is only ever valid in the context that produced it. Once the
        # active set has moved for any other reason, a variable refused earlier
        # deserves another hearing — otherwise one unlucky ordering decides the
        # assemblage permanently.
        (isempty(drop) && isempty(drop_ph_prev)) || empty!(rejected)

        cand = [
            i for i in prob.idx_bounded
                if !(i in active) && !(i in dead) && !(i in rejected) && si[i] > opts.si_tol
        ]

        # What was vetoed is still measured. A rejected variable that remains
        # supersaturated is a violated KKT condition, and the run must not be
        # reported as converged just because the candidate list was filtered.
        veto_violation = any(
            i -> !(i in active) && !(i in dead) && si[i] > opts.si_tol, rejected,
        )

        # ── mixing phases ──
        drop_ph = [
            a for (a, k) in enumerate(act_ph)
                if !prob.phases[k].always_present && refs[a] < opts.si_tol
        ]
        cand_ph = [
            k for k in eachindex(prob.phases)
                if !(k in act_ph) && !all(i in dead for i in prob.phases[k].members) &&
                _phase_tangent(prob, k, u, x_buf, current_g(prob, q)) > opts.si_tol
        ]

        if isempty(drop) && isempty(cand) && isempty(drop_ph) && isempty(cand_ph)
            converged = inner_ok && !veto_violation
            break
        end

        if !isempty(drop)
            keep = setdiff(eachindex(xB), drop)
            active = active[keep]
            xB = xB[keep]
        end
        if !isempty(drop_ph)
            keep = setdiff(eachindex(act_ph), drop_ph)
            act_ph = act_ph[keep]
            refs = refs[keep]
        end
        if !isempty(cand)
            i_best = cand[argmax([si[i] for i in cand])]
            push!(active, i_best)
            push!(xB, 1.0e-9)
            # Admitting a variable may leave the active set unable to support a
            # solution at all, either because the stationarity capacity `m` is
            # already spent or because the entrant's composition is a combination
            # of those already active. Neither is a reason to refuse it — it is
            # violated, so it belongs — but one of the incumbents has to leave.
            #
            # Removal candidates are tried in order of increasing amount: the
            # smallest is the one closest to leaving on its own and the one whose
            # removal perturbs the primal least. Where the entrant is DEPENDENT on
            # the active set, only a variable it is dependent with restores the
            # rank, and trying them in turn finds it. This is a basis exchange, and
            # the choice among admissible incumbents is a heuristic — what is not
            # heuristic is that the set must satisfy the rank and capacity
            # conditions, since otherwise no `y` exists. Termination is unaffected:
            # `seen` records the active sets visited and there are finitely many.
            if !_active_set_supports_a_solution(prob, active, act_ph)
                for j in sort(1:(length(active) - 1); by = j -> xB[j])
                    trial = active[setdiff(eachindex(active), [j])]
                    if _active_set_supports_a_solution(prob, trial, act_ph)
                        active = trial
                        xB = xB[setdiff(eachindex(xB), [j])]
                        break
                    end
                end
            end
            # If nothing worked the entrant itself goes back out: the set it would
            # make is unsolvable whatever leaves.
            if _active_set_supports_a_solution(prob, active, act_ph)
                last_added = i_best
            else
                j = findfirst(==(i_best), active)
                if j !== nothing
                    active = active[setdiff(eachindex(active), [j])]
                    xB = xB[setdiff(eachindex(xB), [j])]
                end
                push!(rejected, i_best)
            end
        end
        if !isempty(cand_ph)
            k_best = cand_ph[argmax([_phase_tangent(prob, k, u, x_buf) for k in cand_ph])]
            push!(act_ph, k_best)
            push!(refs, 1.0e-9)
            # A mole-fraction phase also spends one unit of stationarity capacity.
            if !_active_set_supports_a_solution(prob, active, act_ph)
                for j in sort(eachindex(active); by = j -> xB[j])
                    trial = active[setdiff(eachindex(active), [j])]
                    if _active_set_supports_a_solution(prob, trial, act_ph)
                        active = trial
                        xB = xB[setdiff(eachindex(xB), [j])]
                        break
                    end
                end
            end
            if !_active_set_supports_a_solution(prob, active, act_ph)
                pop!(act_ph)
                pop!(refs)
            end
        end

        drop_ph_prev = drop_ph
        key = (sort(copy(act_ph)), sort(copy(active)))
        key in seen && break
        push!(seen, key)
    end

    # Return the best state the search passed through, not the last one tried.
    #
    # `v` already holds the solution of the inner Newton on that active set, so
    # there is nothing to re-solve: refs, `y` and the bounded amounts are read back
    # out of it and the residual is evaluated once to set `converged`.
    if best_state !== nothing
        act_ph, active, v, W = best_state
        nph = length(act_ph)
        na = length(active)
        refs = [exp(v[a]) for a in 1:nph]
        y = v[(nph + 1):(nph + m)]
        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]
        q = nq == 0 ? Float64[] : v[(nph + m + na + 1):(nph + m + na + nq)]
        converged = maximum(
            abs, _outer_residual(prob, v, W, act_ph, active, bv, x_buf; dead, degenerate),
        ) <= opts.tol
    end

    _invert_phases!(
        prob, W, y, refs, act_ph, active, xB, x_buf;
        dead = dead, g = current_g(prob, q), q = q,
    )
    x = copy(x_buf)
    for (j, i) in enumerate(active)
        x[i] = max(xB[j], 0.0)
    end

    return (;
        x = x, y = y, q = q, active_phases = act_ph, active = active,
        converged = converged, kkt_error = best_res,
    )
end

"""
    kkt_certificate(prob, x, b; floor = 1e-25) -> (; stationarity, feasibility,
                                                    worst_violation, n_interior,
                                                    n_forced_zero, optimal)

Check the KKT conditions at `x`, independently of how it was obtained. For a
convex problem they are sufficient, so `optimal = true` is a **proof**.

# What is checked, and on which variables

A variable is INTERIOR when `xᵢ > floor`. There the condition is the equality
`∇fᵢ + (Aᵀy)ᵢ = 0`, and `y` is obtained from those variables by least squares.
Below `floor` a variable is at its bound, where the condition is the INEQUALITY
`∇fᵢ + (Aᵀy)ᵢ ≥ 0`.

Getting that split wrong is not a detail: imposing the equality on a variable
held at `1e-16` whose stationarity value is `e⁻³⁰⁰` misstates `hᵢ` by 263 units,
and the check then reports a residual of 74 for a point solved to `5e-12`.

Variables carrying a component whose right-hand side has vanished are excluded
from both tests: they are zero by the CONSTRAINT, and the multiplier of a
component nobody supplies is determined by nothing.
"""
function kkt_certificate(
        prob::DualNewtonProblem, x::AbstractVector, b::AbstractVector;
        floor::Float64 = 1.0e-25, tol::Float64 = 1.0e-10, si_tol::Float64 = 1.0e-8,
        q = prob.q0,
    )
    xv = Vector{Float64}(x)
    bv = Vector{Float64}(b)
    gq = current_g(prob, q)
    ∇f = gq .+ current_h(prob, xv, q)

    degenerate = degenerate_components(prob.A, bv)
    dead = isempty(degenerate) ? Set{Int}() :
        Set(j for j in eachindex(xv) if any(abs(prob.A[k, j]) > 0 for k in degenerate))

    # WHICH TEST APPLIES TO WHICH VARIABLE.
    #
    # A member of a mixing phase is NEVER at its bound. Its `hᵢ` is a logarithm of
    # a fraction, so it diverges to `−∞` as the amount goes to zero: `xᵢ = 0` is
    # not attainable, and at the solution its condition is the EQUALITY
    # `∇fᵢ + (Aᵀy)ᵢ = 0`, however small it is. Only a bound-constrained variable —
    # a pure phase, `hᵢ ≡ 0` — can sit exactly at zero and obey the inequality.
    #
    # Confusing the two is not a detail. Applying the inequality `uᵢ ≤ ∇fᵢ` to a
    # phase member truncated at the numerical floor reads `uᵢ ≤ gᵢ − 700`, which no
    # finite multiplier satisfies, and the certificate then reports a
    # supersaturation of several hundred for a composition that is optimal. On a
    # cement without limestone that alone lost four of forty instants.
    #
    # A phase member BELOW the floor is excluded from both tests: it is not a
    # statement about the chemistry but the truncation of `exp(w)` at `w = −700`,
    # and its exact value would satisfy the equality.
    in_phase = Set{Int}()
    for ph in prob.phases, i in ph.members
        push!(in_phase, i)
    end

    interior = [i for i in eachindex(xv) if xv[i] > floor && !(i in dead)]
    at_bound = [
        i for i in eachindex(xv)
            if xv[i] <= floor && !(i in dead) && !(i in in_phase)
    ]

    Ai = Matrix(@view prob.A[:, interior])
    # Pivoted QR, not `\`: when the interior count equals the number of rows the
    # matrix is square and Julia reaches for LU, which throws on a rank-deficient
    # set — and rank deficiency is ordinary here.
    y = qr(transpose(Ai), ColumnNorm()) \ (-∇f[interior])

    stationarity = isempty(interior) ? 0.0 : maximum(abs, ∇f[interior] .+ transpose(Ai) * y)
    # The linear rows are `A x + Aq q − b`, and forgetting `Aq q` reports the
    # parameter itself as an infeasibility: on a kinetic step the residual came
    # out at exactly `Δξ`.
    resid = prob.nq == 0 ? (prob.A * xv .- bv) :
        (prob.A * xv .+ prob.Aq * collect(q) .- bv)
    feasibility = maximum(abs, resid)

    u = -(transpose(prob.A) * y)
    worst = isempty(at_bound) ? -Inf : maximum(u[i] - ∇f[i] for i in at_bound)

    # A mixing phase held ENTIRELY absent is tested by neither of the two above:
    # its members are excluded from `interior` (they are at the floor) and from
    # `at_bound` (they are phase members, whose condition is an equality). So a
    # composition that omits a solid solution which should have formed passed the
    # certificate unexamined — the one soundness hole the certificate had, and
    # exactly the case that matters for a cement, where the C-S-H is a mixing
    # phase. Michelsen's measure is the test such a phase requires.
    worst_phase = -Inf
    absent_phases = Int[]
    for (k, ph) in pairs(prob.phases)
        all(xv[i] <= floor || i in dead for i in ph.members) || continue
        all(i in dead for i in ph.members) && continue   # cannot exist at all
        push!(absent_phases, k)
        worst_phase = max(
            worst_phase, phase_tangent_measure(prob, k, u, xv; g = gq, q = q),
        )
    end
    worst_all = max(worst, worst_phase)

    return (;
        stationarity = stationarity, feasibility = feasibility,
        worst_violation = worst_all, worst_violation_bounded = worst,
        worst_violation_phase = worst_phase, absent_phases = absent_phases,
        n_interior = length(interior),
        n_forced_zero = length(dead),
        optimal = stationarity <= tol && feasibility <= tol && worst_all <= si_tol,
    )
end
