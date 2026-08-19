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

# The two kinds of variable

The distinction is not cosmetic. A pure phase satisfies `gᵢ + hᵢ = uᵢ` when
present and `≥` when absent, and it can be **exactly** zero. A member of a mixing
phase cannot: its activity goes to `−∞` as its fraction goes to zero, so it is
never exactly absent while the phase exists. The active set for a mixing phase is
therefore over the PHASE, and the criterion is a tangent-plane test rather than a
sign of a saturation index.
"""
struct DualNewtonProblem{T <: Real, H}
    A::Matrix{T}
    g::Vector{T}
    h::H
    phases::Vector{SolutionPhase}
    idx_bounded::Vector{Int}
    params::Any
end

function DualNewtonProblem(
        A::AbstractMatrix, g::AbstractVector, h;
        phases::AbstractVector{SolutionPhase},
        idx_bounded::AbstractVector{Int} = Int[],
        params = nothing,
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

    return DualNewtonProblem(
        Matrix{Float64}(A), Vector{Float64}(g), h,
        collect(SolutionPhase, phases), collect(Int, idx_bounded), params,
    )
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
function _mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead)
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
        d[j] = u[i] - prob.g[i] - lnγ
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

function _invert_phases!(prob, W, y, refs, act_ph, active, xB, x_buf; dead = Set{Int}())
    u = -(transpose(prob.A) * y)

    for _ in 1:200
        _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)
        hv = prob.h(x_buf, prob.params)
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
                d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, N, dead)
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
                    r = (u[i] - prob.g[i]) - hv[i]
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

The outer residual at `v = [ln x_ref(φ) for each active phase; y; x_B]`:

  - stationarity of each active phase's reference — one equation per phase, which
    is what fixes that phase's total amount;
  - the equality constraints `A x − b` — `m` equations;
  - stationarity of the active bound-constrained variables — `|P|` equations.

`|Φ| + m + |P|` equations in as many unknowns: some fifteen for a cement, against
forty-seven variables in the interior-point route.
"""
function _outer_residual(
        prob, v, W, act_ph, active, b, x_buf;
        dead = Set{Int}(), degenerate = Int[],
    )
    m = size(prob.A, 1)
    nph = length(act_ph)
    na = length(active)

    refs = [exp(v[a]) for a in 1:nph]
    y = v[(nph + 1):(nph + m)]
    xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]

    _invert_phases!(prob, W, y, refs, act_ph, active, xB, x_buf; dead = dead)

    hv = prob.h(x_buf, prob.params)
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
            d = _mole_fraction_exponents(prob, ph, u, hv, x_buf, refs[a], dead)
            push!(R_ref, _logsumexp(d))
        else
            ir = ph.members[ph.j_ref]
            push!(R_ref, prob.g[ir] + hv[ir] - u[ir])
        end
    end

    Rb = prob.A * x_buf .- b

    # The stationarity of an active bound-constrained variable is `gᵢ + hᵢ = uᵢ`,
    # not `gᵢ = uᵢ`. The two coincide only when `hᵢ = 0`, the case of a pure phase;
    # writing the general form costs nothing there and is the only correct one
    # otherwise.
    Rs = na == 0 ? Float64[] : prob.g[active] .+ hv[active] .- u[active]

    # For a vanished component the balance row carries no information — every
    # variable containing it is pinned at the floor, so the row reads 0 = 0 and
    # leaves `y_k` free, making the Jacobian singular. Replace it by the equation
    # that fixes the multiplier, chosen so the pinning is CONSISTENT with the
    # stationarity condition rather than merely imposed on it.
    # For a vanished component the balance row carries no information — every
    # variable containing it is pinned at the floor, so the row reads 0 = 0 and
    # leaves `y_k` free, making the Jacobian singular. Replace it by the equation
    # that fixes the multiplier, which keeps the system square.
    for k in degenerate
        Rb[k] = y[k] - DEGENERATE_POTENTIAL
    end

    return vcat(R_ref, Rb, Rs)
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
function _phase_tangent(prob, k, u, x_buf)
    ph = prob.phases[k]
    hv = prob.h(x_buf, prob.params)
    s = 0.0
    for (j, i) in enumerate(ph.members)
        # `γ` is read at the current state; for an ideal phase it is one, and for
        # a non-ideal one this is the usual first-order stability test.
        lnγ = 0.0
        s += exp(clamp(u[i] - prob.g[i] - lnγ, -700.0, 50.0))
    end
    return s - 1.0
end

# ── the solve ─────────────────────────────────────────────────────────────────

"""
    dual_newton_solve(prob, b, x0; opts) -> (; x, y, active_phases, active, converged)

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

    # A phase starts active if it is always present or if the guess holds it.
    act_ph = Int[
        k for (k, ph) in pairs(prob.phases)
            if ph.always_present || sum(n0[i] for i in ph.members) > 1.0e-9
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

    # Admit only bound-constrained variables the guess holds in QUANTITY: a trace
    # left over by an interior-point start is not evidence, and two variables both
    # declared stationary over-determine `y`.
    active = Int[i for i in prob.idx_bounded if n0[i] > 1.0e-6 && !(i in dead)]
    xB = Float64[n0[i] for i in active]

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
    h0 = prob.h(x_buf, prob.params)
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

    for yk in y_starts
        for k in degenerate
            yk[k] = DEGENERATE_POTENTIAL
        end
    end

    best = nothing
    for y0 in y_starts
        out = _dual_newton_attempt(
            prob, bv, y0, W, act_ph, refs, active, xB, x_buf, dead, degenerate, m, opts,
        )
        best === nothing && (best = out)
        if out.converged
            best = out
            break
        end
    end
    return best
end

"""
    _dual_newton_attempt(...) -> (; x, y, active_phases, active, converged)

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

    for _ in 1:(opts.max_active_updates)
        nph = length(act_ph)
        na = length(active)
        v = vcat([log(r) for r in refs], y, xB)
        inner_ok = false

        for _ in 1:(opts.maxit)
            R = _outer_residual(prob, v, W, act_ph, active, bv, x_buf; dead, degenerate)
            res = maximum(abs, R)
            opts.verbose && @info "dual-newton" res = res nphases = nph nactive = na
            if res <= opts.tol
                inner_ok = true
                break
            end

            N = length(v)
            J = zeros(N, N)
            W_ref = [copy(w) for w in W]
            for kk in 1:N
                hk = 1.0e-5 * max(abs(v[kk]), 1.0)
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
            na > 0 && minimum(xB) < opts.si_tol && break
            nph > 0 && minimum(@view v[1:nph]) < log(opts.si_tol) && break
        end

        refs = [exp(v[a]) for a in 1:nph]
        y = v[(nph + 1):(nph + m)]
        xB = na == 0 ? Float64[] : v[(nph + m + 1):(nph + m + na)]

        u = -(transpose(prob.A) * y)
        hv = prob.h(x_buf, prob.params)
        si = u .- (prob.g .+ hv)

        # ── bound-constrained variables ──
        drop = [j for j in eachindex(xB) if xB[j] < opts.si_tol]

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
                _phase_tangent(prob, k, u, x_buf) > opts.si_tol
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
            last_added = i_best
        end
        if !isempty(cand_ph)
            k_best = cand_ph[argmax([_phase_tangent(prob, k, u, x_buf) for k in cand_ph])]
            push!(act_ph, k_best)
            push!(refs, 1.0e-9)
        end

        drop_ph_prev = drop_ph
        key = (sort(copy(act_ph)), sort(copy(active)))
        key in seen && break
        push!(seen, key)
    end

    _invert_phases!(prob, W, y, refs, act_ph, active, xB, x_buf; dead = dead)
    x = copy(x_buf)
    for (j, i) in enumerate(active)
        x[i] = max(xB[j], 0.0)
    end

    return (; x = x, y = y, active_phases = act_ph, active = active, converged = converged)
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
    )
    xv = Vector{Float64}(x)
    bv = Vector{Float64}(b)
    ∇f = prob.g .+ prob.h(xv, prob.params)

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
    feasibility = maximum(abs, prob.A * xv .- bv)

    u = -(transpose(prob.A) * y)
    worst = isempty(at_bound) ? -Inf : maximum(u[i] - ∇f[i] for i in at_bound)

    return (;
        stationarity = stationarity, feasibility = feasibility,
        worst_violation = worst, n_interior = length(interior),
        n_forced_zero = length(dead),
        optimal = stationarity <= tol && feasibility <= tol && worst <= si_tol,
    )
end
