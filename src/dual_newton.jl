# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# ── dual_newton.jl ────────────────────────────────────────────────────────────
#
# Exact Newton on the KKT system, in the space of the equality multipliers.
#
# THE PROBLEM. Minimise `f(x) = Σᵢ xᵢ(gᵢ + hᵢ(x))` subject to `A x = b`, `x ≥ 0`,
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
# WHY IT IS WELL CONDITIONED. Parameterising the interior variables by `w = ln x`
# makes their positivity automatic, so no fraction-to-boundary rule applies to
# them — and that rule is what caps the step of the interior-point method in
# `solve!` at every single iteration. Only the bounded variables carry a bound,
# and they are handled by an active set, exactly and finitely.
#
# This is NOT the log reparameterisation of the objective that
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
    DualNewtonProblem(A, g, h; idx_log, idx_bounded, j_ref, params, ...)

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

  - `idx_log`: variables that are strictly positive at any solution, hence
    parameterised by `ln x` and inverted through their stationarity condition.
  - `idx_bounded`: variables that may vanish, handled by an active set.
  - `j_ref`: position **within `idx_log`** of a variable whose `hᵢ` is bounded
    above and therefore *cannot* be inverted — in a chemical system the solvent,
    whose activity is a mole fraction so that `ln a ≤ 0` always, and an arbitrary
    `y` may demand more, for which no finite `x` exists. Its stationarity is
    carried by the outer system instead, where the balance determines it. Pass
    `0` if no such variable exists.
  - `params`: passed through to `h`.
"""
struct DualNewtonProblem{T <: Real, H}
    A::Matrix{T}
    g::Vector{T}
    h::H
    idx_log::Vector{Int}
    idx_bounded::Vector{Int}
    j_ref::Int
    params::Any
end

function DualNewtonProblem(
        A::AbstractMatrix, g::AbstractVector, h;
        idx_log::AbstractVector{Int},
        idx_bounded::AbstractVector{Int},
        j_ref::Int = 0,
        params = nothing,
    )
    isempty(idx_log) && throw(
        ArgumentError(
            "`idx_log` is empty: the multipliers are determined by the stationarity " *
                "of the strictly positive variables, and without any there is nothing " *
                "to determine them."
        )
    )
    return DualNewtonProblem(
        Matrix{Float64}(A), Vector{Float64}(g), h,
        collect(Int, idx_log), collect(Int, idx_bounded), j_ref, params,
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
    _invert_stationarity!(prob, w, y, x_ref, active, xB, x_buf; dead) -> x_log

Solve `hᵢ(x) = uᵢ − gᵢ` for the log-parameterised variables at fixed multipliers
and fixed reference variable, where `u = −Aᵀy`.

For a variable whose `hᵢ` behaves like `ln xᵢ` plus a slowly varying part —
which is what an activity is — `∂hᵢ/∂wᵢ = 1`, so `w += r` is an exact Newton
step and the coupling is what makes the loop iterative rather than one-shot.

The floor is `exp(-700)`: a variable whose stationarity demands `hᵢ = −200` can
never reach it against a floor of `−80`, its residual stays at 120 for ever, and
the loop can never report convergence — which then makes the OUTER Jacobian
meaningless, since that Jacobian is derived on the assumption that these
conditions hold exactly.
"""
function _invert_stationarity!(prob, w, y, x_ref, active, xB, x_buf; dead = Set{Int}())
    u = -(transpose(prob.A) * y)
    target = u[prob.idx_log] .- prob.g[prob.idx_log]
    jr = prob.j_ref

    for _ in 1:200
        fill!(x_buf, 0.0)
        @inbounds for (j, i) in enumerate(prob.idx_log)
            x_buf[i] = j == jr ? x_ref : exp(w[j])
        end
        @inbounds for (j, i) in enumerate(active)
            x_buf[i] = xB[j]
        end
        hv = prob.h(x_buf, prob.params)[prob.idx_log]
        r = target .- hv
        jr > 0 && (r[jr] = 0.0)
        @inbounds for (j, i) in enumerate(prob.idx_log)
            i in dead && (r[j] = 0.0)
        end
        maximum(abs, r) <= 1.0e-14 && break
        @inbounds for j in eachindex(w)
            j == jr && continue
            prob.idx_log[j] in dead && continue
            w[j] = clamp(w[j] + clamp(r[j], -30.0, 30.0), -700.0, 20.0)
        end
    end

    return [j == jr ? x_ref : exp(w[j]) for j in eachindex(w)]
end

"""
    _outer_residual(prob, v, w, active, b, x_buf; dead, degenerate) -> (R, x_log)

The outer residual at `v = [ln x_ref; y; x_B]`, once the inner conditions hold:

  - stationarity of the reference variable — one equation;
  - the equality constraints `A x − b` — `m` equations;
  - stationarity of the variables in the active set — `|P|` equations.

`1 + m + |P|` equations in as many unknowns. For a cement partition that is some
fourteen numbers, against forty-seven variables in the interior-point route.
"""
function _outer_residual(prob, v, w, active, b, x_buf; dead = Set{Int}(), degenerate = Int[])
    m = size(prob.A, 1)
    na = length(active)
    x_ref = exp(v[1])
    y = v[2:(1 + m)]
    xB = na == 0 ? Float64[] : v[(2 + m):(1 + m + na)]

    x_log = _invert_stationarity!(prob, w, y, x_ref, active, xB, x_buf; dead = dead)

    hv = prob.h(x_buf, prob.params)
    u = -(transpose(prob.A) * y)
    jr = prob.j_ref
    ir = prob.idx_log[jr]

    R_ref = prob.g[ir] + hv[ir] - u[ir]

    Rb = Matrix(@view prob.A[:, prob.idx_log]) * x_log .- b
    na > 0 && (Rb .+= Matrix(@view prob.A[:, active]) * xB)

    Rs = na == 0 ? Float64[] : prob.g[active] .- u[active]

    # For a vanished component the balance row carries no information — every
    # variable containing it is pinned at the floor, so the row reads 0 = 0 and
    # leaves `y_k` free, making the Jacobian singular. Replace it by the equation
    # that fixes the multiplier, chosen so the pinning is CONSISTENT with the
    # stationarity condition rather than merely imposed on it. The system stays
    # square.
    for k in degenerate
        Rb[k] = y[k] - DEGENERATE_POTENTIAL
    end

    return vcat(R_ref, Rb, Rs), x_log
end

# ── the solve ─────────────────────────────────────────────────────────────────

"""
    dual_newton_solve(prob, b, x0; opts) -> (; x, y, active, converged)

Solve `prob` for the right-hand side `b`, starting from `x0`.

`x0` supplies a neighbourhood, not a feasible point: this is a Newton method, and
the intended use is to hand it the answer of an interior-point solve. Verify the
result with [`kkt_certificate`](@ref); for a convex problem that certificate is a
proof of **global** optimality and not a plausibility argument.

# Structure

Two levels. The inner one inverts the stationarity of the log-parameterised
variables at fixed multipliers and fixed reference variable — always solvable,
and a few sweeps. The outer is a Newton on `1 + m + |P|` unknowns against the
reference variable's own stationarity, the equality constraints, and the
stationarity of the active set.

The active set is updated **during** the Newton, one variable at a time and the
most violated first. Two variables both declared at their stationarity
over-determine `y` and their rows are then jointly infeasible; admitting a batch
feeds a cycle in which a variable is admitted, driven negative, dropped, and
readmitted. Visited active sets are recorded, which bounds the outer loop by the
number of subsets and therefore terminates.
"""
function dual_newton_solve(
        prob::DualNewtonProblem, b::AbstractVector, x0::AbstractVector;
        opts::DualNewtonOptions = DualNewtonOptions(),
    )
    bv = Vector{Float64}(b)
    n0 = Vector{Float64}(x0)
    m = size(prob.A, 1)
    jr = prob.j_ref
    x_buf = zeros(Float64, length(prob.g))

    degenerate = degenerate_components(prob.A, bv)
    dead = isempty(degenerate) ? Set{Int}() :
        Set(j for j in eachindex(prob.g) if any(abs(prob.A[k, j]) > 0 for k in degenerate))

    w = Float64[log(max(n0[i], 1.0e-30)) for i in prob.idx_log]
    for (j, i) in enumerate(prob.idx_log)
        i in dead && (w[j] = -700.0)
    end
    x_ref = jr > 0 ? max(n0[prob.idx_log[jr]], 1.0e-6) : 1.0

    # Admit only variables the guess holds in QUANTITY. A trace left over by an
    # interior-point start is not evidence, and two variables both declared at
    # their stationarity over-determine `y`. Anything genuinely violated is added
    # back by the update below.
    active = Int[i for i in prob.idx_bounded if n0[i] > 1.0e-6 && !(i in dead)]
    xB = Float64[n0[i] for i in active]

    # `y` from a WEIGHTED least-squares fit of the interior stationarity at the
    # guess. Stationarity holds for every interior variable at the solution, but
    # as a starting point the fit must be driven by the variables that carry
    # weight: an interior-point answer is accurate on the major ones and can be a
    # factor 1e5 out on a 1e-9 trace, and an unweighted fit lets those traces set
    # `y` — which sent the iteration to a solution with no solvent left in it.
    fill!(x_buf, 0.0)
    for (j, i) in enumerate(prob.idx_log)
        x_buf[i] = j == jr ? x_ref : exp(w[j])
    end
    for (j, i) in enumerate(active)
        x_buf[i] = xB[j]
    end
    h0 = prob.h(x_buf, prob.params)
    Al = Matrix(@view prob.A[:, prob.idx_log])
    wgt = [sqrt(max(x_buf[i], 1.0e-30)) for i in prob.idx_log]
    y = qr(transpose(Al * Diagonal(wgt)), ColumnNorm()) \
        (-(prob.g[prob.idx_log] .+ h0[prob.idx_log]) .* wgt)

    converged = false
    seen = Set{Vector{Int}}()

    for _ in 1:(opts.max_active_updates)
        na = length(active)
        v = vcat(log(x_ref), y, xB)
        inner_ok = false

        for _ in 1:(opts.maxit)
            R, _ = _outer_residual(prob, v, w, active, bv, x_buf; dead, degenerate)
            res = maximum(abs, R)
            opts.verbose && @info "dual-newton" res = res nactive = na
            if res <= opts.tol
                inner_ok = true
                break
            end

            # Numerical Jacobian on `1 + m + |P|` variables. The step `h` must
            # stand well clear of the inner tolerance, or the difference quotient
            # measures that tolerance rather than the derivative.
            N = length(v)
            J = zeros(N, N)
            w_ref = copy(w)
            for k in 1:N
                hk = 1.0e-5 * max(abs(v[k]), 1.0)
                vp = copy(v)
                vp[k] += hk
                w_t = copy(w_ref)
                Rp, _ = _outer_residual(prob, vp, w_t, active, bv, x_buf; dead, degenerate)
                J[:, k] .= (Rp .- R) ./ hk
            end
            all(isfinite, J) || break

            # Least squares, not `\`: the KKT matrix is singular whenever the
            # active set is linearly dependent on the components. That is a
            # statement about the problem, not an error, and the minimum-norm step
            # is the right response.
            δ = qr(J, ColumnNorm()) \ (-R)

            α = 1.0
            abs(δ[1]) > 1.0 && (α = min(α, 1.0 / abs(δ[1])))
            for j in 1:na
                dj = δ[1 + m + j]
                if dj < 0 && xB[j] + α * dj < 0
                    α = min(α, -0.9 * xB[j] / dj)
                end
            end

            accepted = false
            for _ in 1:40
                v_t = v .+ α .* δ
                w_t = copy(w_ref)
                R_t, _ = _outer_residual(prob, v_t, w_t, active, bv, x_buf; dead, degenerate)
                if maximum(abs, R_t) < res
                    v, w = v_t, w_t
                    accepted = true
                    break
                end
                α /= 2
            end
            accepted || break

            xB = na == 0 ? Float64[] : v[(2 + m):(1 + m + na)]
            # Leave the moment a variable turns out not to be there: the active
            # set is what must change, and finishing an infeasible subproblem
            # first spends the whole budget on it.
            na > 0 && minimum(xB) < opts.si_tol && break
        end

        x_ref = exp(v[1])
        y = v[2:(1 + m)]
        xB = na == 0 ? Float64[] : v[(2 + m):(1 + m + na)]

        u = -(transpose(prob.A) * y)
        si = u .- prob.g
        drop = [j for j in eachindex(xB) if xB[j] < opts.si_tol]
        candidates = [
            i for i in prob.idx_bounded
                if !(i in active) && !(i in dead) && si[i] > opts.si_tol
        ]

        if isempty(drop) && isempty(candidates)
            converged = inner_ok
            break
        end
        if !isempty(drop)
            keep = setdiff(eachindex(xB), drop)
            active = active[keep]
            xB = xB[keep]
        end
        if !isempty(candidates)
            i_best = candidates[argmax([si[i] for i in candidates])]
            push!(active, i_best)
            push!(xB, 1.0e-9)
        end

        key = sort(copy(active))
        key in seen && break
        push!(seen, key)
    end

    x_log = _invert_stationarity!(prob, w, y, x_ref, active, xB, x_buf; dead = dead)
    x = zeros(Float64, length(prob.g))
    for (j, i) in enumerate(prob.idx_log)
        x[i] = x_log[j]
    end
    for (j, i) in enumerate(active)
        x[i] = max(xB[j], 0.0)
    end

    return (; x = x, y = y, active = active, converged = converged)
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

    interior = [i for i in eachindex(xv) if xv[i] > floor && !(i in dead)]
    at_bound = [i for i in eachindex(xv) if xv[i] <= floor && !(i in dead)]

    Ai = Matrix(@view prob.A[:, interior])
    # Pivoted QR, not `\`: when the interior count equals the number of rows the
    # matrix is square and Julia reaches for LU, which throws on a rank-deficient
    # set — and rank deficiency is ordinary here.
    y = qr(transpose(Ai), ColumnNorm()) \ (-∇f[interior])

    stationarity = isempty(interior) ? 0.0 : maximum(abs, ∇f[interior] .+ transpose(Ai) * y)
    feasibility = maximum(abs, prob.A * xv .- bv)

    u = -(transpose(prob.A) * y)
    worst = isempty(at_bound) ? -Inf : maximum(u[i] - prob.g[i] for i in at_bound)

    return (;
        stationarity = stationarity, feasibility = feasibility,
        worst_violation = worst, n_interior = length(interior),
        n_forced_zero = length(dead),
        optimal = stationarity <= tol && feasibility <= tol && worst <= si_tol,
    )
end
