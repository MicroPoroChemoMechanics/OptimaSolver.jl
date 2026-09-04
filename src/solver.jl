# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── solver.jl ──────────────────────────────────────────────────────────────────
# Main primal-dual interior-point iteration loop.
#
# Algorithm sketch (Wächter & Biegler 2006 / Optima-style):
#
#   1. Initialize (n, y) — from state if warm_start, else feasibility heuristic
#   2. Outer loop over barrier parameter μ:
#      a. Inner loop (Newton iterations):
#         i.   Evaluate KKT residual F(n, y; μ)
#         ii.  Check inner convergence (error < 10*tol or error < μ)
#         iii. Compute Hessian diagonal h
#         iv.  Compute Newton step (dn, dy) via Schur complement
#         v.   Fraction-to-boundary α_max; optionally reduce for unstable vars
#         vi.  Filter line search → accepted (α, n_new, y_new)
#         vii. Update (n, y); add to filter
#      b. Reduce μ: μ ← max(barrier_min, barrier_decay * μ)
#   3. Final KKT check; set state.converged
#
# Warm-start: if state already has n, y from a previous solve, we start there
# with μ initialized to barrier_init (may converge in 1–3 outer iterations).

"""
    solve!(state, prob, can, opts) -> OptimaState

Run the interior-point loop, mutating `state` in-place.

# Arguments
- `state`: `OptimaState` — initial iterate on entry, solution on exit
- `prob`:  `OptimaProblem`
- `can`:   pre-built `Canonicalizer` for `prob.A`
- `opts`:  `OptimaOptions`
"""
function solve!(
        state::OptimaState{T},
        prob::OptimaProblem{T},
        can::Canonicalizer{T},
        opts::OptimaOptions,
    ) where {T}
    ns = prob.ns
    m = prob.m

    # ── Workspace allocations ─────────────────────────────────────────────────
    ws = NewtonStep(ns, m, T)
    grad = zeros(T, ns)
    hf = zeros(T, ns)          # diagonal of ∇²f(n)
    g_pert_h = zeros(T, ns)   # scratch buffer for FD Hessian diagonal
    filter = LineSearchFilter(T)
    h = zeros(T, ns)

    n = state.n
    y = state.y
    μ = state.μ

    # ── Feasibility initialization (ensure An = b, n ≥ lb) ───────────────────
    #
    # Positivity FIRST, then feasibility. The reverse order — which is what this
    # did — has the clamp undo the projection it just performed, and on a system
    # where most candidate species sit at their bound the clamp restores enough
    # matter to leave the start visibly infeasible. `_initialise_feasible!` solves
    # for the basic amounts given the others, so it is exact and there is nothing
    # left to clamp afterwards.
    @inbounds for i in eachindex(n)
        n[i] = max(n[i], prob.lb[i] + eps(T))
    end
    ew0 = prob.A * n .- prob.b
    if maximum(abs, ew0) > sqrt(opts.tol)
        _initialise_feasible!(n, prob, can)
        opts.verbose && println(
            "  start | ‖An-b‖ ", _fmt_sci(maximum(abs, ew0)), " -> ",
            _fmt_sci(maximum(abs, prob.A * n .- prob.b)),
        )
    end

    state.iter = 0

    # The BEST iterate seen, not the last one, is what gets returned.
    #
    # `is_converged` rarely fires on a cement, so the solver's answer is whichever
    # point the iteration budget happened to end on — and that point can be far
    # worse than one it passed through. Measured on an LC³ equilibrium the run
    # reached `err_opt = 1.4e-4` with the element balance at 2.7e-15, then
    # destabilized while `μ` was tightened further and finished at `err_opt = 2.3`
    # with the balance at 8.8e-9. Returning the last point throws away the answer
    # in favor of a worse one, and the caller — a warm start, or a certificate —
    # has no way to know.
    n_best = copy(n)
    y_best = copy(y)
    err_best = T(Inf)
    μ_best = μ

    # Lexicographic, feasibility first.
    #
    # `kkt.error` is `max(error_opt, error_feas)` and on a chemical system the
    # optimality term dominates by orders of magnitude, so ranking on it alone
    # lets an iterate win on optimality while being WORSE on the element balance —
    # and the balance is a hard constraint, not something to trade. Measured on an
    # LC³ budget the returned point carried `‖An − b‖∞ = 2.6e-4` where the
    # iteration had been holding 1e-15 throughout.
    feas_best = T(Inf)
    opt_best = T(Inf)

    function _keep_best!(kkt, μ_now)
        feas_ok = kkt.error_feas <= max(feas_best, T(opts.tol))
        if feas_ok && kkt.error_opt < opt_best
            feas_best = T(kkt.error_feas)
            opt_best = T(kkt.error_opt)
            err_best = T(kkt.error)
            n_best .= n
            y_best .= y
            μ_best = μ_now
        end
        return nothing
    end

    # ── Outer loop: barrier reduction ─────────────────────────────────────────
    for _ in 1:opts.max_iter
        # The barrier subproblem does not have to be solved, only advanced. If it
        # stops advancing, reducing μ is what unlocks it — and refusing to is a
        # deadlock, not caution.
        #
        # The fraction-to-boundary rule caps each step at roughly `s/|δn|`, so a
        # phase on its way out approaches its bound geometrically and the error
        # settles a little ABOVE `κ_ε μ` without crossing it. Measured on an LC³
        # equilibrium the error then drifted upward — 3.17e-4 to 3.36e-4 over a
        # hundred iterations — with `μ` frozen at 1e-5 and `α` shrinking, and the
        # solve stopped on `MaxIters` having tightened nothing. Ipopt requires
        # only `E_μ ≤ κ_ε μ` before reducing; a subproblem that has stalled well
        # short of that has nothing more to give at this `μ`.
        # The best iterate is tracked WITHIN a barrier level, never across them.
        #
        # `err_opt` is the complementarity measure `max |sᵢ gᵢ − μ|`, which is a
        # function of `μ`: an early point at `μ = 1e-4` can carry a smaller number
        # than a far better one at `μ = 1e-10`, simply because the target it is
        # measured against is looser. Comparing across levels therefore hands back
        # a point from the beginning of the run — on calcite in pure water it
        # returned the lifted starting guess, portlandite still sitting at 4.8e-6,
        # and the dual solve that warm-started from it admitted portlandite as a
        # phase and diverged.
        best_err = T(Inf)
        feas_best = T(Inf)
        opt_best = T(Inf)
        n_best .= n
        y_best .= y
        μ_best = μ
        stalled = 0

        # ── Inner loop: Newton iterations for fixed μ ─────────────────────────
        for _ in 1:opts.max_iter
            state.iter += 1

            # Gradient
            eval_gradient!(grad, prob, n)

            # KKT residual
            kkt = kkt_residual(prob, n, y, grad, μ)
            state.error_opt = kkt.error_opt
            state.error_feas = kkt.error_feas
            state.error_feas_abs = kkt.error_feas_abs
            _keep_best!(kkt, μ)

            # Global convergence check
            if is_converged(kkt, opts)
                state.converged = true
                state.n .= n
                state.y .= y
                state.μ = μ
                log_final(state, opts)
                return state
            end

            # Inner convergence: ready to reduce μ
            if should_reduce_barrier(kkt, μ, opts)
                break
            end

            # …or the subproblem has stopped advancing at this μ.
            if kkt.error < best_err * (one(T) - T(1.0e-3))
                best_err = kkt.error
                stalled = 0
            else
                stalled += 1
                stalled >= opts.barrier_stall_iters && break
            end

            # Hessian diagonal of f(n) — diagonal of ∂²f/∂n².
            # For ideal solution (G = Σ nᵢ(μᵢ⁰ + ln nᵢ)), the diagonal is 1/nᵢ.
            # For mixed solid/aqueous problems, pure solids have ∂²G/∂nᵢ² = 0
            # (constant activity); using 1/nᵢ there inflates the Hessian ~10⁵×,
            # making the Newton step negligible and causing linear (not quadratic)
            # convergence. The FD option computes the true diagonal via one
            # gradient evaluation per species.
            # The caller may hand over the exact diagonal through the parameters,
            # the way `A` and `b` are handed over. Both fallbacks below are only
            # approximations of `∂²f/∂nᵢ²`, and on a chemical system the error is
            # not benign: understating the curvature on a trace species makes the
            # line search reject its step, and the iteration stalls on a point
            # that is not the minimum.
            if prob.p isa NamedTuple && haskey(prob.p, :hdiag)
                prob.p.hdiag(hf, n)
            elseif opts.use_fd_hessian
                ε_h = sqrt(eps(T))
                for i in 1:ns
                    Δi = ε_h * max(one(T), abs(n[i]))
                    n[i] += Δi
                    eval_gradient!(g_pert_h, prob, n)
                    hf[i] = max((g_pert_h[i] - grad[i]) / Δi, zero(T))
                    n[i] -= Δi
                end
            else
                hf .= gibbs_hessian_diag(n)
            end
            h .= hessian_diagonal(prob, n, μ, hf)

            # Newton step
            dn, dy = opts.nullspace_step ?
                compute_step_nullspace!(ws, can, h, kkt.ex, kkt.ew) :
                compute_step!(ws, can, h, kkt.ex, kkt.ew)

            # Fraction-to-boundary step limit
            α_max = clamp_step(n, prob.lb, dn)

            # Variable stability: cap step for near-bound variables
            _, ju = classify_variables(n, prob.lb, kkt.ex)
            reduced_step_for_unstable!(dn, ju, n, prob.lb)

            # Clipping individual components takes the step OUT of the null space
            # of `A`, and with it the one property the null-space branch exists to
            # provide: that `A(n + α dn) = An` for every `α`, so feasibility once
            # attained is never lost. Silently, because nothing downstream checks.
            # On an LC³ equilibrium the start was feasible to 3e-10 and the solve
            # finished at `‖An − b‖∞ = 4.5e-3`, drifting a little at every
            # iteration where a variable near its bound was clipped.
            #
            # Re-projecting onto `{A d = 0}` restores it exactly and keeps what the
            # clip was for. The projection is the minimum-norm correction, so it
            # changes the clipped components as little as it can.
            if opts.nullspace_step
                r_dn = prob.A * dn
                if maximum(abs, r_dn) > eps(T) * max(one(T), maximum(abs, dn))
                    dn .-= can.A' * (
                        LinearAlgebra.qr(
                            can.A * can.A', LinearAlgebra.ColumnNorm(),
                        ) \ r_dn
                    )
                end
            end

            # Line search with filter
            f_val = prob.f(n, prob.p)
            α, n_new, y_new, f_new = line_search(
                prob, n, y, dn, dy, f_val, grad, μ, opts;
                filter = filter, α_max = α_max,
            )

            # Bookkeeping: only add to filter during infeasible phase
            θ_new = sum(abs, prob.A * n_new .- prob.b)
            θ_curr_now = sum(abs, prob.A * n .- prob.b)
            if θ_curr_now > sqrt(eps(T))
                add_to_filter!(filter, T(θ_new), T(real(f_new)))
            end

            log_iteration(
                state.iter, μ, kkt, α;
                verbose = opts.verbose, α_max = α_max, dn = dn, dy = dy,
            )

            # If the line search returned a negligible step, the Newton direction
            # gives no progress at this barrier level — break to force barrier
            # reduction rather than spinning through max_iter useless iterations.
            if α <= T(1.0e-10)
                break
            end

            n .= n_new
            y .= y_new

            if state.iter >= opts.max_iter
                # The best iterate replaces the last one only when it is
                # MATERIALLY better. This safeguard exists for the run that
                # destabilizes — on an LC³ equilibrium the error reached 1.4e-4
                # and finished at 2.3, four orders worse — not to second-guess a
                # healthy solve, where best and last differ by rounding and
                # swapping them costs the caller the last few digits it earned.
                if err_best < T(0.1) * kkt.error
                    n .= n_best
                    y .= y_best
                    μ = μ_best
                end
                state.n .= n
                state.y .= y
                state.μ = μ
                eval_gradient!(grad, prob, n)
                kkt_f = kkt_residual(prob, n, y, grad, μ)
                state.error_opt = kkt_f.error_opt
                state.error_feas = kkt_f.error_feas
                state.error_feas_abs = kkt_f.error_feas_abs
                state.converged = is_converged(kkt_f, opts)
                log_final(state, opts)
                return state
            end
        end  # inner

        # Reduce barrier.
        #
        # `reduce_barrier` clamps at `barrier_min`, so `μ < barrier_min` never holds
        # and cannot end the outer loop: the test below asks instead whether the
        # barrier was ALREADY at its floor. If it was, the inner loop has just
        # exited without `is_converged` firing, no further reduction is available,
        # and re-entering the inner loop only re-evaluates the same residual and
        # breaks again on `should_reduce_barrier` — without taking a single step, so
        # not one of those iterations can help. Measured, a warm-started
        # three-species solve reported 312 iterations for work that had finished at
        # 27, and a cement solve spent the same 285 wasted evaluations per call.
        μ <= opts.barrier_min && break
        μ = reduce_barrier(μ, opts)
    end  # outer

    # Final check. The best iterate replaces the last one only when it is
    # materially better — see the note at the max-iteration exit above.
    eval_gradient!(grad, prob, n)
    kkt = kkt_residual(prob, n, y, grad, μ)
    if err_best < T(0.1) * kkt.error
        n .= n_best
        y .= y_best
        μ = μ_best
        eval_gradient!(grad, prob, n)
        kkt = kkt_residual(prob, n, y, grad, μ)
    end
    state.error_opt = kkt.error_opt
    state.error_feas = kkt.error_feas
    state.error_feas_abs = kkt.error_feas_abs
    state.converged = is_converged(kkt, opts)
    state.n .= n
    state.y .= y
    state.μ = μ
    log_final(state, opts)
    return state
end

# ── Public solve interface ────────────────────────────────────────────────────

"""
    solve(prob, opts; u0, y0) -> OptimaResult

Solve the Gibbs minimization problem `prob` and return an `OptimaResult`.

# Arguments
- `prob`:  `OptimaProblem`
- `opts`:  `OptimaOptions` (keyword; defaults to `OptimaOptions()`)
- `u0`:    initial guess for n (keyword; defaults to b/m spread)
- `y0`:    initial guess for y (keyword; defaults to zeros)

# Warm-start
Pass a previous `OptimaResult` as `u0 = prev_result` and the solver will
initialize from `prev_result.n` and `prev_result.y`.
"""
function solve(
        prob::OptimaProblem{T},
        opts::OptimaOptions = OptimaOptions();
        u0 = nothing,
        y0 = nothing,
    ) where {T}
    can = Canonicalizer(prob.A)
    state = _make_initial_state(prob, opts, u0, y0)
    solve!(state, prob, can, opts)
    return OptimaResult(state)
end

"""
    solve(prob, can, opts; u0, y0) -> OptimaResult

Variant that accepts a pre-built `Canonicalizer` (avoids recomputing QR when
`prob.A` is fixed across many solves, e.g. during a temperature scan).
"""
function solve(
        prob::OptimaProblem{T},
        can::Canonicalizer{T},
        opts::OptimaOptions = OptimaOptions();
        u0 = nothing,
        y0 = nothing,
    ) where {T}
    state = _make_initial_state(prob, opts, u0, y0)
    solve!(state, prob, can, opts)
    return OptimaResult(state)
end

# ── Internal helpers ──────────────────────────────────────────────────────────

"""
    _make_initial_state(prob, opts, u0, y0) -> OptimaState

Build the initial `OptimaState` from keyword arguments or sensible defaults.
"""
function _make_initial_state(prob::OptimaProblem{T}, opts::OptimaOptions, u0, y0) where {T}
    m = prob.m

    if u0 isa OptimaResult
        n0 = copy(u0.n)
        y0_vec = copy(u0.y)
    elseif u0 isa AbstractVector
        n0 = convert(Vector{T}, u0)
        y0_vec = y0 isa AbstractVector ? convert(Vector{T}, y0) : zeros(T, m)
    else
        # Default: distribute total mass evenly
        n0 = _default_initial_n(prob)
        y0_vec = zeros(T, m)
    end

    return OptimaState{T}(
        n0, y0_vec, T(opts.barrier_init), 0, false, T(Inf), T(Inf), T(Inf),
    )
end

"""
    _default_initial_n(prob) -> Vector

Simple initial guess: nᵢ = bⱼ / ∑ Aⱼₖ for the first element row that
involves each species, scaled so An ≈ b roughly.
"""
function _default_initial_n(prob::OptimaProblem{T}) where {T}
    n0 = fill(T(1.0e-3), prob.ns)
    # Try to satisfy An = b by distributing b evenly over the species
    for i in 1:(prob.m)
        row_sum = sum(abs, prob.A[i, :])
        if row_sum > zero(T)
            scale = prob.b[i] / row_sum
            for k in 1:(prob.ns)
                if prob.A[i, k] > zero(T)
                    n0[k] = max(n0[k], scale * prob.A[i, k])
                end
            end
        end
    end
    # Ensure strict positivity
    @inbounds for i in eachindex(n0)
        n0[i] = max(n0[i], prob.lb[i] + eps(T))
    end
    return n0
end

"""
    _nnls(E, f; tol) -> v

Lawson-Hanson non-negative least squares: `min ‖E v − f‖₂` subject to `v ≥ 0`.

Used to place the starting point exactly on `A n = b` with `n ≥ lb`. When such a
point exists the residual is zero, so this does not approximate feasibility, it
attains it — and it terminates finitely, because each outer round adds one index
to the passive set and the inner loop only ever releases indices that reached
zero.

Alternating projections onto the affine set and the box converge to the same place
in theory, and far too slowly here to be usable: on an LC³ equilibrium they went
from `‖An − b‖ = 0.86` to `6.8e-3` and then advanced by less than a tenth of a
percent per sweep. The solve that followed never left its starting point.
"""
function _nnls(E::AbstractMatrix{T}, f::AbstractVector{T}; tol::T = T(1.0e-13)) where {T}
    n = size(E, 2)
    v = zeros(T, n)
    passive = falses(n)
    w = E' * f
    for _ in 1:(4 * n)
        # Most promising index still held at zero.
        jbest, wbest = 0, tol
        @inbounds for j in 1:n
            if !passive[j] && w[j] > wbest
                jbest, wbest = j, w[j]
            end
        end
        jbest == 0 && break
        passive[jbest] = true

        for _ in 1:(4 * n)
            P = findall(passive)
            isempty(P) && break
            z = LinearAlgebra.qr(E[:, P], LinearAlgebra.ColumnNorm()) \ f
            if all(>(zero(T)), z)
                fill!(v, zero(T))
                @inbounds for (k, j) in enumerate(P)
                    v[j] = z[k]
                end
                break
            end
            # Fraction-to-zero along v → z, then release whatever reached zero.
            α = T(Inf)
            @inbounds for (k, j) in enumerate(P)
                if z[k] <= zero(T)
                    d = v[j] - z[k]
                    d > zero(T) && (α = min(α, v[j] / d))
                end
            end
            isfinite(α) || (α = zero(T))
            @inbounds for (k, j) in enumerate(P)
                v[j] += α * (z[k] - v[j])
            end
            released = false
            @inbounds for j in P
                if v[j] <= tol
                    v[j] = zero(T)
                    passive[j] = false
                    released = true
                end
            end
            released || break
        end
        w = E' * (f .- E * v)
    end
    return v
end

"""
    _initialise_feasible!(n, prob, can; maxit=100, tol=1e-12)

Make `n` satisfy `A n = b` with `n ≥ lb`, exactly if possible.

The basic/non-basic split is used first, which is how Optima does it and is the
only route that gives EXACT feasibility: hold the non-basic variables where they
are and solve `B n_b = b − N n_n` for the basic ones. `B` is square and
factorized already, so this is one triangular solve, and the residual it leaves
is machine precision rather than an iteration's worth. It works because the basis
is chosen by priority weight — the abundant species — so the budget it is asked to
absorb is small against what it holds.

Why exactness matters, and not marginally. The filter line search only bypasses
its filter when the current point is feasible; while it is not, acceptance needs
either a relative drop of `ls_alpha` in the constraint violation — impossible when
the fraction-to-boundary limit is already below `ls_alpha` — or an Armijo decrease
along a direction that is partly spent restoring feasibility and need not be a
descent direction at all. Measured on an LC³ equilibrium, the start carried
`‖An − b‖∞ = 6.8e-3`, every one of the forty trial steps was refused, and the
solve reported `MaxIters` on the point it started from. Once the start is
feasible the null-space step keeps `A dn = 0`, so feasibility is preserved for the
rest of the solve and never competes with optimality for the same step length.

If a basic amount comes out below its bound the split is abandoned and the point
is projected by alternating projections onto `{A n = b}` and `{n ≥ lb}`. Both sets
are convex and their intersection is non-empty whenever the budget is attainable,
so the alternation converges; it is slower and only approximate, which is why it
is the fallback and not the method.
"""
function _initialise_feasible!(
        n::AbstractVector{T}, prob::OptimaProblem{T}, can::Canonicalizer{T};
        maxit::Int = 100, tol::T = T(1.0e-12),
    ) where {T}
    jb, jn = can.jb, can.jn
    scale0 = max(one(T), maximum(abs, prob.b))

    # ── least disturbance first ──────────────────────────────────────────────
    #
    # The minimum-norm correction is the smallest change that reaches the affine
    # set, and for a caller replaying a trajectory that is the whole purpose of
    # handing over a guess. The routes below instead find *some* feasible point,
    # which is what a cold start needs and precisely what a warm start must not be
    # given: rebuilding the point destroys the correlation between neighboring
    # solves. So the alternating projection is tried FIRST, whatever the starting
    # residual, and the routes below are reached only when it does not get there:
    # they answer "find some feasible point", which is the cold-start question,
    # and giving that answer to a caller who supplied a good guess throws the
    # guess away.
    AAT0 = prob.A * prob.A'
    dmax = one(T)
    @inbounds for i in axes(AAT0, 1)
        dmax = max(dmax, AAT0[i, i])
    end
    @inbounds for i in axes(AAT0, 1)
        AAT0[i, i] += dmax * T(1.0e-14)
    end
    for _ in 1:50
        ew = prob.A * n .- prob.b
        maximum(abs, ew) <= tol && return n
        n .-= prob.A' * (AAT0 \ ew)
        @inbounds for i in eachindex(n)
            n[i] = max(n[i], prob.lb[i] + eps(T))
        end
    end
    maximum(abs, prob.A * n .- prob.b) <= sqrt(eps(T)) * scale0 && return n

    # Strict interiority is part of the contract: this point is handed to a
    # barrier method, and a variable sitting exactly on its bound gives `α = 0`
    # on the first negative step component.
    @inbounds for i in eachindex(n)
        n[i] = max(n[i], prob.lb[i] + eps(T))
    end

    # Everything below RECONSTRUCTS the composition, which is what a cold start
    # needs. The least-disturbance route above has already been tried and did not
    # reach the affine set, so trying it a second time — as an earlier version did,
    # under a `1e-2` threshold — returns whatever it stalled at and never gets
    # here. On an LC³ budget that stall was `2.6e-4`, which is enough to deadlock
    # the filter line search on the first iteration.
    rhs = collect(prob.b)
    @inbounds for j in jn
        for i in eachindex(rhs)
            rhs[i] -= prob.A[i, j] * n[j]
        end
    end
    nb = can.BLU \ rhs
    if all(i -> nb[i] > prob.lb[jb[i]], eachindex(nb))
        @inbounds for (i, j) in enumerate(jb)
            n[j] = nb[i]
        end
        return n
    end

    # ── exact route 2: non-negative least squares on the slacks ─────────────
    #
    # `v = n − lb ≥ 0` with `A v = b − A·lb`. When the budget is attainable — and
    # it is, since it came from a real composition — the NNLS residual is zero, so
    # this lands ON the affine set rather than near it.
    #
    # NNLS returns a solution supported on at most `rank(A)` variables, i.e. with
    # the others at exactly their bound and zero slack. That is unusable as a
    # barrier start: the fraction-to-boundary rule would give `α = 0` on the first
    # negative step component. They are therefore lifted to the slack the barrier
    # itself would give them — at barrier level `μ` a variable held at its bound
    # settles at `s = μ/(∇f)ᵢ`, and with chemical potentials of order 10²–10³ in
    # RT units that is around 1e-6 for the initial `μ = 1e-4`. Lifting a hundred
    # variables by 1e-6 adds 1e-4 mol against a budget of a few moles, and that
    # perturbation is then removed exactly, on the support, by a minimum-norm
    # correction — so the point stays feasible to machine precision AND strictly
    # interior.
    # `v = n − lb ≥ δ` with `A v = b − A·lb`. When the budget is attainable — and
    # it is, since it came from a real composition — the NNLS residual is zero, so
    # this lands ON the affine set rather than near it.
    #
    # The margin `δ` is part of the SAME problem, not a correction applied
    # afterwards. A barrier method cannot be started from a point with zero slack:
    # the fraction-to-boundary rule then gives `α = 0` on the first negative step
    # component. Substituting `v = δ + w` and solving `A w = f − A·δ` for `w ≥ 0`
    # gives a point that is strictly interior AND exactly feasible, both by
    # construction.
    #
    # Lifting after the fact does not work, and the reason is worth recording:
    # NNLS returns a solution supported on at most `rank(A)` variables, and on an
    # LC³ budget that was 10 of 12. Correcting the drift `A·δ` from within that
    # support can only remove its component in `range(A_S)`; the remaining two
    # dimensions stayed, and the start carried `‖An − b‖∞ = 2.6e-4` — enough to
    # deadlock the filter line search on the very first iteration.
    scale = max(one(T), maximum(abs, prob.b))
    δ = T(1.0e-6) * scale
    f_rhs = prob.b .- prob.A * (prob.lb .+ δ)
    w = _nnls(prob.A, f_rhs)
    if maximum(abs, prob.A * w .- f_rhs) <= sqrt(eps(T)) * scale
        n .= prob.lb .+ δ .+ w
        return n
    end

    # ── fallback: alternating projections ───────────────────────────────────
    return _project_alternating!(n, prob; maxit = maxit, tol = tol)
end

"""
    _project_alternating!(n, prob; maxit, tol)

Alternate the projection onto `{A n = b}` with the projection onto `{n ≥ lb}`.

Both sets are convex and their intersection is non-empty whenever the budget is
attainable, so the alternation converges — linearly, and on a cement far too
slowly to be relied on from a cold start, which is why it is reached only when the
point is already close or when the exact routes have failed.
"""
function _project_alternating!(
        n::AbstractVector{T}, prob::OptimaProblem{T}; maxit::Int, tol::T,
    ) where {T}
    AAT = prob.A * prob.A'
    diag_max = one(T)
    @inbounds for i in axes(AAT, 1)
        diag_max = max(diag_max, AAT[i, i])
    end
    @inbounds for i in axes(AAT, 1)
        AAT[i, i] += diag_max * T(1.0e-14)
    end
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(AAT); check = false)
    solve_aat = LinearAlgebra.issuccess(F) ? (r -> F \ r) : (r -> AAT \ r)

    prev = T(Inf)
    for _ in 1:maxit
        # The box projection comes FIRST. Measuring the affine residual before
        # clamping let the loop return on its very first test a point that
        # satisfies `A n = b` and still has entries below `lb` — the one thing
        # this routine exists to rule out. That point then reaches the barrier,
        # where `log(n)` of a non-positive amount is `NaN`, and the solve is
        # lost with nothing in the trace pointing back here.
        @inbounds for i in eachindex(n)
            n[i] = max(n[i], prob.lb[i] + eps(T))
        end
        ew = prob.A * n .- prob.b
        res = maximum(abs, ew)
        res <= tol && break
        res > prev * T(0.999) && break        # stalled: the sets barely overlap here
        prev = res
        n .-= prob.A' * solve_aat(ew)
        @inbounds for i in eachindex(n)
            n[i] = max(n[i], prob.lb[i] + eps(T))
        end
    end
    return n
end
