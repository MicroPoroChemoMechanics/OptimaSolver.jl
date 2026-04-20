# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── solver.jl ──────────────────────────────────────────────────────────────────
# Main primal-dual interior-point iteration loop.
#
# Algorithm sketch (Wächter & Biegler 2006 / Optima-style):
#
#   1. Initialise (n, y) — from state if warm_start, else feasibility heuristic
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
# with μ initialised to barrier_init (may converge in 1–3 outer iterations).

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

    # ── Feasibility initialisation (ensure An ≈ b) ───────────────────────────
    # If the initial n is badly infeasible, project it: add a feasibility step.
    ew0 = prob.A * n .- prob.b
    if maximum(abs, ew0) > sqrt(opts.tol)
        _initialise_feasible!(n, prob, can)
    end
    # Enforce strict positivity
    @inbounds for i in eachindex(n)
        n[i] = max(n[i], prob.lb[i] + eps(T))
    end

    state.iter = 0

    # ── Outer loop: barrier reduction ─────────────────────────────────────────
    for _ in 1:opts.max_iter
        # ── Inner loop: Newton iterations for fixed μ ─────────────────────────
        for _ in 1:opts.max_iter
            state.iter += 1

            # Gradient
            eval_gradient!(grad, prob, n)

            # KKT residual
            kkt = kkt_residual(prob, n, y, grad, μ)
            state.error_opt = kkt.error_opt
            state.error_feas = kkt.error_feas

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

            # Hessian diagonal of f(n) — diagonal of ∂²f/∂n².
            # For ideal solution (G = Σ nᵢ(μᵢ⁰ + ln nᵢ)), the diagonal is 1/nᵢ.
            # For mixed solid/aqueous problems, pure solids have ∂²G/∂nᵢ² = 0
            # (constant activity); using 1/nᵢ there inflates the Hessian ~10⁵×,
            # making the Newton step negligible and causing linear (not quadratic)
            # convergence. The FD option computes the true diagonal via one
            # gradient evaluation per species.
            if opts.use_fd_hessian
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
            dn, dy = compute_step!(ws, can, h, kkt.ex, kkt.ew)

            # Fraction-to-boundary step limit
            α_max = clamp_step(n, prob.lb, dn)

            # Variable stability: cap step for near-bound variables
            _, ju = classify_variables(n, prob.lb, kkt.ex)
            reduced_step_for_unstable!(dn, ju, n, prob.lb)

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

            log_iteration(state.iter, μ, kkt, α; verbose = opts.verbose)

            # If the line search returned a negligible step, the Newton direction
            # gives no progress at this barrier level — break to force barrier
            # reduction rather than spinning through max_iter useless iterations.
            if α <= T(1.0e-10)
                break
            end

            n .= n_new
            y .= y_new

            if state.iter >= opts.max_iter
                state.n .= n
                state.y .= y
                state.μ = μ
                log_final(state, opts)
                return state
            end
        end  # inner

        # Reduce barrier
        μ = reduce_barrier(μ, opts)

        if μ < opts.barrier_min
            break
        end
    end  # outer

    # Final check
    eval_gradient!(grad, prob, n)
    kkt = kkt_residual(prob, n, y, grad, μ)
    state.error_opt = kkt.error_opt
    state.error_feas = kkt.error_feas
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

Solve the Gibbs minimisation problem `prob` and return an `OptimaResult`.

# Arguments
- `prob`:  `OptimaProblem`
- `opts`:  `OptimaOptions` (keyword; defaults to `OptimaOptions()`)
- `u0`:    initial guess for n (keyword; defaults to b/m spread)
- `y0`:    initial guess for y (keyword; defaults to zeros)

# Warm-start
Pass a previous `OptimaResult` as `u0 = prev_result` and the solver will
initialise from `prev_result.n` and `prev_result.y`.
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

    return OptimaState{T}(n0, y0_vec, T(opts.barrier_init), 0, false, T(Inf), T(Inf))
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
    _initialise_feasible!(n, prob, can)

Project n onto the feasibility manifold An = b by a single Newton step
on the feasibility subproblem (holding the direction orthogonal to A fixed).

This corrects gross infeasibility in the initial guess without an inner loop.
"""
function _initialise_feasible!(n::AbstractVector{T}, prob::OptimaProblem{T}, ::Canonicalizer{T}) where {T}
    ew = prob.A * n .- prob.b
    # Solve A Aᵀ Δy = ew  →  Δn = -Aᵀ Δy  (minimum-norm correction).
    # Tikhonov regularisation (same as in compute_step!): when some
    # conservation rows involve only absent species (e.g. Na⁺ row at V=0),
    # the corresponding diagonal of A Aᵀ is near zero and the solve diverges.
    AAT = prob.A * prob.A'
    diag_max = one(T)
    @inbounds for i in axes(AAT, 1)
        diag_max = max(diag_max, AAT[i, i])
    end
    δ_reg = diag_max * T(1.0e-14)
    @inbounds for i in axes(AAT, 1)
        AAT[i, i] += δ_reg
    end
    Δy = AAT \ ew
    n .-= prob.A' * Δy
    # Re-clamp
    @inbounds for i in eachindex(n)
        n[i] = max(n[i], prob.lb[i] + eps(T))
    end
    return n
end
