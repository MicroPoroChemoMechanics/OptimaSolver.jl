# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2020-2024 Allan Leal (original C++ Optima, https://github.com/reaktoro/optima)
# Copyright © 2026 Jean-François Barthélémy (Julia port)

# ── convergence.jl ─────────────────────────────────────────────────────────────
# KKT convergence criteria and iteration logging.
#
# Convergence is declared when the scaled KKT error drops below `tol`:
#
#   error = max(‖ex‖∞, ‖ew‖∞) < tol
#
# where ex = ∇f + Aᵀy - μ/s  (optimality residual)
#       ew = An - b            (feasibility residual)
#
# The barrier parameter μ is reduced only after the inner Newton loop has
# converged for the current μ, following the schedule:
#
#   μ_new = max(barrier_min, barrier_decay * μ)

"""
    is_converged(kkt, opts) -> Bool

Return `true` if the KKT residual is within tolerance.
"""
function is_converged(kkt::KKTResidual, opts::OptimaOptions)
    return kkt.error < opts.tol
end

"""
    should_reduce_barrier(kkt, μ, opts) -> Bool

Return `true` if the inner loop has converged sufficiently to reduce μ.

We use a relaxed inner tolerance: 10× `tol` (or the current μ if larger),
so that we tighten the barrier aggressively when far from the solution and
gently when close.
"""
function should_reduce_barrier(kkt::KKTResidual, μ, opts::OptimaOptions)
    # Reduce μ when the inner loop has converged to the current barrier problem.
    # Use max(tol, μ) so that at μ = barrier_min the threshold equals tol —
    # meaning is_converged will fire first, not this break condition.
    inner_tol = max(opts.tol, μ)
    return kkt.error < inner_tol
end

"""
    reduce_barrier(μ, opts) -> μ_new

Apply the barrier reduction schedule.

    μ_new = max(barrier_min, barrier_decay * μ)
"""
function reduce_barrier(μ, opts::OptimaOptions)
    return max(opts.barrier_min, opts.barrier_decay * μ)
end

"""
    log_iteration(iter, μ, kkt, α; verbose)

Print a one-line iteration summary when `verbose = true`.
"""
function log_iteration(iter::Int, μ, kkt::KKTResidual, α; verbose::Bool)
    if verbose
        println(
            "  iter ",
            lpad(iter, 4), " | μ = ", _fmt_sci(μ),
            " | err_opt = ", _fmt_sci(kkt.error_opt),
            " | err_feas = ", _fmt_sci(kkt.error_feas),
            " | α = ", round(α; digits = 4),
        )
    end
    return nothing
end

"""
    log_final(state, opts)

Print a convergence summary when `verbose = true`.
"""
function log_final(state::OptimaState, opts::OptimaOptions)
    if opts.verbose
        status = state.converged ? "CONVERGED" : "MAX_ITER"
        println(
            "  [OptimaSolver] ", status,
            " after ", state.iter, " iterations",
            " | err = ", _fmt_sci(max(state.error_opt, state.error_feas)),
        )
    end
    return nothing
end

# ── Formatting helper ─────────────────────────────────────────────────────────
_fmt_sci(x) = string(round(Float64(real(x)); sigdigits = 3))
