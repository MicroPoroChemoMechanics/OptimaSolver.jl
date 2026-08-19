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
    # Judged at μ = 0, never at the current barrier level: see `kkt_residual`.
    return kkt.error_0 < opts.tol
end

"""
    should_reduce_barrier(kkt, μ, opts) -> Bool

Return `true` if the inner loop has converged sufficiently to reduce μ.

We use a relaxed inner tolerance: 10× `tol` (or the current μ if larger),
so that we tighten the barrier aggressively when far from the solution and
gently when close.
"""
function should_reduce_barrier(kkt::KKTResidual, μ, opts::OptimaOptions)
    # Reduce μ once the inner loop has solved the CURRENT barrier problem, i.e.
    # when the error falls below `barrier_eps_factor · μ`. `max(tol, ·)` keeps the
    # threshold at `tol` once `μ = barrier_min`, so `is_converged` fires first.
    #
    # The factor defaults to 1, NOT to Ipopt's `κ_ε = 10`, and the difference is
    # not a matter of taste. Ipopt applies `κ_ε` to the *scaled* error `E_μ` of
    # Wächter & Biegler (2006) Eq. (5), divided by `s_d` and `s_c`; the error here
    # is unscaled, so carrying `κ_ε` across is not adopting their criterion but
    # loosening ours by an unjustified factor. Measured: with 10, a warm-started
    # replay of a calcite trajectory came back with element-balance residuals of
    # 4.7e-8 instead of 1.4e-11, because every solve stopped one barrier level
    # short. The keyword is exposed for a caller who does scale the error.
    inner_tol = max(opts.tol, opts.barrier_eps_factor * μ)
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
    log_iteration(iter, μ, kkt, α; verbose, α_max = nothing)

Print a one-line iteration summary when `verbose = true`.

`α_max` is the fraction-to-boundary limit, reported alongside the accepted step
because the two failures it separates are indistinguishable from `α` alone: a
step the filter refuses looks exactly like a step the bounds never allowed.
"""
function log_iteration(
        iter::Int, μ, kkt::KKTResidual, α;
        verbose::Bool, α_max = nothing, dn = nothing, dy = nothing,
    )
    if verbose
        println(
            "  iter ",
            lpad(iter, 4), " | μ = ", _fmt_sci(μ),
            " | err0 = ", _fmt_sci(kkt.error_0),
            " | err_opt = ", _fmt_sci(kkt.error_opt),
            " | err_feas = ", _fmt_sci(kkt.error_feas),
            " | α = ", _fmt_sci(α),
            α_max === nothing ? "" : " | α_max = " * _fmt_sci(α_max),
            dn === nothing ? "" : " | ‖dn‖ = " * _fmt_sci(maximum(abs, dn)),
            dy === nothing ? "" : " | ‖dy‖ = " * _fmt_sci(maximum(abs, dy)),
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
