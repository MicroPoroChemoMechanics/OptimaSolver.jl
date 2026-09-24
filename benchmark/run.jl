# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# ── benchmark/run.jl ──────────────────────────────────────────────────────────
#
# Wall time AND answer of the certified equilibrium route on a frozen cement,
# so that a change to the solver is judged on both at once. A faster solver that
# returns a different composition is a regression, and this is the file that
# says so.
#
#     julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once
#     julia --project=benchmark benchmark/run.jl            # compare to baseline
#     julia --project=benchmark benchmark/run.jl --record   # rewrite the baseline
#
# The cement is `cases/cement107.json`: the CSHQ system of ChemistryLab's CEM IV
# page and its 28-day budget, frozen so that editing that page cannot move the
# benchmark. Only the input is frozen; the database and the activity model are
# ChemistryLab's own.

using ChemistryLab
using OptimaSolver
using DynamicQuantities
using JSON
using Printf

const HERE = @__DIR__
const BASELINE = joinpath(HERE, "baseline.json")

# Loading the wrong OptimaSolver is the one mistake that makes every number
# below meaningless, so it is checked rather than assumed.
let here = realpath(joinpath(HERE, ".."))
    got = realpath(pkgdir(OptimaSolver))
    got == here || error("benchmark loaded OptimaSolver from $got, not from $here")
end

quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
mol(st) = Float64[ustrip(us"mol", x) for x in st.n]

include(joinpath(HERE, "cement_case.jl"))

# What an answer is judged on. Amounts below 1e-10 mol are left out of the
# relative comparison: their relative change is noise and says nothing.
digest(eq, cert, model) = Dict(
    "optimal" => cert.optimal,
    "stationarity" => cert.stationarity,
    "balance" => cert.balance,
    "pH" => pH(eq, model),
    "n" => mol(eq),
)

function timed(label, f)
    GC.gc()
    t = @elapsed r = quiet(f)
    @printf("  %-32s %9.2f s\n", label, t)
    return t, r
end

function run_cases()
    c = cement_case()
    out = Dict{String, Any}()

    # The first call pays compilation; it is reported, and not compared, because
    # its time says more about the Julia version than about the solver.
    t1, (eq1, c1) = timed(
        "cement, cold, first call", () ->
        equilibrate_certified(deepcopy(c.st); model = c.model, b = c.b)
    )
    t2, (eq2, c2) = timed(
        "cement, cold", () ->
        equilibrate_certified(deepcopy(c.st); model = c.model, b = c.b)
    )
    t3, (eq3, c3) = timed(
        "cement, warm (from its answer)", () ->
        equilibrate_certified(eq2; model = c.model, b = c.b)
    )
    b4 = 1.02 .* c.b
    t4, (eq4, c4) = timed(
        "cement, neighbor (+2 %), warm", () ->
        equilibrate_certified(eq2; model = c.model, b = b4)
    )

    out["cement_cold_first"] = merge(digest(eq1, c1, c.model), Dict("time" => t1))
    out["cement_cold"] = merge(digest(eq2, c2, c.model), Dict("time" => t2))
    out["cement_warm"] = merge(digest(eq3, c3, c.model), Dict("time" => t3))
    out["cement_neighbor"] = merge(digest(eq4, c4, c.model), Dict("time" => t4))
    out["species"] = String[symbol(s) for s in c.cs.species]
    return out
end

function compare(new, old)
    println("\n  case                          time (s)  before → now      certified   max rel Δn    ΔpH")
    worst = 0.0
    for key in ("cement_cold", "cement_warm", "cement_neighbor")
        a, b = old[key], new[key]
        na, nb = Float64.(a["n"]), Float64.(b["n"])
        keep = [i for i in eachindex(na) if na[i] > 1.0e-10 && nb[i] > 1.0e-10]
        rel = isempty(keep) ? 0.0 : maximum(abs(nb[i] - na[i]) / na[i] for i in keep)
        dph = abs(b["pH"] - a["pH"])
        worst = max(worst, rel)
        @printf(
            "  %-28s %8.2f → %-8.2f  %-5s → %-5s  %10.2e  %8.2e\n",
            key, a["time"], b["time"], a["optimal"], b["optimal"], rel, dph
        )
    end
    return worst
end

function main()
    record = "--record" in ARGS
    println("OptimaSolver ", pkgversion(OptimaSolver), " from ", pkgdir(OptimaSolver))
    println("ChemistryLab ", pkgversion(ChemistryLab), " from ", pkgdir(ChemistryLab), "\n")
    new = run_cases()
    return if record
        open(io -> JSON.print(io, new, 1), BASELINE, "w")
        println("\nbaseline written to ", BASELINE)
    elseif isfile(BASELINE)
        worst = compare(new, JSON.parsefile(BASELINE))
        @printf("\nworst relative change of an amount above 1e-10 mol: %.2e\n", worst)
    else
        println("\nno baseline yet: run with --record first")
    end
end

main()
