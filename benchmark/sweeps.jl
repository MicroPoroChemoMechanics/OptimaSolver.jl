# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# ── benchmark/sweeps.jl ───────────────────────────────────────────────────────
#
# How many sweeps the inner fixed point of the dual Newton actually runs, and how
# many of its calls stop on the cap rather than on convergence.
#
#     julia --project=benchmark benchmark/sweeps.jl
#
# The count is taken WITHOUT instrumenting the library. The body of
# `_invert_phases!` is read from `src/dual_newton.jl` as it stands, a counter is
# grafted onto its sweep loop, and the copy replaces the method for this session
# only. The library keeps no counter, no global state and no branch in its
# hottest loop; the price is that this script relies on the loop header having
# the form `for <name> in 1:maxsweeps`, and it says so when that stops being true.

using ChemistryLab
using OptimaSolver
using DynamicQuantities
using JSON
using Printf

const HERE = @__DIR__
include(joinpath(HERE, "cement_case.jl"))

function instrumented_source()
    lines = split(read(joinpath(HERE, "..", "src", "dual_newton.jl"), String), '\n')
    i0 = findfirst(l -> startswith(l, "function _invert_phases!("), lines)
    i0 === nothing && error("`function _invert_phases!(` not found in src/dual_newton.jl")
    i1 = findnext(==("end"), lines, i0)
    src = join(lines[i0:i1], '\n')
    m = match(r"\n(\s*)for (\w+) in 1:maxsweeps\n", src)
    m === nothing && error("the sweep loop no longer reads `for <name> in 1:maxsweeps`")
    ind = m.captures[1]
    src = replace(
        src, m.match =>
            "\n$(ind)__last = 0\n$(ind)for __s in 1:maxsweeps\n$(ind)    __last = __s\n"; count = 1
    )
    ret = r"\n(\s*)return _fill_x!\(x_buf, prob, W, refs, act_ph, active, xB\)\nend$"
    occursin(ret, src) || error("the final `return _fill_x!(...)` of `_invert_phases!` moved")
    src = replace(src, ret => s"\n\1push!(_SWEEPS, (__last, maxsweeps))\n\1return _fill_x!(x_buf, prob, W, refs, act_ph, active, xB)\nend")
    return src
end

@eval OptimaSolver const _SWEEPS = Tuple{Int, Int}[]
Base.include_string(OptimaSolver, instrumented_source(), "instrumented _invert_phases!")

quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
c = cement_case()
empty!(OptimaSolver._SWEEPS)
t = @elapsed (eq, cert) = quiet(() -> equilibrate_certified(deepcopy(c.st); model = c.model, b = c.b))
sw = first.(OptimaSolver._SWEEPS)
cap = last.(OptimaSolver._SWEEPS)
n = length(sw)
capped = [sw[i] == cap[i] for i in 1:n]

@printf("cold cement solve: %.1f s (compilation included), certified = %s\n\n", t, cert.optimal)
@printf("calls to _invert_phases!   %9d\n", n)
@printf("sweeps, total              %9d\n", sum(sw))
@printf("sweeps per call            mean %.1f, median %d, max %d\n", sum(sw) / n, sort(sw)[cld(n, 2)], maximum(sw))
@printf("calls that ran to the cap  %9d  (%.1f %%)\n", count(capped), 100count(capped) / n)
@printf(
    "sweeps spent in them       %9d  (%.1f %% of all sweeps)\n",
    sum(sw[capped]; init = 0), 100sum(sw[capped]; init = 0) / sum(sw)
)
println("\nsweeps per call:")
for (lo, hi) in ((1, 1), (2, 5), (6, 20), (21, 50), (51, 100), (101, 199), (200, 200))
    k = count(s -> lo <= s <= hi, sw)
    @printf("  %3d–%-3d  %7d calls  (%5.1f %%)\n", lo, hi, k, 100k / n)
end
