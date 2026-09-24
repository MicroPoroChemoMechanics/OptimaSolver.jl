# Benchmark

Wall time and answer of the certified equilibrium route, measured together on a
frozen cement so that a change to the solver is judged on both at once. A
faster solver that returns a different composition is a regression, and
`run.jl` is the script that reports it as one.

## Running it

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=benchmark benchmark/run.jl                    # against the baseline
julia --project=benchmark benchmark/sweeps.jl                 # inner-loop diagnostic
```

`run.jl --record` rewrites `baseline.json`. The environment resolves
`OptimaSolver` from this checkout and `ChemistryLab` from the sibling checkout
`../../ChemistryLab.jl`; `run.jl` refuses to start if the solver it loaded is
not this one, since a benchmark of the registered release would report the
wrong code as a result.

## What is measured

The case is `cases/cement107.json`: the CSHQ system of ChemistryLab's CEM IV
example and its 28-day budget, with 107 species, 12 components and two solid
solutions. Only the input is frozen; the database and the activity model are
ChemistryLab's own. Four solves are timed: a cold start twice (the first pays
compilation and is reported, not compared), a warm start from the answer, and a
neighboring budget two percent larger from the same answer.

Each answer is compared with the baseline on the certificate and on the largest
relative change of any amount above `1e-10` mol, below which a relative change
is noise. The times in `baseline.json` belong to the machine that recorded them;
the compositions do not.

`sweeps.jl` counts the sweeps of the inner fixed point of the dual Newton
without instrumenting the library: it reads `_invert_phases!` from
`src/dual_newton.jl`, grafts a counter onto its sweep loop and replaces the
method for that session only.

## Recorded

`baseline.json` holds the answers of 0.6.0, before the inner iteration was
stopped on stagnation. On the machine that recorded it:

| solve | 0.6.0 | 0.6.1 |
|:--|--:|--:|
| cement, cold | 142.6 s | 16.8 s |
| cement, cold, first call | 165.0 s | 37.6 s |
| cement, warm | 1.00 s | 0.17 s |
| cement, neighbor, warm | 1.01 s | 0.17 s |

with every answer certified and the largest relative change of an amount
`4.1e-13`.
