# Changelog

## v0.2.1 — Maintenance

- Runic.yml: switch to `workflow_dispatch` only; prevents automatic
  formatting commits that could conflict with local work
- No API changes, no functional changes

## v0.2.0 — Codeberg migration

- Migrated to Codeberg (`MicroPoroChemoMechanics/OptimaSolver.jl`)
- Forgejo workflows: CI, Documentation, Release, Runic, Zenodo
- Registered in MPCM-Registry
- Multi-version documentation deployment (`docs/deploy_docs.jl`)
- Relicensed to LGPL-2.1-or-later

## v0.1.0 — Initial release

Primal-dual interior-point solver for Gibbs-energy minimization.

- `OptimaProblem` / `OptimaOptions` / `OptimaResult` API
- Schur-complement Newton step exploiting diagonal Hessian structure
- Filter line search (Wächter & Biegler 2006)
- Implicit-differentiation sensitivity matrices (`∂n*/∂b`, `∂n*/∂μ⁰`)
- SciML drop-in via `OptimaOptimizer` / `SciMLBase.solve`
- ForwardDiff-compatible throughout
