# SPDX-License-Identifier: LGPL-2.1-or-later
# Copyright © 2025-2026 Jean-François Barthélémy and Anthony Soive (Cerema, UMR MCD)

# The frozen cement, shared by `run.jl` and `sweeps.jl` so that the time and the
# sweep count are measured on one system and not on two that drifted apart.
#
# `cases/cement107.json` is the CSHQ system of ChemistryLab's CEM IV page and its
# 28-day budget (107 species, 12 components, two solid solutions). The input is
# frozen; the database and the activity model are ChemistryLab's own.

function cement_case()
    fx = JSON.parsefile(joinpath(HERE, "cases", "cement107.json"))
    subs = build_species(datapath(fx["database"]); verbose = false)
    byname = Dict(symbol(s) => s for s in subs)
    ss = [
        SolidSolutionPhase(name, [byname[m] for m in members])
            for (name, members) in fx["solid_solutions"]
    ]
    cs = ChemicalSystem(
        [byname[s] for s in fx["species"]], CEMDATA_PRIMARIES;
        solid_solutions = ss
    )
    String[symbol(s) for s in cs.species] == fx["species"] ||
        error("the rebuilt system does not list the species in the frozen order")
    st = ChemicalState(cs)
    for (s, n) in fx["initial_amounts_mol"]
        set_quantity!(st, s, n * u"mol")
    end
    am = fx["activity_model"]
    model = HKFActivityModel(å = am["å"], Ḃ = am["Ḃ"], Kₙ = am["Kₙ"])
    return (; cs, st, b = Float64.(fx["b"]), model)
end
