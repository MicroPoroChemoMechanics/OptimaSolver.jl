using Test
using Aqua
using OptimaSolver
import OptimaSolver: solve!          # solve! is not exported; import explicitly for tests
using LinearAlgebra
using ForwardDiff
import SciMLBase

"""
    aqua_persistent_tasks(m::Module) -> Bool

`Aqua.test_persistent_tasks` in a child process with default bounds checking.

The check precompiles a synthetic package that depends on `m` and waits for a
sentinel written from that package's module body. Julia writes no
precompilation cache when `--check-bounds` is forced, so the body never runs,
the sentinel never appears, and the check fails for a reason that says nothing
about `m`. `Pkg.test()` forces the flag by default.

Running it in a child with `--check-bounds=auto` keeps both halves: the test
suite proper still runs with bounds checking on, and the check still runs.
"""
function aqua_persistent_tasks(m::Module)
    code = """
    using Aqua, $(nameof(m)), Test
    @testset "persistent_tasks" begin
        Aqua.test_persistent_tasks($(nameof(m)))
    end
    """
    cmd = `$(first(Base.julia_cmd())) --check-bounds=auto --startup-file=no --project=$(Base.active_project()) -e $code`
    return success(run(ignorestatus(cmd)))
end

@testset "OptimaSolver" begin
    # Ambiguities, unbound type parameters, undefined exports, dependency
    # hygiene, type piracy and tasks left running at load — none of which the
    # tests below would notice. Nothing is exempted here: the package passes
    # every default check as it stands.
    @testset "Aqua" begin
        # `persistent_tasks` is run apart, see `aqua_persistent_tasks` above.
        Aqua.test_all(OptimaSolver; persistent_tasks = false)
        @test aqua_persistent_tasks(OptimaSolver)
    end
    include("test_canonicalizer.jl")
    include("test_newton.jl")
    include("test_solver.jl")
    include("test_dual_newton.jl")
    include("test_sensitivity.jl")
    include("test_ad.jl")
    include("test_sciml_interface.jl")
    include("test_solver_paths.jl")
    include("test_stability_linesearch.jl")
end
