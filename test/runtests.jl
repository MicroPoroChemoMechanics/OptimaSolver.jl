using Test
using OptimaSolver
import OptimaSolver: solve!          # solve! is not exported; import explicitly for tests
using LinearAlgebra
using ForwardDiff

@testset "OptimaSolver" begin
    include("test_canonicalizer.jl")
    include("test_newton.jl")
    include("test_solver.jl")
    include("test_dual_newton.jl")
    include("test_sensitivity.jl")
    include("test_ad.jl")
end
