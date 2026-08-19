@testset "Newton step" begin
    # Simple 2×4 system: same water/acid conservation as canonicalizer test
    A = Float64[
        2  1  1  2
        1  0  1  0
    ]
    m, ns = size(A)
    can = Canonicalizer(A)

    h = [2.0, 3.0, 1.5, 4.0]     # positive Hessian diagonal
    ex = [0.1, -0.2, 0.05, -0.1]  # optimality residual
    ew = [0.01, -0.005]            # feasibility residual

    ws = NewtonStep(ns, m)
    dn, dy = compute_step!(ws, can, h, ex, ew)

    # Verify KKT: [ H Aᵀ; A 0 ] [dn; dy] = [-ex; -ew]
    lhs_primal = h .* dn .+ A' * dy
    lhs_dual = A * dn
    @test lhs_primal ≈ -ex atol = 1.0e-10
    @test lhs_dual ≈ -ew atol = 1.0e-10

    # clamp_step: no step crosses lower bound
    n = [1.0, 0.5, 2.0, 0.1]
    lb = zeros(4)
    dn_test = [-0.9, 0.1, -0.5, -0.05]
    α = clamp_step(n, lb, dn_test)
    @test α > 0
    @test all(n .+ α .* dn_test .> lb)

    # Step with τ=0.995: should stay strictly inside
    α2 = clamp_step(n, lb, dn_test; τ = 0.995)
    @test all(n .+ α2 .* dn_test .> lb .+ 1.0e-15)
end

@testset "the reduced Hessian is equilibrated before it is factorized" begin

    # `h` is the barrier-augmented curvature, and in a chemical system it spans
    # twenty orders of magnitude. Unequilibrated, the reduced Hessian's condition
    # number passes what Float64 can carry and the Cholesky SUCCEEDS while
    # returning noise: on a cement the step came back at 1e17 and the dual at
    # 1e43, then NaN.
    A = Float64[
        1 0 0 1 2 0;
        0 1 0 1 0 3;
        0 0 1 0 1 1
    ]
    can = Canonicalizer(A)
    h = [1.0, 2.0, 1.5, 1.0e20, 1.0e22, 1.0e18]
    ex = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0]
    ew = zeros(3)

    ws = OptimaSolver.NewtonStep(size(A, 2), size(A, 1))
    dn, dy = OptimaSolver.compute_step_nullspace!(ws, can, h, ex, ew)
    @test all(isfinite, dn)
    @test all(isfinite, dy)
    # A feasible start stays feasible: the null-space step satisfies A dn = -ew.
    @test maximum(abs, A * dn .+ ew) < 1.0e-8
    # The variables with enormous curvature barely move; that is the point.
    @test maximum(abs, dn[4:6]) < 1.0e-15
    # And the step agrees with the Schur-complement branch, which equilibrates.
    ws2 = OptimaSolver.NewtonStep(size(A, 2), size(A, 1))
    dn2, _ = OptimaSolver.compute_step!(ws2, can, h, ex, ew)
    @test maximum(abs, dn .- dn2) < 1.0e-8 * max(1.0, maximum(abs, dn2))

end

@testset "the starting point is made exactly feasible" begin

    # A component total may be NEGATIVE — the H⁺ row of a cement is −2.1 mol — so
    # no basis of abundant species can produce it and solving for the basic
    # amounts is not enough. Lawson–Hanson NNLS on the slacks is, and its residual
    # is zero whenever the budget is attainable.
    A = Float64[
        1 1 1 0 0;
        1 -1 0 1 0;
        0 0 1 0 1
    ]
    b = [2.0, -0.5, 1.0]
    lb = fill(1.0e-16, 5)
    f(n, _) = sum(n .^ 2)
    g!(gr, n, _) = (gr .= 2 .* n)
    prob = OptimaProblem(A, b, f, g!; lb = lb)
    can = Canonicalizer(A)

    n = copy(lb)                       # every variable at its bound
    @test maximum(abs, A * n .- b) > 0.1
    OptimaSolver._initialise_feasible!(n, prob, can)

    @test maximum(abs, A * n .- b) < 1.0e-10       # ON the affine set
    @test all(n .> lb)                             # and strictly interior

    # NNLS itself: exact on a system that admits a non-negative solution.
    v = OptimaSolver._nnls(A, b .- A * lb)
    @test all(v .>= 0)
    @test maximum(abs, A * v .- (b .- A * lb)) < 1.0e-10

end
