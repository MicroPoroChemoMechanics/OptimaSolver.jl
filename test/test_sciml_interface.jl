@testset "SciML interface — OptimaOptimizer" begin
    # The documented entry point of this package is the SciML one: a caller
    # hands an `OptimizationProblem` to `OptimaOptimizer()` and never sees
    # `OptimaProblem`. Nothing exercised it, so the conversion, the constraint
    # extraction, the variable scaling and the warm-start cache were all
    # untested. What follows is the toy Gibbs problem of `test_solver.jl`,
    # driven through that path and checked against its analytic solution.

    μ⁰ = [0.0, 1.0, 2.0]
    G(n, p) = sum(n[i] * (p.μ⁰[i] + log(n[i])) for i in eachindex(n))
    function ∇G!(grad, n, p)
        for i in eachindex(n)
            grad[i] = p.μ⁰[i] + log(n[i]) + 1
        end
        return nothing
    end

    A = ones(1, 3)
    b = [1.0]
    n_analytic = exp.(-μ⁰) ./ sum(exp.(-μ⁰))
    lb = fill(1.0e-16, 3)

    @testset "constructors and cache reset" begin
        alg = OptimaOptimizer()
        @test alg isa SciMLBase.AbstractOptimizationAlgorithm
        @test alg.options.warm_start
        @test alg._cache[] === nothing

        alg2 = OptimaOptimizer(; tol = 1.0e-8, max_iter = 42, warm_start = false)
        @test alg2.options.tol == 1.0e-8
        @test alg2.options.max_iter == 42
        @test !alg2.options.warm_start

        alg3 = OptimaOptimizer(OptimaOptions(tol = 1.0e-9))
        @test alg3.options.tol == 1.0e-9

        @test reset_cache!(alg3) === alg3
        @test alg3._cache[] === nothing
    end

    @testset "constraints carried by `p` (Optima-native path)" begin
        # `_extract_constraints` path 1: `p` is a NamedTuple holding A and b, so
        # no finite differencing happens at all.
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        prob = SciMLBase.OptimizationProblem(
            f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3)
        )
        sol = SciMLBase.solve(prob, OptimaOptimizer(; tol = 1.0e-12))

        @test SciMLBase.successful_retcode(sol)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test sol.u ≈ n_analytic atol = 1.0e-7
        @test abs(only(A * sol.u) - only(b)) < 1.0e-10
        @test sol.objective ≈ G(n_analytic, (μ⁰ = μ⁰,)) atol = 1.0e-9
        # The internal result travels along, so a caller can read the duals.
        @test sol.original isa OptimaResult
        @test sol.original.converged
        @test length(sol.original.y) == 1
    end

    @testset "constraints given as a residual function (finite-difference path)" begin
        # `_extract_constraints` path 3: A and b are recovered by differencing
        # `cons` at `u0`. The answer must be the same as when A and b are handed
        # over directly — that equality is the whole point of the extraction.
        cons!(res, u, p) = (res[1] = sum(u) - 1.0; nothing)
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!, cons = cons!)
        prob = SciMLBase.OptimizationProblem(
            f, copy(lb), (μ⁰ = μ⁰,);
            lb = lb, ub = fill(Inf, 3), lcons = [0.0], ucons = [0.0]
        )
        sol = SciMLBase.solve(prob, OptimaOptimizer(; tol = 1.0e-12))

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ n_analytic atol = 1.0e-6
    end

    @testset "a nonlinear residual is differenced at the right scale" begin
        # `_extract_constraints` prefers a large step, because for the affine
        # residual it documents the quotient is exact and a small step only adds
        # cancellation. A nonlinear residual breaks that reasoning: a
        # log-parameterized equilibrium sends `A exp(x) - b`, where a step of one
        # is not a derivative at all — measured, 72 % off. So affinity is checked
        # per column, and this pins the check.
        x0 = [-2.0, -3.0]
        consexp!(res, x, _) = (res[1] = exp(x[1]) + exp(x[2]) - 1.0; nothing)
        f = SciMLBase.OptimizationFunction((x, q) -> sum(exp.(x)); cons = consexp!)
        prob = SciMLBase.OptimizationProblem(
            f, x0, nothing;
            lb = fill(-30.0, 2), ub = fill(10.0, 2), lcons = [0.0], ucons = [0.0]
        )
        A_nl, _ = OptimaSolver._extract_constraints(prob, x0, prob.p)
        # The exact Jacobian is exp.(x0); a forward difference reaches ~1e-8.
        @test A_nl[1, 1] ≈ exp(x0[1]) rtol = 1.0e-6
        @test A_nl[1, 2] ≈ exp(x0[2]) rtol = 1.0e-6

        # And the affine case still comes back to within an ulp, which is the
        # whole reason the large step is tried first.
        conslin!(res, u, _) = (res[1] = 2u[1] + 3u[2] - 5.0; nothing)
        f2 = SciMLBase.OptimizationFunction((u, q) -> sum(u); cons = conslin!)
        prob2 = SciMLBase.OptimizationProblem(
            f2, [1.0e-16, 1.0e-16], nothing;
            lb = fill(1.0e-16, 2), ub = fill(Inf, 2), lcons = [0.0], ucons = [0.0]
        )
        A2, b2 = OptimaSolver._extract_constraints(prob2, prob2.u0, prob2.p)
        @test A2[1, 1] ≈ 2.0 atol = 1.0e-14
        @test A2[1, 2] ≈ 3.0 atol = 1.0e-14
        @test b2[1] ≈ 5.0 atol = 1.0e-14
    end

    @testset "gradient falls back to ForwardDiff when none is given" begin
        f = SciMLBase.OptimizationFunction(G)
        prob = SciMLBase.OptimizationProblem(
            f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3)
        )
        sol = SciMLBase.solve(prob, OptimaOptimizer(; tol = 1.0e-12))

        @test SciMLBase.successful_retcode(sol)
        @test sol.u ≈ n_analytic atol = 1.0e-7
    end

    @testset "a problem with no constraints is rejected, and says why" begin
        # `_extract_constraints` used to hand back an empty `A` here, which read
        # as support for unconstrained problems. It was not: `Canonicalizer`
        # pivots a QR of `A`, LAPACK returns a degenerate permutation for a
        # zero-row matrix, and the run died on a `BoundsError` with nothing in
        # the message pointing at the cause.
        H(u, p) = sum((u .- p.target) .^ 2)
        f = SciMLBase.OptimizationFunction(H)
        prob = SciMLBase.OptimizationProblem(
            f, fill(0.5, 3), (target = [0.3, 0.7, 1.1],);
            lb = fill(1.0e-16, 3), ub = fill(Inf, 3)
        )
        @test_throws ArgumentError SciMLBase.solve(prob, OptimaOptimizer())

        # Same contract one level down, where the solver itself can state it.
        @test_throws ArgumentError OptimaProblem(
            zeros(0, 3), Float64[], (u, p) -> sum(u), (g, u, p) -> (g .= 1)
        )
    end

    @testset "warm start reuses the cache but never overrides a caller's guess" begin
        # The cache holds the last solution of whatever problem the algorithm
        # object last saw. Honoring it blindly discards an explicit `u0`, which
        # is the defect the source comments describe at length; this pins the
        # behavior both ways.
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        alg = OptimaOptimizer(; tol = 1.0e-12)

        prob1 = SciMLBase.OptimizationProblem(f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3))
        sol1 = SciMLBase.solve(prob1, alg)
        @test SciMLBase.successful_retcode(sol1)
        # A converged solve is cached; a fresh optimizer's cache stays empty.
        @test alg._cache[] !== nothing
        @test alg._cache[].n ≈ sol1.u

        # Same problem again, still at the lower bound: the cache is used and
        # the answer is unchanged.
        sol2 = SciMLBase.solve(prob1, alg)
        @test sol2.u ≈ n_analytic atol = 1.0e-7

        # Now a caller-supplied interior guess. It must be honored, i.e. the
        # cached point is discarded, and the answer must still be the right one.
        prob3 = SciMLBase.OptimizationProblem(
            f, [0.2, 0.3, 0.5], (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3)
        )
        sol3 = SciMLBase.solve(prob3, alg)
        @test sol3.u ≈ n_analytic atol = 1.0e-7

        # A problem of a different size cannot reuse the cache either.
        A4 = ones(1, 4)
        μ⁰4 = [0.0, 1.0, 2.0, 3.0]
        prob4 = SciMLBase.OptimizationProblem(
            f, fill(1.0e-16, 4), (μ⁰ = μ⁰4, A = A4, b = b); lb = fill(1.0e-16, 4), ub = fill(Inf, 4)
        )
        sol4 = SciMLBase.solve(prob4, alg)
        @test SciMLBase.successful_retcode(sol4)
        @test sol4.u ≈ exp.(-μ⁰4) ./ sum(exp.(-μ⁰4)) atol = 1.0e-6

        # With `warm_start = false` the cache is never consulted.
        alg_cold = OptimaOptimizer(; tol = 1.0e-12, warm_start = false)
        sol5 = SciMLBase.solve(prob1, alg_cold)
        @test sol5.u ≈ n_analytic atol = 1.0e-7
    end

    @testset "a solve that runs out of iterations reports it and is not cached" begin
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        prob = SciMLBase.OptimizationProblem(f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3))
        alg = OptimaOptimizer(; tol = 1.0e-14, max_iter = 2)
        sol = SciMLBase.solve(prob, alg)

        @test sol.retcode == SciMLBase.ReturnCode.MaxIters
        @test !sol.original.converged
        # Caching a non-converged point would poison every later warm start.
        @test alg._cache[] === nothing
    end
end
