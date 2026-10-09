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
        # nothing is differentiated at all.
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
        # Near the solution the barrier objective no longer resolves the decrease
        # Armijo asks for, and the step is judged on the KKT residual instead:
        # full steps to the end, 32 iterations measured. Left to rounding, the
        # search halved them at random, took 44 here and ran to `MaxIters` on a
        # machine whose arithmetic differed in the last digits.
        @test sol.original.iterations <= 34
        @test length(sol.original.y) == 1
    end

    @testset "convergence does not depend on the last digits" begin
        # The same solve with its data moved by a few ulp, which is what another
        # machine's arithmetic does to it. Left to rounding near the solution,
        # the line search failed on 34 of 200 such variants and took 32 to 102
        # iterations on the others (26 failures with 0.7.4); judged on the KKT
        # residual where the objective cannot resolve the step, all 200 converge
        # in 32.
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        its = map(1:50) do k
            moved = μ⁰ .+ (((7k) % 17 - 8) .* [1, -1, 1]) .* eps(2.0)
            bk = [1.0 + ((5k) % 17 - 8) * eps(1.0)]
            sol = SciMLBase.solve(
                SciMLBase.OptimizationProblem(f, copy(lb), (μ⁰ = moved, A = A, b = bk); lb = lb, ub = fill(Inf, 3)),
                OptimaOptimizer(; tol = 1.0e-12),
            )
            sol.original.converged ? sol.original.iterations : typemax(Int)
        end
        @test maximum(its) <= 34
    end

    @testset "a forward-mode gradient differentiated again for the Hessian" begin
        # No `grad`: the gradient is built by forward mode, and `OptimaOptimizer()`
        # (`use_fd_hessian = true`) differentiates that gradient to get the exact
        # Hessian diagonal, so two differentiations are nested. Each carries its
        # own tag; with the gradient's config built on a `nothing` tag, as until
        # 0.7.4, nothing kept them apart.
        f = SciMLBase.OptimizationFunction(G)
        prob = SciMLBase.OptimizationProblem(
            f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3)
        )
        sol = SciMLBase.solve(prob, OptimaOptimizer(; tol = 1.0e-12))
        @test sol.u ≈ n_analytic atol = 1.0e-7
    end

    @testset "parameters carrying dual numbers" begin
        # Solved on them: the answer carries the derivatives of the iterations
        # that reached it, those of the softmax at convergence,
        # ∂nᵢ/∂μ⁰₁ = −nᵢ(δᵢ₁ − n₁).
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        solve_at(m1) = SciMLBase.solve(
            SciMLBase.OptimizationProblem(
                f, copy(lb), (μ⁰ = [m1, μ⁰[2], μ⁰[3]], A = A, b = b); lb = lb, ub = fill(Inf, 3)
            ),
            OptimaOptimizer(; tol = 1.0e-12, warm_start = false),
        ).u
        dn = ForwardDiff.derivative(solve_at, μ⁰[1])
        @test dn ≈ -n_analytic .* ([1.0, 0.0, 0.0] .- n_analytic[1]) rtol = 1.0e-6
    end

    @testset "constraints given as a residual function (forward-mode path)" begin
        # `_extract_constraints` path 3: A and b are recovered by differentiating
        # `cons` at `u0`, in forward mode. The answer must be the same as when A and b are handed
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

    @testset "a nonlinear residual gets its exact tangent" begin
        # `_extract_constraints` differentiates the residual by forward mode, so
        # its linearization is exact for an affine residual and is the true
        # tangent of a nonlinear one: a log-parameterized equilibrium sends
        # `A exp(x) - b`, whose Jacobian at `x0` is `exp.(x0)`.
        x0 = [-2.0, -3.0]
        consexp!(res, x, _) = (res[1] = exp(x[1]) + exp(x[2]) - 1.0; nothing)
        f = SciMLBase.OptimizationFunction((x, q) -> sum(exp.(x)); cons = consexp!)
        prob = SciMLBase.OptimizationProblem(
            f, x0, nothing;
            lb = fill(-30.0, 2), ub = fill(10.0, 2), lcons = [0.0], ucons = [0.0]
        )
        A_nl, _ = OptimaSolver._extract_constraints(prob, x0, prob.p)
        # The exact Jacobian is exp.(x0).
        @test A_nl[1, 1] ≈ exp(x0[1]) rtol = 1.0e-6
        @test A_nl[1, 2] ≈ exp(x0[2]) rtol = 1.0e-6

        # And the affine case comes back to within an ulp.
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
    @testset "the warm-start cache does not cross number types" begin
        # A plain answer in the cache, then a solve on dual numbers through the
        # same algorithm object, and the reverse. Until 0.8.2 the first raised a
        # `MethodError` (the cached plain amounts handed to a lift typed on the
        # duals) and so did the second. A solve on duals starts cold: it
        # differentiates the iterations that reach its answer, and must take
        # them.
        f = SciMLBase.OptimizationFunction(G; grad = ∇G!)
        alg = OptimaOptimizer(; tol = 1.0e-12)
        plain = SciMLBase.OptimizationProblem(f, copy(lb), (μ⁰ = μ⁰, A = A, b = b); lb = lb, ub = fill(Inf, 3))
        @test SciMLBase.successful_retcode(SciMLBase.solve(plain, alg))
        solve_at(m1) = SciMLBase.solve(
            SciMLBase.OptimizationProblem(
                f, copy(lb), (μ⁰ = [m1, μ⁰[2], μ⁰[3]], A = A, b = b); lb = lb, ub = fill(Inf, 3)
            ),
            alg,
        ).u
        dn = ForwardDiff.derivative(solve_at, μ⁰[1])
        @test dn ≈ -n_analytic .* ([1.0, 0.0, 0.0] .- n_analytic[1]) rtol = 1.0e-6
        # Nothing on dual numbers is left in the cache for a plain solve to trip on.
        @test alg._cache[] isa OptimaResult{Float64}
        @test SciMLBase.solve(plain, alg).u ≈ n_analytic atol = 1.0e-7
        # Two derivatives in a row, one seed each, through one object: the
        # second may not start from the answer of the first, whose partials are
        # those of the other seed.
        solve2(m) = SciMLBase.solve(
            SciMLBase.OptimizationProblem(
                f, copy(lb), (μ⁰ = [m[1], m[2], μ⁰[3]], A = A, b = b); lb = lb, ub = fill(Inf, 3)
            ),
            alg,
        ).u
        m12 = μ⁰[1:2]
        J = ForwardDiff.jacobian(solve2, m12, ForwardDiff.JacobianConfig(solve2, m12, ForwardDiff.Chunk{1}()))
        @test J ≈ -(Diagonal(n_analytic) .- n_analytic * transpose(n_analytic))[:, 1:2] rtol = 1.0e-6
    end

    @testset "a constraint that is not affine is linearized again, and judged" begin
        # The toy Gibbs problem in the logarithms of its amounts, `x = ln n`, its
        # constraint `Σ exp(xᵢ) = 1`. OptimaSolver handles `A n = b` and took the
        # tangent of this constraint at the start; until 0.8.2 it kept it, and
        # returned `Success` on a point violating the constraint by 3.8e-3, its
        # amounts 2 % off. Linearized again at each answer, the point meets it to
        # better than 1e-6; and a point that does not meet it to the tolerance is
        # not reported a success. (This interior point is made for amounts above
        # a bound near zero; in logarithms it stalls before the tolerance.)
        Gx(x, p) = sum(exp(x[i]) * (p.μ⁰[i] + x[i]) for i in eachindex(x))
        consx!(res, x, _) = (res[1] = sum(exp, x) - 1.0; nothing)
        f = SciMLBase.OptimizationFunction(Gx; cons = consx!)
        x0 = log.([0.6, 0.3, 0.1])
        prob = SciMLBase.OptimizationProblem(
            f, x0, (μ⁰ = μ⁰,); lb = fill(-40.0, 3), ub = fill(5.0, 3), lcons = [0.0], ucons = [0.0],
        )
        sol = SciMLBase.solve(prob, OptimaOptimizer(; tol = 1.0e-12, warm_start = false))
        violation = abs(sum(exp, sol.u) - 1.0)
        @test violation < 1.0e-6
        @test exp.(sol.u) ≈ n_analytic rtol = 1.0e-3
        @test !SciMLBase.successful_retcode(sol) || violation <= 1.0e-12

        # An affine residual is solved as before: its linearization is the
        # constraint, and nothing is solved again. Its residual is evaluated
        # three times: at the start and by the forward-mode Jacobian that
        # extract `A` and `b`, then once at the answer, which agrees with its
        # linearization.
        calls = Ref(0)
        conslin!(res, u, _) = (calls[] += 1; res[1] = sum(u) - 1.0; nothing)
        flin = SciMLBase.OptimizationFunction(G; grad = ∇G!, cons = conslin!)
        plin = SciMLBase.OptimizationProblem(
            flin, copy(lb), (μ⁰ = μ⁰,); lb = lb, ub = fill(Inf, 3), lcons = [0.0], ucons = [0.0],
        )
        s1 = SciMLBase.solve(plin, OptimaOptimizer(; tol = 1.0e-12, warm_start = false))
        @test SciMLBase.successful_retcode(s1)
        @test calls[] == 3
    end
end
