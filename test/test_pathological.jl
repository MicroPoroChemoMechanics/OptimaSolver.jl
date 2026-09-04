@testset "Pathological inputs and diagnostics" begin
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
    prob = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,))

    @testset "a rank-deficient conservation matrix is named, not factorized" begin
        # Row 2 is twice row 1: a conservation law that is a combination of the
        # others. The primal solution is unaffected and the multipliers are
        # simply not unique, but every step here assumes a square basic block.
        # The contract is that this is said plainly rather than surfacing inside
        # a factorization.
        A_dep = [1.0 1.0 1.0; 2.0 2.0 2.0]
        @test_throws ArgumentError OptimaSolver.Canonicalizer(A_dep)
        err = try
            OptimaSolver.Canonicalizer(A_dep)
        catch e
            e
        end
        @test occursin("rank", err.msg)

        # An independent second row is accepted, and the basis is square.
        can = OptimaSolver.Canonicalizer([1.0 1.0 1.0; 1.0 -1.0 0.0])
        @test can.rank_A == 2
        @test length(can.jb) == 2
        @test size(can.B) == (2, 2)
    end

    @testset "refactorize rebuilds the basis factorization" begin
        # Documented for the case where `A` changes under a repartitioning.
        # Nothing called it; what it must guarantee is that the rebuilt object
        # solves the same systems as the original.
        can = OptimaSolver.Canonicalizer(A)
        can2 = OptimaSolver.refactorize(can)

        @test can2.jb == can.jb
        @test can2.jn == can.jn
        @test can2.R ≈ can.R
        rhs = [0.7]
        @test OptimaSolver.solve_B(can2, rhs) ≈ OptimaSolver.solve_B(can, rhs)
    end

    @testset "binding_variable names the variable that blocks the step" begin
        # Diagnostic helper: which bound stops the fraction-to-boundary rule,
        # and at what step length. Never called by the solver, so never checked.
        n = [1.0, 2.0, 3.0]
        lb = zeros(3)

        # Only variable 2 moves down, and its bound cuts the step below one, so
        # it is the one that binds.
        k, α = OptimaSolver.binding_variable(n, lb, [1.0, -5.0, 0.5]; τ = 0.995)
        @test k == 2
        @test α ≈ 0.995 * 2.0 / 5.0

        # A downward step that does not reach the bound within a full step binds
        # nothing: the limit is capped at one and no index is reported.
        k0, α0 = OptimaSolver.binding_variable(n, lb, [1.0, -1.0, 0.5]; τ = 0.995)
        @test k0 == 0
        @test α0 == 1.0

        # Variable 1 is closest to its bound relative to its step.
        k2, α2 = OptimaSolver.binding_variable(n, lb, [-10.0, -1.0, -1.0]; τ = 0.995)
        @test k2 == 1
        @test α2 ≈ 0.995 * 1.0 / 10.0

        # No downward component: nothing binds and the full step is allowed.
        k3, α3 = OptimaSolver.binding_variable(n, lb, [1.0, 2.0, 3.0]; τ = 0.995)
        @test k3 == 0
        @test α3 == 1.0
    end

    @testset "an indefinite reduced Hessian triggers the inertia correction" begin
        # `h` is the barrier-augmented curvature, positive in any real solve.
        # Feeding negative curvature is the only way to reach the ridge loop —
        # the correction Ipopt applies when the plain Cholesky fails. The
        # contract is that a step still comes out, finite and usable.
        can = OptimaSolver.Canonicalizer(A)
        ws = OptimaSolver.NewtonStep(3, 1, Float64)
        ex = [0.1, -0.2, 0.3]
        ew = [0.0]

        # Positive curvature: the plain factorization succeeds, no ridge.
        dn_ok, dy_ok = OptimaSolver.compute_step_nullspace!(ws, can, [1.0, 2.0, 3.0], ex, ew)
        @test all(isfinite, dn_ok) && all(isfinite, dy_ok)

        # Mildly indefinite: the ridge loop raises δ until Cholesky succeeds.
        dn_ind, dy_ind = OptimaSolver.compute_step_nullspace!(ws, can, [1.0, -1.0, -1.0], ex, ew)
        @test all(isfinite, dn_ind)
        @test all(isfinite, dy_ind)

        # Strongly indefinite: no ridge below the ceiling makes it definite, so
        # the Bunch-Kaufman fallback carries the solve instead of it failing.
        dn_bk, dy_bk = OptimaSolver.compute_step_nullspace!(
            ws, can, [1.0, -1.0e8, -1.0e8], ex, ew
        )
        @test all(isfinite, dn_bk)
        @test all(isfinite, dy_bk)
    end

    @testset "an infeasible budget is not silently solved" begin
        # `A n = b` with a negative budget has no solution with `n ≥ lb > 0`.
        # Whatever the solver does, it must not report success.
        prob_bad = OptimaProblem(
            A, [-1.0], G, ∇G!; lb = fill(1.0e-16, 3), p = (μ⁰ = μ⁰,)
        )
        res = solve(prob_bad, OptimaOptions(tol = 1.0e-10, max_iter = 60))
        @test !res.converged
    end

    @testset "verbose logging runs and says what happened" begin
        # The iteration and summary lines are the only diagnostic a caller has
        # when a solve misbehaves; they must at least not throw and must name
        # the outcome.
        function captured(f)
            return mktemp() do path, io
                redirect_stdout(f, io)
                flush(io)
                return read(path, String)
            end
        end

        res = Ref{Any}(nothing)
        s = captured() do
            res[] = solve(prob, OptimaOptions(tol = 1.0e-12, verbose = true))
        end
        @test res[].converged
        @test occursin("iter", s)
        @test occursin("err_opt", s)
        @test occursin("CONVERGED", s)

        s2 = captured() do
            solve(prob, OptimaOptions(tol = 1.0e-18, max_iter = 3, verbose = true))
        end
        @test occursin("MAX_ITER", s2)
    end
end
