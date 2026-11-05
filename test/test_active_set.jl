# SPDX-License-Identifier: LGPL-2.1-or-later

using OptimaSolver
using ForwardDiff
using LinearAlgebra
using Test

# ── The active-set iteration of Optima (`solve_active_set`) ─────────────────────
#
# Each reference is independent of the solver: a closed form for an ideal
# solution, the KKT conditions themselves, the implicit-function theorem for the
# derivatives, and the same problem scaled.

# An ideal solution of species `i` with standard potentials `μ⁰`, and pure phases
# `pure` (linear in their amount): G = Σ_solution nᵢ (μ⁰ᵢ + ln xᵢ) + Σ_pure nᵢ μ⁰ᵢ.
function _as_problem(A, b, μ⁰; pure = Int[], lb = 1.0e-16)
    sol = setdiff(eachindex(μ⁰), pure)
    function G(n, p)
        N = sum(n[i] for i in sol)
        return sum(n[i] * (p.μ⁰[i] + (i in sol ? log(n[i] / N) : zero(eltype(n)))) for i in eachindex(n))
    end
    function ∇G!(g, n, p)
        N = sum(n[i] for i in sol)
        for i in eachindex(n)
            g[i] = p.μ⁰[i] + (i in sol ? log(n[i] / N) : zero(eltype(n)))
        end
        return g
    end
    return OptimaProblem(A, b, G, ∇G!; lb = fill(lb, length(μ⁰)), p = (μ⁰ = μ⁰,))
end

const AS_OPTS = OptimaOptions(tol = 1.0e-12, algorithm = :active_set)

@testset "active set: the iteration of Optima" begin
    @testset "an ideal solution, against its closed form" begin
        # One row, Σ nᵢ = 1: xᵢ ∝ exp(−μ⁰ᵢ).
        μ⁰ = [0.0, 1.0, 2.0]
        r = solve(_as_problem(ones(1, 3), [1.0], μ⁰), AS_OPTS)
        @test r.converged
        @test r.n ≈ exp.(-μ⁰) ./ sum(exp.(-μ⁰)) rtol = 1.0e-10
        # stationarity μ⁰ᵢ + ln xᵢ + y = 0 with Σ xᵢ = 1
        @test r.y ≈ [log(sum(exp.(-μ⁰)))] rtol = 1.0e-10
    end

    @testset "a pure phase that should not form stays on its bound" begin
        # Species 3 is a pure phase of potential 5 for the element of row 1: the
        # solution holds it at a potential below 5, so it is absent, on its bound,
        # with a positive reduced gradient.
        A = [1.0 1.0 1.0; 1.0 0.0 0.0]
        b = [1.0, 0.4]
        μ⁰ = [0.0, 0.5, 5.0]
        r = solve(_as_problem(A, b, μ⁰; pure = [3]), AS_OPTS)
        @test r.converged
        @test r.n[3] == 1.0e-16
        @test A * r.n ≈ b rtol = 1.0e-12
        s3 = μ⁰[3] + transpose(A[:, 3]) * r.y
        @test s3 > 0
    end

    @testset "a start off the balance gets back on it" begin
        A = Float64[2 1 1 2; 1 0 1 0]
        b = [4.0, 1.0]
        μ⁰ = [0.0, 0.5, 1.0, 1.5]
        r = solve(_as_problem(A, b, μ⁰), AS_OPTS; u0 = [5.0, 0.1, 3.0, 0.2])
        @test r.converged
        @test norm(A * r.n .- b, Inf) < 1.0e-12
        # stationarity of every species, the KKT conditions themselves
        N = sum(r.n)
        @test μ⁰ .+ log.(r.n ./ N) .+ transpose(A) * r.y ≈ zeros(4) atol = 1.0e-10
    end

    @testset "the answer does not depend on how much matter the system holds" begin
        A = Float64[2 1 1 2 1; 1 0 1 0 0]
        μ⁰ = [0.0, 0.5, 1.0, 1.5, 30.0]
        x(scale) = (r = solve(_as_problem(A, scale .* [4.0, 1.0], μ⁰; pure = [5], lb = 1.0e-16 * scale), AS_OPTS); r.n ./ scale)
        ref = x(1.0)
        @test x(1.0e-6) ≈ ref rtol = 1.0e-9
        @test x(1.0e6) ≈ ref rtol = 1.0e-9
    end

    @testset "a saddle point is not returned" begin
        # A binary solution with a regular excess `a x(1−x)`, a = 3 > 2: it
        # unmixes, and two instances of it hold the two compositions of the
        # common tangent, x and 1 − x by symmetry. Started from two identical
        # instances at x = 1/2, a stationary point of negative curvature, Newton's
        # method stays there; the inertia correction leaves it.
        a = 3.0
        function G(n, p)
            g = zero(eltype(n))
            for (i, j) in ((1, 2), (3, 4))
                N = n[i] + n[j]
                xi, xj = n[i] / N, n[j] / N
                g += n[i] * log(xi) + n[j] * log(xj) + p.a * N * xi * xj
            end
            return g
        end
        ∇G!(gr, n, p) = (gr .= ForwardDiff.gradient(x -> G(x, p), n); gr)
        A = Float64[1 0 1 0; 0 1 0 1]
        b = [1.0, 1.0]
        prob = OptimaProblem(A, b, G, ∇G!; lb = fill(1.0e-16, 4), p = (a = a,))
        r = solve(prob, AS_OPTS; u0 = [0.5, 0.5, 0.5, 0.5] .+ [1, -1, -1, 1] .* 1.0e-3)
        @test r.converged
        x1 = r.n[2] / (r.n[1] + r.n[2])
        x2 = r.n[4] / (r.n[3] + r.n[4])
        # the common tangent of a symmetric regular solution: ln(x/(1−x)) = a(2x − 1)
        root(x) = log(x / (1 - x)) - a * (2x - 1)
        @test abs(root(min(x1, x2))) < 1.0e-8
        @test min(x1, x2) < 0.2 && max(x1, x2) > 0.8
    end

    @testset "derivatives with respect to the parameters, by the implicit-function theorem" begin
        A = Float64[2 1 1 2; 1 0 1 0]
        b = [4.0, 1.0]
        n_of(μ⁰) = solve(_as_problem(A, b, μ⁰), AS_OPTS).n
        μ⁰ = [0.0, 0.5, 1.0, 1.5]
        J = ForwardDiff.jacobian(n_of, μ⁰)
        # At the answer, stationarity μ⁰ + ln(n/N) + Aᵀy = 0 and A n = b; their
        # differentials give [H Aᵀ; A 0] [dn; dy] = [−dμ⁰; 0].
        n = n_of(μ⁰)
        N = sum(n)
        H = Diagonal(1 ./ n) .- 1 / N
        K = [H transpose(A); A zeros(2, 2)]
        Jref = (K \ [-Matrix(I, 4, 4); zeros(2, 4)])[1:4, :]
        @test J ≈ Jref rtol = 1.0e-8
    end

    @testset "the verbose log says what happened" begin
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        captured(f) = mktemp() do path, io
            redirect_stdout(f, io)
            flush(io)
            read(path, String)
        end
        s = captured(() -> solve(prob, OptimaOptions(tol = 1.0e-12, verbose = true)))
        @test occursin("err_opt", s) && occursin("CONVERGED", s)
        s2 = captured(() -> solve(prob, OptimaOptions(tol = 1.0e-30, max_iter = 2, verbose = true)))
        @test occursin("MAX_ITER", s2)
    end

    @testset "a search that no longer progresses ends on its best iterate" begin
        # A tolerance below the rounding of the problem: the error stops
        # decreasing, and the run ends long before its budget, on an answer as
        # good as the converged one.
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        r = solve(prob, OptimaOptions(tol = 1.0e-30, max_iter = 300))
        @test !r.converged
        @test r.iterations < 100
        μ⁰ = [0.0, 1.0, 2.0]
        @test r.n ≈ exp.(-μ⁰) ./ sum(exp.(-μ⁰)) rtol = 1.0e-10
    end

    @testset "a basis the weights make singular is chosen on the columns" begin
        # Two columns of equal weight carry the first row only; the third, of
        # zero weight, alone carries the second. Ordered by the weights, the two
        # first are chosen and are singular; the basis is then chosen on `A`.
        A = [1.0 1.0 0.0; 0.0 0.0 1.0]
        jb = OptimaSolver._weighted_basis(A, [1.0, 1.0, 0.0])
        @test 3 in jb && rank(A[:, jb]) == 2
        # and where the weights give a regular basis, it is theirs
        @test OptimaSolver._weighted_basis(A, [1.0, 2.0, 1.0]) == [2, 3]
    end

    @testset "a basis is one the matrix supports, the heaviest columns first" begin
        # The second column differs from the first by rounding: weighted before
        # the factorization, the absent third column (alone in the second row)
        # loses to it, and the basis is singular in exact arithmetic.
        A = [1.0 1.0 0.0; 1.0 (1.0 + 1.0e-15) 1.0]
        jb = OptimaSolver._weighted_basis(A, [1.0, 1.0, 1.0e-30])
        @test 3 in jb
        @test cond(A[:, jb]) < 10
        # a heavier column replaces a basic one where it can
        @test OptimaSolver._weighted_basis([1.0 1.0 0.0; 0.0 1.0 1.0], [1.0, 0.0, 5.0]) == [1, 3]
        # a matrix with no column outside its basis
        @test OptimaSolver._weighted_basis([2.0 0.0; 0.0 3.0], [1.0, 1.0]) == [1, 2]
    end

    @testset "a component held by no amount is priced by the dual ratio test" begin
        # Rows X and Y; X₁ carries X and is present, Y₁ and Y₂ carry Y and are on
        # their bounds: the component Y is absent. With Y₂ basic its stationarity
        # sets the multiplier of Y to −g(Y₂) = −1, and Y₁ reads as supersaturated
        # (s = −1) although nothing of Y is there. The ratio test moves the
        # multiplier to the largest value that keeps both carriers at s ≥ 0, here
        # 0, and makes Y₁ basic.
        A = [1.0 0.0 0.0; 0.0 1.0 1.0]
        g = [0.5, 0.0, 1.0]
        jb, jn = [1, 3], [2]
        S = LinearAlgebra.lu(A[:, jb]) \ A[:, jn]
        y = [-0.5, -1.0]
        s = g .+ transpose(A) * y
        @test s ≈ [0.0, -1.0, 0.0]
        OptimaSolver._degenerate_pivots!(y, s, jb, jn, S, [false, true, true], A)
        @test s ≈ g .+ transpose(A) * y
        @test s ≈ [0.0, 0.0, 1.0]
        @test sort(jb) == [1, 2] && jn == [3]
        # nothing to do on a row with a carrier off its bound, or already priced
        s2 = copy(s); y2 = copy(y)
        OptimaSolver._degenerate_pivots!(y2, s2, copy(jb), copy(jn), copy(S), [false, true, true], A)
        @test s2 == s && y2 == y
        jb3, jn3 = [1, 3], [2]
        y3 = [-0.5, -1.0]; s3 = g .+ transpose(A) * y3
        OptimaSolver._degenerate_pivots!(y3, s3, jb3, jn3, LinearAlgebra.lu(A[:, jb3]) \ A[:, jn3], [false, false, true], A)
        @test s3 ≈ [0.0, -1.0, 0.0]
        # no carrier that can take the multiplier up (every coefficient negative),
        # or none that keeps all the others non-negative: left as it is
        for (Srow, s0) in (([-1.0;;], [0.0, -1.0, 0.0]), ([1.0 -1.0], [0.0, -1.0, 0.0, -2.0]))
            ns4 = length(s0)
            jb4, jn4 = [1, 3], setdiff(1:ns4, [1, 3])
            S4 = vcat(zeros(1, length(jn4)), Srow)
            y4 = [0.0, 0.0]; s4 = copy(s0)
            OptimaSolver._degenerate_pivots!(y4, s4, jb4, jn4, S4, [false; trues(ns4 - 1)], zeros(2, ns4))
            @test s4 == s0 && y4 == [0.0, 0.0]
        end
    end

    # An ideal mixture of three species, `∂f/∂n = g⁰ + ln(n/N) + z q`, whose total
    # the parameter fixes (`Ap`) or whose potential the parameter carries (`z`).
    g⁰ = [0.3, -0.5, 1.2]
    mix!(g, n, q, p) = (g .= p.g⁰ .+ log.(n ./ sum(n)) .+ p.z .* q[1]; g)

    @testset "parameters: the total fixed by an external equation" begin
        v!(v, n, q, p) = (v[1] = q[1] - p.c; v)
        prob = OptimaParametricProblem(ones(1, 3), fill(-1.0, 1, 1), [0.0], mix!, v!; p = (g⁰ = g⁰, z = zeros(3), c = 2.0))
        r = solve_active_set(prob, AS_OPTS)
        @test r isa OptimaParametricResult{Float64}
        @test r.converged
        @test r.q ≈ [2.0] rtol = 1.0e-12
        @test r.n ≈ 2 .* exp.(-g⁰) ./ sum(exp.(-g⁰)) rtol = 1.0e-12
        @test max(r.error_opt, r.error_feas, r.error_param) <= AS_OPTS.tol
        # the derivative of the answer with respect to the data, through the solve
        f(c) = solve_active_set(
            OptimaParametricProblem(ones(1, 3), fill(-1.0, 1, 1), [0.0], mix!, v!; p = (g⁰ = g⁰, z = zeros(3), c = c)), AS_OPTS,
        ).n
        @test ForwardDiff.derivative(f, 2.0) ≈ exp.(-g⁰) ./ sum(exp.(-g⁰)) rtol = 1.0e-10
        @test (@inferred solve_active_set(prob, AS_OPTS)) isa OptimaParametricResult{Float64}
    end

    @testset "parameters: a potential closed on the composition" begin
        # `q = asinh(Σ zᵢ nᵢ)`, a Gouy–Chapman-like closure, against a bisection
        # on the potential alone: at a given `q` the composition is explicit.
        z = [1.0, -1.0, 0.0]
        v!(v, n, q, p) = (v[1] = q[1] - asinh(sum(p.z .* n)); v)
        prob = OptimaParametricProblem(ones(1, 3), zeros(1, 1), [1.0], mix!, v!; p = (g⁰ = g⁰, z = z))
        r = solve_active_set(prob, AS_OPTS)
        @test r.converged
        x(ψ) = exp.(-g⁰ .- z .* ψ) ./ sum(exp.(-g⁰ .- z .* ψ))
        lo, hi = -5.0, 5.0
        for _ in 1:200
            mid = (lo + hi) / 2
            (mid - asinh(sum(z .* x(mid)))) > 0 ? (hi = mid) : (lo = mid)
        end
        @test r.q[1] ≈ lo atol = 1.0e-12
        @test r.n ≈ x(lo) rtol = 1.0e-10
        # the same with the Hessian handed over, exact for an ideal mixture
        calls = Ref(0)
        hess(H, n, q) = (calls[] += 1; H .= LinearAlgebra.Diagonal(1 ./ n) .- 1 / sum(n); H)
        rh = solve_active_set(OptimaParametricProblem(ones(1, 3), zeros(1, 1), [1.0], mix!, v!; p = (g⁰ = g⁰, z = z, hess = hess)), AS_OPTS)
        @test rh.converged && calls[] > 0
        @test rh.n ≈ r.n rtol = 1.0e-10
    end

    @testset "parameters: start, bounds, and what is refused" begin
        v!(v, n, q, p) = (v[1] = q[1] - p.c; v)
        p = (g⁰ = g⁰, z = zeros(3), c = 5.0)
        # a parameter held at its bound cannot meet its equation: not converged,
        # and the answer is the best iterate, on the bound
        prob = OptimaParametricProblem(ones(1, 3), fill(-1.0, 1, 1), [0.0], mix!, v!; p, qlb = [0.0], qub = [3.0])
        r = solve_active_set(prob, OptimaOptions(tol = 1.0e-12, max_iter = 30); q0 = [10.0])
        @test !r.converged
        @test r.q[1] <= 3.0
        @test r.error_param >= 1.0
        @test_throws DimensionMismatch solve_active_set(prob, AS_OPTS; q0 = [1.0, 2.0])
        # verbose: one line per iteration and the verdict
        s = mktemp() do path, io
            redirect_stdout(io) do
                solve_active_set(prob, OptimaOptions(tol = 1.0e-12, max_iter = 3, verbose = true))
            end
            close(io)
            read(path, String)
        end
        @test occursin("err_opt", s) && occursin("MAX_ITER", s)
        @test_throws ArgumentError OptimaParametricProblem(zeros(0, 3), zeros(0, 1), Float64[], mix!, v!)
        @test_throws DimensionMismatch OptimaParametricProblem(ones(1, 3), ones(2, 1), [1.0], mix!, v!)
        @test_throws DimensionMismatch OptimaParametricProblem(ones(1, 3), ones(1, 1), [1.0, 2.0], mix!, v!)
        @test_throws DimensionMismatch OptimaParametricProblem(ones(1, 3), ones(1, 1), [1.0], mix!, v!; lb = zeros(2))
        @test_throws DimensionMismatch OptimaParametricProblem(ones(1, 3), ones(1, 1), [1.0], mix!, v!; qlb = zeros(2))
    end

    @testset "inference" begin
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        @test (@inferred solve_active_set(prob, AS_OPTS)) isa OptimaResult{Float64}
    end

    @testset "an unknown algorithm is refused by name" begin
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        @test_throws ArgumentError solve(prob, OptimaOptions(algorithm = :simplex))
    end
end
