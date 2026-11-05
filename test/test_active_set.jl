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

    @testset "inference" begin
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        @test (@inferred solve_active_set(prob, AS_OPTS)) isa OptimaResult{Float64}
    end

    @testset "an unknown algorithm is refused by name" begin
        prob = _as_problem(ones(1, 3), [1.0], [0.0, 1.0, 2.0])
        @test_throws ArgumentError solve(prob, OptimaOptions(algorithm = :simplex))
    end
end
