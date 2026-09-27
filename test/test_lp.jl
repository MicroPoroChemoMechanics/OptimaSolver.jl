# The linear program over pure phases: `lp_start` and the `simplex_start`
# wrapper. Every status is checked against an independent answer: small programs
# are solved by enumerating every basis, infeasible ones are built from a chosen
# Farkas vector, and the multipliers of a chemical toy are compared with the
# pure-phase potentials they must reproduce.

using LinearAlgebra

@testset "lp_start" begin
    # A deterministic generator, so a failure is reproducible from the test alone.
    seed = UInt64(0x2026_0927_0000_0001)
    nextrand() = begin
        seed ⊻= seed << 13
        seed ⊻= seed >> 7
        seed ⊻= seed << 17
        return (seed >> 11) / Float64(1 << 53)
    end

    # The optimum of a small program by brute force: every set of m columns
    # that is a basis with a nonnegative solution, and the best of them.
    function brute_force(A, g, b)
        m, n = size(A)
        best = Inf
        for cols in Iterators.product(ntuple(_ -> 1:n, m)...)
            issorted(collect(cols); lt = <=) || continue
            c = collect(cols)
            B = A[:, c]
            abs(det(B)) > 1.0e-10 || continue
            xB = B \ b
            all(>=(-1.0e-10), xB) || continue
            best = min(best, dot(g[c], xB))
        end
        return best
    end

    @testset "the optimum, against every basis" begin
        for _ in 1:200
            m = 2 + Int(floor(3 * nextrand()))
            n = 5 + Int(floor(5 * nextrand()))
            A = [nextrand() for _ in 1:m, _ in 1:n]
            x0 = zeros(n)
            for _ in 1:m
                x0[1 + Int(floor(n * nextrand()))] = nextrand() + 0.1
            end
            b = A * x0
            g = [2 * nextrand() - 1 for _ in 1:n]
            lp = lp_start(A, g, b)
            @test lp.status === :optimal
            @test lp.balance < 1.0e-12
            @test all(>=(0), lp.x)
            @test dot(g, lp.x) ≈ brute_force(A, g, b) atol = 1.0e-9
            # Complementary slackness: zero reduced cost on the basis, none
            # negative elsewhere, and y solves the basic equations.
            basic = filter(>(0), lp.basis)
            @test all(abs(lp.reduced_costs[j]) < 1.0e-9 for j in basic)
            @test all(>=(-1.0e-9), lp.reduced_costs)
            @test transpose(A[:, basic]) * lp.y ≈ -g[basic] atol = 1.0e-9
        end
    end

    @testset "degenerate programs terminate" begin
        # Ties everywhere: every ratio equal, several columns of equal cost.
        A = [1.0 1.0 1.0 0.0; 0.0 1.0 1.0 1.0]
        lp = lp_start(A, zeros(4), [1.0, 1.0])
        @test lp.status === :optimal
        @test A * lp.x ≈ [1.0, 1.0]
        # And a duplicated row, reported as redundant.
        A2 = [1.0 1.0 0.0; 1.0 1.0 0.0; 0.0 1.0 1.0]
        lp2 = lp_start(A2, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        @test lp2.status === :optimal
        @test length(lp2.redundant_rows) == 1
        @test A2 * lp2.x ≈ [1.0, 1.0, 1.0]
    end

    @testset "infeasible programs come with a Farkas vector that verifies" begin
        for _ in 1:50
            m = 2 + Int(floor(3 * nextrand()))
            n = 4 + Int(floor(6 * nextrand()))
            # Choose z, then columns with Aᵀz ≥ 0 and a b with bᵀz < 0.
            z = [2 * nextrand() - 1 for _ in 1:m]
            A = zeros(m, n)
            for j in 1:n
                a = [2 * nextrand() - 1 for _ in 1:m]
                s = dot(a, z)
                s < 0 && (a .-= (2 * s / dot(z, z)) .* z)
                A[:, j] .= a
            end
            b = [2 * nextrand() - 1 for _ in 1:m]
            bz = dot(b, z)
            bz >= 0 && (b .-= ((bz + 0.5) / dot(z, z)) .* z)
            lp = lp_start(A, zeros(n), b)
            @test lp.status === :infeasible
            @test all(>=(-1.0e-9), transpose(A) * lp.farkas)
            @test dot(b, lp.farkas) < 0
        end
    end

    @testset "a trace budget no species can supply is infeasible" begin
        # REGRESSION. A row of 55 mol next to a trace row of -1e-9 mol whose
        # coefficients are all nonnegative: no nonnegative composition gives a
        # negative total. An absolute phase-I threshold of 1e-8 called this
        # feasible and returned a vertex that violated the second row.
        A = [2.0 1.0 0.0; 0.0 1.0 1.0]
        b = [55.0, -1.0e-9]
        lp = lp_start(A, [0.0, 0.0, 0.0], b)
        @test lp.status === :infeasible
        @test lp.farkas[2] != 0
        @test simplex_start(A, [0.0, 0.0, 0.0], b) === nothing
    end

    @testset "dead columns are held at zero" begin
        A = [1.0 1.0 1.0]
        lp = lp_start(A, [0.0, 1.0, 2.0], [1.0]; dead = (1,))
        @test lp.status === :optimal
        @test lp.x ≈ [0.0, 1.0, 0.0]
        @test isnan(lp.reduced_costs[1])
    end

    @testset "the multipliers are the pure-phase potentials" begin
        # Two elements, three pure phases: one of each element (g = -1 and -2)
        # and a compound of both (g = -3.5), more stable than the pair of
        # elements it is made of (-1 - 2 = -3). The compound alone is the answer.
        A = [1.0 0.0 1.0; 0.0 1.0 1.0]
        g = [-1.0, -2.0, -3.5]
        lp = lp_start(A, g, [1.0, 1.0])
        @test lp.status === :optimal
        @test lp.x ≈ [0.0, 0.0, 1.0]
        # With the compound alone basic, y is not unique (one column, two rows);
        # the reduced cost of the compound is zero and those of the elements are
        # nonnegative whatever the minimum-norm choice.
        @test lp.reduced_costs[3] ≈ 0 atol = 1.0e-12
        @test all(>=(-1.0e-12), lp.reduced_costs)
    end

    @testset "an answer that cannot be verified is not given" begin
        # The iteration limit: two rows need two pivots at least.
        A = [1.0 0.0 1.0; 0.0 1.0 1.0]
        lp = lp_start(A, [1.0, 1.0, 1.5], [1.0, 1.0]; maxit = 1)
        @test lp.status === :iteration_limit
        @test all(iszero, lp.x) && isempty(lp.farkas)
        # simplex_start no longer turns such a program into an answer.
        @test_throws ErrorException simplex_start(A, [1.0, 1.0, 1.5], [1.0, 1.0]; maxit = 1)
        # A phase-I value that says infeasible, and a candidate vector that does
        # not prove it: undecided, never infeasible.
        z_bad = [1.0, -1.0]
        @test !OptimaSolver._is_farkas(A, [1.0, 1.0], z_bad, 1:3, 1.0e-9)
        @test OptimaSolver._phase1_verdict(1.0, 1.0, A, [1.0, 1.0], z_bad, 1:3, 1.0e-9) === :undecided
        # Within the tolerance, the rounding of a feasible program.
        @test OptimaSolver._phase1_verdict(1.0e-11, 1.0, A, [1.0, 1.0], z_bad, 1:3, 1.0e-9) === :feasible
        @test OptimaSolver._phase1_verdict(0.0, 1.0, A, [1.0, 1.0], Float64[], 1:3, 1.0e-9) === :feasible
    end

    @testset "the program of a dual Newton problem" begin
        # Three pure phases, two components, and the second component absent
        # from the budget: its row is degenerate, the species holding it are
        # dead, and its multiplier is the one the dual Newton pins.
        A = [1.0 0.0 1.0; 0.0 1.0 1.0]
        h(x, _) = zeros(length(x))
        prob = DualNewtonProblem(
            A, [-1.0, -2.0, -3.5], h;
            phases = [SolutionPhase([1], 1; always_present = true)], idx_bounded = [2, 3],
        )
        lp = lp_start(prob, [1.0, 0.0])
        @test lp.status === :optimal
        @test lp.x ≈ [1.0, 0.0, 0.0]
        @test lp.y[2] == OptimaSolver.DEGENERATE_POTENTIAL
        @test isnan(lp.reduced_costs[2]) && isnan(lp.reduced_costs[3])
        # With every component present, the same answer as the matrix form.
        @test lp_start(prob, [1.0, 1.0]).x ≈ lp_start(A, [-1.0, -2.0, -3.5], [1.0, 1.0]).x
        # An impossible budget is refused the same way.
        @test lp_start(prob, [-1.0, 1.0]).status === :infeasible
    end

    @testset "simplex_start keeps its answers" begin
        A = [1.0 1.0]
        @test simplex_start(A, [1.0, 3.0], [1.0]) == [1.0, 0.0]
        @test simplex_start(A, [1.0, 3.0], [1.0]; floor = 1.0e-10) == [1.0, 1.0e-10]
        @test simplex_start([1.0 1.0], [1.0, 1.0], [-1.0]) === nothing
    end
end

@testset "lp_start carries derivatives through its vertex" begin
    # For a fixed basis the vertex is x_B = A_B⁻¹ b, so its derivative with
    # respect to b is A_B⁻¹; the pivots are chosen on the values.
    A = [1.0 1.0 0.0; 0.0 1.0 1.0]
    g = [1.0, 0.5, 1.0]
    b0 = [2.0, 1.0]
    J = ForwardDiff.jacobian(bb -> lp_start(A, g, bb).x, b0)
    lp = lp_start(A, g, b0)
    basic = filter(>(0), lp.basis)
    Binv = inv(A[:, basic])
    @test J[basic, :] ≈ Binv atol = 1.0e-12
    @test all(iszero, J[setdiff(1:3, basic), :])
end
