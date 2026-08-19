@testset "Canonicalizer" begin
    # 2 elements, 4 species: H₂O, H⁺, OH⁻, H₂
    # Conservation: [H, O] — rows correspond to elements
    A = Float64[
        2  1  1  2   # H: 2 in H₂O, 1 in H⁺, 1 in OH⁻, 2 in H₂
        1  0  1  0   # O: 1 in H₂O, 0 in H⁺, 1 in OH⁻, 0 in H₂
    ]

    can = Canonicalizer(A)

    @test can.m == 2
    @test can.ns == 4
    @test can.rank_A == 2
    @test length(can.jb) == 2
    @test length(can.jn) == 2

    # Basic columns must span the row space
    @test rank(A[:, can.jb]) == 2

    # Permutation consistency
    @test sort(vcat(can.jb, can.jn)) == 1:4

    # Schur complement is symmetric positive semi-definite
    h = ones(4)
    S = OptimaSolver.schur_complement(can, h)
    @test S ≈ S'
    @test all(eigvals(S) .>= -1.0e-12)

    # solve_B and solve_Bt: round-trip
    rhs = ones(2)
    x = OptimaSolver.solve_B(can, rhs)
    @test A[:, can.jb] * x ≈ rhs atol = 1.0e-12

    xt = OptimaSolver.solve_Bt(can, rhs)
    @test A[:, can.jb]' * xt ≈ rhs atol = 1.0e-12
end

@testset "the basis is chosen independently of column scaling" begin

    # The caller scales each column by the starting value of its variable, and
    # warm-starting from a converged equilibrium spreads those over some ten
    # orders of magnitude. Scaling by a positive diagonal cannot change which
    # columns are independent, so the canonicalizer must return the same rank —
    # it used to lose one, build a non-square basic block, and fail inside LAPACK
    # with "matrix is not square".
    A = Float64[
        1 0 0 1 2 0;
        0 1 0 1 0 3;
        0 0 1 0 1 1
    ]
    m = size(A, 1)
    @test Canonicalizer(A).rank_A == m

    s = [1.0, 1.0e-16, 1.0, 1.0e-14, 1.0e-15, 1.0]
    can = Canonicalizer(A .* s')
    @test can.rank_A == m
    @test size(can.B) == (m, m)

    # and the factorization it hands out is usable
    rhs = [1.0, 2.0, 3.0]
    @test (can.B * OptimaSolver.solve_B(can, rhs)) ≈ rhs

    # A truly redundant conservation law is reported, not discovered later
    # inside a factorization.
    Ar = vcat(A, A[1, :]' .+ A[2, :]')
    @test_throws ArgumentError Canonicalizer(Ar)

end
