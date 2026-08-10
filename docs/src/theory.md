```@meta
CurrentModule = OptimaSolver
```

# Theory

This page describes the mathematical foundations of the Optima solver.
The algorithm closely follows the C++ Optima library by Allan Leal
([github.com/reaktoro/optima](https://github.com/reaktoro/optima)) and the
reference by Leal et al. (2014).

## Problem statement

We seek the equilibrium composition $n \in \mathbb{R}^{n_s}$ minimizing the
Gibbs free energy subject to element-mass conservation:

```math
\min_{n \in \mathbb{R}^{n_s}}\ f(n, p)
\qquad
\text{subject to}
\quad
A n = b,
\quad
n \geq \ell
```

| Symbol | Meaning |
|--------|---------|
| $n_i$ | mole amount of species $i$ |
| $f(n,p)$ | Gibbs free energy (objective) |
| $A \in \mathbb{R}^{m \times n_s}$ | stoichiometric (conservation) matrix |
| $b \in \mathbb{R}^m$ | element abundance vector |
| $\ell \in \mathbb{R}^{n_s}$ | lower bounds (positivity floor $\varepsilon \approx 10^{-16}$) |
| $p$ | parameter tuple $(T, P, \mu^0, \ldots)$ |

For an ideal/dilute aqueous solution the gradient and Hessian diagonal are:

```math
\frac{\partial f}{\partial n_i} = \frac{\mu_i^0}{RT} + \ln n_i + 1,
\qquad
\frac{\partial^2 f}{\partial n_i^2} = \frac{1}{n_i}.
```

## Log-barrier interior-point method

### Barrier augmentation

To enforce $n \geq \ell$ strictly without explicit inequality constraints, a
logarithmic barrier term is added to the objective:

```math
\min_{n}\ \phi_\mu(n) := f(n) - \mu \sum_{i=1}^{n_s} \ln(n_i - \ell_i)
\qquad \text{s.t.} \quad A n = b.
```

The barrier weight $\mu > 0$ is driven to zero over an outer loop; as $\mu \to 0$
the barrier minimizer converges to the original constrained minimizer.

### KKT conditions

At a stationary point $(n^*, y^*)$ of the barrier-augmented Lagrangian

```math
\mathcal{L}(n, y;\, \mu) = \phi_\mu(n) + y^\top (A n - b)
```

the first-order KKT conditions are:

```math
e_x(n,y;\mu) := \nabla_n f(n) + A^\top y - \frac{\mu}{n - \ell} = 0
\qquad (\text{optimality, } n_s \text{ equations})
```

```math
e_w(n) := A n - b = 0
\qquad (\text{feasibility, } m \text{ equations})
```

where division by $n - \ell$ is component-wise.
[`KKTResidual`](@ref) stores $(e_x, e_w)$ together with
$\|e_x\|_\infty$ and $\|e_w\|_\infty$.

### Convergence criterion

The solver declares convergence when

```math
\max\!\bigl(\|e_x\|_\infty,\, \|e_w\|_\infty\bigr) < \texttt{tol}.
```

Near-bound species (slack $n_i - \ell_i \lesssim 10^{-8}\,\max_j(n_j - \ell_j)$)
with $\partial f/\partial n_i \geq 0$ are excluded from $\|e_x\|_\infty$: their
log-barrier term $-\mu/(n_i - \ell_i)$ diverges as $n_i \to \ell_i$, making the
optimality norm artificially large even when the species is correctly absent.

## Newton step via Schur complement

### KKT linear system

At each Newton iteration we solve the $(n_s + m) \times (n_s + m)$ saddle-point
system:

```math
\begin{pmatrix} H & A^\top \\ A & 0 \end{pmatrix}
\begin{pmatrix} \delta n \\ \delta y \end{pmatrix}
=
-\begin{pmatrix} e_x \\ e_w \end{pmatrix}
```

where $H = \operatorname{diag}(h)$ is the barrier-augmented Hessian diagonal:

```math
h_i = \frac{\partial^2 f}{\partial n_i^2} + \frac{\mu}{(n_i - \ell_i)^2}.
```

### Schur complement reduction

Because $H$ is diagonal, $\delta n$ can be eliminated analytically.
From the first block row: $\delta n = -H^{-1}(e_x + A^\top \delta y)$.
Substituting into $A\,\delta n = -e_w$ gives the $m \times m$ Schur system:

```math
S\,\delta y = e_w - A H^{-1} e_x,
\qquad
S = A H^{-1} A^\top \in \mathbb{R}^{m \times m}.
```

Once $\delta y$ is found, $\delta n$ is recovered by back-substitution.

**Implementation.** $S$ is built as a single BLAS GEMM:
$S = \tilde{A} A^\top$ where $\tilde{A}_{ik} = A_{ik}/h_k$ is computed
in-place. The RHS is then the BLAS GEMV $e_w - \tilde{A}\,e_x$, and
$\delta n$ is recovered by the BLAS GEMV $A^\top\!\delta y$.
All three operations reuse the same pre-allocated buffer $\tilde{A}$
(field `AoverH` of [`NewtonStep`](@ref)).

The total cost is $O(n_s m^2)$ to build $S$ and $O(m^3)$ to factor it.
Since $m$ (number of conserved elements) is typically $\leq 15$, this is far
cheaper than factoring the full $(n_s + m)$-dimensional system.

### Numerical conditioning

Two techniques stabilize the Schur solve when some conservation rows correspond
to absent species (zero element budget):

1. **Tikhonov regularization**: add $\delta_{\rm tik} I$ with
   $\delta_{\rm tik} = 10^{-14}\max_i S_{ii}$ before factorization,
   preventing near-zero pivots.
2. **Diagonal equilibration**: scale row and column $i$ by $1/\sqrt{S_{ii}}$
   so all diagonal entries equal 1, reducing the condition number from
   $O(10^7)$ to $O(1)$ in titration-type problems.

## Conservation matrix canonicalization

[`Canonicalizer`](@ref) decomposes $A$ via QR with column pivoting:

```math
A Q = [B \;\; N], \qquad B \in \mathbb{R}^{m \times m}\text{ full rank}.
```

The LU factorization of $B$ is cached and reused across Newton steps, reducing
each back-substitution to $O(m^2)$ rather than $O(m^3)$.
When $A$ is fixed across a sequence of solves (e.g. a temperature scan), pass
the pre-built `Canonicalizer` to `solve` to skip the QR entirely.

## Fraction-to-boundary step limit

Before the line search, the full Newton step is scaled to keep all components
strictly above their lower bounds:

```math
\alpha_{\max} = \min_{i:\, \delta n_i < 0}
\frac{-\tau\,(n_i - \ell_i)}{\delta n_i},
\qquad \tau = 0.995.
```

Additionally, **unstable variables** — species that are near their lower bound
($n_i - \ell_i \lesssim 10^{-8}\max_j(n_j - \ell_j)$) with $e_{x,i} \geq 0$
(gradient pushing toward the bound) — receive a further reduced step:

```math
\delta n_i \;\leftarrow\; \max\!\Bigl(\delta n_i,\;
-\tfrac{\tau}{2}(n_i - \ell_i)\Bigr).
```

This prevents numerical oscillations when a species is in the process of
precipitating or dissolving completely.

## Filter line search

The line search follows Wächter & Biegler (2006). A *filter* is a Pareto set
of pairs $(\theta, \varphi) = (\|An - b\|_1,\, f(n))$; a new point is
*acceptable to the filter* if it is not dominated by any entry already in the
filter.

Starting from $\alpha = \alpha_{\max}$, the algorithm backtracks with factor
$\beta = 0.5$ until the candidate
$(n + \alpha\,\delta n,\; y + \alpha\,\delta y)$ satisfies:

- **Filter acceptance**: not dominated by any entry in the current filter, **and**
- **Sufficient decrease** on the barrier objective (Armijo condition):
```math
\phi_\mu(n + \alpha\,\delta n) \;\leq\;
\phi_\mu(n) + \texttt{ls\_alpha}\cdot\alpha\cdot\nabla\phi_\mu^\top\delta n,
```
  **or** a sufficient feasibility decrease:
```math
\theta(n + \alpha\,\delta n) \;\leq\;
(1 - \texttt{ls\_alpha})\,\theta(n).
```

When the current iterate is already feasible ($\theta \approx 0$), the filter
is bypassed and only the Armijo condition on $\phi_\mu$ is checked, switching
the method to a pure descent algorithm for the final convergence phase.

## Outer barrier loop

The outer loop reduces $\mu$ on a geometric schedule:

```math
\mu_{\text{new}} = \max(\mu_{\min},\; \rho\,\mu),
\qquad \rho = \texttt{barrier\_decay} = 0.1.
```

The inner Newton loop for each fixed $\mu$ runs until the KKT error satisfies

```math
\max(\|e_x\|_\infty,\,\|e_w\|_\infty) < \max(\texttt{tol},\; \mu),
```

so the inner tolerance tightens automatically as $\mu \to 0$, avoiding
unnecessary Newton iterations in the early (exploratory) phase.

## Sensitivity analysis

At convergence $(n^*, y^*)$, the implicit function theorem applied to the
KKT system $F(n, y;\, c) = 0$ gives

```math
\frac{\partial}{\partial c}
\begin{pmatrix} n^* \\ y^* \end{pmatrix}
= -J^{-1} \frac{\partial F}{\partial c},
\qquad
J = \begin{pmatrix} H & A^\top \\ A & 0 \end{pmatrix}.
```

Two parameter families are of direct chemical interest:

### Response to element budgets $\partial n^*/\partial b$

The right-hand side for perturbation of $b_j$ is
$\partial F/\partial b_j = (0;\, -e_j)$, giving:

```math
S\, \frac{\partial y^*}{\partial b_j} = -e_j,
\qquad
\frac{\partial n^*}{\partial b_j} = -H^{-1} A^\top \frac{\partial y^*}{\partial b_j}.
```

**Sanity check**: summing over species $i$,
$\sum_i \partial n_i^*/\partial b_j = 1$ — the extra mole of element $j$ is
fully redistributed among the species.

### Response to standard potentials $\partial n^*/\partial(\mu_k^0/RT)$

The right-hand side for perturbation of $\mu_k^0/RT$ is
$\partial F/\partial (\mu_k^0/RT) = (e_k;\, 0)$, giving:

```math
S\, \frac{\partial y^*}{\partial \mu_k^0} = -\frac{A_{:k}}{h_k},
\qquad
\frac{\partial n^*}{\partial \mu_k^0} =
-H^{-1}\!\left(e_k + A^\top \frac{\partial y^*}{\partial \mu_k^0}\right).
```

**Implementation.** Both sensitivity matrices are computed with a single
batched solve (BLAS TRSM) followed by a BLAS GEMM, rather than $n_s$
sequential scalar solves:

```math
\frac{\partial Y^*}{\partial \mu^0} = S^{-1} \left(-\tilde{A}\right),
\qquad
\frac{\partial N^*}{\partial \mu^0} = -H^{-1}\!\left(I + A^\top \frac{\partial Y^*}{\partial \mu^0}\right),
```

where $\tilde{A}_{ik} = A_{ik}/h_k$ (the same buffer built during the last Newton step).
Using a matrix right-hand side triggers BLAS level-3 (TRSM + GEMM) instead of $n_s$
level-2 (TRSV + GEMV) calls — a significant speedup when $n_s \gtrsim 20$.

The total cost is $O(n_s m^2 + n_s^2 m)$ — negligible compared to the solve itself.

## Variable scaling in the SciML interface

The [`OptimaOptimizer`](@ref) SciML interface automatically scales each species
by its starting value $s_i = \max(n_i^{(0)}, 10^{-10})$:

```math
\tilde{n}_i = n_i / s_i,
\qquad
\tilde{A}_{ij} = s_j A_{ij}.
```

This transforms the scaled problem so all $\tilde{n}_i = O(1)$ at the starting
point, making the Schur complement $\tilde{A} H^{-1} \tilde{A}^\top$ well-conditioned
across the multi-decade concentration ranges typical in chemical speciation
(e.g. pH 1–13 where $[\mathrm{H}^+]$ varies over 12 orders of magnitude).
The scaling is transparent: the returned solution is always in the original units.

## References

- Allan Leal, *Optima* — C++ library for chemical equilibrium optimization,
  ETH Zürich.
  [github.com/reaktoro/optima](https://github.com/reaktoro/optima)

- Leal, A.M.M., Blunt, M.J., LaForce, T.C. (2014).
  Efficient chemical equilibrium calculations for geochemical speciation and
  reactive transport modeling.
  *Geochimica et Cosmochimica Acta*, **131**, 301–322.
  <https://doi.org/10.1016/j.gca.2014.01.006>

- Wächter, A., Biegler, L.T. (2006).
  On the implementation of an interior-point filter line-search algorithm for
  large-scale nonlinear programming.
  *Mathematical Programming*, **106**(1), 25–57.
  <https://doi.org/10.1007/s10107-004-0559-y>
```
