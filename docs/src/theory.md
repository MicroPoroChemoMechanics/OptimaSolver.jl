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

## Convexity, and what it guarantees

Everything below rests on one structural fact, so it is worth establishing it
rather than assuming it.

### The ideal mixing term is convex

Write the ideal part of the Gibbs energy, in $RT$ units, as

```math
\varphi(n) \;=\; \sum_{i} n_i \ln \frac{n_i}{N},
\qquad N = \sum_j n_j ,
```

on the open positive orthant $n > 0$. Then

```math
\frac{\partial \varphi}{\partial n_i}
  = \ln n_i + 1 - (\ln N + 1) = \ln\frac{n_i}{N},
\qquad
\frac{\partial^2 \varphi}{\partial n_i \partial n_j}
  = \frac{\delta_{ij}}{n_i} - \frac{1}{N}.
```

**Claim.** $\nabla^2\varphi \succeq 0$, with null space spanned by $n$ itself.

*Proof.* For any $v \in \mathbb{R}^{n_s}$,

```math
v^\top \nabla^2\varphi\, v
  \;=\; \sum_i \frac{v_i^2}{n_i} \;-\; \frac{1}{N}\Bigl(\sum_i v_i\Bigr)^{2}.
```

By Cauchy–Schwarz, writing $v_i = \bigl(v_i/\sqrt{n_i}\bigr)\sqrt{n_i}$,

```math
\Bigl(\sum_i v_i\Bigr)^{2}
  \;\le\; \Bigl(\sum_i \frac{v_i^{2}}{n_i}\Bigr)\Bigl(\sum_i n_i\Bigr)
  \;=\; N \sum_i \frac{v_i^{2}}{n_i},
```

so $v^\top \nabla^2\varphi\, v \ge 0$. Equality in Cauchy–Schwarz holds exactly
when $v_i/\sqrt{n_i} \propto \sqrt{n_i}$, that is $v \propto n$. $\;\square$

The same computation applies verbatim to the molality convention, where the
solutes carry $\ln\bigl(n_i/(n_w M_w)\bigr)$ and the solvent $\ln x_w$: the
Hessian is again $\operatorname{diag}(1/n_i)$ minus a rank-one term built on the
aqueous block, and the inequality is the same.

### A pure phase contributes a *linear* term

A phase of fixed composition has unit activity, so its potential
$\mu_i = \mu_i^0$ does not depend on $n_i$ and it contributes $\mu_i^0 n_i$ to
$f$. Hence

```math
\frac{\partial^2 f}{\partial n_i^2} \;=\; 0
\qquad\text{for every pure phase,}
```

which is convex, and is also the source of every numerical difficulty in this
solver: the Hessian is singular along those directions by construction, not by
accident.

### Consequences

$f$ is convex on $\{n > 0\}$ and the feasible set $\{An = b,\ n \ge \ell\}$ is a
polyhedron, hence convex. Therefore:

1. **every local minimum is global**, and the minimizer is unique up to the null
   directions above;
2. the constraints being affine, the linearity constraint qualification holds at
   every feasible point, so the **KKT conditions are necessary *and* sufficient**.

Point 2 is what makes verification possible: a candidate $n$ together with
multipliers $(y, z)$ satisfying the KKT system is a *proof* of global optimality,
not evidence for it. Point 1 says that a solver returning different answers from
different starting points is not finding different local minima — it is stopping
short of stationarity.

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

### The optimality error, and why the obvious one cannot work

Let $s_i := n_i - \ell_i > 0$ denote the slacks and

```math
g_L(n,y) \;:=\; \nabla_n f(n) + A^\top y
```

the gradient of the Lagrangian without the bound term. The barrier stationarity
condition is

```math
g_{L,i}(n,y) \;=\; \frac{\mu}{s_i},
\qquad i = 1,\dots,n_s. \tag{$\star$}
```

#### The residual form diverges at the bounds

Measuring $(\star)$ as it stands, i.e. by
$r_i := g_{L,i} - \mu/s_i$, is unusable, and not marginally so.

**Claim.** Along any sequence approaching a solution at which species $i$ is
absent ($s_i \to 0$), $|r_i|$ is unbounded for a fixed *relative* error in $s_i$.

*Proof.* Let $s_i = s_i^{\ast}(1+\eta)$ with $s_i^{\ast}$ the exact slack and
$\eta$ the relative error. At the exact point $g_{L,i} = \mu/s_i^{\ast}$, so

```math
r_i \;=\; \frac{\mu}{s_i^{\ast}} - \frac{\mu}{s_i^{\ast}(1+\eta)}
      \;=\; \frac{\mu}{s_i^{\ast}}\,\frac{\eta}{1+\eta}
      \;\xrightarrow[\;s_i^{\ast}\to 0\;]{}\; \infty
```

for any fixed $\eta \neq 0$. $\;\square$

Numerically: with $\mu = 10^{-4}$ and $s_i = 10^{-15}$ — an ordinary state of
affairs for a mineral that is not present — the term $\mu/s_i$ alone is
$10^{11}$. On a cement equilibrium this quantity started at $4.5\times10^{11}$
and never fell below $2\times 10^{9}$, so `tol = 1e-4` failed exactly as
`tol = 1e-10` did: **no tolerance was attainable**, and the failure was not one
of accuracy but of the criterion itself.

Two things followed, both silent. The barrier update
[`should_reduce_barrier`](@ref) requires the inner loop to reach
$\max(\texttt{tol}, \mu)$ before $\mu$ may fall, so $\mu$ stayed at its initial
$10^{-4}$ for the whole solve; and every solve exhausted `max_iter` and returned
whatever iterate it had reached, while the *feasibility* error $\|An-b\|_\infty$
was meanwhile reaching $10^{-14}$.

The guard intended to prevent this — excluding variables within
$10^{-6}\,\ell_i$ of their bound — could not fire either: with
$\ell_i = 10^{-16}$ that threshold is $10^{-22}$, so nothing was ever excluded.

#### The complementarity form is bounded and equivalent

Introduce the bound multipliers $z \in \mathbb{R}^{n_s}$ explicitly. The KKT
system of the barrier subproblem is

```math
\underbrace{g_L(n,y) - z = 0}_{\text{dual feasibility}},
\qquad
\underbrace{s_i z_i = \mu}_{\text{complementarity}},
\qquad
\underbrace{An - b = 0}_{\text{primal feasibility}},
\qquad s, z > 0 .
```

Dual feasibility *defines* $z = g_L$; substituting it into complementarity gives
the equivalent scalar conditions

```math
E_i(n,y) \;:=\; s_i\, g_{L,i}(n,y) \;-\; \mu \;=\; 0 . \tag{$\star\star$}
```

**Claim.** $(\star)$ and $(\star\star)$ have the same solutions, and $E_i = s_i r_i$
is bounded where $r_i$ is not.

*Proof.* Since $s_i > 0$ strictly at any interior point, $E_i = s_i r_i$ vanishes
iff $r_i$ does, which gives the equivalence. For the boundedness, $g_{L,i}$ is
finite at any interior point and $s_i$ is bounded above by the total amount, so
$|E_i| \le s_i|g_{L,i}| + \mu < \infty$; whereas the computation above shows
$|r_i| = |E_i|/s_i \to \infty$ as $s_i \to 0$ at fixed $E_i$. $\;\square$

This is the measure Ipopt reports, with the bound multiplier eliminated by dual
feasibility rather than carried as a variable (Wächter & Biegler 2006, §3.5). On
the same cement equilibrium it reads $5.8\times10^{-3}$ where the residual form
read $4.5\times 10^{11}$ — fourteen orders of magnitude, on the same iterate.

The feasibility error is **not** rescaled: $An - b$ is in moles and already
means something absolute.

### Convergence criterion

```math
\max\!\bigl(\max_i |E_i|,\ \|An-b\|_\infty\bigr) < \texttt{tol}.
```

### What this method does not do

Making the error measure meaningful does not make the iteration converge, and it
is honest to separate the two.

With the error now readable, the trace on a cement equilibrium shows the step
capped by the fraction-to-boundary rule of
[Fraction-to-boundary step limit](@ref) at **every** iteration — $\alpha$ equal
to $\alpha_{\max}$ throughout, falling below $10^{-2}$ — so convergence is
linear and `tol` is not reached. The mechanism is the one identified above: a
pure phase has $\partial^2 f/\partial n_i^2 = 0$ exactly, so its Newton direction
is governed by the barrier term alone and is large, and the boundary cuts it.

Three remedies were implemented and measured, and all three are *rejected*
because they moved the chemistry the wrong way:

| remedy | effect on the KKT error | effect on the answer |
|:--|:--|:--|
| reduce $\mu$ on stalling | barrier finally falls | worst element imbalance $1.1 \to 15.2$ mol |
| project each iterate onto $An=b$ | feasibility $10^{-7}\to10^{-9}$ | pore solution pH $12.58 \to 14.32$ |
| primal-dual with explicit $z$ | feasibility to $10^{-15}$ | aluminate assemblage lost entirely |

The structural fix is not a modification of this iteration but a different
formulation: solving the KKT system directly in the space of element potentials,
where the aqueous species are parameterized by $\ln n$ so that positivity is
automatic and the fraction-to-boundary rule has nothing to act on. That is
implemented in `ChemistryLab.DualEquilibriumSolver`, which uses this solver to
reach a neighborhood and then certifies the result.

## Newton on the KKT system in multiplier space

The interior-point method above minimizes `f` by walking the interior. This
section describes the alternative the package also provides,
[`dual_newton_solve`](@ref), which solves the KKT conditions directly. It is the
Brinkley–Karpov formulation of the geochemical Gibbs-minimization codes, stated
here for the general convex program.

### The formulation

Let `u := -A^\top y`. The stationarity conditions split by the nature of the
variable:

```math
\text{interior } i:\quad h_i(x) = u_i - g_i
\qquad\Longleftrightarrow\qquad
x_i \text{ recovered from its own condition,}
```

```math
\text{bounded } i:\quad g_i = u_i \ \text{ if } x_i > 0,
\qquad g_i \ge u_i \ \text{ if } x_i = 0 .
```

The second line is a **stability criterion**: a bounded variable is positive
exactly when `u_i - g_i` vanishes, and zero when that index is negative. In a
chemical system it is the statement that a phase is present iff it is saturated.

### Why it is well conditioned

Parameterizing the interior variables by ``w = \ln x`` makes their positivity
automatic. The fraction-to-boundary rule of
[Fraction-to-boundary step limit](@ref) — which caps the interior-point step at
*every* iteration on a cement equilibrium — therefore has nothing to act on for
them. Only the bounded variables carry a bound, and they are handled by an
active set, exactly.

This is **not** the log reparameterization that `variable_space = Val(:log)`
performs, and the distinction matters. Composing the objective with `exp` gives

```math
\frac{\partial^2 (f\circ\exp)}{\partial w_i^2}
  = x_i\bigl(\nabla f_i + 1\bigr),
```

which is negative wherever ``\nabla f_i < -1`` — for a chemical potential of
order ``-200``, everywhere. `f ∘ exp` is not convex. Here the logarithm is
applied to the KKT **equations**, solved as a square nonlinear system;
convexity of the original problem is what makes that system's solution unique.

### The reference variable

One interior variable may have `h_i` **bounded above**, so that its condition
`h_i = u_i - g_i` has no solution for an arbitrary `y`. In a chemical system
that is the solvent: its activity is a mole fraction, so ``\ln a \le 0`` always.
Inverting it is not merely slow, it can be **infeasible**, and an inner loop
that included it can never report convergence — which then invalidates the outer
Jacobian, since that Jacobian is derived on the assumption that the inner
conditions hold exactly.

Such a variable is declared through `j_ref` and carried by the outer system,
where the equality constraints determine it.

### Phases with no solvent

The paragraph above is written for a phase that *has* a solvent. A solid
solution has none: **every** member is a mole fraction, so every `h_i` is bounded
above by zero, and the argument that singled out one variable now applies to all
of them at once. Inverting `h_i = u_i - g_i` member by member is then impossible
whenever the right-hand side is positive, and an iteration that tries answers by
sending the member's amount to infinity.

Fixing the reference's amount and inverting the others does not repair it. With
`x_\text{ref}` held and ideal mixing, the fixed point gives
``x_i = \Sigma\,e^{u_i-g_i}`` with ``\Sigma = x_\text{ref} + \sum_{i\neq\text{ref}} x_i``,
hence

```math
\Sigma = \frac{x_\text{ref}}{1 - S},
\qquad S = \sum_{i \neq \text{ref}} e^{u_i - g_i},
```

which has **no positive solution once ``S \ge 1``** — precisely the situation in
which the phase is supersaturated at the current `y` and the reference cannot
hold it back. Measured on a cement, the four C-S-H end-members all ran to the
clamp at ``e^{20} = 4.85\times10^{8}`` mol, and the outer Jacobian was then
computed on that.

What the potentials *do* determine, always, is the **composition**. Stationarity
of every member gives ``x_i/N = e^{u_i - g_i - \ln\gamma_i}``, so

```math
\frac{x_i}{N} = \operatorname{softmax}_i\bigl(u_i - g_i - \ln\gamma_i\bigr),
\qquad
\underbrace{\log\!\sum_i e^{\,u_i - g_i - \ln\gamma_i}}_{\text{the phase equation}} = 0 ,
```

with the phase total `N` as the outer unknown. Both are evaluated with the
maximum factored out, so nothing overflows for any `y`. The phase equation is
the same expression as the tangent-plane measure below, which is what one would
want: a phase is admitted and held stationary by one quantity, not by two that
could disagree.

`SolutionPhase` carries this as `mole_fraction`. It must stay `false` for an
aqueous phase: there the solutes' activities are molalities, unbounded above,
and their dependence on `u` is exactly what determines the multipliers — remove
it and `y` is barely determined at all.

### Degenerate components

A row `k` with ``b_k = 0`` need not be degenerate. With ``x \ge 0``,

```math
\sum_i A_{ki} x_i = 0
\quad\Longrightarrow\quad
x_i = 0 \ \ \forall i:\ A_{ki}\neq 0
```

**only if the non-zero entries of the row share a sign** — a sum of non-negative
terms vanishes only term by term. A row with entries of both signs permits
cancellation and forces nothing. [`degenerate_components`](@ref) implements
exactly this test.

The distinction is not academic: the `H+` row of a chemical system carries `+1`
for `H+` and `−1` for `OH-`, so its zero total is the ordinary state of pure
water. Declaring it degenerate removes the entire acid–base system and returns
pH 7.000 with the solid undissolved.

Where a row *is* degenerate its multiplier is determined by nothing and the
Jacobian is singular in that direction. The row is then replaced by
``y_k = \texttt{DEGENERATE\_POTENTIAL}``, which keeps the system square and makes
the pinning of those variables consistent with their stationarity rather than
merely imposed on it.

### Termination

The outer loop admits **one** variable per round, the most violated, and records
the active sets it has visited. Since there are finitely many subsets and each
round either terminates or visits a new one, the loop terminates. Admitting a
batch instead feeds a cycle in which a variable is admitted, driven negative,
dropped, and readmitted — observed on a cement without limestone as a solve
converged to `2e-12` of the *wrong* subproblem, with an excluded variable
violated by 10.9.

### What an active set must satisfy, and how it is searched

Two conditions decide whether an active set can support a solution at all, and
both are consequences of the KKT system rather than choices.

A bound-constrained variable held active imposes `u_i = g_i`, i.e.
``a_i^\mathsf{T} y = -g_i`` — one **linear** equation in `y`. A mole-fraction
mixing phase imposes ``\log\sum_i e^{u_i-g_i} = 0``, one more, nonlinear but still
a condition on `y` alone. A phase with a solvent does not, since its reference
equation involves the composition. Hence

```math
\#\{\text{active bounded}\} + \#\{\text{active mole-fraction phases}\} \;\le\; m ,
```

which is Gibbs' phase rule in the form the dual system takes, and in addition
``\operatorname{rank} A[:,\text{active}] = |\text{active}|`` — two dependent
columns demand a fixed relation between their `g_i` that no database satisfies,
which is exactly what two polymorphs of one composition amount to. An active set
breaking either condition has no solution: the residual cannot reach zero for any
iterate, and the least-squares step spreads the violation over the rows instead of
removing it.

Complementarity supplies the leaving rule. For a bound-constrained variable the
conditions are ``x_i \ge 0``, ``s_i \le 0``, ``x_i s_i = 0`` with
``s_i = u_i - g_i - h_i``. Testing only ``x_i \to 0`` covers one half: a variable
held active while **undersaturated** violates complementarity as plainly, and no
Newton iteration repairs it, because its own equation ``s_i = 0`` is the one that
cannot hold. That test is meaningful only on a point that solves the current
subproblem — while the inner Newton is still working, ``s_i`` is a transient.

The search over sets is a **descent method**, and the quantity it descends is the
KKT error of the whole problem — stationarity, element balance, and the worst
violation among the phases held absent — not the residual of the subproblem the
current set defines. The distinction decides the outcome: a set that omits a phase
the solution needs solves its own equations exactly, so ranking states on that
residual rewards leaving phases out. The best state visited is what the solver
returns.

### Exchange, not rejection

An admission that fails to converge is not evidence against the variable
admitted. The inner Newton stops the moment an active variable falls below its
bound, and that is the *departing* variable announcing itself: the entrant stays,
the drop list removes the other, and the two together are the active-set
exchange.

Reading that failure as the entrant's fault is what an earlier version did, and
the consequence was silent. On a cement, ettringite and monosulphate compete for
the same sulfate, so admitting one necessarily drives the other out; rejecting
ettringite permanently let the solve converge — to `2e-12` stationarity and
`5e-12` element balance — onto an assemblage in which ettringite was **absent and
supersaturated by 14.8**. Only the certificate caught it.

The entrant is reconsidered only when the Newton failed with *nothing* leaving,
which is the genuine over-determination this guard was written for: two bound
variables declared stationary whose formulas are dependent modulo the mixing
phases, where the residual cannot reach zero at all. Even then the veto lasts
only as long as the context that produced it, and a vetoed variable that is still
supersaturated prevents the run from being reported as converged — the candidate
list is filtered, the KKT conditions are not.

### The certificate

[`kkt_certificate`](@ref) checks the conditions at any point, whatever produced
it. For a convex program they are sufficient, so a certificate is a proof of
global optimality.

Two splits decide whether the check means anything. A variable **at its bound**
obeys the inequality, not the equality: imposing the equality on an amount held
at `1e-16` whose stationarity value is `e^{-300}` misstates `h_i` by 263 units,
and the check then reports a residual of 74 for a point solved to `5e-12`. And a
variable carrying a **degenerate component** is excluded from both tests, for the
reason above.

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

The rank and the basis order are two questions, and each needs its own answer.

For the **rank**, the column scaling is noise:
``\operatorname{rank}(A\,\mathrm{diag}(s)) = \operatorname{rank}(A)`` for any
positive `s`. But the pivoted-QR test compares each pivot to the largest, and the
SciML interface scales columns by each variable's starting value; warm-starting
from a converged equilibrium spreads those over ten orders of magnitude and the
smallest honest pivot falls below the tolerance. The rank came out one short, `B`
was built with `m-1` columns, and the failure surfaced as a "matrix is not square"
error from LAPACK. It is therefore read off a column-equilibrated copy.

For the basis **order**, that same scaling is exactly the information wanted. The
null-space step asks the basic variables to absorb the infeasibility, through
``\delta n_b = B^{-1}(-e_w)``, so they must be the ones that can move — the
abundant species, not a trace ion pinned at its bound. Equilibrating before
pivoting throws that away: on an LC³ equilibrium the basis then held species at
`1e-16`, the particular solution asked them for `1e4` mol, and the dual step came
back at `1e31`. The order is taken from the matrix as given, which is how Optima
prioritizes its own basis.

A genuinely rank-deficient `A` — a conservation law that is a combination of the
others — is reported as such instead of being discovered inside a factorization.

## The reduced Hessian must be equilibrated

With the canonical partition, the step may be taken in the null space of `A`:

```math
\delta n = \delta n_p + Z\,\delta z,
\qquad
Z^\mathsf{T} H Z = R^\mathsf{T}\mathrm{diag}(h_b)R + \mathrm{diag}(h_n) .
```

`h` is the barrier-augmented curvature ``\nabla^2 f + \mu/s^2``. In a chemical
system the amounts span ten orders of magnitude, so `h` spans twenty and more — on
a cement it ran from 2.5 on the solvent to ``10^{27}`` on a species at its bound.
The reduced Hessian inherits that spread, its condition number passes anything
Float64 can carry, and the Cholesky then *succeeds* while returning a direction
that is noise: ``\|\delta n\|_\infty = 4.5\times10^{17}``,
``\|\delta y\|_\infty = 3.1\times10^{43}``, and `NaN` two iterations later.

Scaling by the square root of the diagonal is exact — with
``D = \mathrm{diag}(\sqrt{\operatorname{diag} K})``, solving
``(D^{-1}KD^{-1})(D\,\delta z) = D^{-1} r`` returns the same ``\delta z`` — and it
normalizes every diagonal entry to one. The Schur-complement branch already did
this; the null-space branch, which is the default, did not.

## Convergence, and the level it is measured at

The barrier subproblem's stationarity is `sᵢ (∇f + Aᵀy)ᵢ = μ`, so

```math
E_\mu = \max_i \bigl| s_i (\nabla f + A^\mathsf{T} y)_i - \mu \bigr|
```

vanishes at the solution of *that* subproblem, for **any** `μ`. It is the right
quantity to decide when to tighten the barrier, and the wrong one to decide that
the problem is solved: a point satisfying it at `μ = 10^{-4}` is `O(10^{-4})` from
the optimum. Convergence is therefore declared on

```math
E_0 = \max\Bigl( \max_i \bigl| s_i (\nabla f + A^\mathsf{T} y)_i \bigr|,\; \|An-b\|_\infty \Bigr),
```

the same residual at `μ = 0` — Ipopt's `E_0`, Wächter & Biegler (2006),
Algorithm A, step 2. It is the true KKT error and cannot be satisfied at a loose
barrier.

That test only becomes attainable with a barrier schedule the inner Newton can
keep up with. At `κ_μ = 0.1` the barrier outruns it and `E_0` plateaus just above
the tolerance without crossing it — 312 iterations to reach `9.999\times10^{-11}`
against `10^{-10}`. At Ipopt's `κ_μ = 0.2` the same problem converges in 30.

## Feasibility of the starting point

The barrier method must be started **on** ``\{A n = b,\; n \ge \ell\}``, not near
it, and the reason is the line search. The filter is bypassed only when the current
point is feasible; while it is not, a step is accepted either because the
constraint violation drops by a relative ``\alpha_{\rm ls}`` — unreachable once
the fraction-to-boundary limit is itself below ``\alpha_{\rm ls}`` — or by Armijo
along a direction partly spent restoring feasibility, which need not be a descent
direction at all. Measured on an LC³ equilibrium, a start carrying
``\|An-b\|_\infty = 6.8\times10^{-3}`` had all forty trial steps refused at every
barrier level, and the solve reported `MaxIters` on the point it began at.

Positivity is enforced first and feasibility after, so the clamp cannot undo the
projection. Two exact routes are then tried in order:

1. **Solve for the basic amounts** given the others, ``B n_b = b - N n_n``. One
   triangular solve on a factorization that already exists, and this is how Optima
   does it.
2. **Non-negative least squares** on the slacks ``v = n - \ell``, by Lawson–Hanson.
   Needed because a component total can be negative — the `H⁺` row of a cement is
   ``-2.1`` mol — and no basis of abundant species can produce it. NNLS terminates
   finitely and its residual is zero whenever the budget is attainable, which it is
   whenever the budget came from a real composition.

NNLS returns a solution supported on at most ``\operatorname{rank}(A)`` variables,
the rest at exactly their bound with zero slack — unusable as a barrier start,
since the first negative step component gives ``\alpha = 0``. They are lifted to
the slack the barrier itself would give them: a variable held at its bound settles
at ``s = \mu/(\nabla f)_i``, about ``10^{-6}`` for the initial ``\mu = 10^{-4}``
with chemical potentials of order ``10^2``–``10^3`` in RT units. The matter that
adds is then removed exactly, on the support, by a minimum-norm correction, so the
point is feasible to machine precision **and** strictly interior.

Alternating projections onto the two sets converge to the same place in theory and
far too slowly to be usable — on that same problem they reached ``6.8\times10^{-3}``
and then advanced by less than a tenth of a percent per sweep. They remain only as
a last fallback.

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

## The warm-start cache, and the caller's initial point

`OptimaOptimizer` carries a cache of its previous solution, and with
`warm_start = true` it may reuse it. The semantics matter enough to state
precisely, because the natural implementation is wrong.

Let the caller supply an initial point $u_0$. The cache holds $n^{\text{prev}}$,
the solution of *whatever problem this algorithm object last solved* — not
necessarily this one, since the object is reusable and reused.

**The rule.** The cache is consulted only when

1. it has the same dimension as the current problem, **and**
2. $u_0$ carries no interior information, i.e. every component sits at its lower
   bound, $u_{0,i} \le 100\,\ell_i$ for all $i$.

Otherwise $u_0$ is used as given.

Condition 2 is the substantive one. Starting from $n^{\text{prev}}$ regardless
discards an initial point the caller chose deliberately, and the consequence is
not academic: a chemical-kinetics run re-speciating at every accepted step leaves
its *final* composition in the cache, so replaying the same trajectory through
the same algorithm object started every solve from the 28-day state. On an
ordinary Portland cement that returned a pore solution at pH 14.2 with 0.31 mol
of ettringite and no monosulphate, where honoring the caller's guess gives
pH 12.58 with the sulfate entirely in monosulphate — the same trajectory, the
same constraints, the same guess. It also silently defeated the caller's own
warm-start logic *during* the run.

The cache is therefore a convenience for repeated solves where the caller has
nothing better to offer, and never an override.

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
