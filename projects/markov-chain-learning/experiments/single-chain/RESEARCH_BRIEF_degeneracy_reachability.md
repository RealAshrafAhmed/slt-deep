# Research brief: degeneracy ⇄ profile-flatness ⇄ optimization reachability

A prompt + experiment design for a new session. Two linked ideas about singular
(over-parameterized) models. The throughline: **one geometric quantity — the dimension /
"size" of the true-parameter variety $W_0$ (equivalently the RLCT $\lambda$) — controls three
things usually studied separately: (1) generalization, (2) why profile likelihood goes flat,
(3) how easily optimization reaches the true set from random init.** The novelty is the
*link*, not the discovery of degeneracy.

Setting we have working code for: a small `MarkovTransformer` (~910 params, vocab 5, len 10)
whose DGP is a single analytic Gambler's-Ruin Markov chain `q`. KL(q‖p_w) is computable
(log q exact). We can vary net width to change degeneracy. Tools already built:
`torch_bdn.bn.fit_constrained` (freeze raw coords, minimize rest),
`fit_constrained_directions` (pin coefficients along an orthonormal basis, e.g. eigenvectors),
`top_diagonal_curvature_indices`, `approx_hessian`. See HANDOFF_profile_kl.md for the
profile/KL machinery and the hard-won facts (population KL, minibatch-SGD inner solver,
flat 2-param profile, etc.).

---

## IDEA 1 — Profile likelihood is flat because the true set is a positive-dim variety

**Claim.** Let $W_0 = \{w : \mathrm{KL}(q\|p_w)=0\}$ be the true-parameter set, an analytic
variety of dimension $d_0$ (on its smooth locus). Profile out all but $k$ coordinates:
$\tilde K(\alpha)=\min_{\text{rest}} \mathrm{KL}$. Then $\tilde K \equiv 0$ wherever the
$k$-coordinate constraint fails to *isolate* $W_0$ — i.e. as long as the projection
$\pi_k: W_0\to\mathbb R^k$ is submersive, fixing $k$ coords just slides you along $W_0$. The
"number of compensating directions" is $d_0 - \mathrm{rank}(\text{constraint on }W_0)$. With
$d_0 \gg k$ this holds generically ⇒ the profile is flat. (This is the rigorous version of
what we observed: 2-param profile flat to 5 decimals while the *slice* is richly structured.)

**Why this is a gap.** Classical modified-profile-likelihood (Barndorff-Nielsen; Cox–Reid;
Severini) corrects profile bias but ASSUMES the nuisance block Hessian $j_{\lambda\lambda}$ is
invertible — its corrections divide by $|j_{\lambda\lambda}|$ / require Fisher orthogonality.
In the singular case that block is rank-deficient (det ≈ 0), so the classical machinery does
not apply. Watanabe's SLT handles the *integrated* object (free energy, RLCT) over $W_0$ but
never the *profile* (min-out) construction. So "profile likelihood on a singular model with a
variety of true parameters" appears genuinely unwritten.

**Caveats to nail (these are where the real math is):**
- $W_0$ is a VARIETY not a manifold — $d_0$ can jump at singular points; the clean statement
  is on the smooth locus, the singular locus needs resolution-of-singularities (Hironaka /
  Watanabe). Don't claim a global constant $d_0$.
- **Second-order flat ≠ on the variety.** Near-zero Hessian eigenvalues (local, quadratic)
  are NOT the same as actually lying on $W_0$. Eigenvectors are a LOCAL LINEAR frame; the
  curvature coordinates are nonlinear, so a straight eigenvector line leaves the curved valley
  at finite radius. Any "count of degenerate directions" must distinguish (a) #{small Hessian
  eigvals}, (b) true $\dim W_0$, (c) numerically-small-but-not-flat.

**Test (cheap, code exists):**
1. Eigendecompose H(KL) at the population-KL minimizer w*; count eigvals below thresholds
   (λ_max·1e-3, 1e-6). Report the spectrum + degenerate count (MEASURE, don't assume).
2. Profile (minimize-the-rest) along the 2 STIFFEST eigenvectors at SMALL radius (±0.3, fine
   grid) → expect a non-flat bowl (incompressible direction). Along the 2 SOFTEST eigenvectors
   → expect ~flat (compensable). Stiff-curved vs soft-flat at small radius = the signature.
3. Confirm a single raw-coordinate profile is flat (already seen) — it projects into $W_0$.

---

## IDEA 2 — More degenerate ⇒ bigger $W_0$ ⇒ easier/faster to reach from random init

**Intuition (user's).** As net width grows, the number of degenerate directions grows, so the
true set $W_0$ is bigger / higher-dimensional, so a random perturbation is more likely to
*hit* (or flow to) $W_0$, and the mean optimization path to the minimum gets shorter. More
degenerate = bigger target = easier to hit from more initializations.

**The quantity that has to be made precise (user's own caveat — this is the deep part).**
"Reachability" is not just $\dim W_0$; at finite sample N the relevant object is the **measure
of the sub-ε low-loss tube around $W_0$**. SLT says this scales as $V(\epsilon)\sim
\epsilon^{\lambda}$ with $\lambda$ the RLCT (which encodes how degenerate $W_0$ is near w*),
and the empirical-loss tube width in the non-degenerate directions scales like $N^{-1/2}$. So
reachability couples THREE things: $d_0$ (variety dim), the codimension $d-d_0$ (how many
directions must be "hit"), and N (how thick the tube is). The theorem must be quantitative in
$(d_0, d, N)$ — "bigger set, easier to hit" is the hand-wave that needs replacing.

**Also control these confounds (or the experiment proves nothing):**
- Steps-to-reach depends on optimizer/lr/conditioning, which ALSO change with width. Fix the
  optimizer; measure steps-to-(KL<ε); normalize for per-step cost and for conditioning
  (report κ = λ_max/λ_min_nonzero too). Otherwise you measure conditioning, not geometry.
- Distinguish "reached $W_0$" (population KL < ε on a large held-out sample) from "fit the
  training sample" (empirical loss small) — the over-parameterized net can do the latter
  without the former (the Hypothesis-B failure mode from HANDOFF; use held-out KL).

**Conjecture to state and test.** Across widths W (hence model dims d(W)):
- the degenerate count / estimated $\dim W_0$ grows with W,
- the RLCT $\lambda$ (estimate via SLT free-energy / λ̂ from SGLD, or proxy) tracks it,
- the **mean steps-to-(held-out KL < ε) from random init DECREASES** as degeneracy grows,
- and the decrease is PREDICTED by a function of $(d_0, d-d_0, N)$ — ideally the same $\lambda$
  that predicts generalization. **The link (one quantity → all three) is the contribution.**

**Experiment design (build this):**
1. Width sweep: train the MarkovTransformer at several widths (d_model / depth), several random
   seeds each, on the same DGP. For each: estimate $\dim W_0$ (degenerate Hessian count at the
   solution) and/or λ̂; record steps-to-(held-out KL < ε) from random init (the reachability
   metric); record conditioning κ.
2. Plot reachability (mean + variance over seeds) vs degeneracy/λ̂/width; test whether a
   single curve in the predicted $(d_0,d,N)$ variable collapses them.
3. Sample-size cross-check: vary N at fixed width — tube thickness $\sim N^{-1/2}$ should move
   reachability in the predicted direction (this is the N-dependence the user flagged).
4. Tie back to IDEA 1: at each width, confirm profile-flatness scales with the measured
   degenerate count (more degenerate ⇒ flatter profiles ⇒ more compensating directions).

---

## What is known vs. (likely) new — frame the writeup honestly

- KNOWN: SLT/RLCT for generalization (Watanabe); overparam nets reach zero loss from random
  init (NTK / lazy training) — but proved via kernel arguments, NOT via $\dim W_0$; classical
  (modified) profile likelihood — only for invertible nuisance blocks; sloppy-models geometry
  (Transtrum–Sethna): stiff/soft directions, curved thin "hyperribbon" model manifold.
- LIKELY NEW: (i) profile likelihood on a singular model where the true set is a positive-dim
  variety, and the flatness theorem with the $d_0 - \mathrm{rank}$ count; (ii) the bridge that
  the SAME degeneracy ($\dim W_0$ / RLCT) governing generalization ALSO governs profile-
  flatness AND optimization reachability — one geometric quantity, three consequences.

## Pressure-tests before claiming a theorem
- Does "#small Hessian eigvals" actually equal a meaningful $\dim W_0$, or is it second-order-
  flat-but-not-on-the-variety? (Probe with finite-radius profiles, not just the Hessian.)
- Reachability needs the MEASURE of basins / the stable manifold of the flow, not just
  $\dim W_0$ — random-init reachability is a dynamics statement; degeneracy helps but the clean
  probability needs the tube-volume argument, with N.
- Variety vs manifold: handle the singular locus (resolution of singularities) or restrict
  claims to the smooth locus and say so.

## References to anchor it
- Watanabe, *Mathematical Theory of Bayesian Statistics* (2018) — RLCT/free energy, readable.
- Watanabe, *Algebraic Geometry and Statistical Learning Theory* (2009) — resolution of
  singularities, the variety picture.
- Amari & Nagaoka, *Methods of Information Geometry* — Fisher metric / the regular theory.
- Barndorff-Nielsen & Cox, *Inference and Asymptotics*; Cox–Reid (1987) — (modified) profile
  likelihood, and exactly the non-degeneracy assumption that fails here.
- Transtrum, Machta, Sethna — "sloppy models" / geometry of nonlinear least squares (curved
  thin model manifold; stiff vs soft).
