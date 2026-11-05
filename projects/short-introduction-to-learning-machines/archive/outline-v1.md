# Outline: Learning Problems Seen from Parameter Space

(Working title. Alternative: *A Geometric View of Learning Problems*.)

Working outline for agreeing the strategy before more prose is written.
`note.tex` is a draft of Part I. It is kept as raw material and will be cut to fit this outline.

## Thesis

Every learning problem, from fitting a Gaussian to training an LLM, has the
same mathematical shape. A utility defines what counts as a good answer. That answer is pulled back through a model map to a loss on parameter space, and the learner is judged by regret. Problems differ in two ways only:

1. **The geometry of the model map.** This covers fibres, the set of optimal parameters W₀, and singularities.
2. **What can actually be computed.** This means exact solutions,
   numerical approximation, and stochastic approximation.

The note names these objects and keeps four things apart: what the problem
gives, what the modeller chooses, what is assumed, and what is approximated.

**Positioning: a view, not the view.** Every computable learner, at a given sample
size, produces its predictive distribution through a map from a finite-dimensional
state. Several theories of learning study quantities attached to this map: Fisher
information, stochastic complexity, Bayes free energy, log-loss regret, and the
volume of near-optimal parameters. In the regular case many of these reduce to the
parameter count d/2, and in the singular case to the RLCT λ. The note presents the
map as their common object; no unification is claimed. The map is studied up to
reparametrisation, and the quantities worth naming are those that are invariant.
Part I is shared by all views; the view proper starts in Part II. Computation
happens in parameter space, which is why its geometry matters to practice.

**Stance: all models are wrong.** The DGP q exists (Part I), but the model class is
an idealisation that does not contain it in general. The target is the best
distribution the idealisation offers, the KL projection p₀, and its usefulness is
measured by regret under q. Realisability is a special case, not the default. The
condition that makes the unrealisable case well behaved is *relatively finite
variance* (Watanabe): it makes p₀ unique even when W₀ is not a point.

## Running examples: one dataset, a ladder of idealisations

Adult height, as recorded (finite precision, so the records lie in a finite set of
bins). Every model below is fitted to the same records, so what changes is only the
idealisation.

| Model | Status | What it shows |
|---|---|---|
| Categorical over height bins (multinomial) | saturated: contains every q on the finite record set | regular; the only model that is guaranteed right, at the cost of one parameter per bin |
| Gaussian | wrong but useful: two parameters | misspecified, K(w₀) small but > 0; usefulness measured by regret under q |
| Mixture of two Gaussians | wrong but more useful; matches the known structure (two sexes) | singular; p₀ sits near the locus where the components merge (height is not bimodal) |
| Same, with sex recorded (optional) | conditional model q(y \| x) | the hidden label becomes an input; the singularity disappears |

Then the sequence examples:

| Data | Model | What it shows |
|---|---|---|
| Phone accelerometer, activities labelled | HMM | a hidden state in time; label switching carries over from the mixture |
| Natural text | transformer | no model is believed true; chosen for flexibility, judged by held-out loss |

Supporting detail and candidate sources are in `design-notes.md`.

---

## Introduction (written last)

- How to look at an LLM, or any trained model, through this lens: a short
  tour from the problem to the loss to the solution to the computation.
- Thesis and ledger preview: given / chosen / assumed / approximated.
- What is not claimed: no new results; every link is a cited theorem, a
  labelled assumption, or a definition.

## Part I: Framing (from the problem to the loss)

**Goal:** arrive at L(w), K(w) and L_n(w) with every step labelled.
**Budget:** about 8–10 pages. This is less than the current draft, which spends
most of its length here.

| § | Content | Status tag |
|---|---|---|
| I.1 Problem | decision problem, utility, Bayes action, regret; three cases (acting, predicting with a given utility, predicting with the utility left open); Breiman's two cultures as a difference in model choice | given |
| I.2 Records | observable → measurement kernel → encoding → finite record; probability as a chosen language; A1, after which the law of the records exists | chosen + assumed |
| I.3 Sampling symmetry | exchangeability, de Finetti, q as the directing measure; the finite version (Diaconis–Freedman); sequences (many paths vs one path); actions as records (ignorability, positivity); testable and designable | assumed (checkable) |
| I.4 What there is to learn | I(X;Y) = regularity, H(Y\|X) = noise; excess entropy for sequences; world symmetries produce regularity on the quotient (noise outsourcing). Compression and the structure function become one remark | property of q (given, unknown) |
| I.5 Models | model map W → Δ(𝒳) as parametrisation ∘ output map; image, fibres, identifiability; the KL projection p₀ as the target (realisability as the special case p₀ = q); regular vs singular (definition only, as structure is Part II) | chosen |
| I.6 Utility → loss | regret bound via Pinsker; log score regret = KL; Bernardo; L = H(Y\|X) + K; empirical L_n; Bottou–Bousquet decomposition (approximation / estimation / optimisation) as the map of Parts II–III | derived |
| I.7 What can be computed | given / computable / tractable / estimable. q is unknown, not uncomputable; expectations, argmins and integrals over W are the objects Part III must approximate. Sets up the Part II / Part III split | classification |

**Change from the draft:** draft §4 (regularity) and §5 (world symmetry) merge
into I.4 at about half their current length. Kolmogorov complexity material
moves to one remark plus citations.

## Part II: Exact solutions (assume expectations, integrals and argmins are available)

**Goal:** the "teeth" of the note. Show what the loss landscape on W looks like
and what different solution concepts return.
**Budget:** about 10–12 pages.

- **II.1 The optimal set W₀.**
  - W₀ = argmin K.
  - W₀ is the fibre over the KL projection p₀ (over q in the realisable special case).
  - Relatively finite variance: the condition under which p₀ is unique, so that
    every point of W₀ gives the same distribution. It holds for realisable models
    and fails when two different distributions are equally close to q.
  - The approximation error is K(w₀).
- **II.2 Reparametrisation, symmetries and invariants.**
  - Two kinds of transformation:
    - reparametrisations φ: W′ → W (same image, new coordinates): which
      quantities are intrinsic;
    - symmetries of a fixed map (p∘φ = p): these generate fibres. Examples:
      permutation of components or hidden units, ReLU rescaling, softmax shift.
      The quotient W/~; fibres from symmetry vs fibres from degeneracy.
  - Smooth case (diffeomorphisms): Fisher information transforms as a metric;
    Jeffreys prior is invariant; the MLE is equivariant; MAP is not; the natural
    gradient is invariant and plain gradient descent is not.
  - Singular case: resolution of singularities is a reparametrisation; λ is
    invariant under it.
  - Practical consequence: Hessian flatness is not invariant (Dinh et al. 2017),
    while λ is.
  - Continuous symmetries give conserved quantities of gradient flow (Noether
    for learning: layer balancing, broken by SGD). Named and cited only.
  - Running example: the mixture's label switching (a discrete symmetry) and the
    merging of components (a degeneracy).
- **II.3 Kinds of singularity.**
  - (a) non-identifiability at a point (mixture with two equal components:
    the mixing weight is undetermined);
  - (b) W₀ a positive-dimensional set (an over-specified mixture: a weight
    goes to 0, or two components coincide);
  - (c) degenerate Fisher information from a factorised parametrisation
    (the transformer's embedding × unembedding product).
  - Label switching carries over from the mixture to the HMM.
  - The RLCT λ as the invariant, cited informally: λ = d/2 for regular
    models, λ < d/2 for singular ones.
- **II.4 Solution concepts as functions of the data.**
  - MLE: argmin L_n.
  - MAP: argmin of L_n − (1/n) log φ.
  - Bayes posterior as a Gibbs measure, and the tempered posterior.
  - Bayes predictive: it lies outside the model image in general.
  - Failure modes: the mixture MLE does not exist (the likelihood is
    unbounded); MAP depends on the parametrisation.
  - Recommended reference point: the Bayes predictive.
- **II.5 Convergence in three layers.**
  - Pointwise: SLLN, or the ergodic theorem for chains.
  - Uniform: ULLN, giving consistency to W₀.
  - Fluctuations: for regular models, asymptotic normality and AIC/BIC; for
    singular models, Watanabe's results (generalisation error ≈ λ/n, WAIC,
    free energy ≈ nL_n + λ log n).
  - Regular and singular models differ only in the fluctuation layer.
  - Devroye: there is no uniform rate without a model.
- **II.6 Sequences: activity and text.**
  - Chain-rule loss; per-step loss = entropy rate + KL rate.
  - Activity: an HMM; prediction runs through the belief state over activities.
  - Text: the same loss, an unrealisable target, a transformer as the map.
  - Many-path vs one-path convergence.

## Part III: Inexact computation

**Goal:** separate approximation that is a choice from approximation that is
forced, and name what the computed object actually is.
**Budget:** about 6–8 pages.

- **III.1 What cannot be computed.**
  - Expectations under q: only samples are available.
  - argmin over a non-convex W.
  - Integrals over W in high dimension.
- **III.2 Numerical approximation.**
  - Deterministic methods with error bounds: quadrature, the Laplace
    approximation, full-batch gradient descent and Newton's method.
  - These reach stationary points, not W₀.
- **III.3 Stochastic approximation.**
  - Robbins–Monro.
  - SGD: one-pass SGD on fresh samples is stochastic approximation of L
    itself, while multi-epoch SGD works on L_n.
  - This is required by scale, not chosen for convenience. It needs step-size
    conditions.
- **III.4 Sampling.**
  - MCMC and SGLD for the posterior.
  - Localised posteriors around a trained point, mentioned only; the details
    belong to the parking lot.
- **III.5 What the algorithm returns.**
  - A random variable whose law depends on the algorithm. It is not exactly
    the MLE, MAP or Bayes estimate.
  - Optimisation error, the third Bottou–Bousquet term.
  - Implicit regularisation is named, not explained.
- **III.6 Evaluation.**
  - Held-out loss estimates L(ŵ) without bias only under train/test
    exchangeability. This links back to I.3.

## Part IV: Examples (each follows the same template)

**Template:**
1. problem and utility;
2. records and measurement;
3. sampling symmetry, and how it fails;
4. regularity and world symmetries;
5. model map, fibres and singularities;
6. loss;
7. solution concept;
8. computation;
9. ledger row.

- **IV.1 Cat classification.**
  - Labels as noisy measurements, and translation invariance.
  - CNN followed by softmax; ReLU and permutation fibres.
  - SGD.
- **IV.2 Science example** (to choose).
  - A case where the parameter is the answer, so identifiability matters.
- **IV.3 Robotics.**
  - Actions as records, with a fixed environment kernel and a chosen policy.
  - Ignorability and on-policy data collection.
  - SE(3) equivariance.
- **IV.4 LLM.**
  - Documents are exchangeable (many paths); the chain rule over tokens.
  - Transformer followed by softmax; its fibres.
  - Pre-training as approximately one-pass stochastic approximation of L.
  - The II.6 transformer scaled up. The objects stay the same.
  - Fine-tuning and RLHF as a change of utility: one paragraph, with the
    detail in the parking lot.

## Part V: Summary

- **Dictionary:** practitioner term ↔ mathematical name (e.g. "training loss"
  = L_n, "a minimum" ↔ W₀ or a fibre, "loss of model capacity" ↔ approximation
  error K(w₀)).
- **Ledger:** given / chosen / assumed / approximated, with one row per
  example.
- **Connections:** one row per theory of learning:
  the quantity it studies, how it attaches to the map, its regular special case,
  its singular form, and what it is invariant under. Rows: information geometry
  (Fisher metric; the regular local picture), MDL (stochastic complexity;
  (d/2) log n vs λ log n), Bayes (free energy, posterior predictive), individual
  sequences (log-loss regret), optimisation (natural gradient; conserved quantities),
  volume of near-optimal parameters (λ), PAC/VC (function-class complexity: weakly
  attached to the map, stated honestly).
- **Not covered:** a pointer to the parking lot.

## Parking lot (later papers)

- Reparametrisation groups as an organising principle: what is the right group in
  the singular case (birational, measure-preserving, posterior-predictive-preserving
  maps)? Do they give different invariants? Conservation laws as a route to structure,
  as in physics.

- The prior as a push-forward.
- Markov de Finetti.
- RLHF reverse-KL structure.
- Local learning coefficient estimation in practice.
- Universal prediction and individual sequences.
- Algorithmic statistics in depth.
- Positive-dimensional singularities and small-sample findability: why more
  singular optimal sets may be easier to reach with few samples. Position against
  the volume-scaling meaning of the RLCT (Watanabe), Bayesian preference for singular
  regions at small n, flat minima (Hochreiter & Schmidhuber 1997), and SGD-as-sampler
  results (Mingard et al. 2021).

---

## Decisions needed before writing

1. **Weight of Part I.** Should it be cut to about 8–10 pages, with
   regularity and world symmetry merged into one section (I.4)?
   *Recommendation: yes.* This is where the current draft feels distracting.
2. **Running examples.** Data-first table above. Open: the sources for each
   dataset (see `design-notes.md`), the bin width for heights, and mixture of
   Gaussians vs mixture of Poissons as the singular step.
3. **Bayes as the reference.** Should the Bayes predictive be the reference
   point in II.4, with MLE and MAP presented as its limits or approximations?
   *Recommendation: yes.*
4. **Science example.** Still open (ion channels dropped as too technical).
5. **Depth of Watanabe.** Cite his results informally in II.2, II.3 and II.5 only
   (the RLCT, generalisation error, WAIC), with no analyticity assumptions?
   *Recommendation: yes,* as already agreed.
6. **Order of writing.** Write Part II first, since it carries the teeth, then
   cut Part I down to exactly what Part II needs?
