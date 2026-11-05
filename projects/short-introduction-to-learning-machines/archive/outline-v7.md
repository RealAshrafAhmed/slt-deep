# Outline v7: Five Postulates and One Map

*The mathematical structure of learning problems, Part 1: belief formation.*
(Alternative titles: *World, Agent, Kernel*; *Learning as Stochastic Realization*;
*What a Learning Machine Is Made Of*.)

Consolidates v1–v6, `design-notes.md`, `verification.md`, `note.tex`,
`prior-art.md` and the discussion of 2026-10-24/25. Supersedes v6. The
**Provenance** section at the end records where every earlier idea went, so
nothing is lost silently.

Register: a physics paper. Postulates (structures), derived notions, a dictionary,
examples; the literature in Appendix B. Every statement in the body is a
definition, a labelled assumption, a cited theorem (named, not reproduced), or an
example.

## Scope

**Belief formation only.** An agent observes records of a world and forms
predictive beliefs about what it will observe next. Nothing it does changes the
world. This is the setting of statistical inference, scientific modelling, and an
LLM without tools. Actions (reports, designs, control) are Note 2, an extension of
this note. "Control" is used in the control-theory sense (an input that steers the
world's dynamics) and only to name what is out of scope; one paragraph in §0 says
why this is a scope choice and not a limit of the Bayesian view (Savage's normal
form; Wald, Feldbaum's dual control, Åström, Bayes-adaptive MDPs).

**Ontology left open.** The note commits to a measure-free structure
(Ω, 𝓕, {X_t}, (𝓖_t)) and to the existence of a law on the records. Whether the
law is in the world (ensemble, propensity) or in the agent (de Finetti, B&S)
changes one footnote and no theorem.

**Contribution.** Each postulate has prior art (Appendix B, `prior-art.md`). The
contribution is the assembly: no existing treatment keeps the world side
(P1–P3), the realization (P5) and the singular parametrisation (§3) in one frame.
Said once, in §0.

Length: ~12k words; five posts at the section breaks.

## What the note claims

1. Belief formation in statistics, ML and scientific modelling shares one
   structure, five postulates. Each community has formalised a subset; a table
   shows which.
2. A learning machine is a **stochastic realization** of a kernel: an internal
   state that splits past from future, a map from states to predictive kernels,
   and a filter. Every predictive model is one, explicitly or as the trivial
   realization; training a sequence model is approximate realization from data.
3. The parameter space is a **parametrisation** of the realization, not a chart on
   it. Everything the machine computes is either pulled back along the map (hence
   invariant under reparametrisation by construction) or is a coordinate fact.
   This is the one principle.
4. A representation is judged by **predictive alignment** (its kernel is close to
   the world's) and **efficiency** (its state carries the predictive information
   and nothing more). **Structural alignment**, correspondence with a named hidden
   observable, is a separate and optional claim.

Not claimed: new theorems; a unification beyond the table; a position on where
probability lives.

## Running examples

| Example | Symmetry | Realization | Carries |
|---|---|---|---|
| Adult heights (NHANES, binned): categorical → Gaussian → two-Gaussian mixture → mixture with sex recorded | exchangeable | S = W; S = W × {label} for the mixture | the map, image, fibres, the singular point; pullbacks; a hidden observable (sex): hidden, learnable, identifiable **[V: NHANES; Schilling et al.; height means/SD]** |
| Text and an LLM | exchangeable documents, stationary within | trivial: S = context window | the main example; kernels not laws; everything at scale; one closing paragraph per section |
| Transformers on HMM data; Othello-GPT | stationary | declared latent (HMM) vs learned (transformer) | a predictor that recovers the minimal realization; efficiency; the two alignments **[V: Shai et al. 2024; Li et al. 2023; Nanda et al. 2023]** |

The HMM example is also where the toy-transformer work on this branch connects.

---

## §0 Introduction (~900 words)

- The same objects under different names: likelihood / cross-entropy; Fisher
  information / Gauss–Newton; posterior predictive / ensemble; hidden state /
  latent / parameter / residual stream / belief state; probe / estimator of an
  observable; "state space model" / realization.
- The four claims in four sentences; the diagram (§3) in one figure.
- Scope and the Bayesian-control paragraph. The contribution sentence.
- The ledger: given / chosen / assumed / approximated; one line per section.

## §1 Five postulates (~3,000 words)

Each postulate: the structure, one example, one physics counterpart in a
sentence; citations by theorem name only.

**P1 (World).** A measurable space (Ω, 𝓕) of complete histories, past and future,
with a law P. One history ω* occurs; averaging over Ω is averaging over what is not
yet known about ω*. Ω is canonical by default, Ω = 𝒳^T, and enlarged only to name
a hidden observable (a ledger entry). *Physics:* ensemble over trajectories; the
microstate is not needed.

**P2 (Observables).** An observable is a measurable function on Ω. The instrument
reads X_t: Ω → 𝒳, 𝒳 finite (finite precision). These generate the agent's
filtration 𝓖_t and its limit 𝓖_∞.
- What the agent knows at t is the atom A_t(ω*) = {ω : X_s(ω) = X_s(ω*), s ≤ t};
  the predictive law is P(· | 𝓖_t), supported on it. "The random experiment at
  time t" is (A_t(ω*), P(· | 𝓖_t)).
- **Hidden** is a property, not a kind: τ is hidden at t if not 𝓖_t-measurable.
- **Learnable**: τ is 𝓖_∞-measurable mod P. Doob: E[τ | 𝓖_n] → E[τ | 𝓖_∞] a.s.,
  learnable iff the limit is τ. Not learnable ⇒ no agent and no data suffice.
- A sample is X_{1:n}(ω*). "Different trials" means either coordinates of the one
  history (the only kind that yields data) or alternative futures in A_t(ω*)
  (counterfactuals, present only inside expectations).
- *Physics:* coarse-graining; Mori–Zwanzig (the observed variables alone are not
  Markov: memory and noise); Koopman (observables primary).
**[V: Doob; Ay & Löhr 2015 for the sub-σ-algebra formulation]**

**P3 (Symmetry; assumption A1).** A group G on the index set with P invariant.
Ergodic decomposition P = ∫ P_θ Π(dθ); one history identifies its own component
by averages. θ is the **macrostate**: an observable of ω*, constant on G-orbits,
𝓖_∞-measurable, hence learnable. A1 is the ergodic hypothesis for the record
process: one history stands in for the ensemble. The table:

| G | Components | Theorem | Example |
|---|---|---|---|
| permutations (exchangeable) | i.i.d. laws | de Finetti; Hewitt–Savage; Diaconis–Freedman (finite) | heights |
| exchangeable units, sequential within | laws on units | de Finetti on units | documents |
| shifts (stationary) | stationary ergodic laws | ergodic decomposition | tokens within a document; HMM data |

  Without a symmetry one history identifies nothing beyond itself; options are a
  weaker regularity (mixing), designing the symmetry in (random sampling; Note
  2), or non-learnability. A1 is testable toward rejection (test martingales).
  Under the epistemic reading, the P_θ are *derived* from the agent's own
  exchangeable law (B&S ch. 4) and "objective" means intersubjective agreement.
**[V: Kallenberg 2005; Lauritzen; Orbanz & Roy; Al-Najjar & Shmaya; Vovk et al.]**

**P4 (Meeting space: kernels).** What is learned is a kernel from the conditioned
past to the predicted future: κ(dx_{t+1} | x_{1:t}) for an LLM, κ(dy | x) for a
GLM. The law of the conditioning variable is unmodelled and enters only as a
weight. The meeting space 𝒦 is the space of such kernels; the world's point is
κ_θ. Laws ↔ kernel families (Ionescu-Tulcea), so nothing is lost; but the agent
makes no claim about the marginal and is not wrong about it, and fibres are
relative to the weight.
**[V: Ionescu-Tulcea; Blackwell; Hutter / Ortega for kernels on histories]**

**P5 (Agent: a stochastic realization).** A theory is a triple (S, p, π):
- an internal state process S_t with the **splitting property**: past ⊥ future
  given S_t (Markovian splitting subspace in the Gaussian case; conditional
  independence in general);
- a map p: S → 𝒦 (the kernel over the future given the state);
- a filter π_t ∈ Δ(S) updated from the past, so the agent's predictive is
  ∫ p_s(· | x_{1:t}) π_t(ds).
Facts (stochastic realization theory), each one sentence with its name:
- *Trivial realization:* S = the past. An autoregressive transformer is this, the
  context window as S.
- *Minimal realization:* the forward predictor space, the conditional law of the
  future given the past (Knight's prediction process; Shalizi–Crutchfield's causal
  states). Minimal by inclusion, not dimension (Lindquist–Picci); unique up to
  isomorphism; a functional of κ_θ alone.
- *Internal vs external:* a realization is internal if S_t is measurable w.r.t.
  the observables (constructed from data), external if it needs an enlarged Ω
  (Lindquist–Picci). A declared latent (HMM states, a mixture label) is a choice
  of realization; a generator is not in general the minimal predictor (Löhr–Ay).
  The further claim that an external S coincides with a named hidden observable
  of the world is **optional** and labelled when made.
- *Non-uniqueness:* many realizations, one kernel (Gaussian case: a family of
  minimal realizations; OOMs; PSRs). This is a fibre one level above W.
- *The filter is itself a realization:* the prediction process is Markov
  (Knight); the mixed-state presentation of Shai et al. is this object.
- *The realization problem:* construct (S, p) from κ_θ or from data (Akaike's
  canonical correlations; subspace identification). **Training a sequence model
  is approximate stochastic realization from one history.**
A parameter space W parametrises S (or S = W); the model map is W → S → 𝒦.
Learning is the agent's dynamics driven by the record stream: on beliefs (the
filter; Bayes on W) or on coordinates (gradient flow).
**[V: Lindquist & Picci; Akaike; Knight; Shalizi & Crutchfield; Löhr & Ay;
Jaeger; Littman–Sutton–Singh; Van Overschee & De Moor]**

**The cross-community table.** Which postulates each formalises: Kolmogorov
(P1–P2); ergodic theory, Kallenberg, Lauritzen (P3); Blackwell–Le Cam (P2, P4);
B&S (P3, P5 with S = Θ); stochastic realization theory, Lindquist–Picci (P5
linear); computational mechanics (P5 minimal, general); filtering theory (P5
filter); Hutter / prequential (P4, P5 trivial); supervised ML (P4, W); SLT
(W → 𝒦); Ay–Löhr (P2 sub-σ-algebra, P5 minimal world model); Bayesian mechanics
(P5 with the world claim assumed).

**Ledger.** Given: the record process. Chosen: the instrument, the realization,
the coordinates. Assumed: A1; standard Borel on Ω and S. Derived: θ,
learnability, the filter.

## §2 The meeting space and what lives on it (~1,500 words)

All pointwise in the conditioning variable, averaged under its unmodelled law.

- Convexity: mixtures of kernels; averaging happens here.
- Divergence: expected conditional KL; relative entropy rate via the chain rule.
  Log score regret = KL (Gibbs). Justified without decision theory: code length
  (prequential; MDL); the f-divergence that decomposes along the chain rule.
  Forward pointer: bounds every bounded decision's regret (Note 2; the
  Proposition of `note.tex` §7 moves there). **[V: Pinsker; Bernardo; chain-rule
  characterisation]**
- Fisher–Rao pointwise then averaged; Čencov. **[V]**
- Observables as functionals of a kernel: one-step predictive; entropy rate h;
  predictive information I(past; future) = excess entropy E.
- **Regularity and noise** (one paragraph, from the companion): I(past; future)
  is what a representation can carry; H(future | past) is noise.
  **[V: Bialek–Nemenman–Tishby; block entropy]**
- KL projection κ₀; uniqueness under relatively finite variance. **[V: Watanabe
  2018]**
- The minimal realization as a functional of the kernel; C_μ ≥ E. **[V]**

**For an LLM.** Next-token kernel across documents; its minimal realization is
what a perfect predictor must track; its predictive information bounds what any
context window can carry.

## §3 The map and its pullbacks (~2,800 words; the core)

```
   Δ(W) ──p_#──►  Δ(𝒦)  ──mean──►  predictive kernel
     │
     W ──► S ──p──►  𝒦  ◄──κ_·──  Θ           (agent ── data ── world)
                     │
   K, L_n, F, f∘p ◄── pull back ── KL, Fisher–Rao, observables f
```

Agent (W, S) and world (Θ) both map into 𝒦 and never into each other; the
empirical kernel sits in 𝒦.

- **3.1 The map.** W → S → 𝒦; image, fibres (relative to the weight),
  identifiability. Heights ladder: four maps, one dataset.
- **3.1b Charts versus parametrisations.** Regular case = chart (injective
  immersion; Amari; Watanabe's "regular"). Two failures: non-injective (fibres
  from symmetry; W/~), rank drop (image not a manifold; degenerate pullback).
  The right statement: a parametrisation of a subset of 𝒦, not a chart.
  Resolution of singularities restores coordinates locally after a
  non-injective blow-up; λ, m survive. History in one paragraph (Gauss →
  Riemann → Whitney → singularity theory; algebraic geometry always singular;
  statistics inherited the chart view, Watanabe brought the other). What
  survives: pullbacks of functions, tensors, observables for any smooth map; no
  ambient manifold needed (Ay et al.; Pistone–Sempi; physics on convex state
  spaces). **Physics box:** gauge orbit spaces (Singer), superspace (Ebin,
  Fischer), singular reduction (Sjamaar–Lerman), moduli (Luna): a singular point
  is one with a larger stabiliser; do geometry on the stratified quotient;
  legitimate observables are invariants. The two alternatives placed. One
  sentence on the realization-level fibre (many realizations, one kernel) as the
  same phenomenon one level up. **[V rows as listed]**
- **3.2 Pullbacks.** KL → K(w); log loss → L = h + K; data → L_n; Fisher–Rao →
  F = JᵀgJ, degenerate at rank drops; λ in one paragraph; observable f → f∘p.
- **3.3 Observables of the agent; learnability on both sides.** τ on S or W is a
  legitimate observable of the agent iff constant on fibres. Twin on the world
  side: a hidden observable is learnable iff 𝓖_∞-measurable. Heights: component
  mean difference, label switching, the merged point; recording sex.
- **3.4 The principle, three contrasts.** Fisher vs Hessian in coordinates
  (Dinh; Lau et al. App. C); MLE vs MAP; predictive vs posterior mean of w. Two
  priors differing only along fibres give the same belief on kernels.
- **3.5 Three errors on the diagram** (Bottou–Bousquet).

**For an LLM.** S = context window, W = weights, p = next-token kernel; fibres
from permutations, rescalings, embedding × unembedding; "a feature" legitimate
iff constant on fibres.

## §4 Learning as a process on W, seen on 𝒦 (~1,800 words)

- **4.1 Solution concepts as measures on W.** MLE, MAP, posterior (Gibbs,
  tempered), law of an SGD run; mixture MLE does not exist; MAP not invariant.
- **4.2 Push-forward and the predictive.** Averaging on kernels, not W;
  ensembles = posterior predictives.
- **4.3 Convergence on 𝒦.** Pointwise / uniform / fluctuations (regular d/2 vs
  singular λ); many units vs one long path. One paragraph; cited.
- **4.4 What can be computed.** Expectations (samples only), argmin (non-convex),
  integrals (high-dimensional); deterministic vs stochastic approximation;
  one-pass SGD ≈ stochastic approximation of L; the returned object is a random
  point with an algorithm-dependent law. Named, not analysed (Note 2).
- **4.5 Evaluation.** Held-out loss estimates the pulled-back KL under
  exchangeability of units (A1).

## §5 Representations: alignment and efficiency (~2,000 words)

- **5.1 Two senses of alignment.** *Predictive*: the agent's kernel → κ_θ (or
  κ₀); all that log loss sees. *Structural*: the agent's S corresponds to the
  minimal realization of κ_θ (world-free) or, under the optional claim, to a
  named hidden observable. Well posed only for quantities constant on fibres
  (agent side) and learnable (world side).
- **5.2 Efficiency.** Sufficient: S carries I(past; future). Efficient: nothing
  more; carrying more = distinguishing pasts with the same future = encoding
  noise, overfitting at the level of representation. Causal states are the
  unique minimal sufficient realization; information bottleneck as the
  variational form, recovering them in the zero-distortion limit.
- **5.3 Science and prediction as two requests.** Prediction asks for predictive
  alignment; science asks for structural alignment with a *named* observable and
  so needs learnability and identifiability. Breiman's two cultures as a choice
  of realization.
- **5.4 Predictors recover the minimal realization.** Shai et al.: belief-state
  geometry in the residual stream, with generator non-uniqueness stated; Othello:
  the board is the predictive state of the game. Reason: optimal prediction needs
  the predictive information, which the minimal realization carries. So
  predictive alignment forces structural alignment *with the minimal
  realization*; the optional world claim is the step from "the minimal
  realization" to "the board exists". Interpretability = estimating an observable
  on W, legitimate iff constant on fibres.
- **5.5 The instrumentalist reading.** Rissanen, Dawid, Grünwald: a model is a
  code for the past–future regularity. Everything above survives; only the
  optional world claim is dropped.

**For an LLM.** Pre-training asks for predictive alignment only; what it learns
about hidden structure is what the minimal realization of text requires; the
context window bounds the predictive information it can carry.

## §6 Field guide (~800 words + table)

Rows: likelihood / cross-entropy / perplexity; Fisher information; Hessian
sharpness; weight norm / decay; posterior predictive / ensemble; posterior mean of
w; calibration; belief state; probe / concept vector; identifiability; AIC / BIC /
WAIC; λ; hidden state / latent / residual stream (a realization; invariant up to
isomorphism of realizations); context window (trivial realization); predictive
information / excess entropy; statistical complexity C_μ; "state space model"
(a declared realization); "superposition" / "collapse" (on W; not invariant
unless restated on S). Columns: postulate; lives on; pulled back / pushed forward;
invariant under reparametrisation?

## §7 Dictionary, neighbours, and what comes next (~900 words)

- **Dictionary** (structural rows; analogies marked): phase space ↔ Ω;
  observable ↔ X_t; ensemble ↔ P, π; ergodic hypothesis ↔ A1; macrostate ↔ θ;
  coarse-graining ↔ the instrument; Mori–Zwanzig ↔ records alone not Markov;
  Takens ↔ the trivial realization recovers the state; Kalman state ↔ S;
  gauge redundancy ↔ fibres; canonical transformation ↔ reparametrisation;
  energy ↔ K; metric ↔ F.
- **Nearest neighbours**, one paragraph each: computational mechanics and Shai et
  al. (P5, §5; no W side); Ay–Jost–Lê–Schwachhöfer and Ay–Löhr (§3, P2; no
  realization); Watanabe / Lau et al. (§3; no world side); Bayesian mechanics
  (P5 with the world claim assumed; "gauge" in another sense).
- **Note 2**: actions (reports, designs, control); regret bounds; dual control;
  tuning an LLM as a change of utility; equivalence of learning machines;
  invariants of the learning dynamics; "only invariants are worth measuring"
  (one paragraph, labelled motivation).
- Open questions.

## Appendix A: notation and ledger (1 page)
## Appendix B: sources, one line per postulate and derived notion (from `prior-art.md`)

---

## Provenance: where every earlier idea went

| Idea | Origin | v7 location |
|---|---|---|
| Ledger: given / chosen / assumed / approximated | v1 thesis | one line per section; Appendix A |
| Height ladder; saturated categorical; Poisson alternative | v1, design-notes | Running examples (Poisson dropped) |
| HMM / accelerometer; 2×2 generator design; BabyLM | v1, design-notes | HMM/transformer example in §5.4; accelerometer, 2×2, BabyLM → parking lot |
| Problem → utility → loss chain; regret bound via Pinsker; Bernardo log score | v1 I.1, I.6; `note.tex` §1, §7 | utility and regret → Note 2; log score justified by code length in §2 |
| Records, measurement kernel, probability as language | v1 I.2; `note.tex` §2 | P2 (one paragraph); companion |
| Exchangeability, de Finetti, Diaconis–Freedman, ignorability | v1 I.3; `note.tex` §3 | P3; ignorability → Note 2 |
| Regularity vs noise; excess entropy; Kolmogorov | v1 I.4; `note.tex` §4 | §2 one paragraph; Kolmogorov → companion |
| World symmetries, noise outsourcing | `note.tex` §5 | companion |
| Model map, image, fibres, regular/singular definitions | v1 I.5; `note.tex` §6 | P5, §3.1 |
| W₀, relatively finite variance, KL projection | v1 II.1 | §2, §3.2 |
| Reparametrisation vs symmetry; Fisher as metric; Jeffreys; natural gradient; Dinh | v1 II.2 | §3.1b, §3.4 |
| Kinds of singularity; λ; RLCT | v1 II.3 | §3.2 one paragraph |
| Solution concepts; Bayes predictive as reference | v1 II.4 | §4.1–4.2 |
| Convergence in three layers; Devroye | v1 II.5 | §4.3 |
| Sequences: chain rule, entropy rate | v1 II.6 | P4, §2 |
| Inexact computation; SGD as stochastic approximation; evaluation | v1 III | §4.4–4.5 |
| Applications template (cat, science, robotics, LLM) | v1 IV | LLM throughout; others → parking lot / Note 2 |
| Dictionary; connections table; PAC/VC honesty | v1 V | §6, §7 |
| Parking lot (prior as push-forward, Markov de Finetti, RLHF, LLC, findability) | v1 | unchanged, plus Note 2 items |
| Question-per-section; one figure per section; parallels boxes; blog split | v2 | kept as form; physics box in §3.1b; five posts |
| Four readings (optimisation, selection, building, approximation) | v2 §4 | §4 and Note 2 |
| Safety motivation | v2 §6 | §7, one paragraph, labelled |
| Control-theory dictionary; data steer across fibres; holonomy; control bundles | v3 | Note 2 |
| Diagram; prediction then averaging; four communities | v4 | §3 diagram; P5 filter + §4.2; cross-community table |
| Six objects; action channels; Bayesian control paragraph; act–state independence | v5 | five postulates; Scope; §0 |
| Charts vs parametrisations; physics box; ambient manifold not needed | v5 §3.1b | §3.1b |
| Hidden = non-measurable; learnable = 𝓖_∞; atom; trials; canonical Ω | discussion 10-25 | P1–P2 |
| Kernels as meeting space; realization; efficiency; instrumentalism | v6 | P4, P5, §5 |
| Stochastic realization theory (splitting, internal/external, minimality by inclusion, non-uniqueness, realization problem) | discussion 10-25 | P5 |
| Prior-art ranking; "assembly" as contribution | `prior-art.md` | Scope; §7 neighbours; Appendix B |

## Decisions needed

1. **P5 depth.** State splitting, minimality by inclusion and internal/external
   as definitions with Lindquist–Picci and Shalizi–Crutchfield named (as here),
   or restrict to causal states only? *Recommendation: as here; the Gaussian case
   is where the theorems are cleanest and the non-uniqueness is explicit.*
2. **"Training is approximate realization."** State as a reading (as here) or
   as a claim with subspace identification as the precedent? *Recommendation: a
   reading, one paragraph, with Van Overschee–De Moor and Defourneau–Petreczky
   cited.*
3. **Canonical Ω by default.** *Recommendation: yes.*
4. **§5.4 example.** HMM/transformer worked, Othello one paragraph.
   *Recommendation: yes.*
5. **Order of writing.** §1 → §3 → §5 → §2 → §4 → §6–7 → §0.
