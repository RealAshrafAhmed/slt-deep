# Outline v6: Five Postulates and One Map

*The mathematical structure of learning problems, Part 1: belief formation.*
(Alternative titles: *What a Learning Machine Is Made Of*; *World, Agent, Kernel*.)

Supersedes v5. Written in the register of a physics paper: postulates (structures,
not literature), derived notions, a dictionary, and the literature in a sources
appendix. Every statement in the body is a definition, a labelled assumption, a
cited theorem (named, not reproduced), or an example.

## Scope

**Belief formation only.** An agent observes records of a world and forms
predictive beliefs about what it will observe next. Nothing the agent does changes
the world. This is the setting of statistical inference, scientific modelling, and
an LLM without tools. Actions (reports, designs, control) are Note 2, an extension
of this note. Where a term from Note 2 is needed to name what is out of scope,
"control" is used in the control-theory sense: an input that steers the world's
dynamics.

**Ontology left open.** The note commits to a measure-free structure and to the
existence of a law on the records. Whether that law is in the world (ensemble,
propensity) or in the agent (de Finetti, B&S) changes one footnote and no theorem.

Length: ~12k words; five posts at the section breaks.

## What the note claims

1. Belief formation in statistics, ML and scientific modelling shares one
   structure: five postulates. Each community has formalised a subset; a table
   shows which. **Shown, not asserted.**
2. The learning machine is a *realization* of a kernel: an internal state space, a
   map from states to predictive kernels, and a filter. Every predictive model has
   one, explicitly or as the trivial realization. **Definition plus cited
   theorems (realization theory).**
3. The parameter space is a *parametrisation* of the realization, not a chart on
   it. Everything the machine computes is either pulled back along the map (and
   so invariant under reparametrisation by construction) or is a coordinate fact.
   **The one principle.**
4. A representation is judged by two standards: predictive alignment (the kernel
   is close to the world's) and efficiency (the state carries the predictive
   information and nothing more). Structural alignment, correspondence with a
   hidden observable, is a separate and optional claim. **Definitions plus cited
   theorems (predictive information, causal states).**

Not claimed: new theorems; a unification beyond the table; a position on whether
probability is in the world.

## Running examples

| Example | Symmetry | Realization | Carries |
|---|---|---|---|
| Adult heights (NHANES, binned): categorical → Gaussian → two-Gaussian mixture → mixture with sex recorded | exchangeable | S = W (Gaussian); S = W × {label} (mixture) | the map, image, fibres, singular point; pullbacks; a hidden observable (sex) and whether it is learnable **[V: NHANES; Schilling et al.; height means/SD]** |
| Text and an LLM | exchangeable documents, stationary within | trivial realization: S = context window | the main example; kernels, not laws; everything at scale; a closing paragraph per section |
| Transformers on HMM data; Othello-GPT | stationary | declared latent (HMM) vs learned (transformer) | a predictor that recovers the minimal realization; efficiency; alignment **[V: Shai et al. 2024; V-new: Li et al. 2023; Nanda et al. 2023]** |

---

## §0 Introduction (~900 words)

- The same objects under different names: likelihood / cross-entropy; Fisher
  information / Gauss–Newton; posterior predictive / ensemble; hidden state /
  latent / parameter / residual stream; probe / estimator of an observable.
- The four claims in four sentences; the diagram (§3) in one figure.
- Scope: belief formation; Note 2 adds actions. One paragraph on why this is not a
  limitation of the Bayesian view (Savage normal form; Bayesian control: Wald,
  Feldbaum's dual control, Åström, Bayes-adaptive MDPs). **[V-new: dates]**
- The ledger: given / chosen / assumed / approximated; one line per section.

## §1 Five postulates (~2,800 words)

Written as postulates with one example each; the cross-community table at the
end; citations only by theorem name.

**P1 (World).** A measurable space (Ω, 𝓕) of complete histories, past and future,
and a law P on it. One history ω* occurs. Averaging over Ω is averaging over what
is not yet known about ω*. Ω may be taken canonical, Ω = 𝒳^T, and enlarged only to
name hidden observables (a ledger entry).

**P2 (Observables).** An observable is a measurable function on Ω. The instrument
reads a family X_t: Ω → 𝒳, 𝒳 finite (finite precision). These generate the
agent's filtration 𝓖_t = σ(X_s : s ≤ t) and its limit 𝓖_∞. What the agent knows
at time t is the atom A_t(ω*) = {ω : X_s(ω) = X_s(ω*), s ≤ t}; the predictive law
is P(· | 𝓖_t), supported on it.
- *Hidden* is a property, not a kind: τ is hidden at t if not 𝓖_t-measurable.
- *Learnable*: τ is 𝓖_∞-measurable mod P. Doob: E[τ | 𝓖_n] → E[τ | 𝓖_∞] a.s.;
  learnable iff the limit is τ. Not learnable ⇒ no agent, no data suffice.
  **[V-new: Doob 1949; Schervish ch. 1]**
- A sample is X_{1:n}(ω*): one history, seen through the instrument. "Different
  trials" means either coordinates of the one history (the only kind that gives
  data) or alternative futures in A_t(ω*) (counterfactuals; enter only in
  expectations).

**P3 (Symmetry; assumption A1).** A group G acting on the index set with P
invariant. Ergodic decomposition: P = ∫ P_θ Π(dθ); one history identifies its own
component θ by averages (ergodic theorem). θ is the **macrostate**: an observable
of ω*, constant on G-orbits, 𝓖_∞-measurable, hence learnable. A1 is the ergodic
hypothesis for the record process: one history stands in for the ensemble. Table:

| G | Components | Theorem | Example |
|---|---|---|---|
| permutations (exchangeable) | i.i.d. laws | de Finetti; Hewitt–Savage; Diaconis–Freedman (finite) | heights |
| exchangeable units, sequential within | laws on units | de Finetti on units | documents |
| shifts (stationary) | stationary ergodic laws | ergodic decomposition | tokens within a document; HMM data |

  Without a symmetry, one history identifies nothing beyond itself. Options:
  weaker regularity (mixing); design the symmetry in (random sampling makes
  exchangeability true by construction: Note 2); accept non-learnability. A1 is
  testable toward rejection (test martingales). **[V: Hewitt–Savage/Kallenberg;
  Diaconis–Freedman; ergodic decomposition; Vovk et al. 2003. V-new: Kallenberg
  2005 as the single reference]**

**P4 (Meeting space: kernels).** What is learned is a kernel from the conditioned
past to the predicted future, κ(dx_{t+1} | x_{1:t}) (an LLM) or κ(dy | x) (a
GLM). The law of the conditioning variable is unmodelled and enters only as a
weight. The meeting space 𝒦 is the space of such kernels; the world's point in it
is κ_θ. Laws and kernel families are equivalent (Ionescu-Tulcea), so nothing is
lost, but the emphasis is on the kernel: the agent makes no claim about the
marginal and is not wrong about it.

**P5 (Agent: a realization).** A theory is a triple (S, p, π):
- an internal state space S;
- a map p: S → 𝒦 (a kernel over the future given the state);
- a filter π_t ∈ Δ(S), updated from the past, so that the agent's predictive is
  ∫ p_s(· | x_{1:t}) π_t(ds).
Facts (realization theory):
- the trivial realization always exists: S = the past, π_t = δ_{x_{1:t}}; an
  autoregressive transformer is this, with the context window as S;
- the minimal realization exists and is unique up to isomorphism: the causal
  states (equivalence classes of pasts with the same conditional future), a
  functional of the kernel alone; Knight's prediction process; Kalman's
  realization theory in the linear case;
- a declared latent (HMM states, a mixture label) is a choice of realization
  between the two. The further claim that S coincides with a hidden observable of
  the world is **optional** and labelled when made.
A parameter space W parametrises S (or S = W); the model map is the composite
W → S → 𝒦. Learning is the agent's dynamics driven by the record stream: on
beliefs (the filter; Bayes on W) or on coordinates (gradient flow).
**[V-new: Shalizi & Crutchfield 2001 (minimality, uniqueness); Knight 1975;
Kalman 1963 / Kalman–Falb–Arbib 1969; Littman–Sutton–Singh 2002]**

**The cross-community table.** Which postulates each formalises and which it
leaves implicit: Kolmogorov probability (P1–P2), ergodic theory and Kallenberg's
symmetries (P3), Blackwell–Le Cam experiments (P2, P4), Bayesian statistics / B&S
(P3, P5 with S = Θ), computational mechanics (P5 minimal), systems theory /
realization (P5), filtering theory (P5 filter), supervised ML (P4, W), SLT (W → 𝒦).

**Ledger.** Given: the record process. Chosen: the instrument (X_t), the
realization (S, p), the coordinates W. Assumed: A1; standard Borel on Ω and S.
Derived: θ, learnability, the filter.

## §2 The meeting space and what lives on it (~1,500 words)

**Goal:** name the structures on 𝒦 that get pulled back, all pointwise in the
conditioning variable and averaged under its (unmodelled) law.

- Convexity (mixtures of kernels; averaging happens here).
- Divergence: expected conditional KL; for processes the relative entropy rate,
  decomposed by the chain rule into one-step terms. The log score and its regret =
  KL (Gibbs). Justified without decision theory: code length (prequential, MDL);
  the f-divergence that decomposes along the chain rule. Forward pointer: bounds
  every bounded decision's regret (Note 2). **[V: Pinsker; Bernardo 1979. V-new:
  chain-rule characterisation of KL among f-divergences]**
- The Fisher–Rao form, pointwise then averaged; Čencov's theorem. **[V-new:
  Čencov 1982; Ay et al. 2017]**
- Observables as functionals of a kernel: the one-step predictive, the entropy
  rate h, the predictive information I(past; future) = excess entropy E.
- **The learnable part of the past–future relation.** I(past; future) is what a
  representation can carry; H(future | past) is noise. **Regularity and noise**
  in one paragraph (from the companion). **[V: block entropy; Bialek, Nemenman &
  Tishby 2001]**
- The KL projection κ₀ of κ_θ onto a subset; uniqueness under relatively finite
  variance. **[V: Watanabe 2018 ch. 3]**
- The minimal realization as a functional of the kernel: causal states; C_μ ≥ E.
  **[V: Shalizi & Crutchfield 2001]**

**For an LLM.** The world's kernel is next-token given context across documents;
its minimal realization is whatever a perfect next-token predictor must track;
its predictive information is finite and sets what any context window can carry.

**Ledger.** All properties of κ_θ (given, unknown).

## §3 The map and its pullbacks (~2,800 words; the core)

**Goal:** the diagram, and the one principle.

```
   Δ(W) ──p_#──►  Δ(𝒦)  ──mean──►  predictive kernel
     │
     W ──► S ──p──►  𝒦  ◄──κ_·──  Θ           (agent ── data ── world)
                     │
   K, L_n, F, f∘p ◄── pull back ── KL, Fisher–Rao, observables f
```

The agent (W, S) and the world (Θ) both map into 𝒦 and never into each other.
The data, the empirical kernel, sit in 𝒦.

- **3.1 The map.** W → S → 𝒦; image, fibres, identifiability. Fibres are
  relative to the weight (the law of the conditioning variable). Heights ladder:
  four maps, one dataset.
- **3.1b Charts versus parametrisations.** Unchanged from v5: regular case =
  chart; two failures (non-injective; rank drop); the right statement (a
  parametrisation of a subset of 𝒦, not a chart); resolution of singularities;
  history in one paragraph; what survives (pullbacks of functions, tensors,
  observables for any smooth map); no ambient manifold needed; the physics box
  (gauge redundancy, singular reduction, stabilisers); the two alternatives
  placed. **[V-new rows as in v5]**
- **3.2 Pullbacks, one at a time.** KL → K(w); log loss → L(w) = h + K(w); data →
  L_n(w); Fisher–Rao → F(w) = JᵀgJ, degenerate where the map loses rank; λ as the
  invariant replacing d there, one paragraph; observable f → f∘p.
  **[V: RLCT volume scaling; invariance under resolution]**
- **3.3 Observables of the agent, and learnability on both sides.** A quantity
  τ on S (or W) is a legitimate observable of the agent iff constant on fibres
  (factors through p). Twin statement on the world side: a hidden observable is
  learnable iff 𝓖_∞-measurable (P2). Heights: the component mean difference is
  constant on fibres up to label switching, not at the merged point; recording
  sex makes it a conditional observable and learnable.
- **3.4 The principle, with three contrasts.** Pulled-back quantities transform
  by the chain rule under reparametrisation; quantities defined only on W do not.
  Fisher information vs Hessian in coordinates (Dinh et al. 2017); MLE
  (equivariant) vs MAP (prior density in coordinates); the predictive (pushed
  forward) vs the posterior mean of w. Two priors differing only along fibres give
  the same belief on kernels. **[V: Dinh; Jeffreys; Lehmann–Casella]**
- **3.5 Three errors on the diagram.** Bottou–Bousquet: approximation = K(w₀);
  estimation = L_n vs L; optimisation = where the process stops. **[V-new]**

**For an LLM.** S = context window (trivial realization), W the weights, p the
next-token kernel. Fibres from permutations, rescalings, embedding × unembedding.
"A feature" is a candidate observable on W, legitimate only if constant on fibres.

**Ledger.** S, p, W chosen; every pulled-back quantity derived; λ a cited property
of the chosen map against κ_θ.

## §4 Learning as a process on W, seen on 𝒦 (~1,800 words)

- **4.1 Solution concepts as measures on W.** MLE, MAP (point masses); the Bayes
  posterior (Gibbs measure, tempered); the law of an SGD run. Failure modes: the
  mixture MLE does not exist; MAP is not invariant. **[V-new: Kiefer–Wolfowitz;
  Day 1969]**
- **4.2 Push-forward and the predictive.** A measure on W → a measure on 𝒦 → its
  mean, which may leave the image. Averaging happens on kernels, not on W.
  Ensembles and posterior predictives are one construction.
- **4.3 Convergence, stated on 𝒦.** Towards κ_θ or κ₀; pointwise (SLLN /
  ergodic), uniform (ULLN; consistency to the fibre W₀), fluctuations (regular
  d/2 vs singular λ), cited. Many histories' coordinates (exchangeable units) vs
  one long path. **[V: free energy; Devroye]**
- **4.4 What can be computed.** Expectations under κ_θ (samples only); argmin on
  non-convex W; integrals on W. Deterministic vs stochastic approximation;
  one-pass SGD ≈ stochastic approximation of L. The returned object is a random
  point with an algorithm-dependent law; named, not analysed (Part 2).
- **4.5 Evaluation.** Held-out loss estimates the pulled-back KL; unbiased only
  under exchangeability of units (A1).

**For an LLM.** Pre-training ≈ one-pass stochastic approximation over
exchangeable documents; a checkpoint is a sample from an algorithm-dependent law
on W.

**Ledger.** Everything approximated; say by what.

## §5 Representations: alignment and efficiency (~2,000 words)

**Goal:** what a good internal state is, without deciding whether it is "real".

- **5.1 Two senses of alignment.** *Predictive*: the agent's kernel → κ_θ (or
  κ₀). All that log loss and held-out evaluation see. *Structural*: the agent's
  S corresponds to the minimal realization of κ_θ (a functional of the kernel,
  world-free) or, under the optional assumption, to a hidden observable.
  Structural is not implied by predictive in general; it is well posed only for
  quantities constant on fibres (agent side) and learnable (world side).
- **5.2 Efficiency.** A realization is *sufficient* if S carries I(past; future);
  it is *efficient* if it carries nothing more. Carrying more = distinguishing
  pasts with the same future = encoding noise: overfitting at the level of
  representation. The causal states are the unique minimal sufficient
  realization (C_μ ≥ E). The information bottleneck is the variational form:
  compress the past subject to keeping I(S; future). **[V-new: Tishby, Pereira
  & Bialek 1999; Still 2009; Shalizi & Crutchfield 2001]**
- **5.3 Science and prediction as two requests.** Prediction asks for predictive
  alignment; the state is a means. Science asks for structural alignment with a
  *named* hidden observable (an effect size, a board, a disease state) and so
  needs learnability (world side) and identifiability (agent side). Same
  mathematics; different functional requested. Breiman's two cultures as a
  difference in the realization chosen.
- **5.4 Predictors recover the minimal realization.** Evidence: transformers on
  HMM data linearly represent the belief-state (mixed-state) geometry, i.e. the
  minimal realization's filter (Shai et al. 2024); Othello-GPT represents the
  board (Li et al. 2023; Nanda et al. 2023), which is the predictive state of the
  game. Reason, from §2: optimal prediction requires the predictive information,
  and the minimal realization is what carries it. So predictive alignment forces
  structural alignment *with the minimal realization*, and the optional world
  claim is exactly the step from "the minimal realization" to "the board exists".
  Interpretability = estimating an observable on W, legitimate iff constant on
  fibres (§3.3). **[V: Shai et al.; V-new: Othello papers]**
- **5.5 The instrumentalist reading.** "Forget whether κ_θ is in the world": a
  model is a code for the regularity in the past–future relation (Rissanen;
  Dawid; Grünwald). Everything in §1–§5 survives under it; only the optional
  world claim is dropped. **[V-new: Rissanen 1989; Dawid 1984; Grünwald 2007]**

**For an LLM.** Pre-training asks only for predictive alignment; what it learns
about hidden structure is whatever the minimal realization of text requires, and
the context window bounds how much of I(past; future) it can carry.

**Ledger.** Alignment and efficiency are derived standards; the world claim is an
assumption, made or not.

## §6 Field guide (~800 words + table)

One row per quantity people compute; which postulate it belongs to, whether it is
pulled back or pushed forward, whether invariant under reparametrisation. Rows as
in v5 (likelihood / cross-entropy / perplexity; Fisher information; Hessian
sharpness; weight norms; posterior predictive / ensemble; posterior mean of w;
calibration; belief state; probe / concept vector; identifiability; AIC / BIC /
WAIC; λ) plus:

| Term | Postulate | Lives on | Pulled back / pushed forward | Invariant? |
|---|---|---|---|---|
| hidden state, latent, residual stream | P5 (a realization) | S | — | up to isomorphism of realizations |
| context window | P5 (trivial realization) | S = past | — | yes |
| predictive information, excess entropy | P4 observable | 𝒦 | — | yes |
| statistical complexity C_μ | P5 minimal | 𝒦 | — | yes |
| representation "collapse" / "superposition" | — | W | no | no, unless restated on S |

## §7 Dictionary and what this makes possible (~800 words)

- **Dictionary** (structural rows only; analogies marked): phase space /
  microstate ↔ Ω, ω; observable ↔ X_t; ensemble ↔ P, π; ergodic hypothesis ↔ A1;
  macrostate ↔ θ; coarse-graining ↔ the instrument; Mori–Zwanzig memory + noise
  ↔ the records alone are not Markov, the realization restores it; Takens
  reconstruction ↔ the trivial realization recovers the state; gauge redundancy
  ↔ fibres; canonical transformation ↔ reparametrisation; energy ↔ K; metric ↔ F.
  **[V-new: Takens 1981 hypotheses; Mori–Zwanzig form; Koopman 1931]**
- **Note 2** (actions): reports, designs, control; regret bounds; dual control;
  tuning an LLM as a change of utility; equivalence of learning machines;
  invariants of the learning dynamics; "only invariants are worth measuring" for
  interpretability and safety (one paragraph, labelled motivation).
- Open questions, one list.

## Appendix A: notation and ledger (1 page)

## Appendix B: sources, one line per postulate and derived notion

P1–P2 Kolmogorov; Kallenberg *Foundations*; Schervish ch. 1 (θ as a function of
the observables). P3 Kallenberg 2005; Lauritzen 1984/1988; Diaconis–Freedman.
P4 Blackwell 1951/53; Le Cam; Torgersen; Ionescu-Tulcea. P5 Kalman 1963;
Knight 1975; Shalizi & Crutchfield 2001; Littman–Sutton–Singh 2002; Bain & Crisan
(filtering). Meeting-space geometry: Čencov; Amari–Nagaoka; Ay–Jost–Lê–Schwachhöfer.
Singular maps: Watanabe 2009/2018; Hironaka. Predictive information and
bottleneck: Bialek–Nemenman–Tishby 2001; Tishby–Pereira–Bialek 1999; Still 2009.
Instrumentalism: Rissanen 1989; Dawid 1984; Grünwald 2007; Shafer & Vovk 2019.
Agent/world separation without actions: Shafer & Vovk (Forecaster vs Reality).
Physics box: Singer 1978; Ebin 1970; Sjamaar & Lerman 1991; Luna 1973;
Mori 1965; Zwanzig 1973; Takens 1981; Koopman 1931.

---

## Changes from v5

| v5 | v6 |
|---|---|
| Six objects, with actions and utility | Five postulates; actions to Note 2 |
| Hidden state Z_t as primitive | Hidden = not 𝓖_t-measurable; learnable = 𝓖_∞-measurable |
| Meeting space Δ(𝒳^T) | Kernels 𝒦; the conditioning law is a weight |
| Model = (S, p) | Model = a realization (S, p, π); trivial and minimal realizations as theorems |
| Alignment: predictive vs structural (with the world) | Structural alignment with the minimal realization (world-free); world claim optional; efficiency as a second standard |
| B&S framing in the body | Postulates in the body; literature in Appendix B |

## Reuse of `note.tex`

§1 (problem) → Note 2. §2 (records) → P2, one paragraph. §3 (sampling symmetry)
→ P3. §4 (regularity and noise) → §2, one paragraph (returns from the companion).
§5 (world symmetries) → companion. §6 (models) → P5 and §3.1. §7 (utility → KL)
→ §2 (code-length justification kept; regret bound to Note 2).

## Verification: new rows (add to `verification.md`)

Doob 1949 (posterior → 𝓖_∞-conditional); Kallenberg 2005 (symmetry table);
Shalizi & Crutchfield 2001 (causal states: minimal, unique, C_μ ≥ E); Knight
1975; Kalman 1963 / Kalman–Falb–Arbib 1969; Littman–Sutton–Singh 2002; Tishby et
al. 1999; Still 2009; Bialek et al. 2001; Rissanen 1989; Dawid 1984; Grünwald
2007; Shafer & Vovk 2019; chain-rule characterisation of KL among f-divergences;
Čencov 1982; Ay et al. 2017; Bottou & Bousquet 2008; Kiefer–Wolfowitz 1956 / Day
1969; Li et al. 2023; Nanda et al. 2023; Hironaka 1964; Singer 1978; Ebin 1970;
Fischer 1970; Sjamaar & Lerman 1991; Luna 1973; Pistone & Sempi 1995; Mori 1965;
Zwanzig 1973; Takens 1981; Koopman 1931; Bayesian control dates (Wald 1947,
Feldbaum 1960–61, Åström 1965, Martin 1967, Duff 2002).

## Decisions needed

1. **Canonical Ω.** State P1 with Ω = 𝒳^T by default and enlarge only when a
   hidden observable is named? *Recommendation: yes.*
2. **Minimal-realization theorem.** State Shalizi–Crutchfield's minimality and
   uniqueness as the one theorem of P5, or also Kalman's linear case?
   *Recommendation: both named, one sentence each.*
3. **§5.4 example.** HMM/transformer (exact minimal realization known; links to
   the toy-transformer work on this branch) as the worked example, Othello in one
   paragraph. *Recommendation: yes.*
4. **Order of writing.** §1 (postulates) first, then §3, then §5, then the rest.
