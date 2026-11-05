# Outline v5: Six Objects and One Map

*The mathematical structure of learning problems, Part 1.*
(Alternative titles: *What a Learning Machine Is Made Of*; *The Objects of Learning*.)

Distilled from `outline.md` (v1), v2–v4, `design-notes.md`, `verification.md` and
`note.tex`. This is the outline to write from. The principle for inclusion is
rigour: every section is a list of definitions, labelled assumptions and cited
theorems, and nothing enters unless a later section or the field guide (§6) uses it.

## Scope

**The setting is Bernardo & Smith's:** an agent that observes and reports, and
possibly chooses what to observe, but does not change the world. This is the
setting of an LLM without tools, and of statistical inference and scientific
modelling. The top objective is to show the underlying structure for an LLM.

**Terminology kept for later.** Actions are classified by the object they can
change:

| Channel | Changes | In this note |
|---|---|---|
| **report** | nothing in the world; only the consequence | yes: prediction, estimation, classification, tests |
| **design** | what is recorded (objects 2–3) | yes, briefly: experiment design (Lindley) |
| **control** | the world's subsequent states (object 1) | no: RL, robotics, LLMs with tools ("agents"), performative prediction. Part 2 or a companion |

"Control" is used in the control-theory sense: an input that steers the world's
dynamics. The note uses the word only to name what is out of scope.

Control is outside this note by choice, not because Bayesian theory stops at
reports. Savage's framework handles interaction in normal form (acts are contingent
plans; the state includes how the world responds to each plan), and Bayesian
control is a mature literature: sequential analysis (Wald 1947), bandits (Thompson
1933; Gittins 1979), Bayesian adaptive and **dual control** (Bellman & Kalaba;
Feldbaum 1960–61: a control action both steers and informs, so control and design
do not separate), POMDP belief states (Åström 1965), Bayes-adaptive MDPs (Martin
1967; Duff 2002), Bayesian games (Harsanyi 1967–68). B&S concentrate on inference.
One paragraph in §0 says this. **[V-new: all dates and attributions in this
paragraph]**

## What the note claims

1. Every learning problem in statistics, ML and scientific modelling is a decision
   problem under uncertainty (Wald; Savage; Bernardo & Smith). **Cited, not
   claimed.**
2. Such a problem is specified by six objects. Each community has formalised the
   ones it needed and left the others implicit. **Shown by a table, not asserted.**
3. The learning machine is an internal representation with a map into the space of
   laws of the records (object 6). Everything computable about the machine is
   either pulled back along this map, or lives only on the parameter space. The
   first kind is invariant under reparametrisation by construction; the second
   kind is not. **This is the one principle the note states**, and the field guide
   applies it to the quantities each community computes.
4. Fields differ in which objects are fixed, which are learned, and which
   functional of the world the utility asks about. Not in the mathematics.

Not claimed: no new theorems; no unification beyond the table; no claim about
which theory is right. Part 2 (equivalence of learning machines, invariants of the
learning process, control) is pointed to in §7 and nowhere else.

Length: ~12k words, under an hour; splits into five posts at the section breaks.

## What is dropped, and where it goes

| Material | From | Goes to |
|---|---|---|
| Probability as a chosen language; regularity and noise; world symmetries; Kolmogorov / structure function | v1 I.2–I.4, `note.tex` §2–5 | Foundations companion (`note.tex` §§2–5 is the draft) |
| Convergence in three layers; Watanabe free energy, WAIC; singularity taxonomy | v1 II.3, II.5 | one paragraph in §4 plus citations; rest to Part 2 |
| Equivalence group; invariants of dynamics; conserved quantities; control bundles | v3 | Part 2 |
| Control: RL, environment kernels, regimes, ignorability, performative prediction | v1 I.3, v4 §5 | Part 2 / companion; named in Scope and §7 |
| Cat; robotics; science template | v1 Part IV | parking lot |
| Safety motivation | v2 §6 | one paragraph in §7, labelled as motivation |
| Control-theory dictionary | v3 | notes only; not in the text |

The `verification.md` rows that survive are marked **[V]** below. Rows not marked
move to the companion's list. New rows are marked **[V-new]**.

## Notation (the appendix, introduced in this order in §1)

- **The world.** A filtered probability space (Ω, 𝓕, (𝓕_t)_{t∈T}, P). A point
  ω ∈ Ω is one complete history. T is ℕ (arrival or time order) or a finite index
  set.
- **Processes on it.** Hidden states Z_t: Ω → 𝒵; records X_t: Ω → 𝒳.
- **Two filtrations.** The world's 𝓕_t = σ(Z_s, X_s : s ≤ t); the agent's
  𝓖_t = σ(X_s : s ≤ t) ⊂ 𝓕_t. *Hidden* means the gap between them.
- **Three kinds of unknown.**

| Unknown | Measurable w.r.t. | Example |
|---|---|---|
| ω | 𝓕 | the full history |
| Z_t (transient hidden state) | 𝓕_t | the board position; the topic of a document so far |
| θ (stable state of the world) | the invariant σ-field | the height distribution; the law of documents |

- **The experiment.** θ ↦ P_θ ∈ Δ(𝒳^T), the law of the record process given θ
  (a regular conditional law; Blackwell's experiment). By Ionescu-Tulcea it is the
  family of one-step kernels P_θ(dx_{t+1} | x_{1:t}).
- **Beliefs and the model.** An internal representation S, a map p: S → Δ(𝒳^T)
  (in practice one-step kernels p_s(dx_{t+1} | x_{1:t})), a parametrisation
  W → S, and beliefs π ∈ Δ(S).
- **Actions.** A set A of reports (and designs), a bounded utility u.
- **Standing assumption.** All spaces standard Borel, so that conditioning,
  disintegration and the representation theorems apply.

## Running examples

| Example | Symmetry | Carries |
|---|---|---|
| Adult heights (NHANES, binned): categorical → Gaussian → two-Gaussian mixture → mixture with sex recorded | exchangeable | the map, image, fibres, singular point; pullbacks; the ladder of idealisations; a hidden quantity (sex difference) and whether it is constant on fibres **[V: NHANES; Schilling et al. bimodality; height means/SD]** |
| Text and an LLM | exchangeable documents; sequential within a document | the main example: everything at scale; the two-level structure; one closing paragraph per section |
| Sequence models that represent hidden state: Othello-GPT; chess-playing language models; transformers on HMM data | stationary sequences | a model trained only to predict that tracks Z_t (§5) **[V-new: Li et al. 2023; Nanda et al. 2023; Karvonen 2024; V: Shai et al. 2024]** |

All three are passive: the models report, they do not control.

---

## §0 Introduction (~900 words)

- The same objects under different names: likelihood / cross-entropy;
  Fisher information / Gauss–Newton; posterior predictive / ensemble;
  state / parameter / latent; probe / estimator of a functional.
- The claims above, in four sentences.
- The diagram (§3) previewed in one figure.
- Scope: the report setting; what "control" means and why it is left for later.
- The ledger discipline: given / chosen / assumed / approximated, one line at the
  end of each section.

## §1 Six objects (~2,800 words)

**Goal:** define the objects once, show where each already has a rigorous home.

| # | Object | Definition in the note | Rigorous home |
|---|---|---|---|
| 1 | State of the world, with dynamics | (Ω, 𝓕, (𝓕_t), P); the stable state θ and the hidden process Z_t | Savage's state; de Finetti's directing measure (B&S ch. 4); POMDP state |
| 2 | Observables and the map state → observables | records X_t; the experiment θ ↦ P_θ ∈ Δ(𝒳^T); observables as functionals on Δ(𝒳^T) | Blackwell's experiment (1951, 1953); Le Cam; POMDP observation kernel |
| 3 | Measurement and data | the agent's filtration 𝓖_t; the observed path x_{1:n} | `note.tex` §2–3 (compressed) |
| 4 | Beliefs | π ∈ Δ(S), updated by conditioning on 𝓖_t | B&S posterior; POMDP belief state (Åström 1965; Kaelbling, Littman & Cassandra 1998) |
| 5 | Actions and consequences | reports (and designs); bounded utility u | Savage; B&S ch. 2; `note.tex` §1 |
| 6 | The model | internal representation S, map p: S → Δ(𝒳^T), parametrisation W → S | B&S: subjective choice; POMDP: assumed known; ML: the architecture; **not studied as a space by either** |

Sub-sections:

- **1.1 The decision problem (object 5).** Actions, outcomes, bounded utility; the
  three cases (acting on a report; predicting with a utility; predicting with the
  utility left open). Act–state independence (Savage; B&S ch. 2) stated as the
  defining property of the report setting. Reuse `note.tex` §1, removing the
  driving example. **[V-new: act–state independence in Savage and B&S ch. 2]**
- **1.2 The world (object 1).** The filtered probability space; ω, Z_t, θ; the
  two filtrations. The world's law of the records is a process law, not a law on
  𝒳.
- **1.3 Symmetry as an assumption.** A1 is a family indexed by a group G acting on
  T. G-invariant laws form a convex set; its extreme points are the ergodic laws;
  every invariant law decomposes uniquely over them; the realised path identifies
  its own component almost surely. **The stable state θ is that component**, which
  is why it can be learned. Table of cases:

| G | Extreme points | Theorem | Example |
|---|---|---|---|
| permutations (exchangeable) | i.i.d. laws ≅ Δ(𝒳) | de Finetti; Hewitt–Savage; finite form Diaconis–Freedman | heights |
| exchangeable units, sequential within | laws on units ≅ Δ(𝒳*) | de Finetti on units | documents |
| shifts (stationary) | stationary ergodic laws | ergodic decomposition | text within a document; HMM data |
| Markov exchangeable (recurrent) | Markov chains | Diaconis–Freedman 1980 | optional remark |

  With no symmetry: one path identifies nothing beyond itself. The options are a
  weaker regularity (mixing), designing the symmetry in (randomisation, i.i.d.
  sampling of units), or accepting the target is not learnable. Testability of A1
  toward rejection (conformal test martingales). **[V: Hewitt–Savage/Kallenberg;
  Diaconis–Freedman finite bound; Kolmogorov extension/Birkhoff/ergodic
  decomposition; Vovk et al. 2003. V-new: Kallenberg 2005 *Probabilistic
  Symmetries* as the single reference for the table; Diaconis–Freedman 1980
  recurrence condition]**
- **1.4 Observables and experiments (object 2).** Blackwell's definition: an
  experiment is a kernel from states to laws of observations. Comparison of
  experiments (sufficiency / garbling) as the order on "which measurement system
  is more informative", stated without proof. A hidden quantity θ ↦ τ(θ) is
  observable in principle iff it factors through θ ↦ P_θ. **[V-new: Blackwell
  1951, 1953; Le Cam 1964; Torgersen 1991]**
- **1.5 Measurement and records (object 3).** One paragraph from `note.tex` §2:
  device as a Markov kernel, finite records; the agent's filtration 𝓖_t. Design
  (choosing the experiment) as the second action channel, one paragraph
  (Lindley 1956). **[V: Lindley]**
- **1.6 The model (object 6).** An internal representation S and a map
  p: S → Δ(𝒳^T), given by one-step kernels; a parametrisation W → S. Examples:
  S = W (Gaussian); S = W × {labels} (mixture); S = W × hidden states (HMM);
  S = Δ(𝒳) (nonparametric). The model's own symmetry (i.i.d., time-homogeneous)
  is part of the choice and may differ from the world's. Image, fibres,
  identifiability, regular vs singular (definition only), from `note.tex` §6.
  **[V: regular-model definition matches Watanabe]**
- **1.7 Beliefs (object 4).** π ∈ Δ(S); updating by conditioning; the
  predictive is the one-step conditional
  P̂(dx_{n+1} | x_{1:n}) = ∫ p_s(dx_{n+1} | x_{1:n}) π_n(ds) (Dawid's prequential
  view). De Finetti read the agent's way: an agent with exchangeable predictives
  behaves as if S = Δ(𝒳) with a prior (B&S ch. 4), so Δ(𝒳) is the universal
  internal representation that A1 licenses and every parametric S is a
  restriction of it. **[V-new: B&S ch. 4; Dawid 1984]**
- **1.8 Four communities, one table.** Which objects each fixes, learns, or leaves
  implicit: Bayesian statistics, Blackwell–Le Cam decision theory, POMDPs,
  supervised ML, scientific modelling. **[V-new: Wald 1950; DeGroot 1970;
  Berger 1985; B&S §2, §4, §6.1]**
- **1.9 Prediction then averaging.** When u depends on the world only through
  future observables, the Bayes action factors through the predictive: push the
  belief forward to a measure on laws, average it, then maximise expected utility.
  The log score when the use is left open (Bernardo 1979). When u depends on a
  hidden quantity (θ or Z_t), the action does not factor through the predictive.
  **[V: Bernardo 1979 uniqueness conditions]**

**Ledger.** Given: object 5 and the record process. Chosen: the measurement
system, S, p, W. Assumed: A1 (gives θ), standard Borel. Derived: beliefs from the
rest by conditioning.

## §2 The space of laws of the records (~1,500 words)

**Goal:** name the structures on Δ(𝒳^T) that later get pulled back.

- Convexity: mixtures, so averaging happens here.
- Divergences and scoring rules. KL between process laws and the relative entropy
  rate lim (1/n) KL(P_{1:n} ‖ Q_{1:n}); the chain rule turns it into a sum of
  one-step KLs. Proper scoring rules (Gneiting & Raftery 2007); the log score and
  its regret = KL (Gibbs). In the i.i.d. case everything reduces to KL(q ‖ p).
  Regret bound for any bounded utility (Proposition from `note.tex` §7, kept).
  **[V: Pinsker constant and TV convention; Bernardo 1979]**
- The Fisher–Rao metric and Čencov's theorem (the unique metric invariant under
  sufficient statistics); its rate version for processes. **[V-new: Čencov 1982;
  Ay, Jost, Lê & Schwachhöfer 2017]**
- Observables as functionals of a process law: the one-step predictive, the
  entropy rate h, the excess entropy E = I(past; future), the Bayes action of a
  decision problem. **[V: block entropy and excess entropy]**
- The KL projection p₀ onto a subset; uniqueness under relatively finite
  variance. **[V: Watanabe 2018 ch. 3]**
- **Predictive states:** the minimal sufficient summary of the past for the
  future (Shalizi & Crutchfield 2001; Littman, Sutton & Singh 2002). A functional
  of the law alone. This is the bridge from the observables to Z_t, used in §5.
  **[V: statistical complexity ≥ excess entropy; V-new: PSRs]**

**For an LLM.** The world's law is a law on documents; within a document it is a
process law; its predictive state is whatever a perfect next-token predictor must
track.

**Ledger.** All of this is a property of the world's law (given, unknown) or of the
utility (given).

## §3 The map and its pullbacks (~2,800 words; the core)

**Goal:** the diagram, and the one principle.

```
   Δ(W) ──p_#──►  Δ(Δ(𝒳^T))  ──mean──►  predictive
     │
     W ──► S ──p──►  Δ(𝒳^T)  ◄──P_·──  Θ          (agent ── data ── world)
                      │
   K, L_n, F, f∘p ◄── pull back ── KL rate, Fisher–Rao, observables f
```

The agent (W, S) and the world (Θ, Z) both map into Δ(𝒳^T) and never into each
other. The data sit in the middle.

- **3.1 The map.** p = output map ∘ parametrisation. Image, fibres,
  identifiability. Heights ladder: four maps, one dataset.
- **3.1b Charts versus parametrisations.** The coordinate view and where it
  stops.
  - *Regular case = chart.* If p is an injective immersion, the image is an
    embedded submanifold and W is a global chart on it; the pulled-back
    Fisher–Rao tensor is a Riemannian metric; reparametrisations are
    diffeomorphisms. This is information geometry (Amari), and "regular" in
    Watanabe's sense is exactly this condition.
  - *Two failures.* (i) Not injective: fibres from symmetries; W is a redundant
    cover of the image, and W/~ is the candidate for coordinates. (ii) Rank
    drops: the image is not a manifold at that point, and the pulled-back tensor
    is degenerate. Under analyticity the image is an analytic set with singular
    strata. Standard example t ↦ (t², t³) (a cusp); the heights mixture at the
    merged point; (a, b) ↦ N(ab, 1).
  - *The right statement.* Object 6 is a **parametrisation** of a subset of the
    space of laws, not a chart on it. In differential geometry a parametrisation
    is a map into the ambient space and a chart is a map out of the manifold; they
    coincide only when the parametrisation is regular. Singular images are the
    subject of singularity theory of smooth maps (Whitney, Thom, Arnold) on the
    differential side and of algebraic / analytic geometry on the other.
    Resolution of singularities (Hironaka 1964) restores local coordinates: a
    manifold U and a proper map g: U → W with K∘g in normal-crossing form. The
    coordinates return locally, after a non-injective blow-up, and λ and m are
    what survive it. **[V-new: Hironaka 1964; Watanabe 2009 ch. 2–3]**
  - *History, one paragraph.* In geometry the parametrisation came first (Gauss),
    then the abstract manifold (Riemann; Whitney), then singularity theory as a
    response to its limits; algebraic geometry always had singular images.
    Statistics inherited the chart view (Rao, Čencov, Amari) and took up the
    parametrisation view only with Watanabe. **[V-new: dates for Veblen–Whitehead,
    Whitney 1955, Hartigan 1985, Fukumizu & Amari 2000, Watanabe 2001]**
  - *What survives.* Pullback of functions, tensors and observables is defined
    for any smooth map, not only diffeomorphisms. K, L, F = JᵀgJ and f∘p are
    always well defined and invariant under reparametrisation as objects on W;
    what fails is reading F as a metric. Hence the principle of §3.4 holds in the
    singular case, and this section is framed around pullbacks, not charts.
  - *No ambient manifold is needed.* Pullback needs only the structures being
    pulled back (a function, a quadratic form, functionals), not a manifold
    structure on the space of laws. Parametrised measure models (Ay et al.) are
    set up for exactly this; Pistone–Sempi give a Banach-manifold structure when
    one is wanted. Physics does the same on non-manifold state spaces (convex sets
    of states in statistical mechanics; density matrices in quantum mechanics;
    Wasserstein space). **[V-new: Pistone & Sempi 1995; Ay et al. 2017;
    Bengtsson & Życzkowski; Lott–Villani, Sturm]**
  - *Physics box: redundancy and its quotient.* Gauge theory: the physical space
    is the quotient of connections by the gauge group, a stratified space whose
    singular strata are the connections with extra symmetry; no global gauge
    fixing (Gribov; Singer 1978); physical observables are gauge-invariant
    functions. General relativity: metrics modulo diffeomorphisms are singular at
    metrics with isometries (Ebin 1970; Fischer 1970). Mechanics: symplectic
    reduction by a non-free action gives a stratified space (Sjamaar & Lerman
    1991). Moduli spaces: singular points sit at objects with non-trivial
    automorphisms (Luna 1973). The common sentence: **a singular point of a
    parametrised family is one whose stabiliser is larger than generic; the space
    to do geometry on is the stratified quotient; the legitimate observables are
    the invariants of the redundancy.** Fibres = orbits; "constant on fibres" =
    gauge invariance. **[V-new: Singer 1978; Kondracki & Rogulski 1983;
    Ebin 1970; Fischer 1970; Sjamaar & Lerman 1991; Luna 1973]**
  - *The two alternatives, placed.* "Model = coordinates on the image" is the
    regular special case. "Singularity is an optimisation nuisance" is the view
    the note argues against, since the singular structure governs selection,
    sample complexity and identifiability (§4, §5).
- **3.2 Pullbacks, one at a time.**
  - utility → expected loss L(w); log score → L(w) = h + K(w) with K the relative
    entropy rate of P_θ to p_w (i.i.d.: H(q) + KL(q ‖ p_w)); data → L_n(w), the
    average one-step log loss along the observed path.
  - Fisher–Rao → Fisher information F(w) = JᵀgJ, degenerate where the map loses
    rank (singular models); λ named as the invariant that replaces d there, one
    paragraph. **[V: RLCT volume scaling; invariance under resolution]**
  - observable f on the space of laws → f∘p on W.
- **3.3 Hidden quantities and two senses of alignment.** A quantity τ on S is
  learnable from records iff it is constant on fibres (factors through p).
  - *Predictive alignment:* the predictive converges to P_θ (or to p₀). This is
    all that log loss and held-out evaluation see.
  - *Structural alignment:* a correspondence between the agent's S (its hidden
    component) and the world's Z_t or θ. Stronger than predictive alignment and
    not implied by it in general; well posed only for quantities constant on
    fibres.
  - Heights: the mean difference between components is identifiable up to label
    switching, not at all at the merged point; recording sex makes it a
    conditional observable. **[V: Schilling et al.]**
- **3.4 The principle, with three contrasts.** Pulled-back quantities transform by
  the chain rule under a reparametrisation; quantities defined only on W do not.
  - Fisher information vs Hessian in coordinates (Dinh et al. 2017). **[V]**
  - MLE (equivariant) vs MAP (depends on the prior density in coordinates).
    **[V: Jeffreys; Lehmann–Casella]**
  - the predictive (pushed forward) vs the posterior mean of w (lives on W). Two
    priors that differ only along fibres give the same belief on laws.
- **3.5 Three errors on the diagram.** Bottou–Bousquet: approximation = K(w₀);
  estimation = L_n vs L; optimisation = where the process stops. **[V-new:
  Bottou & Bousquet 2008]**

**For an LLM.** S = W, p given by next-token kernels conditioned on a context
window. Fibres from permutations, rescalings, embedding × unembedding. "A feature"
is a candidate observable on W, meaningful only if constant on fibres.

**Ledger.** S, p, W are chosen; every pulled-back quantity is derived; λ is a cited
property of the chosen map against the world's law.

## §4 Learning as a process on W, seen on the space of laws (~1,800 words)

**Goal:** name what an algorithm returns, and where averaging happens.

- **4.1 Solution concepts as measures on W.** MLE, MAP (point masses); the Bayes
  posterior (Gibbs measure, tempered posterior); the law of an SGD run. Failure
  modes: the mixture MLE does not exist (unbounded likelihood); MAP is not
  invariant. **[V-new: Kiefer & Wolfowitz 1956; Day 1969]**
- **4.2 Push-forward and the predictive.** A measure on W becomes a measure on
  laws; its mean is the predictive, which may leave the image. Averaging happens
  on laws, not on W. Ensembles and posterior predictives are the same
  construction. For sequences the predictive is the one-step conditional (§1.7).
- **4.3 Convergence, stated on laws.** Towards P_θ or p₀; pointwise (SLLN or
  ergodic theorem), uniform (ULLN; consistency to the fibre W₀), fluctuations
  (regular d/2 vs singular λ). Cited, one paragraph. Many paths (exchangeable
  units) vs one path (a single stationary sequence). **[V: free energy;
  Devroye]**
- **4.4 What can be computed.** Expectations under the world's law (samples only);
  argmin over non-convex W; integrals over W. Deterministic vs stochastic
  approximation; one-pass SGD as stochastic approximation of L, multi-epoch of
  L_n. The returned object is a random point whose law depends on the algorithm.
  Named, not analysed (Part 2).
- **4.5 Evaluation.** Held-out loss estimates the pulled-back utility; unbiased only
  under train/test exchangeability of units (back to A1).

**For an LLM.** Pre-training ≈ one-pass stochastic approximation of L over
exchangeable documents; a checkpoint is a sample from an algorithm-dependent law on
W.

**Ledger.** Everything here is approximated; say by what.

## §5 Science and prediction: what the utility asks for (~1,800 words)

**Goal:** fields differ by the action and by which functional of the world the
utility asks about.

| Field | Action | Utility asks about | Alignment needed |
|---|---|---|---|
| Prediction | report a predictive distribution | observables | predictive |
| Estimation | report a parameter | a functional of θ | identifiability of that functional |
| Science | report a claim about θ or Z; design the experiment | a hidden functional, judged by its future use | structural |
| Control | — | — | out of scope (Part 2) |

- **5.1 Prediction.** The predictive is sufficient; the parameter is a means
  (Breiman's two cultures as a difference in object 6 only).
- **5.2 Science.** Identifiability (constant on fibres) is the question;
  experiment design chooses object 2 to make the wanted functional identifiable or
  better estimated (Lindley 1956; Blackwell's order).
- **5.3 Hidden structure is captured anyway.** Models trained only to predict
  represent features of the hidden state Z_t: Othello-GPT represents the board
  (Li et al. 2023; Nanda et al. 2023); chess-playing language models estimate
  latent variables (Karvonen 2024); transformers on HMM data represent
  belief-state geometry (Shai et al. 2024). AlphaZero (McGrath et al. 2022) shows
  the same in the control setting, one sentence. The reason, from §2: optimal
  prediction requires the predictive state, and the predictive state is a
  functional of the world's law that summarises Z_t. So predictive alignment
  forces part of structural alignment. Interpretability is estimation of an
  observable pulled back to W, and §3.3 says when it is well posed.
  **[V: Shai et al.; V-new: the Othello, chess and AlphaZero papers]**

**For an LLM.** Pre-training asks only for predictive alignment; what it learns
about hidden structure is whatever the predictive state of text requires.

**Ledger.** One row per field: given / chosen / assumed / approximated.

## §6 Field guide (~800 words + table)

One row per quantity people compute; each row says which object it belongs to,
whether it is pulled back or pushed forward, and whether it is invariant under
reparametrisation. First-pass rows (each to be checked):

| Term | Object | Lives on | Pulled back / pushed forward | Invariant? |
|---|---|---|---|---|
| likelihood, cross-entropy, perplexity | utility (5) via map (6) | W | pulled back | yes |
| Fisher information, Gauss–Newton | Fisher–Rao on laws | W | pulled back | yes, as a tensor |
| Hessian sharpness | — | W | no | no |
| weight norm, weight decay | — | W | no | no |
| posterior predictive, ensemble | beliefs (4) | laws | pushed forward | yes |
| posterior mean of w | beliefs (4) | W | no | no |
| calibration | observable on the (prediction, outcome) law | laws | — | yes |
| belief state, posterior | (4) | Δ(S) | — | yes, as a belief on laws |
| probe, concept vector | observable (2) | W | candidate pullback | iff constant on fibres |
| identifiability | (6) vs (2) | W | — | — |
| AIC / BIC / WAIC | estimators of pulled-back utility | W | — | WAIC yes; BIC counts d |
| learning coefficient λ | (6) against the world's law | W | pulled-back volume | yes |

## §7 What this makes possible (~700 words)

- Equivalent learning machines: same image and same pullbacks ⇒ the same learner
  in different coordinates (Part 2).
- Only invariants are worth measuring: diagnostics, interpretability and safety
  arguments built on coordinate facts can be changed without changing behaviour.
  One paragraph, labelled motivation.
- The learning process as its own object (Part 2).
- **Control.** When the agent's actions change the world (RL, tools, deployment
  at scale), act–state independence fails. The objects stay; the world's law
  becomes a family indexed by the agent's policy, and the fixed object is the
  world's kernel. One paragraph, pointing to Part 2.
- Open questions, one list.

## Appendix: notation and the ledger (1 page)

The Notation section above, plus the per-section ledger lines collected in one
table.

---

## Reuse of `note.tex`

| `note.tex` | v5 |
|---|---|
| §1 The problem | §1.1 (drop the driving example; add act–state independence) |
| §2 Records | §1.5, one paragraph; rest to companion |
| §3 Sampling symmetry | §1.3, generalised from exchangeability to a symmetry group |
| §4 Regularity, §5 World symmetries | companion |
| §6 Models | §1.6 and §3.1, extended to S and process laws |
| §7 Utility → KL | §2 (Proposition kept) and §3.2 |

## Verification: new rows to add to `verification.md`

| Claim | Source |
|---|---|
| Inference as a decision problem; estimation, testing, prediction as special utilities | Bernardo & Smith 1994 §2, §6.1; Wald 1950; DeGroot 1970; Berger 1985 |
| Act–state independence as part of the decision framework | Savage 1954; B&S ch. 2 |
| Parameters as labels for limiting frequencies; representation theorems; agent-side reading of de Finetti | B&S ch. 4 |
| Table of symmetries and representation theorems | Kallenberg 2005; Diaconis–Freedman 1980 (Markov; recurrence) |
| Experiment as a kernel; comparison of experiments; sufficiency / garbling | Blackwell 1951, 1953; Le Cam 1964; Torgersen 1991 |
| Prequential forecasting | Dawid 1984 |
| Belief MDP; belief as sufficient state | Åström 1965; Kaelbling, Littman & Cassandra 1998 |
| Čencov: Fisher–Rao unique invariant metric | Čencov 1982; Ay et al. 2017 |
| Approximation / estimation / optimisation decomposition | Bottou & Bousquet 2008 |
| Unbounded likelihood of Gaussian mixtures | Kiefer & Wolfowitz 1956; Day 1969 |
| Predictive state representations | Littman, Sutton & Singh 2002 |
| Learned sequence models represent hidden state | Li et al. 2023; Nanda et al. 2023; Karvonen 2024; McGrath et al. 2022 |
| Resolution of singularities; normal-crossing form of K after blow-up | Hironaka 1964; Watanabe 2009 ch. 2–3 |
| Gauge orbit space is stratified; no global gauge fixing | Singer 1978; Kondracki & Rogulski 1983 |
| Superspace singular at metrics with isometries | Ebin 1970; Fischer 1970 |
| Singular symplectic reduction | Sjamaar & Lerman 1991 |
| Singular points of a quotient ↔ larger stabilisers | Luna 1973 |
| Manifold structure on spaces of laws for infinite 𝒳; parametrised measure models | Pistone & Sempi 1995; Ay et al. 2017 |
| Geometry on non-manifold state spaces | Bengtsson & Życzkowski 2006/2017; Lott & Villani 2009; Sturm 2006 |
| History: Veblen–Whitehead, Whitney 1955, Hartigan 1985, Fukumizu & Amari 2000, Watanabe 2001 | dates and content |

## Decisions needed

1. **Sequence example for §5.3.** Othello-GPT (cleanest evidence), the HMM /
   belief-state case (links to the toy-transformer work on this branch), or both?
   *Recommendation: both; Othello for the text, HMM as the worked example since
   its Z_t and predictive state are known exactly.*
2. **Markov exchangeability row in §1.3.** Keep as a remark, or drop?
3. **Design in §1.5 and §5.2.** One paragraph each (as here), or move design to
   Part 2 with control?
4. **Order of writing.** §1 first (it fixes the vocabulary), then §3, then the
   rest.
