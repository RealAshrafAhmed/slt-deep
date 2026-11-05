# Outline: Two Realizations of One Record Process (research note; was v8)

*The mathematical structure of learning problems, Part 1: belief formation.*
(Alternative titles: *World, Instrument, Agent*; *Learning as Stochastic
Realization*; *Microstates, Observables and Learning Machines*.)

Consolidates v7 and the discussion of 2026-10-25 (world as a system; the law of
the records as a property of the one trajectory; stationarity as an information
statement; the utility → representation → energy chain). Supersedes v7 (now in archive/). The
**Provenance** table at the end maps every earlier idea to its place.

Register: a physics paper. Postulates (structures), derived notions, a
"connective tissue" table, examples; literature in Appendix B. Every statement in
the body is a definition, a labelled assumption, a cited theorem (named, not
reproduced), or an example.

## The picture in one paragraph

A world is a dynamical system: microstates, a dynamics, and an instrument that
reads a few observables. The records are the instrument's output along the one
trajectory that occurs. An agent is a second system: an internal state, a map from
states to predictive kernels, and a filter. Both are *realizations* of the same
record process: the world's is the one that generated it, the agent's is the one
that predicts it. What the agent can learn is the invariant part of the record
process, which exists as a property of the trajectory when the trajectory is
typical; how much it must learn is set by the utility, through a rate–distortion
curve whose endpoint is the minimal predictive realization; and what it computes
lives in coordinates that are a parametrisation, not a chart, of its own
realization, so that only quantities pulled back along that map are about the
agent at all.

## Scope

**Belief formation only.** The agent does not change the world. Actions (reports,
designs, control) are Note 2. "Control" is used in the control-theory sense and
only to name what is out of scope; §0 says in one paragraph why this is a scope
choice and not a limit of the Bayesian view.

**Ontology left open, and now less needed.** The note commits to a measure-free
structure and to A1. Under A1 the law of the records is the asymptotic empirical
law of the realised trajectory, which exists without anyone assuming it. The
ensemble reading (a law on histories) and the subjectivist reading (the agent's
coherent law) remain as equivalent descriptions; one footnote.

**Contribution.** Each postulate has prior art (`prior-art.md`, Appendix B). The
contribution is the assembly, and the connective-tissue table (§7) is its
summary. Said once in §0.

Length: ~12k words; five posts.

## What the note claims

1. Belief formation in statistics, ML, scientific modelling and the physics of
   measurement shares one structure: five postulates. Each community has
   formalised a subset; a table shows which.
2. A learning machine is a **stochastic realization** of a kernel; the world is
   another realization of the same kernel. Training a sequence model is
   approximate realization from one history.
3. The parameter space is a **parametrisation** of the agent's realization, not a
   chart. Everything the machine computes is either pulled back along the map
   (invariant by construction) or a coordinate fact. The one principle.
4. What must be learned is not the law of the history but a point on the
   **predictive rate–distortion curve** fixed by the utility and reachable within
   the history; the minimal predictive realization is its endpoint. A
   representation is judged by predictive alignment and efficiency; structural
   alignment with a named hidden observable is a separate, optional claim.

Not claimed: new theorems; unification beyond the tables; a position on where
probability lives.

## Running examples

| Example | World as a system | Symmetry | Agent's realization | Carries |
|---|---|---|---|---|
| Adult heights (NHANES, binned) | people arriving; instrument reads height to a bin | exchangeable | S = W (Gaussian); S = W × {label} (mixture) | the map, fibres, singular point; pullbacks; sex as a hidden observable: hidden, learnable, identifiable |
| Text and an LLM | authors, topics, intent; instrument reads tokens | exchangeable documents, stationary within | trivial: S = context window | kernels not laws; I(T) classes; the main example; a closing paragraph per section |
| HMM-generated sequences; Othello-GPT | a declared generator (M = hidden chain, h = emission) | stationary | learned transformer | the agent recovers the minimal predictive realization, not the generator; efficiency; the two alignments |

The HMM example is where the toy-transformer work on this branch connects.

---

## §0 Introduction (~900 words)

- The same objects under different names: likelihood / cross-entropy; Fisher
  information / Gauss–Newton; posterior predictive / ensemble; microstate /
  hidden state / latent / residual stream / belief state; parameter /
  macrostate / ergodic component; probe / observable; state-space model /
  realization.
- The one-paragraph picture; the four claims; the diagram (§3) in a figure.
- Scope; the Bayesian-control paragraph; the contribution sentence.
- The ledger: given / chosen / assumed / approximated; one line per section.

## §1 Five postulates (~3,200 words)

Each postulate: the structure, one example, the physics counterpart in a
sentence; citations by theorem name only.

**P1 (World: a system).** A state space M of microstates, a dynamics Φ_t on M
(deterministic or stochastic; generators interacting), and an **instrument**
h: M → 𝒳 reading finitely many values. The **history space** Ω is the set of
trajectories; ω* is the one that occurs; the records are X_t = h(Z_t), the
instrument's output along ω*. The microstate Z_t is the maximal observable; it is
never needed. Variability in the records has two sources: the dynamics on M, and
the many-to-one instrument. *Physics:* phase space, flow, coarse-graining.
Realization theory: the world is an **external realization** of the record
process (Lindquist–Picci; Löhr–Ay's "generator").

**P2 (Observables and the agent's filtration).** An observable is a measurable
function on Ω. The instrument's X_t generate the agent's filtration 𝓖_t and its
limit 𝓖_∞. What the agent knows at t is the atom A_t(ω*).
- **Hidden** is a property, not a kind: τ is hidden at t iff not 𝓖_t-measurable.
- **Learnable**: τ is 𝓖_∞-measurable (Doob: E[τ | 𝓖_n] → E[τ | 𝓖_∞]).
- "Different trials" = coordinates of the one history (the only kind that yields
  data) or alternative futures in A_t(ω*) (counterfactuals, inside expectations
  only).
- *Physics:* Mori–Zwanzig (the observed variables alone are not Markov: memory +
  noise); Koopman (observables primary); Takens (the record past can reconstruct
  the state).
**[V: Doob; Ay & Löhr 2015]**

**P3 (Symmetry; assumption A1; the law of the records).** A group G on the index
set under which the record process is invariant. Consequences, in order:
- *Existence.* The **law of the records** is the asymptotic empirical law along
  ω*: limits of time averages of cylinder indicators, hence the empirical kernel.
  These exist when ω* is typical (Birkhoff). The law is a property of the
  trajectory and the dynamics; no ensemble and no agent are required. The
  ensemble over Ω is the equivalent description under the ergodic hypothesis
  (one history stands in for the ensemble) and is kept for conditioning and for
  the subjectivist reading.
- *Identification.* Ergodic decomposition: the record law splits into components;
  ω* identifies its own component θ, the **macrostate**: an observable, constant
  on G-orbits, 𝓖_∞-measurable, hence learnable. The table of symmetries
  (exchangeable; exchangeable units with stationary interiors; stationary).
- *Stationarity, restated.* A function of a stationary history is stationary, so
  non-stationary records mean either a slow component relative to the window, or
  no invariant at any scale. **A drifting generator is a hidden observable:
  promote it to the state and look for the invariant one level up** (a
  time-varying parameter → a hidden chain; regime switching → Markov
  exchangeability). A1, finally: *an invariant exists at some level of this
  hierarchy that the available history can identify.* Testable toward rejection
  (test martingales). Without it, one history identifies nothing.
- *Epistemic reading:* the P_θ are derived from the agent's exchangeable law
  (B&S ch. 4); "objective" = intersubjective agreement.
**[V: Birkhoff / typicality; Gray (asymptotically mean stationary); Kallenberg
2005; Lauritzen; Orbanz & Roy; Al-Najjar & Shmaya; Vovk et al.]**

**P4 (Meeting space: kernels).** What is learned is a kernel from the conditioned
past to the predicted future. The law of the conditioning variable is unmodelled
and enters as a weight; the agent makes no claim about it. 𝒦 = the space of such
kernels; the world's point κ_θ is the empirical kernel of P3; fibres are relative
to the weight. Existence of the kernel is P3's existence; its *stability* (same
function of the past at all times) is A1. **[V: Ionescu-Tulcea; Blackwell; Hutter]**

**P5 (Agent: an internal realization).** A theory is (S, p, π): a state process
with the **splitting property** (past ⊥ future given S_t); a map p: S → 𝒦; a
filter π_t. Facts, one sentence each:
- trivial realization S = past (a transformer's context window);
- minimal realization = the conditional law of the future given the past
  (Knight's prediction process; causal states); minimal by inclusion
  (Lindquist–Picci); unique up to isomorphism; a functional of κ_θ alone;
- **internal vs external**: the agent's S is internal (built from records); the
  world's M is external (P1). A declared latent is a choice between them; a
  generator is not in general the minimal predictor (Löhr–Ay). The claim that
  an agent's S coincides with a named observable of M is optional and labelled;
- non-uniqueness: many realizations, one kernel (a fibre one level above W);
- the filter is itself a realization (the prediction process is Markov; Shai et
  al.'s mixed-state presentation);
- the **realization problem**: construct (S, p) from κ_θ or from data (Akaike;
  subspace identification). *Training a sequence model is approximate
  realization from one history.* (A reading, one paragraph.)
A parameter space W parametrises S; the model map is W → S → 𝒦. Learning is the
agent's dynamics driven by the records: on beliefs (filter; Bayes on W) or on
coordinates (gradient flow).
**[V: Lindquist & Picci; Akaike; Knight; Shalizi & Crutchfield; Löhr & Ay;
Jaeger; Littman–Sutton–Singh; Van Overschee & De Moor]**

**The cross-community table** (which postulates each formalises): classical and
statistical mechanics (P1, P3 ergodic hypothesis); Kolmogorov (P2); ergodic
theory, Kallenberg, Lauritzen (P3); Blackwell–Le Cam (P2, P4); B&S (P3, P5 with
S = Θ); stochastic realization, Lindquist–Picci (P1 external, P5 internal);
computational mechanics (P5 minimal, general); filtering (P5 filter); Hutter /
prequential (P4, P5 trivial); supervised ML (P4, W); SLT (W → 𝒦); Ay–Löhr (P2,
P5); Bayesian mechanics (P5 with the world claim assumed).

**Ledger.** Given: the world's system and the instrument. Chosen: the
realization, the coordinates. Assumed: A1; standard Borel. Derived: the law of
the records, θ, learnability, the filter.

## §2 The meeting space, information, and what must be learned (~2,000 words)

All pointwise in the conditioning variable, averaged under its unmodelled law.

- Convexity; divergence (expected conditional KL; relative entropy rate via the
  chain rule); log-score regret = KL, justified by code length (prequential,
  MDL); Fisher–Rao pointwise, Čencov. **[V]**
- Observables as functionals of a kernel: one-step predictive; entropy rate h;
  predictive information I(T) = I(past of length T; future).
- **Regularity and noise.** I(past; future) is what a representation can carry;
  H(future | past) is noise, paid as the floor of the loss and not modelled.
- **Stationarity as an information statement.** The growth of I(T) classifies
  processes (Bialek–Nemenman–Tishby): saturates at E (finite state suffices);
  ~ (d/2) log T (a finite-dimensional generator is being learned; the regular
  rate); ~ T^α (infinitely many things to learn; "non-stationary at every scale"
  seen from inside); ~ T (noise). Adding observables changes the class: a
  sufficient observable makes the records Markov and collapses I(T) to a
  constant; a missing slow variable pushes it up. "How stationary, given my
  observables and history" = "which class, and where T sits on it".
  **[V: Bialek, Nemenman & Tishby 2001]**
- **What must be learned: the utility bound.** The target is not the law of ω.
  (i) A bounded utility sets a KL tolerance ε on the kernel (Pinsker; Note 2;
  under the log score the tolerance is KL itself). (ii) The agent then needs
  I(S; future) ≥ I(past; future) − δ(ε), not all of it: sufficiency *for a
  utility* is weaker than Bayes sufficiency (Blackwell's garbling order).
  (iii) The **predictive rate–distortion curve** gives the minimal memory
  I(S; past) achieving that; the causal states are its zero-distortion endpoint;
  the utility picks a point on it (Marzen & Crutchfield; Still et al.). (iv)
  Retained non-predictive information is dissipated: the minimal energy cost of
  an agent is set by its non-predictive memory (Still, Sivak, Bell & Crooks).
  *Learning is reaching, at the precision the utility requires, the point on this
  curve that the history length can support.* **[V: Marzen & Crutchfield 2016;
  Still et al. 2012; Blackwell decision sufficiency]**
- KL projection κ₀; relatively finite variance. Minimal realization as a
  functional of the kernel; C_μ ≥ E. **[V]**

**For an LLM.** Next-token kernel across documents; its I(T) class is an
empirical question (scaling laws are statements about it); the context window
bounds the memory, hence where on the curve the model can sit.

## §3 The map and its pullbacks (~2,800 words; the core)

```
   Δ(W) ──p_#──►  Δ(𝒦)  ──mean──►  predictive kernel
     │
     W ──► S ──p──►  𝒦  ◄──h, Φ──  M            (agent ── records ── world)
                     │
   K, L_n, F, f∘p ◄── pull back ── KL, Fisher–Rao, observables f
```

Agent (W, S) and world (M) are two realizations meeting in 𝒦; the empirical
kernel sits there; neither maps into the other.

- **3.1 The map.** W → S → 𝒦; image, fibres (relative to the weight),
  identifiability. Heights ladder.
- **3.1b Charts versus parametrisations.** As in v7: regular case = chart; two
  failures; the right statement; resolution of singularities; history (Gauss →
  Riemann → Whitney → singularity theory; statistics inherited the chart view,
  Watanabe brought the other); what survives (pullbacks for any smooth map; no
  ambient manifold needed); the physics box (gauge orbit spaces, superspace,
  singular reduction, moduli: a singular point has a larger stabiliser; geometry
  on the stratified quotient; observables = invariants); the realization-level
  fibre as the same phenomenon one level up; the two alternatives placed.
- **3.2 Pullbacks.** KL → K; log loss → L = h + K; data → L_n; Fisher–Rao → F =
  JᵀgJ; λ in one paragraph; observable f → f∘p.
- **3.3 Observables of the agent; learnability on both sides.** Legitimate
  observable of the agent iff constant on fibres; learnable hidden observable of
  the world iff 𝓖_∞-measurable. Heights: component mean difference; recording
  sex.
- **3.4 The principle, three contrasts.** Fisher vs Hessian (Dinh; Lau et al.);
  MLE vs MAP; predictive vs posterior mean of w.
- **3.5 Three errors on the diagram** (Bottou–Bousquet), now with a fourth
  reading: approximation error is distance from κ_θ to the image; the utility
  bound of §2 says how much of it matters.

**For an LLM.** S = context window; W = weights; fibres from permutations,
rescalings, embedding × unembedding; "a feature" legitimate iff constant on
fibres.

## §4 Learning as a process on W, seen on 𝒦 (~1,800 words)

- **4.1 Solution concepts as measures on W.** MLE, MAP, posterior, law of an SGD
  run; mixture MLE does not exist; MAP not invariant.
- **4.2 Push-forward and the predictive.** Averaging on kernels, not W.
- **4.3 Convergence on 𝒦.** Pointwise / uniform / fluctuations; many units vs one
  long path (the two senses of "trial", P2).
- **4.4 What can be computed.** Expectations, argmins, integrals; deterministic
  vs stochastic approximation; the returned object is a random point with an
  algorithm-dependent law. One sentence: these limits are the computational side
  of the §2 bound (energy and representation size).
- **4.5 Evaluation.** Held-out loss under exchangeability of units.

## §5 Representations: alignment and efficiency (~2,000 words)

- **5.1 Two senses of alignment.** Predictive (kernel → κ_θ or κ₀; all that log
  loss sees). Structural (S ↔ the minimal predictive realization, world-free; or,
  under the optional claim, S ↔ a named observable of M). Well posed only for
  quantities constant on fibres and learnable.
- **5.2 Efficiency relative to a tolerance.** Sufficient: carries I(past;
  future) − δ. Efficient: nothing more; excess = distinguishing pasts with the
  same future = encoding noise, overfitting at the level of representation.
  Causal states = the zero-distortion endpoint; information bottleneck = the
  variational form; a utility picks the point (§2).
- **5.3 Science and prediction as two requests.** Prediction: predictive
  alignment; science: structural alignment with a *named* observable of M,
  needing learnability and identifiability. Breiman's two cultures as a choice
  of realization.
- **5.4 Predictors recover the minimal realization, not the generator.** Shai et
  al. (belief-state geometry; generator non-uniqueness stated); Othello (the
  board is the predictive state). Predictive alignment forces structural
  alignment *with the minimal realization*; the optional claim is the step from
  there to "the board exists". Interpretability = estimating an observable on W,
  legitimate iff constant on fibres.
- **5.5 The instrumentalist reading.** A model is a code for the past–future
  regularity (Rissanen, Dawid, Grünwald); everything survives, the optional
  claim is dropped.

## §6 Field guide (~800 words + table)

Rows as in v7 (likelihood … λ; hidden state / residual stream; context window;
predictive information; C_μ; state-space model; superposition / collapse) plus:
"non-stationary" (a statement about I(T) and the window, P3/§2); "scaling law"
(an empirical I(T) class); "memory" / "KV cache" (I(S; past), the rate side).

## §7 Connective tissue (~1,200 words + table)

The table the note exists for. One idea per row; its name in each field; the
theorem that makes the row structural (or "analogy").

| Idea | Physics | Probability / ergodic theory | Statistics | Systems theory | ML | Status |
|---|---|---|---|---|---|---|
| complete description | microstate, phase space | ω ∈ Ω, path space | — | state of the generator | — | structural |
| what is read | observable; coarse-graining | measurable function; X_t | random variable; experiment (Blackwell) | output map h | feature / input | structural |
| what is not read | — | not 𝓖_t-measurable | latent | unobserved state | hidden state / latent | definition |
| one trajectory stands for the ensemble | ergodic hypothesis | Birkhoff; ergodic decomposition | de Finetti; extremal families | — | i.i.d. / stationarity assumption | theorem (given A1) |
| the learnable invariant | macrostate; conserved quantity | invariant σ-field; ergodic component | parameter θ | — | "the true distribution" | theorem |
| why records alone are not Markov | Mori–Zwanzig memory + noise | non-Markov projection | — | non-minimal output | "long-range dependence" | theorem |
| state = what splits past from future | — | Markov property | sufficient statistic | splitting subspace; minimal realization | hidden state; belief state | theorem (Lindquist–Picci; Knight) |
| the minimal predictor | — | prediction process | minimal sufficient statistic | predictor space | causal states / ε-machine | theorem |
| generator ≠ minimal predictor | — | — | — | external vs internal realization | HMM vs transformer belief states | theorem (Löhr–Ay) |
| reconstructing state from records | Takens embedding | — | — | realization problem; subspace identification | sequence-model training; context window | theorem / reading |
| redundancy in the description | gauge orbit; stratified quotient | — | non-identifiability | non-uniqueness of realizations | weight symmetries; fibres | structural |
| only invariants are physical | gauge invariance; general covariance | — | reparametrisation invariance; Fisher as a tensor | — | λ vs Hessian sharpness | theorem (Čencov; Watanabe; Lau et al.) |
| singular points have extra symmetry | reducible connections; isotropy | — | singular models | — | merged components; rank-deficient layers | theorem (Luna; Watanabe) |
| energy | Hamiltonian | — | expected log loss | — | training loss | analogy (structural for Gibbs posteriors) |
| how much must be learned | — | — | decision-theoretic sufficiency (Blackwell garbling) | — | utility-bounded error | theorem |
| memory vs predictive power | dissipation bound (Still et al.) | — | — | — | predictive rate–distortion; information bottleneck | theorem |
| regularity vs noise | — | entropy rate | — | — | I(past; future) vs H(future \| past) | definition |
| how learnable, from inside | — | — | parametric (d/2) log T vs nonparametric | — | I(T) growth class; scaling laws | theorem (BNT) |

- One paragraph per row group: measurement (rows 1–6), state and realization
  (7–10), redundancy and invariance (11–13), what must be learned (14–18).
- **Nearest neighbours**, one paragraph each: computational mechanics + Shai et
  al.; Ay–Jost–Lê–Schwachhöfer + Ay–Löhr; Watanabe / Lau et al.; Bayesian
  mechanics (assumes what P5 leaves optional).
- **Note 2**: actions; regret bounds (Pinsker chain made explicit); dual control;
  tuning as a change of utility; equivalence of learning machines; invariants of
  the learning dynamics; "only invariants are worth measuring" (motivation).
- Open questions.

## Appendix A: notation and ledger
## Appendix B: sources, one line per postulate and row (from `prior-art.md`)

---

## Provenance (additions to the v7 table)

| Idea | Origin | v8 location |
|---|---|---|
| World as a system (M, Φ, h); two sources of variability; external realization | discussion 10-25 | P1 |
| Law of records = asymptotic empirical law of ω*; exists by typicality; ensemble as equivalent description | discussion 10-25 | P3 existence; Scope |
| Kernel existence = P3; kernel stability = A1 | discussion 10-25 | P4 |
| Non-stationarity = slow component or no invariant; promote drift to hidden state; A1 as "an invariant at some reachable level" | discussion 10-25 | P3 stationarity |
| I(T) growth classes as the information form of stationarity; role of observables | discussion 10-25; BNT 2001 | §2 |
| Utility → KL tolerance → decision sufficiency → rate–distortion → dissipation | discussion 10-25 | §2 "what must be learned"; §4.4; §5.2 |
| Connective-tissue table | user's aim | §7 |
| Everything in v7's provenance table | v1–v7 | unchanged locations, renumbered where §2 grew |

## Decisions needed

1. **P1 as a system.** State M, Φ, h explicitly (as here), or keep Ω primitive and
   mention the system as the physics reading only? *Recommendation: as here;
   it is what makes "two realizations" literal.*
2. **Existence by typicality.** Make Birkhoff/typicality the primary route to
   the law of the records, with the ensemble secondary? *Recommendation: yes;
   it removes an assumption and matches A1.*
3. **The utility chain in Note 1.** Keep (i)–(iv) in §2 with the regret bound
   deferred to Note 2, or defer the whole chain? *Recommendation: keep; it is
   the answer to "why not learn the law of ω", which Note 1 must give.*
4. **§7 table size.** Eighteen rows as drafted, or cut to the twelve with a
   theorem in the status column? *Recommendation: keep analogies but mark them.*
5. **Order of writing.** §1 → §7 table (it disciplines everything) → §3 → §2 →
   §5 → §4 → §6 → §0.
