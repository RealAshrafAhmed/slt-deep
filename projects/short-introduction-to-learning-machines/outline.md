# Outline: Two Realizations of One Record Process (research note; was v8)

*The mathematical structure of learning problems, Part 1: belief formation.*
(Alternative titles: *World, Instrument, Learner*; *Learning as Stochastic
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

Learning, under passivity, involves two systems: a world with microstates and
dynamics, and a learner with internal states, an update rule and a learning machine.
the learner is driven by the records of an instrument that reads a few observables of an
autonomous world. The learner's performance is scored with a utility over unseen records (P6).

Under the assumptions that the world's dynamics and instrument do not depend on
when one looks (P1) and that there exists a description of the regularities of
the records that grows more slowly than the history (A1), the records have a
law: a kernel from the records seen to those not yet seen. The law splits into information
the seen records carry about the unseen, and noise, whose level can be learned but whose
values cannot be predicted.

The coupling runs one way: the world drives the learner, and the learner's
reports do not affect the records. What *can* be learned is fixed by the
records; what *must* be learned is fixed by the utility. If the utility asks for
full prediction, a learner that meets it becomes a *realization* of the record
process in the sense of stochastic realization theory.

Finally, two learners are equivalent when a map between their internal states
and parameters preserves their predictions and their updates. Quantities that
differ between equivalent learners are gauge, a redundancy of the description;
those that do not are observables of the learner, and laws of learning, if
there are any, are relations among them.

This framing attempts to:

1. give a shared language, and notation, for the objects of learning (world,
   instrument, records, law, learner, utility), in which each community's
   formalism reads as a view of part of one structure, so that a newcomer can
   see the whole structure before specialising and a specialist can see where
   their problem sits in it (claim 1; §1, §7).

2. place very different practices on one structure: fitting a two-parameter
   Gaussian to heights, Bayesian inference, and training an LLM differ in the
   learner (its state, update rule and learning machine) and in the utility,
   not in the structure (running examples; P5; §4.1).

3. separate what is given (world, instrument), chosen (utility, learner), assumed
   (P1, A1, passivity) and derived (the law, what can be learned), and so differentiate between
   quantities about the world from quantities about the learner and about its description (ledger; §3).

4. take a step towards stating learners, their performance and any laws of
   learning invariantly, under a stated equivalence of learners rather than in
   a chosen parametrisation (claim 3; §3).

5. make legible what singular learning theory is after, for readers who know
   the framing but not SLT, or SLT but not the framing: why the parameter space
   is a parametrisation and not a chart, why invariants such as λ and not the
   Hessian, and where SLT's results sit in the larger structure (§3, §4).

6. set up the passive case as the base the active case extends: Note 2 drops
   passivity and keeps the rest (§0; Note 2).

7. contribute to a theoretical side for a largely experimental science: not
   only more mathematics, but a discipline about which quantities are about the
   learner and which are about its description (§0, "An experimental science").

## Scope

**Belief formation under passivity.** The learner's utility scores its reports
against the records (P6), and its reports do not affect the records. §0 says why
prediction comes first even for utilities over actions, and why passivity is the
condition under which passive prediction is exactly the prediction needed.

**Ontology: one stance, one assumption.** The note commits to realism as a
stance (P1: the records are the readout of an autonomous world) and to finite
reach as its one substantive assumption (A1, P3). Neither is proved; the root of
"the law of the records" is philosophy, probed by science. Passive records can
reject specific model classes and can confirm A1 only in the limit; interventions
(repetition, varied conditions, an enlarged instrument) test specific invariance
claims and are Note 2. The structure stays measure-free: the law of the records
is *derived* from A1, as the kernel of an eventually stable short code for the
records (codes and probabilities are interchangeable, Kraft). The ensemble
reading (a law on histories) and the subjectivist reading (the learner's
coherent law) remain as equivalent descriptions; one footnote.

**Contribution.** Each postulate has prior art (`prior-art.md`, Appendix B). The
contribution is the assembly, and the connective-tissue table (§7) is its
summary. Said once in §0.

Length: ~13k words; five posts.

## What the note claims

1. Belief formation in statistics, ML, scientific modelling and the physics of
   measurement shares one structure: six postulates. Each community has
   formalised a subset; a table shows which.
2. The world and the learner are **coupled**, not identified. The world realizes
   the record process externally; the learner becomes a **stochastic
   realization** of it only when its utility asks for full prediction. Training a
   sequence model is approximate realization from one history.
3. Learners are compared by **equivalence**: a map between internal states that
   preserves predictions and updates. What differs between equivalent learners
   is gauge; what does not is an observable of the learner; laws of learning
   can only be relations among observables. The parameter space is the
   parameter-level case: a **parametrisation** of the learner's realization, not
   a chart. The one principle.
4. What must be learned is not the law of the history but a point on the
   **predictive rate–distortion curve** fixed by the utility and reachable within
   the history; the minimal predictive realization is its endpoint. A
   representation is judged by predictive alignment and efficiency; structural
   alignment with a named hidden observable is a separate, optional claim.
5. The postulates double as a **checklist for trusting records**. Data snooping,
   performative prediction, confounding and non-ergodicity are each the failure
   of one named assumption (§1, "How records mislead"); each community knows its
   own row.

Not claimed: new theorems; unification beyond the tables; a position on where
probability lives.

## Running examples

| Example | World as a system | Symmetry | Learner's realization | Carries |
|---|---|---|---|---|
| Adult heights (NHANES, binned) | people arriving; instrument reads height to a bin | exchangeable | S = W (Gaussian); S = W × {label} (mixture) | the map, fibres, singular point; pullbacks; sex as a hidden observable: hidden, learnable, identifiable |
| Text and an LLM | authors, topics, intent; instrument reads tokens | exchangeable documents, stationary within | trivial: S = context window | kernels not laws; I(T) classes; the main example; a closing paragraph per section |
| HMM-generated sequences; Othello-GPT | a declared generator (M = hidden chain, h = emission) | stationary | learned transformer | the learner recovers the minimal predictive realization, not the generator; efficiency; the two alignments |

The HMM example is where the toy-transformer work on this branch connects.

---

## §0 Introduction (~900 words)

- **An experimental science.** Deep learning works like experimental physics:
  rich phenomenology (scaling laws, emergence, grokking) and little theory that
  says which measured quantities are meaningful. What theoretical physics added
  to experimental physics was not only mathematics but a discipline about which
  quantities are physical: invariants, gauge, observables (Klein; the hole
  argument; Dirac; Kretschmann; §3.1b). The analogue here: which quantities are
  about the learner and which are about its description. *Evidence that the
  question is live:* sharpness was proposed as an explanation of
  generalisation, shown to depend on the parametrisation (Dinh et al. 2017),
  and replaced by invariant candidates such as the learning coefficient λ (Lau
  et al.): the field learning the gauge lesson the hard way. Phrase it about the
  field, with this evidence, not about researchers' training.
- **Positioning against the same analogy.** Others use it (Roberts & Yaida,
  *The Principles of Deep Learning Theory*; Bahri et al., "Statistical
  Mechanics of Deep Learning"; the scaling-law literature). They mostly build
  theory in coordinates (expansions in parameters or width); this note's
  contribution to the analogy is the invariance discipline.
- **Two audiences, named.** Newcomers, who should see the whole structure before
  specialising and so stand on firm ground when they narrow down; specialists,
  who should see where their problem sits and what it shares with others. The
  running examples and the "For an LLM" paragraphs carry the first; the
  postulates and tables the second. Saying so keeps a reviewer from reading the
  newcomer-facing passages as padding.
- The same objects under different names: likelihood / cross-entropy; Fisher
  information / Gauss–Newton; posterior predictive / ensemble; microstate /
  hidden state / latent / residual stream / belief state; parameter /
  macrostate / ergodic component; probe / observable; state-space model /
  realization.
- The one-paragraph picture; the five claims; the diagram (§3) in a figure.
- **Why prediction first.** For any utility, including utilities over actions,
  the optimal actor depends on its history only through a predictive state
  (Åström's belief-state reduction; predictive state representations,
  Littman–Sutton–Singh; certainty equivalence in LQG; the internal model
  principle, Francis–Wonham). Prediction is the utility-agnostic interface: one
  predictive model serves every downstream utility (the argument for
  pretraining).
- **The catch.** Acting needs *action-conditional* predictions ("what happens if
  I do a"), and passive prediction supplies "what happens next". They coincide
  only when the records were not confounded with the actions that produced them;
  otherwise a passively trained predictor used to act treats its own actions as
  evidence (Ortega et al. 2021, "delusions"). Predicting everything is also more
  than any one utility needs (value equivalence, Grimm et al. 2020), and when
  actions change what is recorded, learning and acting couple (dual control).
  One paragraph; the treatment is Note 2.
- **Under passivity** the catch disappears: reports do not affect records, so
  passive prediction is the prediction needed. This is why the note assumes
  passivity (P6) rather than merely scoping actions out.
- Scope; the contribution sentence.
- The ledger: given / chosen / assumed / approximated; one line per section.

## §1 Six postulates (~3,900 words)

Each postulate: the structure, one example, the physics counterpart in a
sentence; citations by theorem name only.

**P1 (World: a system).** A state space M of microstates, a dynamics Φ_t on M
(deterministic or stochastic; generators interacting), and an **instrument**
h: M → 𝒳 reading finitely many values. The **history space** Ω is the set of
trajectories; ω* is the one that occurs; the records are X_t = h(Z_t), the
instrument's output along ω*. The microstate Z_t is the maximal observable; it is
never needed. Variability in the records has two sources: the dynamics on M, and
the many-to-one instrument. **Autonomy:** Φ and h do not depend on when one
looks; states change, the rule does not. This is where time invariance lives,
not in the records (which need not be stationary). **Status: a realist stance,
not a theorem.** For a given M it has content: it excludes a law that changes
with the calendar, rewritten from step to step with no state to account for it.
It loses that content only if M may be chosen after the records are seen: every
sequence is the output of *some* autonomous system (M = the sequence space, Φ =
the shift, h = read the first symbol), and a clock coordinate makes any
time-dependent rule autonomous. A1 (P3) is what keeps M honest when it must be
enlarged: "if it fails, enlarge the state" is legitimate only within finite
reach; unlimited, it excuses any failure (Duhem–Quine).
*Physics:* phase space, flow, coarse-graining; time-translation invariance of
the laws (Noether); the experimenter picks the system boundary so the law is
invariant.
Realization theory: the world is an **external realization** of the record
process (Lindquist–Picci; Löhr–Ay's "generator").

**P2 (Observables and the learner's filtration).** An observable is a measurable
function on Ω. The instrument's X_t generate the learner's filtration 𝓖_t and its
limit 𝓖_∞. What the learner knows at t is the atom A_t(ω*).
- **Hidden** is a property, not a kind: τ is hidden at t iff not 𝓖_t-measurable.
- **Learnable**: τ is 𝓖_∞-measurable (Doob: E[τ | 𝓖_n] → E[τ | 𝓖_∞]).
- "Different trials" = coordinates of the one history (the only kind that yields
  data) or alternative futures in A_t(ω*) (counterfactuals, inside expectations
  only).
- *Physics:* Mori–Zwanzig (the observed variables alone are not Markov: memory +
  noise); Koopman (observables primary); Takens (the record past can reconstruct
  the state).
**[V: Doob; Ay & Löhr 2015]**

**P3 (Finite reach A1; the law of the records; symmetry).** The one substantive
assumption, what it buys, and what a symmetry adds.
- **A1 (finite reach).** The part of the world the records depend on can be
  reached from the records at sublinear cost: the **sophistication** (Koppel) of
  the record prefixes grows sublinearly, i.e. the model part of their shortest
  two-part description is eventually o(n), and from some point on the model is
  refined by further records but not rewritten. Equivalently: an autonomous
  realization of the records (P1) whose relevant state has sublinear description
  length. *Sublinear, not bounded:* parametric models already cost ~(d/2) log n
  (§2), and sophistication of infinite sequences diverges in simple cases
  (Grassberger). Informal shorthand: the records are *eventually compressible*
  (the note's term, not the literature's).
- *Status.* A1 has the form "there is an N such that for all n > N …": not
  provable, not refutable with certainty, **verifiable in the limit** (Gold;
  Kelly). Passive records can reject a given model class (test martingales);
  interventions test specific invariance claims (Note 2). Measuring stick: white
  noise satisfies A1 (short model, zero regular part; the level is learnable, the
  values are not); a kernel rewritten at every step does not; a kernel that
  varies unlearnably is absorbed into the noise level. "No law" is always
  relative to a class of models and tests: every sequence has *some* law (a
  point mass), and a randomised forecaster can be calibrated on any sequence
  (Foster–Vohra), so the content lies in the class.
- *The law of the records (derived).* The eventually stable short code *is* the
  kernel from past to future: code lengths are conditional log-probabilities
  (Kraft). Its two parts are the regular part (model) and the noise (data given
  model). No measure is posited; once one is introduced, the kernel exists by
  disintegration and splits by the Doob decomposition. **Learnable along the
  realized path, stationary or not:** a learner whose predictive law dominates
  the truth converges to its predictions (Blackwell–Dubins merging), with total
  error bounded by the truth's description length (Solomonoff; Hutter). This
  learns the kernel along ω*, not the law of ω: what P4 says is learned.
  Characterisation of which (non-stationary) classes admit a consistent
  predictor: Ryabko.
- *Symmetry: what one history identifies.* A group G on the index set (times,
  units, observers) under which the record process is invariant. Then, in
  addition: time averages of cylinder indicators converge along a typical ω*
  (Birkhoff; Gray's asymptotically mean stationary processes), giving the
  **asymptotic empirical law**, the frequency form of the law above; and by
  ergodic decomposition ω* identifies its own component θ, the **macrostate**:
  an observable, constant on G-orbits, 𝓖_∞-measurable, hence learnable. The
  table of symmetries (exchangeable; exchangeable units with stationary
  interiors; stationary). Symmetry is about *identifying θ*, not about being
  learnable: the doubling-block sequence (0s and 1s in blocks of length 1, 2,
  4, …) has no limiting frequencies yet a short description, so A1 holds and it
  is perfectly predictable by a learner whose class contains it.
- *Stationarity, restated.* A function of a stationary history is stationary, so
  non-stationary records mean either a slow component relative to the window, or
  state beyond finite reach. **A drifting generator is a hidden observable:
  promote it to the state and look for the invariant one level up** (a
  time-varying parameter → a hidden chain; regime switching → Markov
  exchangeability). The promotion is legitimate exactly while A1 holds for the
  enlarged state.
- *Epistemic reading:* the P_θ are derived from the learner's exchangeable law
  (B&S ch. 4); "objective" = intersubjective agreement, made precise by merging:
  learners with mutually absolutely continuous priors come to agree.
**[V: Koppel 1987; Vitányi 2006 (sophistication); Vereshchagin–Vitányi
(structure function); Grassberger; Gold 1967; Kelly 1996; Kraft; Blackwell &
Dubins 1962; Solomonoff; Hutter; Ryabko 2010; Foster & Vohra 1998; Shafer & Vovk;
Birkhoff / typicality; Gray (asymptotically mean stationary); Kallenberg 2005;
Lauritzen; Orbanz & Roy; Al-Najjar & Shmaya; Vovk et al.]**

**P4 (Meeting space: kernels).** What is learned is a kernel from the conditioned
past to the predicted future. The law of the conditioning variable is unmodelled
and enters as a weight; the learner makes no claim about it. 𝒦 = the space of such
kernels; the world's point κ_θ is the kernel of P3 (the empirical kernel, under a
symmetry); fibres are relative to the weight. The kernel's *stability* (the same
function of past and state at all times) comes from P1's autonomy; that it can be
*reached* from the records is A1. **[V: Ionescu-Tulcea; Blackwell; Hutter]**

**P5 (Learner: states, update rule, learning machine).** A learner is
(W, S, π, p): parameters W; internal states S; an update rule
π_w: S × 𝒳 → S driven by the records (the **filter**, in the sense of filtering
theory); and the **learning machine** w ↦ p_w, with p_w: S → Δ(𝒳^U) a map from
internal states to distributions over **U, the unseen records the utility
scores** (P6) (Watanabe's p(x | w), extended by a state). U is a set of indices
from P3's index set not yet seen: the next time step, the one h steps on, a
block; another unit (heights); a held-out point; y given x (classification, via
P4's conditioning variable). A report about a function of them (a mean, an
event, a state) is a functional of the distribution on 𝒳^U. The state
summarises the records seen, s = σ_w(records seen), so the learner's kernel is
the composite **κ_w = p_w ∘ σ_w ∈ 𝒦**, from what has been seen to a distribution
on 𝒳^U.
- *The utility sets U.* A one-step learner has U = {n+1}; a direct h-step
  forecaster U = {n+h}; a block forecaster U = {n+1, …, n+h}. Each is a learning
  machine. A learner trained for one U need not define predictions for another,
  and the question of whether its predictions cohere into one law arises only
  when it is used beyond its utility.
- *Full prediction is the most demanding case, not the general one.* U = all
  unseen records, possibly infinitely many (the law exists by Kolmogorov
  extension / Ionescu-Tulcea, but nothing requires it), **and** a strictly
  proper score (e.g. log) on them. Only this pair asks for the whole conditional
  law: a large U with a coarse utility (the sign of a long-run mean) still asks
  for little. It is the zero-distortion end of §2's rate–distortion curve, and
  the only case where the learner can be a realization (the splitting property
  is stated against the whole future). In the sequential case it
  reduces to the one-step kernel, p_w(x_{t+1} | x_{≤t}) = p_w(x_{t+1} | s_t):
  chained, one-step kernels give the law of the whole future (Ionescu-Tulcea),
  by averaging over the unseen intermediate records, i.e. running the kernel on
  the learner's own imagined histories. Under the log score the reduction is
  exact (chain rule for KL); for a utility on a horizon, chaining (iterated) and
  fitting the h-step kernel (direct) differ under misspecification (Marcellino,
  Stock & Watson 2006), and §2's rate–distortion point decides. Time order is a
  special index set, not part of the definition: without one, coherence across
  orderings is a property of the learner (automatic for a Bayesian predictive).
- *Every passive learner has a learning machine,* explicit or not. Under
  passivity its reports are a function of the records seen (with its prior and
  any internal randomness), so S = records seen with p = that function is one,
  trivially; the canonical one identifies histories that lead to the same
  reports now and after any further records (the Nerode quotient), minimal and
  unique up to isomorphism. It may be degenerate: a modeller who posits a
  deterministic law x = f(t; w) has point-mass predictions, and the fitting loss
  supplies the noise model (least squares: Gaussian; absolute error: Laplace).
  A learner that reports only decisions or point forecasts has the part its
  utility reads. The claim is about the machine up to equivalence: that a
  *particular* S (a layer's activations, a brain's variables) is the state is a
  gauge choice (§3). That this representation is the useful one for comparing
  learners is argued, not assumed (equivalence, observables).
- *S is not a latent variable.* S is whatever the prediction depends on from the
  records seen: always present (all of them qualify), possibly trivial. A latent
  Z with p_w(x) = ∫ p_w(x, z) dz is an optional way to parametrise p_w (a
  declared latent, below); most problems, and an LLM, have none.
- *Two timescales.* w changes slowly, by learning (fitting; Bayes on W, where
  the slow state is the posterior and the prediction the mixture); s changes
  fast, by filtering. S is trivial exactly when w alone splits seen from unseen
  records (i.i.d. given w): then κ_w is the density p(x | w), Watanabe's setting.
- *Cases.* Gaussian fit: S trivial given w; as a running learner, the slow state
  is the sufficient statistics. LLM: w = weights, S = context window, σ_w =
  append and truncate. HMM / RNN: S = belief or hidden vector, π_w = the filter
  or the recurrence.
The learner is an **internal realization** of the record process when S has the
**splitting property** (past ⊥ future given S_t) and κ_w = κ_θ: a property it may
reach, not an assumption. Facts, one sentence each:
- trivial realization S = past (a transformer's context window);
- minimal realization = the conditional law of the future given the past
  (Knight's prediction process; causal states); minimal by inclusion
  (Lindquist–Picci); unique up to isomorphism; a functional of κ_θ alone;
- **internal vs external**: the learner's S is internal (built from records); the
  world's M is external (P1). A declared latent is a choice between them; a
  generator is not in general the minimal predictor (Löhr–Ay). The claim that
  a learner's S coincides with a named observable of M is optional and labelled;
- non-uniqueness: many realizations, one kernel (a fibre one level above W);
- **equivalence**: a morphism of learners is a map φ: S → S′ with
  p′_{w′}∘φ = p_w and φ(π_w(s, y)) = π′_{w′}(φ(s), y) for every record y. A one-way,
  non-invertible φ is a *reduction* (S′ coarse-grains S); equivalence is a
  two-way map or a common reduction (**bisimulation**: Park; Milner; the
  Myhill–Nerode quotient is the minimal one). Each class contains the minimal
  realization, unique up to isomorphism (Kalman's state-space isomorphism
  theorem; Lindquist–Picci). Quantities that vary within a class are **gauge**;
  those constant on classes are **observables of the learner** (§3.3); laws of
  learning are relations among observables (e.g. Bayes generalization error
  ≈ λ/n, C_μ ≥ E, I(S; future) ≤ I(past; future));
- **the choice of equivalence is the content** (Kretschmann's lesson, §3.1b):
  permuting hidden units preserves predictions and commutes with SGD, so it is
  an equivalence of SGD learners; ReLU rescaling preserves predictions but not
  gradients, so the rescaled network is a different SGD learner that agrees
  today. For Bayesian learners every reparametrisation (prior transformed with
  it) is an equivalence, and the two notions coincide;
- the filter is itself a realization (the prediction process is Markov; Shai et
  al.'s mixed-state presentation);
- the **realization problem**: construct (S, π_w, p_w) from κ_θ or from data (Akaike;
  subspace identification). *Training a sequence model is approximate
  realization from one history.* (A reading, one paragraph.)
The model map is w ↦ κ_w, W → 𝒦 (Watanabe's map when S is trivial). Learning is the
learner's dynamics driven by the records: on beliefs (filter; Bayes on W) or on
coordinates (gradient flow).
**[V: Lindquist & Picci; Akaike; Knight; Shalizi & Crutchfield; Löhr & Ay;
Jaeger; Littman–Sutton–Singh; Van Overschee & De Moor; Kalman 1963; Park 1981;
Milner 1989]**

**P6 (Utility; passivity).** The learner issues a report a ∈ A (a prediction, a
point forecast, a label, an answer) and the records that arrive score it:
u: A × 𝒳^U → ℝ, where U is the set of unseen records the utility reads (P5).
**Passivity:** reports do not affect the records. Utility is on
(report, record) pairs, not on internal states (those are coordinates, P5 and
§3; they enter on the cost side, as rate in §2) and not on records alone
(preferences over outcomes are inert when the learner cannot change them; they
become live in Note 2). **What is scored:** U may be the next record, the one h
steps on, a block, or (through a function of them) a mean or an event; U is
what the learner must predict (P5), and records outside it are never evaluated.
Facts, one sentence each:
- under passivity the best report for *any* u depends on the past only through
  the learner's predictive kernel: a* = argmax_a E_κ[u(a, X)] (the Bayes act;
  Blackwell sufficiency);
- **proper scoring rules** are the case A = predictive distributions where the
  best report is the honest one; the log score is the local one and asks for the
  whole kernel (Gneiting–Raftery; Dawid);
- coarser utilities need only a compression of the kernel: the rate–distortion
  curve of §2;
- examples: next-token log loss (LLM pretraining); a forecast scored by later
  measurements under a fixed instrument (science); an answer graded against a
  reference record (task solving, while the grade is a fixed function of answer
  and reference).
**[V: Wald; Savage; Blackwell; Gneiting & Raftery 2007; Dawid]**

**The cross-community table** (which postulates each formalises): classical and
statistical mechanics (P1, P3 ergodic hypothesis); Kolmogorov (P2); ergodic
theory, Kallenberg, Lauritzen (P3); Blackwell–Le Cam (P2, P4); B&S (P3, P5 with
S = Θ); stochastic realization, Lindquist–Picci (P1 external, P5 internal);
computational mechanics (P5 minimal, general); filtering (P5 filter); Hutter /
prequential (P4, P5 trivial); supervised ML (P4, W); SLT (W → 𝒦); Ay–Löhr (P2,
P5); Bayesian mechanics (P5 with the world claim assumed); statistical decision
theory, Wald–Savage, proper scoring rules (P6).

**Ledger.** Given: the world's system and the instrument. Chosen: the utility,
the realization, the coordinates. Assumed as a stance: realism (P1: an
autonomous world, read by a fixed instrument). Assumed, substantive: finite
reach (A1, verifiable in the limit); passivity; standard Borel once a measure is
introduced. Probed by intervention: specific invariances (Note 2). Derived: the
law of the records, learnability along the path; under a symmetry, θ and the
empirical law; the filter; whether the learner is a realization.

**How records mislead (~700 words).** Each way records mislead is the failure of
one named assumption, so the postulates double as a checklist (claim 5). One
paragraph per row; the closed-loop treatment of rows 2–3 is Note 2.

| Assumption | Failure | Known as |
|---|---|---|
| Instrument is fixed (P1) | what is recorded or analysed is chosen after seeing the records | data snooping; garden of forking paths; adaptive data analysis (Dwork et al. 2015); selection bias |
| Passivity (P6) | the learner's reports change the records | performative prediction (Perdomo et al. 2020); self-fulfilling forecasts |
| Records used in the regime that generated them | other agents' actions in the records are confounded with microstates the instrument does not read, and the learner's use replaces those actions | confounding; delusions in sequence models (Ortega et al. 2021); shift under intervention |
| Finite reach (A1); symmetry (P3) | the history does not reach the state the records depend on, or is not representative of its component | regime change; non-ergodicity; survivorship |

- *Row 1.* Looking at the data and then choosing what to test makes the
  instrument a function of the records: the loop closes although nobody acted
  on the world.
- *Row 3.* The world contains other agents; their actions appear in the records
  and were caused partly by unread microstates (an expert's knowledge, a
  clinician's judgement, an author's intent). A passive predictor learns the law
  of the records correctly, and that law includes those agents' policies. When
  the learner's own outputs take their place, they are no longer coupled to the
  unread microstates, and treating them as the same evidence is the error. For an
  LLM: text written by someone who knew is evidence about what they knew; text
  the model generates is not.
- *Invariance, made precise.* The "invariant part of the record process" is
  invariant *along the trajectory* (P3). Invariance across interventions, which
  identifies causal structure (invariant causal prediction, Peters–Bühlmann–
  Meinshausen 2016), is Note 2.
**[V: Dwork et al. 2015; Perdomo et al. 2020; Ortega et al. 2021; Peters et al.
2016]**

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
  under the log score the tolerance is KL itself). (ii) The learner then needs
  I(S; future) ≥ I(past; future) − δ(ε), not all of it: sufficiency *for a
  utility* is weaker than Bayes sufficiency (Blackwell's garbling order).
  (iii) The **predictive rate–distortion curve** gives the minimal memory
  I(S; past) achieving that; the causal states are its zero-distortion endpoint;
  the utility picks a point on it (Marzen & Crutchfield; Still et al.). (iv)
  Retained non-predictive information is dissipated: the minimal energy cost of
  a learner is set by its non-predictive memory (Still, Sivak, Bell & Crooks).
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
   Δ(W) ──κ_#──►  Δ(𝒦)  ──mean──►  predictive kernel
     │
     W ────κ────►  𝒦  ◄──h, Φ──  M              (learner ── records ── world)
                   │
   K, L_n, F, f∘κ ◄── pull back ── KL, Fisher–Rao, observables f

   κ_w = p_w ∘ σ_w :  seen ──σ_w──►  S  ──p_w──►  Δ(𝒳^U)
```

Learner (W, S) and world (M) meet in 𝒦: the world's system outputs the records,
the learning machine outputs predictions; the empirical kernel sits there;
neither maps into the other.

- **3.1 The map.** W → 𝒦, w ↦ κ_w = p_w ∘ σ_w (Watanabe's p(x | w) when S is
  trivial); image, fibres (relative to the weight),
  identifiability. Heights ladder.
- **3.1b Charts versus parametrisations.** As in v7: regular case = chart; two
  failures; the right statement; resolution of singularities; history (Gauss →
  Riemann → Whitney → singularity theory; statistics inherited the chart view,
  Watanabe brought the other); what survives (pullbacks for any smooth map; no
  ambient manifold needed); the physics box (gauge orbit spaces, superspace,
  singular reduction, moduli: a singular point has a larger stabiliser; geometry
  on the stratified quotient; observables = invariants). *Why invariants, as
  physics settled it:* identity (Klein's Erlangen programme: properties of a
  geometry are invariants of its group); evidence (Einstein's hole argument and
  point-coincidence resolution; Dirac: gauge-dependent quantities are not fixed
  by the dynamics; Elitzur: gauge is redundancy, not symmetry). In learning: the
  records constrain only the predictions, so what varies within a class is fixed
  by initialisation and optimiser, not by the records. Kretschmann (1917):
  covariance alone is empty; the content is the choice of equivalence. Gribov:
  no global gauge fixing, a singular orbit space; the counterpart of
  "parametrisation, not chart"; the realization-level
  fibre as the same phenomenon one level up; the two alternatives placed.
- **3.2 Pullbacks.** KL → K; log loss → L = h + K; data → L_n; Fisher–Rao → F =
  JᵀgJ; λ in one paragraph; observable f → f∘κ.
- **3.3 Observables of the learner; learnability on both sides.** Legitimate
  observable of the learner iff constant on equivalence classes (P5); fibres of
  W → 𝒦 are the parameter-level case for prediction-equivalence; learnable hidden observable of
  the world iff 𝓖_∞-measurable. Heights: component mean difference; recording
  sex.
- **3.4 The principle, three contrasts.** Fisher vs Hessian (Dinh; Lau et al.;
  sharpness is gauge under prediction-equivalence, possibly an observable under
  SGD-equivalence);
  MLE vs MAP; predictive vs posterior mean of w.
- **3.5 Three errors on the diagram** (Bottou–Bousquet), now with a fourth
  reading: approximation error is distance from κ_θ to the image; the utility
  bound of §2 says how much of it matters.

**For an LLM.** S = context window; W = weights; fibres from permutations,
rescalings, embedding × unembedding (permutations are SGD-equivalences;
rescalings are not); "a feature" legitimate iff constant on the equivalence
classes in use.

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
| one trajectory stands for the ensemble | ergodic hypothesis | Birkhoff; ergodic decomposition | de Finetti; extremal families | — | i.i.d. / stationarity assumption | theorem (given A1 and a symmetry) |
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
- **Note 2**: drop passivity and the fixed instrument; the separation question
  (when belief formation can be solved before acting, and when it cannot);
  action-conditional prediction; invariance under intervention; rows 1–3 of "How
  records mislead" in closed loop; actions; regret bounds (Pinsker chain made explicit); dual control;
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
| Law of records = asymptotic empirical law of ω*; exists by typicality; ensemble as equivalent description | discussion 10-25 | P3 symmetry (frequency form); Scope — revised 11-07 |
| Kernel existence = P3; kernel stability = A1 | discussion 10-25 | P4 — revised 11-07 (stability from P1, reach from A1) |
| Non-stationarity = slow component or no invariant; promote drift to hidden state; A1 as "an invariant at some reachable level" | discussion 10-25 | P3 stationarity — A1 revised 11-07 to finite reach |
| I(T) growth classes as the information form of stationarity; role of observables | discussion 10-25; BNT 2001 | §2 |
| Utility → KL tolerance → decision sufficiency → rate–distortion → dissipation | discussion 10-25 | §2 "what must be learned"; §4.4; §5.2 |
| Connective-tissue table | user's aim | §7 |
| Learner (not agent); instrument as the world's output map; coupling primitive, realization derived; learning machine = map from states to predictions; filter = update rule | discussion 11-05 | picture; P5; claim 2 |
| Utility on (report, record); passivity as an assumption; why prediction first; the catch | discussion 11-05 | P6; §0 |
| How records mislead: one failure per assumption; confounding via other agents | discussion 11-05 | §1; claim 5 |
| Equivalence of learners (morphism preserving predictions and updates; bisimulation); gauge vs observables; laws as relations among observables; why invariants (Klein, hole argument, Dirac, Kretschmann, Gribov) | discussion 11-05 | picture; claim 3; P5; §3.1b; §3.3 |
| Realism as a stance (autonomous Φ, fixed h), vacuous alone (shift construction); A1 as finite reach (sublinear sophistication), verifiable in the limit; law of records derived from an eventually stable code (Kraft); learnability without stationarity (Blackwell–Dubins, Solomonoff); symmetry demoted to identification of θ; "enlarge the state" only within finite reach | discussion 11-06/07 | picture; Scope; P1; P3; P4; ledger |
| Learning machine = w ↦ p_w, p_w: S → Δ(𝒳^U), U = the unseen records the utility scores (not all unseen records; full prediction = U all + a strictly proper score, the most demanding case); one-step kernel = sequential reduction, exact under log score); learner's kernel κ_w = p_w ∘ σ_w; S (always present, possibly trivial) ≠ latent Z (optional); S trivial iff w splits seen from unseen records (Watanabe's case); model map W × S → Δ(𝒳^U), not W → S → 𝒦 | discussion 11-07 | picture; P5; §3 diagram; §3.1 |
| Picture paragraph rewritten: "Learning, under passivity, …"; order world → records → law → learner → realization; can vs must be learned; every passive learner has a learning machine (trivial S = records seen; Nerode quotient; degenerate deterministic case; claims about a particular S are gauge) | discussion 11-07 | picture; P5 |
| Picture paragraph as a narrative arc (details to the body); A1 phrased on the regularities, not the records; "What the framing is for": six motivations mapped to claims (shared language and notation; Gaussian / Bayes / LLM on one structure; given–chosen–assumed–derived; invariance under equivalence; making SLT legible; passive as the base for active). Claim 5 kept as a by-product, not a motivation | discussion 11-07 | picture |
| Deep learning as an experimental science needing a theoretical side; physics' addition was the invariance discipline, not only mathematics; sharpness (Dinh et al. 2017) → λ (Lau et al.) as evidence; positioning against Roberts & Yaida, Bahri et al.; two audiences (newcomers, specialists) | discussion 11-07 | §0; motivations 1, 7 |
| Everything in v7's provenance table | v1–v7 | unchanged locations, renumbered where §2 grew |

## Decisions needed

1. **P1 as a system.** State M, Φ, h explicitly (as here), or keep Ω primitive and
   mention the system as the physics reading only? *Recommendation: as here;
   it makes the world's external realization literal.*
2. **Existence by typicality.** *Superseded (11-07):* the law of the records
   now rests on finite reach (A1), with Birkhoff/typicality the frequency form
   under a symmetry. Open: whether §1 states A1 via sophistication (algorithmic)
   or via a dominating predictive law (Blackwell–Dubins), with the other as a
   remark. *Recommendation: sophistication in the text, merging as the theorem
   that uses it.*
3. **The utility chain in Note 1.** Keep (i)–(iv) in §2 with the regret bound
   deferred to Note 2, or defer the whole chain? *Recommendation: keep; it is
   the answer to "why not learn the law of ω", which Note 1 must give.*
4. **§7 table size.** Eighteen rows as drafted, or cut to the twelve with a
   theorem in the status column? *Recommendation: keep analogies but mark them.*
5. **Title.** "Two Realizations of One Record Process" states as a premise what
   the note now derives. *Recommendation: World, Instrument, Learner* or
   *Learning as Stochastic Realization*.
6. **Separation note.** Make "when does learning separate from acting?" a short
   bridge note, or the opening of Note 2? *Recommendation: opening of Note 2; §0
   carries the one-paragraph version.*
7. **Order of writing.** §1 → §7 table (it disciplines everything) → §3 → §2 →
   §5 → §4 → §6 → §0.
