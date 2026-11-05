# Outline: What a Learning Machine Is Made Of (blog post; was learning-framework v9)

*Part 1: learning without changing the world.* (Alternative titles: *World, Instrument, Machine*;
*Two Machines, One Record*.)

The same picture as the research note (`../short-introduction-to-learning-machines/outline.md`), written for new researchers in applied ML with a bachelor's degree in
computer science, mathematics or physics. Terms are introduced by the rules in
`glossary.md`: idea first, name second, one new term per paragraph, theorem names
in parentheses, other fields' vocabulary collected in §6–§7. **Bold** marks the
point where a term is introduced; it is used freely afterwards.

## The note in one paragraph

The world is a machine we cannot see inside. An instrument reads a few numbers off
it, and those readings are our data. A learning machine is another, with
its own memory and its own assumptions about the world's machine, built to do a few
tasks well. As the world evolves, the learning machine predicts what comes next and gives the answer that scores best under each task's **utility**. This note names the parts of
that picture. Once they are named, a language model and the Gaussian fitted in a
second-year statistics course turn out to have the same parts: a world, an
instrument, one long record, an unknown probability law, a utility, a
statistical model, and a learning process. What differs is
the geometry of the model, how much of the world stays hidden, and the
challenges that come with the sheer dimension of every part. The
note shows the two side by side at every step. Its aim is to see the forest
rather than the trees: one structure in which these parts, and the questions
asked about them, each have a place. Modern mathematics did this for geometry by
studying a space apart from the coordinates written on it and asking which
quantities survive a change of coordinates; the note takes a first step of the
same kind for learning, and tests whether the structure earns its keep by
showing how statistics, physics, systems theory and machine learning have each
treated the same objects under different names. Along the way it says what one
stream of readings can and cannot teach, and which of the quantities computed
during training are about the model rather than about how its numbers happen to
be written down.

## Who it is for, and what it covers

- **Reader:** comfortable with probability (random variables, conditional
  probability, the law of large numbers), linear algebra and basic machine
  learning, and with the language of undergraduate classical and statistical
  mechanics: microstate, phase space, dynamics, observable, ensemble, macrostate,
  coarse-graining, the ergodic hypothesis. These physics terms are the note's
  working vocabulary: each is defined in one sentence when first used, so a
  reader without physics can follow, and then used freely. No measure theory or
  information geometry assumed.
- **Covers:** deciding without changing the world, with prediction as the step
  every such decision needs. The
  learner reads, and it may also decide: which parameter value is best, whether
  to reject a null hypothesis, which forecast to issue. A utility scores each
  decision, so the model can be tuned to a goal other than prediction, but the
  decision cannot change what the world does next. This covers statistics
  (estimation and testing as decisions), scientific modelling, and a language
  model without tools.
- **Leaves to Note 2:** learners whose actions change the world (agents, tools,
  control), and choosing what to measure.
- **Not claimed:** new theorems. Each piece is known in some field; the
  contribution is putting them in one picture, and the table in §7 that shows
  where each piece comes from.
- **Purpose:** see the next section.
- **Stance, for two audiences.** To the reader coming from machine learning:
  statistics is not tests and p-values; it is the history space, the law of
  the records, predictive distributions, the posterior, the gap (KL) and
  predictive information, and these are the objects you already compute daily
  under other names. To the reader coming from statistics: the organising idea
  of the second half is *invariance*, studying an object up to change of
  description, which is ordinary in modern mathematics but not yet part of the
  statistician's language. The rigour this asks for is not a nuisance; it is
  what lets you say which computed quantities are about the learning machine
  and which are about how its numbers happen to be written down.
- The rigorous synthesis for researchers is the research note and a later
  paper (after Note 2); this note takes from them only what a new researcher
  needs.

Length: about 12,000 words, five posts.

## Purpose

The note has three parts, and the purpose is stated in three paragraphs so that
nothing in the body is written without a reason in this section. A Gaussian fit
and a language model trained on many tasks are two examples of one learning
problem, not two frameworks; the three paragraphs are what a reader needs in
order to see that and to know, at every part, what can undermine it, what limits
it puts on any learning machine, and what the remedy is.

**1. The picture: what the world gives, and what one record can teach (§1–§2).**
The world is a machine nobody sees inside; an instrument coarse-grains its state
to a few numbers, and the readings along the one history that occurs are the
data. Readings vary for two reasons, the world moves and many states give the
same reading, and the second reason is why hidden quantities exist at all. Some
hidden quantities are learnable from a long enough record and some never are;
which is which is decided by what the instrument records, not by how much data
there is. One history can stand in for many only under a symmetry, and the
symmetry is what makes the law of the records exist, as a property of the one
history (typicality), with the ensemble as an equivalent description. The
strongest symmetry, exchangeability, gives the simplest laws: a law on a single
reading, the empirical distribution as the whole of the regularity and the order
as pure noise, a finite summary that settles. That is why the statistician's
tools are so clean, and why they are so heavily used; it is a special case.
Stationarity alone gives a memory that never settles, which is stochastic
filtering, and a language model is that case at scale. What the record contains
splits into regularity, the part the past tells about the future, and noise, the
part no predictor can remove and that is paid as the floor of the loss;
regularity is relative to a split and equals compressibility. How the regularity
grows with the length of the past is the information form of "how stationary":
it levels off, grows like a log, grows like a power, or grows linearly, and
recording an extra observable moves a process between these classes. When the
symmetry seems to fail, the usual cause is a hidden quantity that changes slowly
compared with the window; the remedy is to promote it to the state and look for
the symmetry one level up, and the limit is that if no level within reach of the
data has a symmetry, nothing is learnable. Finally, a use decides how much must
be learned: a bounded utility sets a tolerance on the gap (KL), which is why the
gap is the loss when the use is left open and why a fixed task may need far less
than the whole law; the tolerance sets how much regularity the memory must hold,
which sets the least memory on a memory-versus-accuracy curve whose endpoint is
the smallest perfect memory, and holding more than that costs energy.

**2. The learner: a second machine, and what is chosen in building it (§1.7,
§2, §3.1–3.2, §5).** A learner is a machine with memory: a state, an update rule,
a prediction rule (a realization). The simplest memory is the whole past (the
context window); the smallest perfect memory is unique and depends only on the
world's next-step rule (causal states); the machine that produced the data and
the smallest machine that predicts it are usually different, so "the model's
memory is a real thing in the world" is a further claim. The learner's family of
rules is written with weights, and the map from weights to rules is where every
choice lands. Its image is a subset of the space of next-step rules, and it is
not in general a manifold. When the map is one-to-one and of full rank the image
is an embedded submanifold, the weights are honest coordinates, the pulled-back
Fisher information is a Riemannian metric, and the whole of information geometry
is recovered. When only one-to-one fails, the fibres are symmetries of the
description and the quotient by them restores coordinates. When the rank drops,
the image has a sharp point, the pulled-back Fisher information is degenerate,
and no metric exists there; the weights are then a description, not a chart, and
coordinates return only locally, after a blow-up (resolution of singularities).
What survives in every case is the pullback of functions and quadratic forms,
which needs no manifold structure on the space of rules at all. Whether the
family is singular depends on the family *and on what is recorded*: recording
the hidden label removes the singularity from the mixture. The two machines are
different objects with different internal spaces and meet only at the next-step
rule; when the family contains the world's rule the model is realizable, the
statistician's "well-specified", and that is where the gap can reach zero, the
macrostate is a point of the family, the memory needed is known in advance, and
nearly every classical theorem applies. It is the special case; in the general
case the target is a projection and the learner's memory need not resemble the
world's. A memory is judged by two separate standards: predictive alignment,
which is all the loss can see, and structural alignment with the smallest
perfect memory, or, as a further labelled claim, with a named hidden quantity;
it is sufficient if it holds the regularity the use requires and efficient if it
holds nothing else, and trained predictors are found to approach the efficient
one.

**3. Training: how the machine is found, and which of its numbers mean
anything (§3.3–3.6, §4).** The score, the gap and the Fisher information are
defined on rules and read on weights by pullback; the training loss is the floor
plus the gap, so the gap is exactly the regularity not captured. What training
returns is a distribution over weights: the maximum-likelihood point, which for a
mixture does not exist; the posterior; the end point of a stochastic-gradient
run, whose law depends on the optimiser. What the user receives is the
push-forward of that distribution to rules, a predictive distribution, and
averaging rules is not averaging weights. Convergence is a statement on rules,
in three layers: pointwise, uniform, fluctuations; the regular and singular
cases differ only in the third, where the usual parameter count is replaced by
an invariant of the map, the learning coefficient, read off after the blow-up.
The Bayesian view is the one in which the regular and the singular case are the
same statement. Averages under the world, minimisers over a non-convex
landscape, and integrals over weights are all out of exact reach, so every
returned object is a random point whose law is set by the algorithm; these
computational limits are the other side of the memory-versus-accuracy curve.
Three errors remain (the family cannot express the truth, the data are finite,
the optimiser did not finish), and the utility of part 1 says how much of the
first one matters. One test governs all of it: a quantity is about the machine
only if it survives rewriting the weights; Fisher information, maximum
likelihood and averaged predictions do, Hessian sharpness, a prior-weighted
maximum and averaged weights do not, and a probe finds something only if it is
constant on fibres on the learner's side and learnable on the world's side.

**Ingredient checklist.** Every item below must appear in the body in the
section named; a draft that drops one is incomplete.

| Ingredient | Part | Section |
|---|---|---|
| world, instrument, coarse-graining; two sources of variability | 1 | §1.1–1.2 |
| histories; hidden vs learnable (Doob); what is recorded decides learnability | 1 | §1.3–1.4 |
| Mori–Zwanzig: readings alone need memory and noise | 1 | §1.4 aside |
| symmetry → law exists by typicality; ensemble as equivalent description | 1 | §1.5 |
| exchangeable = strongest symmetry → simplest laws; type class; order is noise | 1 | §1.5 |
| stationary → filter; i.i.d. is the filter whose state converges | 1 | §1.5, §1.7 |
| regularity vs noise; floor of the loss; regularity = compressibility, relative to a split | 1 | §2 |
| growth of predictive information as "how stationary"; observables move the class | 1 | §2 |
| slow hidden variable; promote and level up; no level ⇒ nothing learnable | 1 | §1.5, §2 |
| utility → tolerance on the gap; why KL when the use is open; fixed task needs less | 1 | §2 |
| memory-versus-accuracy curve; smallest perfect memory as endpoint; energy cost | 1 | §2, §5.2 |
| machine with memory; trivial vs minimal; producer ≠ predictor | 2 | §1.7 |
| map weights → rules; image is a subset, not always a manifold; fibres | 2 | §3.1 |
| regular case: chart, embedded image, Fisher–Rao metric, information geometry recovered | 2 | §3.2 |
| quotient by fibres restores coordinates when only injectivity fails | 2 | §3.2 |
| rank drop: sharp point, degenerate Fisher, no metric; blow-up restores coordinates locally | 2 | §3.2 |
| pullback works for any smooth map; no ambient manifold needed | 2 | §3.2 |
| singularity depends on the family and on what is recorded | 2 | §3.1, §3.5 |
| realizability = well-specified; what it buys; why statisticians insist; it is the special case | 2 | §2 |
| predictive vs structural alignment; sufficient vs efficient memory | 2 | §5.1–5.2 |
| pullback of score, gap, Fisher; loss = floor + gap = regularity not captured | 3 | §3.3 |
| solution concepts as distributions on weights; mixture MLE does not exist; MAP not invariant | 3 | §4.1 |
| push-forward → predictive distribution; average rules not weights | 3 | §4.2 |
| convergence in three layers; regular vs singular differ only in fluctuations; learning coefficient | 3 | §4.3 |
| Bayesian view: regular and singular as one statement | 3 | §4.3 |
| what can be computed; stochastic approximation; returned object is random | 3 | §4.4 |
| held-out loss needs exchangeability of units | 3 | §4.5 |
| three errors; utility says how much approximation error matters | 3 | §3.6 |
| the one test: survives reparametrisation; three contrasts; what a probe can find | 3 | §3.4–3.5 |

## What the note says

1. **One framework, two examples.** A language model and a classical statistical
   model are the same kind of object: a world, an instrument, one long record,
   a symmetry, a next-step rule, a predictor with memory, a family of rules, a
   score. Statistics, machine learning, scientific modelling and the physics of
   measurement each have a precise language for some of these parts and are
   vague about the others; naming all of them once lets a student keep the
   whole picture in view while working on any special case. (§1, the table)
2. **Simple laws come from strong symmetry.** Exchangeability gives a law on a
   single reading, a finite sufficient statistic, and a memory that settles;
   that is why the classical tools are so clean, and why statisticians exploit
   it so heavily. Stationarity alone gives a memory that never settles: that is
   stochastic filtering, and a language model is the same case at scale. The
   i.i.d. model is not a separate theory; it is the filter whose state
   converges. (§1.5, §1.7)
3. **Each part can fail, and each failure has a remedy in the same parts.** No
   symmetry at any reachable level; an observable that is not learnable;
   readings that carry too little predictive information; a family that cannot
   express the world's rule; an optimiser that did not finish. The remedies:
   look for the symmetry one level up; record the slow variable; add memory;
   enlarge the family, or accept the closest rule as the target; average
   predictions, not weights. (§1.5, §2, §3.6, §4, §5.6)
4. **The two machines meet at the next-step rule, and realizability is where
   they coincide.** A learner is a **machine with memory** that reproduces a
   next-step rule (realization theory). The world, seen through the instrument,
   is another such machine; the producer and the smallest predictor are usually
   different. Training a sequence model is building the second machine from the
   output of the first. Realizability, the statistician's "well-specified
   model", is the case where the learner's family contains the world's rule:
   the gap can reach zero, the macrostate is the parameter, and the memory
   needed is known in advance. That is why statisticians care so much about it,
   and it is the special case, not the general one. (§1.7, §2, §5)
5. **Training lives on the map from weights to rules, and only invariants of
   the map are quantities.** Many weight settings give the same predictions.
   Quantities computed from the predictions mean the same thing however the
   weights are written; quantities computed from the weights directly do not.
   Training returns a distribution over weights (the posterior, a stochastic
   optimiser's end point) and the user receives its push-forward, a predictive
   distribution; this is the Bayesian view, and it is the one in which the
   classical and the singular case are the same statement. The shape of the
   family near its redundant points is a question in algebraic geometry
   (blow-ups), and the effective number of parameters is the first invariant it
   yields (Watanabe). (§3, §4)
6. **Memory is sized to the use.** We never need to learn everything about the
   world. A use for the predictions fixes how accurate they must be, and that
   fixes how much memory the machine needs. The smallest memory that predicts
   perfectly is a well-defined object, and trained predictors are found to
   approach it. (§2, §5)

## Running examples

| Example | The world | What is read | The learner |
|---|---|---|---|
| Adult heights (NHANES) | people arriving at a clinic | height, to the nearest centimetre | a Gaussian; a mixture of two Gaussians |
| Text | authors, topics, intentions | the next token | a transformer, whose memory is its context window |
| Sequences from a hidden Markov model; Othello games | a small hidden machine we wrote ourselves | its output symbols; the moves | a transformer trained only to predict |

Heights and text are carried side by side through every section: each part is
introduced for the Gaussian, then for the language model, then named. Heights
carry the ideas about weights (§3). Text carries everything at scale. The
hidden-Markov case is the one where we know the world's machine exactly, so we can
check what the learner's memory holds (§5). It is also where the toy-transformer
work on this branch connects.

**Running device.** A two-column table (Gaussian | language model), one row per
part, shown empty in §0 and filled one row at a time; the completed table closes
§5. Rows: world, instrument, record, symmetry and macrostate, next-step rule,
memory, family of rules, weights, score and gap, target, the move, what training
returns, the check, learnable, not learnable.

---

## §0 Introduction (~900 words)

- **Opening scene.** Two different sets of weights give a network exactly the same
  predictions. Rescaling one layer and unscaling the next turns a sharp minimum
  into a flat one without changing a single prediction. Which facts about a trained
  model are facts about the model, and which are facts about how we wrote it down?
- **Same things, different names.** One short list: "cross-entropy" and
  "likelihood"; "ensemble" and "posterior predictive"; "hidden state", "latent",
  "belief state" and "residual stream"; "probe" and "estimator". The note gives
  each pair one name.
- **The fields this draws on**, one plain sentence each: probability and
  statistics (how to reason from data); statistical physics (how a few readings
  relate to a detailed world); systems theory (how to build a machine with memory
  that reproduces a signal); information geometry and singular learning theory
  (the shape of the space of models); computational mechanics (the smallest
  machine that predicts a process).
- The four things the note says; the main diagram (§3) as a figure.
- **What is left out** and why that is a choice: learners whose actions change
  the world are well studied too (Note 2); here the learner reads and decides,
  but its decisions do not change the world.
- **Bookkeeping.** Each section ends with one line: what was given, what was
  chosen, what was assumed, what was approximated.
- The empty two-column table, as a figure. "Each part fills a row."

## §1 The world, the instrument, and the record (~3,200 words)

Each part: the idea in plain words, one example, then the name.

**1.1 The world.** The world at one instant has a complete description, its
**microstate**: the positions and velocities of everything, or for our purposes
everything that could matter. The set of all microstates is the **phase space**,
and the world moves through it by its **dynamics**. Nobody sees the microstate,
and no result in the note depends on knowing it; it is there to make clear what
the data are, and what they are not.
- *Example:* a person walking into the clinic, with everything true about them.

**1.2 The instrument and the records.** We see the world only through an
**instrument** that reads a few numbers off the microstate: a height to the nearest
centimetre, a token. Reading a few numbers off a detailed state is
**coarse-graining**. The readings in order are the **records**. Two things make
records vary: the world changes, and many different microstates give the same
reading.
- *Example:* two people of the same height but different sex give the same
  reading.

**1.3 Histories.** Everything that happens, past and future, is one **history**.
Only one history actually occurs; we see part of it, through the instrument. All
possible histories form the **history space**. Probability enters as a way of
speaking about the histories that agree with what we have seen so far but differ
in what comes next.
- *Sampling, made precise.* Our data are the readings of the one history that
  occurs. "Repeating the experiment" means one of two things: the next person
  through the door (another reading in the same history, the only kind that gives
  data), or what might have happened instead (another history, which we never
  see and use only to compute averages).

**1.4 Observables, hidden and learnable.** In physics an **observable** is a
quantity that depends on the microstate. Here it is any quantity that depends on
the history: a height, a person's sex, the average height of everyone who will
ever come in. Some observables are read by the instrument; the rest are
**hidden** at a given time, meaning that what has been read so far does not
determine them. An observable is **learnable** if the complete record, if we waited
forever, would determine it. Your best guess of an observable converges to the
truth exactly when it is learnable (Doob's theorem).
- *Example:* sex is hidden if not recorded, and not learnable from heights alone:
  no amount of height data tells you the sex of the next person. The average
  height of everyone is learnable.
- *Physics aside:* when you track only a few readings of a detailed world, the
  rest shows up as memory and noise in those readings (Mori–Zwanzig). That is why
  readings alone are rarely enough to predict the next one without some memory.

**1.5 One history must be enough.** We have one history, not many copies of the
world. Learning from it needs a **symmetry**: a way of reordering the records that
does not change the odds. Two cases cover the examples:
- **exchangeable:** shuffling the records does not change the odds (people
  arriving at the clinic in random order);
- **stationary:** shifting the records in time does not change the odds (the
  middle of a long text looks statistically like any other part).
A symmetry has two consequences. First, the **ergodic hypothesis** holds for the
records: one long history gives the same averages as an **ensemble** of many
copies of the world. So almost every history has well-defined long-run
frequencies, and these are what "the distribution of the data" means: the
**law** of the records (the ergodic theorem). Second, a history has a
**macrostate**: the part of the law that a long run reveals and that never
changes, its *type* (de Finetti's theorem and its generalisations).
- *When the symmetry seems to fail:* the world may have something that changes
  slowly compared with how long we have watched. The fix is to treat that slowly
  changing thing as hidden and look for a symmetry one level up (a regime that
  switches, rather than a parameter that drifts). If no level has a symmetry
  within reach of the data, nothing is learnable.
- *Footnote:* the law exists without anyone assuming it, as a property of the one
  history (the Boltzmann view). Describing it through the ensemble (the Gibbs
  view) gives the same averages and is the more common way to say it.

**1.6 What is learned: a next-step rule.** A predictor does not need the whole
law. It needs the odds of the next reading given the past: the **next-step rule**
(a "conditional distribution", or "kernel"). A language model learns the rule for
the next token given the context; a regression model learns the rule for the
output given the input. How often each past or input occurs is left unmodelled;
it only decides which cases matter most. The world has one next-step rule; every
model proposes another; they are compared there.

**1.7 The learner: a machine with memory.** A predictor of a stream needs memory.
A learner is a **machine with memory**: a **state** that holds what it has kept
from the past, an **update rule** that changes the state with each new reading,
and a **prediction rule** that turns the state into odds for the next reading.
The technical name for such a machine, when it reproduces a given next-step rule,
is a **realization** (from systems theory). Three facts carry the rest of the note:
- **The simplest memory is the whole past.** A transformer is this: its memory is
  its context window.
- **There is a smallest memory that predicts perfectly**: group together all pasts
  that lead to the same odds for the future; the group a past belongs to is all
  that needs to be remembered. It is unique, and it depends only on the world's
  next-step rule (named **causal states** in §5).
- **The machine that produced the data and the smallest machine that predicts it
  are usually different.** A hidden Markov model can be smaller than its best
  predictor. Saying that a learner's memory "is" something real in the world is a
  further claim, made only when needed.
Training a sequence model is building such a machine from one long record.
- *The classical case, in these terms.* With exchangeable records the state can
  be a finite summary of the past (a sufficient statistic: the running mean and
  variance for the Gaussian) and it settles as the record grows. With
  stationarity alone the state must keep tracking: this is stochastic
  filtering (the Kalman filter, the forward pass of a hidden Markov model),
  and a language model is the same case with a learned update rule. The i.i.d.
  model is the filter whose state converges, not a different kind of object.

**1.8 Weights.** In practice the machine is written with weights: numbers that
set its update and prediction rules. Weights are how we write a machine down; §3
is about the difference.

**Who has said what** (one table, end of §1): which fields have precise language
for which parts. Physics: world, dynamics, instrument, symmetry. Probability:
histories, observables. Statistics: the law, its type (the parameter). Systems
theory: machines with memory. Computational mechanics: the smallest memory.
Machine learning: next-step rules and weights. Singular learning theory: the map
from weights to rules.

**Bookkeeping.** Given: the world and the instrument. Chosen: the learner's
memory and its weights. Assumed: a symmetry. Derived: the law, the type, what is
learnable.

## §2 Comparing rules, and how much must be learned (~2,000 words)

- **The gap between two rules.** Score a rule by the log-probability it gave to
  each reading that happened (the **log score**). The average shortfall of a
  model's score against the world's own rule is the **KL divergence**, "the gap".
  It is zero only when the rules agree, and it has a second meaning: the extra
  bits per reading needed to encode the data with the model's rule instead of the
  true one.
- **The floor.** Part of the surprise in each reading cannot be removed by any
  predictor; it is the noise in the world as seen through the instrument. The
  training loss is this floor plus the gap. Only the gap can be learned away.
  What the record contains splits the same way: **regularity**, the part the
  past tells about the future, and **noise**, the rest. Regularity is what a
  code can exploit (log score = code length, so predicting well and compressing
  well are the same thing), and it is always relative to a split: input versus
  output, past versus future. For exchangeable records the split is explicit:
  the empirical distribution is the regularity and the order is noise.
- **How much the past tells about the future.** The **predictive information** is
  how much the last T readings tell you about the future. How it grows with T says
  what kind of process you face:

  | Growth with T | What it means |
  |---|---|
  | levels off | a finite memory suffices |
  | grows like (d/2) log T | you are learning d numbers about the world |
  | grows like a power of T | there is always more to learn |
  | grows linearly | it is mostly noise |

  Recording more observables can move a process down this table: if you record
  the slow variable, the drift disappears (Bialek, Nemenman & Tishby).
- **Why the gap, and when less is enough.** Any use of the predictions with a
  bounded payoff loses at most a multiple of the square root of the gap
  (Pinsker). So a model with a small gap is good for every use at once,
  including uses not yet specified; that is why the gap is the loss when the
  use is left open, as for a language model, and the log score is the one
  honest score for stating a distribution (Bernardo). A fixed use often needs
  far less: for classification only the most probable label must be right, so
  a task-specific learner can model only what it needs (Vapnik's advice), while
  a general-purpose learner models all of the rule.
- **How much must be learned.** We never need perfect predictions. A use for the
  predictions sets how large a gap is tolerable. That sets how much predictive
  information the memory must hold. And that sets the least memory needed: the
  **memory versus accuracy curve** (predictive rate–distortion; the information
  bottleneck). The smallest perfect memory of §1.7 is the end of the curve.
  Memory that holds things that do not help prediction also costs energy to
  maintain (Still et al. 2012). So the real task is to reach the point on this
  curve that the use requires and the data allow.
- **The best a model can do.** If the model cannot express the world's rule, its
  target is the closest rule it can express. How far that is from the truth is
  the error that more data will never remove.
- **Where the two machines meet.** The world's machine and the learner's are
  different objects with different internal spaces; the only place they are
  compared is the next-step rule. When the learner's family contains the
  world's rule, the model is **realizable** (a statistician says
  **well-specified**). Then the gap can reach zero, the floor is all that
  remains of the loss, the macrostate of §1.5 is a point of the family, and
  the memory needed is known before training. Nearly every classical theorem
  (consistency, the asymptotic normality of estimators, the usual information
  criteria) assumes this case, which is why so much effort goes into checking
  it. It is the special case. The general case, where the family does not
  contain the rule, is the one above: the target is a projection, and the
  learner's memory and the world's need not resemble each other (§5.1). Watanabe's
  theory is stated for the gap under prediction with the log score; a use other
  than prediction sets a different gap and a different target (one sentence,
  pointer to the research note).

**For a language model.** Scaling curves are measurements of how the predictive
information of text grows; the context window bounds how much memory the model
can bring to bear.

**Bookkeeping.** All of this is a property of the world's rule, which is given and
unknown.

## §3 Weights are not the machine (~2,800 words; the core)

The diagram: weights → machine → next-step rule ← world. The learner and the world
both arrive at a next-step rule and are compared there. The data are a rough
version of the world's rule.

- **3.1 The map from weights to rules.** Every weight setting gives a rule. The
  set of weight settings giving the same rule is a **fibre**. A model is
  **identifiable** if every rule comes from exactly one weight setting. Heights:
  in a two-Gaussian mixture, swapping the two components gives the same
  distribution (a fibre of two points); when the components coincide, the mixing
  weight can be anything (a fibre that is a whole line).
- **3.2 Coordinates or description?** When the map is one-to-one and smooth with
  no flat directions, the weights are honest coordinates on the set of rules the
  model can express, and the usual geometry applies (information geometry). This
  is the **regular** case. When it is not, the model is **singular**, and the
  weights are a *description* (a **parametrisation**) rather than coordinates (a
  **chart**). Picture: the curve traced by t ↦ (t², t³) has a sharp point at the
  origin; t describes the curve, but is not a coordinate there. Every network with
  hidden units has such points.
  - *The image is a set of rules, not always a manifold.* Two things can fail.
    If the map is many-to-one but of full rank, the fibres are symmetries of
    the description; dividing them out (the **quotient**) gives honest
    coordinates again and information geometry is recovered on the quotient.
    If the rank drops, the image has a sharp point, the pulled-back Fisher
    information is degenerate there, and no metric exists at that point; no
    quotient repairs it. Coordinates return only locally, after a **blow-up**
    that replaces the sharp point by a smooth surface (resolution of
    singularities, Hironaka), and that is where the learning coefficient is
    read off.
  - *What still works:* any quantity defined on rules can be read on weights by
    first going to the rule (**pullback**). This works for singular models too,
    and it needs no manifold structure on the space of rules at all: a
    function, a quadratic form, an observable can be pulled back along any
    smooth map. What fails is reading the pulled-back Fisher information as a
    metric.
  - *History, one paragraph:* geometry started with descriptions (Gauss) and moved
    to coordinates (Riemann); statistics inherited coordinates; singular models
    brought the descriptions back (Watanabe).
  - *Physics box:* physics met the same problem with redundant descriptions of the
    same physical state (gauge theories, general relativity). Its lesson: the
    sharp points are where the description has extra symmetry, and the only
    meaningful quantities are those that do not depend on the description.
- **3.3 The quantities, read through the map.** The training loss, the gap, and
  the **Fisher information** (how quickly the rule changes when the weights
  move) are all defined on rules and read on weights. Near a singular point the
  usual count of parameters is wrong; the right count is the **effective number
  of parameters** (the learning coefficient λ), which does not depend on how the
  weights are written.
- **3.4 One test for any quantity.** A **reparametrisation** rewrites the weights
  without changing the rules they give (rescale one layer, unscale the next). A
  quantity computed from rules survives it; a quantity computed from weights
  directly does not. Three pairs:
  - Fisher information (survives) vs the Hessian's sharpness (does not; Dinh et
    al. 2017);
  - maximum likelihood (survives) vs a prior-weighted maximum (does not);
  - averaging predictions (survives) vs averaging weights (does not).
- **3.5 What a probe can find.** A quantity computed from the weights says
  something about the machine only if it is the same across a fibre. On the world
  side, a hidden quantity can be learned only if it is learnable (§1.4). Heights:
  the difference between the component means is meaningful up to swapping, and
  meaningless when the components coincide. Recording sex makes it learnable.
- **3.6 Three sources of error.** The model cannot express the truth; the data are
  finite; the optimiser did not finish (Bottou & Bousquet). §2 says how much of the
  first one matters for a given use.

**For a language model.** Permuting neurons, rescaling layers, and the
embedding–unembedding product all create fibres. A "feature" found in the weights
is about the model only if it survives these.

## §4 What training returns (~1,800 words)

- **4.1 Answers as distributions over weights.** Maximum likelihood, the posterior,
  and the result of a stochastic gradient descent run are all distributions over
  weights (a single point is a special case). For a mixture, maximum likelihood
  does not exist: the likelihood can be made arbitrarily large.
- **4.2 Average predictions, not weights.** Sending each weight setting to its rule
  and averaging the rules (**push-forward**, then average) gives the posterior
  predictive or an ensemble; the result can be better than any single setting.
- **4.3 Convergence is about rules.** With more data the predictions approach the
  best expressible rule, in three layers: the loss of each rule settles (law of
  large numbers); it settles for all rules at once, so the minimiser settles
  (consistency); and the fluctuations around the limit have a size, set by the
  effective number of parameters. The regular and singular cases differ only in
  the third layer. In the Bayesian view the statement is the same for both:
  the posterior predictive's gap shrinks at a rate set by the learning
  coefficient, which equals half the parameter count in the regular case and
  is smaller otherwise (Watanabe). One paragraph, cited.
- **4.4 What can be computed.** Averages under the world (only samples),
  minimisation over a non-convex landscape, and integrals over weights are all out
  of exact reach; what training returns is a random weight setting whose
  distribution depends on the optimiser. These limits are the computational side
  of the memory-versus-accuracy limit of §2.
- **4.5 Held-out loss.** It estimates the gap fairly only if the held-out records
  are exchangeable with the training ones (§1.5).

## §5 What the learner's memory holds (~2,000 words)

- **5.1 Two kinds of agreement.** **Predictive alignment:** the learner's next-step
  rule matches the world's (or gets as close as it can). This is all the loss can
  see. **Structural alignment:** the learner's memory matches the smallest perfect
  memory, or, as a further claim, a real hidden quantity in the world. The second
  is stronger and does not follow from the first in general.
- **5.2 Good memory.** Memory is **sufficient** if it holds all the predictive
  information the use requires, and **efficient** if it holds nothing else.
  Memory that tells apart pasts with the same future is memorising noise:
  overfitting at the level of the representation. The smallest perfect memory,
  the **causal states**, is the efficient memory at zero tolerance.
- **5.3 Science and prediction ask for different things.** Prediction needs
  predictive alignment only. Science asks about a named hidden quantity (an effect
  size, a disease state), so it needs that quantity to be learnable from the
  records and meaningful in the model (same across a fibre).
- **5.4 Predictors find the smallest memory.** Transformers trained on
  hidden-Markov data hold the belief states of the smallest predictor in their
  residual stream, not the states of the generating machine (Shai et al. 2024).
  Othello-GPT holds the board, which is exactly what is needed to predict the next
  move. The reason: predicting well requires the predictive information, and the
  smallest perfect memory is what holds it. Saying "the model knows the board" is
  the further step from "the model holds the smallest memory" to "the board is
  real".
- **5.5 Whether the world's rule is 'real'.** None of §1–§5 depends on it. A model
  can be read as a compression of the regularity in the records (Rissanen, Dawid,
  Grünwald), and everything above still holds.
- **5.6 Using the picture.** The filled two-column table. Then, one paragraph
  each, pointing back to a row: *what goes wrong* (a slow hidden variable read as
  "distribution shift"; probing for a quantity that is not learnable; a
  weight-space diagnostic read as a fact about the model; memory that holds noise;
  a held-out set that broke the symmetry) and *how to build better* (record the
  slow variable; size the memory to the use; measure invariants, not coordinates;
  make the quantity you care about learnable by what you record).

## §6 Field guide (~800 words + table)

A table of terms people use, each with: what it is in this note; whether it is
about rules or about weights; whether it survives reparametrisation. Rows:
cross-entropy / likelihood / perplexity; Fisher information; Hessian sharpness;
weight norm and weight decay; posterior predictive and ensembles; average of
weights; calibration; belief state; probe and concept vector; identifiability;
AIC, BIC, WAIC; effective number of parameters; hidden state, latent, residual
stream; context window; predictive information; statistical complexity;
state-space model; superposition; "non-stationary"; scaling law; KV cache.

## §7 One picture across fields (~1,200 words + table)

The connective-tissue table of the research note, with plain-language row titles:

| Idea | Physics | Probability | Statistics | Systems theory | Machine learning |
|---|---|---|---|---|---|
| the complete description | microstate | a point of the history space | — | the generator's state | — |
| what is read | observable; coarse-graining | random variable | the experiment | output | input / token |
| what is not read | — | not determined by the readings | latent variable | unobserved state | hidden state |
| one history stands in for many | ergodic hypothesis | ergodic theorem | de Finetti | — | i.i.d. assumption |
| what a long run reveals | macrostate | the type of the history | the parameter | — | "the data distribution" |
| why readings alone are not enough | memory and noise (Mori–Zwanzig) | — | — | — | long-range dependence |
| memory that suffices | — | Markov property | sufficient statistic | state of a realization | hidden state; belief state |
| the smallest such memory | — | prediction process | minimal sufficient statistic | minimal realization | causal states |
| producer ≠ predictor | — | — | — | external vs internal realization | generator vs belief states |
| rebuilding the state from readings | Takens embedding | — | — | system identification | training a sequence model |
| redundant descriptions | gauge symmetry | — | non-identifiability | many realizations | weight symmetries; fibres |
| only description-free quantities count | gauge invariance | — | reparametrisation invariance | — | λ vs sharpness |
| sharp points have extra symmetry | special configurations | — | singular models | — | merged components; dead units |
| how much must be learned | — | — | sufficiency for a use | — | tolerance on the gap |
| memory versus accuracy | energy cost of memory | — | — | — | information bottleneck |
| how learnable, from inside | — | — | parametric vs nonparametric | — | growth of predictive information; scaling laws |

Each row gets a status (a theorem, a definition, or an analogy), a one-paragraph
explanation in the text, and its sources in Appendix B.

- **Closest earlier work**, one paragraph each: computational mechanics and Shai
  et al.; information geometry of agents (Ay and coauthors); singular learning
  theory (Watanabe; Lau et al.); Bayesian mechanics (Friston and coauthors), which
  assumes the learner's memory is a real feature of the world.
- **Note 2:** learners whose actions change the world; when two learners are
  the same; quantities of the
  training process that survive reparametrisation.
- Open questions.

## Appendix A: symbols and bookkeeping
## Appendix B: sources (from `prior-art.md`), one line per part and per §7 row
## Appendix C: the careful version

For mathematically careful readers, one page restating §1 with measurable spaces,
σ-algebras, filtrations, regular conditional distributions and standard Borel
assumptions, so the body can stay plain without losing rigour.

---

## What changed from the research outline

| research note | blog post |
|---|---|
| Postulates P1–P5 | §1.1–1.8, each "idea, example, name" |
| filtration, atom, 𝓖_∞-measurable | "what has been read so far"; "determined by the complete record"; symbols in Appendix C |
| ergodic decomposition, invariant σ-field | "the type of a history"; macrostate as the physics word |
| kernel, meeting space 𝒦 | next-step rule; "compared there" |
| stochastic realization, splitting, internal/external | machine with memory; producer vs predictor |
| relative entropy rate, Čencov, Fisher–Rao | the gap (KL); Fisher information only |
| predictive rate–distortion, decision sufficiency, dissipation | memory versus accuracy curve; "a use sets the tolerance"; energy cost of memory |
| chart vs parametrisation, pullback | coordinates vs description; "read on weights by first going to the rule" |
| connective-tissue table with theorem names | same rows with plain titles; theorem names in Appendix B |
| measure-theoretic rigour in the body | Appendix C |

All content of the research outline is kept; nothing was removed except vocabulary. Its
provenance table applies unchanged.

## Decisions needed

1. **"Next-step rule"** as the body's word for a kernel / conditional
   distribution, with "kernel" introduced once? Or keep "conditional
   distribution" throughout, which CS readers already know?
2. **"Machine with memory"** for a realization, with "realization" named once?
3. **Appendix C** for the measure-theoretic version, so the body has no σ-algebras?
4. **"Macrostate" vs "type"** for the long-run invariant: with physics now in the
   reader's background, macrostate leads and "type" is its plain synonym. Keep
   both, or drop "type"?
