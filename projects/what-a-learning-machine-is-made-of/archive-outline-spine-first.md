> **Superseded.** The user preferred v9's opening (two machines) with the purpose stated after it. v9 now carries the purpose paragraph, the two-column running device and the "using the picture" section from this draft. Kept for reference only.

# Outline v10: The Same Machine

*From a Gaussian fit to a language model: the parts of a learning problem, named
once.*

**Purpose.** A note for new graduate students (and anyone who builds, studies or
uses these systems) that shows one thing: a large language model and a
second-year statistical model have the same structure. Same parts, same
questions, same failure modes; more complexity, many special cases. A student who
has read it should be able to answer "what is the difference between an LLM and
what you learned in statistical inference?" with: *nothing in structure; the
differences are in which parts are large, which are hidden, and which are
approximated.*

**What it is not.** Not the research synthesis. The connective-tissue table, the
realization-theory framing, the prior-art positioning and the rigorous postulates
(v8, v9, `prior-art.md`) are the outline of a later paper, to be written after
Part 2. This note borrows from them only what a student needs to see the picture.

**Reader.** Bachelor's in computer science, mathematics or physics; comfortable
with probability and basic machine learning; knows or will accept undergraduate
physics vocabulary (microstate, observable, ensemble, macrostate). Writing rules in
`glossary.md`.

**Length.** 8–10k words; four posts.

## The note in one paragraph

Fit a Gaussian to a column of heights and you have done everything a language
model does: you decided what the world is and what your instrument reads from it;
you assumed that one long record is enough to learn from; you chose a family of
rules for predicting the next reading; you wrote that family down with a few
numbers; you scored the rules against the record and moved the numbers to improve
the score; and you checked the result on readings you had held back. A language
model has the same parts. What changes is that the world is text instead of
people, the instrument reads tokens instead of centimetres, the rule needs a
memory of the past instead of none, the numbers are billions instead of two, and
nothing can be computed exactly. Every part has a name, and most parts have two or
three names in different fields. This note gives each part one name, shows the
Gaussian and the language model side by side at every step, and uses the
picture for three things: to see what can go wrong, to see how to build better
machines, and to translate between fields.

## The spine: one table, grown through the note

Every section adds one row. By the end the reader has this:

| Part | Gaussian fit to heights | Language model | The name used |
|---|---|---|---|
| the world | people arriving at a clinic | people writing documents | world, microstate |
| the instrument | a ruler, to the nearest cm | tokenisation | instrument, coarse-graining |
| the record | n heights | a corpus of tokens | record, history |
| what makes one record enough | shuffling does not matter | shuffling documents does not matter; within a document, position does not matter | symmetry (exchangeable, stationary); macrostate |
| what is predicted | the next height | the next token given the context | next-step rule |
| the memory | none | the context window | machine with memory; state |
| the family of rules | all Gaussians | all transformers of a given shape | model; image |
| the numbers | (mean, sd) | billions of weights | weights; parametrisation |
| the score | log-likelihood | cross-entropy | log score; the gap (KL) |
| the best possible | the closest Gaussian to the true heights | the closest transformer to the true next-token rule | target; approximation error |
| the move | solve for mean and sd | SGD | training; optimisation error |
| what training returns | a point | a random point whose law depends on the run | solution concept |
| the check | held-out log-likelihood | held-out loss, perplexity | evaluation; estimation error |
| what can be learned | the mean and sd of the population | whatever the next-token rule requires the memory to track | learnable; the smallest memory |
| what cannot be learned | a person's sex from height | an author's intent beyond what text reveals | hidden, not learnable |

## Running examples

Two only, carried all the way through: **heights** (NHANES; Gaussian, then a
two-Gaussian mixture when the note needs a hidden variable and a singular point)
and **text** (a language model). The hidden-Markov/transformer experiment
appears once, in §3, as the demonstration that a predictor's memory ends up
holding what prediction requires.

---

## §0 Why the picture matters (~700 words)

- Open with the question a statistician gets asked, and the answer the note
  defends.
- Three uses of the picture, each in two sentences, each pointing to a section:
  *what goes wrong* (§4), *how to build better* (§4), *same thing, two names* (§5).
- The spine table, empty, as a figure. "Each section fills a row."
- One paragraph: what is deliberately left out (learners that act on the world;
  tuning to a goal; Note 2), and that the research version with the full
  mathematics comes later.
- Bookkeeping line introduced: given / chosen / assumed / approximated.

## §1 The problem (~2,500 words)

Each part: the idea, the Gaussian column, the language-model column, the name.
One new term per paragraph. Physics vocabulary used freely after one-sentence
definitions.

- **1.1 World, instrument, record.** The world has a complete state (microstate)
  we never see; an instrument coarse-grains it to a few numbers; the readings in
  order are the record. Both examples. Variation in the record has two sources:
  the world moves, and many microstates give the same reading.
- **1.2 One record must be enough.** We have one history, not many worlds. A
  symmetry (shuffling, shifting) makes one long record stand in for the ensemble
  (the ergodic hypothesis), gives the record a law (its long-run frequencies),
  and gives it a macrostate: what a long run reveals and what never changes.
  Heights: exchangeable; the macrostate is the height distribution. Text:
  exchangeable across documents, stationary within; the macrostate is the law of
  text. When the symmetry seems to fail, something slow is hidden; record it or
  go up a level.
- **1.3 What gets predicted.** Not the whole law: the next-step rule, the odds of
  the next reading given the past. For heights there is no past that matters.
  For text, the past is the context. How often each past occurs is not modelled.
- **1.4 Observables: hidden and learnable.** Any quantity that depends on the
  history is an observable. Hidden: not fixed by what has been read. Learnable:
  fixed by the complete record. Sex is hidden and not learnable from heights;
  the population mean is learnable. Your best guess converges to the truth
  exactly when the quantity is learnable (Doob).
- **1.5 The machine.** A predictor of a stream is a machine with memory: a state,
  an update rule, a prediction rule. The Gaussian has no memory. The transformer's
  memory is its context window: the simplest memory, the whole recent past.
  There is a smallest memory that predicts perfectly (group pasts with the same
  future); §3 returns to it.
- **1.6 Weights.** A family of rules written down with numbers. The map from
  weights to rules is the object §2 is about.
- **1.7 Score and target.** The log score; the gap (KL) as the average shortfall
  against the world's own rule; the floor (noise) that no predictor removes; the
  target as the closest rule the family can express; three errors: cannot
  express, finite data, optimiser stopped (Bottou–Bousquet).
- **1.8 Training and checking.** Solve vs iterate; what each returns; held-out
  evaluation and why it needs the symmetry of §1.2.

**Bookkeeping line.** Given: world, instrument, score. Chosen: family, weights,
memory, optimiser. Assumed: symmetry. Approximated: the target (by the family),
the law (by the record), the minimum (by the optimiser).

## §2 Weights are not the machine (~2,000 words)

The one piece of geometry the note needs, kept to what a student can use.

- **2.1 Fibres.** Many weight settings, one rule. Swapping the two components of a
  mixture; permuting neurons; rescaling one layer and unscaling the next. The set
  of weights giving one rule is a fibre. Identifiable = every fibre is a point.
- **2.2 The test.** A reparametrisation rewrites weights without changing rules. A
  quantity computed from rules survives it; one computed from weights alone does
  not. Three pairs: Fisher information vs Hessian sharpness; maximum likelihood
  vs a prior-weighted maximum; averaging predictions vs averaging weights.
- **2.3 Singular points.** Where the components of a mixture coincide the mixing
  weight stops mattering and the usual count of parameters is wrong. Every network
  with hidden units has such points. The right count is the effective number of
  parameters (λ), which survives reparametrisation. Said in four sentences, cited,
  not developed.
- **2.4 What a probe can find.** A quantity read off the weights is about the
  machine only if it is constant on fibres. Heights: the gap between component
  means is meaningful up to swapping, meaningless at the singular point.

**Bookkeeping line.** Everything here is a property of the chosen family.

## §3 What the memory holds (~1,500 words)

- **3.1 Enough, and no more.** A good memory holds what the past says about the
  future (predictive information) and nothing else. Holding more is memorising
  noise. The smallest perfect memory is a well-defined object that depends only
  on the world's next-step rule (causal states).
- **3.2 How much must be learned.** A use for the predictions fixes a tolerable
  gap; the gap fixes how much predictive information is needed; that fixes the
  least memory (the memory-versus-accuracy curve). We never learn the law of
  the world; we reach a point on this curve.
- **3.3 The experiment.** Transformers trained on hidden-Markov sequences hold
  the belief states of the smallest predictor in their residual stream, not the
  states of the machine that generated the data (Shai et al. 2024). Othello-GPT
  holds the board, which is what predicting the next move needs. Lesson: a
  predictor's memory converges to what prediction requires, which may or may not
  be something "real" in the world; saying it is real is a further claim.
- **3.4 Two kinds of agreement.** Predictive (the rule matches) vs structural (the
  memory matches a named hidden quantity). Prediction needs the first; science
  asks for the second, and only for learnable, identifiable quantities.

**Bookkeeping line.** Derived from the world's rule; the "real" claim is optional.

## §4 Using the picture (~1,500 words)

**What goes wrong**, one paragraph each, each pointing back to a row:
- the record was not long enough for a slow variable (§1.2): drift, "distribution
  shift";
- a hidden quantity was asked for that is not learnable (§1.4): the probe finds
  noise;
- a weight-space diagnostic was read as a fact about the model (§2.2): sharpness,
  weight norms, cosine similarities between neurons;
- the memory holds noise (§3.1): representation overfitting;
- the held-out set broke the symmetry (§1.8): contamination.

**How to build better**, one paragraph each:
- record the slow variable instead of fighting drift;
- choose the memory for the predictive information the use needs, not the
  maximum available;
- measure invariants (Fisher, λ, held-out gap), not coordinates;
- when a hidden quantity matters, make it learnable by what you record (the
  heights-with-sex example).

**The same machine**, closing: the filled spine table, with one sentence per row
on what grew from the Gaussian to the LLM and what did not change.

## §5 Dictionary (~600 words + table)

Terms with two or three names: cross-entropy / log-likelihood / log score;
ensemble / posterior predictive; hidden state / latent / belief state / residual
stream; parameter / macrostate / ergodic component; feature / observable;
state-space model / machine with memory; distribution shift / broken symmetry;
sufficient statistic / minimal memory; sharpness vs effective parameters. One
line each: the name used here, the fields that use the others, whether it is
about rules or about weights.

## §6 Where to go next (~400 words)

- Note 2: learners that act (choosing what to record; agents; tuning to a goal).
- The research version: the full mathematics, the realization-theory framing, the
  connective-tissue table across physics, probability, statistics, systems theory
  and machine learning (`outline-v9.md`).
- Reading list, ten items, one line each.

## Appendix: symbols and the careful statements (1 page)

---

## Relation to earlier outlines

| Earlier material | In v10 |
|---|---|
| Five postulates (v8/v9 §1) | §1, compressed, Gaussian-and-LLM at every step |
| Charts vs parametrisations, physics box, resolution of singularities | §2.3 in four sentences; rest to the research paper |
| Stochastic realization theory, internal vs external | §1.5 and §3.3 in plain terms; theory to the research paper |
| I(T) growth classes, rate–distortion, dissipation | §3.2 in one paragraph; detail to the research paper |
| Connective-tissue table (v8 §7) | §5 dictionary (short); full table to the research paper |
| Prior art, contribution statement | research paper only |
| Field guide (v8 §6) | merged into §4 "what goes wrong" and §5 |
| Ledger | the bookkeeping line, kept |
| Height ladder, text, HMM/transformer | kept; HMM used once in §3.3 |

## Decisions needed

1. **Spine table as the organising device**, filled row by row? *Recommendation:
   yes; it is the thesis made visible.*
2. **Mixture of Gaussians**: introduce it in §1 (as the second height model) or
   only in §2 when fibres and singular points are needed? *Recommendation: §2;
   §1 stays Gaussian vs LLM only.*
3. **§4 "what goes wrong" list**: the five above, or add "reward hacking" as a
   pointer to Note 2?
4. **Length**: 8–10k words (four posts) rather than 12k?
