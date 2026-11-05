# Outline v4: The Structures Underneath Learning Problems

*Part 1 of 2.* (Working titles: *Spaces, Maps and Pullbacks: the Mathematics of
Learning Problems*; *What Learning Machines Compute, Named*.)

Replaces v3. v3 developed Part 2 (equivalence of learning machines, invariants of
the dynamics, control bundles) before Part 1 had named the objects it uses. It also
leaned on the control-theory metaphor more than the material supports.

**Part 1 (this note)** names the mathematical structures that the statistics, ML, RL
and scientific-modelling communities all use, often under different names or none.
It places them in one framework and says how to think about the quantities each
field computes.

**Part 2 (later)** uses the framework: when two learning machines are the same,
which quantities are invariant, and why only invariants are worth measuring. Part 1
ends by pointing to it and keeps the claims it needs to a few concrete examples.

Length target: under an hour (~10–12k words), or four to five posts.

## The picture

One diagram carries the note:

```
              learning process (w_t, or a posterior on W)
                       │ push forward
                       ▼
   W  ──── p ────►  Δ(𝒳)  ──── U ────►  ℝ
 parameters       distributions        utility, observables
   ▲                    │
   └──── pull back ─────┘
   loss, Fisher information, observables as functions on W
```

- **On Δ(𝒳)** live the things the problem is about: the target q, utilities and
  scoring rules, divergences, the Fisher–Rao metric, mixtures, and observables
  (expectations, quantiles, decisions).
- **The model map** p: W → Δ(𝒳) is where learning takes place, because computation
  happens in coordinates.
- **Pullbacks** bring the structures of Δ to W: utility → loss; KL → K(w);
  Fisher–Rao → Fisher information; an observable f → f∘p.
- **Push-forwards** take what happens on W back to Δ: a training trajectory becomes
  a path of predictive distributions; a posterior on W becomes a distribution over
  distributions, and its average is the Bayes predictive.

**The one principle stated in Part 1:** a quantity pulled back from Δ is invariant
under reparametrisation of W by construction. A quantity defined only on W (weight
norms, Hessian in coordinates, the posterior mean of w) is not. Part 2 builds on
this.

## Thesis

Prediction, parameter estimation, scientific inference and control are all choices
of an action under uncertainty, judged by a utility that is realised later. Each
learning problem is fixed by **what the action is** and **which functional of the
world the utility asks about**. The mathematical structures (the map, the
structures on the space of distributions, their pullbacks to parameters, the
learning process and its push-forward) are shared across all of them. Seeing them
named makes clear which quantities people compute are about the world, which are
about the model, and which are about the coordinates.

## Running examples

| Example | Role |
|---|---|
| Adult heights: categorical, Gaussian, two-Gaussian mixture, mixture with sex recorded | the map, image, fibres, pullbacks, the ladder of idealisations (§2–4) |
| A scientific question on the same data: "what is the mean height difference between the sexes?" | utility on a hidden quantity; it is learnable only if it is constant on fibres (§1, §3) |
| Chess (AlphaZero) or Othello | a control problem whose learned model carries the hidden structure of the game (§5) |
| Transformer on text | everything at scale; a short paragraph at the end of §2–5 |

---

## §0 Introduction (~1,000 words)

- The same objects under different names: "likelihood" and "cross-entropy",
  "Fisher information" and "Gauss–Newton", "posterior predictive" and "ensemble",
  "probe" and "estimator of a functional".
- The diagram and the one principle.
- Plan; what is not claimed (no new theorems; every link is cited, defined, or
  labelled as an assumption); Part 2 named.

## §1 Every learning problem is a decision problem (~2,000 words)

**Goal:** establish the general setting before any model appears.

- **Actions, states, information, utility.** A decision problem: choose an action
  from information so as to maximise a utility that depends on an unknown state.
- **Four kinds of action, one setting:**
  - *prediction:* the action is a distribution over future observables;
  - *estimation:* the action is a value of a parameter, judged later;
  - *science:* the action is a claim about a hidden quantity, whose utility is the
    future value of the discovery;
  - *control:* the action changes the world, and the utility is future reward.
- **The sequential case is the general one.** Choosing a parameter now for a later
  use is choosing an action now for a later reward. RL, viewed this way, contains the
  others. Prediction is RL with one step and the log score. Experiment design is RL
  where the reward is information.
- **Prediction, then averaging.** When the utility depends on the world only through
  future observables, a decision splits into two steps: form the predictive
  distribution, then choose the action that maximises the expected utility under it.
  The predictive distribution is sufficient for every such decision, which is why
  the log score is the default when the use is left open (Bernardo 1979). When the
  utility depends on a hidden quantity (science), prediction is not enough.
- **Where this view comes from.** Statistics has held it since Wald. ML often
  does not state it.
  *Citations to settle:* Wald (1950), *Statistical Decision Functions*; Savage
  (1954); Raiffa & Schlaifer (1961); DeGroot (1970), *Optimal Statistical
  Decisions*; Berger (1985); Bernardo & Smith (1994), ch. 2–3; Lindley (1956) and
  Chaloner & Verdinelli (1995) for design. Bridging work in ML: Duff (2002)
  Bayes-adaptive MDPs; Ghavamzadeh et al. (2015) Bayesian RL survey; Hutter (2005)
  AIXI; Levine (2018) control as inference; Silver et al. (2021) "Reward is
  enough"; Russo & Van Roy (2014) information-directed sampling.
- **Ledger line:** given: the problem and utility; chosen: what counts as an action.

## §2 The space of distributions and what lives on it (~2,000 words)

**Goal:** name the structures on Δ(𝒳) before parameters appear.

- **Records and q.** One paragraph: a measurement system fixes the records;
  exchangeability gives a fixed law q (de Finetti; Diaconis–Freedman). For sequences
  and actions, q is a kernel. Detail goes to the Foundations companion.
- **Structures on Δ:**
  - *convexity:* mixtures, so averaging happens here;
  - *divergences:* KL and f-divergences; proper scoring rules (Gneiting & Raftery
    2007), with KL as the regret of the log score;
  - *the Fisher–Rao metric:* the unique metric invariant under sufficient
    statistics (Čencov 1982);
  - *observables:* functionals on Δ (means, quantiles, conditional expectations,
    the Bayes action of a decision problem). Utilities are observables of the
    joint (action, outcome) law.
- **What q contains.** Regularity and noise, hidden structure (latent variables,
  states), and the predictive state: the minimal summary of the past that is
  sufficient for the future (causal states, Shalizi & Crutchfield 2001; predictive
  state representations, Littman, Sutton & Singh 2002).
- **For an LLM:** q is a law on documents; the predictive state of text is
  whatever a perfect next-token predictor would need to track.

## §3 The model map and its pullbacks (~2,500 words, the core)

**Goal:** show what each structure on Δ becomes on W, and which become invariant.

- **The map.** Parametrisation followed by an output map; image, fibres,
  identifiability; all models are wrong, so the target is the KL projection p₀.
  Heights ladder.
- **Pullbacks, one at a time:**
  - utility → expected loss L(w); KL → K(w); data → L_n(w);
  - Fisher–Rao → Fisher information F(w), degenerate where the map loses rank
    (singular models); λ named as the invariant that replaces the parameter count
    there (Watanabe, cited in one paragraph);
  - observable f → f∘p: what the model predicts about a quantity.
- **Hidden quantities and identifiability.** A scientific quantity is a function
  θ(w) on W. It can be learned from data only if it is constant on fibres, so that
  it factors through p. Heights: the mean difference between components is
  identifiable up to label switching, and not at all when the components merge.
  Recording sex makes it a conditional observable.
- **The principle with examples.** Pulled-back quantities are invariant under
  reparametrisation; the rest are coordinate facts:
  - Fisher information vs Hessian sharpness (Dinh et al. 2017);
  - MLE (equivariant) vs MAP (not);
  - the predictive mean of p_w (invariant) vs the posterior mean of w (not).
- **For an LLM:** the image is a tiny subset of Δ; fibres come from permutations
  and rescalings; "a feature" is a candidate observable, and it is meaningful only
  if it is constant on fibres.

## §4 Learning as a process on W, seen on Δ (~1,800 words)

**Goal:** name what a learning algorithm is and what it returns.

- **Solution concepts** as measures on W: MLE and MAP (point masses), the Bayes
  posterior (a Gibbs measure), the law of an SGD run.
- **Push forward to Δ.** A measure on W becomes a measure on distributions;
  averaging it gives the predictive, which can lie outside the image. Prediction,
  then averaging, again: first form the push-forward, then average in Δ, not in W.
- **Convergence is a statement in Δ:** towards p₀, at a rate governed by the
  pulled-back structure (regular vs singular; cited).
- **Three errors** (Bottou–Bousquet) located on the diagram: approximation (q to
  the image), estimation (L_n vs L), optimisation (where the process stops).
- **Evaluation:** held-out loss estimates the pulled-back utility, and is unbiased
  only under train/test exchangeability.

## §5 Science, prediction and control in one framework (~2,000 words)

**Goal:** show the differences between fields as differences in the action and in
the functional the utility asks about, not in the mathematics.

- **Science:** the utility is on a hidden quantity θ; identifiability and
  experiment design are central; the parameter is the answer.
- **Prediction:** the utility is on observables; the parameter is a means; the
  predictive distribution is the answer.
- **Control:** the utility is future reward; the policy is the answer.
- **Yet the learned model captures hidden structure anyway.** Systems trained only to
  predict or to win represent features of the hidden state:
  - AlphaZero learns human chess concepts (McGrath et al. 2022, PNAS; Schut et al.
    2023);
  - Othello-GPT represents the board (Li et al. 2023; Nanda et al. 2023);
  - chess-playing language models estimate latent variables (Karvonen 2024);
  - transformers represent belief-state geometry (Shai et al. 2024).
  The reason, stated through §2: optimal prediction or control requires the
  predictive state, and the predictive state is a functional of the hidden
  structure.
- **What this means:** the line between science and engineering is which functional
  of q the utility requires, and whether the map makes it identifiable. Reading
  hidden structure out of a trained model (interpretability) is estimation of an
  observable pulled back to W.

## §6 A field guide (~1,000 words + one table)

The table a reader keeps. One row per quantity people compute:

| Field term | Object in the framework | Lives on | Pulled back / pushed forward | Invariant under reparametrisation? |
|---|---|---|---|---|
| cross-entropy, likelihood | L_n, pullback of the log score | W | pulled back | yes (as a function of p_w) |
| Fisher information, Gauss–Newton | pullback of Fisher–Rao | W | pulled back | yes (as a tensor) |
| Hessian sharpness | second derivative in coordinates | W | — | no |
| weight decay, weight norms | structure on W only | W | — | no |
| posterior predictive, ensemble | average of the push-forward | Δ | pushed forward | yes |
| posterior mean of w | average on W | W | — | no |
| calibration | observable on the (prediction, outcome) law | Δ | — | yes |
| value function, return | utility of a policy | Δ (of trajectories) | pulled back to policy parameters | yes |
| probe, concept vector | estimator of an observable on W | W | candidate pullback | only if constant on fibres |
| AIC / BIC / WAIC | estimators of pulled-back utility | W | — | WAIC yes; BIC counts parameters |
| identifiability | θ constant on fibres | W | — | — |

Rows to be checked one by one.

## §7 Toward Part 2 (~600 words)

What naming the structures makes possible:
- **Equivalent learning machines:** different maps with the same image and
  pullbacks are the same learner seen in different coordinates.
- **Only invariants are worth measuring:** for diagnostics, interpretability and
  safety arguments.
- **The learning process itself:** what can be reached, what is selected among
  equally good solutions, and the role of the optimiser (where v3's material goes).
- Open questions in one list.

---

## What moves out

- v3 §§2, 4, 5 (equivalence group, invariants of the dynamics, control bundles)
  → Part 2. The control-theory dictionary is kept as notes, not as a frame.
- v1 Part I foundations detail → Foundations companion (`note.tex` §§1–5).
- Full Watanabe treatment → citations in §3–4 only.
- HMM, cat, robotics template → parking lot.

## Decisions needed

1. **"Prediction, then averaging."** I read this as two linked splits: decision =
   predictive distribution, then expected utility under it; and Bayes =
   push-forward to Δ, then average there. Is that what you meant, or something else?
2. **Hidden-structure example for §5:** chess (AlphaZero; closest to your framing),
   Othello-GPT (cleanest evidence), or a transformer on HMM-generated sequences
   (belief states; links to the toy-transformer work on this branch)?
3. **The science example:** keep it on the heights data (sex difference as θ), or use
   a separate case where the parameter is the answer?
4. **Citations for "statistics knew, ML often doesn't."** Is the list in §1 the right
   starting set, or do you have a preferred source?
