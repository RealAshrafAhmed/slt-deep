# Outline v2: The Geometry of Learning Machines

*A short introduction.* (Alternatives: *Learning as Geometry*; *The Shape of a
Learning Problem*; *A Short Introduction to the Geometry of Learning*.)

Proposed replacement for `outline.md`. That outline is a plan for a monograph:
five parts, about 40 pages, and a template applied to four examples. This one is a
plan for an essay that can be read in under an hour (about 12,000 words), or as six
posts of about 2,000 words each. The model is a short introduction in the style of
the geometric control theory primers. Those primers take a field that engineers know
well, name the one geometric object underneath it, show what is invariant, and show
what the change of view buys. They are not surveys.

## The one object

A learning machine is a map from a finite-dimensional parameter space to
predictions:

  w ∈ W  ↦  p_w ∈ Δ(𝒳)

Paired with the unknown law of the data q, this map induces one function on W:

  K(w) = KL(q ‖ p_w)

**The note is about the shape of K on W.** Training is motion on this landscape.
Model selection measures its volume. Model building chooses its shape. Approximation
asks how far its floor sits above zero. The four practices usually get four separate
theories; here they are four readings of one geometric object.

## Thesis, in one paragraph (the reader should be able to repeat it)

Everything a learning machine can do is decided by the shape of one function on
its parameter space. This shape is almost never a bowl. It has flat valleys created
by redundancy and pinched points where descriptions collapse (singularities), and
these features, not the parameter count, govern how the machine trains, how much
data it needs, and which solutions it prefers. Mathematics has mature tools for such
shapes: algebraic geometry, gauge symmetry and statistical mechanics. Learning theory
has only begun to use them. If we want machines that are both powerful and safe, we
need to read this geometry rather than guess at it.

## What changes from v1

| v1 | v2 | Why |
|---|---|---|
| Five parts, ~40 pp | Prologue + six sections + epilogue, ~12k words | One sitting, or a six-post series |
| Part I builds the setup in 7 sections | Setup is one section (§1), told as a single chain | The reader needs the landscape, not the foundations of probability |
| Ledger (given / chosen / assumed / approximated) is the thesis | Ledger is a recurring device: one line at the end of each section | It is a discipline, not the payoff |
| Geometry is one of several parts | Geometry is the spine; every section is a question about K on W | This is the "teeth" |
| Four applications in Part IV with a template | Height ladder throughout; the LLM returns at the end of each section | One example that grows beats four that repeat |
| Connections to other theories in a final table | Parallels with physics and mathematics placed where they help; one table at the end | They motivate the view, so they belong in the argument |
| Safety absent | §6 says why the geometry matters for safe and capable systems | This is the motivation |

## Form

- Each section opens with **one question** a practitioner already asks, and closes
  with **what the geometry answers**.
- Each section has **one figure** of W, drawn for the height ladder. Figures carry
  the argument. A reader who skims the figures and captions should get the thesis.
- Each section has **one parallel**: a box of three or four sentences naming the
  physics or mathematics object that has the same structure, labelled as an
  analogy where it is one.
- Each section ends with **one ledger line**: what was given, chosen, assumed or
  approximated in this step.
- **No proofs.** Every claim is a cited theorem, a definition, or flagged as a
  conjecture. Equations appear only when they are the clearest sentence.

## Running example: one dataset, a ladder of maps

Adult heights, as recorded (NHANES, binned). The data never changes; only the map
does.

| Rung | Map W → Δ(𝒳) | What the landscape shows |
|---|---|---|
| Categorical over bins | saturated simplex | a bowl; floor at zero; one parameter per bin |
| Gaussian | 2 parameters | a bowl; floor above zero (wrong but useful) |
| Two-Gaussian mixture | 5 parameters | a valley with a pinch: label switching (symmetry) and merging components (singularity) |
| Mixture with sex recorded | conditional model | the pinch disappears: singularity belongs to the map and to what is recorded, not to the world |

The horizon example is a transformer on text: the same objects at scale, with no
rung where the map is right. It appears at the end of each section in one
paragraph: "and for an LLM…".

(The HMM / accelerometer example and the 2 × 2 generator design move to a follow-up
post; see Parking lot.)

---

## Prologue: a trained network is a point (~800 words)

- Open with a concrete scene: a trained model is a vector of numbers. Two very
  different vectors can give the same model; two nearby vectors can give very
  different ones. Which properties belong to the vector and which to the model?
- Introduce the map w ↦ p_w and the landscape K as the one object.
- State the thesis paragraph.
- Name the inspiration honestly: the geometric view of control systems
  turned a collection of recipes into a theory by finding the right space and its
  invariants. Learning has the space (W) and the function (K); the invariants are
  known in part (Fisher metric, RLCT) and are not yet standard tools.
- Roadmap of the six questions.

**Figure 0.** The map from W to the space of distributions: a fibre collapses to a
point, the image is a curved subset, and q sits off it.

## §1 Where does the landscape come from? (~2,000 words)

**Question:** "My loss is cross-entropy. Why that, and what is it measuring?"

One chain, each step one paragraph, each with its status tag:

1. **Problem → utility** (given). Acting or predicting; the log score when the
   use of the prediction is left open (Bernardo).
2. **Records → q** (chosen, then assumed). A measurement system fixes what is
   recorded. Exchangeability (designable, testable) gives a fixed law q (de Finetti,
   Diaconis–Freedman). One paragraph; the rest goes to a footnote.
3. **Model → map** (chosen). Parametrisation followed by an output map; its image is
   a curved subset of Δ(𝒳). All models are wrong: the target is the KL projection
   p₀, not q.
4. **Utility → loss** (derived). Log-score regret is KL; L(w) = H + K(w). The data
   give L_n, an estimate of L.
5. **Three errors** (Bottou–Bousquet) as three facts about the landscape:
   approximation = height of the floor K(w₀); estimation = distance between the
   landscapes L_n and L; optimisation = where the algorithm stops.

**Answer:** cross-entropy is the unique loss that measures distance to q itself,
and K is the only place q enters. Everything else on W is chosen.

**Figure 1.** The height ladder: four images in Δ(𝒳) and the KL distance from q to
each.

**Parallel.** Energy function on configuration space. The loss plays the role of an
energy; the parameters are coordinates on a configuration space; q is the
environment that fixes the energy.

**Ledger line.** Given: utility. Chosen: records, map. Assumed: exchangeability.

*Cut from v1 Part I:* probability as a chosen language, regularity and noise,
world symmetries, Kolmogorov material. These were the distracting parts; they become
one remark and citations, or the companion "Foundations" post.

## §2 What does the landscape look like? (~2,500 words, the core)

**Question:** "Why does my model have so many equally good solutions, and does it
matter?"

- **The floor W₀.** The set of best parameters is a fibre of the map over p₀. It is
  a point only for identifiable maps. Relatively finite variance (Watanabe) is the
  condition under which every point of W₀ gives the same prediction, even when the
  model is wrong.
- **Two sources of flatness:**
  - *Symmetry* (p∘φ = p): label switching in the mixture; permutation of hidden
    units; ReLU rescaling; softmax shift. The quotient W/~ is the real parameter
    space.
  - *Degeneracy*: the mixture whose components merge, where the mixing weight is no
    longer determined. Here the floor is not a smooth valley; it is pinched.
- **Singularities.** Where the floor self-intersects or pinches, the map loses
  rank and the Fisher information is degenerate. These maps are the rule, not the
  exception: every neural network is singular (Watanabe; Wei et al.).
- **The height ladder walks through it:** a bowl (categorical), a bowl with a raised
  floor (Gaussian), a pinched valley (mixture), a bowl again (sex recorded).

**Answer:** yes, it matters. The redundancy is structured, and the structure,
not the count of parameters, is the information about the model.

**Figure 2.** The mixture landscape in two coordinates (mean separation × mixing
weight): the floor as a cross, not a point.

**Parallel.** Gauge symmetry: a redundant description where only the quotient is
physical. Algebraic geometry: singular varieties, and resolution of singularities
(Hironaka) as the way to see them.

**For an LLM.** The embedding × unembedding product, attention head permutations
and layer-norm rescaling all create fibres; the floor of a transformer is a
high-dimensional singular set.

**Ledger line.** All structure in this section is a property of the chosen map,
evaluated against q.

## §3 What doesn't depend on how we write it down? (~1,500 words)

**Question:** "Is a flat minimum a good minimum?"

- **Reparametrisation vs symmetry.** A reparametrisation changes the coordinates
  but not the image; a symmetry moves along a fibre. Only quantities that survive
  reparametrisation are about the model.
- **The smooth case:** Fisher information is a metric (Amari). The MLE is
  equivariant; MAP is not; the natural gradient is invariant, while plain gradient
  descent is not.
- **Flatness is not invariant** (Dinh et al. 2017): the Hessian at a minimum can be
  made arbitrarily sharp by rescaling.
- **The invariant that replaces it:** the learning coefficient λ, read from how the
  volume of near-optimal parameters scales, V(ε) ~ ε^λ (log ε)^(m−1). λ = d/2 for
  regular maps, λ < d/2 for singular ones. It is an effective dimension that
  coordinates cannot change.
- **Height ladder:** the Gaussian has λ = 1; the mixture near its merged floor has a
  λ below 5/2 (cite, or compute numerically and label as such).

**Answer:** "Flat" is a statement about coordinates. The right question is how
much volume sits near the floor, and that has an invariant answer: λ.

**Figure 3.** The same minimum in two parametrisations: Hessians differ, volume
scaling does not.

**Parallel.** General covariance: physical laws are written in invariants of the
metric, not in coordinates. Dimension theory: λ is to a singular floor what
dimension is to a manifold.

**Ledger line.** Invariants are derived; their estimation from data is
approximated (§5).

## §4 What does the geometry buy? Four readings of one landscape (~3,000 words)

The payoff section. One subsection per practice, each 600–800 words, each ending
with a one-sentence takeaway.

### 4a. Optimisation: motion on the landscape
- Gradient flow and SGD as dynamics on W, steered by data order, step size and
  batch size, which act as controls.
- Continuous symmetries give conserved quantities of gradient flow (layer balancing;
  Kunin et al. 2021, Zhao et al.). SGD noise breaks them and pushes along fibres.
- The fibres and pinches shape which part of W₀ is reached, not only whether it is.
- *Takeaway:* the optimiser moves on the landscape you chose; its symmetries
  constrain the path.
- *Parallel:* Noether's theorem; geometric control (reachable sets of a controlled
  flow). State as analogy.

### 4b. Model selection: the volume of the landscape
- Bayes marginal likelihood as a partition function:
  Z_n = ∫ exp(−n L_n(w)) φ(w) dw, with n as inverse temperature.
- Free energy F_n ≈ n L_n(w₀) + λ log n (Watanabe): accuracy against volume. BIC is
  the regular special case (λ = d/2); it over-penalises singular models.
- Generalisation error ≈ λ/n; WAIC as the computable estimate.
- Height ladder: Gaussian vs mixture decided by n L_n + λ log n. At small n the
  Gaussian wins; at larger n the mixture does. This is a phase transition in n.
- *Takeaway:* Occam's razor is a volume, and singular models are cheaper than their
  parameter count says.
- *Parallel:* statistical mechanics: energy vs entropy; phase transitions as a
  change of dominant region in Z_n.

### 4c. Model building: choosing the shape
- Building a model is choosing a map, so it is choosing the landscape.
- Building symmetries in (equivariance) quotients W in advance.
- What is recorded changes the geometry: sex recorded removes the singularity.
- Saturated vs idealised: the categorical has the floor at zero but λ = (bins − 1)/2;
  the Gaussian has a raised floor but λ = 1. Regret under q decides.
- *Takeaway:* architecture and data design are geometric choices with predictable
  consequences.

### 4d. Approximating the unknown
- The floor height K(w₀) is the approximation error, and it is visible only through
  held-out loss.
- The posterior predictive lies outside the image of the map: averaging over the
  floor can approximate q better than any single point.
- *Takeaway:* "all models are wrong" becomes a measurable quantity: the floor height,
  plus a geometry for doing better than any point on it.

**Figure 4.** Free energy of Gaussian and mixture vs n for the height data: the
crossing point.

**For an LLM.** Pre-training is approximately one-pass stochastic approximation of
L itself; scaling curves are statements about floor height and λ; capabilities may
arrive as changes of dominant region (phase transitions), which are observable.

## §5 What can we actually compute? (~1,200 words)

**Question:** "My optimiser returned a vector. Which of the objects above is it?"

- Three things cannot be computed exactly: expectations under q, argmins on a
  non-convex W, and integrals over W in high dimension.
- Approximation that is a choice (deterministic: Laplace, full-batch descent) vs
  approximation forced by scale (stochastic: SGD, SGLD).
- The algorithm returns a random point whose law depends on the algorithm: not the
  MLE, MAP or Bayes estimate exactly.
- The invariants of §3 can be estimated locally around a trained point (local
  learning coefficient; cite and point to follow-up).
- Held-out loss is honest only under train/test exchangeability (back to §1).

**Answer:** you have a sample from an algorithm-dependent distribution on W, and the
geometry tells you what to measure about it.

**Figure 5.** SGD trajectories on the mixture landscape, ending on different parts
of the floor.

**Ledger line.** Everything in this section is approximated; say by what.

## §6 Why this matters for safe and powerful systems (~1,500 words)

**Question:** "Why should anyone building general systems care about the shape of
K?"

- **Structure in W is structure in behaviour.** Fibres and singularities determine
  which functions are easy to reach and which are preferred at a given n. That is a
  handle on what a model will learn before it learns it.
- **Development is observable.** If capabilities appear as phase transitions, they
  show up as changes in invariants (λ) that can be tracked during training
  (developmental interpretability). Label this as a programme, with its evidence.
- **Interpretability needs invariants.** Features read off coordinates can be
  artefacts of the parametrisation; quantities invariant under reparametrisation
  cannot.
- **Alignment changes the landscape.** Fine-tuning and RLHF change the utility,
  and so they change K. The geometry says what the new floor looks like relative to
  the old one.
- **The borrowed toolbox.** Physics and modern mathematics have spent a century on
  exactly these structures: gauge redundancy, partition functions, phase
  transitions, singularities, conserved quantities. The similarity is structural,
  not a metaphor; where it is only an analogy, say so.
- **Honest limits.** Most results are asymptotic in n; estimation of λ at scale is
  young; the link from geometry to safety is a research bet, not a theorem.

## Epilogue: a dictionary (~500 words + two tables)

- **Practitioner ↔ geometry:** training loss = L_n; "a minimum" = a point of a fibre
  of W₀; "capacity loss" = floor height K(w₀); "flat minimum" → volume scaling, λ;
  "effective parameters" = λ; "grokking / emergence" → a candidate phase transition.
- **Learning ↔ physics and mathematics:**

| Learning | Physics / mathematics | Status |
|---|---|---|
| K on W | energy on configuration space | structural |
| fibres from symmetry, W/~ | gauge redundancy, quotient | structural |
| singular floor, λ | singular varieties, resolution, log canonical threshold | theorem (Watanabe) |
| Z_n, n | partition function, inverse temperature | structural |
| model selection in n | phase transition | theorem in examples; conjecture at scale |
| conserved quantities of gradient flow | Noether's theorem | theorem (for gradient flow) |
| Fisher metric, natural gradient | Riemannian geometry, covariance | theorem |
| training as steered flow | geometric control, reachable sets | analogy |

- Further reading, one line each: Watanabe (2009, 2018), Amari (2016), Grünwald
  (2007), Bottou–Bousquet (2008), Dinh et al. (2017), Kunin et al. (2021), and the
  developmental interpretability literature.

---

## Blog-series split (if not published as one essay)

1. *A trained network is a point* — Prologue + §1
2. *The shape of the floor* — §2
3. *What coordinates hide* — §3
4. *One landscape, four practices* — §4a–4b
5. *Building and approximating* — §4c–4d + §5
6. *Why the geometry matters* — §6 + Epilogue

Each post stands alone with a two-sentence recap of the one object at the top.

## Parking lot

Everything from the v1 parking lot stays. Added from v1's body:
- Foundations post: probability as a chosen language, records and measurement
  kernels, exchangeability in depth, regularity vs noise, world symmetries,
  Kolmogorov material (draft `note.tex` §§1–5 is the raw material).
- Sequences: HMM on accelerometer data; the 2 × 2 generator design.
- Worked applications: cat classification, a science example, robotics with actions
  as records.
- Convergence in three layers (pointwise / uniform / fluctuations) in full.

## Decisions needed

1. **Format.** One essay (~12k words) or the six-post series? *Recommendation:*
   write as one essay with the post breaks marked; split at publication.
2. **Title.** *The Geometry of Learning Machines: A Short Introduction*?
3. **Fate of `note.tex`.** Keep §§1–5 as the Foundations companion; compress §§6–7
   into §1 here. *Recommendation: yes.*
4. **Strength of the safety section.** Present §6 as motivation and research
   programme, with every link labelled theorem / evidence / conjecture?
   *Recommendation: yes.*
5. **Figures.** Six figures, all computed on the height data (plus one schematic).
   Compute λ for the mixture numerically, or cite a known value?
6. **HMM.** Drop to the parking lot as proposed, or keep one paragraph in §2?
