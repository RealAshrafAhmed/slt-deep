# Outline v3: Learning Systems up to Change of Coordinates

*A short introduction to the geometry of learning.* (Working title. Alternatives:
*Invariants of Learning*; *A Geometric View of Learning Machines*.)

Replaces v2. In v2 the subject shrank to the shape of K on W, and the substance
became a retelling of Watanabe. In v3 the subject is the **learning system**: a
map, plus the dynamics that move along it, studied up to the transformations that
should not matter. The claim is that the **invariants of this system govern
learning**. Watanabe's λ is one of these invariants. It is cited, not developed.

Length target as in v2: under an hour (~12k words) or a six-post series.

## The model: what geometric control theory did

Nonlinear control was a set of case-by-case recipes until a change of view in the
1970s (Hermann–Krener, Sussmann, Jurdjevic, Brockett). In that view a control
system is a family of vector fields on a manifold. Two systems are treated as the
same when they differ by a change of state coordinates and a state-dependent
relabelling of the controls (**feedback equivalence**). The questions people care
about then become questions about invariants:

| Question | Invariant |
|---|---|
| Where can the system go? | the Lie algebra of the vector fields (accessibility, Chow–Rashevskii) |
| What can we tell apart from the outputs? | observability rank and indistinguishable states |
| Which path is optimal? | Pontryagin structure, curvature (Agrachev) |
| When are two systems the same? | feedback invariants |

A trained learner already has all of this structure:

| Control theory | Learning |
|---|---|
| state x ∈ M | parameters w ∈ W |
| controls u | data points z (or minibatches) |
| vector fields f(x, u) | update fields V_z(w) = −G(w)⁻¹ ∇ℓ(z, w) |
| output y = h(x) | the model map p: W → behaviours (predictive distributions, policies) |
| indistinguishable states | fibres of p |
| feedback equivalence | reparametrisation of W and a change of metric or preconditioner G |

**The note sets up this dictionary and shows what follows from it.** Where it
reproduces known results (natural gradient, flatness, conserved quantities,
implicit bias, λ, MDP homomorphisms), it shows each one as a statement about an
invariant. Where it does not, it states the question this view raises.

## Thesis

A learning system has three parts: a **map** from parameters to behaviour, a
**metric** that turns losses into motion, and **data** that steer that motion.
Learning happens in the map, which is where coordinates live. But what a system
can represent, identify, reach and select, and how much data it needs to do so,
is governed by quantities that do not depend on those coordinates. Naming the
invariants, and keeping each separate from the choices that are not invariant, is
the programme.

Three consequences that give the note substance beyond prior work:

1. **Data steer only across fibres.** Every update field lies in the image of
   G⁻¹Jᵀ, where J is the Jacobian of the map. First-order learning therefore
   cannot move along the directions the output does not see. Motion along fibres,
   which selects among equally good solutions, comes from second-order effects:
   noise, finite step size and the commutators of data fields. *Implicit bias is a
   property of the pair (map, metric), not of the model.* This places Smith et al.
   (2021) and Li, Wang & Arora (2022) in one picture.
2. **Learning is open loop in supervised learning and closed loop in RL.** In
   supervised learning the controls z are exogenous (exchangeable draws from q).
   In RL the data a learner sees depend on its own state through its policy. The
   environment is itself a control system, with a **control bundle** of actions
   over states, and a policy is a section of that bundle. RL is therefore two
   control systems in feedback. Feedback equivalence of the environment is the
   continuous form of MDP homomorphisms and bisimulation (Ravindran–Barto; Ferns,
   Panangaden & Precup).
3. **Invariants are what is worth measuring.** Any diagnostic read off coordinates
   (Hessian sharpness, individual weights, attention scores) can be changed by a
   reparametrisation that leaves behaviour fixed. Monitoring, interpretation and
   safety arguments should be built on invariants. This is the link to safe systems.

## The equivalence group (the organising table)

Each quantity in the note is tagged with what it is invariant under:

| Transformation | Acts on | Example | Invariant under it |
|---|---|---|---|
| Reparametrisation φ: W′ → W | coordinates | change of units, normalisation | image, fibres up to φ, Fisher metric, λ |
| Symmetry of the map (p∘φ = p) | moves along fibres | permutation, ReLU rescaling, label switching | the behaviour; quotient W/~ |
| Change of metric G ("feedback") | the update fields | SGD vs natural gradient vs Adam | the image of the update fields only up to G; trajectories change |
| World symmetry (acting on Z and on behaviours) | data and outputs | translation, SE(3), relabelling of actions | equivariant maps; MDP quotient |

Plain gradient descent fixes G = I in one coordinate system. It is the one choice
that is *not* invariant, and much of what is called "the inductive bias of deep
learning" comes from this choice.

## Running examples

Kept small and natural. Each one carries a single idea.

| Example | Where | Carries |
|---|---|---|
| Adult heights: Gaussian and two-Gaussian mixture | §2–3 | image, fibres, label switching, merging components (singular point); identifiability = observability |
| A factorised layer (embedding × unembedding, or a low-rank fit to real data) | §3–4 | a continuous symmetry; conserved quantity of gradient flow; drift along the fibre under SGD |
| Pendulum or cart-pole | §5 | a classic control system; control bundle; policy as section; closed loop |
| Transformer on text | end of each section | the same objects at scale |

---

## Prologue (~1,000 words)

- The scene: two very different weight vectors give the same model; one change of
  units makes a sharp minimum flat. Which facts are about the learner and which are
  about the coordinates?
- What geometric control theory did for control: find the space, the
  equivalence, the invariants. The first table above.
- Thesis and the three consequences.

## §1 The learning system (~1,800 words)

Setup, compressed from v1 Part I into one chain with ledger tags:

- Problem and utility (given) → records and q (chosen, assumed: exchangeability)
  → map p: W → behaviours (chosen) → loss ℓ(z, w) (derived: log score).
- New element: the **dynamics**. A metric G (chosen) turns each data point into a
  vector field V_z on W. The training algorithm is a rule for choosing controls z_t
  (sampling, batching, order) and step sizes.
- The learning system is the triple (p, G, data process). Name the three spaces:
  behaviours B, parameters W, data Z.

**Figure 1.** W → B with a fibre collapsing to a point; one data field V_z drawn
on W.

## §2 What counts as the same system (~1,500 words)

- The equivalence group table above, built up one row at a time with an example
  each.
- Reparametrisation vs symmetry: new coordinates vs motion along a fibre.
- Changing the metric is feedback: same map, same data, different motion.
- What is invariant and what is a choice. Dinh et al. (flatness is not invariant)
  as the warning case; the natural gradient as the invariant case (Amari).

## §3 Invariants of the map: what can be represented and told apart (~2,000 words)

- **Image:** what can be represented; distance from q to the image = approximation
  error (all models are wrong; KL projection p₀).
- **Fibres = indistinguishable states:** identifiability is observability of the
  output map. Heights: label switching (symmetry) and merging components
  (degeneracy).
- **Rank and singularities:** where the map loses rank, the local degrees of freedom
  drop. Every network with hidden units has such points.
- **Volume near the optimum:** λ governs Bayesian selection and sample
  requirements (Watanabe). One paragraph and one citation: it is the invariant of
  the map under the Bayesian learning rule, one row among several.

**Governs:** representation, identification, effective dimension.

## §4 Invariants of the dynamics: what can be reached and what is selected (~2,500 words, the new core)

- **The learning distribution.** The span of the update fields at w lies in
  G⁻¹ range(Jᵀ). First-order learning moves only across fibres.
- **Brackets and drift.** Finite steps and noise produce motion along commutators
  [V_z, V_z′] and along the noise covariance. On a fibre of optima this gives a slow
  drift that selects a solution (Li, Wang & Arora 2022; Smith et al. 2021 modified
  loss). Data order matters through commutators.
- **Conserved quantities.** A continuous symmetry of the loss that also preserves G
  gives a conserved quantity of gradient flow (layer balancing; Kunin et al. 2021).
  Change G and the conservation law changes, so these are invariants of (p, G), not
  of p.
- **Accessibility.** The reachable set from initialisation under data controls:
  which parts of W₀ can be reached at all?
- Factorised-layer example: conserved a² − b² under gradient flow; drift under SGD;
  both depend on the metric.

**Governs:** which of the equally good solutions you get, and why the optimiser is
part of the model.

*Open question to state (not claim):* is the effect of data order a holonomy, a net
displacement along a fibre after a loop of controls, with the commutators of the data
fields as its curvature?

## §5 Control bundles: learning to act (~2,000 words)

- The environment as a control system: states S, a **control bundle** A → S (the
  admissible actions at each state), dynamics.
- A policy is a section of the bundle of action distributions; a parametrised policy
  is a map W → sections. The learner's map now lands in a space of sections.
- **Closed loop:** the policy decides which data the learner sees. Two control
  systems in feedback; the ignorability and positivity conditions of v1 become
  conditions on this loop.
- **Feedback equivalence of environments:** state-dependent relabelling of actions
  and change of state coordinates. Discrete case: MDP homomorphisms; metric case:
  bisimulation metrics. Invariants of the environment govern transfer and sample
  efficiency; equivariant policies are maps compatible with the group.
- Pendulum or cart-pole example; then RLHF as a closed loop over text.

**Governs:** what a learner must represent, what it can safely ignore, and when
learning to act differs from learning to predict.

## §6 What the invariants buy (~1,500 words)

One short subsection each, answered as "which invariant decides this":

- **Optimisation:** the learning distribution and its brackets.
- **Model selection:** volume invariants (λ) and held-out regret.
- **Model building:** choosing the map, metric and recorded variables is choosing
  the invariants. Recording sex removes the heights singularity; equivariance
  quotients W in advance.
- **Approximating the unknown:** distance from q to the image; averaging over a fibre
  of optima can leave the image.
- **Safe systems:** measure invariants, not coordinates; closed-loop learners need
  control-theoretic guarantees (reachability, observability), not only statistical
  ones. Mark what is theorem, evidence or conjecture.

## §7 Open questions (~800 words)

The substance a reader takes away and later papers pick up:

1. A complete set of invariants for (map, metric) under feedback equivalence. Which
   ones predict generalisation?
2. Data order as holonomy; curriculum as a choice of control path.
3. Accessibility of the optimal set: are some solutions unreachable by first-order
   learning from typical initialisations?
4. The right group in the singular case (birational, measure-preserving,
   predictive-preserving maps).
5. Control bundles with learned state spaces: when the agent's state is itself the
   output of a learned map.
6. Conservation laws as a route to structure, as in physics.

## Epilogue: dictionary (~500 words)

Three columns: practitioner term ↔ learning-system object ↔ control or physics
counterpart, each tagged theorem / structural / analogy.

---

## What moves out

- v1 Part I detail (probability as a language, regularity, world symmetries,
  Kolmogorov) → a Foundations companion; `note.tex` §§1–5 is the raw material.
- Full Watanabe treatment (convergence layers, WAIC, free energy derivation) →
  citations in §3 only.
- HMM / accelerometer, cat, science, robotics template → parking lot.

## To verify before writing (add to `verification.md`)

- "Control bundle" terminology and the fibre-bundle formulation of control systems
  (Sussmann; Bullo & Lewis 2005; Jurdjevic 1997).
- Exact statements of Li, Wang & Arora (2022) and Smith et al. (2021), and whether
  the commutator form of the data-order effect is already published.
- Conserved quantities under non-Euclidean metrics (Kunin et al. 2021; Zhao et al.
  2023; Marcotte et al. 2023).
- The continuous analogue of MDP homomorphisms (van der Pol et al. 2020; Panangaden
  et al. 2024) and its relation to feedback equivalence.

## Decisions needed

1. **Is the triple (map, metric, data process) the right object?** Or should the
   metric stay implicit, with the note focused on the map and its fibres?
2. **How far into RL?** One section (as here), or a parallel half of the note?
3. **RL example:** pendulum / cart-pole (classic GCT, physical), or something with
   natural data?
4. **Holonomy:** state it as an open question (as here), or work it out on the
   factorised layer before writing?
