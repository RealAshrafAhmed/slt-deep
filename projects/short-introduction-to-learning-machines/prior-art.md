# Prior art for the framing

Search run 2026-10-25 (web-only; ~35 searches; full texts read for Shai et al. 2024,
Lau et al. 2023, Ramstead et al. 2023, Ay & Löhr 2015, Watanabe 2022,
Sakthivadivel 2022). Watanabe's books and Knight 1975 were not re-read. "Not found"
means not found, not "does not exist".

The five points checked (numbering as in `outline-v7.md`):

- **P1–P2** world as histories Ω with a law; observables = measurable functions;
  hidden = not measurable w.r.t. the agent's filtration; learnable macrostate θ
  from a symmetry.
- **P4** meeting space = kernels from conditioned past to predicted future; the
  conditioning law unmodelled, enters as a weight.
- **P5** agent = realization (state space, state → kernel, filter); trivial
  realization S = past (autoregressive transformer); minimal = causal states /
  Knight / Kalman–Lindquist–Picci; "S exists in the world" optional.
- **§3** W → kernels is a parametrisation, not a chart; loss, Fisher, observables
  are pullbacks and so invariant; coordinate-only quantities are artefacts; gauge
  analogy.
- **§5** efficiency: S carries I(past; future) and nothing more; predictive vs
  structural alignment.

## Verdict

Each point has solid prior art on its own. No single source states the
combination. Specifically not found: hidden-ness defined by non-measurability
w.r.t. the agent's filtration *in a learning context* (Ay–Löhr 2015 is closest);
kernels with the conditioning law demoted to a weight stated as a principle (it is
unremarked convention in Hutter and in supervised SLT); a neural architecture
treated as a realization with S = past identified with the autoregressive
transformer; W → kernels as a singular parametrisation with pullbacks *together
with* a world side; the predictive-vs-structural alignment split (Shai et al. and
Löhr–Ay each have half).

The contribution is therefore the assembly, and the note should say so in one
sentence, citing the sources below for each point.

## Ranked closest work

1. **Computational mechanics.** Shalizi & Crutchfield 2001 (J. Stat. Phys. 104;
   cond-mat/9907176); Shalizi & Shalizi 2004 (CSSR; cs/0406011) which states that
   the causal-state construction "is essentially the same as" Knight's prediction
   process, that ε is "the unique minimal sufficient statistic", with a recursive
   filter S_{t+1} = T(S_t, X_{t+1}); Löhr & Ay 2009 "Non-sufficient memories that
   are sufficient for prediction" and "On the generative nature of prediction"
   (generative HMM ≠ minimal predictor; generative complexity behaves well under
   ergodic decomposition); Löhr 2012. Covers P5 and §5 nearly in full; P1 "no
   primitive hidden space" in spirit. Lacks P4 as a principle, all of §3.
2. **Shai, Marzen, Teixeira, Gietelink Oldenziel, Riechers 2024**, NeurIPS,
   arXiv:2405.15943; Piotrowski, Riechers, Filan, Shai 2025, arXiv:2502.01954.
   Belief states of the mixed-state presentation found linearly in the residual
   stream; explicit that "there are infinitely many distinct HMMs that can
   generate the same stochastic process" and that "prediction is substantially
   more complicated than generation". The only work connecting causal-state
   realization to a trained parametric model. Lacks P1 formalisation, P4, §3.
3. **Ay, Jost, Lê, Schwachhöfer 2017** *Information Geometry* (ch. 3,
   parametrized measure models: a differentiable map into measures, not required
   injective or immersive; Fisher and Amari–Chentsov as pullbacks; invariance
   under Markov morphisms); **Lê 2020** "Diffeological statistical models"
   (Mathematics 8(2):167; arXiv:1912.02090) explicitly for singular ML models;
   **Ay & Löhr 2015** "The Umwelt of an embodied agent" (Theory Biosci. 134;
   arXiv:1603.08389): the agent's accessible events as a sub-σ-algebra, kernels,
   and a minimal world model observationally equivalent for the agent. Together
   the cleanest statements of §3 and of P2's definition of hidden; never
   assembled with realization theory or predictive information.
4. **Watanabe 2009, 2018, 2022 (arXiv:2206.05630) + Lau, Furman, Wang, Murfet,
   Wei 2023 (LLC; arXiv:2308.12108).** Singular parametrisation; the learning
   coefficient as a birational invariant of the fibre; Lau et al. App. C: LLC
   "invariant to local diffeomorphism of the parameter space"; Hessian-based
   measures "undermined by their failure to stay invariant" (citing Dinh et al.
   2017). Lacks the world side, realizations, predictive information.
5. **Still 2009** (EPL 85:28005); **Still, Crutchfield & Ellison 2010** "Optimal
   causal inference" (Chaos 20:037111): information bottleneck past → future
   recovers causal states in the zero-distortion limit; **Bialek, Nemenman &
   Tishby 2001** "Predictability, complexity, and learning" (Neural Comp. 13):
   predictive information as the unique complexity measure, with growth rate
   tied to learnable parametric dimension; **Marzen & Crutchfield 2016, 2017**
   (predictive rate-distortion; nearly maximally predictive features in RNNs).
   Covers §5 efficiency. Lacks P4, §3.

## Other relevant work

- **Ramstead, Sakthivadivel, Heins, Koudahl, Millidge, Da Costa, Klein, Friston
  2023** "On Bayesian mechanics" (Interface Focus 13; arXiv:2205.11543);
  Sakthivadivel 2022 (arXiv:2204.11900); critique Biehl, Pollock & Kanai 2021
  (Entropy 23(3):293). Internal states parametrise beliefs over external states
  via a synchronisation map "only defined when the inverse of μ is invertible on
  its image". Assumes a primitive external state space (what P5 leaves optional);
  "gauge" is used for the choice of constraint function, not for parameter
  orbits. One paragraph in §7.
- **Hutter 2005** *Universal AI* (environments as chronological semimeasures on
  full histories: "neither stationarity, nor the Markov property, nor complete
  accessibility"); **Ortega 2011** PhD; **Ortega & Braun 2013** (Proc. R. Soc. A
  469); **Ortega, Braun, Dyer, Kim, Tishby 2015** (arXiv:1512.06789). Kernels on
  histories; the trivial realization; no symmetry or minimality.
- **Lauritzen 1988** *Extremal Families*; **Orbanz & Roy 2015** (IEEE TPAMI 37:
  symmetry ⇒ ergodic decomposition ⇒ parameter = extremal family); **Al-Najjar &
  Shmaya 2014** "Learning the ergodic decomposition" (arXiv:1406.6670); **Shalizi
  2009** (EJS 3: posterior convergence for dependent data, all hypotheses wrong).
  Sources for P3.
- **Stochastic realization theory.** Akaike 1974 "Stochastic theory of minimal
  realization" (IEEE TAC 19) and 1975 (SIAM J. Control 13, canonical variables);
  Faurre 1976; Picci 1976 (Proc. IEEE 64); Lindquist & Picci 1979 (SIAM J.
  Control Optim. 17), 1985 (SIAM J. Control Optim. 23), 2015 *Linear Stochastic
  Systems* (Springer): minimal Markovian splitting subspace = "subspace of
  smallest size containing all the information from the past needed in
  predicting the future", minimality "by subspace inclusion rather than by
  dimension", internal vs external realizations, forward predictor space;
  Ruckebusch 1976/1980; Knight 1975 (Ann. Prob. 3) and 1992 *Foundations of the
  Prediction Process*; Jaeger 2000 (OOMs: "many different OOMs generate the same
  process"); Littman, Sutton & Singh 2002 (PSRs); Thon & Jaeger 2015 (JMLR);
  Van Overschee & De Moor 1996 (subspace identification); Defourneau & Petreczky
  2019 (realization theory of RNNs, arXiv:1903.05609). Sources for P5. *Dates
  before 1990 from memory; verify.*
- **Reparametrisation and quotients in deep learning.** Dinh, Pascanu, Bengio &
  Bengio 2017; Kristiadi, Dangel & Hennig 2023 (arXiv:2302.07384); Amari, Park &
  Ozeki 2006 (Neural Comp. 18); Sun & Nielsen 2019 (lightlike neuromanifolds);
  Puechmorel 2023 "Pullback bundles and the geometry of learning" (Entropy 25,
  not read); Dong & Cheng 2026 (arXiv:2603.21502: "realization map", orbits,
  quotient geometry); a 2026 complete symmetry classification of shallow ReLU
  networks (arXiv:2604.14037). Sources for §3; none has a world side.
- **Categorical probability.** Fritz 2020 (Adv. Math. 370); Cho & Jacobs 2019;
  Jacobs *Structured Probabilistic Reasoning*; Fritz, Gonda & Perrone 2021 (de
  Finetti as a categorical limit, arXiv:2105.02639); Culbertson & Sturtz 2014;
  Smithe 2021 (Bayesian lenses; "dynamical realisation" of an open game,
  arXiv:2109.04461, 2101.10483); Capucci, Gavranović, Hedges & Rischel 2021
  (arXiv:2105.06332); Gavranović et al. 2024 survey (arXiv:2408.14014); Lê
  "Probabilistic morphisms and Bayesian supervised learning" (Mat. Sbornik, not
  read). Kernels as morphisms; abstract realizations; no minimality, no
  singular pullbacks, no predictive information.
- **Instrumentalism / prequential.** Dawid 1984 (JRSS-A); Dawid & Vovk 1999
  (Bernoulli); Rissanen 1989; Grünwald 2007; Grünwald & Kotłowski 2010
  (arXiv:1002.0757); Shafer & Vovk 2019 (Forecaster vs Reality, no actions).
- **Biehl 2017** PhD (arXiv:1704.02716): deriving rather than assuming the
  agent/world split. Tangential.

## Not found

No "learning as realization" paper in the P5 sense; no learning-theory text
defining hidden by non-measurability w.r.t. the agent's filtration other than
Ay–Löhr; no principle-level statement of P4; no work combining computational
mechanics with SLT or pullback invariance (Timaeus/Hoogland HMM in-context
learning work did not surface; may exist unpublished); no explicit
predictive-vs-structural alignment dichotomy.
