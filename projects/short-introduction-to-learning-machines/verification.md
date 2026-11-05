# Verification list

Claims cited from memory. Each must be checked against the source before the
text is final. Status: `open` / `checked` / `corrected` / `dropped`.

## In note.tex (Part I)

| Claim | Source | Status |
|---|---|---|
| Diaconis–Freedman finite bound: 2ck/n (c-element space), k(k−1)/n general; norm convention for TV | Diaconis & Freedman 1980, Ann. Probab. 8(4) (abstract checked via search; norm convention open) | open |
| de Finetti for standard Borel spaces, uniqueness of μ, a.s. convergence of empirical measures to Q | Hewitt & Savage 1955; Kallenberg 2005 Thm 1.1 | open |
| Kolmogorov extension, Birkhoff, ergodic decomposition as stated | Kallenberg 2002 | open |
| Adaptive actions depending only on past records keep likelihood inference valid (ignorability) | Rubin 1976; Berger & Wolpert 1988 | open |
| Conformal test martingales test exchangeability online with validity | Vovk, Nouretdinov & Gammerman 2003 | open |
| Block entropy H(X_{1:L}) ≈ E + hL, E = I(past; future) | Crutchfield & Feldman 2003; Bialek, Nemenman & Tishby 2001 | open |
| Shannon entropy = expected Kolmogorov complexity up to O(K(P)) | Grünwald & Vitányi 2003 | open |
| Structure function / algorithmic sufficient statistic statements | Gács, Tromp & Vitányi 2001; Vereshchagin & Vitányi 2004 | open |
| Invariant conditional ⇔ Y = f(η, M(X)); hypotheses (compact G, standard Borel); theorem number | Bloem-Reddy & Teh 2020 | open |
| Strict generalisation benefit of invariant/equivariant models: exact setting | Elesedy & Zaidi 2021 | open |
| Exchangeable sequence uniform on type class given type; log\|T\| ≈ nH | Csiszár 1998; Cover & Thomas ch. 11 | open |
| Definition of regular model (identifiable, smooth, Fisher positive definite) matches Watanabe's | Watanabe 2009 ch. 1 | open |
| Log score: unique smooth, proper, local utility for reporting distributions; conditions (e.g. \|Y\| > 2) | Bernardo 1979 | open |
| Expected-information criterion for experimental design | Lindley 1956 | open |
| Bernardo framed the log score as the utility of pure scientific inference | Bernardo 1979 | open |
| Pinsker TV ≤ sqrt(KL/2) with TV = sup_A \|q(A) − p(A)\| and KL in nats | Cover & Thomas Lemma 11.6.1 | open |

## Discussed, not yet in the text

| Claim | Source | Status |
|---|---|---|
| Probability of non-stochastic strings is negligible under any computable source | Vereshchagin & Shen 2017 survey | open |
| No universal pointwise estimation of conditional probabilities for stationary ergodic processes | Bailey 1976; Ryabko 1988 | open |
| Universal prediction in the Cesàro sense for stationary ergodic processes | Algoet 1992 | open |
| Where Watanabe discusses de Finetti | Watanabe 2009 or 2018 | open |
| Rotation invariance for all n ⇒ scale mixture of i.i.d. Gaussians | Freedman 1963; Kingman 1972 | open |
| Symmetry + sufficiency ⇒ exponential families (exact statement) | Lauritzen 1988; Diaconis & Freedman 1984 | open |
| Maximal invariant sufficient for family of G-invariant laws | Hall, Wijsman & Ghosh 1965 | open |
| Conditioning not computable in general; computable with suitable densities | Ackerman, Freer & Roy, J. ACM 2019 | open |
| Finite-set models ↔ computable distributions, equivalent up to log terms | Gács–Tromp–Vitányi 2001; Vereshchagin–Vitányi 2004 | open |
| Predictive information grows as (d/2) log T for regular d-parameter families | Bialek, Nemenman & Tishby 2001 | open |
| Statistical complexity C_μ ≥ excess entropy E | Shalizi & Crutchfield 2001 | open |
| Brudno: K(x_{1:n})/n → h a.s. for stationary ergodic processes | Brudno 1983 | open |
| BabyLM 10M (strict-small): size, source composition, document boundaries, license; 2023 vs 2024 edition | Warstadt et al. 2023; BabyLM 2024 call | open |
| Transformers trained on Markov-chain data (statistical induction heads; attention on Markov inputs) | Edelman et al. 2024; Makkuva et al. 2024 | open |
| HMM observation process is generally not finite-order Markov | standard; e.g. Ephraim & Merhav 2002 survey | open |
| Transformers trained on HMM data linearly represent belief-state (mixed-state) geometry | Shai et al. 2024 | open |
| Human height is not bimodal; equal-weight, equal-variance two-normal mixture is bimodal iff mean gap > 2σ | Schilling, Watkins & Watkins 2002 | open |
| NHANES public height/sex data | CDC NHANES | open |
| Secular trend in adult height by birth cohort | NCD-RisC 2016 (eLife) | open |
| UCI HAR dataset: smartphone accelerometer, labelled activities | Anguita et al. 2013 | open |
| HMMs for animal movement behaviour states | Langrock et al. 2012; McClintock et al. 2020 | open |
| Relatively finite variance: definition (E f ≥ c E f², f = log p₀/p_w); implies p₀ unique; holds when realisable; theory extends to unrealisable case | Watanabe 2018 (Mathematical Theory of Bayesian Statistics), ch. 3 | open |
| US adult height: means ≈ 175 / 162 cm, SD ≈ 7 cm | NHANES 2015–18 (CDC anthropometric reference) | open |

## Parameter-space view and invariants (outline II.2, Part V)

| Claim | Source | Status |
|---|---|---|
| Amari and coauthors studied singular regions and plateaus in neural network learning | Fukumizu & Amari 2000; Amari, Park & Ozeki 2006 | open |
| Bayes marginal likelihood as a code length; stochastic complexity (d/2) log n in the regular case | Rissanen 1996; Grünwald 2007 | open |
| Free energy nL_n + λ log n; λ = d/2 when regular | Watanabe 2009, 2018 | open |
| Log-loss regret against a parametric class ≈ (d/2) log n (NML/Shtarkov); what is known for singular classes | Shtarkov 1987; Rissanen 1996; Grünwald 2007 | open |
| RLCT invariant under resolution of singularities (birational invariant) | Watanabe 2009 | open |
| RLCT volume scaling V(ε) ~ ε^λ (−log ε)^{m−1} | Watanabe 2009 | open |
| Hessian flatness changes under reparametrisations that keep the function | Dinh, Pascanu, Bengio & Bengio 2017 | open |
| Natural gradient is reparametrisation invariant | Amari 1998 | open |
| Jeffreys prior invariance; MLE equivariance; MAP not invariant | standard; e.g. Jeffreys 1946; Lehmann & Casella | open |
| ReLU rescaling symmetry gives balanced layers under gradient flow | Du, Hu & Lee 2018 | open |
| Symmetries give conservation laws of gradient flow, broken by SGD | Kunin et al. 2021 | open |
| Complete characterisation of conserved quantities for ReLU networks | Marcotte, Gribonval & Peyré 2023 | open |
| HMC leapfrog is symplectic and volume preserving, leaving the target invariant | Neal 2011 | open |
| Volume of near-optimal parameters and SGD as an approximate Bayesian sampler | Mingard et al. 2021 | open |

## Postulates, realization theory and sources (outline v7)

All cited from memory or from a web-only search (see `prior-art.md`); check before use.

| Claim | Source | Status |
|---|---|---|
| Posterior E[τ \| 𝓖_n] → E[τ \| 𝓖_∞] a.s.; learnable iff limit is τ | Doob 1949; Schervish 1995 ch. 1 | open |
| Symmetry table: exchangeable / stationary / Markov-exchangeable components and theorems | Kallenberg 2005 *Probabilistic Symmetries*; Diaconis & Freedman 1980 (recurrence condition) | open |
| Symmetry ⇒ ergodic decomposition ⇒ parameter = extremal family | Lauritzen 1984, 1988; Orbanz & Roy 2015 | open |
| A Bayesian observing a stationary process learns to predict as if it knew its ergodic component | Al-Najjar & Shmaya 2014 | open |
| Posterior convergence for dependent data under misspecification | Shalizi 2009 (EJS 3) | open |
| Agent's accessible events as a sub-σ-algebra; minimal world model observationally equivalent for the agent | Ay & Löhr 2015 | open |
| Experiment as a kernel; comparison / garbling order | Blackwell 1951, 1953; Le Cam 1964; Torgersen 1991 | open |
| Laws on 𝒳^ℕ ↔ families of one-step kernels | Ionescu-Tulcea (Kallenberg *Foundations*) | open |
| Chain-rule decomposition characterises KL among f-divergences (exact statement) | to locate (Csiszár?) | open |
| Fisher–Rao as the unique invariant metric; parametrised measure models; pullback without ambient manifold | Čencov 1982; Ay, Jost, Lê & Schwachhöfer 2015 (PTRF 162), 2017 book ch. 3; Lê 2020 | open |
| Prequential forecasting; no true distribution | Dawid 1984; Dawid & Vovk 1999; Rissanen 1989; Grünwald 2007 | open |
| Forecaster vs Reality protocol without actions | Shafer & Vovk 2019 | open |
| Splitting subspace: past ⊥ future given state; minimal Markovian splitting subspace; minimality by inclusion; internal vs external realizations; forward predictor space; non-uniqueness of minimal realizations | Lindquist & Picci 1979, 1985, 2015; Ruckebusch 1976/1980 | open |
| Stochastic realization via canonical correlations | Akaike 1974 (IEEE TAC 19), 1975 (SIAM J. Control 13) | open |
| Stochastic realization of Gaussian processes | Faurre 1976; Picci 1976 | open |
| Prediction process: conditional law of the future given the past is itself Markov | Knight 1975 (Ann. Prob. 3); Knight 1992 | open |
| Causal states: unique minimal sufficient statistic; recursive filter; C_μ ≥ E; link to Knight stated | Shalizi & Crutchfield 2001; Shalizi & Shalizi 2004 | open |
| Generative HMM ≠ minimal predictor; generative complexity under ergodic decomposition | Löhr & Ay 2009 (both papers); Löhr 2012 | open |
| OOMs / PSRs / sequential systems: many generators, one process | Jaeger 2000; Littman, Sutton & Singh 2002; Thon & Jaeger 2015 | open |
| Subspace identification = realization from data | Van Overschee & De Moor 1996 | open |
| Realization theory for RNNs | Defourneau & Petreczky 2019 | open |
| Information bottleneck recovers causal states in the zero-distortion limit | Still, Crutchfield & Ellison 2010; Tishby, Pereira & Bialek 1999; Still 2009 | open |
| Predictive information as the unique complexity measure; growth rate ↔ learnable dimension | Bialek, Nemenman & Tishby 2001 | open |
| Nearly maximally predictive features in RNNs/reservoirs | Marzen & Crutchfield 2017 (PRE 95) | open |
| Transformers linearly represent mixed-state belief geometry; generator non-uniqueness stated | Shai et al. 2024 (NeurIPS; arXiv:2405.15943); Piotrowski et al. 2025 (arXiv:2502.01954) | open |
| Othello-GPT represents the board; linear probes | Li et al. 2023; Nanda et al. 2023 | open |
| LLC invariant under local diffeomorphism; Hessian measures fail invariance | Lau, Furman, Wang, Murfet & Wei 2023 (App. C, E) | open |
| Watanabe: model and prior as "fictional candidates"; unrealisable case | Watanabe 2022 (arXiv:2206.05630); 2018 book | open |
| Reparametrisation invariance via implicit Riemannian metric | Kristiadi, Dangel & Hennig 2023 | open |
| Realization map, orbits, quotient geometry of shallow nets | Dong & Cheng 2026 (arXiv:2603.21502); symmetry classification arXiv:2604.14037 | open |
| Singularities in neuromanifolds | Amari, Park & Ozeki 2006; Fukumizu & Amari 2000 | open |
| Bayesian mechanics: internal states parametrise beliefs; synchronisation map; "gauge" as choice of constraint | Ramstead et al. 2023; Sakthivadivel 2022; Biehl, Pollock & Kanai 2021 | open |
| Environments as chronological semimeasures on full histories | Hutter 2005; Ortega 2011; Ortega & Braun 2013 | open |
| De Finetti in Markov categories | Fritz, Gonda & Perrone 2021 | open |
| Approximation / estimation / optimisation decomposition | Bottou & Bousquet 2008 | open |
| Unbounded likelihood of Gaussian mixtures | Kiefer & Wolfowitz 1956; Day 1969 | open |
| Resolution of singularities; normal-crossing form of K | Hironaka 1964; Watanabe 2009 ch. 2–3 | open |
| Gauge orbit space stratified; no global gauge fixing | Singer 1978; Kondracki & Rogulski 1983 | open |
| Superspace singular at metrics with isometries | Ebin 1970; Fischer 1970 | open |
| Singular symplectic reduction | Sjamaar & Lerman 1991 | open |
| Singular points ↔ larger stabilisers | Luna 1973 | open |
| Banach-manifold structure on densities | Pistone & Sempi 1995 | open |
| Geometry on non-manifold state spaces | Bengtsson & Życzkowski; Lott & Villani 2009; Sturm 2006 | open |
| Takens embedding: hypotheses and 2d+1 delays | Takens 1981 | open |
| Mori–Zwanzig projected equation: memory kernel + noise | Mori 1965; Zwanzig 1973 | open |
| Koopman operator on observables | Koopman 1931 | open |
| History: Gauss 1827; Riemann 1854; Veblen–Whitehead 1931/32; Whitney 1936, 1955; Hartigan 1985; Watanabe 2001 | dates and content | open |
| Bayesian control precedents: Wald 1947; Feldbaum 1960–61 (dual control); Bellman & Kalaba; Åström 1965; Martin 1967; Duff 2002; Harsanyi 1967–68 | dates and attributions | open |
| Act–state independence in Savage and B&S ch. 2 | Savage 1954; Bernardo & Smith 1994 | open |

## Added for outline v8 (world as system; existence by typicality; information form of stationarity; utility chain)

| Claim | Source | Status |
|---|---|---|
| Time averages along a typical trajectory exist and define the empirical law / empirical kernel (cylinder indicators) | Birkhoff 1931; Kallenberg *Foundations* (ergodic theorem); typicality (Boltzmann; Goldstein 2001 "Boltzmann's approach") | open |
| Asymptotically mean stationary processes: ergodic theorem holds for transients | Gray, *Probability, Random Processes and Ergodic Properties* (2nd ed. 2009) | open |
| A measurable function of a stationary process is stationary | standard | open |
| Predictive information I(T) growth classes: finite / log T with coefficient d/2 / power law; interpretation | Bialek, Nemenman & Tishby 2001 | open |
| Predictive rate–distortion; causal states as the zero-distortion endpoint | Marzen & Crutchfield 2016 (J. Stat. Phys. 163); Still, Crutchfield & Ellison 2010 | open |
| Non-predictive retained information is dissipated; minimal energy cost of prediction | Still, Sivak, Bell & Crooks 2012 (PRL 109:120604) | open |
| Sufficiency for a decision problem is weaker than Bayes sufficiency; garbling order | Blackwell 1951, 1953; Le Cam 1964 | open |
| External vs internal realization; generator ≠ minimal predictor | Lindquist & Picci 1985/2015; Löhr & Ay 2009 | open |
