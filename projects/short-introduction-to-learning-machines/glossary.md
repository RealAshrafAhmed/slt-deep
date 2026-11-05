# Jargon audit and glossary

Audience: a reader with a bachelor's degree in computer science, mathematics or
physics. They know probability (random variables, conditional probability,
expectation, the law of large numbers), linear algebra, basic analysis, and some
machine learning. From undergraduate physics they know, or will accept with a
one-sentence reminder, the language of classical and statistical mechanics:
microstate, phase space, dynamics, observable, ensemble, macrostate, coarse-graining,
the ergodic hypothesis. They do not know measure theory beyond the words, ergodic
theory as mathematics, systems theory, information geometry or algebraic geometry.

Consequence: **physics vocabulary is part of the note's working language.** Each
physics term is still defined in one plain sentence when first used, so a
computer-science reader can follow, but it is then used freely and need not be
replaced by a plain phrase.

Rules for the prose:

1. **Idea first, name second.** Every technical term is introduced by a plain
   sentence that says what the thing is and why we need it, then the name is
   given in bold, then the name is used.
2. **One new term per paragraph**, at most.
3. **Named theorems appear in parentheses or footnotes**, never as the subject of
   a sentence: "one long run tells you the average (the ergodic theorem)".
4. **Physics words are working vocabulary**, defined once in a plain sentence and
   then used freely: "the full description of the world at one instant, its
   **microstate**". Specialised physics (Mori–Zwanzig, gauge theory, Takens) still
   gets a gloss in place and stays out of the main line of argument.
5. **A term from another field is given once, in a table (§7) or a footnote**,
   not in running text.
6. **Symbols are introduced with a sentence in words** and reused without change.
   No symbol appears before its sentence.
7. The field guide (§6) and the connective-tissue table (§7) are where the
   jargon of each field is collected; the body stays plain.

## Decision table

Columns: term as used in v8 → plain phrase to use in the body → decision
(**keep** and introduce; **replace** by the plain phrase; **footnote/table only**).

### The world and what we see

| Term | Plain phrase | Decision |
|---|---|---|
| microstate, phase space | the complete description of the world at one instant; the set of all such descriptions | keep, working vocabulary, defined in §1.1 |
| dynamics Φ_t | the rule that moves the world from one instant to the next | keep, working vocabulary |
| instrument h | the device that reads a few numbers off the world | keep "instrument"; the map is "what the instrument reads" |
| history, trajectory, path space Ω | the complete record of everything that happens, past and future; the set of all possible such records | keep "history"; "history space" for Ω; drop "trajectory" and "path space" except once |
| ω*, "the one that occurs" | the history that actually happens | keep the symbol after one sentence |
| observable | a quantity that depends only on the history (any function of it); the physics notion extended from microstates to histories | keep, working vocabulary, defined in §1.4 |
| measurable function, σ-algebra, 𝓕 | — | footnote only ("for the mathematically careful: measurable with respect to …") |
| filtration 𝓖_t | what has been read by time t | replace in body with "what has been read so far"; the symbol 𝓖_t kept; the word "filtration" in a footnote |
| atom A_t(ω*) | all histories that agree with what has been read so far | replace; symbol kept |
| hidden | not determined by what has been read so far | keep, defined this way |
| learnable | determined by the complete record, if you waited forever | keep, defined this way |
| Doob's theorem | "your best guess converges to the truth exactly when the truth is learnable" | parentheses |
| regular conditional distribution, disintegration | the probability of the next reading given the past | footnote only |
| standard Borel | — | footnote only |
| coarse-graining | reading only a few numbers off a detailed world | keep, working vocabulary; it is what the instrument does |
| Mori–Zwanzig | "when you track only a few variables, the rest shows up as memory and noise" | parentheses, once |
| Koopman | — | §7 table only |
| Takens | "the recent past of a single reading can stand in for the hidden state" | parentheses, once |

### One history is enough

| Term | Plain phrase | Decision |
|---|---|---|
| symmetry, group G acting on the index set | a reordering or shift of the records that does not change the odds | keep "symmetry", introduced by example (shuffling, shifting) |
| exchangeable | shuffling the records does not change the odds | keep, defined |
| stationary | shifting the records in time does not change the odds | keep, defined |
| ergodic, ergodic hypothesis | one long run gives the same averages as many independent runs | keep, working vocabulary ("the ergodic hypothesis: one history stands in for the ensemble") |
| ergodic decomposition, invariant σ-field, extremal family | the records split into types; a long run reveals its type | "type" in prose; the mathematical names in a footnote |
| macrostate θ | the type of the history: the part that a long run reveals and that never changes | keep, working vocabulary; "type" may be used as its plain synonym |
| typicality, Birkhoff | almost every history has well-defined long-run averages | parentheses |
| law, law of the records | the long-run frequencies of the records (what "the distribution of the data" means) | keep "the law"; define as long-run frequencies |
| ensemble | imagining many copies of the world | keep, working vocabulary |
| de Finetti, Hewitt–Savage, Diaconis–Freedman, Kallenberg | — | footnote |
| test martingale | a betting test that can reject the symmetry | parentheses |
| asymptotically mean stationary | — | footnote |

### What gets learned

| Term | Plain phrase | Decision |
|---|---|---|
| Markov kernel, kernel κ | the rule giving the odds of the next reading given the past | replace with **"next-step rule"** in the body; "kernel" introduced once as the technical name and used in §3–§4 where symbols are needed |
| conditioning variable, weight | the past, or the input; "how often each input occurs" | replace |
| meeting space 𝒦 | the set of all next-step rules | replace; symbol kept |
| Ionescu-Tulcea | — | footnote |
| Blackwell experiment, garbling, comparison of experiments | "one instrument is more informative than another if its readings could be degraded into the other's" | one sentence, parentheses |
| relative entropy rate, KL divergence | the average surprise of the truth under the rule, minus the unavoidable part; "the gap" | keep **KL** after defining "gap"; "relative entropy" never |
| log score, proper scoring rule | being scored by the log-probability you assigned to what happened | keep "log score", defined |
| Fisher–Rao, Čencov | the natural way to measure small changes of a rule | keep **Fisher information** only; Čencov in footnote |
| entropy rate h | the unavoidable surprise per step | replace; symbol kept |
| predictive information I(T), excess entropy E | how much the last T readings tell you about the future | keep "predictive information"; "excess entropy" in table |
| rate–distortion, information bottleneck | the least memory needed to predict within a tolerance | keep "memory vs accuracy curve"; names in parentheses |
| dissipation (Still et al.) | memory that does not help prediction costs energy | one sentence, cited |
| decision-theoretic sufficiency | a summary that is good enough for the use you have in mind | replace |
| KL projection κ₀, relatively finite variance | the best rule the model can express; (a technical condition making it unique) | replace; condition in footnote |

### The learner

| Term | Plain phrase | Decision |
|---|---|---|
| realization, stochastic realization theory | a machine with a memory that produces the same next-step rule | introduce as **"a machine with memory"**, then "realization" as the technical name, once |
| internal state, state space S | the memory | keep "memory" in prose, "state" S in symbols |
| splitting property, Markov property | the memory holds everything from the past that matters for the future | replace; "Markov" in parentheses |
| filter π_t | how the memory is updated by each new reading | replace with "the update rule"; symbol kept |
| trivial realization | the memory is the whole past | replace |
| minimal realization, predictor space, causal states, ε-machine, prediction process | the smallest memory that still predicts as well as possible | keep "smallest memory"; "causal states" named once in §5 |
| internal vs external realization | memory built from the readings vs memory inside the world | replace |
| generator vs predictor | the machine that produced the data vs the machine that predicts it | keep as plain phrase |
| splitting subspace, Lindquist–Picci, Knight, Akaike, subspace identification | — | footnote / table |
| mixed-state presentation, belief state | the probabilities over hidden states given the past | keep "belief state", defined |
| parametrisation vs chart | the weights describe the machine, but several weight settings can be the same machine | keep both words, introduced with the cusp picture |
| fibre | the set of weight settings that give the same machine | keep, defined |
| identifiable | every machine has exactly one weight setting | keep, defined |
| singular, regular | "singular": weights where the usual counting of parameters breaks down | keep, defined by example (two merged components) |
| resolution of singularities, Hironaka, blow-up, normal crossing | — | footnote |
| learning coefficient λ, RLCT | the effective number of parameters near the best weights | keep "effective number of parameters" then λ |
| pullback, push-forward | reading a quantity defined on rules as a quantity on weights, by composing; sending weights to rules | keep both, introduced with one sentence each |
| reparametrisation | changing the way weights are written without changing the machines they describe | keep, defined |
| gauge, orbit, stratified quotient, stabiliser | — | physics box only, each glossed in place |
| Gibbs measure, tempered posterior | — | footnote |
| Bottou–Bousquet decomposition | the three sources of error: the model cannot express the truth; the data are finite; the optimiser did not finish | replace; name in parentheses |
| stochastic approximation | — | parentheses after "SGD" |
| predictive / structural alignment | predicting as well as the truth / the memory matching the smallest memory (or a real hidden quantity) | keep, each defined |
| efficiency (of a representation) | memory that holds nothing that does not help prediction | keep, defined |
| instrumentalist reading | "whether the rule is 'really there' does not matter" | replace |

### Fields and names

| Term | Decision |
|---|---|
| Bernardo & Smith, Savage, Wald, Blackwell, Le Cam, Watanabe, Amari, Shalizi–Crutchfield, Lindquist–Picci, Lauritzen, Friston, Hutter, Dawid, Rissanen, Grünwald | citations in parentheses or footnotes; never the subject of a sentence in the body; collected in Appendix B and §7 |
| computational mechanics, information geometry, singular learning theory, realization theory, statistical mechanics, decision theory | named once each in §0 as "the fields this note draws on", with one plain sentence per field; afterwards only in §7 |

## Terms kept, in order of introduction

*Physics working vocabulary* (defined once, used freely): microstate · phase space
· dynamics · observable · coarse-graining · ensemble · ergodic hypothesis ·
macrostate.

*Introduced in the note:* history · instrument · hidden · learnable · symmetry
(exchangeable, stationary) · type · the law (long-run frequencies) ·
next-step rule (kernel) · the gap (KL) · log score · predictive information ·
memory vs accuracy curve · machine with memory (realization) · state · update
rule · smallest memory (causal states) · belief state · weights · fibre ·
identifiable · singular · parametrisation vs chart · pullback · push-forward ·
reparametrisation · effective number of parameters (λ) · Fisher information ·
predictive alignment · structural alignment · efficiency.

About thirty new terms over twelve thousand words, about one every four hundred
words, plus eight physics terms the reader is assumed to recognise.
