# Design notes

Supporting material for `outline.md`. These points may go into the note itself, but
they don't belong in the outline.

## Running examples: all models are wrong

The examples do not look for data where a model is true. They take one DGP and
compare idealisations of it, judged by regret under q.

- **Records are finite, so a saturated model exists.** Heights are recorded to a
  finite precision, so the record set is a finite set of bins. Under exchangeability
  the categorical (multinomial) model over the bins contains q exactly. It is
  regular in the interior of the simplex. It is the "nonparametric" model made
  finite, and every other model is a restriction of it.
- **Gaussian: a two-parameter idealisation.** It is wrong (pooled height is a mixture
  of two near-normal subpopulations), but close: K(w₀) is small. It shows the trade
  between estimation error (2 vs many parameters) and approximation error K(w₀),
  measured on held-out data.
- **Mixture of two Gaussians: an idealisation matching the known structure.** Still
  wrong in detail, but closer. It is singular, and the pooled distribution is not
  bimodal (Schilling, Watkins & Watkins 2002), so p₀ sits near the locus where the
  components merge. Relatively finite variance is the condition under which the
  singular theory still applies even though the mixture is not exactly right.
- **With sex recorded (optional).** The hidden label becomes an input, the model
  becomes conditional and regular, and the singularity disappears. This shows that
  singularity is a property of the model and what is recorded, not of the world.
- **Alternative singular step: mixture of Poissons**, for count data with
  overdispersion (e.g. counts with heterogeneous rates). Same structure (label
  switching, merging components), with discrete records. Height keeps one dataset
  throughout; Poisson needs a second one.
- Source for height: a public survey such as NHANES.

Parallel for text (possible remark): the saturated model for documents is the
categorical over all token sequences, which is impossible to fit. n-gram models and
transformers are restrictions of it, just as the Gaussian and the mixture are for height.

- **Labelled accelerometer data → HMM.** The person is in one activity at a time
  (walking, sitting, climbing stairs), activities persist, and the sensor signal
  varies within each one. Labelled public datasets exist (e.g. UCI Human Activity
  Recognition, Anguita et al. 2013). Honest caveat: activity durations are not
  exactly geometric, so the Markov assumption is a motivated approximation rather
  than a derived one. Alternative with the same role: animal movement (resting /
  foraging / travelling from GPS tracks), which is widely modelled with HMMs in
  ecology.
- **Natural text → transformer.** No mechanism gives a model. The model is chosen
  for flexibility and judged by held-out loss. q exists by exchangeability of
  documents but is not realisable.

Possible extra: a small transformer trained on the activity sequences, i.e. a flexible
model on data with a well-motivated model. It shows the same target reached through a
different map.

## Generator question (from an earlier discussion)

On real text, q exists by exchangeability of documents (I.3), but it is not
realisable by any finite model and is known only through held-out loss. This is the
approximation term K(w₀) > 0. A 2 × 2 design separates the gap: data {annotated HMM
corpus, broad natural corpus} × model {HMM, transformer}. Moving across a row changes
only the parametrisation; moving down a column changes only what is known about q.

Broad natural corpus for Part IV if needed: BabyLM 10M (Warstadt et al. 2023). Its
mixture of sources is a concrete de Finetti mixture, and per-source held-out loss
shows the shift from training data to deployment.
