# Handoff: KL profile / level-set diagnostics for the Markov-chain transformer

Paste everything below into a fresh LLM session to continue. It captures the goal,
what's already built, the hard-won facts, the open question, and the exact next experiment.

---

## Goal

In `projects/markov-chain-learning/experiments/single-chain/notebooks/05_analysis.ipynb`
we want **2D KL level-set plots over pairs of parameters** to **diagnose an MCMC sampler**
on an over-parameterized, singular transformer (a `MarkovTransformer`, ~910 params,
vocab 5, max_len 10). The data-generating process (DGP) is a single Gambler's-Ruin Markov
chain `q` with analytic transition matrix `T` (so `log q` is exact).

The intended object is the **profile KL**: at each grid point fix 2 parameters and
**minimize the loss over all the other params** (NOT a slice that freezes them), then plot
the resulting KL. The user has successfully used this profile method before on
LOWER-dimensional models (mixtures of 3/4/5 components with a single-component DGP).

## Definitions (agreed)

- Population KL: `F(w) = E_{x~q}[log q(x) - log p(x|w)] = KL(q||p_w)`, estimated by
  Monte-Carlo over M fresh draws from `q`. Since `E_q[-log q] = H` (analytic entropy
  floor), `F(w) = mean-NLL(w on the M sample) - H_FLOOR`. No prior, no train/val split.
- `H_FLOOR = -(pi[:,None]*T*log T).sum()` where `pi` is the stationary dist. ≈ 1.2616.
- Profile (what we want) = fix 2 coords, **minimize over the rest**. Slice = freeze the
  rest at w* (cheaper, but overstates rigidity).

## What's already built (committed to the working tree, not git-committed)

1. **`packages/torch_bdn/src/torch_bdn/bn/constrained_fit.py`** (new, exported from
   `torch_bdn.bn`):
   - `fit_constrained(model, loss_fn, batches, *, fixed_indices, fixed_values=None,
     init_theta=None, epochs, lr, optimizer_kwargs=None, full_data=None, restore=True)`
     — runs a normal training loop but ZEROS the gradients of `fixed_indices` each step
     (+ hard re-pin), so chosen raw-parameter coordinates stay fixed. `batches` is a
     zero-arg callable returning a one-epoch iterable of (x,y) minibatches.
   - `fit_constrained_directions(... basis, coeffs, origin ...)` — same idea but pins the
     COEFFICIENTS along orthonormal `basis` columns (for eigenvector axes): projects
     span(basis) out of grad+step each step, re-pins; supports `cosine_decay` and
     `avg_frac` (late-iterate averaging for minibatch SGD).
   - `top_diagonal_curvature_indices(H, k)` — indices of k largest Hessian diagonal entries.
   - Returns `ConstrainedFitResult(loss, theta, epochs_run/epochs, free_grad_norm)`.

2. **05_analysis.ipynb** cells:
   - `d1331132` — population-KL setup: draws M=3000 from q, defines `loss_fn` (mean CE,
     ignore PAD), `H_FLOOR`, `population_kl(model)`; finds the population-KL minimizer
     `theta_star`/`wstar_model` via minibatch SGD (`fit_constrained`, empty fixed set,
     300 epochs, lr 3e-3, batch 256); computes Hessian `H_star`; picks axes
     `AXIS_I, AXIS_J` = top-2 diagonal-curvature params.
   - `0d142f8f` — self-contained loader rebuilding `sample_size_results` (per-N in-sample
     Hessians) for the SEPARATE eigenspectrum-vs-N figure in `6072eef9`.
   - `bb39335d` — the PROFILE grid (15x15, ±2): per-cell minibatch SGD (`profile_cell`,
     60 epochs, cosine decay, late-iterate-averaged KL, warm-started), freezing AXIS_I,J.
     Plots raw KL + quantile-rank level sets. **This comes out FLAT (~0 everywhere).**
   - `dace88e2` — 4-panel NLL SLICES at w* (others fixed at MLE), ±3, 41x41, forward-eval
     only: {params, eigvecs} x {stiff, sloppy}.

## Hard-won facts (verified empirically this session)

- **Training (04) is pure NLL, no prior** (`Adam(lr=3e-3)`, `CrossEntropyLoss`, no
  weight_decay). The saved "MAP" checkpoints are really EARLY-STOPPED MLEs.
- **Minimizing the IN-SAMPLE empirical loss is ill-posed**: the over-parameterized model
  sends ||theta||→∞ (loss→-∞), never converges. Fix = profile the POPULATION KL (bounded
  below by 0).
- **Full-batch Adam at high lr is unstable**; **minibatch SGD self-regularizes ||theta||**
  (noise acts as implicit reg). Read each cell as the mean late-iterate KL.
- A **negative estimated KL** is the signature that M is too small (inner optimizer overfits
  the MC sample).
- **The PROFILE over 2 raw params is FLAT to 5 decimals** — and this is NOT a pinning bug:
  verified the pinned param is held exactly, ||theta|| stays bounded, yet KL→0. The other
  ~908 params re-adapt to compensate the fixed pair.
- **The SLICE (others fixed) is NOT flat**: along a single param KL rises 0→0.55→2.4→...→12.8
  at displacement +0.5..+20; along the top eigenvector even steeper (KL=246 at +10).
- 4-panel slice maxima (±3): params-stiff 2.75, params-sloppy ~0, eigvec-stiff 73.8,
  eigvec-sloppy 24.9 (the last is locally flat but globally curved = higher-order/algebraic
  singularity behaviour).
- Hessian at w*: eig range ~[-0.012, 2.45]; even the largest DIAGONAL entries are only ~0.26
  (single params carry little curvature; curvature lives in eigenvector combinations).

## The OPEN QUESTION (what to resolve next)

Why is the 2-parameter PROFILE flat? Two hypotheses:
- (A) Real: the model is so over-parameterized that fixing 2 coords is non-binding (the
  zero-KL set is huge), so the params are genuinely unidentifiable. (Note: earlier I called
  the zero-KL set a "manifold" and did a dimension count — that was sloppy; it's a real
  analytic VARIETY, can be non-smooth/self-intersecting, no clean global dimension. Don't
  repeat that hand-wave.)
- (B) **Artifact of under-sampling the profiling**: with M=3000 the inner minimizer overfits
  the MC sample, driving the ESTIMATED KL to ~0 even if the true population KL at the fixed
  (alpha,beta) is > 0. User finds (B) plausible and notes the profile method worked on
  low-dim models. Larger M should then reveal real profile structure.

## NEXT EXPERIMENT (agreed plan)

Test (B) directly, staying cheap and near the MLE:
- Axes = the **raw parameter pair** AXIS_I, AXIS_J (the object that came out flat).
- Grid: ±0.2 from w* (small → in the local quadratic regime, fast, warm-startable). E.g. a
  short line `alpha ∈ {-0.2,-0.1,0,0.1,0.2}, beta=0`, or a small 5x5.
- Sample sizes: **M ∈ {5k, 10k, 20k, 50k}**.
- For each (M, cell): minimize-the-rest on the M-sample, then record BOTH:
  (a) **KL evaluated on the M-sample** (what the optimizer sees), and
  (b) **KL evaluated on a fixed large HELD-OUT sample (e.g. 100k draws from q)**.
- Read: if (a)→0 but (b) stays > 0 → hypothesis (B) confirmed (profiling overfits; flatness
  is a sampling artifact; bigger M fixes it). If (a)≈(b)→0 as M grows → flatness is real
  (genuine compensation / unidentifiability).

This is now IMPLEMENTED as notebook cell `c8bf815d` (right after the 4-panel slice cell
`dace88e2`), but NOT yet run. The cell: keeps the same 2-param profile (fix AXIS_I,AXIS_J,
minimize the rest with minibatch SGD), but evaluates only **4 small offsets (±0.1,±0.1)**
from w* (not the 15x15 grid), for **M ∈ {5k,10k,20k,50k}**, and reports the profiled KL
**on the M-sample** (the value the minimizer produces, per the user's request — no held-out).
It plots profiled-KL vs M, one line per offset. READ: if the lines RISE with M, the flat
profile was the net minimizing the EMPIRICAL (not population) KL on too few draws
(hypothesis B); if they stay ~0 for all M, the compensation/flatness is genuine.
Run the setup cell `d1331132` first (it defines loss_fn, H_FLOOR, KL_BATCH/LR, _anchor_ckpt,
etc. that this cell reuses). Expensive part: per M a 200-epoch anchor fit + 4×60-epoch cells;
M=50k is the slow one.

## HYPOTHESIS 2 — why the 2-param profile is flat, and how to get a non-flat one

**Hypothesis (degenerate-set compensation):** The Hessian at w* has a large degenerate
subspace (~700 near-zero eigenvalues out of 910). That means the min-NLL set is an ~700-dim
"level set of equivalent parameters". When we fix 2 coordinates and minimize the other 908,
the optimizer just slides ALONG that ~700-dim equivalence set to a point that still achieves
min NLL — so the profile is flat. This is the honest version of the earlier (badly stated)
"the zero set is huge" claim: count the degenerate directions = dimension of the equivalence
set; fixing 2 coords does not escape it. Consistent with everything observed.

**Corollaries:**
- To get a NON-flat profile you must constrain a direction in the NON-degenerate complement,
  i.e. a **stiff eigenvector** (large eigenvalue, ~200-dim complement). Fixing a coefficient
  along a stiff eigenvector cannot be compensated by sliding along the degenerate set, so KL
  rises. Single raw parameters fail because they project mostly into the degenerate set.
- **Eigenvectors are only a LOCAL LINEAR frame** (tangent directions of the geometry AT w*).
  The true zero-set is a curved analytic variety — the "curvature coordinates" are nonlinear.
  So a straight eigenvector line is valid only in a SMALL neighborhood of w*; far out (±2,±3)
  the line leaves the curved valley and the values stop meaning "along the soft/stiff
  direction" (this is why the sloppy-eigvec SLICE rose at ±3 — the straight cut wandered off
  the bent flat valley into a steep region).

**Test plan:**
1. Quantify the degeneracy: eigendecompose H(F) at w*, count eigenvalues below a small
   threshold (e.g. λ_max·1e-3 and 1e-6). Confirm ~700 are degenerate (the claim should be
   MEASURED, not assumed). Report the count + the spectrum.
2. Profile along the 2 STIFFEST eigenvectors (use `fit_constrained_directions`, already in
   `torch_bdn.bn`), at SMALL radius (±0.3) with a FINE grid (e.g. 15–21 pts) so the local
   linear frame stays valid. Expect a non-flat bowl with the star at the bottom.
3. Contrast: also profile along the 2 SOFTEST eigenvectors at the same small radius — expect
   ~flat (still compensable / in the degenerate set). Stiff-curved vs soft-flat at small
   radius is the clean signature.
4. (Optional, second-order) To follow the curved valley further than the local frame allows,
   recompute H along the path and re-derive the eigenbasis at a few radii, rather than
   extrapolating one straight eigenvector line.

**Read:** stiff-eigvec profile rises near w* (real, incompressible curvature) while
soft-eigvec profile stays flat (degenerate/compensable) → confirms the degenerate-set
hypothesis AND gives the correct way to draw level sets for sampler diagnosis (stiff
eigenvectors, small radius). NOTE this is distinct from Hypothesis B (under-sampling): both
can be tested; if the M-sweep (cell `c8bf815d`) shows flatness is M-independent, that points
to this degeneracy explanation rather than under-sampling.

## Environment notes

- Run from repo root `/Users/ashrafahmed/workspace/slt-deep` with `uv run --active python3`
  (the active venv has torch; a bare `uv run` may rebuild and drop torch).
- Project packages: `sys.path.insert(0, ".../projects/markov-chain-learning/packages")` for
  `pytorch_models` (MarkovTransformer) and `markov_chain` (sample_sequences).
- macOS has **no `timeout` command**; use nbconvert's `--ExecutePreprocessor.timeout`.
- The 15x15 x 60-epoch profile grid takes ~15 min; keep diagnostics small/standalone.
- Cached frame/anchor (if present): `/tmp/frame_cache.pt`.
