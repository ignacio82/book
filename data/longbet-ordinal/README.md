# Data for the ordinal LongBet chapter

These compact extracts come from the ordinal comparison rerun in September 2026 with the current engine.
All 20 prescribed seeds, 91000–91019, are retained in each of the two advanced
scenarios. The chapter's two-profile illustration is calculated directly from
the probit model; it is not an additional fitted-model benchmark.

## Files and targets

- `panel-results.csv`: 180 method-by-panel rows from 40 independent panels.
  Probability losses use the probability scale; multiply RMSE by 100 for
  percentage points. Blank metrics are inapplicable, not zero.
- `decision-draws.csv.gz`: paired draws for exposure 4 of the first
  prespecified rare-rating panel, seed 91000. `top_att` concerns category code
  4 (five stars); `lowest_att` concerns code 0 (one star). Four chains each
  contribute the same number of retained draws (`provenance.json` records it),
  stored in chain-major order. Both columns average effects over the same
  held-out treated account-week cells at exposure 4.
- `provenance.json`: experiment settings, library versions, source hashes,
  the original primary comparison, and the decision example definition.
- `checksums.csv`: SHA256 hashes checked when the chapter executes.

The decision thresholds (at least +3 points in five-star probability and
at least -20 points in one-star probability) are an illustrative teaching
rule added for the book. They were not a prespecified hypothesis of the
simulation comparison. The event is recomputed from paired draws, and its
own R-hat, mean ESS, and mean MCSE are checked with R's `posterior` package.
Each chain must visit both event states. Quantile-tail ESS is undefined for
a Bernoulli indicator when a tail threshold produces a constant sequence;
the indicator's mean is the probability being estimated and diagnosed here.
No individual cross-world rating transition is inferred.

For each panel, `top_att_mse` averages squared five-star ATT error equally over
six exposures. `category_cate_mse` averages squared errors in all five
conditional category effects across the 900 held-out treated cells.
The chapter's RMSE is the square root of mean panel MSE, not the mean of
panel RMSEs. The category-effect metric is a secondary, descriptive comparison.

## Generating mechanisms

Each panel has 240 training accounts, eight consecutive weeks, and four
independent Uniform(-1,1) baseline covariates. Adoption is balanced across
weeks 3, 4, 5, and never. A separate random stream generates 240 independent
evaluation accounts with the same design. Errors are independent over accounts
and weeks, with no missing outcomes or unit random effects. Calendar and
exposure stay within the training support. No evaluation values or known
effects enter fitting or tuning.

Let h(s) = 1 - exp(-s/2), with effects zero before adoption. Both scenarios
use cutpoints (-infinity, 0, 0.7, 1.4, 2, infinity).

- Rare top rating: untreated location -0.6 + 0.55*x0 - 0.4*x1 + 0.08*(t-4);
  treated location adds 0.35 + 0.4*h(s) + 0.1*x2. Residual SD is 1 in both arms.
- Changing dispersion: untreated location 0.9 + 0.3*x0 - 0.2*x1 + 0.05*(t-4);
  treated location adds 0.1*h(s). Untreated SD is 0.65; treated SD is
  0.65 + 1.3*h(s).

True potential category probabilities are normal-CDF differences. All methods
predict the same evaluation accounts under their assigned history and under
never treatment at the same calendar weeks. Their difference is evaluated
against the known probability effect, not a simulated individual effect.

## Methods and computation

Ordinal LongBet uses the September 2026 engine (`longbet-jax`): four chains
started from the prior, the package's forest defaults (20 prognostic and 60
treatment trees of depth at most 8), fixed treatment-only coding, no exposure
splits in the treatment forest (the trajectory carries the exposure profile),
no random intercept, and CPU execution. Binary LongBet uses the same settings
on the highest-category indicator.

Each chain discards 2,000 burn-in sweeps, runs 4,000 further sweeps and keeps
every fourth draw. R-hat > 1.01 or bulk/tail ESS < 400 on any monitored
quantity triggers one diagnostic-only extension: 8,000 burn-in sweeps plus
16,000 sampling sweeps, again keeping every fourth draw. The 30 category ATTs
and three free cutpoints are monitored for ordinal fits; six highest-category
ATTs are monitored for binary fits. `panel-results.csv` records, per fit, the
number of attempts and the diagnostic failures before and after the extension;
`provenance.json` counts the rare-rating fits that still fail after it.
Diagnostics do not certify every conditional account-level effect.

The maximum-likelihood probit ordinal regression uses three-df natural splines
for each covariate, week and exposure factors, and treatment interactions with
all covariate splines (`ordinal::clm`). A second fit adds exposure-specific
log residual scales.

The histogram boosted classifier is a multiclass S-learner on covariates,
calendar week, exposure, and treatment. Training-account-grouped three-fold
CV selects log-loss performance across 7/15/31 leaves and L2 penalties 1/10;
250 iterations, learning rate .05, minimum leaf size 15, no early stopping.
Fit times include tuning and diagnostic extensions. The timings come from
shared machines running other work concurrently and are descriptive, not a
controlled speed benchmark.

The primary rare-rating contrast was set before confirmatory fitting. Its
97.5% paired bootstrap interval uses 50,000 resamples of the 20 panel losses,
seed 77331. The conservative level originally allowed for two questions;
the planned nonlinear comparison failed pilot diagnostics under an earlier
engine revision and was deferred before confirmatory accuracy results were
examined. That allocation was not reassigned. Twenty panels provide
approximate bootstrap tails and limited coverage evidence. Other comparisons
are descriptive.

## Reproduction

The book is self-contained for regenerating every displayed number and figure:

```bash
./render.sh longbet_ordinal.qmd
```

`R/longbet-ordinal.R` verifies the extract hashes, panel counts, decision draw
order, finite values, and the rare-rating diagnostic gate. The chapter then
rebuilds comparison tables and probabilities using the saved draws. Include
the updated `_freeze/longbet_ordinal/` in the same change when publishing.

To refresh these extracts from the original completed experiment, with a
Python environment containing NumPy:

```bash
python tools/import-longbet-ordinal.py /path/to/ordinal-comparison
```

The author's experiment directory is
`/home/ignacio/vignettes/ordinal-comparison-v2`. Its `README.md`, `protocol.md`,
`protocol-correction.md`, `benchmark.py`, `baselines.R`, and `run.py` document
and regenerate the full fits. The import script checks every hash in
that experiment's manifest before exporting any book data. The source
manifest and selected draw-file hashes are recorded in `provenance.json`.
Re-exporting deliberately preserves every confirmatory panel and uses the
first seed for the worked decision; it does not select favorable fits.

The compact book extract omits full fitted forests and per-cell MCMC arrays.
To study a different population, threshold rule, model revision, or subgroup
posterior, retain or regenerate the corresponding model output and validate
that new target. A frozen book example is not a substitute for that analysis.
