# Implementation Plan: Rewriting the LongBet Chapter on `longbet-jax`

Rewrite [`longbet.qmd`](file:///home/ignacio/book/longbet.qmd) (2,974 lines)
against **`longbet-jax`**, which is now the only LongBet engine. The R/C++
XBART backend is gone by design.

**Read §1 before touching the chapter.** This is not a cosmetic port. The R API
changed in ways that fail *silently* — positional arguments now bind to
different parameters, and a trace-indexing idiom used a dozen times through the
chapter now selects the wrong draws instead of erroring. A rewrite that skips §1
produces a chapter that renders and reports wrong numbers.

Engine repo: `/home/ignacio/longbet-jax` (monorepo: Python engine in
`src/longbet/`, R package in `R/`). Verified state as of 2026-09-06: 92 Python
tests pass / 1 skipped; R `R CMD check` reports `Status: OK` both with the
Python engine installed and with no Python at all.

---

## 1. Breaking changes: what will silently produce wrong output

Work through this table first. Every row is a real difference between the
engine the chapter was written against and the one it will now run on.

| # | Old behaviour the chapter relies on | New behaviour | Failure mode |
| :- | :--- | :--- | :--- |
| 1 | `predict.longbet(fit, x, x_trt, z, t)` — positional | Signature is `predict(object, x, z, t = NULL, x_trt = NULL, ...)` | **Silently misbound**, see §2.1. Whether it then errors depends on the shapes involved. |
| 2 | Trace arrays include burn-in; `post <- (burnin + 1):ncol(...)` | Traces contain **only post-burn-in draws**; `ncol == num_sweeps` | **Silent.** Drops the first `num_burnin` *retained* draws and keeps nothing extra. |
| 3 | `pcat = 2` marks trailing unordered categoricals | `pcat` is a **hard error** | Loud — but the fix is a modelling change, see §2.2. |
| 4 | `att_stability()` returns `width_ratio`, `reliable` | Neither exists. Returns `ess_median`, `ess_min`, `ess_tail_median`, `rhat_max`, `mcse_median`, `ess_ok`, `rhat_ok`; `is_reliable` is always `NA` | Loud (`NULL` columns), but the surrounding prose is now false — see §2.5. |
| 5 | `get_att()` → `att`, `intervals`, `att_full` | Adds `exposure`; `intervals` is `2 x S` | Mostly compatible. |
| 6 | `get_catt()` → `catt`, `intervals` | Returns `catt`, `sd`, `lower`, `upper` — no `intervals` | Loud. |
| 7 | `b0 = b1 = 1` fixed | `b_scaling = TRUE` by default: **`b0`, `b1` are sampled** | Silent modelling change; `beta_values` is scale-arbitrary, see §2.6. |
| 8 | Single chain; convergence judged by sweep count | `num_chains` available; R̂ reported | The chapter has no multi-chain narrative yet. |
| 9 | `x_trt = x` was the idiom | Omitting `x_trt` shares one column block; passing it creates a **second duplicate block** | Silent waste + a different (needlessly split) model. |
| 10 | `predict()` could omit inputs used at fit | Every fitted input must be supplied again | Loud, and deliberately so. |

### 1.1 Search-and-destroy checklist

Run these from `/home/ignacio/book`. Each must return nothing before the
chapter is considered ported.

```bash
grep -n "pcat"                     longbet.qmd   # → one-hot instead (§2.2)
grep -n "burnin + 1)"              longbet.qmd   # → delete the slice (§2.3)
grep -n "model_params\$burnin"     longbet.qmd   # → same
grep -n "width_ratio\|\$reliable"  longbet.qmd   # → gone (§2.5)
grep -n "predict\.longbet("        longbet.qmd   # → name every argument (§2.1)
# Positional predict calls specifically — the dangerous ones:
grep -nE "predict\.longbet\([^,]+, *[a-z_]+, *[a-z_]+," longbet.qmd
```

Known counts in the current chapter, as a progress check: `pcat` at lines 415,
426, 565, 583, 1788, 1902; the burn-in slice at 638, 1286–1288, 1644–1645, 1928,
1972–1984 (inline `` `r ` `` expressions — easy to miss), 2096, 2198; positional
`predict.longbet()` at 1167, 1553, 1793.

---

## 2. What each change requires

### 2.1 Name every argument

R matches named arguments first, then fills the *remaining* formals with the
positional ones in order. So the chapter's idiom does not bind the way it reads.
Verified against the new signature
`predict(object, x, z, t = NULL, x_trt = NULL, ...)`:

```r
predict.longbet(lb_fit, x, x, z_all, t = week_study, random_seed = 1)
#   t and random_seed are taken by name, leaving object, x, z, x_trt
#   -> object = lb_fit
#   -> x      = x
#   -> z      = x        <-- the covariate matrix, used as the treatment panel
#   -> x_trt  = z_all    <-- the treatment panel, used as covariates
```

Nothing warns. Whether it then errors is an accident of shape: here `z = x` is
`3000 x 5`, so the engine infers `T = 5` and rejects a 14-long `t` with a
message about `t`, which points at the wrong argument entirely. Change the panel
dimensions and the same call could run to completion on a transposed model.

```r
# RIGHT
pred_all <- predict(lb_fit, x = x, z = z_all, t = week_study, random_seed = 1)
```

Name every argument to `longbet()` and `predict()` throughout, and prefer the
S3 generic `predict()` over calling `predict.longbet()` directly. The note at
line 543 about `predict.longbet(t = NULL)` behaviour on `google/longbet` no
longer applies and should be cut.

### 2.2 Categoricals: one-hot, and say why

The engine has no notion of an unordered categorical — every column is ordered
numeric. `vertical` has three levels, so `as.integer(factor(vertical))` would
impose a false ordering (Apparel < Home < Electronics).

```r
vert <- factor(vertical)
x <- cbind(
  base_level, base_slope, log(listings),      # continuous
  fulfilled,                                  # binary: ordered == unordered
  model.matrix(~ vert - 1)[, -1]              # one-hot, drop one level
)
colnames(x) <- c("base_level", "base_slope", "log_listings",
                 "fulfilled", paste0("vertical_", levels(vert)[-1]))
```

`fulfilled` is binary, so it needs no encoding — a split on a two-level column
is the same either way. Note also that the "continuous columns first" ordering
was a C++ backend requirement and no longer matters; keep it only if it aids
readability.

**Prose to write (§ "Covariates handed to the model"):** state this as a real
modelling difference from the reference implementation, not a detail. Trees can
still recover an interaction between the one-hot columns, but each split now
isolates one level rather than partitioning an arbitrary ordering.

### 2.3 Burn-in is already discarded

```r
# WRONG under the new engine — drops 60 of the 250 retained draws
post_sw  <- (lb_fit$model_params$burnin + 1):ncol(lb_fit$gamma_draws)
gamma_i  <- rowMeans(lb_fit$gamma_draws[, post_sw])

# RIGHT
gamma_i  <- rowMeans(lb_fit$gamma_draws)
```

`model_params$burnin` is still on the object (it records what was requested),
which is exactly why this is dangerous: the idiom keeps working and keeps
returning a number. Delete every slice; do not "fix" them to a different range.

### 2.4 Sampler settings that actually converge

The chapter's current `num_sweeps = 200, num_burnin = 60` does **not** converge
on a panel this size. Measured on a 2,500 x 20 panel (50,000 cells), 20 trees
per forest, on CPU:

| burn-in | sweeps | `n_skip` | chains | retained draws | R̂ max | ESS median | ESS min | sample | predict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 60 | 1 | 1 | 60 | — | **2.2** | 1.7 | 5.5s | 4.2s |
| 20 | 60 | 1 | 4 | 240 | **1.874** | 7.3 | 6.5 | 3.4s | 23.8s |
| 200 | 300 | 1 | 2 | 600 | 1.092 | 44.1 | 17.4 | 3.3s | 51.4s |
| 250 | 250 | 1 | 4 | 1000 | 1.060 | 251.8 | 49.9 | 3.1s | 89.9s |
| 250 | 500 | 1 | 4 | 2000 | 1.031 | 694.5 | 137.4 | 2.8s | 162.9s |
| 250 | 250 | 2 | 4 | 1000 | 1.030 | 544.6 | 231.7 | 2.7s | 98.3s |
| 250 | 250 | 4 | 4 | 1000 | 1.024 | 796.3 | 398.0 | 3.0s | 115.8s |
| **500** | **250** | **2** | **4** | **1000** | **1.012** | **743.3** | **499.8** | **3.1s** | **106.7s** |
| 500 | 500 | 2 | 4 | 2000 | 1.015 | 1272.8 | 664.8 | 2.9s | 188.4s |

Three facts the chapter should absorb, because they are counter-intuitive:

- **Burn-in is the binding constraint, not sweeps.** Twenty burn-in iterations
  leaves R̂ at 1.87 — the chain is still transient, and no number of retained
  draws fixes that. Going from 250 to 500 burn-in at otherwise identical
  settings roughly doubled the worst-case ESS (232 → 500) and was the only
  change that brought R̂ under 1.02.
- **Thinning is nearly free ESS.** The four 1000-draw rows all cost the same to
  predict (~90–116s), but differ by a factor of three in ESS. Sampling is cheap
  (~3s); *prediction* scales with retained draws. Thinning spends the cheap
  resource to save the expensive one.
- **More draws is the worst way to buy ESS.** The 2000-draw rows have the best
  absolute ESS but cost 163–188s to predict — and the 1000-draw row with longer
  burn-in beats the 2000-draw row with short burn-in on worst-case ESS
  (500 vs 137) at 60% of the prediction cost.

**These are now the engine's shipped defaults** (`num_burnin = 2000,
num_sweeps = 250, n_skip = 2, num_chains = 4`). Pass them explicitly in the
chapter's headline fit anyway: the chapter should show the reader what a
converged configuration looks like rather than hiding it behind a default, and
the settings are the subject of @sec-longbet-sampler.

Out of the box on that panel with one seed: R̂ 1.012, ESS 743 median / 500
worst-case, ATT within 0.047 of the truth, 7s to fit and 111s to predict.

**Do not take that single fit as evidence the settings are enough.** Repeating
it across four seeds gave a maximum R̂ between 1.021 and 1.100 and a worst-case
ESS between 30 and 500.

That spread is real, and it got worse once chains were started
overdispersed — which is the only way R̂ means anything. Under that diagnostic
the defaults give a median R̂ around 1.01–1.03 and a worst-case ESS in the tens.

But the quantity the chapter reports is fine. With four chains sitting at
visibly different points on the β–ν ridge (`b1` spanning −0.18 to 0.61, opposite
signs, `beta` spanning −0.10 to 3.10), the **per-chain ATT means agree to three
decimal places** at every exposure time, with between-chain spread 0.12–0.36 of
the within-chain spread:

| S | treated cells | per-chain ATT means | truth |
| ---: | ---: | :--- | ---: |
| 1 | 362 | 1.188, 1.199, 1.197, 1.205 | 1.20 |
| 8 | 250 | 3.406, 3.402, 3.414, 3.408 | 3.39 |
| 18 | 26 | 5.121, 5.114, 5.132, 5.140 | 5.09 |

**This is the chapter's argument, made concrete.** @sec-longbet-sampler already
claims the identified quantity is what to diagnose and β is not. Here is the
demonstration: chains that disagree wildly about β and `b1` agree to three
decimals about the ATT. Show both, side by side. It is a far stronger version of
the point than the prose alone.

Two practical consequences for the chapter:

- Report `n_treated` beside R̂, and consider trimming the reported event-time
  axis where support falls away — at week 14 only wave 1 contributes.
- Do not chase `rhat_max` below a threshold by re-rolling seeds. Report what the
  fit gave, and use the per-chain ATT agreement as the evidence that the
  reported quantity is sound.

### 2.5 The `att_stability()` narrative must be rewritten, not ported

This is the largest prose change in the chapter, and the current text is now
false in a specific, checkable way.

- `width_ratio` and `reliable` **no longer exist**. The split-half width ratio
  was deliberately removed: it read 0.93 for a fit that under-covered and 0.96
  for one that did not. The chapter already says this (lines ~770–784) — that
  argument is now embodied in the software rather than explained around it, and
  the prose should say so.
- **No reliability verdict is issued.** `is_reliable` is `NA`. The reference
  implementation's threshold was calibrated against measured coverage of the
  XBART *sweep* sampler and does not transfer to a Metropolis sampler with real
  chains. `ess_ok` and `rhat_ok` are threshold checks on the numbers, documented
  as exactly that.
- **"Effective sweeps" is the wrong noun now.** These are effective *draws* from
  a Markov chain with a stationary distribution, not effective sweeps of a
  sweep-based approximation. Rename throughout (lines 759, 782, 2800, 2818–2825).
- **R̂ is new and belongs here.** It requires `num_chains >= 2`; with one chain
  the function warns and reports `NA`.
- **`by_exposure$n_treated`** gives the number of treated cells behind each
  exposure time. In this rollout the late event times are reached only by wave
  1, so they carry the least data and will show the worst diagnostics — a fact
  about the design, not the sampler. Plot or tabulate it beside R̂; it is the
  difference between "the sampler failed" and "you asked a question the data
  cannot answer at week 14".

New chunk:

```r
stab <- att_stability(lb_pred)
stab$summary %>%
  transmute(`Draws (chains x sweeps)` = total_draws,
            `Chains`                  = num_chains,
            `ESS (median)`            = ess_median,
            `ESS (worst event time)`  = ess_min,
            `Tail ESS (median)`       = ess_tail_median,
            `Max R-hat`               = rhat_max,
            `Worst R-hat at event time` = rhat_max_at_exposure,
            `... backed by (cells)`   = rhat_max_n_treated,
            `MCSE of the ATT`         = mcse_median) %>%
  kable(digits = c(0, 0, 1, 1, 1, 3, 0, 0, 5), caption = paste(
    "How much of the posterior the sampler actually saw. ESS is",
    "rank-normalized and computed on the ATT series --- the identified",
    "quantity --- not on beta or on a leaf."
  ))
```

Keep the chapter's existing point that the estimate settles before the interval
does; that survives the engine change and is worth more than ever, because R̂
now gives it a second, independent witness.

### 2.6 `b0`/`b1` are sampled, so `beta_values` is scale-arbitrary

`b_scaling` defaults to `TRUE`. Two consequences:

- The chapter's ridge discussion (@sec-longbet-sampler) gets *stronger*: the
  engine now has two explicit moves along the β–ν ridge (the conjugate `b0`/`b1`
  draw and a Metropolis move on `log c`), both on by default. Describe them.
- Anything that reads `lb_fit$beta_values` as if it were the effect trajectory
  is now meaningless in level and sign — the ridge is traversed on purpose.
  Line 1645's `beta_post <- rowMeans(lb_fit$beta_values[, post_sweeps])` should
  become a statement *about* non-identification, or be replaced by the ATT from
  `get_att()`, which is identified. Line 672's CV-of-β diagnostic is a good
  illustration of the ridge and can stay — reframed as "this is what a
  non-identified coordinate looks like", with the identified ATT beside it.

---

## 3. Environment

The Python engine is **not on PyPI yet**. `.onLoad` declares
`py_require("longbet-jax>=0.1,<0.2")`, which cannot resolve until it is
published, so install it explicitly.

```bash
# Python engine
cd /home/ignacio/longbet-jax && pip install .          # or ".[cuda12]" for NVIDIA

# R package (same repo)
R -e 'remotes::install_local("/home/ignacio/longbet-jax")'
```

In R, point reticulate at the environment holding the engine before
`library(longbet)`:

```r
library(reticulate)
use_virtualenv("/home/ignacio/longbet-jax/.venv", required = TRUE)
library(longbet)
```

**Rendering.** The book renders in Docker (`book-v72`). That image does not yet
contain the engine. Either extend it, or reuse the pattern in
`/home/ignacio/longbet-jax/Dockerfile.rtest`, which installs the Python engine
into `/opt/longbet-venv` and sets `RETICULATE_PYTHON`. Verify before rendering:

```r
longbet_devices()   # e.g. "cpu:0" — proves the bridge is live
```

**Smoke test before rewriting anything.** Deliberately shorter than the shipped
defaults so it finishes in a minute; it checks that the bridge works, not that
the chain converged. Run it before touching the chapter, so a broken environment
is never mistaken for a broken port:

```r
set.seed(1); N <- 200; Tn <- 10
x <- matrix(rnorm(N * 3), N, 3)
z <- matrix(0, N, Tn); z[1:100, 5:Tn] <- 1
s <- ifelse(z == 1, pmax(col(z) - 4, 0), 0)
y <- 0.5 * x[, 1] + 1.2 * sqrt(s) + matrix(rnorm(N * Tn, 0, 0.3), N, Tn)
fit  <- longbet(y = y, x = x, z = z, t = 1:Tn, num_chains = 2,
                num_burnin = 200, num_sweeps = 200)
pred <- predict(fit, x = x, z = z, t = 1:Tn, summary_only = TRUE)
get_att(pred)$att            # should climb like 1.2 * sqrt(s)
att_stability(pred)$summary  # rhat_max should be near 1
```

---

## 4. Editorial principles

1. **LongBet is the model; `longbet-jax` is the implementation.** Present the
   DGP once, in full, and do not narrate a migration. The reader does not need
   to know there was ever a C++ backend.

2. **The estimand is a contrast against never-treated, not the BCF contrast.**
   This is the single most important thing for the chapter to state correctly:

   $$
   \tau_t(X_i, S) = b_1\,\beta_S\,\nu(X_i, S, t) \;-\; b_0\,\beta_0\,\nu(X_i, 0, t)
   $$

   Under control the exposure index is $0$, not $S$: **both** the multiplier and
   the forest's input change. It is *not* $(b_1 - b_0)\,\beta_S\,\nu$. Every
   prediction evaluates the treatment forest twice, on the factual exposure and
   on a copy of the design with $S \equiv 0$.

3. **Diagnostics go on the ATT.** The treatment term is a product the likelihood
   does not identify, so β alone has no stable scale to converge to. Report R̂
   and ESS on the ATT and CATT series, never on β or a leaf.

4. **Quote only measured numbers**, with the configuration attached. Every
   figure in §2.4 and §6 below was measured on this machine, on CPU. Do not
   quote GPU figures — none were measured.

5. **Two front doors, one engine.** The chapter stays in R. Add
   `::: {.panel-tabset}` R/Python pairs at the fit and predict chunks only;
   more than that is noise.

---

## 5. Section-by-section blueprint

Line numbers are from the current 2,974-line file and are approximate; anchor on
the `@sec-` labels.

### §1 Introduction (≈1–132)

Present the full DGP once, including every term the model actually has:

$$
Y_{it} = \alpha\,\mu(X_i, t, X^{\text{tv}}_{it})
       + b_{Z_{it}}\,\beta_{S_{it}}\,\nu(X_i, S_{it}, t, X^{\text{trt,tv}}_{it})
       + \gamma_i + \epsilon_{it}
$$

Do not drop $\alpha$ or $b_{Z_{it}}$ here and reintroduce them later — the
current draft states two different equations in two places.

Add the implementation callout, and state the sampler honestly:

```markdown
::: {.callout-note title="Implementation"}
LongBet is implemented in `longbet-jax`: a vectorized Metropolis-Hastings
sampler written in JAX on top of `bartz`, with multiple chains, R-hat, and
optional GPU execution. It is reached from R via `library(longbet)` and from
Python via `from longbet import LongBet`.
:::
```

**Acceptance:** the equation in §1 matches the one in §11's parameter guidance,
symbol for symbol.

### §2 What a panel model buys you in an experiment (≈133–185)

Unchanged. Precision / resolution / reach; "randomization makes the estimate
credible, the model makes it useful." No API surface here.

### §3 Business problem & simulation (≈186–515)

Simulation design unchanged: 3,000 sellers, four weekly launch waves plus a 40%
holdout, non-parallel baselines across three verticals, 14-week study window,
52-week business case, 12% take rate, $500/seller/year.

Changes: the one-hot encoding of §2.2, and drop the "C++ backend requires
continuous columns first" sentence.

**Replace the hand-written rollout figure with `plot_rollout()`.** The chapter
currently builds the wave/holdout tile chart with `expand_grid()` plus a
`case_when()` over `LAUNCH[wave]`. The package now does this:

```r
plot_rollout(
  z = Z[, week_study], t = week_study,
  labels = c(`4` = "W1", `6` = "W2", `8` = "W3", `10` = "W4"),
  never_treated_label = "Holdout",
  colors = c(`Not yet treated` = "grey85",
             Treated = PAL[["LongBet"]],
             `Never treated` = "grey62"),
  title = "Four randomized launch waves and a holdout"
) + theme(...)          # an ordinary ggplot; adjust as usual
```

Two reasons to prefer it over the hand-written version, both worth a sentence in
the text: the cohorts are **derived from `z`** rather than restated from
`LAUNCH`, so the figure cannot drift from the data it describes; and
`rollout_summary()` — which `plot_rollout()` is a thin layer over — carries the
same `exposure` column the sampler uses, so the picture and the event-time axis
of the ATT are guaranteed to agree. Writing it by hand is exactly where an
off-by-one in event time creeps in.

**Acceptance:** `ncol(x)` matches `colnames(x)`; no `pcat` anywhere; the rollout
figure comes from `plot_rollout()` and the wave labels match `LAUNCH`.

### §4 Fitting (`@sec-longbet-fitting`, ≈516–617)

```r
fit_time <- system.time(
  lb_fit <- longbet(
    y = y_train, x = x, z = z_train, t = week_study,
    num_burnin = 2000, num_sweeps = 250, n_skip = 2, num_chains = 4,  # the defaults
    num_trees_pr = 60, num_trees_trt = 60,
    sig_knl = 1, lambda_knl = 2,
    random_intercept = TRUE,
    random_seed = 42
  )
)
```

Note the deliberate omissions: no `pcat`, and no `x_trt` — omitting it lets both
forests share one block of columns instead of duplicating them.

Prediction:

```r
lb_pred <- predict(
  lb_fit, x = x, z = z_eval, t = week_eval,
  summary_only = TRUE, random_seed = 42
)
```

Say what `summary_only` does and does not do: it bounds **memory** — the panel
is reduced in blocks of cells so the `N x T x draws` array is never built, and
the quantiles stay exact — but it does not reduce the *time*, which scales with
retained draws.

Add the R/Python tabset here.

**Acceptance:** the fit chunk runs; `dim(lb_fit$beta_values)` is
`(S_max + 1) x 1000`; `ncol(lb_fit$gamma_draws) == 1000`.

### §5 Checking the sampler (`@sec-longbet-sampler`, ≈618–784)

The most-changed section. Rewrite around three things:

1. **This is a real Markov chain.** Four chains, each started from a draw of the
   priors on `beta`, `gamma`, `b0`, `b1` and the variances — not from a common
   point — so R̂ is a convergence statistic rather than a measure of how far two
   random streams drifted. A stationary distribution, and R̂ on the ATT at each
   event time.
2. **The β–ν ridge**, and the two moves that traverse it (§2.6). This is the
   place to show the CV of β (huge, by construction) beside the CV of the ATT
   (small) — non-identification made visible.
3. **`att_stability()`** per §2.5, including why no verdict is issued.

Trace plots must be of the **ATT at a fixed event time**, coloured by chain —
not of β.

**Acceptance:** no reference to `width_ratio`, `reliable`, or "effective
sweeps"; and the reported R̂ is the one the rendered fit produced. Do **not**
make `rhat_max < 1.05` an acceptance criterion — §2.4 shows it is routinely
exceeded at the thin end of the event-time axis on a design like this one, and a
criterion that forces the author to re-roll seeds until it passes is worse than
no criterion. Report `rhat` and `n_treated` per exposure time and let the reader
see where the estimate stops being supported.

### §6 Average effect (`@sec-longbet-scorecard`, ≈785–1058)

Unchanged in substance. `get_att(lb_pred)` now also returns `exposure`, so the
event-time axis no longer has to be reconstructed by hand.

**Keep** the `event_time_draws()` helper (chunk `alignment-helpers`, ≈792). It is
not redundant with `get_att()` — it aligns arbitrary *subsets* of units and the
known truth, which `get_att()` does not do. One correction to make while there:
it derives the event-time axis as `P <- max(rowSums(z))`, the count of treated
periods, whereas the engine derives it as elapsed time since adoption. The two
agree on this design (contiguous treatment, unit-spaced weeks) but not in
general. Align the helper with `get_att()$exposure` so a subset ATT and the
overall ATT are guaranteed to sit on the same axis.

**Acceptance:** the scorecard still compares LongBet against unadjusted and
covariate-adjusted DiD; `event_time_draws()` on all units reproduces
`get_att(lb_pred)$att` to within floating point.

### §7 What the average hides (≈1059–1221)

Heterogeneity by vertical and catalog size. `get_catt()` now returns
`catt`, `sd`, `lower`, `upper` (no `intervals` list). Line 1167's positional
`predict.longbet()` call must be named.

### §8 The business decision (`@sec-longbet-decision`, ≈1222–1509)

Economics unchanged. Line 1284–1288 needs the burn-in fix (§2.3) and a
correction: the comment says "`predict()` leaves the unit effect out" — it no
longer does. `muhats0` and `preds` now **include** $\gamma_i$, so the manual
`gamma_i` add-back must be removed or the effect is double-counted.

**Acceptance:** the profit calculation adds $\gamma_i$ exactly once.

### §9 Forecasting past the window (`@sec-longbet-forecast`, ≈1510–1697)

The GP projection with the constant mean marginalized into the kernel, so the
forecast reverts to the *estimated common level* rather than to zero. Line
1553's positional call must be named. `random_seed` is honoured now — it was
silently dropped in an earlier build, so the reproducibility claim is safe to
make.

Add the caveat the engine's docs carry: past the fitted horizon the projection
is a statement about the prior on the β/ν split, not a reading of the data.
Interval width grows there, and that is the honest signal.

### §10 Observational panels (`@sec-longbet-observational`, ≈1698–1861)

Targeted selection: `ps` goes to the prognostic forest only. Two notes worth
adding — `ps` now accepts per-cell values (`N x T`), which is the principled
form under staggered adoption where a single scalar per unit is not the right
object; and a propensity supplied at fit **must** be supplied again at predict,
by design.

Line 1787–1793: named arguments, one-hot, no `pcat`.

### §11 Deep dives (`@sec-longbet-limits`, ≈1862–2597)

- **`@sec-longbet-baseline`:** $\gamma_i$ via conjugate Gibbs, drawn *after* the
  treatment fit so a treated unit's post-adoption periods do not drag its
  baseline up. Worth adding: γ and μ(X_i) are both unit-constant, so they are
  separated only by their priors — correlation between $\hat\gamma$ and truth
  plateaus around 0.9 even with a long chain. That is a property of the model,
  not of the sampler.
- **`@sec-longbet-serial`:** unchanged.
- **`@sec-longbet-timevarying`:** the arguments are now real and named `x_tv`
  (prognostic) and `x_trt_tv` (treatment), each `N x T x P`.
- **Parameter guidance:** replace `pcat` with the one-hot note. Add
  `num_chains`, `n_skip`, `kernel_type` (`"se"`, `"matern32"`, `"matern52"`,
  `"ar1"`), and `split_time_trt`. `lambda_knl = 2` is safe: the GP conditional
  works through a Cholesky factor and never forms the precision matrix, which is
  what made that regime fragile before.

### §12 Operating characteristics (`@sec-longbet-operating`, ≈2598–2853)

- Monte Carlo replications with `summary_only = TRUE`; see the budget note
  below before choosing the replication count.
- The under-coverage demonstration at 80 sweeps still works and is now *better
  motivated*: R̂ = 1.87 at 20 burn-in is an independent, non-circular witness
  that the short run has not converged.
- **Separate the two failure modes.** A short run fails at *every* event time; a
  thinly supported event time fails only at the end of the axis. §2.4's tables
  distinguish them, and `n_treated` lets the chapter show which one it is
  looking at. Conflating them is the easy mistake here, and it would undercut
  the sampler-length argument by blaming the sampler for a design limit.
- Coverage measured on the engine's own test suite: **96.8%** for a nominal 95%
  ATT interval over 20 replications of a staggered design — slightly
  conservative. Cite with that provenance or re-measure on this design.
- Performance: re-measure on the render machine with
  `benchmarks/bench_scaling.py` (which now blocks). See §6 for why two earlier
  generations of figures for this chapter are withdrawn.
- Budget: at the shipped defaults a 42,000-cell fit-and-predict is roughly three
  minutes, so 40 Monte Carlo replications is about two hours. Cache it, or cut
  the count and say so.

### §13 Conclusion (≈2854–2975)

Installation per §3. Note that R and Python call the same engine, so results
agree exactly rather than approximately, and that the R getters delegate to the
engine for that reason.

---

## 6. Measured numbers the chapter may cite

Measured on one development machine, CPU only, at the shipped defaults (2,000
burn-in, 250 draws thinned by 2, 4 chains — 2,500 iterations per chain),
`T = 20`, 20 trees per forest, **blocking on JAX's asynchronous dispatch**:

| N | cells | compile | sample | ms/iter | predict | total | ATT ESS | s / eff. draw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 10,000 | 7.7s | 30.8s | 12.3 | 22.4s | 60.8s | 507 | 0.12 |
| 2,500 | 50,000 | 5.5s | 93.2s | 37.3 | 76.9s | 175.6s | 668 | 0.26 |

- **Sampling is the larger cost**, ahead of prediction at both sizes, and grows
  with the panel: 12.3 to 37.3 ms per iteration for five times the cells.
- **Compile time is fixed** at five to eight seconds.
- **Prediction is real but secondary**, scaling with *retained* draws rather
  than iterations.

**Two earlier generations of figures for this chapter are withdrawn**, and
neither should be quoted:

1. "~3.7s pure fit / ~7.8s total" — measured at settings that did not converge.
2. "Sampling is nearly flat across a twentyfold increase in panel size" and
   "prediction dominates" — measured without blocking on asynchronous dispatch.
   `fit()` returns before the sampler runs, so cost was attributed to whichever
   call blocked first. Both claims are the **opposite** of what a blocking
   measurement shows.

The chapter's panel is 3,000 x 14 = 42,000 cells, close to the second row, so
budget roughly three minutes per fit-and-predict at the defaults. Re-measure on
the render machine rather than quoting the table above, and if you time anything
yourself call `jax.block_until_ready()`.

Other citable results, all from the engine's test suite:

| Claim | Value | Source |
| :--- | :--- | :--- |
| ATT interval coverage, nominal 95% | 96.8% over 20 replications | `tests/test_calibration.py` |
| Residual SD recovery (truth 0.25) | 0.258 | `tests/test_dgp_recovery.py` |
| ATT correlation with truth | 0.98 | same |
| Per-cell CATT correlation with truth | 0.97 | same |
| $\hat\gamma$ correlation with truth | 0.89 (ceiling ≈0.90 on this design) | same |

**Do not claim:**

- any GPU speedup — none was measured, no GPU was available;
- any comparison against the C++ engine — it is gone and was not re-benchmarked;
- ROCm support — only CUDA 12/13 extras exist;
- convergence from a single fit. R̂ and ESS vary substantially seed to seed on
  this model (§2.4). If the chapter quotes a diagnostic, it should be the one
  the rendered fit actually produced, not one from a run that happened to look
  good.

---

## 7. Execution checklist

1. **Environment**
   - [ ] Python engine installed; `library(longbet); longbet_devices()` works.
   - [ ] §3 smoke test gives a rising ATT and R̂ near 1.
   - [ ] Render image contains the engine.

2. **Mechanical port** (do this before any prose)
   - [ ] Every `grep` in §1.1 returns nothing.
   - [ ] One-hot encoding in place; `pcat` gone.
   - [ ] All burn-in slices deleted, including inline `` `r ` `` expressions.
   - [ ] All `predict()` calls use named arguments.
   - [ ] `getMus()`/`preds` double-counting of $\gamma_i$ removed (§8).
   - [ ] Chapter renders end to end.

3. **Substantive rewrite**
   - [ ] §1 DGP stated once, in full.
   - [ ] Estimand stated as the never-treated contrast (§4.2).
   - [ ] §5 rewritten: chains, R̂, the two ridge moves, no verdict.
   - [ ] §12 numbers re-measured or re-sourced per §6.

4. **Verify**
   - [ ] `quarto render longbet.qmd` in Docker, zero errors.
   - [ ] `stab$summary$rhat_max < 1.05` in the rendered output.
   - [ ] Every number in the prose appears in a chunk output or §6.
   - [ ] Commit.

## 8. Open items owned by the author, not the implementer

- **Publish** `ignacio82/longbet-jax` and the Python package to PyPI, so §3's
  install instructions can be one line and `py_require()` can resolve.
- **Decide whether the chapter re-measures coverage** on its own design or cites
  the engine's 96.8%. The former is stronger and costs ~80 minutes of compute.
