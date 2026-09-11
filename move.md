# Implementation plan: longitudinal causal inference reorganization

- **Status:** Reviewed; recommended with the corrections and implementation gates below.
- **Repository:** `/home/ignacio/book` (use repository-relative paths when implementing).
- **Reviewed against:** `8c04a40302ce5542242cbd869289edaf980ea623`, September 9, 2026.
- **Audience:** The AI or maintainer implementing the reorganization.
- **Scope of this review:** Improve this plan only. The book has not been reorganized.

## 1. Recommendation and boundaries

Proceed. Bringing BSTS, synthetic control, and LongBet together gives readers a
useful route through causal questions involving time. Keeping BART and BCF
before LongBet respects its modeling prerequisites. Splitting LongBet at the
decision and forecasting sections creates three coherent reading units.

Describe the overall order as a **learning progression: foundations, study
design, models, then longitudinal applications**. It still uses more than one
organizing principle; this change does not create a pure taxonomy of causal
identification strategies. BSTS and synthetic control are alternatives for some
questions, not prerequisites for LongBet. Aggregate versus individual data is
not a strict boundary between these methods.

Moving matching to the end of the experiments part is reasonable **if the text
explicitly marks the transition to observational identification**. Matching is
one adjustment method within target trial emulation, not another name for that
framework or a way to recreate randomization. A target trial also specifies its
population, strategies, assignment, follow-up, outcomes, estimand, and analysis.
See [Hernán and Robins (2016)](https://dlab.epfl.ch/teaching/spring2022/cs727/papers/hernan2016using.pdf).

Preserve the numerical examples and their limitations. Do not change estimands,
priors, sampler budgets, seeds, policy thresholds, or diagnostic criteria as
part of reorganizing the book. Do not add a full DiD, event-study, or longitudinal
g-methods chapter in this change. Provide a brief orientation to the methods
covered and the important gaps instead.

### Corrections to the original proposal

| Finding | Required correction |
| --- | --- |
| Later LongBet sections are described as independent simulations. | They reuse the rollout, helpers, and sometimes fitted objects. Extract shared dependencies before splitting. |
| The proposed `sim_data` object does not exist. | Use an explicit data contract built from the actual objects in section 4. |
| The proposed export includes `lb_pred` at the end of chapter 1. | `counterfactual` already calls `rm(lb_pred)`. Preserve required observed summaries before that point; retain the memory cleanup. |
| Fixed line-number slicing is called deterministic. | Split at unique heading markers, retaining surrounding fences, callouts, inline R, and labels. |
| Keeping `longbet.html` is said to preserve all bookmarks. | Moved section, figure, table, and equation fragments need compatibility routing too. |
| Keeping `other_methods.qmd` on disk is treated as sufficient. | A source omitted from the book does not guarantee published HTML. Generate and verify a compatibility page. |
| `freeze: auto` is treated as a shared data cache. | Freeze, knitr caches, and reusable R artifacts have different roles. See section 5. |
| The runbook stages sources but omits frozen results. | Regenerate affected `_freeze/` output; `docs/` and local caches are ignored. |
| The execution environment is assumed consistent. | Docker uses JAX; current CI installs `ignacio82/longbet@v0.7.2`. Validate frozen publishing separately from execution. |

## 2. Target structure and editorial changes

Use these part titles consistently in landing-page metadata and prose:

| Part | Landing page | Title / role |
| --- | --- | --- |
| 1 | `index.qmd` | Retain the existing Introduction / foundations content. |
| 2 | `rct.qmd` | **Randomized Experiments and Observational Comparisons** |
| 3 | `glm.qmd` | **Generalized Linear Models** |
| 4 | `stochastictrees.qmd` | **Stochastic Trees and Heterogeneous Effects** |
| 5 | `longitudinal.qmd` | **Longitudinal and Panel Causal Inference** |

Replace only `book.chapters` with this structure; preserve unrelated project,
format, bibliography, sharing, analytics, sidebar, and publication settings:

```yaml
  chapters:
    - part: index.qmd
      chapters:
        - chapter_01.qmd
        - potential_outcome.qmd
        - baseline.qmd
        - bayes101.qmd
        - decisions_first.qmd
        - rce.qmd
        - surrogates.qmd
    - part: rct.qmd
      chapters:
        - rct_basic.qmd
        - factorial.qmd
        - iv.qmd
        - coordination.qmd
        - strength.qmd
        - matching.qmd
    - part: glm.qmd
      chapters:
        - blm.qmd
        - meta-analysis.qmd
        - beyond_normality.qmd
        - hurdle.qmd
    - part: stochastictrees.qmd
      chapters:
        - bart.qmd
        - bcf.qmd
        - causalPropensity.qmd
        - phobart.qmd
        - bad.qmd
    - part: longitudinal.qmd
      chapters:
        - causalimpact.qmd
        - bsynth.qmd
        - longbet.qmd
        - longbet_decisions.qmd
        - longbet_extensions.qmd
    - references.qmd
```

This retains all 25 existing content chapters and adds two, for **27 content
chapters**, excluding the five part pages and `references.qmd`. Only the obsolete
part landing page is retired. All existing content-chapter filenames survive.

### Matching and the experiments landing page

- Update `rct.qmd`'s title, description, and share description. Keep `rct.html`.
  Introduce matching as the final bridge to comparisons without randomized
  assignment; make the additional identifying assumptions explicit.
- Keep `matching.qmd` titled **Matching**, and retain its examples and code.
  Add a short section before the mechanics explaining how a comparison group
  fits into a target trial protocol. Include a small table applying eligibility,
  treatment/comparator, time zero and follow-up, outcome, estimand, and analysis
  to the existing bootcamp example. Label new protocol details as illustrative;
  do not imply the existing simulation implements a complete longitudinal
  emulation. Add the reference above to `references.bib` after checking for an
  existing entry.
- State that covariates used for matching precede treatment; observed balance
  does not establish absence of unmeasured confounding. Assess covariate balance
  and overlap, not just the distribution of a fitted propensity score. Preserve
  the King–Nielsen discussion and the Charles/Ozzy example.
- Specify the comparison's target population, and explain that exclusions for
  lack of overlap can change it. Do not promise an ATE if the example estimates
  an ATT or an effect in the matched population.
- Add a forward link to `bcf.qmd`: propensity information can help mitigate
  regularization-induced confounding under the model's assumptions. Do not say
  that including a propensity score eliminates confounding.

### The longitudinal landing page

Create `longitudinal.qmd` with this metadata, then write a concise orientation
meeting the requirements below:

```yaml
---
title: "Longitudinal and Panel Causal Inference"
description: "Causal effects across time: aggregate interventions, donor panels, and staggered rollouts with heterogeneous effects."
aliases:
  - /other_methods.html
share:
  permalink: "https://book.martinez.fyi/longitudinal.html"
  description: "Causal inference across time: assumptions, estimation, and decisions"
  linkedin: true
  email: true
  mastodon: true
---
```

1. Open with regional campaigns, market interventions, and staggered feature
   rollouts. Distinguish calendar time from time since adoption.
2. Add a method-selection table: **question/data**, **chapter**, and **source of
   identification / central assumptions**. Cover BSTS, synthetic control,
   randomized LongBet rollouts, and observational LongBet panels. Explain
   unaffected controls and stability of the counterfactual relationship for
   BSTS; donor suitability, pretreatment fit, no anticipation, and no donor
   contamination for synthetic control; randomized launch order for the main
   LongBet example; and the stronger adoption-timing assumptions for its
   observational extension. Derive details from the chapters and their cited
   sources; do not give all methods a universal parallel-trends assumption.
3. Explain that pretreatment history helps model baselines but does not
   automatically remove unobserved or time-varying confounding. Separate serial
   dependence, a modeling and uncertainty issue, from confounding itself.
4. Link to all five chapters. Label decisions and extensions as continuations
   of the core LongBet example. Give a shortcut for readers familiar with BCF;
   do not imply that PhoBART or BAD is required for LongBet.
5. Locate DiD/event studies as relevant neighboring methods and existing
   benchmarks. Describe a selected toolkit, not comprehensive coverage of
   longitudinal causal inference. Adding time-varying covariates to a model
   does not by itself solve treatment-confounder feedback.
6. Include a visible link to `matching.qmd` for readers arriving from the old
   non-experimental methods URL.

Use links to `.qmd` sources and named section references rather than hardcoded
part/chapter numbers. Bayesian intervals remain conditional on modeling and
identification assumptions; avoid promising automatically calibrated uncertainty.

### Other navigation and summaries

- In `stochastictrees.qmd`, update title/descriptions, replace the paragraph
  listing BART/BCF/LongBet with a roadmap for BART, BCF, causal propensity,
  PhoBART, and BAD, and link forward to `longbet.qmd`. Avoid describing every
  chapter here as cross-sectional causal estimation: ordinal modeling and
  adaptive designs have different roles. Do not imply that every package
  mentioned uses the same engine or sampler.
- Add contextual transitions in `causalimpact.qmd` and `bsynth.qmd` if needed to
  orient readers to the new part; retain their filenames and examples.
- Update `_quarto.yml`'s abstract, `README.md`'s topic/build descriptions, and
  `llms.txt`'s abstract to agree with the new order and shared execution workflow.
  Keep the business decision focus. Audit `index.qmd` and other source prose;
  change only descriptions affected by this migration.
- Remove `other_methods.qmd` from the book and remove its obsolete source once
  the `longitudinal.qmd` alias is verified. Its history remains in Git. Generated
  `docs/other_methods.html` must exist after a clean render.

## 3. Split LongBet by heading, preserving the analysis

The reviewed source has **3,361 lines, 161,706 bytes, and 62 executable R chunks**.
These are inventory checks, not slicing instructions. Later revisions may
legitimately change them.

Read the complete original file into memory or a temporary snapshot **before
writing any replacement**. Find each marker exactly once, in increasing order:

```text
## The Decision {#sec-longbet-decision}
## Projecting Past the End of the Study {#sec-longbet-forecast}
```

Use half-open ranges: core ends immediately before the first marker; decisions
begins there and ends before the second; extensions begins at the second and
retains the rest. In the reviewed source these start at lines 1333 and 2308.
Verify both markers fall outside fenced code/divs. Do not interpret the `## R`
and `## Python` tab headings as chapter boundaries.

| Destination | Exact title | Content retained |
| --- | --- | --- |
| `longbet.qmd` | **LongBet: Dynamic Treatment Effects in Staggered Rollouts** | Introduction, model/features, randomized design, rollout DGP, fit, sampler checks, ATT/DiD/ANCOVA, heterogeneous effects. |
| `longbet_decisions.qmd` | **LongBet: Decisions and Multiple Outcomes** | `sec-longbet-decision` through the multi-outcome innovation-correlation discussion. |
| `longbet_extensions.qmd` | **LongBet: Forecasting, Observational Panels, and Diagnostics** | Forecasting, observational panels, baseline variants, serial correlation, time-varying covariates, operating characteristics, absorbing treatment, design notes, conclusion, further reading. |

For all three chapters:

- Supply one YAML frontmatter block. Preserve `format.html.css: longbet.css`,
  sharing flags, and citations; set title, description, and share permalink to
  the chapter's own content and filename. The CSS is already generic enough to
  share; change selectors only if visual validation finds a need.
- Retain existing explicit section, figure, table, and equation labels in their
  destination. Preserve chunk labels/settings unless a documented dependency
  extraction requires changes. Cross-reference labels must be unique book-wide.
- Start with a brief orientation, prerequisites, and links to the other two
  chapters. Continuations must explain the reused seller panel, outcome scale,
  calendar, estimand, and objects they load.
- Keep the R/Python tabs and visible mathematical/modeling code. Shared helpers
  must not turn the tutorial into unexplained `source()` calls. Use one
  executable definition per helper/model specification; display that source or
  explain the call where taught rather than maintaining a second code copy.
- Add short conclusions to core and decisions. Rewrite the inherited final
  conclusion to summarize the sequence, replacing stale “above” and “this
  chapter” references. Audit inline R expressions as well as fenced chunks.

Keep a concise assumptions reminder in core and alongside decisions even though
the longer practical discussion moves to chapter 3. Preserve the distinctions
between observed effects, model-based subgroup generalization, future
projections, and the annual economic scenario. Do not recast the assumed annual
tail as an identified forecast or remove failed-diagnostic caveats.

## 4. Shared execution contract — implement before the split

**Each chapter must execute in a fresh R process, in any order.** A reader may
start by rendering extensions. It must not depend on a previous chapter's R
session or require that chapter to have been rendered first.

Use shared R modules plus regenerable local artifacts. Do not source or knit an
entire earlier `.qmd` to obtain its environment. Do not use `save.image()`, export
`ls()` indiscriminately, or substitute invented “fast evaluation draws” for the
fitted posterior.

### Verified incoming dependencies

This identifies important dependencies, not every free variable. Complete the
inventory, including inline R, during extraction.

| Consumer | Dependencies outside its proposed chapter |
| --- | --- |
| All chapters | Setup imports, JAX engine guards, engine/dependency fingerprints, `diagnose_draws()`, `PAL`, shared simulation/helper definitions. |
| Decision/economics | `lb_fit`, `pred_all`, `tau_hat_draws`, `tau_hat`, `tau_truth_S`, `S_target`, `n`, `week_study`, `mu0_true`, `w_i`, `h_grow()`, `h_fade()`, and seller attributes including `listings` and `broad`. |
| Multi-outcome simulation | `n`, `weeks_all`, `week_study`, `tmat`, `S`, `level_i`, `drift_i`, `season()`, `vertical`, `listings`, `fulfilled`, `y0`, `w_i`, `tau_true`, `x`, `z_train`, and effect-shape helpers. It takes every fourth original seller. |
| Forecasting | `lb_fit`, `lb_att`, `x`, `z_train`, `z_ext`, study/future calendars, `S_max`, `truth_att`, `truth_att_ext`, and diagnostics. |
| Observational simulation | Original seller characteristics, calendar, latent baseline parameters, `tmat`, `Tn`, `LAUNCH`, effect-shape helpers, `S_observed`, and diagnostics. It changes adoption and outcomes, not the entire underlying population. |
| Baseline variants / serial dependence | `x`, `y_train`, `z_train`, `week_study`, `lb_fit`, `lb_att`, `truth_att`, `S_observed`, `y`, and `is_hold`; residual ACF also needs `v_full_no` from earlier in extensions. |
| Time-varying covariates | `n`, `week_study`, `tau_true`, `y0`, `x`, `z_train`, and diagnostics. |
| Monte Carlo | `week_base`, `week_study`, `LAUNCH`, `season()`, `h_grow()`, `h_fade()`, `event_time_truth()`, `did_event_study()`, and `diagnose_draws()`. Its local DGP still uses these outer definitions. |

### Proposed modules

These are new files to implement, not existing repository APIs:

| File | Responsibility |
| --- | --- |
| `R/longbet-common.R` | Shared initialization/compatibility checks, palette, diagnostics, event-time alignment, effect-shape and seasonal functions, benchmark helpers. No fits merely from sourcing. |
| `R/longbet-rollout.R` | Deterministic rollout builder with explicit seed/calendar inputs and a named return list. One definition of the original DGP and model configuration. |
| `R/longbet-artifacts.R` | Load-or-build functions for the core fit and reusable predictions/summaries; validation and fingerprints. No chapter rendering or fits merely from sourcing. |

The rollout return value must explicitly include calendars and `LAUNCH`; seller
attributes/assignment (`vertical`, `fulfilled`, `listings`, `seller_sd`, `strata`,
`wave`, `launch`, `broad`); baseline/effect parameters (`level_i`, `drift_i`,
`w_i`); panels (`tmat`, `S`, `Z`, `mu0_true`, `y0`, `tau_true`, `y`); and model
inputs (`x`, `y_train`, `z_train`, `z_ext`). Include dimensions and lookback
summaries needed by retained code. Keep helpers in source, not serialized
environments. Derive `truth_att`, `truth_att_ext`, `S_observed`, `S_max`, and
`is_hold` from this returned data.

Qualify values through the return list or bind an explicit, validated list of
names within the chapter execution environment. Make imported dependencies
visible to the implementer.

Preserve the original R random-draw sequence starting with seed 1982, scoped
seed 271828 for the multi-outcome supplement, seed 8123 for observational
scenarios, seed 515 for promotions, and all current fit/prediction/Monte Carlo
seeds. Record `RNGkind()`. Moving helpers must not consume random numbers.
Verify the original DGP and extracted builder agree before expensive fitting.

`did_event_study()` closes over chapter variables. Monte Carlo copies it and
rebinds its environment with `environment(mc_did) <- environment()`. Make inputs
explicit and update both callers, or intentionally preserve and verify closure
behavior. Moving it to another environment can silently change the estimator.

### Reusable artifacts

Use `cache/longbet/`, already ignored by Git, with a small shared load-or-build
implementation rather than a new workflow framework:

1. **Core fit:** `lb_fit` for the original `y_train/x/z_train/calendar` and exact
   current sampler specification. All three chapters reuse this fit.
2. **Observed-panel results:** preserve `lb_att` and observed diagnostics before
   `lb_pred` is removed. Retain or regenerate full observed predictions only for
   consumers needing their arrays.
3. **Common-launch predictions:** `pred_all` under `z_all` (everyone launches in
   week 11, prediction seed 1), or the exact subset of arrays/derived summaries
   needed by the existing heterogeneous-effect and economic calculations.
   Preserve `S_target`, terminal effect draws, and draw correspondence across
   sellers/weeks. This is a different treatment scenario from `lb_pred`.

Do not save all three full objects into one mandatory payload. Load components
only when needed; avoid retaining full observed and common-launch arrays
simultaneously. If reducing to policy inputs, preserve the operations and order:
posterior medians of baseline predictions, residual variance adjustment,
draw-wise `expm1()` sums, and terminal draws. Do not replace an expectation of a
nonlinear function with the function of an expectation.

Forecast sensitivity predictions belong to extensions; multi-outcome fits
belong to decisions. Each chapter keeps its expensive supplements cached. The
shared core loader must not run these supplements.

Each artifact needs a named envelope: schema version, dependency fingerprint,
model/prediction configuration including seeds, and payload. Keys must depend
on producing source/data, the R/Python engine source fingerprint, relevant
dependency versions, and upstream artifact keys. Prediction keys also
distinguish scenario, horizon, and settings. Business thresholds and prose
changes must not invalidate the core fit artifact.

Create directories on demand. For missing/incompatible artifacts, announce the
rebuild and compute the real result. For corruption, report and rebuild or fail
clearly. Validate fields, dimensions, calendar/row alignment, and chain-major
draw order. Write a temporary file in the same directory and rename only after
successful serialization/validation. Never reuse an interrupted fit. Render
chapters sequentially so two processes do not build the same artifact at once.

Use `saveRDS()`/`readRDS()` with the JAX R wrapper's supported serialized model
state, not a naked reticulate object. The nearby `longbet-jax` source inspected
during review implements `raw_model` serialization and rehydration; that is
evidence for the approach, **not proof about whichever Docker image is used
later**. Check the installed implementation and verify save/load/predict across
two fresh R processes with the same prediction inputs/seed. Exercise
rehydration, not merely whether `readRDS()` returns a list.

## 5. Caching, environment, and publication

Keep project-level `execute.freeze: auto`. It reuses execution output during a
full project render; a single-document render executes code and may reuse valid
knitr chunks. Frozen output contains rendered results, not a shared R session.
See [Quarto execution management](https://quarto.org/docs/projects/code-execution.html).

| Location | Role and migration requirements |
| --- | --- |
| `_freeze/` | Tracked output for publishing. Regenerate `longbet/`, add both new chapter directories, and refresh other changed executable chapters. Let Quarto write metadata/figures; do not split or hand-edit old `html.json` files to impersonate execution. |
| `*_cache/` | Ignored per-document knitr caches. A moved chunk may execute again in its new document even if its text is unchanged. |
| `cache/longbet/` | Ignored validated model artifacts for reuse across independently executed chapters, including when a prior chapter is frozen. |
| `docs/` | Ignored build output. Verify locally; do not force-add it. |

Setup currently hashes `knitr::current_input()` for `cache.extra`. Include shared
R source hashes and upstream artifact keys after extraction. Conservatively
invalidating a knitr chunk is acceptable if its loader reuses the unchanged
expensive artifact. External helper edits can still be missed by document-level
freeze because setup never executes. Document in `README.md`: **after changing
shared LongBet R files, explicitly render all three LongBet chapters to refresh
their frozen outputs**. A successful frozen build does not validate new helpers.

Use the Docker route supplied here. At review time, the local `book` image
reported Quarto **1.9.38**, matching CI; host `quarto` and `Rscript` were not on
PATH. Recheck during implementation:

```bash
docker run --rm book quarto --version
docker run --rm -v "$PWD:/book" book quarto inspect
```

Verify the installed JAX engine with existing chapter guards, including
`LongBetMulti`, `longbet_multi`, and `PRECISION_CACHE_VERSION`. Record image ID
and installed LongBet revision/dependency fingerprint. If rebuilding, use a
verified commit via `--build-arg LONGBET_REF=...` rather than introducing an
unrelated engine upgrade. Preserve engine source hashing: package version alone
does not identify sampler fixes.

Current `.github/workflows/publish.yml` installs retired C++ package
`ignacio82/longbet@v0.7.2`; it cannot freshly execute these JAX chapters. Retain
the repository's publish-from-committed-freeze workflow for this migration and
prove complete frozen results work in a clean copy without local caches. Do
not rely on CI to create missing LongBet output. Aligning CI with JAX is a
separate infrastructure change if fresh CI execution is desired; do not remove
compatibility guards to make old CI pass.

## 6. Cross-references and compatibility URLs

### Internal references

Inventory labels and source links before moving content. Search tracked source
and metadata, not generated `docs/`, `*.knit.md`, or `_freeze/` prose. Include
`@sec-*`, `@fig-*`, `@tbl-*`, `@eq-*`, local `#fragment` links, `.qmd` links,
hardcoded `.html` URLs, and “earlier/later in this chapter” phrasing.

Retain labels in destination chapters; update direct links to the new file.
Quarto resolves book-wide cross-references; a final full-book render must confirm
destinations and numbering. See
[Quarto book cross-references](https://quarto.org/docs/books/book-crossrefs.html).

### Old part URL

`aliases: [/other_methods.html]` on `longitudinal.qmd` generates the old page.
Check it after rendering into empty output. The destination must show links to
the time-based methods and matching. Do not redirect still-existing
`causalimpact.html`, `bsynth.html`, or `matching.html`; their content URLs stay.

### Moved LongBet fragments

Keep bare `longbet.html` and retained fragments working. Route fragments for
moved content to the corresponding new chapter **and the same fragment**:

| Old URL | New destination |
| --- | --- |
| `longbet.html#sec-longbet-decision` | `longbet_decisions.html#sec-longbet-decision` |
| `longbet.html#sec-longbet-multi` | `longbet_decisions.html#sec-longbet-multi` |
| `longbet.html#fig-longbet-multi-joint` | `longbet_decisions.html#fig-longbet-multi-joint` |
| `longbet.html#sec-longbet-forecast` | `longbet_extensions.html#sec-longbet-forecast` |
| `longbet.html#sec-longbet-serial` | `longbet_extensions.html#sec-longbet-serial` |
| `longbet.html#sec-longbet-operating` | `longbet_extensions.html#sec-longbet-operating` |

Cover every moved authored heading and public figure/table/equation target,
including automatic heading IDs. Preserve automatic IDs explicitly when needed:
repeated headings or extraction can change them. Discover final public IDs from
the original render; chunk names alone are insufficient. Quarto UI and generated
code-line anchors are outside this compatibility guarantee; report that limit.

Implement page-local `assets/longbet-legacy-links.js`, included only by
`longbet.qmd`, with an explicit allowlisted map from old fragment to new file
derived from the inventory. On initial load and `hashchange`, route only mapped
fragments with `location.replace()`, retaining query string and fragment. Leave
bare URLs, retained anchors, and unknown hashes alone; malformed encoding must
not throw. Do not route by the broad `sec-longbet-*` prefix: several such
sections stay. Add visible links near the core chapter's top to both
continuations for readers without JavaScript.

Why a script: Quarto documents whole-page and fragment aliases in
[its redirects guide](https://quarto.org/docs/websites/website-navigation.html#redirects).
However, a minimal book fixture tested during review with Quarto 1.9.38
generated a whole-page alias successfully but did **not** add a fragment router
to an old page that remained an active chapter. Do not assume aliases alone
preserve fragments on the retained `longbet.html`.

## 7. Implementation sequence and validation gates

Complete each gate before advancing. These steps describe local implementation
when this plan is assigned to an implementing agent; pushing, publishing, and
unrelated analysis changes are outside the reorganization.

### Gate A — baseline and inventory

1. Read applicable instructions and inspect `git status`. Preserve unrelated
   work. At review time `move.md` was untracked. A branch does not back up
   untracked files; snapshot original plan/source outside render inputs if
   needed. Create a branch if appropriate; do not reset the user's checkout.
2. Record source revision, chapter list, section/chunk/label inventory, original
   public anchor locations, and environment fingerprints. Read the original
   file fully before overwriting it.
3. Capture baseline DGP outputs and key numerical summaries from matching
   source/engine results. `.knit.md`, `docs/`, and `_freeze/` can be stale: verify
   provenance before treating them as baselines. If no valid baseline exists,
   establish one with the recorded environment instead of promising equality to
   unverified output.

### Gate B — dependencies while the monolith still works

1. Implement the shared modules/artifact contract. Adapt the original monolith
   to use them before moving sections. Preserve visible teaching code.
2. Check original/extracted DGP equality for assignment, row order, covariates,
   observed outcomes, and truth with identical seed/environment. Verify DiD and
   adjusted benchmark equality too.
3. Validate artifact invalidation and save/load/predict in separate R processes.
   Reuse a verified fit where possible. Tiny fits may test mechanics but cannot
   validate scientific conclusions or supply published chapter draws.
4. Preserve seller × time × draw axes for full R predictions, target × draw for
   diagnostic inputs, and chain-major draw order. Check scenarios/horizons;
   compatible dimensions alone are insufficient.

### Gate C — split and integrate

1. Split at section 3's markers. Wire each chapter's setup to its dependencies
   through shared modules/loaders. Use descriptive setup chunk names. Every
   original chunk/substantive section must have one destination or a documented
   shared-source replacement.
2. Apply landing-page, matching, summary, and `_quarto.yml` edits. Preserve
   filenames, citation keys, frontmatter conventions, tabs, and CSS.
3. Implement internal links, old part alias, and LongBet fragment map. Add
   `/longbet_decisions_files/` and `/longbet_extensions_files/` to `.gitignore`,
   matching the existing `/longbet_files/` rule.
4. Inspect the project and parse extracted R code. `quarto inspect` checks
   configuration; it does not prove chapter execution or link resolution.

### Gate D — execution and numerical checks

1. In a disposable working copy, execute `longbet_extensions.qmd` first with
   empty LongBet knitr/artifact caches. This proves it builds core dependencies
   without rendering earlier chapters. Then execute decisions and core in fresh
   R processes, retaining shared artifacts. Verify core-fit reuse and record
   its build count/key. Run sequentially to bound memory.
2. Compare verified baseline outputs: rollout/truth; event-time ATT/benchmarks;
   seller effects; policy table/rankings/sensitivity; multi-outcome probabilities
   and recovery; forecast grid; observational/baseline comparisons; promotion
   results; Monte Carlo summaries; sampler flags. Reusing the same posterior
   should preserve derived values up to floating-point tolerance. If re-fitting
   introduces platform variation, quantify against Monte Carlo error and explain
   changed displayed values; do not silently accept changed conclusions.
3. Diagnostics must remain computed, reported, and interpreted faithfully.
   Scientific diagnostics may legitimately fail in the original example. The
   migration passes by preserving those failures and qualifications, not by
   changing thresholds or claiming every fit converged.
4. Exercise warm-cache behavior: repeated requests must not resample the core
   fit. Changed simulation seed/model specification must invalidate the right
   artifact; changed policy thresholds must reuse the fit.

The empty-cache execution is the expensive computation check. Reuse verified
artifacts and compatible knitr results for final execution; do not repeat cold
MCMC runs without changed inputs or an unresolved failure.

### Gate E — final rendering and links

After final source edits, refresh affected chapters using the existing wrapper:

```bash
./render.sh rct.qmd
./render.sh matching.qmd
./render.sh stochastictrees.qmd
./render.sh longitudinal.qmd
./render.sh longbet.qmd
./render.sh longbet_decisions.qmd
./render.sh longbet_extensions.qmd
./render.sh
```

Also render `causalimpact.qmd`, `bsynth.qmd`, or `index.qmd` if their source
changed. A chapter already executed with final source at Gate D need not
execute again merely to repeat this list.

Make a clean publication copy containing intended tracked sources and fresh
`_freeze/`, **without** `docs/`, `.quarto/`, `cache/`, or `*_cache/`, and perform
a full-book render. Include new files even before committing; an export of HEAD
alone omits them. Confirm frozen LongBet results are reused and no local model
artifacts are needed. Do not run `make clean` in the main checkout: it removes
all `_freeze`, caches, and `docs`, including unrelated results.

Check the site, ideally with a local HTTP server and browser:

- Five ordered parts, 27 content chapters, references, correct previous/next
  navigation, and each chapter listed once.
- No new unresolved citations/cross-references; every moved label resolves to
  its intended page/numbered target. Record unrelated existing warnings.
- Bare `longbet.html`, retained fragments, moved section/figure/table fragments,
  unknown hashes, and moved hashes with query strings behave correctly. Verify
  every mapped destination exists; exercise representative routes in a browser,
  including a same-page hash change.
- `other_methods.html` exists in clean output and reaches the new landing page;
  matching is easy to find there.
- Correct titles, descriptions, share URLs, equation/table overflow behavior,
  mobile layout, plots, and R/Python tabs in all three LongBet chapters.
- Fresh `_freeze/` results and referenced figures for changed executable
  chapters. No reliance on stale moved figures. Remove only obsolete artifacts
  of changed chapters after the clean render proves them unnecessary.

### Gate F — handoff

Review the final diff and `git diff --check`. List sources, shared modules,
compatibility script, metadata, and affected `_freeze/` changes. Do not stage
the whole repository indiscriminately or force-add ignored models/HTML. Commit
only if requested or already authorized by the working convention; no automatic
push or publication is needed.

Report commands/checks passed, environment, preservation of baseline values and
diagnostic interpretations, URLs tested, and unverified environment-dependent
behavior. If a gate cannot run, leave it explicitly incomplete rather than
declaring implementation complete or the book ready to publish.

## 8. Definition of done

- [ ] Target navigation implemented; every existing content chapter remains.
- [ ] Matching's observational status and relationship to target trial emulation
      accurately explained in the chapter and part introduction.
- [ ] Three LongBet chapters preserve the original analysis and limitations.
- [ ] Shared dependencies have one executable source and explicit data contracts.
- [ ] Any LongBet chapter executes first in a fresh R process; repeated requests
      reuse valid artifacts and stale inputs invalidate them.
- [ ] Serialized fits predict in a new process; draw alignment survives.
- [ ] Internal references and promised legacy URLs work in a clean site build.
- [ ] Summaries, metadata, sharing URLs, README guidance, and styling agree.
- [ ] Fresh frozen results support publishing without local model caches.
- [ ] Validation evidence and remaining limitations recorded for the user.
