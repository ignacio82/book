# R/longbet-common.R
# Shared setup for the LongBet chapters: environment checks, sampler budgets, the model
# settings every chapter uses, convergence diagnostics, event-time alignment, and the
# design-based benchmark the chapters compare against.

verify_longbet_environment <- function() {
  stopifnot(
    requireNamespace("longbet", quietly = TRUE),
    packageVersion("longbet") >= "1.0.0",
    !("longbet_cpp" %in% ls(asNamespace("longbet")))
  )
  lb_engine <- reticulate::import("longbet", convert = FALSE)
  stopifnot(
    reticulate::py_has_attr(lb_engine, "LongBet"),
    reticulate::py_has_attr(lb_engine, "LongBetConfig"),
    reticulate::py_has_attr(lb_engine, "LongBetMulti"),
    is.function(longbet::longbet_multi)
  )
  invisible(TRUE)
}

get_longbet_engine_fingerprint <- function() {
  # Identify the engine by what its code says, not by where it happens to be installed: hash the
  # Python sources keyed on their package-relative path, plus the R front door's version. A
  # path-sensitive hash would change when the same revision is installed in another library, which
  # would invalidate every cached fit for no reason.
  lb_engine <- reticulate::import("longbet", convert = FALSE)
  lb_python_dir <- dirname(reticulate::py_to_r(lb_engine$`__file__`))
  py_files <- list.files(lb_python_dir, pattern = "\\.py$", full.names = TRUE, recursive = TRUE)
  stopifnot(length(py_files) > 0L)
  sums <- tools::md5sum(py_files)
  names(sums) <- sub(paste0("^", normalizePath(lb_python_dir, winslash = "/"), "/?"), "",
                     normalizePath(names(sums), winslash = "/"))
  sums <- sums[order(names(sums))]
  digest::digest(list(python = sums, r_version = as.character(packageVersion("longbet"))),
                 algo = "sha256")
}

get_longbet_dependencies <- function() {
  list(
    R = R.version.string,
    packages = installed.packages()[, c("Package", "Version")],
    python = reticulate::py_run_string(
      "import importlib.metadata\nchapter_deps = {k: importlib.metadata.version(k) for k in ['jax', 'bartz', 'numpy', 'scipy', 'arviz']}"
    )$chapter_deps
  )
}

# LONGBET_QUICK=1 shrinks every sampling budget to a smoke-test size and sends the fitted
# artifacts to a separate cache, so a quick render never overwrites the published one.
longbet_quick <- function() nzchar(Sys.getenv("LONGBET_QUICK"))

# Four chains started from the prior and 2,000 burn-in sweeps everywhere. `n_skip` sets how
# long the chains run after burn-in: n_skip x 1,000 sweeps, of which every n_skip-th is kept.
lb_budget <- function(n_skip = 8) {
  if (longbet_quick()) {
    return(list(num_chains = 4L, num_burnin = 30L, num_sweeps = 30L, n_skip = 1L))
  }
  list(num_chains = 4L, num_burnin = 2000L, num_sweeps = 1000L, n_skip = as.integer(n_skip))
}

# Model settings shared by every fit in these chapters. The simulated panels have no calendar
# structure in the untreated outcome, and both forests are told so; the seller intercept
# carries each unit's level.
LB_MODEL <- list(sig_knl = 1, lambda_knl = 2, random_intercept = TRUE,
                 split_time_ps = FALSE, split_calendar_trt = FALSE)

GATE_ESS  <- 400
GATE_RHAT <- 1.01

PAL <- c(LongBet = "#0072B2", `Group-time DiD` = "#D55E00", Truth = "#000000",
         `Broad catalog` = "#009E73", `Narrow catalog` = "#CC79A7")

# A stand-in for a fitted model when only the chain layout matters (multi-outcome children,
# summaries returned by a worker process).
chain_layout <- function(num_chains, num_sweeps) {
  list(model_params = list(num_chains = num_chains, num_sweeps = num_sweeps))
}

# Diagnostics on a [targets x draws] matrix whose columns are chain-major: all retained draws
# of chain 1, then chain 2, and so on, which is the layout every fit returns.
diagnose_draws <- function(draws, fit, label) {
  draws <- as.matrix(draws)
  chains <- as.integer(fit$model_params$num_chains)
  per_chain <- as.integer(fit$model_params$num_sweeps)
  stopifnot(chains >= 2L, ncol(draws) == chains * per_chain)
  arr <- aperm(array(t(draws), c(per_chain, chains, nrow(draws))), c(2, 1, 3))
  engine <- reticulate::import("longbet", convert = FALSE)
  res <- engine$att_stability(att_draws = reticulate::r_to_py(arr), min_ess = GATE_ESS,
                              max_rhat = GATE_RHAT, alpha = 0.025, warn = FALSE)
  d <- reticulate::py_to_r(res$by_exposure)
  bytarget <- tibble::tibble(target = seq_len(nrow(draws)), ess_bulk = as.numeric(d$ess_bulk),
                             ess_tail = as.numeric(d$ess_tail), rhat = as.numeric(d$rhat),
                             mcse = as.numeric(d$mcse))
  pass <- with(bytarget, is.finite(rhat) & rhat <= GATE_RHAT & is.finite(ess_bulk) &
                 ess_bulk >= GATE_ESS & is.finite(ess_tail) & ess_tail >= GATE_ESS)
  list(summary = tibble::tibble(Quantity = label,
                                `R-hat (max)` = max(bytarget$rhat, na.rm = TRUE),
                                `Bulk ESS (min)` = min(bytarget$ess_bulk, na.rm = TRUE),
                                `Tail ESS (min)` = min(bytarget$ess_tail, na.rm = TRUE),
                                Passed = sprintf("%d of %d", sum(pass), length(pass)),
                                ok = all(pass)),
       bytarget = bytarget)
}

# Align an [n x T x draws] array of unit-level effects on time since adoption and average over
# a subset of units: returns an [S x draws] matrix.
event_time_draws <- function(tauhat, z, keep = rep(TRUE, nrow(z))) {
  P <- max(rowSums(z)); D <- dim(tauhat)[3]
  num <- matrix(0, P, D); den <- numeric(P)
  patt <- apply(z, 1, paste0, collapse = "")
  for (p in unique(patt[keep & rowSums(z) > 0])) {
    ii <- which(patt == p & keep)
    cols <- which(z[ii[1], ] == 1); k <- length(cols)
    num[1:k, ] <- num[1:k, ] + apply(tauhat[ii, cols, , drop = FALSE], c(2, 3), sum)
    den[1:k] <- den[1:k] + length(ii)
  }
  out <- num / den; out[is.nan(out)] <- NA_real_; out
}

event_time_truth <- function(m, z, keep = rep(TRUE, nrow(z))) {
  as.vector(event_time_draws(array(m, c(dim(m), 1)), z, keep))
}

curve_summary <- function(draws, truth = NULL, label = NULL) {
  out <- tibble::tibble(s = seq_len(nrow(draws)), estimate = rowMeans(draws),
                        lower = apply(draws, 1, stats::quantile, 0.025),
                        upper = apply(draws, 1, stats::quantile, 0.975))
  if (!is.null(truth)) out$truth <- truth[seq_len(nrow(draws))]
  if (!is.null(label)) out$segment <- label
  out
}

score_curve <- function(est, lo, hi, truth) {
  tibble::tibble(
    RMSE = sqrt(mean((est - truth)^2)), `Mean error` = mean(est - truth),
    `Interval width` = mean(hi - lo),
    `Contained truth` = sprintf("%d of %d", sum(truth >= lo & truth <= hi), length(truth)))
}

# Fits run in fresh worker processes, LONGBET_WORKERS at a time. Each batch gets a new cluster,
# so the memory a fit and its prediction allocate is returned when the batch ends.
lb_workers <- function() max(1L, as.integer(Sys.getenv("LONGBET_WORKERS", "1")))

run_parallel <- function(tasks, fn, export = character()) {
  run_batch <- function(batch) {
    cl <- parallel::makeCluster(length(batch))
    on.exit(parallel::stopCluster(cl), add = TRUE)
    invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages({
      library(longbet); library(dplyr); library(tibble); library(purrr) })))
    parallel::clusterExport(cl, c("LB_HELPERS", export), envir = globalenv())
    invisible(parallel::clusterEvalQ(cl, for (f in LB_HELPERS) if (file.exists(f)) source(f)))
    parallel::parLapply(cl, batch, fn)
  }
  batches <- split(seq_along(tasks), ceiling(seq_along(tasks) / lb_workers()))
  out <- unlist(lapply(batches, function(ix) run_batch(tasks[ix])), recursive = FALSE, use.names = FALSE)
  names(out) <- names(tasks)
  out
}

# ---------------------------------------------------------------------------------------
# The design-based benchmark.
#
# For each adoption cohort and each period since its launch, the change from a common
# pre-launch window, minus the same change among units that never adopt, pooled across
# cohorts with sample-size weights. This is the group-time estimator of Callaway and
# Sant'Anna specialised to a balanced panel with never-treated controls and no covariate
# adjustment, which is the case these chapters simulate. `strata` reweights the controls to
# each cohort's randomisation strata when the design has them; `keep` restricts every arm to
# a subgroup, which is how the same estimator runs inside a segment.
# ---------------------------------------------------------------------------------------
group_time_did <- function(y, cohort, periods, pre_periods, keep = NULL, strata = NULL,
                           adjust = NULL) {
  n <- nrow(y)
  if (is.null(keep)) keep <- rep(TRUE, n)
  if (is.null(strata)) strata <- factor(rep("all", n))
  colnames(y) <- periods
  D <- y - rowMeans(y[, match(pre_periods, periods), drop = FALSE])
  if (!is.null(adjust)) D <- adjust$D
  never <- !is.finite(cohort) & keep
  launches <- sort(unique(cohort[is.finite(cohort) & keep]))
  max_s <- max(vapply(launches, function(g) sum(periods >= g), integer(1)))
  purrr::map_dfr(seq_len(max_s), function(s) {
    gs <- launches[vapply(launches, function(g) (g + s - 1) <= max(periods), logical(1))]
    nt <- vapply(gs, function(g) sum(cohort == g & keep), integer(1))
    gs <- gs[nt > 1]; nt <- nt[nt > 1]
    a <- nt / sum(nt)
    col <- match(gs + s - 1, periods)
    treated_mean <- vapply(seq_along(gs), function(j) mean(D[cohort == gs[j] & keep, col[j]]), numeric(1))
    var_treated <- vapply(seq_along(gs), function(j) stats::var(D[cohort == gs[j] & keep, col[j]]) / nt[j], numeric(1))
    control_weights <- vapply(gs, function(g) {
      nt_str <- table(factor(strata[cohort == g & keep], levels = levels(strata)))
      nh_str <- table(factor(strata[never], levels = levels(strata)))
      st <- as.character(strata[never])
      as.numeric(nt_str[st] / sum(nt_str) / nh_str[st])
    }, numeric(sum(never)))
    control_combo <- as.vector((D[never, col, drop = FALSE] * control_weights) %*% a)
    est <- sum(a * treated_mean) - sum(control_combo)
    control_var <- length(control_combo) * stats::var(control_combo)
    if (!is.null(adjust)) {
      influence <- numeric(sum(never))
      for (j in seq_along(gs)) {
        target_x <- colMeans(adjust$design[cohort == gs[j] & keep, , drop = FALSE])
        contrast_x <- target_x - as.vector(crossprod(adjust$design[never, , drop = FALSE], control_weights[, j]))
        w <- control_weights[, j] + as.vector(adjust$design[never, , drop = FALSE] %*% adjust$inv_cross %*% contrast_x)
        influence <- influence + a[j] * w * D[never, col[j]] / sqrt(1 - adjust$leverage)
      }
      control_var <- sum(influence^2)
    }
    se <- sqrt(sum(a^2 * var_treated) + control_var)
    tibble::tibble(s = s, estimate = est, lower = est - 1.96 * se, upper = est + 1.96 * se)
  })
}

# Holdout-fitted covariate regression, the linear cousin of the prognostic forest: regress the
# differenced outcome on baseline covariates among never-treated units and work with residuals.
did_covariate_adjustment <- function(y, cohort, periods, pre_periods, x) {
  never <- !is.finite(cohort)
  colnames(y) <- periods
  D <- y - rowMeans(y[, match(pre_periods, periods), drop = FALSE])
  xdf <- as.data.frame(x)
  for (nm in names(xdf)) if (length(unique(xdf[[nm]])) == 2L) xdf[[nm]] <- factor(xdf[[nm]])
  design <- stats::model.matrix(~ ., data = xdf)
  inv_cross <- solve(crossprod(design[never, , drop = FALSE]))
  leverage <- rowSums((design[never, , drop = FALSE] %*% inv_cross) * design[never, , drop = FALSE])
  D_adj <- D
  for (cc in seq_len(ncol(D))) {
    fit <- stats::lm(D[never, cc] ~ ., data = xdf[never, , drop = FALSE])
    D_adj[, cc] <- D[, cc] - stats::predict(fit, newdata = xdf)
  }
  list(D = D_adj, design = design, inv_cross = inv_cross, leverage = leverage)
}

# Absolute paths to these helper files, recorded when they are sourced, so that a worker process
# can source them too and reach every helper and simulator by name.
LB_HELPERS <- normalizePath(file.path("R", c("longbet-common.R", "longbet-sim.R", "longbet-artifacts.R")),
                            mustWork = FALSE)
