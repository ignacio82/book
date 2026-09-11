# R/longbet-common.R
# Shared initialization, compatibility checks, palette, diagnostics, event-time alignment,
# effect-shape and seasonal functions, and benchmark helpers.

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
  lb_io <- reticulate::import("longbet._io", convert = TRUE)
  stopifnot(identical(as.integer(lb_io$PRECISION_CACHE_VERSION), 1L))
  invisible(TRUE)
}

get_longbet_engine_fingerprint <- function() {
  lb_engine <- reticulate::import("longbet", convert = FALSE)
  lb_python_dir <- dirname(reticulate::py_to_r(lb_engine$`__file__`))
  lb_source_files <- sort(c(
    list.files(lb_python_dir, pattern = "\\.py$", full.names = TRUE, recursive = TRUE),
    list.files(system.file("R", package = "longbet"), full.names = TRUE),
    system.file("contract", "longbet-api.yaml", package = "longbet")
  ))
  stopifnot(length(lb_source_files) > 0L, all(file.exists(lb_source_files)))
  digest::digest(tools::md5sum(lb_source_files), algo = "sha256")
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

PAL <- c(
  LongBet = "#0072B2",
  `DiD vs holdout` = "#D55E00",
  Truth = "#000000",
  `Broad catalog` = "#009E73",
  `Narrow catalog` = "#CC79A7"
)

# Diagnose target-by-draw matrices. Columns must retain the fit's chain-major
# order: all draws of chain 1, then all draws of chain 2, and so on.
diagnose_draws <- function(draws, fit, label) {
  draws <- as.matrix(draws)
  chains <- as.integer(fit$model_params$num_chains)
  per_chain <- as.integer(fit$model_params$num_sweeps)
  stopifnot(
    is.numeric(draws), nrow(draws) > 0L,
    length(chains) == 1L, !is.na(chains), chains >= 2L,
    length(per_chain) == 1L, !is.na(per_chain), per_chain >= 4L,
    ncol(draws) == chains * per_chain,
    length(label) == 1L, !is.na(label)
  )

  # R reads matrices column-major, so transpose first. The intermediate axes
  # are draw-within-chain, chain, target; then put chain first for ArviZ.
  arr <- aperm(array(t(draws), c(per_chain, chains, nrow(draws))), c(2, 1, 3))
  engine <- reticulate::import("longbet", convert = FALSE)
  result <- engine$att_stability(
    att_draws = reticulate::r_to_py(arr),
    min_ess = 400, max_rhat = 1.01, alpha = 0.025, warn = FALSE
  )
  diagnostics <- reticulate::py_to_r(result$by_exposure)
  bytarget <- data.frame(
    target = seq_len(nrow(draws)),
    ess_bulk = as.numeric(diagnostics$ess_bulk),
    ess_tail = as.numeric(diagnostics$ess_tail),
    rhat = as.numeric(diagnostics$rhat),
    mcse = as.numeric(diagnostics$mcse)
  )
  finite_extreme <- function(x, fn) {
    if (all(is.na(x))) return(NA_real_)
    fn(x, na.rm = TRUE)
  }
  ok <- with(bytarget,
    all(is.finite(rhat) & rhat <= 1.01 &
        is.finite(ess_bulk) & ess_bulk >= 400 &
        is.finite(ess_tail) & ess_tail >= 400))
  list(
    summary = data.frame(
      label = as.character(label),
      rhat_max = finite_extreme(bytarget$rhat, max),
      ess_min = finite_extreme(bytarget$ess_bulk, min),
      ess_tail_min = finite_extreme(bytarget$ess_tail, min),
      diagnostic_ok = ok
    ),
    bytarget = bytarget
  )
}

# Align an [n x T x draws] array of unit-level effects on time-since-adoption
# and average over a subset of units, returning a [S x draws] matrix.
event_time_draws <- function(tauhat, z, keep = rep(TRUE, nrow(z))) {
  P <- max(rowSums(z))
  D <- dim(tauhat)[3]
  num <- matrix(0, P, D)
  den <- numeric(P)
  patt <- apply(z, 1, paste0, collapse = "")
  for (p in unique(patt[keep & rowSums(z) > 0])) {
    ii <- which(patt == p & keep)
    cols <- which(z[ii[1], ] == 1)
    k <- length(cols)
    num[1:k, ] <- num[1:k, ] + apply(tauhat[ii, cols, , drop = FALSE], c(2, 3), sum)
    den[1:k] <- den[1:k] + length(ii)
  }
  out <- num / den
  out[is.nan(out)] <- NA_real_
  out
}

# Same alignment for a known [n x T] matrix of true effects.
event_time_truth <- function(m, z, keep = rep(TRUE, nrow(z))) {
  as.vector(event_time_draws(array(m, c(dim(m), 1)), z, keep))
}

season <- function(v, t) {
  ifelse(v == "Apparel",     0.22 * sin(2 * pi * (t - 3) / 26),
  ifelse(v == "Home",        0.12 * cos(2 * pi * (t - 1) / 26),
                             0.35 * (t / 26)^2 - 0.05))
}

h_grow <- function(s) 0.30 * (1 - exp(-s / 6))
h_fade <- function(s) 0.20 * (s / 1.5) * exp(1 - s / 1.5)

score <- function(est, lo, hi, truth) {
  tibble::tibble(
    RMSE = sqrt(mean((est - truth)^2)),
    `Mean error` = mean(est - truth),
    `Interval width` = mean(hi - lo),
    `Contained truth` = sprintf("%d of %d", sum(truth >= lo & truth <= hi), length(truth))
  )
}

# Difference-in-differences event study helper.
# Evaluates variables from its enclosing or bound environment:
# is_hold, S_observed, LAUNCH, week_study, wave, weeks_all, strata,
# adjust_design, adjust_inv_cross, adjust_leverage.
did_event_study <- function(keep = rep(TRUE, n), D = dY, adjusted = FALSE) {
  hold <- is_hold & keep
  purrr::map_dfr(1:S_observed, function(s) {
    ws <- c("W1", "W2", "W3", "W4")
    ws <- ws[LAUNCH[ws] + s - 1 <= max(week_study)]
    nt <- vapply(ws, function(w) sum(wave == w & keep), integer(1))
    ws <- ws[nt > 1]; nt <- nt[nt > 1]
    a  <- nt / sum(nt)

    col <- match(as.integer(LAUNCH[ws] + s - 1), weeks_all)
    treated_mean <- vapply(seq_along(ws), function(j)
      mean(D[wave == ws[j] & keep, col[j]]), numeric(1))
    var_treated_mean <- vapply(seq_along(ws), function(j)
      var(D[wave == ws[j] & keep, col[j]]) / nt[j], numeric(1))

    control_weights <- vapply(ws, function(w) {
      nt_str <- table(factor(strata[wave == w & keep], levels = levels(strata)))
      nh_str <- table(factor(strata[hold], levels = levels(strata)))
      st <- as.character(strata[hold])
      as.numeric(nt_str[st] / sum(nt_str) / nh_str[st])
    }, numeric(sum(hold)))
    h_combo <- as.vector((D[hold, col, drop = FALSE] * control_weights) %*% a)

    est <- sum(a * treated_mean) - sum(h_combo)
    control_variance <- length(h_combo) * var(h_combo)
    if (adjusted) {
      stopifnot(all(keep))
      influence <- numeric(sum(hold))
      for (j in seq_along(ws)) {
        target_x <- colMeans(adjust_design[wave == ws[j], , drop = FALSE])
        contrast_x <- target_x - as.vector(crossprod(
          adjust_design[hold, , drop = FALSE], control_weights[, j]))
        weights <- control_weights[, j] + as.vector(
          adjust_design[hold, , drop = FALSE] %*% adjust_inv_cross %*% contrast_x)
        influence <- influence + a[j] * weights * D[hold, col[j]] /
          sqrt(1 - adjust_leverage)
      }
      control_variance <- sum(influence^2)
    }
    se <- sqrt(sum(a^2 * var_treated_mean) + control_variance)
    tibble::tibble(s = s, estimate = est, lower = est - 1.96 * se, upper = est + 1.96 * se)
  })
}
