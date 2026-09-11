# R/longbet-artifacts.R
# Load-or-build functions for the core fit and reusable predictions/summaries.
# Validation, fingerprints, atomic caching in cache/longbet/. No fits merely from sourcing.

longbet_cache_dir <- function() {
  dir_path <- file.path("cache", "longbet")
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  }
  dir_path
}

get_core_model_params <- function() {
  list(
    num_burnin = 2000L,
    num_sweeps = 250L,
    n_skip = 2L,
    num_chains = 4L,
    num_trees_pr = 60L,
    num_trees_trt = 60L,
    sig_knl = 1.0,
    lambda_knl = 2.0,
    sigma_prior_a = 2.0,
    sigma_prior_b = 1.0,
    random_intercept = TRUE,
    random_seed = 42L
  )
}

validate_core_fit <- function(fit) {
  if (!inherits(fit, "longbet")) return(FALSE)
  if (is.null(fit$raw_model) || !is.raw(fit$raw_model) || length(fit$raw_model) == 0L) return(FALSE)
  if (is.null(fit$beta_values) || ncol(fit$beta_values) != 1000L) return(FALSE)
  if (is.null(fit$sigma0_draws) || length(fit$sigma0_draws) != 1000L) return(FALSE)
  if (as.integer(fit$model_params$num_chains) != 4L) return(FALSE)
  if (as.integer(fit$model_params$num_sweeps) != 250L) return(FALSE)
  TRUE
}

# Safely serialize an artifact to cache with atomic rename
write_artifact_atomic <- function(artifact, target_path) {
  tmp_file <- tempfile(
    pattern = paste0("tmp_", basename(target_path), "_"),
    tmpdir = dirname(target_path)
  )
  on.exit({
    if (file.exists(tmp_file)) unlink(tmp_file)
  }, add = TRUE)
  saveRDS(artifact, tmp_file)
  # Validate read-back before replacing
  check <- tryCatch(readRDS(tmp_file), error = function(e) NULL)
  if (is.null(check)) {
    stop(sprintf("Failed to validate serialized artifact at '%s'", tmp_file))
  }
  file.rename(tmp_file, target_path)
  invisible(target_path)
}

get_or_create_core_fit <- function(rollout = NULL, force = FALSE) {
  if (is.null(rollout)) {
    rollout <- simulate_longbet_rollout()
  }
  target_file <- file.path(longbet_cache_dir(), "core_fit.rds")
  engine_fp <- tryCatch(get_longbet_engine_fingerprint(), error = function(e) "unknown")
  model_params <- get_core_model_params()
  data_hash <- digest::digest(list(rollout$y_train, rollout$x, rollout$z_train, rollout$week_study), algo = "sha256")
  fit_key <- digest::digest(list(schema = 1L, engine = engine_fp, params = model_params, data = data_hash), algo = "sha256")

  if (!force && file.exists(target_file)) {
    cached <- tryCatch(readRDS(target_file), error = function(e) NULL)
    if (!is.null(cached) && is.list(cached) && identical(cached$schema_version, 1L)) {
      if (validate_core_fit(cached$payload)) {
        message("[Artifact] Reusing validated core fit from cache/longbet/core_fit.rds")
        return(cached$payload)
      }
    }
    message("[Artifact] Cached core fit invalid or incompatible; rebuilding...")
  }

  # Check if knitr cache in longbet_cache/html has a valid fit we can harvest
  knitr_cache_files <- list.files("longbet_cache/html", pattern = "^fit_.*\\.RData$", full.names = TRUE)
  fit_harvested <- NULL
  for (kf in knitr_cache_files) {
    db_base <- sub("\\.RData$", "", kf)
    if (file.exists(paste0(db_base, ".rdb")) && file.exists(paste0(db_base, ".rdx"))) {
      e <- new.env()
      res <- tryCatch({
        lazyLoad(db_base, envir = e)
        if (exists("lb_fit", envir = e) && validate_core_fit(e$lb_fit)) e$lb_fit else NULL
      }, error = function(e) NULL)
      if (!is.null(res)) {
        fit_harvested <- res
        message(sprintf("[Artifact] Initialized core fit from existing verified cache (%s)", basename(db_base)))
        break
      }
    }
  }

  lb_fit <- if (!is.null(fit_harvested)) {
    fit_harvested
  } else {
    message("[Artifact] Fitting core LongBet model (4 chains x 250 sweeps)...")
    longbet::longbet(
      y = rollout$y_train, x = rollout$x, z = rollout$z_train, t = rollout$week_study,
      num_burnin = model_params$num_burnin, num_sweeps = model_params$num_sweeps,
      n_skip = model_params$n_skip, num_chains = model_params$num_chains,
      num_trees_pr = model_params$num_trees_pr, num_trees_trt = model_params$num_trees_trt,
      sig_knl = model_params$sig_knl, lambda_knl = model_params$lambda_knl,
      sigma_prior_a = model_params$sigma_prior_a, sigma_prior_b = model_params$sigma_prior_b,
      random_intercept = model_params$random_intercept,
      random_seed = model_params$random_seed
    )
  }

  if (!validate_core_fit(lb_fit)) {
    stop("Core fit failed validation: corrupted or incomplete model state.")
  }

  envelope <- list(
    schema_version = 1L,
    artifact_type = "core_fit",
    key = fit_key,
    engine_fingerprint = engine_fp,
    model_params = model_params,
    data_hash = data_hash,
    created_at = Sys.time(),
    payload = lb_fit
  )
  write_artifact_atomic(envelope, target_file)
  message("[Artifact] Saved core fit to cache/longbet/core_fit.rds")
  lb_fit
}

validate_observed_att <- function(att) {
  if (!is.list(att)) return(FALSE)
  if (is.null(att$att) || is.null(att$intervals) || is.null(att$att_full)) return(FALSE)
  if (length(att$att) < 14L || ncol(att$att_full) != 1000L) return(FALSE)
  TRUE
}

get_or_create_observed_att <- function(lb_fit, rollout = NULL, force = FALSE) {
  if (is.null(rollout)) {
    rollout <- simulate_longbet_rollout()
  }
  target_file <- file.path(longbet_cache_dir(), "observed_att.rds")

  if (!force && file.exists(target_file)) {
    cached <- tryCatch(readRDS(target_file), error = function(e) NULL)
    if (!is.null(cached) && is.list(cached) && identical(cached$schema_version, 1L)) {
      if (validate_observed_att(cached$payload)) {
        message("[Artifact] Reusing observed ATT from cache/longbet/observed_att.rds")
        return(cached$payload)
      }
    }
  }

  # Check if knitr cache has predict
  knitr_pred_files <- list.files("longbet_cache/html", pattern = "^predict_.*\\.RData$", full.names = TRUE)
  att_harvested <- NULL
  for (kf in knitr_pred_files) {
    db_base <- sub("\\.RData$", "", kf)
    if (file.exists(paste0(db_base, ".rdb")) && file.exists(paste0(db_base, ".rdx"))) {
      e <- new.env()
      res <- tryCatch({
        lazyLoad(db_base, envir = e)
        if (exists("lb_att", envir = e) && validate_observed_att(e$lb_att)) e$lb_att else NULL
      }, error = function(e) NULL)
      if (!is.null(res)) {
        att_harvested <- res
        message(sprintf("[Artifact] Initialized observed ATT from existing verified cache (%s)", basename(db_base)))
        break
      }
    }
  }

  lb_att <- if (!is.null(att_harvested)) {
    att_harvested
  } else {
    message("[Artifact] Predicting observed-panel ATT...")
    lb_pred <- predict(lb_fit, x = rollout$x, z = rollout$z_train,
                       t = rollout$week_study, summary_only = TRUE,
                       random_seed = 42)
    longbet::get_att(lb_pred, alpha = 0.05)
  }

  if (!validate_observed_att(lb_att)) {
    stop("Observed ATT failed validation.")
  }

  envelope <- list(
    schema_version = 1L,
    artifact_type = "observed_att",
    created_at = Sys.time(),
    payload = lb_att
  )
  write_artifact_atomic(envelope, target_file)
  message("[Artifact] Saved observed ATT to cache/longbet/observed_att.rds")
  lb_att
}

validate_common_launch <- function(cl) {
  if (!is.list(cl)) return(FALSE)
  needed <- c("tau_hat_draws", "tau_hat", "tau_truth_S", "S_target", "initial_lift", "mu0_hat", "sigma2_hat", "gmv_week", "z_all")
  if (!all(needed %in% names(cl))) return(FALSE)
  if (nrow(cl$tau_hat_draws) != 3000L || ncol(cl$tau_hat_draws) != 1000L) return(FALSE)
  if (length(cl$gmv_week) != 3000L) return(FALSE)
  TRUE
}

get_or_create_common_launch <- function(lb_fit, rollout = NULL, force = FALSE) {
  if (is.null(rollout)) {
    rollout <- simulate_longbet_rollout()
  }
  target_file <- file.path(longbet_cache_dir(), "common_launch.rds")

  if (!force && file.exists(target_file)) {
    cached <- tryCatch(readRDS(target_file), error = function(e) NULL)
    if (!is.null(cached) && is.list(cached) && identical(cached$schema_version, 1L)) {
      if (validate_common_launch(cached$payload)) {
        message("[Artifact] Reusing common launch summaries from cache/longbet/common_launch.rds")
        return(cached$payload)
      }
    }
  }

  # Check if knitr cache has counterfactual
  knitr_cf_files <- list.files("longbet_cache/html", pattern = "^counterfactual_.*\\.RData$", full.names = TRUE)
  pred_all_harvested <- NULL
  for (kf in knitr_cf_files) {
    db_base <- sub("\\.RData$", "", kf)
    if (file.exists(paste0(db_base, ".rdb")) && file.exists(paste0(db_base, ".rdx"))) {
      e <- new.env()
      res <- tryCatch({
        lazyLoad(db_base, envir = e)
        if (exists("pred_all", envir = e)) e$pred_all else NULL
      }, error = function(e) NULL)
      if (!is.null(res)) {
        pred_all_harvested <- res
        message(sprintf("[Artifact] Harvesting common launch prediction from existing verified cache (%s)", basename(db_base)))
        break
      }
    }
  }

  n <- rollout$n
  week_study <- rollout$week_study
  z_all <- cbind(matrix(0L, n, sum(week_study < 11)),
                 matrix(1L, n, sum(week_study >= 11)))
  S_target <- ncol(z_all) - sum(week_study < 11)

  pred_all <- if (!is.null(pred_all_harvested)) {
    pred_all_harvested
  } else {
    message("[Artifact] Running common-launch prediction under week 11 adoption...")
    predict(lb_fit, x = rollout$x, z = z_all, t = week_study, random_seed = 1)
  }

  tau_hat_draws <- pred_all$tauhats[, ncol(z_all), ]
  tau_hat <- rowMeans(tau_hat_draws)
  tau_truth_S <- rollout$w_i * h_grow(S_target) + (1 - rollout$w_i) * h_fade(S_target)

  recent <- tail(seq_along(week_study), 4)
  sigma2_hat <- median(as.numeric(lb_fit$sigma0_draws)^2) * lb_fit$sdy^2
  mu0_hat <- apply(pred_all$muhats0[, recent, , drop = FALSE], c(1, 2), median)
  gmv_week <- rowMeans(exp(mu0_hat + sigma2_hat / 2))

  exposure_columns <- which(week_study >= 11)
  tau_path_draws <- pred_all$tauhats[, exposure_columns, , drop = FALSE]
  initial_lift <- apply(expm1(tau_path_draws), c(1, 3), sum)
  rm(pred_all, tau_path_draws)
  invisible(gc(verbose = FALSE))

  payload <- list(
    S_target = S_target,
    tau_hat_draws = tau_hat_draws,
    tau_hat = tau_hat,
    tau_truth_S = tau_truth_S,
    recent = recent,
    sigma2_hat = sigma2_hat,
    mu0_hat = mu0_hat,
    gmv_week = gmv_week,
    initial_lift = initial_lift,
    z_all = z_all
  )

  if (!validate_common_launch(payload)) {
    stop("Common launch payload failed validation.")
  }

  envelope <- list(
    schema_version = 1L,
    artifact_type = "common_launch",
    created_at = Sys.time(),
    payload = payload
  )
  write_artifact_atomic(envelope, target_file)
  message("[Artifact] Saved common launch summaries to cache/longbet/common_launch.rds")
  payload
}
