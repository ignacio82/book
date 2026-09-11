# R/longbet-rollout.R
# Deterministic rollout builder with explicit seed/calendar inputs and a named return list.
# One definition of the original DGP and model configuration.

simulate_longbet_rollout <- function(
  seed = 1982,
  n = 3000,
  week_base = 1:6,
  week_study = 7:24,
  week_future = 25:30,
  LAUNCH = c(W1 = 11, W2 = 12, W3 = 13, W4 = 14, Holdout = Inf)
) {
  set.seed(seed)
  weeks_all <- c(week_base, week_study, week_future)
  Tn <- length(weeks_all)

  vertical <- sample(c("Apparel", "Home", "Electronics"), n,
                     replace = TRUE, prob = c(0.40, 0.35, 0.25))
  fulfilled <- rbinom(n, 1, 0.45)
  listings <- pmax(1, round(exp(rnorm(n, log(35), 0.9))))
  seller_sd <- rnorm(n)

  strata <- interaction(
    vertical,
    cut(listings, unique(quantile(listings, 0:3 / 3)), include.lowest = TRUE),
    fulfilled, drop = TRUE
  )
  deal_waves <- function(k) {
    arms <- c("W1", "W2", "W3", "W4", "Holdout")
    props <- c(0.15, 0.15, 0.15, 0.15, 0.40)
    sample(rep(arms, diff(round(k * cumsum(c(0, props))))))
  }
  wave <- character(n)
  for (s in levels(strata)) {
    idx <- which(strata == s)
    wave[idx] <- deal_waves(length(idx))
  }
  launch <- LAUNCH[wave]

  level_i <- 6.4 + 0.55 * seller_sd + 0.45 * log(listings / 35) +
             0.25 * fulfilled + rnorm(n, 0, 0.35)
  drift_i <- 0.004 * seller_sd +
             c(Apparel = 0.000, Home = 0.002, Electronics = 0.008)[vertical]

  tmat <- matrix(rep(weeks_all, each = n), n, Tn)
  mu0_true <- matrix(level_i, n, Tn) +
              outer(drift_i, weeks_all, "*") +
              season(matrix(vertical, n, Tn), tmat)
  y0 <- mu0_true + matrix(rnorm(n * Tn, 0, 0.28), n, Tn)

  S <- tmat - matrix(launch, n, Tn) + 1
  S[!is.finite(S) | S < 0] <- 0
  S <- matrix(as.numeric(S), n, Tn)
  Z <- matrix(as.integer(S > 0), n, Tn)

  w_i <- plogis((log(listings) - log(50)) / 0.35)
  tau_true <- ifelse(S > 0, w_i * h_grow(S) + (1 - w_i) * h_fade(S), 0)
  y <- y0 + tau_true

  broad <- listings >= 50

  wb <- week_base - mean(week_base)
  base_level <- rowMeans(y[, week_base])
  base_slope <- as.vector((y[, week_base] %*% wb) / sum(wb^2))

  vert <- factor(vertical)
  x <- cbind(
    base_level, base_slope, log(listings),
    fulfilled,
    model.matrix(~ vert - 1)[, -1]
  )
  colnames(x) <- c("base_level", "base_slope", "log_listings",
                   "fulfilled", paste0("vertical_", levels(vert)[-1]))

  y_train <- y[, week_study]
  z_train <- Z[, week_study]
  z_ext <- Z[, c(week_study, week_future)]

  list(
    seed = seed,
    n = n,
    week_base = week_base,
    week_study = week_study,
    week_future = week_future,
    weeks_all = weeks_all,
    LAUNCH = LAUNCH,
    Tn = Tn,
    vertical = vertical,
    fulfilled = fulfilled,
    listings = listings,
    seller_sd = seller_sd,
    strata = strata,
    wave = wave,
    launch = launch,
    broad = broad,
    level_i = level_i,
    drift_i = drift_i,
    w_i = w_i,
    tmat = tmat,
    S = S,
    Z = Z,
    mu0_true = mu0_true,
    y0 = y0,
    tau_true = tau_true,
    y = y,
    wb = wb,
    base_level = base_level,
    base_slope = base_slope,
    x = x,
    y_train = y_train,
    z_train = z_train,
    z_ext = z_ext
  )
}

# Bind rollout components and standard derived quantities into an environment
unpack_rollout <- function(rollout, envir = parent.frame()) {
  for (nm in names(rollout)) {
    assign(nm, rollout[[nm]], envir = envir)
  }
  # Derived convenient quantities used across chapters
  assign("truth_att", event_time_truth(rollout$tau_true[, rollout$week_study], rollout$z_train), envir = envir)
  assign("truth_att_ext", event_time_truth(rollout$tau_true[, c(rollout$week_study, rollout$week_future)], rollout$z_ext), envir = envir)
  assign("S_observed", max(rowSums(rollout$z_train)), envir = envir)
  assign("S_max", max(rowSums(rollout$z_ext)), envir = envir)
  assign("is_hold", rollout$wave == "Holdout", envir = envir)
  invisible(rollout)
}
