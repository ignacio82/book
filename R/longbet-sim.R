# R/longbet-sim.R
# Every simulated panel used by the LongBet chapters. Each generator returns the data the
# model sees plus the quantities the chapter scores against, so no chapter recomputes a truth.

# ---------------------------------------------------------------------------------------
# The marketplace rollout: one shape over exposure whose size varies smoothly with catalog
# breadth. Sellers differ in level; the untreated path is flat over the calendar.
# ---------------------------------------------------------------------------------------
h_shape   <- function(s) 1 - exp(-s / 4)
amplitude <- function(listings) 0.30 * stats::plogis((log(listings) - log(40)) / 0.8)

simulate_rollout <- function(seed = 7, n = 400, week_base = 1:4, week_study = 5:20,
                             week_future = 21:24,
                             LAUNCH = c(W1 = 9, W2 = 10, W3 = 11, Holdout = Inf),
                             sd_eps = 0.25) {
  set.seed(seed)
  weeks_all <- c(week_base, week_study, week_future); Tn <- length(weeks_all)
  vertical  <- sample(c("Apparel", "Home", "Electronics"), n, TRUE, c(.40, .35, .25))
  fulfilled <- stats::rbinom(n, 1, 0.45)
  listings  <- pmax(1, round(exp(stats::rnorm(n, log(35), 0.9))))
  seller_sd <- stats::rnorm(n)
  strata <- interaction(vertical,
                        cut(listings, unique(stats::quantile(listings, 0:2 / 2)), include.lowest = TRUE),
                        fulfilled, drop = TRUE)
  deal <- function(k) sample(rep(c("W1", "W2", "W3", "Holdout"),
                                 diff(round(k * cumsum(c(0, .2, .2, .2, .4))))))
  wave <- character(n)
  for (st in levels(strata)) { ii <- which(strata == st); wave[ii] <- deal(length(ii)) }
  launch <- LAUNCH[wave]

  level_i <- 6.4 + 0.55 * seller_sd + 0.45 * log(listings / 35) + 0.25 * fulfilled +
    stats::rnorm(n, 0, 0.35)
  tmat <- matrix(rep(weeks_all, each = n), n, Tn)
  mu0_true <- matrix(level_i, n, Tn)                      # flat over the calendar by construction
  y0 <- mu0_true + matrix(stats::rnorm(n * Tn, 0, sd_eps), n, Tn)

  S <- tmat - matrix(launch, n, Tn) + 1
  S[!is.finite(S) | S < 0] <- 0
  S <- matrix(as.numeric(S), n, Tn)
  Z <- matrix(as.integer(S > 0), n, Tn)
  a_i <- amplitude(listings)
  tau_true <- ifelse(S > 0, a_i * h_shape(S), 0)
  y <- y0 + tau_true

  vert <- factor(vertical)
  x <- cbind(log_listings = log(listings), fulfilled = fulfilled,
             stats::model.matrix(~ vert - 1)[, -1])
  colnames(x)[3:4] <- paste0("vertical_", levels(vert)[-1])
  base_level <- rowMeans(y[, match(week_base, weeks_all), drop = FALSE])

  list(n = n, week_base = week_base, week_study = week_study, week_future = week_future,
       weeks_all = weeks_all, LAUNCH = LAUNCH, Tn = Tn, vertical = vertical,
       fulfilled = fulfilled, listings = listings, seller_sd = seller_sd, strata = strata,
       wave = wave, launch = launch, level_i = level_i, a_i = a_i, tmat = tmat, S = S, Z = Z,
       mu0_true = mu0_true, y0 = y0, tau_true = tau_true, y = y, x = x,
       base_level = base_level, sd_eps = sd_eps,
       y_train = y[, match(week_study, weeks_all), drop = FALSE],
       z_train = Z[, match(week_study, weeks_all), drop = FALSE],
       z_ext = Z[, match(c(week_study, week_future), weeks_all), drop = FALSE],
       broad = listings >= 50, is_hold = wave == "Holdout")
}

# Adoption chosen by the marketplace instead of by lottery. "observed" ranks on catalog size
# and the seller's own level, both of which the model sees; "hidden" ranks on a trait the
# covariates never record, which also lifts the untreated path from the first launch on.
simulate_observational <- function(ro, scenario = c("observed", "hidden"), seed = 8123) {
  scenario <- match.arg(scenario)
  set.seed(seed)
  n <- ro$n; Tn <- ro$Tn
  u_i <- stats::rnorm(n)
  y0s <- matrix(ro$level_i, n, Tn) + matrix(stats::rnorm(n * Tn, 0, ro$sd_eps), n, Tn)
  if (scenario == "hidden") {
    y0s <- y0s + 0.15 * u_i * matrix(rep(ro$weeks_all >= min(ro$LAUNCH), each = n), n, Tn)
  }
  score <- if (scenario == "observed")
    1.0 * scale(log(ro$listings))[, 1] + 0.8 * scale(ro$level_i)[, 1] + stats::rnorm(n, 0, .8)
  else 1.5 * u_i + stats::rnorm(n, 0, .5)
  q <- stats::quantile(score, c(.40, .60, .80))
  wv <- ifelse(score >= q[3], "W1", ifelse(score >= q[2], "W2", ifelse(score >= q[1], "W3", "Holdout")))
  Sm <- ro$tmat - matrix(ro$LAUNCH[wv], n, Tn) + 1
  Sm[!is.finite(Sm) | Sm < 0] <- 0
  Sm <- matrix(as.numeric(Sm), n, Tn)
  Zm <- matrix(as.integer(Sm > 0), n, Tn)
  tau <- ifelse(Sm > 0, ro$a_i * h_shape(Sm), 0)
  list(y = y0s + tau, z = Zm, S = Sm, tau = tau, wave = wv, launch = ro$LAUNCH[wv], x = ro$x)
}

# A promotional calendar the marketplace sets: it lifts GMV on its own and makes the optimizer
# worth more in the same week, so it belongs in both forests.
simulate_promo <- function(ro, seed = 515) {
  set.seed(seed)
  n <- ro$n; Tw <- length(ro$week_study)
  promo <- stats::plogis(matrix(rep(stats::rnorm(Tw, 0, 0.9), each = n), n, Tw) +
                           matrix(stats::rnorm(n * Tw, 0, 0.7), n, Tw))
  tau_study <- ro$tau_true[, match(ro$week_study, ro$weeks_all), drop = FALSE]
  tau_promo <- tau_study * (1 + 1.2 * promo)
  y_promo <- ro$y0[, match(ro$week_study, ro$weeks_all), drop = FALSE] + 0.8 * promo + tau_promo
  list(promo = promo, tau_promo = tau_promo, y = y_promo,
       treated = ro$z_train == 1, bins = dplyr::ntile(promo[ro$z_train == 1], 10))
}

# ---------------------------------------------------------------------------------------
# Three outcomes on one rollout: GMV, seller hours, and a complaint indicator, with signed
# effects and correlated innovations within a seller-week.
# ---------------------------------------------------------------------------------------
simulate_multi <- function(seed = 11, n = 300, tt = 12, launch = c(W1 = 5, W2 = 7, Holdout = Inf)) {
  set.seed(seed)
  listings <- pmax(1, round(exp(stats::rnorm(n, log(35), 0.9))))
  fulfilled <- stats::rbinom(n, 1, 0.45); x3 <- stats::rnorm(n)
  wave <- sample(rep(names(launch), diff(round(n * cumsum(c(0, .3, .3, .4))))))
  tm <- matrix(rep(seq_len(tt), each = n), n, tt)
  S <- tm - matrix(launch[wave], n, tt) + 1
  S[!is.finite(S) | S < 0] <- 0
  S <- matrix(as.numeric(S), n, tt)
  Z <- matrix(as.integer(S > 0), n, tt)
  w <- stats::plogis((log(listings) - log(40)) / 0.8)
  amp <- list(gmv = -0.10 + 0.30 * w + 0.08 * fulfilled,
              seller_hours = 0.10 - 0.30 * w - 0.08 * fulfilled,
              complaint = 0.25 - 0.55 * w - 0.15 * fulfilled)
  h <- function(s) 1 - exp(-s / 3)
  u <- matrix(stats::rnorm(n * tt), n, tt); v <- matrix(stats::rnorm(n * tt), n, tt)
  e3 <- matrix(stats::rnorm(n * tt), n, tt)
  e_gmv <- u; e_hours <- 0.60 * u + 0.80 * v
  e_compl <- 0.45 * u + 0.10 * v + sqrt(0.7875) * e3
  gmv_mu0   <- 6.4 + 0.45 * log(listings / 35) + 0.25 * fulfilled + 0.2 * x3 + stats::rnorm(n, 0, 0.35)
  hours_mu0 <- log(3) + 0.40 * log(listings / 35) - 0.20 * fulfilled + stats::rnorm(n, 0, 0.25)
  compl_mu0 <- -1.3 + 0.25 * log(listings / 35) - 0.35 * fulfilled
  y <- list(
    gmv = matrix(gmv_mu0, n, tt) + matrix(amp$gmv, n, tt) * h(S) + 0.25 * e_gmv,
    seller_hours = matrix(hours_mu0, n, tt) + matrix(amp$seller_hours, n, tt) * h(S) + 0.30 * e_hours,
    complaint = 1L * (matrix(compl_mu0, n, tt) + matrix(amp$complaint, n, tt) * h(S) + e_compl > 0))
  list(n = n, tt = tt, y = y,
       x = cbind(log_listings = log(listings), fulfilled = fulfilled, x3 = x3),
       z = Z, S = S, wave = wave, listings = listings, fulfilled = fulfilled,
       amp = amp, h = h, compl_mu0 = compl_mu0,
       errors = list(gmv = e_gmv, seller_hours = e_hours, complaint = e_compl))
}

# ---------------------------------------------------------------------------------------
# Five-star ratings: a latent response observed through ordered thresholds, one shape over
# exposure whose size varies with a covariate.
# ---------------------------------------------------------------------------------------
simulate_ordinal <- function(seed = 91, n = 240, tt = 8, cuts = c(-Inf, 0, 0.7, 1.4, 2.0, Inf)) {
  set.seed(seed)
  x <- matrix(stats::runif(n * 4, -1, 1), n, 4, dimnames = list(NULL, paste0("x", 0:3)))
  adopt <- sample(rep(c(3, 4, 5, Inf), length.out = n))
  tm <- matrix(rep(seq_len(tt), each = n), n, tt)
  S <- tm - matrix(adopt, n, tt) + 1
  S[!is.finite(S) | S < 0] <- 0
  S <- matrix(as.numeric(S), n, tt)
  Z <- matrix(as.integer(S > 0), n, tt)
  eta0 <- -0.6 + 0.55 * x[, 1] - 0.4 * x[, 2] + 0.08 * (tm - 4)
  tau <- (0.35 + 0.1 * x[, 3]) * (0.5 + (1 - exp(-S / 2))) * (S > 0)
  y <- matrix(as.integer(cut(eta0 + tau + matrix(stats::rnorm(n * tt), n, tt), cuts, labels = FALSE)) - 1L, n, tt)
  list(n = n, tt = tt, x = x, z = Z, S = S, y = y, eta0 = eta0, tau = tau, cuts = cuts, adopt = adopt)
}

# ---------------------------------------------------------------------------------------
# Two panels for the comparison with the design-based estimator: an additive one where the
# effect is the same for every unit, and one where its size varies smoothly with a covariate.
# ---------------------------------------------------------------------------------------
simulate_additive <- function(seed = 101, n = 300, tt = 8, p = 4) {
  set.seed(seed)
  x <- matrix(stats::rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
  alpha <- stats::rnorm(n, 0, 1.2)
  adopt <- sample(c(3, 5, 7, Inf), n, TRUE, rep(.25, 4)); tg <- seq_len(tt)
  z <- outer(adopt, tg, function(a, t) as.numeric(t >= a))
  s <- outer(adopt, tg, function(a, t) pmax(0, t - a + 1))
  tau <- ifelse(z == 1, 1.4 * s, 0)
  y <- outer(alpha, rep(1, tt)) + matrix(stats::rnorm(n * tt, 0, 0.6), n, tt) + tau
  list(n = n, tt = tt, tg = tg, x = x, z = z, s = s, tau = tau, y = y, adopt = adopt)
}

simulate_heterogeneous <- function(seed = 303, n = 400, tt = 8, p = 4) {
  set.seed(seed)
  x <- matrix(stats::rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
  alpha <- stats::rnorm(n, 0, 1.0)
  adopt <- sample(c(3, 5, 7, Inf), n, TRUE, rep(.25, 4)); tg <- seq_len(tt)
  z <- outer(adopt, tg, function(a, t) as.numeric(t >= a))
  s <- outer(adopt, tg, function(a, t) pmax(0, t - a + 1))
  amp <- 1.5 + 0.9 * x[, 1]
  tau <- ifelse(z == 1, amp * log(1 + s), 0)
  y <- outer(alpha + 0.5 * x[, 1], rep(0, tt), "+") + matrix(stats::rnorm(n * tt, 0, 0.5), n, tt) + tau
  list(n = n, tt = tt, tg = tg, x = x, z = z, s = s, tau = tau, y = y, adopt = adopt, amp = amp)
}

# ---------------------------------------------------------------------------------------
# Randomized encouragement with absorbing adoption. Adoption is a first-passage time with
# known weekly hazards, so the expected effect of the offer is available in closed form.
# ---------------------------------------------------------------------------------------
expected_arm_stats <- function(h, g) {
  tt <- length(h)
  surv <- cumprod(1 - h)
  p_adopt <- h * c(1, surv[-tt])
  list(cdf = cumsum(p_adopt),
       eg = vapply(seq_len(tt), function(t) sum(p_adopt[1:t] * g(t - (1:t) + 1)), numeric(1)))
}

expected_truth <- function(hz0, hz1, tau_base, g) {
  n <- nrow(hz0[[1]]); tt <- ncol(hz0[[1]]); k <- length(hz0)
  citt_y <- citt_d <- p_at <- p_nt <- matrix(0, n, tt)
  for (level in seq_len(k)) {
    for (i in seq_len(n)) {
      s0 <- expected_arm_stats(hz0[[level]][i, ], g)
      s1 <- expected_arm_stats(hz1[[level]][i, ], g)
      citt_d[i, ] <- citt_d[i, ] + (s1$cdf - s0$cdf) / k
      citt_y[i, ] <- citt_y[i, ] + tau_base[i] * (s1$eg - s0$eg) / k
      p_at[i, ]   <- p_at[i, ] + s0$cdf / k
      p_nt[i, ]   <- p_nt[i, ] + (1 - s1$cdf) / k
    }
  }
  list(citt_y = citt_y, citt_d = citt_d, p_at = p_at, p_nt = p_nt)
}

first_adoption <- function(hazard, v) {
  hit <- v < hazard
  out <- rep(Inf, nrow(hit))
  any_hit <- rowSums(hit) > 0
  out[any_hit] <- apply(hit[any_hit, , drop = FALSE], 1, function(r) which(r)[1])
  out
}

exposure_from <- function(adopt_week, tt) {
  s <- outer(adopt_week, seq_len(tt), function(a, t) pmax(0, t - a + 1))
  s[!is.finite(s)] <- 0
  s
}

# Concierge onboarding: adoption rises with technical readiness once offered, and the
# productivity benefit compounds with exposure for accounts that are both ready and scaled.
onboarding_structure <- function(x, tt = 8, t_enc = 4, tau_high = 9.0, tau_low = 1.2) {
  n <- nrow(x); x1 <- x[, "x1"]; x2 <- x[, "x2"]
  p0_level <- c(0.05, 0.20)
  p_comp <- 0.20 + 0.65 * stats::plogis(3 * x1)
  hz0 <- lapply(p0_level, function(p) matrix(p, n, tt))
  hz1 <- lapply(hz0, function(h) { h[, t_enc:tt] <- p_comp; h })
  qualifying <- (x1 > 0) & (x2 > -0.2)
  tau_base <- tau_low + (tau_high - tau_low) * qualifying * (0.5 + 0.5 * stats::pnorm(x2))
  list(hz0 = hz0, hz1 = hz1, qualifying = qualifying, tau_base = tau_base,
       truth = expected_truth(hz0, hz1, tau_base, log1p))
}

generate_encouragement_data <- function(n, tt = 8, t_enc = 4, seed = 42, sd_eps = 1.5) {
  set.seed(seed)
  x <- cbind(x1 = stats::rnorm(n), x2 = stats::rnorm(n), x3 = stats::rnorm(n))
  u <- stats::rbinom(n, 1, 0.5)
  z_u <- stats::rbinom(n, 1, 0.5)
  z <- matrix(0, n, tt); z[z_u == 1, t_enc:tt] <- 1
  st <- onboarding_structure(x, tt, t_enc)
  v  <- matrix(stats::runif(n * tt), n, tt)
  h0 <- st$hz0[[1]] * (u == 0) + st$hz0[[2]] * (u == 1)
  h1 <- st$hz1[[1]] * (u == 0) + st$hz1[[2]] * (u == 1)
  a0 <- first_adoption(h0, v); a1 <- first_adoption(h1, v)
  s0 <- exposure_from(a0, tt); s1 <- exposure_from(a1, tt)
  d0 <- (s0 > 0) * 1; d1 <- (s1 > 0) * 1
  d  <- ifelse(z == 1, d1, d0); s <- ifelse(z == 1, s1, s0)
  alpha_i <- stats::rnorm(n, 10, 1.5)
  eps <- matrix(stats::rnorm(n * tt, 0, sd_eps), n, tt)
  mu  <- alpha_i + 0.5 * col(eps) - 2.5 * u + 0.8 * x[, "x1"] - 0.5 * x[, "x2"]
  y0 <- mu + st$tau_base * log1p(s0) + eps
  y1 <- mu + st$tau_base * log1p(s1) + eps
  y  <- ifelse(z == 1, y1, y0)
  list(n = n, tt = tt, t_enc = t_enc, y = y, d = d, z = z, x = x, z_u = z_u, u = u, s = s,
       d0 = d0, d1 = d1, y0 = y0, y1 = y1, qualifying = st$qualifying, tau_base = st$tau_base,
       true_citt_y = st$truth$citt_y, true_citt_d = st$truth$citt_d,
       true_p_at = st$truth$p_at, true_p_nt = st$truth$p_nt)
}

# A nudge whose benefit arrives late and whose sign depends on workflow fit: compliance is
# unrelated to benefit, so ranking on adoption would recruit exactly the wrong users.
nudge_structure <- function(x, tt = 8, t_nudge = 3, slope_pos = 3.5, slope_neg = -2.5) {
  n <- nrow(x); x1 <- x[, "x1"]; x2 <- x[, "x2"]
  p0_of <- function(u) ifelse(x1 > 0.8, 0.18, ifelse(u == 1 & x1 > 0.2, 0.12, 0.02))
  p1 <- ifelse(x1 < -0.8 & x2 < -0.6, 0.05, ifelse(x2 > -0.4, 0.85, 0.35))
  hz0 <- lapply(0:1, function(u) matrix(p0_of(u), n, tt))
  hz1 <- lapply(hz0, function(h) { h[, t_nudge:tt] <- p1; h })
  beneficiary <- (x1 > 0) & (x2 > -0.2)
  slope <- ifelse(beneficiary, slope_pos, slope_neg)
  g <- function(s) pmax(0, s - 2)
  list(hz0 = hz0, hz1 = hz1, beneficiary = beneficiary, slope = slope, g = g,
       truth = expected_truth(hz0, hz1, slope, g))
}

generate_nudge_data <- function(n, tt = 8, t_nudge = 3, seed = 42, sd_eps = 1.0) {
  set.seed(seed)
  x <- cbind(x1 = stats::rnorm(n), x2 = stats::rnorm(n), x3 = stats::rnorm(n))
  u <- stats::rbinom(n, 1, 0.5)
  z_u <- stats::rbinom(n, 1, 0.5)
  z <- matrix(0, n, tt); z[z_u == 1, t_nudge:tt] <- 1
  st <- nudge_structure(x, tt, t_nudge)
  stopifnot(all(st$hz1[[1]] >= st$hz0[[1]]), all(st$hz1[[2]] >= st$hz0[[2]]))
  v  <- matrix(stats::runif(n * tt), n, tt)
  h0 <- st$hz0[[1]] * (u == 0) + st$hz0[[2]] * (u == 1)
  h1 <- st$hz1[[1]] * (u == 0) + st$hz1[[2]] * (u == 1)
  a0 <- first_adoption(h0, v); a1 <- first_adoption(h1, v)
  s0 <- exposure_from(a0, tt); s1 <- exposure_from(a1, tt)
  d0 <- (s0 > 0) * 1; d1 <- (s1 > 0) * 1
  d  <- ifelse(z == 1, d1, d0)
  alpha_i <- stats::rnorm(n, 10, 1.5)
  eps <- matrix(stats::rnorm(n * tt, 0, sd_eps), n, tt)
  mu  <- alpha_i + 0.4 * col(eps) - 2.0 * u + 0.6 * x[, "x1"] - 0.3 * x[, "x2"]
  y0 <- mu + st$slope * st$g(s0) + eps
  y1 <- mu + st$slope * st$g(s1) + eps
  y  <- ifelse(z == 1, y1, y0)
  list(n = n, tt = tt, t_nudge = t_nudge, y = y, d = d, z = z, x = x, z_u = z_u,
       d0 = d0, d1 = d1, beneficiary = st$beneficiary, slope = st$slope,
       complier_T = d1[, tt] == 1 & d0[, tt] == 0, always_T = d0[, tt] == 1,
       never_T = d1[, tt] == 0,
       true_citt_y = st$truth$citt_y, true_citt_d = st$truth$citt_d)
}
