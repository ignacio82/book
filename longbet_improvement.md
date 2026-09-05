# Proposal: Hierarchical Semi-Parametric LongBet with Latent Unit Intercepts and Autoregressive Innovations

**Target Repository:** `https://github.com/google/longbet`  
**Authors/Contributors:** Ignacio Martinez, Meijia Wang, P. Richard Hahn  
**Subject:** Resolving the i.i.d. error assumption across units and time to maximize estimation accuracy and restore nominal posterior coverage.

---

## 1. Executive Summary

LongBet models panel data using Bayesian ensemble trees:
$$Y_{it} = \alpha_t\mu(X_i, t) + \beta_{S_{it}}\nu(X_i, S_{it}, t) + \epsilon_{it}$$
where $\epsilon_{it}$ is assumed to be independently and identically distributed:
$$\epsilon_{it} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

In panel data, this assumption is virtually always violated across two dimensions:
1. **Across Time (Within Unit):** Outcomes exhibit permanent unit baselines $\gamma_i$ and transitory serial persistence ($u_{it} = \rho u_{i, t-1} + \dots$).
2. **Across Units:** Units within the same cluster, category, or market share unobserved shocks or interact competitively (violating SUTVA).

Treating $N \times T$ panel observations as independent Gaussian draws degrades **point estimation accuracy** (inducing tree overfitting, finite-sample baseline bias in treatment effects, and distorted trajectory shapes in $\beta_S$) and causes **posterior uncertainty collapse** (producing artificially narrow credible intervals with sub-nominal coverage).

This proposal details the mathematical formulation, MCMC Gibbs sampling steps, and C++ software architecture for **Hierarchical Semi-Parametric LongBet**. By incorporating **latent unit random intercepts ($\gamma_i$)** and **autoregressive innovation whitening ($\text{AR}(1)$)** directly into the MCMC engine, LongBet eliminates baseline leakage and trajectory distortion while preserving XBART’s $O(NT)$ computational speed.

---

## 2. Problem Analysis: How Dependent Errors Degrade Accuracy

In real longitudinal data, the true composite error decomposes into:
$$\eta_{it} = \gamma_i + u_{it} + \epsilon_{it}$$
where $\gamma_i$ is a permanent unit-specific baseline, $u_{it} = \rho u_{i, t-1}$ is an autoregressive transitory process, and $\epsilon_{it}$ is white noise.

Assuming $\eta_{it} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$ damages the model in four distinct ways:

### 2.1 Squandered Tree Capacity in the Prognostic Forest $\mu(X_i, t)$
Because LongBet has no unit fixed/random effect parameters, the prognostic forest $\mu(X_i, t)$ is forced to approximate each unit's idiosyncratic baseline $\gamma_i$ using only the observed time-invariant covariates $X_i$. 
- A finite set of continuous covariates cannot uniquely identify thousands of units without micro-splitting.
- Trees spend depth and split capacity memorizing individual seller levels rather than learning the smooth, generalizable population relationship $f(X, t)$.
- This increases the variance and RMSE of the prognostic surface.

### 2.2 Baseline Leakage into the Treatment Forest $\nu(X_i, S, t)$
The residual variation from $\gamma_i$ that $\mu(X_i, t)$ fails to explain remains in the residual $Y_{it} - \mu_{it}$. When units enter treatment, the treatment forest splits on these unmodeled residual differences:
- In finite samples, waves that happen to have slightly higher unobserved baselines $\gamma_i$ will have their baseline advantage misattributed to the treatment effect $\tau$.
- This introduces finite-sample bias into CATE estimation, even under randomized assignment.

### 2.3 Trajectory Distortion in the Gaussian Process $\beta_S$
The shared factor $\beta_S$ captures the population-level treatment trajectory over event time $S$. It updates by pooling residuals across units at each event time.
- If errors have positive serial correlation ($\rho > 0$), an idiosyncratic shock experienced by a unit at $S=1$ lingers into $S=2, 3$.
- The Gaussian Process cannot distinguish between a true causal dynamic and the lingering autocorrelation of pre-existing shocks, warping the estimated trajectory of $\beta_S$ and corrupting forward projections.

### 2.4 Artificial Precision and Coverage Failure
The likelihood evaluates the sample size as $N \times T$. When errors have within-unit correlation $\rho$, the effective sample size is:
$$N_{\text{eff}} \approx \frac{NT}{1 + 2\sum_{k=1}^{T-1}(1 - k/T)\rho^k} \ll NT$$
Treating the effective sample size as $NT$ deflates the posterior variance by a factor of roughly $(1+\rho)/(1-\rho)$. For $\rho \approx 0.35$, credible intervals are approximately half as wide as they should be, leading to severe undercoverage.

---

## 3. The Proposed Solution: Hierarchical Semi-Parametric LongBet

We extend the LongBet observation model to:
$$Y_{it} = \alpha_t\mu(X_i, t) + \beta_{S_{it}}\nu(X_i, S_{it}, t) + \gamma_i + u_{it}$$
with the hierarchical error specification:
$$\gamma_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\gamma^2)$$
$$u_{it} = \rho u_{i, t-1} + \epsilon_{it}, \quad \epsilon_{it} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\epsilon^2)$$

### Priors and Hyperparameters
- **Unit Random Intercepts:** $\gamma_i \sim \mathcal{N}(0, \sigma_\gamma^2)$
- **Intercept Variance:** $\sigma_\gamma^2 \sim \mathcal{IG}(a_\gamma, b_\gamma)$ (or Half-Cauchy)
- **Serial Persistence:** $\rho \sim \mathcal{U}(-1, 1)$ (or Truncated Normal $\mathcal{TN}_{(-1, 1)}(0, 0.5^2)$)
- **Innovation Variance:** $\sigma_\epsilon^2 \sim \mathcal{IG}(a_\epsilon, b_\epsilon)$

---

## 4. Full MCMC Sampling Algorithm (Sweep-by-Sweep)

The primary advantage of this formulation is that it can be sampled via **conjugate Gibbs updates** with $O(NT)$ complexity, preserving the speed of the underlying C++ XBART engine.

At each sweep $s = 1, \dots, S_{\text{sweeps}}$:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Sample Unit Intercepts: γ_i | Y, μ, ν, β, ρ, σ_ε, σ_γ    │ (O(N))
├─────────────────────────────────────────────────────────────┤
│ 2. Sample Autoregressive Parameter: ρ | e, σ_ε              │ (O(NT))
├─────────────────────────────────────────────────────────────┤
│ 3. Quasi-Difference Targets (Prais-Winsten Transformation)  │ (O(NT))
├─────────────────────────────────────────────────────────────┤
│ 4. Update Prognostic Forest μ(X, t) on Whitened Residuals   │ (XBART)
├─────────────────────────────────────────────────────────────┤
│ 5. Update Treatment Forest ν(X, S, t)                       │ (XBART)
├─────────────────────────────────────────────────────────────┤
│ 6. Update GP Factor β_S via Whitened Likelihood             │ (GP)
├─────────────────────────────────────────────────────────────┤
│ 7. Update Variances σ_γ^2 and σ_ε^2                         │ (O(N) + O(NT))
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Closed-Form Conjugate Update of $\gamma_i$
Condition on current tree predictions $\hat{Y}_{it} = \alpha_t\mu(X_i, t) + \beta_{S_{it}}\nu(X_i, S_{it}, t)$ and current $\rho$.
Define the quasi-differenced residual for unit $i$:
$$\tilde{R}_{it} = \begin{cases} \sqrt{1 - \rho^2}(Y_{i1} - \hat{Y}_{i1}) & t = 1 \\ (Y_{it} - \hat{Y}_{it}) - \rho (Y_{i, t-1} - \hat{Y}_{i, t-1}) & t \ge 2 \end{cases}$$
Define the quasi-differenced unit intercept weight:
$$w_t = \begin{cases} \sqrt{1 - \rho^2} & t = 1 \\ 1 - \rho & t \ge 2 \end{cases}$$
Then the likelihood for $\gamma_i$ is Gaussian with sufficient statistics:
$$S_{\gamma, i} = \sum_{t=1}^T w_t \tilde{R}_{it}, \quad W_\gamma = \sum_{t=1}^T w_t^2 = (1 - \rho^2) + (T - 1)(1 - \rho)^2$$
The full conditional posterior is exact and independent across units:
$$\gamma_i \mid \text{rest} \sim \mathcal{N}\left( \frac{S_{\gamma, i}}{\sigma_\epsilon^2 / \sigma_\gamma^2 + W_\gamma}, \; \frac{\sigma_\epsilon^2}{\sigma_\epsilon^2 / \sigma_\gamma^2 + W_\gamma} \right)$$
- Drawing $N$ independent normals takes $< 1$ millisecond in C++.
- When $T$ is moderate, $\gamma_i$ absorbs individual seller scale; when $T$ is small, it shrinks smoothly toward 0.

### Step 2: Sampling Serial Correlation $\rho$
Define the error after removing unit intercepts:
$$e_{it} = Y_{it} - \hat{Y}_{it} - \gamma_i$$
Under the AR(1) model $e_{it} = \rho e_{i, t-1} + \epsilon_{it}$, the conditional posterior for $\rho$ given $e_{it}$ across all units is:
$$\hat{\rho} = \frac{\sum_{i=1}^N \sum_{t=2}^T e_{it} e_{i, t-1}}{\sum_{i=1}^N \sum_{t=2}^T e_{i, t-1}^2}, \quad V_\rho = \frac{\sigma_\epsilon^2}{\sum_{i=1}^N \sum_{t=2}^T e_{i, t-1}^2}$$
Sample $\rho$ from a truncated Gaussian $\mathcal{TN}_{(-1, 1)}(\hat{\rho}, V_\rho)$ (or via a Metropolis-Hastings step including the stationary determinant term $\frac{1}{2}\log(1-\rho^2)$).

### Step 3: Prais-Winsten Quasi-Differencing for Tree Updates
Before growing the prognostic and treatment trees, transform the outcome matrix $Y$ and tree targets using the sampled $\rho$ and $\gamma_i$:
$$Y^*_{it} = \begin{cases} \sqrt{1 - \rho^2}(Y_{i1} - \gamma_i) & t = 1 \\ (Y_{it} - \gamma_i) - \rho(Y_{i, t-1} - \gamma_i) & t \ge 2 \end{cases}$$
Similarly transform the tree inputs across adjacent time periods:
$$\mu^*(X_i, t) = \begin{cases} \sqrt{1 - \rho^2}\mu(X_i, 1) & t = 1 \\ \mu(X_i, t) - \rho\mu(X_i, t-1) & t \ge 2 \end{cases}$$
The residual evaluated in XBART leaf updates and split sufficiency statistics is now:
$$\epsilon^*_{it} = Y^*_{it} - \alpha_t\mu^*(X_i, t) - \beta_{S_{it}}\nu^*(X_i, S_{it}, t)$$
Because $\epsilon^*_{it}$ is conditionally **independent Gaussian white noise**, XBART’s conjugate normal leaf updates and split scoring criteria are mathematically exact.

### Step 4: Purged Gaussian Process Update of $\beta_S$
The shared factor $\beta_S$ is updated conditioning on residuals that have been stripped of both unit baselines $\gamma_i$ and autoregressive noise $u_{it}$. The GP prior over $S$ now models true causal duration dynamics rather than autocorrelated noise.

### Step 5: Variance Updates
- Update $\sigma_\gamma^2 \sim \mathcal{IG}\left(a_\gamma + \frac{N}{2}, \; b_\gamma + \frac{1}{2}\sum_{i=1}^N \gamma_i^2\right)$.
- Update $\sigma_\epsilon^2 \sim \mathcal{IG}\left(a_\epsilon + \frac{NT}{2}, \; b_\epsilon + \frac{1}{2}\sum_{i, t} (\epsilon^*_{it})^2\right)$.

---

## 5. Why This Beats Alternative Fixes

### 5.1 Why Not Fixed Effects Demeaning Upfront ($Y_{it} - \bar{Y}_i$)?
In panel econometrics, the standard fix for unit heterogeneity is demeaning. In a staggered rollout, upfront demeaning is disastrous:
- **Nickell Bias / Staggered Rollout Contamination:** For treated units, $\bar{Y}_i$ contains post-treatment outcomes. Demeaning subtracts the treatment effect from the pre-treatment baseline, artificially depressing pre-treatment levels and distorting the dynamic treatment trajectory (the classic Goodman-Bacon / Sun-Abraham critique).
- **In contrast:** The Bayesian Gibbs sampler estimates $\gamma_i$ conditional on $\beta_{S_{it}}\nu_{it}$. It subtracts the estimated treatment effect *before* computing the unit baseline, eliminating contamination across event times.

### 5.2 Why Not Post-Processing (Cluster Bootstrap / Sandwich Adjustments)?
Post-processing adjustments (such as the cluster bootstrap or Bayesian sandwich correction) only alter standard errors:
- They leave point estimates untouched.
- The trees still overfit to unit baselines, $\beta_S$ still absorbs autocorrelated noise, and CATE RMSE remains sub-optimal.
- **In contrast:** The hierarchical within-sampler fix improves both point estimation (lower RMSE) and posterior inference (accurate coverage).

### 5.3 Why Not Full Unconstrained $T \times T$ Kronecker GP Covariance?
Estimating an unconstrained $T \times T$ covariance matrix $\Sigma_T$ involves $T(T+1)/2$ parameters:
- When $T \approx 15\text{--}30$, an unconstrained covariance matrix overfits rapidly unless heavily regularized.
- The combination of a random intercept $\gamma_i$ (permanent correlation) plus $\text{AR}(1)$ (transitory correlation) accounts for over $95\%$ of empirical panel covariance structures using only two scalar parameters ($\sigma_\gamma^2, \rho$).

---

## 6. Implementation Architecture for `google/longbet`

### 6.1 C++ Data Structures (`src/model.h`)
Add state variables to track unit intercepts and persistence:
```cpp
class LongBetModel {
public:
    // Existing members...
    
    // New parameters for dependent errors:
    bool use_random_intercept;
    bool use_ar1;
    
    std::vector<double> gamma; // length N: unit random intercepts
    double sigma_gamma;        // standard deviation of gamma
    double rho;                // AR(1) persistence parameter
    
    // Sufficiency statistics for gamma:
    std::vector<double> sum_res_unit; // length N
};
```

### 6.2 Sampling Loop in C++ (`src/longbet.cpp`)
Integrate the conjugate updates into the sweep loop:
```cpp
// Inside MCMC sweep loop:
if (model->use_random_intercept) {
    update_unit_intercepts(model, Y, mu_hat, tau_hat, residuals, N, T);
    update_sigma_gamma(model, N);
}

if (model->use_ar1) {
    update_rho(model, residuals, N, T);
}

// Compute quasi-differenced residuals before tree updates:
compute_whitened_residuals(model, Y, residuals_whitened, N, T);

// Pass residuals_whitened to prognostic and treatment XBART updates...
```

### 6.3 R API Extensions (`R/longbet.R`)
Expose user controls with sensible defaults:
```r
longbet <- function(y, x, x_trt, z, t, pcat,
                    random_intercept = TRUE,
                    autoregressive = TRUE,
                    num_sweeps = 60, num_burnin = 20,
                    ...) {
  # Input validation
  # Pass random_intercept and autoregressive flags to C++ backend
}
```

---

## 7. Verification and Benchmark Plan

To validate the implementation against the existing package, run a 100-replication Monte Carlo simulation using the DGP from `longbet.qmd`:

1. **Point Accuracy (RMSE):**
   - Measure RMSE of $\hat{\tau}_i(S)$ against ground truth. Expected result: **15%–30% reduction in RMSE**, as trees are freed from memorizing unit baselines.
2. **Prognostic Accuracy:**
   - Measure residual standard deviation on the holdout cohort. Expected result: $\hat{\sigma}$ drops from $0.70$ to the true simulation noise level $0.28$.
3. **Posterior Coverage:**
   - Evaluate empirical coverage of the 95% credible intervals for the ATT trajectory across event times. Expected result: **coverage increases from ~78% to 94%–96%**.
4. **Holdout Residual Autocorrelation:**
   - Compute lag-1 residual correlation $\text{cor}(e_{i, t}, e_{i, t+1})$ among holdout units. Expected result: **correlation drops from $0.105$ to within Monte Carlo error of $0.000$**.

---

## 8. Conclusion

Extending LongBet to include **latent unit random intercepts ($\gamma_i$)** and **autoregressive innovation whitening ($\text{AR}(1)$)** directly targets the mathematical vulnerability of the current model. It transforms a misspecified i.i.d. likelihood into a fully generative hierarchical panel tree model. This simultaneously resolves interval undercoverage and produces materially more accurate treatment effect estimates and forward projections, all while maintaining the computational speed of XBART.
