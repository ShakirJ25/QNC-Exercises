import numpy as np
import scipy.stats as st
import pandas as pd

# Set random seed for reproducibility (random number generator)
np.random.seed(42)

# Parameters
true_mu = 10.0
true_sigma = 2.0
sample_sizes = [5, 10, 20, 40, 80, 160, 1000]
alpha = 0.05
z_crit = st.norm.ppf(1 - alpha/2)

#Confidence Intervals based on methods
def z_interval_known_sigma(x, sigma=true_sigma):
    """Frequentist CI with known sigma (Z-interval)."""
    n = len(x)
    mean = x.mean()
    se = sigma / np.sqrt(n)
    lo = mean - z_crit * se
    hi = mean + z_crit * se
    return lo, hi

def bootstrap_percentile(x, B=5000):
    """Bootstrap percentile CI."""
    rng = np.random.default_rng(12345)
    n = len(x)
    boot_means = [rng.choice(x, size=n, replace=True).mean() for _ in range(B)]
    lo = np.percentile(boot_means, 100 * alpha/2)
    hi = np.percentile(boot_means, 100 * (1 - alpha/2))
    return lo, hi

def bayesian_credible_interval(x, mu0=0.0, kappa0=1e-6, alpha0=1e-3, beta0=1e-3):
    """Bayesian credible interval for mean with Normal-Inverse-Gamma prior."""
    n = len(x)
    mean = x.mean()
    ssq = np.sum((x - mean) ** 2)

    # Posterior parameters
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean) / kappa_n
    alpha_n = alpha0 + n / 2
    beta_n = beta0 + 0.5 * ssq + (kappa0 * n * (mean - mu0) ** 2) / (2 * kappa_n)

    # Posterior distribution for mu is Student-t
    df = 2 * alpha_n
    scale = np.sqrt(beta_n / (alpha_n * kappa_n))
    lo = mu_n + st.t.ppf(alpha/2, df) * scale
    hi = mu_n + st.t.ppf(1 - alpha/2, df) * scale
    return lo, hi

# Simulating datasets

results = []
for n in sample_sizes:
    x = np.random.normal(true_mu, true_sigma, size=n)

    z_lo, z_hi = z_interval_known_sigma(x)
    b_lo, b_hi = bootstrap_percentile(x)
    bayes_lo, bayes_hi = bayesian_credible_interval(x)

    results.append({
        "n": n,
        "sample_mean": x.mean(),
        "Z_CI": (z_lo, z_hi),
        "Bootstrap_CI": (b_lo, b_hi),
        "Bayesian_CI": (bayes_lo, bayes_hi)
    })

# Convert to DataFrame for pretty output
df = pd.DataFrame(results)
pd.set_option("display.precision", 4)
print(df)
