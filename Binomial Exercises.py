# EXERCISE 1
import math

# parameters
n = 10      # total quanta
p = 0.2     # probability of release per quantum

# compute binomial probabilities
probabilities = []
for k in range(n + 1):
    prob = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    probabilities.append(prob)

# print results
for k, prob in enumerate(probabilities):
    print(f"P({k} quanta released) = {prob:.6f}")




# EXERCISE 2
import math

# parameters
n = 14   # total quanta
k = 8    # observed quanta released

# function to compute binomial likelihood
def likelihood(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# compute likelihood for specific p values
p_values = [0.1, 0.7] + [i/10 for i in range(1, 11)]  # 0.1, 0.7, and all deciles 0.1–1.0
results = {}

for p in p_values:
    results[p] = likelihood(n, k, p)

# print results
for p in sorted(results.keys()):
    print(f"Likelihood of p={p:.1f}: {results[p]:.6e}")

# find maximum likelihood
best_p = max(results, key=results.get)
print(f"\nMaximum likelihood at p={best_p:.1f} with likelihood {results[best_p]:.6e}")




# EXERCISE 3 
import math

# parameters
n = 14
observations = [8, 5]  # two experiments: 8 released, then 5 released

# likelihood function for a single experiment
def likelihood_single(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# total likelihood across all observations
def total_likelihood(n, observations, p):
    L = 1.0
    for k in observations:
        L *= likelihood_single(n, k, p)
    return L

# total log-likelihood across all observations
def total_log_likelihood(n, observations, p):
    logL = 0.0
    for k in observations:
        Lk = likelihood_single(n, k, p)
        if Lk > 0:
            logL += math.log(Lk)
        else:
            return float("-inf")  # avoid log(0)
    return logL

# check for p = 0.1 (as in the question)
p_test = 0.1
L_01 = total_likelihood(n, observations, p_test)
logL_01 = total_log_likelihood(n, observations, p_test)
print(f"For p={p_test}:")
print(f"  Total likelihood     = {L_01:.6e}")
print(f"  Total log-likelihood = {logL_01:.6f}")

# now compute likelihoods for deciles 0.1–1.0
p_values = [i/10 for i in range(1, 11)]
results = {}
for p in p_values:
    results[p] = (total_likelihood(n, observations, p),
                  total_log_likelihood(n, observations, p))

# print results
print("\nLikelihoods and log-likelihoods at deciles:")
for p, (L, logL) in results.items():
    print(f"p={p:.1f}:  L={L:.6e},  logL={logL:.6f}")

# find maximum likelihood estimate (MLE)
best_p = max(results, key=lambda p: results[p][0])
print(f"\nMaximum likelihood at p={best_p:.1f}")

# finer grid search (higher resolution, e.g., 0.01 steps)
fine_p_values = [i/100 for i in range(1, 100)]
fine_results = {p: total_likelihood(n, observations, p) for p in fine_p_values}
best_p_fine = max(fine_results, key=fine_results.get)
print(f"Refined MLE (step=0.01): p={best_p_fine:.2f}")

# even finer grid (0.001 steps)
finer_p_values = [i/1000 for i in range(1, 1000)]
finer_results = {p: total_likelihood(n, observations, p) for p in finer_p_values}
best_p_finer = max(finer_results, key=finer_results.get)
print(f"Refined MLE (step=0.001): p={best_p_finer:.3f}")



# Exercise 4
import math
import numpy as np

# parameters
n = 14  # total quanta per experiment

# observed counts (k -> count)
counts = {
    0: 0, 1: 0, 2: 3, 3: 7, 4: 10,
    5: 19, 6: 26, 7: 16, 8: 16, 9: 5,
    10: 5, 11: 0, 12: 0, 13: 0, 14: 0
}

# single observation probability
def binom_prob(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# total log-likelihood
def log_likelihood(n, counts, p):
    logL = 0.0
    for k, c in counts.items():
        if c > 0:
            pk = binom_prob(n, k, p)
            if pk > 0:
                logL += c * math.log(pk)
            else:
                return float("-inf")  # avoid log(0)
    return logL

# search over candidate p values with resolution 0.01
p_candidates = np.arange(0.01, 1.00, 0.01)
logL_values = {p: log_likelihood(n, counts, p) for p in p_candidates}

# find maximum likelihood estimate
p_hat = max(logL_values, key=logL_values.get)

print(f"Maximum likelihood estimate (p-hat) = {p_hat:.2f}")
print(f"Log-likelihood at p-hat = {logL_values[p_hat]:.6f}")



#Exercise 5
import math
from scipy.stats import binom

# Parameters
n = 14           # total quanta
k_obs = 7        # observed releases
p_null = 0.3     # Null hypothesis: true release probability (baseline condition)

# Step 1: Compute p-hat (MLE from data)
p_hat = k_obs / n

# Step 2: Probability of observing exactly k_obs under Null Hypothesis
p_value_exact = binom.pmf(k_obs, n, p_null)

# Step 3: (Optional) two-sided p-value: probability of observing outcomes
# as or more extreme than k_obs, relative to expected mean n*p_null
mean_null = n * p_null
if k_obs > mean_null:
    # probability of k >= k_obs
    p_value_twosided = sum(binom.pmf(k, n, p_null) for k in range(k_obs, n + 1))
else:
    # probability of k <= k_obs
    p_value_twosided = sum(binom.pmf(k, n, p_null) for k in range(0, k_obs + 1))

# Print results
print(f"Observed p-hat = {p_hat:.3f}")
print(f"Probability of exactly {k_obs} releases under H0 (p=0.3): {p_value_exact:.5f}")
print(f"Two-sided p-value under H0: {p_value_twosided:.5f}")