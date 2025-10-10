import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Load the combined dataset
data = pd.read_csv("all_subints_with_phase.csv")

# Spearman correlation
rho, pval = spearmanr(data["phase"], data["intensity"])
print(f"Spearman correlation: rho={rho:.3f}, p={pval:.3e}")

# Kendall tau correlation
tau, pval_tau = kendalltau(data["phase"], data["intensity"])
print(f"Kendall tau: tau={tau:.3f}, p={pval_tau:.3e}")

# Quick interpretation
if pval < 0.05:
    print("✅ Significant correlation detected with Spearman")
else:
    print("⚠️ No significant correlation with Spearman")

if pval_tau < 0.05:
    print("✅ Significant correlation detected with Kendall tau")
else:
    print("⚠️ No significant correlation with Kendall tau")