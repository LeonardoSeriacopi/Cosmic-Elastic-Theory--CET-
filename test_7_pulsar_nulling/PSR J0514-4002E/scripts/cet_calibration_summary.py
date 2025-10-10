#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Orbital Calibration Summary (versão compatível com bootstrap_summary.csv)
Autor: CET Lab
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Arquivos de entrada ====
FILE_PERIOD = "orbital_period_candidate.csv"
FILE_BOOT_SUMMARY = "bootstrap_summary.csv"
FILE_BOOT_PERIODS = "bootstrap_periods.csv"
OUT_CSV = "cet_calibration_summary.csv"
OUT_PNG = "calibration_overview.png"

# ==== Leitura ====
for f in [FILE_PERIOD, FILE_BOOT_SUMMARY, FILE_BOOT_PERIODS]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Arquivo não encontrado: {f}")

period_df = pd.read_csv(FILE_PERIOD)
boot_summary_df = pd.read_csv(FILE_BOOT_SUMMARY)
boot_periods_df = pd.read_csv(FILE_BOOT_PERIODS)

# ==== Extração de valores ====
p_detected = period_df["best_period_days"].iloc[0]
p_ref = period_df["pb_ref_days"].iloc[0]
boot_median = period_df["bootstrap_median"].iloc[0]
boot_std = period_df["bootstrap_std"].iloc[0]
rel_err = period_df["rel_error_%"].iloc[0]

# Estatísticas bootstrap
boot_mean = boot_summary_df["mean_period_days"].iloc[0]
boot_std_s = boot_summary_df["std_period_days"].iloc[0]
ci_low = boot_summary_df["ci_low_days"].iloc[0]
ci_high = boot_summary_df["ci_high_days"].iloc[0]
n_boot = int(boot_summary_df["n_boot"].iloc[0])

boot_periods = boot_periods_df["detected_period_days"].values

# Consistência: fração dentro de ±10% do valor detectado
within_10 = np.mean(np.abs(boot_periods - p_detected) / p_detected <= 0.1)
within_10_pct = 100 * within_10

# Δlog10(P)
delta_logp = np.log10(p_detected / p_ref)

# ==== Montagem do resumo ====
summary = pd.DataFrame([{
    "Pulsar": "J0514−4002E",
    "P_detected_days": p_detected,
    "P_ref_days": p_ref,
    "Delta_logP": delta_logp,
    "Bootstrap_median_days": boot_median,
    "Bootstrap_std_days": boot_std,
    "Bootstrap_mean_days": boot_mean,
    "Bootstrap_std_summary_days": boot_std_s,
    "Bootstrap_ci_low_days": ci_low,
    "Bootstrap_ci_high_days": ci_high,
    "Bootstrap_samples": n_boot,
    "Bootstrap_consistency_pct": within_10_pct,
    "Relative_error_%": rel_err
}])

summary.to_csv(OUT_CSV, index=False)
print(f"✅ Summary saved: {OUT_CSV}")

# ==== Gráfico comparativo ====
sns.set(style="whitegrid")
plt.figure(figsize=(7,5))
plt.bar(["Detected", "Reference", "Bootstrap mean"],
        [p_detected, p_ref, boot_mean],
        color=["#ff6666", "#4472c4", "#6aa84f"])
plt.ylabel("Orbital Period (days)")
plt.title("CET — Orbital Period Calibration Summary")
plt.text(0, p_detected*1.05, f"{p_detected:.3f} d", ha="center", fontsize=9)
plt.text(1, p_ref*1.05, f"{p_ref:.3f} d", ha="center", fontsize=9)
plt.text(2, boot_mean*1.05, f"{boot_mean:.3f} d", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()

print(f"📊 Plot saved: {OUT_PNG}")
print("✅ Calibration summary complete.")