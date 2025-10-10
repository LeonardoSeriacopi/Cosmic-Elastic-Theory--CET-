# cet_tail_validation.py
# CET — Orbital Tail Asymmetry Validation
# Autor: [Seu Nome]
# Data: 2025-10-05

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

print("\n=== CET Tail Validation: Orbital Morphology Analysis ===")

# === 1. Entrada de dados ===
try:
    df = pd.read_csv("variance_phase_curve.csv")
except FileNotFoundError:
    df = pd.read_csv("variance_phase_curve-1.csv")

phase_col = [c for c in df.columns if "phase" in c.lower()][0]
var_col = [c for c in df.columns if "var" in c.lower()][0]

phase = df[phase_col].values
variance = df[var_col].values

# === 2. Normalização ===
variance_norm = (variance - np.min(variance)) / (np.max(variance) - np.min(variance))

# === 3. Análise de assimetria e curtose ===
skewness = skew(variance_norm)
kurt = kurtosis(variance_norm)

# === 4. Detecção da cauda longa / curta ===
# Definimos o "centro" da curva e comparamos as médias das metades
mid = int(len(phase) / 2)
left_mean = np.mean(variance_norm[:mid])
right_mean = np.mean(variance_norm[mid:])

tail_ratio = right_mean / left_mean if left_mean != 0 else np.nan

if tail_ratio > 1.1:
    tail_type = "→ Cauda longa em fase tardia (fase > 0.5)"
elif tail_ratio < 0.9:
    tail_type = "→ Cauda longa em fase inicial (fase < 0.5)"
else:
    tail_type = "→ Curva aproximadamente simétrica"

# === 5. Relatório no terminal ===
print(f"Assimetria (skewness): {skewness:.3f}")
print(f"Curtose (kurtosis): {kurt:.3f}")
print(f"Relação cauda direita/esquerda: {tail_ratio:.2f}")
print(tail_type)

# === 6. Gráfico anotado ===
plt.figure(figsize=(8, 4))
plt.plot(phase, variance_norm, color='blue', lw=2)
plt.title("CET — Orbital Tail Validation (Variance Curve Morphology)")
plt.xlabel("Orbital Phase (folded)")
plt.ylabel("Normalized Variance of Intensity")

# Marca o ponto médio e regiões de cauda
plt.axvline(0.5, color='gray', ls='--', alpha=0.6)
plt.text(0.05, 0.9, f"Skew = {skewness:.2f}\nKurt = {kurt:.2f}\n{tail_type}",
         transform=plt.gca().transAxes, fontsize=9,
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig("results_CET_tail_validation.png", dpi=150)
plt.show()

# === 7. Salva o resumo ===
summary = pd.DataFrame({
    "skewness": [skewness],
    "kurtosis": [kurt],
    "tail_ratio": [tail_ratio],
    "tail_type": [tail_type]
})
summary.to_csv("tail_validation_summary.csv", index=False)

print("\n✅ Tail validation concluída com sucesso.")
print("Gráfico salvo em: results_CET_tail_validation.png")
print("Resumo salvo em: tail_validation_summary.csv")