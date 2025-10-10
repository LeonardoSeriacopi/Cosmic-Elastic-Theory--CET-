import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load orbital parameters from ATNF filtered CSV
atnf = pd.read_csv("atnf_filtered_pilot.csv")

# Expect columns PB (days), T0 (MJD reference epoch)
PB_days = float(atnf["PB"].values[0])    # orbital period in days
T0 = float(atnf["T0"].values[0])         # reference epoch in MJD

print(f"Orbital period (days): {PB_days}, Reference epoch T0: {T0}")

# Load observation catalog
catalog = pd.read_csv("catalogo_observacoes.csv")

# Collect results from all subint CSVs
all_data = []

for idx, row in catalog.iterrows():
    filename = row["filename"]
    mjd_start = row["MJD_start"]

    subint_file = filename.replace(".fits", "_subint.csv")
    if not os.path.exists(subint_file):
        continue

    subint = pd.read_csv(subint_file)

    # Compute absolute MJD for each subint
    times_mjd = mjd_start + subint["OFFS_SUB_s"] / 86400.0

    # Compute orbital phase
    phases = ((times_mjd - T0) / PB_days) % 1.0

    # Store results
    df_temp = pd.DataFrame({
        "phase": phases,
        "intensity": subint["INTENSIDADE_MEDIA"],
        "file": filename
    })
    all_data.append(df_temp)

# Combine all into single dataframe
if len(all_data) == 0:
    raise RuntimeError("No subint data found.")
data = pd.concat(all_data, ignore_index=True)

# Plot: Intensity vs Orbital Phase
plt.figure(figsize=(10,5))
plt.scatter(data["phase"], data["intensity"], s=10, alpha=0.7)
plt.xlabel("Orbital Phase")
plt.ylabel("Mean Intensity")
plt.title("Pulsar Emission vs Orbital Phase")
plt.grid(True)
plt.tight_layout()
plt.savefig("intensity_vs_orbital_phase.png", dpi=150)
plt.close()

# Plot: Histogram by phase
plt.figure(figsize=(10,5))
plt.hist(data["phase"], bins=20, weights=data["intensity"], alpha=0.7)
plt.xlabel("Orbital Phase")
plt.ylabel("Summed Intensity")
plt.title("Histogram of Pulsar Intensity by Orbital Phase")
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_orbital_phase.png", dpi=150)
plt.close()

# Save combined dataset with phase and intensity
data.to_csv("all_subints_with_phase.csv", index=False)
print("✅ Combined dataset saved to all_subints_with_phase.csv")

print("✅ Analysis done. Plots saved as intensity_vs_orbital_phase.png and histogram_orbital_phase.png")