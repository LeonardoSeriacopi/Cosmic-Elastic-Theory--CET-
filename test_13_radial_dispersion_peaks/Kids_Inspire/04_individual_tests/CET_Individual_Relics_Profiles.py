import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from astropy.cosmology import Planck18 as cosmo

# ============================================================
# CONFIG
# ============================================================
sns.set_theme(style="whitegrid")

piso = 0.31      # Mpc
R_MAX = 5.2      # Mpc
NBINS = 30

# ============================================================
# LOAD MASTER TABLE (Paper 5)
# ============================================================
df_master = pd.read_csv("CET_Paper5_Master.csv")

summary = []

# ============================================================
# ATLAS FIGURE
# ============================================================
fig_atlas, axes = plt.subplots(4, 4, figsize=(22, 18), sharex=True)
axes = axes.flatten()
atlas_idx = 0

# ============================================================
# MAIN LOOP — GALAXY BY GALAXY
# ============================================================
for _, row in df_master.iterrows():

    gid = row["ID"].strip()
    file_name = f"{gid}.csv"

    if not os.path.exists(file_name):
        print(f"[SKIP] Missing file for {gid}")
        continue

    # ----------------------------
    # LOAD SENSOR DATA
    # ----------------------------
    df = pd.read_csv(file_name)

    z_lens = row["z_lens"]
    D_A = cosmo.angular_diameter_distance(z_lens).value  # Mpc

    # Angular → Physical
    df["dist_mpc"] = np.deg2rad(df["dist_deg"]) * D_A

    # Observable
    if "gamma_t" in df.columns:
        obs = np.abs(df["gamma_t"].values)
    else:
        obs = np.sqrt(df["e1"]**2 + df["e2"]**2)

    # ----------------------------
    # RADIAL PROFILE (RMS)
    # ----------------------------
    bins = np.linspace(0.1, R_MAX, NBINS)
    centers = 0.5 * (bins[:-1] + bins[1:])

    profile = []

    for b in range(len(bins) - 1):
        mask = (df["dist_mpc"] >= bins[b]) & (df["dist_mpc"] < bins[b+1])
        vals = obs[mask]

        if len(vals) > 10:
            profile.append(np.sqrt(np.mean(vals**2)))
        else:
            profile.append(np.nan)

    profile = np.array(profile)

    # ----------------------------
    # PEAK & BOUNDARY (FROM MASTER)
    # ----------------------------
    r_peak = row["dist_pico_mpc"]
    r_boundary = row["limite_transbordo_mpc"]

    # ----------------------------
    # INDIVIDUAL FIGURE
    # ----------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(centers, profile, color="#1f2933", lw=2, label="Dispersion Profile")

    plt.axvspan(0, r_peak, color="#2563eb", alpha=0.12, label="Stasis (Potential)")
    plt.axvline(r_peak, color="#2563eb", ls="--", lw=2, label="Peak")

    if not np.isnan(r_boundary):
        plt.axvline(r_boundary, color="#dc2626", lw=2, label="Transition Boundary")

    plt.axvline(piso, color="#6b7280", ls=":", lw=1.5, label="Universal Floor")

    plt.xlabel("Radial Distance [Mpc]", fontsize=12)
    plt.ylabel("Dispersion (RMS)", fontsize=12)

    plt.title(
        f"{gid} | logρ = {row['Idade Estrutural (logρ)']:.2f}",
        fontsize=14, weight="bold"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"CET_Fingerprint_{gid}.png", dpi=200)
    plt.close()

    # ----------------------------
    # ATLAS PANEL
    # ----------------------------
    ax = axes[atlas_idx]

    ax.plot(centers, profile, color="#111827", lw=1.5)
    ax.axvspan(0, r_peak, color="#2563eb", alpha=0.12)
    ax.axvline(r_peak, color="#2563eb", ls="--", lw=1)

    if not np.isnan(r_boundary):
        ax.axvline(r_boundary, color="#dc2626", lw=1)

    ax.axvline(piso, color="#6b7280", ls=":", lw=1)

    ax.set_title(gid, fontsize=10, weight="bold")

    if np.any(~np.isnan(profile)):
        ax.set_ylim(
            np.nanmin(profile) * 0.95,
            np.nanmax(profile) * 1.05
        )

    atlas_idx += 1

    # ----------------------------
    # SUMMARY
    # ----------------------------
    summary.append({
        "ID": gid,
        "z_lens": z_lens,
        "log_rho": row["Idade Estrutural (logρ)"],
        "r_peak_mpc": r_peak,
        "r_boundary_mpc": r_boundary,
        "N_sources": row["N_sources"]
    })

# ============================================================
# FINALIZE ATLAS & SAVE SUMMARY
# ============================================================
for j in range(atlas_idx, len(axes)):
    fig_atlas.delaxes(axes[j])

fig_atlas.suptitle(
    "CET Volumetric Fingerprints — Individual Relic Galaxies",
    fontsize=22, weight="bold"
)

fig_atlas.tight_layout()
fig_atlas.savefig("CET_Atlas_Relics_Individual.png", dpi=300)

pd.DataFrame(summary).to_csv(
    "CET_Relics_Individual_Profile_Summary.csv",
    index=False
)

print("[OK] Individual relic analysis completed.")