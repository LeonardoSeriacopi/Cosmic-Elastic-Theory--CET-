
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("CEERS_catalog_with_zcet.csv")
df["delta_z"] = df["z_spec"] - df["z_cet"]
df_valid = df.dropna(subset=["delta_z", "z_spec", "density_rhocrit"])

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_valid["z_spec"],
    df_valid["delta_z"],
    c=df_valid["density_rhocrit"],
    cmap="plasma",
    alpha=0.9,
    edgecolors="none"
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
cbar = plt.colorbar(scatter)
cbar.set_label("Normalized Density (ρ / ρ_crit)")
plt.xlabel("Observed Redshift (z_spec)")
plt.ylabel("Δz = z_spec - z_CET")
plt.title("Figure 4: Δz vs z_spec colored by density (CEERS)")
plt.grid(True)
plt.tight_layout()
plt.savefig("figura4_deltaz_vs_zspec_colored_by_density.png", dpi=300)
plt.show()
