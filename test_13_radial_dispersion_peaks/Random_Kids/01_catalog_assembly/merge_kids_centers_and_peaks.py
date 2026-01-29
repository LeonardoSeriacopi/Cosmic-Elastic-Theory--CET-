import pandas as pd

centers = pd.read_csv("kids_10k_centers.csv")
peaks   = pd.read_csv("kids_dispersion_peaks.csv")

# padroniza nomes
centers = centers.rename(columns={"Z": "Z_center"})
peaks   = peaks.rename(columns={"Z": "Z_peak"})

df = centers.merge(peaks, on="ID", how="inner")
df.to_csv("anchor_out_merged_with_proxies.csv", index=False)

print("Merged rows:", len(df))
print(df.head())