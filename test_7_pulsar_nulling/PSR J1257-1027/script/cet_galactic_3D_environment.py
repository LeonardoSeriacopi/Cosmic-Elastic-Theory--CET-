#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CET 3D Galactic Context Analyzer
--------------------------------
Computes galactic coordinates (l, b, Z) for a pulsar and visualizes its
location relative to the Galactic disk — a validation of CET environmental prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# === Pulsar data ===
source_name = "J1257"
ra_str = "12:57:04.0"
dec_str = "-10:27:06.0"

# === Approximate distance estimate ===
# If you have DM and a model distance, replace this value
# Typical isolated pulsars have distances between 0.5 and 3 kpc
distance_kpc = 1.8   # example (can be updated from DM later)

# === Convert to galactic coordinates ===
coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg), frame="icrs")
l = coord.galactic.l.deg
b = coord.galactic.b.deg

# === Compute Galactic position in kpc ===
r = distance_kpc
l_rad = np.deg2rad(l)
b_rad = np.deg2rad(b)

# Convert to Galactocentric (Sun at X=-8.2 kpc)
R_sun = 8.2  # kpc
x = R_sun - r * np.cos(b_rad) * np.cos(l_rad)
y = -r * np.cos(b_rad) * np.sin(l_rad)
z = r * np.sin(b_rad)

# === Classify region by Z-height ===
if abs(z) < 0.3:
    region = "Galactic Disk"
    interpretation = "Dense medium; weak CET activity expected."
elif abs(z) < 1.5:
    region = "Transition Zone"
    interpretation = "Intermediate density; moderate CET elastic modulation."
else:
    region = "Galactic Halo / Edge"
    interpretation = "Low-density gradient; strong CET intermittency regime."

# === Print summary ===
print("\n=== CET Galactic 3D Position Report ===")
print(f"Source: {source_name}")
print(f"RA: {ra_str}, Dec: {dec_str}")
print(f"Galactic l: {l:.2f}°, b: {b:.2f}°")
print(f"Distance: {r:.2f} kpc")
print(f"Galactic height Z: {z:.2f} kpc")
print(f"Region: {region}")
print(f"CET Interpretation: {interpretation}")

# === Save report ===
df = pd.DataFrame([{
    "source": source_name,
    "ra": ra_str,
    "dec": dec_str,
    "l_deg": l,
    "b_deg": b,
    "distance_kpc": r,
    "x_kpc": x,
    "y_kpc": y,
    "z_kpc": z,
    "region": region,
    "interpretation": interpretation
}])
df.to_csv("galactic_3D_environment_report.csv", index=False)
print("✓ Saved → galactic_3D_environment_report.csv")

# === 3D Visualization ===
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection="3d")

# Draw galactic plane (disk)
plane_size = 12
ax.plot_surface(
    *np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                 np.linspace(-plane_size, plane_size, 10)),
    np.zeros((10,10)), color='gray', alpha=0.15, zorder=1
)

# Plot the pulsar
ax.scatter(x, y, z, color="red", s=80, label=f"{source_name}")
ax.text(x, y, z+0.2, source_name, color="red", fontsize=9)

# Mark Sun and Galactic Center
ax.scatter(0, 0, 0, color="gold", marker="*", s=120, label="Galactic Center")
ax.scatter(R_sun, 0, 0, color="orange", marker="o", s=60, label="Sun")

# Axes and labels
ax.set_xlabel("X (kpc, toward Galactic Center)")
ax.set_ylabel("Y (kpc, rotation direction)")
ax.set_zlabel("Z (kpc, height)")
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)
ax.set_zlim(-5, 5)
ax.set_title("CET Galactic 3D Context — Disk vs Halo Position", fontsize=12)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("galactic_3D_environment_map.png", dpi=200)
plt.show()

print("✓ Saved → galactic_3D_environment_map.png")