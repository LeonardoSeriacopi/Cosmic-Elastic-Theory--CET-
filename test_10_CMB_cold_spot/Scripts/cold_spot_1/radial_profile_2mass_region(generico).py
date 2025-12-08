#!/usr/bin/env python3
"""
radial_profile_2mass_region.py

Radial number–density profile around an arbitrary sky centre
(using a 2MASS cutout) + constant-density and linear-gradient fits.

Use this to compare:
 - CMB Cold Spot (classic)
 - any other cold / control region

by just changing the centre (L_CENTER, B_CENTER) and the input file.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


# =======================
# USER PARAMETERS
# =======================

# 2MASS cutout around the region you want to study
INFILE = "2mass.tsv"          # mude se usar outro recorte, ex: "2mass_cs2.tsv"

# Label for printing / filenames
REGION_LABEL = "Cold Spot 1"  # ex: "Cold Spot 2", "Control region", etc.

# Centre of the region in GALACTIC coordinates (deg)
# (para o Cold Spot clássico: l=209, b=-57)
L_CENTER = 209.0
B_CENTER = -57.0

# Radial bins (deg) – mesmos que usamos antes
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Magnitude range in K-band used in previous tests
KMIN = 8.0
KMAX = 16.0

# Output text file
OUTFILE = "radial_profile_2mass_region_results.txt"


# =======================
# HELPER FUNCTIONS
# =======================

def build_radial_profile(theta_deg, k_mag, r_bins, kmin, kmax):
    """
    Compute N, area, density and poisson error in each radial annulus.
    """
    mask_mag = (k_mag >= kmin) & (k_mag < kmax)
    theta = theta_deg[mask_mag]

    r_mid = []
    N = []
    area = []
    dens = []
    sigma = []

    for r_in, r_out in zip(r_bins[:-1], r_bins[1:]):
        sel = (theta >= r_in) & (theta < r_out)
        n = np.sum(sel)
        a = np.pi * (r_out**2 - r_in**2)  # deg^2
        if n > 0:
            rho = n / a
            err = np.sqrt(n) / a
        else:
            rho = 0.0
            err = 0.0

        r_mid.append(0.5 * (r_in + r_out))
        N.append(n)
        area.append(a)
        dens.append(rho)
        sigma.append(err)

    return (np.array(r_mid), np.array(N), np.array(area),
            np.array(dens), np.array(sigma))


def fit_constant_density(r, dens, sigma):
    """
    Weighted fit to a constant density ρ(r) = ρ0.
    """
    w = 1.0 / (sigma**2)
    rho0 = np.sum(w * dens) / np.sum(w)

    chi2 = np.sum(((dens - rho0) / sigma)**2)
    dof = len(dens) - 1
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return rho0, chi2, chi2_red, dof


def fit_linear_density(r, dens, sigma):
    """
    Weighted linear fit ρ(r) = a + b r.
    Returns a, b, σ_a, σ_b, χ², χ²_red, dof.
    """
    w = 1.0 / (sigma**2)
    Sw = np.sum(w)
    Swr = np.sum(w * r)
    Swr2 = np.sum(w * r**2)
    Swd = np.sum(w * dens)
    Swrd = np.sum(w * r * dens)

    Delta = Sw * Swr2 - Swr**2

    a = (Swr2 * Swd - Swr * Swrd) / Delta
    b = (Sw * Swrd - Swr * Swd) / Delta

    # Covariance matrix
    sigma_a2 = Swr2 / Delta
    sigma_b2 = Sw / Delta
    sigma_a = np.sqrt(sigma_a2)
    sigma_b = np.sqrt(sigma_b2)

    chi2 = np.sum(((dens - (a + b * r)) / sigma)**2)
    dof = len(dens) - 2
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return a, b, sigma_a, sigma_b, chi2, chi2_red, dof


# =======================
# MAIN
# =======================

def main():
    print("=========================================================")
    print(f"[INFO] Reading 2MASS cutout: {INFILE}")
    df = pd.read_csv(INFILE, sep="\t", comment="#")

    # RA/Dec in degrees
    ra = pd.to_numeric(df["RAJ2000"], errors="coerce").values
    dec = pd.to_numeric(df["DEJ2000"], errors="coerce").values
    k_mag = pd.to_numeric(df["Kmag"], errors="coerce").values

    mask_valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(k_mag)

    ra = ra[mask_valid]
    dec = dec[mask_valid]
    k_mag = k_mag[mask_valid]

    print(f"[INFO] Valid rows (RA/Dec/Kmag): {ra.size}")
    print(f"[INFO] RA min/max  = {ra.min():6.3f} – {ra.max():6.3f} deg")
    print(f"[INFO] Dec min/max = {dec.min():6.3f} – {dec.max():6.3f} deg")

    # Convert to Galactic and compute angular distance to centre
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs").galactic
    l = coord.l.deg
    b = coord.b.deg

    cs_center = SkyCoord(l=L_CENTER*u.deg, b=B_CENTER*u.deg, frame="galactic")
    sep = cs_center.separation(SkyCoord(l=l*u.deg, b=b*u.deg, frame="galactic"))
    theta_deg = sep.deg

    print(f"[INFO] θ min/max to centre ({L_CENTER},{B_CENTER}) = "
          f"{theta_deg.min():.2f} – {theta_deg.max():.2f} deg")

    # Radial profile
    print("=========================================================")
    print(f"[INFO] Building radial profile for {REGION_LABEL} "
          f"(K in [{KMIN:.1f}, {KMAX:.1f}])...")
    r_mid, N, area, dens, sigma = build_radial_profile(
        theta_deg, k_mag, RADIAL_BINS, KMIN, KMAX
    )

    print("[INFO] Radial profile:")
    print(" r_in  r_out   N      area_deg2   dens[1/deg2]   sigma")
    for i in range(len(r_mid)):
        r_in = RADIAL_BINS[i]
        r_out = RADIAL_BINS[i+1]
        print(f" {r_in:4.1f}  {r_out:4.1f}  {N[i:6d}  "
              f"{area[i:10.2f}  {dens[i:10.3f}  {sigma[i:8.3f}]")

    # Constant-density fit
    rho0, chi2_c, chi2r_c, dof_c = fit_constant_density(r_mid, dens, sigma)

    # Linear fit
    a, b, sigma_a, sigma_b, chi2_l, chi2r_l, dof_l = fit_linear_density(
        r_mid, dens, sigma
    )
    sig_b = b / sigma_b if sigma_b > 0 else np.nan

    print("=========================================================")
    print("[RESULT] Constant-density fit (LCDM-like baseline):")
    print(f"  rho0_const     = {rho0:.3f} sources/deg^2")
    print(f"  chi2_const     = {chi2_c:.3f}")
    print(f"  chi2_red_const = {chi2r_c:.3f} (dof={dof_c})")
    print("---------------------------------------------------------")
    print("[RESULT] Linear fit ρ(r) = a + b r:")
    print(f"  a = {a:.3f} ± {sigma_a:.3f} sources/deg^2")
    print(f"  b = {b:.3f} ± {sigma_b:.3f} sources/deg^2/deg "
          f"({sig_b:.2f} sigma)")
    print(f"  chi2_linear    = {chi2_l:.3f}")
    print(f"  chi2_red_linear= {chi2r_l:.3f} (dof={dof_l})")
    print("=========================================================")

    # Save to text file
    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write(f"# Radial profile for region: {REGION_LABEL}\n")
        f.write(f"# Centre (l,b) = ({L_CENTER:.3f}, {B_CENTER:.3f}) deg\n")
        f.write(f"# K range: [{KMIN:.1f}, {KMAX:.1f}]\n")
        f.write("#\n")
        f.write("# r_in  r_out   r_mid   N   area_deg2   dens   sigma\n")
        for i in range(len(r_mid)):
            r_in = RADIAL_BINS[i]
            r_out = RADIAL_BINS[i+1]
            f.write(f"{r_in:6.3f} {r_out:6.3f} {r_mid[i:6.3f} "
                    f"{N[i]:8d} {area[i:10.3f} {dens[i:10.3f} {sigma[i:10.3f}\n")
        f.write("#\n")
        f.write("# Constant-density fit:\n")
        f.write(f"# rho0_const   = {rho0:.6f}\n")
        f.write(f"# chi2_const   = {chi2_c:.6f}\n")
        f.write(f"# chi2_red     = {chi2r_c:.6f}\n")
        f.write("#\n")
        f.write("# Linear fit ρ(r) = a + b r:\n")
        f.write(f"# a        = {a:.6f}\n")
        f.write(f"# sigma_a  = {sigma_a:.6f}\n")
        f.write(f"# b        = {b:.6f}\n")
        f.write(f"# sigma_b  = {sigma_b:.6f}\n")
        f.write(f"# b/sigma_b= {sig_b:.3f}\n")
        f.write(f"# chi2_lin = {chi2_l:.6f}\n")
        f.write(f"# chi2r_lin= {chi2r_l:.6f}\n")

    print(f"[INFO] Results written to: {OUTFILE}")


if __name__ == "__main__":
    main()