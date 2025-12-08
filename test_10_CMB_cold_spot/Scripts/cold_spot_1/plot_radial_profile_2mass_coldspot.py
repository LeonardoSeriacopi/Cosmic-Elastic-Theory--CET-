#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plot_radial_profile_2mass_coldspot.py

Plots:
 - the observed radial number–density profile around the CMB Cold Spot
 - the MEAN radial profile from random centres
with error bars.

Requires the file produced by radial_profile_2mass_coldspot_mcmean.py:
    radial_profile_coldspot_vs_random.txt
"""

import numpy as np
import matplotlib.pyplot as plt

infile = "resultado_radial_profile_2mass_coldspot_mcmean.txt"

def load_profile(fname):
    r_in, r_out, r_mid = [], [], []
    N_obs, area = [], []
    dens_obs, sigma_obs = [], []
    dens_rand, sigma_rand = [], []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue

            r_in.append(float(parts[0]))
            r_out.append(float(parts[1]))
            r_mid.append(float(parts[2]))
            N_obs.append(int(parts[3]))
            area.append(float(parts[4]))
            dens_obs.append(float(parts[5]))
            sigma_obs.append(float(parts[6]))
            dens_rand.append(float(parts[7]))
            sigma_rand.append(float(parts[8]))

    return (np.array(r_in), np.array(r_out), np.array(r_mid),
            np.array(N_obs), np.array(area),
            np.array(dens_obs), np.array(sigma_obs),
            np.array(dens_rand), np.array(sigma_rand))


def main():

    (r_in, r_out, r_mid,
     N_obs, area,
     dens_obs, sigma_obs,
     dens_rand, sigma_rand) = load_profile(infile)

    # ==============================
    # Figure 1: observed vs random
    # ==============================
    plt.figure(figsize=(6.2, 4.7))

    plt.errorbar(
        r_mid, dens_obs, yerr=sigma_obs,
        fmt="o-", capsize=3, label="Cold Spot (observed)",
        color="tab:blue"
    )

    plt.errorbar(
        r_mid, dens_rand, yerr=sigma_rand,
        fmt="s--", capsize=3, label="Random centres (mean)",
        color="tab:orange"
    )

    plt.xlabel(r"Radius $\theta$ around Cold Spot [deg]")
    plt.ylabel(r"2MASS number density [deg$^{-2}$]")
    plt.title("Radial Number–Density Profile: Cold Spot vs Random Sky")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig_coldspot_radial_profile.png", dpi=300)

    # ==============================
    # Figure 2: density excess
    # ==============================
    delta = dens_obs - dens_rand
    sigma_delta = np.sqrt(sigma_obs**2 + sigma_rand**2)

    plt.figure(figsize=(6.2, 4.7))

    plt.errorbar(
        r_mid, delta, yerr=sigma_delta,
        fmt="o-", capsize=3, color="tab:green"
    )

    plt.axhline(0.0, linestyle=":", color="black")

    plt.xlabel(r"Radius $\theta$ around Cold Spot [deg]")
    plt.ylabel(r"Density excess $\Delta \rho$ [deg$^{-2}$]")
    plt.title("Cold Spot Density Excess Relative to Random Sky")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig_coldspot_density_excess.png", dpi=300)

    print("[OK] Figures saved as:")
    print("     fig_coldspot_radial_profile.png")
    print("     fig_coldspot_density_excess.png")


if __name__ == "__main__":
    main()