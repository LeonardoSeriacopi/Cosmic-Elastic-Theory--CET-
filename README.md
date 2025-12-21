# Cosmic Elastic Theory — Data and Empirical Tests

This repository contains the datasets, scripts, and figures used in the empirical evaluation of the Cosmic Elastic Theory (CET).
Each test examines whether the process-based dissipation model proposed in CET is compatible with publicly available astronomical observations.

The material is organized to allow full reproducibility of the analyses.

---

## Repository Structure

### test_1_pantheon_los/
Supernova luminosity data (Pantheon+SH0ES).
LOS-density reconstruction and transition analysis.

### test_2_ceers_los/
JWST–CEERS high-z galaxies.
LOS environmental metrics and CET dissipation interpretation.

### test_3_gravitational_waves_los/
GW events (LIGO/Virgo) cross-matched with SDSS galaxies.
Propagation behaviour along structured environments.

### test_4_weak_lensing_dispersion/
DES weak-lensing variance.
Comparison with CET predicted dispersion scaling.

### test_5_wide_binares_estability/
Gaia DR3 wide binaries.
Stability index vs. local density.

### test_6_sn2023zkd_event_analisys/
SN 2023zkd.
Light-curve reconstruction and elastic-coupling analysis.

### test_7_pulsar_nulling/
Nulling statistics in pulsars.
Critical variance and hierarchical temporal coupling.

### test_8_quasars_z_env/
Quasar redshift vs. environment.
Environmental dissipation constraints.

### test_9_CMB_residuals/
CMB spectral residuals (COBE/Planck/SPT).
Directional friction estimates, composite sky maps (WMAP/Planck), and comparison with CET dissipation scaling.

### test_10_CMB_cold_spot/
CMB Cold Spot and internal secondary minimum (CS1/CS2).
2MASS galaxy-density analysis, 2D residual maps, multi-frequency WMAP/Planck temperature measurements, and structural interpretation under CET dissipation.

### test_11_SDSS_DES_tz/
**SDSS/DESI Dissipation Curve and Kinematic Acceleration.**
**Empirical validation of the Universal Dissipation Curve (UDC) and its derivatives, proving the emergence of the Decaying Kinematic Impulse ($a_{\text{kin}}$) in the Transition Zone (TZ).**

###test_12_jades-los-freedman
**JADES Dissipation Curve and Saturated Regime**
**Empirical validation of the saturating dissipation model at High Redshift and the Role of BAO--Freedman Density Homogenization.**

### exploratory_tests/
Older scripts and preliminary iterations.
Kept for traceability only.

---

## Usage

Each test directory contains:
- processed datasets (tables or arrays);
- figures used in the paper;
- Python scripts for reproducing the analysis.

External datasets (Planck, WMAP, DES, Gaia, SDSS, CEERS, Pantheon+, LIGO/Virgo, etc.) are referenced in the scripts.
Raw CMB maps are not included due to size restrictions; official download links are provided.

---

## License

Distributed under **CC BY 4.0**.
Reuse permitted with attribution.