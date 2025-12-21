# JADES LOS–Freedman Analysis  
### Model-independent test of photon dissipation using JADES and GOODS-S

This repository contains the full data-processing pipeline, intermediate products,
and final figures used in the *LOS–Freedman analysis of the JADES GOODS-S sample*,
as presented in the Cosmic Elastic Theory (CET) framework.

The goal of this analysis is to isolate **non-geometric contributions to the photon
signal** by removing the luminosity–distance dependence and probing residual
dissipation along the line of sight (LOS), in a fully model-independent way.

---

## Scientific context

In standard ΛCDM, once the geometric distance contribution is removed from the
magnitude–redshift relation, no large-scale structured residuals are expected.
Residuals should fluctuate around zero with no systematic dependence on redshift
or environment.

In contrast, *Cosmic Elastic Theory (CET)* predicts a dissipative component in
photon propagation, leading to a characteristic *two-regime behavior* in the
corrected residuals:
1. A quasi-linear decay at high initial excitation (high redshift);
2. A saturation regime as isotensive equilibrium is approached.

The LOS–Freedman method implemented here is designed to test this prediction
directly.

---

## Repository structure

jades-los-freedman/ 
├── scripts/    # Full data-processing and analysis pipeline
├── data/       # Final datasets used in figures and statistical tests 
├── figures/    # Figures generated from the final datasets

---

## Scripts (scripts/)

The analysis is fully reproducible and organized as a sequential pipeline:

1. 00_clean_GOODSS_catalog.py  
   Cleans and prepares the GOODS-S source catalogs.

2. 01_build_GOODSS_3D_catalog.py  
   Builds the 3D CANDELS GOODS-S tracer catalog (RA, DEC, z).

3. 02_compute_LOS_GOODSS_basic.py  
   Computes basic 2D LOS environment metrics.

4. 03_compute_LOS_GOODSS_3D.py  
   Computes 3D LOS counts using CANDELS tracers.

5. 04_compute_LOS_cone3D.py  
   Computes LOS metrics using a cone-based (CEERS-like) prescription.

6. 05_apply_LOS_to_JADES.py  
   Attaches LOS information to individual JADES sources.

7. 06_compute_JADES_Freedman_residuals.py  
   Computes LOS–Freedman corrected residuals.

8. 07_add_distance_proxy.py  
   Adds a distance proxy and performs distance normalization tests.

Scripts are intended to be run in numerical order.

---

## Data (data/)

This folder contains the *final datasets* used in the analysis and figures:

### CSV tables
- jades_dissipation_curve_binned.csv  
- jades_dissipation_curve_distnorm.csv  
- jades_dissipation_curve_BAO.csv  
- jades_dissipation_curve_BAO_binned.csv  
- jades_residual_models_fit.csv  
- z_over_dproxy_by_bin.csv

### FITS catalogs
- candels_GOODSS_3D.fits  
- jades_GOODSS_LOS3D_freedman_BAO.fits  
- jades_GOODSS_LOS3D_freedman_distnorm.fits

These files are sufficient to reproduce all plots and statistical results shown
in the associated analysis.

---

## Figures (figures/)

Key figures generated from the final datasets include:

- *jades_cet_dissipation_curve.png*  
  Main 2×1 figure showing:
  - Upper panel: ⟨Δμ⟩ vs. redshift with CET-like saturating fit  
  - Lower panel: residuals with respect to the CET-like model

- Additional diagnostic and comparison figures:
  - Dissipation curves
  - Residual distributions
  - Model-independent redshift scaling tests

---

## Reproducibility

All results presented in the analysis can be reproduced using:
- Python ≥ 3.9  
- NumPy  
- Astropy  
- Matplotlib  
- Pandas  

No proprietary software or non-public datasets are required beyond publicly
available JADES and CANDELS catalogs.

---

## Citation and data preservation

This repository is intended to be archived on *Zenodo*, where a DOI will be
assigned.  
Please cite the Zenodo record when using these scripts or datasets.

---

## Notes

This repository is designed for *scientific transparency*, not minimal size.
Intermediate steps are kept explicit in the pipeline to allow independent
verification of each stage of the LOS–Freedman procedure.

---

## Contact

For questions, discussions, or independent validation attempts, please open an
issue or contact the author via the associated CET project page.
seriacopileonardo@gmail.com