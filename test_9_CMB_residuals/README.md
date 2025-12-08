# 📄 README — CET CMB Analysis Repository

This repository contains the scripts, processed outputs, and figures used in the Cosmic Elastic Theory (CET) analysis of CMB spectral structure, frequency–dependent residuals, and directional dissipation signatures.

Raw CMB datasets (Planck, WMAP, FIRAS, SPT) are **not included** due to file size and licensing restrictions.  
Only processed data products (plots, residual tables, fitted parameters) are provided in:

```
/scripts
/figures
/results
```

---

# 🔗 Raw Data Sources (Official Archives)

To reproduce the results, download the original maps from their official public archives.

## **Planck 2018 — Full Mission Maps**
Planck Legacy Archive (PLA):  
https://pla.esac.esa.int

Recommended files:
- `COM_CMB_IQU-smica_2048_R3.00_full.fits`
- LFI frequency maps (30, 44, 70 GHz)
- HFI frequency maps (100–857 GHz)

---

## **WMAP 9-Year Sky Maps**
NASA LAMBDA Archive:  
https://lambda.gsfc.nasa.gov/product/map/dr5/m_products.cfm

Bands used:
- K (23 GHz)
- Ka (33 GHz)
- Q (41 GHz)
- V (61 GHz)
- W (94 GHz)

---

## **COBE FIRAS — Absolute CMB Spectrum**
NASA LAMBDA — FIRAS Products:  
https://lambda.gsfc.nasa.gov/product/cobe/firas_products.html

Used for:
- Absolute temperature spectrum  
- Low-resolution spectral residuals (ΔI/I)

---

## **SPT — High-Frequency Spectral Residuals**
South Pole Telescope Public Data:  
https://pole.uchicago.edu/public/data/

Used for:
- Spectral residuals (90–150 GHz)
- High-ℓ small-scale spectrum comparison

---

# 📦 Repository Structure

```
/scripts
    CET_CMB_residual_fit.py
    CET_CMB_directional_fit.py
    CET_CMB_WMAP_offsky.py
    CET_CMB_WMAP_v5.7_balance_5bands

/results
    cmb_residuals.csv
    cmb_residual_spectrum_dataset.txt
    planck_cet_spectral_fit_results.txt
    CET_directional_gammas_v2.1.txt

/figures
    CET_CMB_residual_fit.png
    CET_CMB_WMAP_v5.7_balance_5bands.png
    CET_CMB_Combined_v7.1_resampled.png
    CET_CMB_v4.5.png
    
```

---

# 🧪 Reproducibility Instructions

After downloading raw maps, place them in:

```
/data/raw/
```

Expected structure:

```
data/raw/WMAP/*.fits
data/raw/Planck/*.fits
data/raw/FIRAS/*.fits
data/raw/SPT/*.fits
```

Paths can be modified in each script.

---

# 📘 Citation

If you use this repository, please cite:

**L. S. Seriacopi**  
*Cosmic Elastic Theory: Cosmic Acceleration and General Relativity as Local Phase-Transition Phenomena*  
Research Square (2025).

**L. S. Seriacopi**  
*Cosmic Elastic Theory II: Tension Decay Duality and the Singular Mode*  
Research Square (2025).

---

# 🛠 Additional Support

If needed, I can also generate:

- `environment.yml` or `requirements.txt`
- GitHub badges (DOI, license, Python version)
- A clean `/docs` folder for GitHub Pages
- A Zenodo deposition metadata template
- A .gitignore tailored for FITS + numpy + results

Just say the word!