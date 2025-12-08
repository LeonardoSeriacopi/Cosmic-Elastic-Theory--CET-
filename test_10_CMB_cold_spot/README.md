README — CMB Cold Spot Analysis (Test 10)
=========================================

This directory contains the full analysis pipeline applied to the CMB Cold Spot
(CS1) and its internal secondary minimum (CS2).  
All major tests are organized into high-level folders, each containing the
subdirectories:

    cold_spot_1/   → analysis centered on the primary Cold Spot
    cold_spot_2/   → analysis centered on the secondary minimum

Below is a summary of the main scientific procedures and scripts, without listing
the full directory tree.

----------------------------------------------------------------------
1. Data/
----------------------------------------------------------------------

Contains all catalogs and map inputs used throughout the tests.

• 2MASS (VizieR cutouts)  
    - Tab-separated catalogs around CS1 and CS2  
    - Used for galaxy-density statistics, radial profiles, and 2D density maps

• WMAP + Planck (thermodynamic temperature maps)  
    - HEALPix-format maps  
    - Used for multi-frequency aperture photometry at both Cold Spot centres

• SDSS / DES (optional tests)  
    - Additional catalogs tested for cross-validation  
    - Not used in the final pipeline

----------------------------------------------------------------------
2. Scripts/
----------------------------------------------------------------------

Python scripts used to execute all scientific analyses.

• analyze_2mass_coldspot.py  
    - Loads 2MASS catalog  
    - Computes angular separation to CS1/CS2  
    - Computes:
        • galaxy counts in disk (R ≤ 5°)  
        • galaxy counts in ring (5° < R ≤ 10°)  
        • angular densities  
        • magnitude histograms  
    - Outputs:
        • disk / ring filtered catalogs  
        • histogram tables

• map_density_2mass.py  
    - Builds 2D density maps around CS1 and CS2  
    - Produces:
        • absolute density maps  
        • residual maps (disk minus ring average)  
        • contour-enhanced versions  
    - Used for the main figure included in the manuscript

• cmb_temperature_multifreq.py  
    - Loads WMAP and Planck LFI maps  
    - Computes mean thermodynamic temperature in apertures of 1°, 2°, and 5°  
    - Compares CS1 vs CS2  
    - Tests frequency-independence to rule out foreground contamination

----------------------------------------------------------------------
3. Results/
----------------------------------------------------------------------

Stores all numeric outputs from the analyses:

• disk/ring catalogs  
• density values and radial profiles  
• magnitude histograms  
• multi-frequency CMB temperature tables  
• residual density statistics

These files are directly referenced in the manuscript.

----------------------------------------------------------------------
4. Figures/
----------------------------------------------------------------------

Contains all figures used in the text, including:

• Galaxy-density radial profiles  
• 2D density maps (CS1, CS2)  
• Residual maps with contour overlays  
• Multi-frequency CMB aperture-temperature plots  
• Final polished figure used in the CMB–LSS interpretation section

----------------------------------------------------------------------
5. Summary of Main Scientific Tests
----------------------------------------------------------------------

1. Galaxy-density test (2MASS)  
    - Measured density contrast between disk and ring  
    - Revealed:
        • positive radial density gradient around CS1  
        • dense compact core + outer depletion ring around CS2  
    - Morphology incompatible with simple ΛCDM isotropic expectations

2. 2D density residual maps  
    - Show explicit asymmetries and shell-like void structure around CS2  
    - Confirm internal structure within the Cold Spot region

3. Multi-frequency CMB test (WMAP + Planck)  
    - Both CS1 and CS2 are consistently cold in all frequency bands  
    - CS2 is colder in small apertures  
    - No frequency dependence → foreground explanation ruled out

4. Joint CMB + 2MASS interpretation  
    - The Cold Spot is not a single supervoid  
    - Instead, the region contains:
        • a large-scale gradient  
        • an internal secondary minimum  
    - The structure aligns with CET expectations of anisotropic dissipation,
      contrasting with ΛCDM Gaussian isotropy.

----------------------------------------------------------------------

End of README.