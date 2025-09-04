Scripts – Test 11 (SN 2023zkd)



This directory contains the scripts used for the processing and analysis of the Type Ibn supernova SN 2023zkd.

The workflow spans from the acquisition of raw photometry to the fitting of dissipative relaxation models.



📥 1. Data acquisition



fetch\_phot\_sn2023zkd.py

Queries the ZTF (IRSA) public archive and downloads raw photometry (g/r/i).

Applies basic quality filters (catflags == 0, mag\_err < 1) and produces a CSV file with columns:

mjd, band, mag, mag\_err.





🔄 2. Pre-processing and merging



merge\_photometry\_sn2023zkd.py

Merges photometry from TNS, ATLAS, and ZTF into a single chronological dataset.

Serves as the baseline catalog for subsequent light-curve analyses.



make\_raw\_vs\_clean\_panels.py

Compares raw versus cleaned/binned photometry.

Applies σ-clipping and time-binning (typically Δt ≈ 3 days), and generates side-by-side figures.





🌈 3. Color and bolometric reconstruction



make\_colors\_gr.py

Computes the temporal evolution of colors (e.g., g–r) from the cleaned dataset.



make\_bolometric\_from\_clip.py

Integrates multi-band photometry to reconstruct the bolometric curve.

Outputs temperature, photospheric radius, and bolometric luminosity.



merge\_color\_with\_bol.py

Merges the color time series with bolometric luminosity.

Produces combined CSVs and optional correlation plots.





⚡ 4. Energy and mass-loss rates



compute\_energy\_mdot.py

Integrates bolometric luminosity over time to estimate radiated energy.

Also computes mass-loss rates (ṁ) given shock velocity, wind velocity, and efficiency parameters.





🌀 5. Relaxation modeling



fit\_relaxation\_model.py

Fits a storage + dissipative relaxation model to the bolometric curve.

Returns parameters such as characteristic relaxation timescale (τ), immediate vs. delayed energy fractions, and fit quality metrics.

