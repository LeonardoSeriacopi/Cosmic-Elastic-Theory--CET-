🌌 **Cosmic Elastic Theory** (CET)

The Cosmic Elastic Theory (CET) reinterprets cosmology through the lens of an elastic spacetime medium, where cosmic acceleration and emergent gravity arise from thermodynamic tension, deformation, and phase transitions — without invoking dark energy or dark matter.

📂 **Repository Structure**
CET/
│
├── Foundations/                # Theoretical groundwork
│   ├── Theoretical_Foundation/ 
│   └── Analysis/               
│
├── Tests/                      # Empirical validations
│   ├── test_1_pantheon/        
│   ├── test_2_ceers/           
│   ├── test_3_jades/           
│   ├── test_4_eridanus/        
│   ├── test_5_cluster_stability/
│   ├── test_6_quasars/         
│   ├── test_7_des_dr1/         
│   └── test_8_gravitational_waves/   # (new)
│        ├── data/              # Raw + processed datasets
│        ├── scripts/           # Analysis pipeline
│        ├── figures/           # Output plots
│        └── CET_GW_LOS.pdf     # Full report (linked to OSF)
│
├── LICENSE_CC_BY_4.0
├── LICENSE_MIT
└── README.md

🔬 **Scientific Scope**

Foundational Theory
Redshift as elastic deformation.
Critical pressure and dual regimes of spacetime.

Empirical Tests

Supernovae (Pantheon+)
Galaxy surveys (JADES, CEERS, DES)
Quasar residuals (DR16Q)
Cluster stability without dark matter
Elastic response in supervoids (Eridanus)
Weak lensing variance (DES Y3A2)

NEW: Line-of-sight density suppression in gravitational-wave ringdowns.

⚙️ **Workflow for Each Test**

Every test follows a transparent pipeline:
Data preparation → load raw catalogues, apply event/galaxy filters.
Scripts → analysis with regression models (OLS/Ridge, permutation tests).
Outputs → figures (PNG), tables (CSV/MD), and short summaries (JSON).
PDF Report → full write-up in LaTeX, archived on OSF.

**Example (Test 8 – GW LOS suppression):**

# Run multivariate regression and LOS contribution test
python scripts/finalize_los_analysis.py posterior_with_density.csv results_los6 \
       --events "GW150914_095045,GW151012_095443,GW170104_101158,..."

# Build tables from results
python scripts/make_tables.py results_los6

🌐 **Open Access**

All preprints and reports are archived at OSF:
📂 https://osf.io/vpq76

🤝 **Contributions**

This is an open research initiative exploring cosmology from first principles of tension, deformation, and thermodynamics.
Ideas, forks, critiques, and collaborations are welcome.