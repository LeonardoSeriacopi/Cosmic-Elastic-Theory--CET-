🌌 **Cosmic Elastic Theory** (CET)

The Cosmic Elastic Theory (CET) reinterprets cosmology through the lens of an elastic spacetime medium, where cosmic acceleration and emergent gravity arise from thermodynamic tension, deformation, and phase transitions — without invoking dark energy or dark matter.

📂 **Repository Structure**
CET/
│
├── Foundations/                # Theoretical groundwork
│   ├── Theoretical\_Foundation/
│   └── Analysis/  
│
├── Tests/                      # Empirical validations
│   ├── test\_1\_pantheon/        

│   ├── test\_2\_ceers/           

│   ├── test\_3\_jades/           

│   ├── test\_4\_eridanus/        

│   ├── test\_5\_cluster\_stability/

│   ├── test\_6\_quasars/         

│   ├── test\_7\_des\_dr1/         

│   ├── test\_8\_gravitational\_waves/   # LOS suppression

│   └── test\_9\_ringdown\_suppression/  # (new)

│        ├── data/              # Raw + processed datasets

│        ├── scripts/           # Nonlinear + cap models

│        ├── figures/           # Output plots

│        └── Testing the Maximum Tension Hypothesis in Cosmic Elastic Theory.pdf       
├── LICENSE\_CC\_BY\_4.0
├── LICENSE\_MIT
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

python scripts/finalize\_los\_analysis.py posterior\_with\_density.csv results\_los6   
--events "GW150914\_095045,GW151012\_095443,GW170104\_101158,..."

# Build tables from results

python scripts/make\_tables.py results\_los6

🌐 **Open Access**

All preprints and reports are archived at OSF:
📂 https://osf.io/vpq76

🤝 **Contributions**

This is an open research initiative exploring cosmology from first principles of tension, deformation, and thermodynamics.
Ideas, forks, critiques, and collaborations are welcome.

