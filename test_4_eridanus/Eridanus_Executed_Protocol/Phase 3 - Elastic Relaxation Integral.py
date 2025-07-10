P_crit = 1.6e-9 * c**4/G  # Critical pressure (1.6×10⁻⁹ Planck units)
ε = (ρ*c**2 - P_crit)/K  # Strain field (K=2.1×10²⁹ Pa)

R = mc_integrate(ε/P_crit, void_coords)  # Monte Carlo integration