# Parâmetros medidos do Eridanus
ν_eridanus = 0.327
Δz_eridanus = -0.0180

# Fórmula CET para convergência (derivada da razão de Poisson)
κ_CET = -0.7 * ν_eridanus * (Δz_eridanus / 0.02)  # Normalizado por z_ref=0.02
print(f"Predição CET: κ = {κ_CET:.4f}")
