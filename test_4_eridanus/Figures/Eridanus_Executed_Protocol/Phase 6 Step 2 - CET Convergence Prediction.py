# Calculate κ_CET from elastic operators
def kappa_CET(D, D_prime, ρ):
    return np.mean(D * D_prime) / np.mean(ρ**2)  # Eq. 6 in paper
    
κ_pred = kappa_CET(D_operator(ρ), D_prime_operator(ρ), ρ)