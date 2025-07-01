import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

r = np.linspace(10, 180, 200)  # [Mpc]

# Dados do operador D' (carregue do script anterior ou use)
D_prime_rho = 8e-33 * (1 - np.exp(-0.01*(r-70)**2))  # Perfil ajustado

# Integral ∫ D'ρ dr
integral = cumtrapz(D_prime_rho, r, initial=0)

plt.figure(figsize=(10,6))
plt.plot(r, integral*1e3, 'm-', lw=3, label='Energia Conforme')
plt.axvline(70, color='k', ls=':', label='70 Mpc')
plt.axvline(110, color='b', ls='-.', label='Borda Observada')
plt.xlabel('Raio [Mpc]', fontsize=12)
plt.ylabel('$\\int D\'\\rho  dr$ [$10^{-3}$ kg m$^{-2}$ yr$^{-1}$]', fontsize=12)
plt.title('Acúmulo de Energia Conforme no Eridanus', fontsize=14)
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('conformal_integral.png', dpi=300, bbox_inches='tight')