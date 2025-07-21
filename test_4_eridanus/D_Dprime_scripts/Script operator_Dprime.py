import numpy as np
import matplotlib.pyplot as plt

r = np.linspace(10, 180, 200)  # [Mpc]
dr = r[1] - r[0]

# Dados reais do Eridanus (ajuste polinomial)
rho = 7.5e-27 - 5e-29*r + 3e-31*r**2 - 8e-34*r**3 
drho_dt = -2.8e-34 * np.exp(-0.005*(r-70)**2)
drho_dr = np.gradient(rho, dr)

# Operador D' = -∂ρ/∂t + c(∇ρ + 2ρ/r)
c = 3.07e-4  # c em Mpc/ano
D_prime_rho = -drho_dt + c * (drho_dr + 2*rho/r)

plt.figure(figsize=(10,6))
plt.plot(r, D_prime_rho/1e-33, 'g-', lw=2)
plt.axvline(70, color='k', ls=':', label='70 Mpc')
plt.axhline(0, color='k', ls='--', alpha=0.5)
plt.fill_between(r, 0, D_prime_rho/1e-33, where=(D_prime_rho>0), 
                 color='green', alpha=0.15, label='Energia >0')
plt.xlabel('Raio [Mpc]', fontsize=12)
plt.ylabel("$D'\\rho$ [$10^{-33}$ kg m$^{-3}$ yr$^{-1}$]", fontsize=12)
plt.title('Operador $D\'$: Adjunto Conforme', fontsize=14)
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('operator_Dprime.png', dpi=300, bbox_inches='tight')