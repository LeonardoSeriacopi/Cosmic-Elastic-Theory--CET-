import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CET DISCRETE DISSIPATION – DEMO FIGURE
# ---------------------------------------------------------
# This script builds a toy model of the CET discrete process:
#   - A photon with initial energy E0 propagates along D
#   - At each step, the medium processes a small fraction of E
#   - The dissipative redshift z_diss is the sum of discrete
#     increments Δz_i = Γ_proc(E_i, K_i) * ΔD
#   - Coupling K(D) decays as the medium approaches isotension
#
# The goal is to illustrate:
#   * Discrete interactions
#   * Accumulated dissipative redshift
#   * Saturation toward a finite z_diss^max
# =========================================================


# -----------------------------
# 1) Model parameters (toy)
# -----------------------------
D_max = 3000.0   # maximum path length [Mpc] (toy scale)
N_steps = 800    # number of discrete interaction steps

# Photon & medium parameters (dimensionless toy values)
E0      = 1.0    # initial normalized photon energy
K0      = 1.0    # initial coupling (maximal dynamical response)

# Characteristic scales (how fast E and K decay)
D_E  = 1000.0    # energy-decay scale [Mpc]
D_K  = 150.0     # coupling-decay scale [Mpc] (transition zone)

# CET-like process parameters (toy form of Gamma_proc)
A       = 0.02   # overall normalization of process rate
beta    = 1.0    # power-law dependence on energy
alpha   = 0.5    # exponential saturation with energy
lambdaR = 0.5    # modulation strength via R(E, K)


# -----------------------------
# 2) Helper: CET process rate
# -----------------------------
def R_effective(E, K):
    """
    Simple dimensionless modulation function R(E, K).
    In CET language: encodes how dynamical coupling modulates
    the process rate. Here we use a bounded function that
    decreases as K -> 0 and grows mildly with E.
    """
    # Avoid division by zero; add a small floor
    K_eff = max(K, 1e-6)
    return (E / (1.0 + E)) * (K_eff / (1.0 + K_eff))


def gamma_proc(E, K):
    """
    Toy version of the CET process rate:
        Gamma_proc(E, K) = A * E^beta * exp(-alpha * E) * (1 + lambdaR * R(E, K))
    This is not meant to be numerically realistic, only to show:
        - growth with energy (E^beta)
        - saturation at high E (exp(-alpha E))
        - dependence on coupling K via R(E, K)
    """
    return A * (E ** beta) * np.exp(-alpha * E) * (1.0 + lambdaR * R_effective(E, K))


# -----------------------------
# 3) Discrete propagation
# -----------------------------
# Uniform steps in comoving distance
D = np.linspace(0.0, D_max, N_steps)
dD = D[1] - D[0]

# Arrays to store evolution
E_arr      = np.zeros(N_steps)
K_arr      = np.zeros(N_steps)
z_diss_arr = np.zeros(N_steps)
dz_events  = np.zeros(N_steps)  # individual increments Δz_i

# Initial conditions
E_arr[0] = E0
K_arr[0] = K0
z_diss_arr[0] = 0.0

for i in range(1, N_steps):
    # Approximate energy decay (toy: exponential with distance)
    E_arr[i] = E0 * np.exp(-D[i] / D_E)

    # Coupling decays faster: the medium approaches isotension
    K_arr[i] = K0 * np.exp(-D[i] / D_K)

    # Local process rate at this step
    gamma_i = gamma_proc(E_arr[i], K_arr[i])

    # Discrete increment of dissipative redshift
    dz_i = gamma_i * dD
    dz_events[i] = dz_i

    # Cumulative dissipative redshift
    z_diss_arr[i] = z_diss_arr[i - 1] + dz_i


# -----------------------------
# 4) Plot: discrete vs cumulative
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Cumulative dissipative redshift curve
ax.plot(D, z_diss_arr, label="Cumulative $z_{\\mathrm{diss}}(D)$")

# Discrete events: mark every Nth point so it doesn't clutter
step_mark = max(1, N_steps // 60)  # about ~60 markers max
ax.scatter(
    D[::step_mark],
    z_diss_arr[::step_mark],
    s=15,
    alpha=0.7,
    label="Discrete processing steps"
)

ax.set_xlabel("Comoving distance $D$ [Mpc]")
ax.set_ylabel("Dissipative redshift $z_{\\mathrm{diss}}$")
ax.set_title("CET discrete dissipation: accumulated $z_{\\mathrm{diss}}$ from discrete steps")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# -----------------------------
# 5) Optional: plot E(D) and K(D)
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(D, E_arr, label="Photon energy $E(D)$")
ax2.plot(D, K_arr, label="Coupling $K(D)$", linestyle="--")

ax2.set_xlabel("Comoving distance $D$ [Mpc]")
ax2.set_ylabel("Normalized value")
ax2.set_title("CET toy evolution of energy and coupling")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()