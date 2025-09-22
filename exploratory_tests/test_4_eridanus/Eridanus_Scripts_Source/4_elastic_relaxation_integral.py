import numpy as np
import matplotlib.pyplot as plt

def elastic_relaxation(P_grid, P_crit=1.6e-9, K=2.1, R_void=150):
    """
    Computes the elastic relaxation ε(P) across a pressure grid.

    Parameters:
    - P_grid (array): Array of pressure values [Pa]
    - P_crit (float): Critical pressure threshold for causal decoupling [Pa] (default = 1.6e-9)
    - K (float): Relaxation stiffness parameter (default = 2.1)
    - R_void (float): Void radius used for normalization [Mpc] (default = 150)

    Returns:
    - ε_grid (array): Elastic relaxation profile
    """
    # Elastic relaxation formula from CET
    ε_grid = 1 - np.exp(-K * (P_crit / P_grid)**0.5) * np.exp(-P_grid / P_crit)

    return ε_grid

# Example usage:
if __name__ == "__main__":
    P_vals = np.logspace(-12, -8, 500)
    ε_vals = elastic_relaxation(P_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(P_vals, ε_vals)
    plt.xscale('log')
    plt.xlabel("Local Pressure $P$ [Pa]")
    plt.ylabel("Elastic Relaxation $\\varepsilon(P)$")
    plt.title("Elastic Relaxation Curve (CET)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("elastic_relaxation_curve.png", dpi=300)
    plt.show()