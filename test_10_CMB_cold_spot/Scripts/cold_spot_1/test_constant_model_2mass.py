import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# -------------------------
# CONFIGURAÇÕES
# -------------------------
infile = "2mass.tsv"

# Centro do Cold Spot (galáctico)
l_cs = 209.0 * u.deg
b_cs = -57.0 * u.deg
coord_cs = SkyCoord(l=l_cs, b=b_cs, frame="galactic")

# Bins radiais (mesmos usados antes)
radial_bins = np.array([0, 2, 4, 6, 8, 10], dtype=float)

# -------------------------
# FUNÇÃO PARA CÁLCULO DE ÁREA
# -------------------------
def area_annulus(r_in, r_out):
    return np.pi * (r_out**2 - r_in**2)

# -------------------------
# PARTE PRINCIPAL
# -------------------------
def main():
    print("=========================================================")
    print("[INFO] Lendo catálogo 2MASS recortado:", infile)

    df = pd.read_csv(infile, sep="\t", comment="#", low_memory=False)

    # Garantir conversão numérica
    df["RAJ2000"] = pd.to_numeric(df["RAJ2000"], errors="coerce")
    df["DEJ2000"] = pd.to_numeric(df["DEJ2000"], errors="coerce")
    df = df.dropna(subset=["RAJ2000", "DEJ2000"])

    print("[INFO] Linhas com RA/Dec válidos:", len(df))

    # Coordenadas 2MASS → SkyCoord
    coords = SkyCoord(ra=df["RAJ2000"].values * u.deg,
                      dec=df["DEJ2000"].values * u.deg,
                      frame="icrs")

    # Distância angular ao Cold Spot
    theta = coords.separation(coord_cs).deg
    df["theta"] = theta

    print("[INFO] θ min/max:", f"{theta.min():.2f} – {theta.max():.2f} deg")
    print("=========================================================")

    # -------------------------
    # PERFIL RADIAL CONHECIDO
    # -------------------------
    counts = []
    areas = []
    densities = []
    errors = []

    print("[INFO] Montando perfil radial...")

    for i in range(len(radial_bins) - 1):
        r1 = radial_bins[i]
        r2 = radial_bins[i + 1]

        mask = (theta >= r1) & (theta < r2)
        N = mask.sum()
        area = area_annulus(r1, r2)

        dens = N / area
        sigma = np.sqrt(N) / area if N > 0 else 0

        counts.append(N)
        areas.append(area)
        densities.append(dens)
        errors.append(sigma)

        print(f" Bin {r1:3.1f}–{r2:3.1f} deg | N={N:6d} | dens={dens:8.2f} | σ={sigma:6.2f}")

    counts = np.array(counts)
    densities = np.array(densities)
    errors = np.array(errors)

    print("=========================================================")

    # -------------------------
    # AJUSTE DO MODELO CONSTANTE
    # -------------------------
    print("[INFO] Ajustando modelo de densidade constante...")

    # Melhor estimativa da densidade constante (weighted mean)
    weights = 1 / errors**2
    rho0 = np.sum(weights * densities) / np.sum(weights)

    print(f"[RESULT] Densidade constante ajustada = {rho0:.3f} fontes/deg²")

    # χ² do ajuste
    chi2_val = np.sum(((densities - rho0) / errors)**2)
    dof = len(densities) - 1  # 1 parâmetro ajustado
    chi2_red = chi2_val / dof

    # p-value = probabilidade de obter chi² > observado
    p_value = 1 - chi2.cdf(chi2_val, dof)

    print("=========================================================")
    print("[RESULTADOS ESTATÍSTICOS]")
    print(f" χ² total           = {chi2_val:.3f}")
    print(f" χ² reduzido        = {chi2_red:.3f}")
    print(f" Graus de liberdade = {dof}")
    print(f" p-value            = {p_value:.5f}")
    print("=========================================================")

    # Salvar resultados
    with open("constant_model_test_results.txt", "w") as f:
        f.write("Teste do modelo de densidade constante\n")
        f.write("=======================================\n")
        f.write(f"Densidade constante ajustada: {rho0:.3f}\n")
        f.write(f"chi2 total: {chi2_val:.3f}\n")
        f.write(f"chi2 reduzido: {chi2_red:.3f}\n")
        f.write(f"Graus de liberdade: {dof}\n")
        f.write(f"p-value: {p_value:.6f}\n\n")

        f.write("PERFIL RADIAL USADO:\n")
        for i in range(len(counts)):
            f.write(f"Bin {radial_bins[i]}–{radial_bins[i+1]} deg: dens={densities[i]:.3f}, σ={errors[i]:.3f}\n")

    print("[OK] Resultados salvos em constant_model_test_results.txt")

# -------------------------
# EXECUTAR
# -------------------------
if __name__ == "__main__":
    main()