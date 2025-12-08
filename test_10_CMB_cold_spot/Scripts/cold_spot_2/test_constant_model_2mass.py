import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# -------------------------
# CONFIGURAÇÕES
# -------------------------
infile = "2mass.tsv"

# Centro do SEGUNDO Cold Spot (galáctico)
# antes: l_cs = 209.0, b_cs = -57.0
l_cs = 208.4349 * u.deg
b_cs = -55.8545 * u.deg
coord_cs = SkyCoord(l=l_cs, b=b_cs, frame="galactic")

# (opcional) imprimir o centro em ICRS só pra conferência
coord_icrs = coord_cs.icrs
print("=========================================================")
print("[INFO] Centro do SEGUNDO Cold Spot:")
print(f"  Galáctico: l = {l_cs.value:.4f} deg, b = {b_cs.value:.4f} deg")
print(f"  ICRS/J2000: RA = {coord_icrs.ra.deg:.4f} deg, Dec = {coord_icrs.dec.deg:.4f} deg")
print("=========================================================")

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

    # Distância angular ao Cold Spot 2
    theta = coords.separation(coord_cs).deg
    df["theta"] = theta

    print("[INFO] θ min/max:", f"{theta.min():.2f} – {theta.max():.2f} deg")
    print("=========================================================")

    # -------------------------
    # PERFIL RADIAL OBSERVADO
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

        dens = N / area if area > 0 else np.nan
        sigma = np.sqrt(N) / area if N > 0 else 0.0

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
    weights = 1.0 / (errors**2)
    rho0 = np.sum(weights * densities) / np.sum(weights)

    print(f"[RESULT] Densidade constante ajustada = {rho0:.3f} fontes/deg²")

    # χ² do ajuste
    chi2_val = np.sum(((densities - rho0) / errors)**2)
    dof = len(densities) - 1  # 1 parâmetro ajustado
    chi2_red = chi2_val / dof

    # p-value = probabilidade de obter chi² > observado
    p_value = 1.0 - chi2.cdf(chi2_val, dof)

    print("=========================================================")
    print("[RESULTADOS ESTATÍSTICOS]")
    print(f" χ² total           = {chi2_val:.3f}")
    print(f" χ² reduzido        = {chi2_red:.3f}")
    print(f" Graus de liberdade = {dof}")
    print(f" p-value            = {p_value:.5f}")
    print("=========================================================")

    # Salvar resultados (se quiser separar, pode trocar o nome do arquivo)
    with open("constant_model_test_results_cs2.txt", "w", encoding="utf-8") as f:
        f.write("Teste do modelo de densidade constante (SEGUNDO Cold Spot)\n")
        f.write("===========================================================\n")
        f.write(f"Densidade constante ajustada: {rho0:.3f}\n")
        f.write(f"chi2 total: {chi2_val:.3f}\n")
        f.write(f"chi2 reduzido: {chi2_red:.3f}\n")
        f.write(f"Graus de liberdade: {dof}\n")
        f.write(f"p-value: {p_value:.6f}\n\n")

        f.write("PERFIL RADIAL USADO:\n")
        for i in range(len(counts)):
            f.write(
                f"Bin {radial_bins[i]}–{radial_bins[i+1]} deg: "
                f"dens={densities[i]:.3f}, sigma={errors[i]:.3f}, N={counts[i]}\n"
            )

    print("[OK] Resultados salvos em constant_model_test_results_cs2.txt")


# -------------------------
# EXECUTAR
# -------------------------
if __name__ == "__main__":
    main()