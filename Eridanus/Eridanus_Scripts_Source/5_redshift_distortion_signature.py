from scipy.optimize import curve_fit

# Suponha que R_void_bins e Δz_observed estão definidos
def alpha_model(r, alpha):
    return alpha * r

alpha_opt, _ = curve_fit(alpha_model, R_void_bins, Δz_observed)
alpha = alpha_opt[0]
Δz_predito = alpha * R_void

# Visualização
principal_axis = df['x_c'] / np.max(np.abs(df['x_c']))
plt.scatter(principal_axis, df['z'], s=5, label="Dados")
plt.axhline(y=Δz_predito, color='r', linestyle='--',
            label=f"Predição CET: Δz = {Δz_predito:.4f}")
plt.legend()
plt.xlabel("Eixo principal normalizado")
plt.ylabel("Redshift")
plt.title("Assinatura observável da distorção de redshift")
plt.show()