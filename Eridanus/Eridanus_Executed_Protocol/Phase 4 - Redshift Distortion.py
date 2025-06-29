# redshift_distortion.py (novo)
α = 0.042  # Constante calibrada em Pantheon+
Δz_pred = α * R_void

# Carregar dados observacionais
principal_axis = np.loadtxt("gradient_data.txt")  # Dados do arquivo Gradiente

plt.figure(figsize=(10,6))
plt.scatter(principal_axis[:,0], principal_axis[:,1], s=3, alpha=0.7, label="Dados Eridanus")
plt.axhline(y=Δz_pred, color='r', linestyle='--', 
            label=f"Predição CET: Δz = {Δz_pred:.4f}")
plt.xlabel("Posição no Eixo Principal (normalizada)")
plt.ylabel("Δz")
plt.legend()
plt.savefig("eridanus_redshift_distortion.png")