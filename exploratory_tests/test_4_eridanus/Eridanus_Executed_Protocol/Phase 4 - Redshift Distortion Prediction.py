Δz_pred = 0.042 * R  # Calibrated from Pantheon+
plt.plot(principal_axis, Δz_obs, label="DES-Y3")
plt.axhline(Δz_pred, c='r', ls='--', label=f"CET Prediction: {Δz_pred:.4f}")