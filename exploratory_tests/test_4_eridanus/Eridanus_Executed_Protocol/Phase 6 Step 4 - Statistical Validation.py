from scipy.stats import linregress

# Pixel-by-pixel comparison (5σ clipping)
valid_pixels = mask & (np.abs(kappa_des) < 5*np.std(kappa_des))
slope, intercept, r_value, p_value, _ = linregress(
    κ_CET_smoothed[valid_pixels], 
    κ_des_smoothed[valid_pixels]