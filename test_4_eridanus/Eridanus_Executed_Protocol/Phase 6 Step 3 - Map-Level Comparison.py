# Smooth both maps with 10 arcmin Gaussian beam
κ_des_smoothed = hp.smoothing(kappa_des, fwhm=np.radians(10/60)) 
κ_CET_smoothed = hp.smoothing(hp.ud_grade(κ_pred, nside_out=2048), fwhm=...)

# Compute cross-correlation
cl = hp.anafast(κ_des_smoothed * mask, κ_CET_smoothed * mask)