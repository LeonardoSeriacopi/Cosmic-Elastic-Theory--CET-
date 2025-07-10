# Load DES Y6 shear catalog (Eridanus field)
import healpy as hp
kappa_des = hp.read_map("DESY6_kappa_eridanus.fits")  
mask = (hp.read_map("DESY6_mask.fits") > 0.8)  # 80% completeness cut