# Carregar dados do Eridanus
eridanus_data = load_eridanus_data()
coords = calc.comoving_distance(
    eridanus_data['ra'], 
    eridanus_data['dec'], 
    eridanus_data['z']
)

# Calcular densidades
densities = []
for i in range(len(coords)):
    indices = get_local_neighbors(coords, i, radius=15)  # 15 Mpc
    densities.append(calc.void_density(coords[indices])[0])

# Aplicar CET
z_cet = cet_correction(eridanus_data['z'], densities)