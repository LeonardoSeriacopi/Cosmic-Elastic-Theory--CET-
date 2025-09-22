# 2_void_boundary_topology.py
# Baseado em: tda_void_eridanus_script_debug.py

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Scaler
from sklearn.preprocessing import MinMaxScaler

# Carrega dados normalizados (assumindo que coords_norm já foi calculado)
coords = pd.read_csv("2_processed_eridanus_comoving.csv")
coords_norm = MinMaxScaler().fit_transform(coords[['x', 'y', 'z']])

# Executa homologia persistente
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
diagrams = VR.fit_transform([coords_norm])
diagram = diagrams[0]

# Filtra H1 e H2 com limiar de persistência
h1 = diagram[diagram[:, 0] == 1]  # Ciclos
h2 = diagram[diagram[:, 0] == 2]  # Cavidades

# Aplica filtro de durabilidade em H1
persistence = h1[:, 2] - h1[:, 1]
min_persistence = 0.05 * np.max(persistence)  # 5% do valor máximo
h1_filtered = h1[persistence > min_persistence]

# Junta os componentes filtrados para a fronteira do void
void_boundary = np.concatenate((h1_filtered, h2))

# Exporta se desejar
np.savetxt("void_boundary_filtered.csv", void_boundary, delimiter=",", header="dim,birth,death", comments='')

print("Void boundary extracted and filtered.")
