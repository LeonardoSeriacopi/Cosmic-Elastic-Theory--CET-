import numpy as np
from astropy.cosmology import Planck18
from scipy.spatial import cKDTree, Voronoi
from scipy.interpolate import LinearNDInterpolator

class CosmicDensityCalculator:
    def __init__(self, cosmology=Planck18):
        self.cosmo = cosmology
        self.rho_crit0 = 3 * self.cosmo.H0.value**2 / (8 * np.pi * 4.3009e-9)  # g/cm³
    
    def critical_density(self, z):
        """Critical density at redshift z"""
        E_z = np.sqrt(self.cosmo.Om0 * (1+z)**3 + self.cosmo.Ode0)
        return self.rho_crit0 * E_z**2
    
    def comoving_distance(self, ra, dec, z):
        """Convert to comoving coordinates"""
        dist = self.cosmo.comoving_distance(z).value  # Mpc
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        x = dist * np.cos(dec_rad) * np.cos(ra_rad)
        y = dist * np.cos(dec_rad) * np.sin(ra_rad)
        z_coord = dist * np.sin(dec_rad)
        
        return np.vstack([x, y, z_coord]).T
    
    def isolated_density(self, coords, n_neighbors=7):
        """Density for isolated objects"""
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=n_neighbors+1)
        
        # Adaptive radius (distance to nth neighbor)
        r_n = dists[:, n_neighbors]
        volumes = (4/3) * np.pi * r_n**3
        return n_neighbors / volumes
    
    def cluster_density(self, coords, radius=2.0):
        """Fixed-radius density for clusters"""
        tree = cKDTree(coords)
        counts = tree.query_ball_point(coords, r=radius, return_length=True)
        volume = (4/3) * np.pi * radius**3
        return np.array(counts) / volume
    
    def void_density(self, coords, grid_resolution=10):
        """Density field for voids using Voronoi tessellation"""
        vor = Voronoi(coords)
        
        # Calculate Voronoi cell densities
        cell_densities = 1 / np.array([self._cell_volume(vor, i) for i in range(len(coords))])
        
        # Create interpolation function
        interp = LinearNDInterpolator(coords, cell_densities)
        
        # Generate grid
        x_min, x_max = coords[:,0].min(), coords[:,0].max()
        y_min, y_max = coords[:,1].min(), coords[:,1].max()
        z_min, z_max = coords[:,2].min(), coords[:,2].max()
        
        grid_x, grid_y, grid_z = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
            np.linspace(z_min, z_max, grid_resolution)
        )
        
        # Interpolate densities
        grid_density = interp(grid_x, grid_y, grid_z)
        return grid_x, grid_y, grid_z, grid_density
    
    def _cell_volume(self, vor, index):
        """Calculate volume of a Voronoi cell"""
        from scipy.spatial import ConvexHull
        vertices = vor.vertices[vor.regions[vor.point_region[index]]]
        hull = ConvexHull(vertices)
        return hull.volume

# Example usage
if __name__ == "__main__":
    # Mock data: isolated galaxies
    rng = np.random.default_rng(42)
    ra = rng.uniform(0, 360, 100)
    dec = rng.uniform(-90, 90, 100)
    z = rng.uniform(0.01, 0.05, 100)
    
    calc = CosmicDensityCalculator()
    coords = calc.comoving_distance(ra, dec, z)
    
    # Environment-aware density calculation
    densities = []
    for i in range(len(coords)):
        # Create neighborhood subset
        tree = cKDTree(coords)
        indices = tree.query_ball_point(coords[i], r=10)  # 10 Mpc radius
        
        if len(indices) < 20:  # Isolated environment
            density = calc.isolated_density(coords[indices])[0]
        elif len(indices) > 100:  # Cluster environment
            density = calc.cluster_density(coords[indices])[0]
        else:  # Field environment
            density = np.median(calc.isolated_density(coords[indices]))
        
        densities.append(density)
    
    print(f"Calculated densities: {densities[:5]}...")