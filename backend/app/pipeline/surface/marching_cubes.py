import numpy as np
from skimage.measure import marching_cubes

def extract_mesh(sdf: np.ndarray, level=0.0):
    """
    Extract mesh from SDF using Marching Cubes.
    """
    verts, faces, normals, _ = marching_cubes(sdf, level=level)
    return verts, faces, normals
