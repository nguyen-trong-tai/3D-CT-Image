import numpy as np
import scipy.ndimage as ndi

def mask_to_sdf(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Convert binary mask to Signed Distance Field.
    """
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(1 - mask)
    sdf = outside - inside

    return sdf * spacing[0]  # physical units
