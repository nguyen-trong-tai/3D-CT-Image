import numpy as np
from typing import Tuple

def crop_roi(
    volume: np.ndarray,
    bbox: Tuple[int, int, int, int, int, int]
) -> np.ndarray:
    """
    Crop ROI from volume.

    bbox = (z, y, x, depth, height, width)
    """
    z, y, x, d, h, w = bbox
    return volume[z:z+d, y:y+h, x:x+w]
