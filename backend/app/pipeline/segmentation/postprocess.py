import numpy as np
import scipy.ndimage as ndi

def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labeled, range(1, n+1))
    return (labeled == (sizes.argmax() + 1)).astype(np.uint8)
