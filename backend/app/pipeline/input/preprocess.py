import numpy as np
import scipy.ndimage as ndi

def window_hu(volume: np.ndarray, wl=-600, ww=1600) -> np.ndarray:
    """Apply lung window."""
    low = wl - ww // 2
    high = wl + ww // 2
    volume = np.clip(volume, low, high)
    return (volume - low) / (high - low)

def resample(volume: np.ndarray, spacing, target=(1.0, 1.0, 1.0)):
    """Resample volume to isotropic spacing."""
    zoom = tuple(s / t for s, t in zip(spacing, target))
    return ndi.zoom(volume, zoom, order=1), target
