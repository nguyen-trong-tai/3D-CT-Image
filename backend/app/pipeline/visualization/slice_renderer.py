import numpy as np

def overlay_mask(slice_img: np.ndarray, mask: np.ndarray, alpha=0.4):
    """
    Overlay segmentation mask on CT slice.
    """
    overlay = slice_img.copy()
    overlay[mask > 0] = overlay.max()
    return (1 - alpha) * slice_img + alpha * overlay
