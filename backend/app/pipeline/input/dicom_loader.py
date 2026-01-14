import pydicom
import numpy as np
from typing import Tuple, Dict

def load_dicom_series(dicom_files: list) -> Tuple[np.ndarray, Dict]:
    """
    Load a DICOM series into a 3D volume.

    Returns
    -------
    volume : np.ndarray (Z, Y, X)
    metadata : dict {spacing, origin, orientation}
    """
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)

    slope = float(slices[0].RescaleSlope)
    intercept = float(slices[0].RescaleIntercept)
    volume = volume * slope + intercept

    spacing = (
        float(slices[0].SliceThickness),
        float(slices[0].PixelSpacing[0]),
        float(slices[0].PixelSpacing[1]),
    )

    return volume, {"spacing": spacing}
