import modal
import numpy as np

stub = modal.Function.lookup(
    "lung-tumor-3d-demo",
    "run_segmentation_pipeline"
)

def run_pipeline_remote(volume, spacing, bbox):
    result = stub.remote(volume, spacing, bbox)
    return result