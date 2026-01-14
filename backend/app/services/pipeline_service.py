from app.pipeline.input.preprocess import window_hu, resample
from app.pipeline.roi.roi_cropper import crop_roi
from app.pipeline.segmentation.inference import run_inference
from app.pipeline.segmentation.postprocess import largest_component
from app.pipeline.implicit.sdf import mask_to_sdf
from app.pipeline.surface.marching_cubes import extract_mesh

def run_roi_pipeline(volume, spacing, bbox, model):
    """
    Full pipeline: ROI → Seg → SDF → Mesh
    """
    roi = crop_roi(volume, bbox)
    roi = window_hu(roi)
    prob = run_inference(model, roi)
    mask = largest_component(prob > 0.5)
    sdf = mask_to_sdf(mask, spacing)
    verts, faces, normals = extract_mesh(sdf)
    return verts, faces, normals
