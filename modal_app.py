import modal
import torch
import numpy as np

app = modal.App("lung-tumor-gpu-pipeline")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch==2.1.2",
        "numpy",
        "scipy",
    )
)

@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
)
class GPUPipeline:

    def __enter__(self):
        # Load models ONCE
        self.seg_model = torch.load(
            "/model/seg_unet.pt",
            map_location="cuda"
        ).eval().cuda()

        self.sdf_model = torch.load(
            "/model/sdf_unet.pt",
            map_location="cuda"
        ).eval().cuda()

    @modal.method()
    def run(
        self,
        roi_volume: np.ndarray,
        spacing: tuple,
    ):
        """
        ROI → Segmentation → Neural SDF
        """
        # ---------- Segmentation ----------
        with torch.no_grad():
            x = torch.from_numpy(roi_volume)[None, None].cuda()
            seg_prob = torch.sigmoid(self.seg_model(x))
            mask = (seg_prob > 0.5).float()

        # ---------- Neural SDF ----------
        with torch.no_grad():
            sdf = self.sdf_model(mask)   # regression
            sdf = sdf[0, 0].cpu().numpy()

        return {
            "sdf": sdf,
            "spacing": spacing,
        }
