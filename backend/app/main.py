from fastapi import FastAPI
from app.api import upload, volume, segmentation, reconstruction, mesh

app = FastAPI(
    title="Lung Tumor 3D Reconstruction Demo",
    description="Academic research demo: CT → Segmentation → Implicit → Mesh",
    version="1.0"
)

app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(volume.router, prefix="/volume", tags=["Volume"])
app.include_router(segmentation.router, prefix="/segment", tags=["Segmentation"])
app.include_router(reconstruction.router, prefix="/reconstruct", tags=["Reconstruction"])
app.include_router(mesh.router, prefix="/mesh", tags=["Mesh"])
