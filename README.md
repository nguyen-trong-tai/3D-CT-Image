# Lung Tumor 3D Reconstruction (Research Demo)

This repository contains an **academic research and demonstration system** that visualizes an end-to-end pipeline from **lung CT images** to **3D tumor surface models**.

The project is designed for:
- Thesis / capstone project
- Medical imaging and AI research demonstration
- Educational visualization of 3D reconstruction pipelines

This project is **not** intended for clinical diagnosis or medical use.

---

## Pipeline Overview

The system follows the pipeline below:
Intput: CT Volume
→ ROI Selection
→ Tumor Segmentation (Deep Learning)
→ Implicit Surface Modeling (SDF / Neural SDF)
→ Marching Cubes
→ 3D Mesh Visualization

Key ideas:
- Explicit separation between **voxel-based** and **implicit** representations
- Use of **implicit surfaces (SDF)** to improve surface continuity
- Modular design for research analysis and comparison

---
---

## Dataset

The pipeline is designed for lung CT datasets such as:
- **LIDC-IDRI**
- **LUNA16** (optional)

Medical datasets are **not included** in this repository due to size and privacy constraints.

---

## GPU Acceleration (Modal)

GPU-based inference is handled using **Modal**:
- Segmentation and neural SDF run on GPU
- Inference-only (no training on Modal)
- GPU resources are allocated per invocation

This approach is well-suited for research demos and limited compute budgets.

---

## Installation (Local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Modal Setup
``` bash
pip install modal
modal setup
modal deploy modal_app.py
```
## Running the Backend
``` bash
cd backend
uvicorn app.main:app --reload
```
 or 
``` bash
python main.py
``` 
## Scope & Limitations
- Research and visualization only
# Author
...
