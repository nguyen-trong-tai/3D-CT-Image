# Product Requirements Document
## Lung Tumor 3D Reconstruction Research Platform

**Version:** 1.0  
**Date:** January 2026  
**Status:** Draft  
**Project Type:** Academic Research / Thesis Demonstration

---

## 1. Executive Summary

This document defines requirements for a web-based research demonstration system that transforms lung CT imaging data into interactive 3D tumor models. The platform visualizes an end-to-end AI pipeline consisting of ROI selection, segmentation, implicit surface modeling, and mesh reconstruction.

**Primary Purpose:** Academic demonstration and thesis defense visualization  
**Target Users:** AI researchers, medical imaging students, technical reviewers  
**Core Technology:** FastAPI backend with GPU-accelerated AI models, React frontend with Three.js rendering  
**Deployment:** Serverless GPU compute (e.g., Modal, Runpod, Replicate)

---

## 2. Product Vision & Objectives

### 2.1 Vision Statement

Create a transparent, modular research platform that demonstrates the complete pipeline from volumetric CT data to explicit 3D mesh reconstruction, enabling researchers to validate each processing stage independently.

### 2.2 Primary Objectives

1. **Educational Transparency**: Expose each pipeline stage with clear visualization
2. **Research Validation**: Enable verification of segmentation and reconstruction quality
3. **Modular Architecture**: Ensure pipeline robustness even when optional components fail
4. **Technical Demonstration**: Showcase state-of-the-art medical imaging AI techniques
5. **Accessibility**: Browser-based deployment requiring no specialized software

### 2.3 Success Metrics

- Successful end-to-end processing of standard lung CT datasets (LUNA16, LIDC-IDRI)
- Sub-15 second processing time for typical nodules (20-50mm diameter)
- Mesh reconstruction accuracy validated against ground truth segmentations
- Clear visual differentiation between pipeline stages in UI
---

## 3. Scope Definition

### 3.1 In Scope

**Core Pipeline:**
- Optional ROI selection via bounding box interface (user choice)
- Full volume segmentation option (skip ROI selection)
- Deep learning-based tumor segmentation (on ROI or full volume)
- Implicit surface modeling (SDF)
- Iso-surface extraction via Marching Cubes
- 2D slice visualization with segmentation overlay
- 3D mesh rendering in browser
- Mesh quality validation tools

**Data Handling:**
- DICOM series upload and parsing
- MetaImage (.mhd + .raw) format support
- Temporary processing storage (session-based)
- Mesh export (.glb, .obj formats)

**Technical Infrastructure:**
- FastAPI backend with async processing
- Serverless GPU inference
- React frontend with WebGL rendering
- RESTful API design

### 3.2 Out of Scope

- Clinical diagnostic capabilities
- Multi-organ segmentation
- PACS/HIS/EMR integration
- Persistent patient data storage
- Longitudinal tumor tracking
- Treatment planning features
- Regulatory compliance (HIPAA, FDA, CE marking)
- Real-time collaboration features
- User authentication (initial version)

### 3.3 Future Extensions (Non-Blocking)

- Automatic nodule detection as ROI proposal
- Multi-nodule batch processing
- Neural implicit surface representations
- Uncertainty quantification
- Comparative visualization (multiple segmentation methods)
- Interactive mesh editing

---

## 4. User
<!-- ### 4.1 Primary Persona: Graduate Researcher

**Background:** AI/ML graduate student working on medical imaging thesis  
**Goals:** Demonstrate novel segmentation or reconstruction methods  
**Technical Level:** High (Python, PyTorch/TensorFlow, medical imaging fundamentals)  
**Use Cases:**
- Validate segmentation accuracy on benchmark datasets
- Compare implicit vs explicit surface representations
- Generate publication-quality 3D visualizations
- Debug pipeline failures with detailed logging

### 4.2 Secondary Persona: Thesis Committee Member

**Background:** Academic reviewer with medical imaging or computer vision expertise  
**Goals:** Evaluate technical validity and novelty  
**Technical Level:** Medium-High (understands concepts but may not code)  
**Use Cases:**
- Review end-to-end pipeline architecture
- Assess segmentation quality on various nodule types
- Verify mesh reconstruction fidelity
- Evaluate edge case handling

### 4.3 Tertiary Persona: Medical Imaging Student

**Background:** Undergraduate/early graduate student learning medical AI  
**Goals:** Understand medical image processing workflow  
**Technical Level:** Medium (learning phase)  
**Use Cases:**
- Visualize CT data and segmentation
- Interact with 3D models
- Understand pipeline modularity
- Explore failure modes safely -->

---

## 5. Functional Requirements

### 5.1 Data Input Module (must have)

**FR-1.1: CT Volume Upload** 
- Accept DICOM series (multi-file .dcm) or folder have series Dicom 
- Accept MetaImage format (.mhd + .raw)
- Maximum volume size: 512×512×512 voxels
- File size limit: 500MB per upload
- Validate DICOM metadata (Modality=CT)

**FR-1.2: Volume Preprocessing**
- Extract voxel data and spacing information
- Normalize Hounsfield Units to standardized range
- Handle anisotropic spacing (warn if Z-spacing > 2.5mm)
- Generate axial slice thumbnails for visualization
<!-- 
**FR-1.3: Data Quality Checks**
- Detect missing slices in DICOM series
- Warn on low-resolution scans (pixel spacing > 1mm)
- Flag non-chest CT scans (based on FOV or metadata)
- Validate bit depth and value ranges -->

### 5.2 ROI Selection Module (Optional) (must have)

**FR-2.1: Processing Mode Selection**
- User chooses between:
  - **ROI Mode**: Manual bounding box selection (faster, focused)
  - **Full Volume Mode**: Segment entire CT volume (comprehensive, slower)
- Display mode comparison: processing time, memory usage, use cases
- Default: ROI Mode (recommended for single nodule demos)

**FR-2.2: Manual Bounding Box Selection (ROI Mode)**
- Display axial slices with scrollable viewer
- Enable 2D bounding box drawing on current slice
- Propagate bounding box across Z-axis (adjustable height)
- Visual feedback: box dimensions in mm and voxels
- Minimum ROI size: 16×16×16 voxels
- Maximum ROI size: 128×128×128 voxels

<!-- **FR-2.3: ROI Refinement (ROI Mode)**
- Adjust box position and size via drag handles
- Slice-by-slice box modification
- ROI preview with cropped volume
- Reset to full volume option -->
In the scope of this thesis, full-volume mode is demonstrated on limited cases and primarily serves as a conceptual extension.
**FR-2.4: Full Volume Processing (Full Volume Mode)**
- Skip ROI selection entirely
- Process entire CT volume through segmentation
- Downsample volume if dimensions exceed model limits (512³)
- Display warning about increased processing time (est. 30-60s)
- Memory requirement check (minimum 8GB GPU recommended)

**FR-2.5: ROI Validation (ROI Mode Only)**
- Ensure ROI contains lung tissue (HU range check)
- Warn if ROI includes significant artifacts
- Display cropped statistics (mean HU, volume)

### 5.3 Automatic Detection Module (Optional, Future)

**FR-3.1: Nodule Detection**
- Run pre-trained detection model on full volume
- Return bounding box proposals with confidence scores
- Display top-K detections (K≤10) on slice viewer
- Allow selection of detection to initialize ROI

**FR-3.2: Detection Failure Handling**
- If detection fails, fall back to manual ROI selection
- Display clear message: "No nodules detected, please select ROI manually"
- Never block segmentation pipeline due to detection failure
- Log detection failures for debugging (not shown to user)

### 5.4 Segmentation Module (must have)

**FR-4.1: Tumor Segmentation**
- Input: Cropped ROI volume (ROI Mode) OR Full CT volume (Full Volume Mode)
- Model: 3D U-Net or equivalent architecture
- Model variants:
  - **Lightweight model** for ROI (64³-128³ input, ~50M params)
  - **Full-resolution model** for entire volume (512³ input, ~100M params)
- Output: Binary mask or probability map (same dimensions as input)
- Inference mode only (no training in application)

**FR-4.2: Multi-Nodule Detection (Full Volume Mode)**
- Connected component analysis on segmentation output
- Separate each detected nodule into individual segments
- Rank by volume/confidence
- Allow user to select which nodule(s) to reconstruct
- Display count of detected nodules

**FR-4.3: Segmentation Post-Processing**
- Connected component analysis
  - ROI Mode: keep largest component
  - Full Volume Mode: keep all components above threshold OR allow multi-selection
- Optional morphological operations (closing, hole filling)
- Minimum segment volume threshold (64 voxels)
- Export segmentation mask as .npy or .nii.gz

**FR-4.4: Segmentation Visualization**
- Overlay segmentation mask on 2D slices (colored transparency)
- Full Volume Mode: use different colors for multiple nodules
- Toggle mask visibility
- Display segmentation statistics (volume in mm³, extent)
- Slice-by-slice navigation with mask overlay

**FR-4.5: Segmentation Quality Checks**
- Warn if segmentation is empty
- ROI Mode: warn if segmentation touches ROI boundary (potential crop issue)
- Full Volume Mode: warn if >10 nodules detected (possible false positives)
- Calculate and display segmentation confidence (if probabilistic)

### 5.5 Implicit Surface Modeling Module (must have)

**FR-5.1: Continuous Field Construction**
- Convert discrete segmentation mask to implicit representation
- Supported methods:
  - Signed Distance Field (SDF)
  - Probabilistic level set
- Differentiable formulation for future neural implicit extensions

**FR-5.2: Field Properties**
- Resolution: 1-2× segmentation mask resolution
- Smooth interpolation at mask boundaries
- Consistent zero-level set alignment with segmentation

**FR-5.3: Field Validation**
- Verify field gradient magnitude is reasonable
- Check for NaN or infinite values
- Export field as .npy for debugging

### 5.6 Surface Extraction Module (must have)

**FR-6.1: Marching Cubes Implementation**
- Extract iso-surface at level = 0.5 (for probabilistic) or 0.0 (for SDF)
- Adaptive grid resolution based on tumor size
- Generate triangle mesh with normals
- Typical output: 10K-100K triangles

**FR-6.2: Mesh Post-Processing**
- Remove disconnected components (keep largest)
- Smooth mesh (Laplacian smoothing, configurable iterations)
- Decimate mesh if triangle count > 200K
- Validate mesh is manifold (no holes)

**FR-6.3: Mesh Quality Metrics**
- Triangle count, vertex count
- Surface area (mm²)
- Volume (mm³)
- Bounding box dimensions
- Aspect ratio and triangle quality distribution

**FR-6.4: Mesh Export**
- Export as .glb (binary glTF, preferred for web)
- Export as .obj (text format, for external tools)
- Include scale and origin metadata

### 5.7 2D Visualization Module (must have)

**FR-7.1: Slice Viewer**
- Display axial slices with adjustable window/level
- Scroll through slices with mouse wheel or slider
- Preset windowing: Lung (-600, 1600), Mediastinum (50, 400)
- Zoom and pan controls
- Slice index and Z-position display

**FR-7.2: Overlay Rendering**
- Segmentation mask overlay (adjustable opacity, color)
- ROI bounding box outline
- Mesh contour overlay (intersection of 3D mesh with 2D slice plane)

**FR-7.3: Comparison Tools**
- Side-by-side view: original vs. segmentation
- Toggle between different pipeline outputs
- Difference visualization (for validation)

### 5.8 3D Visualization Module (must have)

**FR-8.1: WebGL Mesh Rendering**
- Three.js-based 3D viewer
- Orbit camera controls (rotate, zoom, pan)
- Lighting: ambient + directional lights
- Material: Phong shading with adjustable specularity

**FR-8.2: Mesh Interaction**
- Rotate model with mouse drag
- Zoom with scroll
- Reset camera view button
- Toggle wireframe mode
- Adjust transparency

**FR-8.3: Coordinate System**
- Display anatomical axes (Superior-Inferior, Anterior-Posterior, Left-Right)
- Grid plane at mesh base
- Scale reference (10mm cube)

**FR-8.4: Mesh Quality Visualization**
- Color-code by curvature or normal variation
- Highlight potential artifacts (isolated triangles, high aspect ratio)
- Display cross-sections for internal structure inspection

### 5.9 Pipeline Orchestration

**FR-9.1: Processing Workflow**

**ROI Mode (Default):**
- Step 1: Upload and validate CT volume
- Step 2: Select ROI via bounding box
- Step 3: Run segmentation on ROI
- Step 4: Compute implicit surface
- Step 5: Extract mesh
- Step 6: Visualize results

**Full Volume Mode:**
- Step 1: Upload and validate CT volume
- Step 2: Skip ROI selection (full volume processing)
- Step 3: Run segmentation on entire volume
- Step 4: Detect multiple nodules (if present)
- Step 5: Select nodule(s) for reconstruction
- Step 6: Compute implicit surface for each selected nodule
- Step 7: Extract mesh(es)
- Step 8: Visualize results (with multi-object rendering)

**FR-9.2: Progress Tracking**
- Display current pipeline stage
- Estimated time remaining per stage
  - ROI Mode: 15-30 seconds total
  - Full Volume Mode: 30-90 seconds total
- Detailed logs accessible via toggle

**FR-9.3: Intermediate Result Caching**
- Cache segmentation mask for re-running surface extraction
- Cache implicit field for experimenting with iso-values
- Full Volume Mode: cache all detected nodules separately
- Session-based storage (no persistent DB)

**FR-9.4: Error Recovery**
- Allow re-running individual stages without full re-upload
- Checkpoint intermediate outputs
- Clear error messages with suggested actions
- Full Volume Mode: if memory error, suggest switching to ROI Mode

### 5.10 Validation
<!-- 
**FR-10.1: Mesh-Segmentation Concordance**
- Overlay mesh contour on 2D slices
- Compute Dice coefficient between mesh rasterization and segmentation
- Highlight discrepancies (color-coded per-slice)

**FR-10.2: Volume Comparison**
- Compare segmentation volume vs. mesh volume
- Expected: <5% difference for high-quality reconstruction
- Flag large discrepancies for review

**FR-10.3: Ground Truth Comparison (Optional)**
- Upload reference segmentation
- Compute metrics: Dice, Hausdorff distance, surface distance
- Display error heatmap on mesh -->

---

## 6. Non-Functional Requirements

### 6.1 Performance

**NFR-1.1: Processing Time**
- ROI Mode:
  - Segmentation inference: <10 seconds for 64³ ROI on GPU
  - Implicit surface modeling: <5 seconds
  - Marching Cubes extraction: <3 seconds
  - End-to-end: <30 seconds
- Full Volume Mode:
  - Segmentation inference: 20-40 seconds for 512³ volume on GPU
  - Multi-nodule separation: <10 seconds
  - Implicit surface per nodule: <5 seconds
  - Marching Cubes per nodule: <3 seconds
  - End-to-end (single nodule): <60 seconds
  - End-to-end (multiple nodules): <90 seconds

**NFR-1.2: UI Responsiveness**
- Slice viewer rendering: 60 FPS
- 3D mesh rendering: 30 FPS minimum for 50K triangle mesh
- Full Volume Mode: support rendering up to 5 nodules simultaneously (10K triangles each)
- API response time: <500ms for metadata endpoints

**NFR-1.3: Scalability**
- Support concurrent processing of up to 3 users (demo environment)
- Memory management:
  - ROI Mode: 4GB GPU VRAM minimum
  - Full Volume Mode: 8GB GPU VRAM minimum (16GB recommended)

---

## 7. System Architecture

### 7.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend                     │
└────────────────────┬────────────────────────────────┘
                     │ HTTPS / REST API
┌────────────────────▼────────────────────────────────┐
│              FastAPI Backend                        │
│  ┌──────────────────────────────────────────────┐   │ 
│  │  Endpoints: /upload, /segment, /reconstruct  │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────┘
                     │ gRPC / HTTP
┌────────────────────▼────────────────────────────────┐
│          Serverless GPU Worker                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Modal / Runpod / Replicate                  │   │
│  ├──────────────┬──────────────┬────────────────┤   │
│  │ Segmentation │ Implicit SDF │ Marching Cubes │   │
│  │ (...)        │ (SDF Conv.)  │ (skimage)      │   │
│  └──────────────┴──────────────┴────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 7.2 Technology Stack

**Frontend:**
- React 18+ with TypeScript
- Three.js (r150+) for 3D rendering
- Cornerstone.js or custom canvas for 2D DICOM viewer
- Axios for API communication
- Tailwind CSS for styling

**Backend:**
- FastAPI (Python 3.10+)
- PyTorch 2.0+ for model inference
- pydicom for DICOM parsing
- SimpleITK for MetaImage handling
- scikit-image for Marching Cubes

**Deployment:**
- Vercel for frontend hosting
- Modal Labs for GPU serverless functions

**Model Hosting:**
- Model weights stored in HuggingFace Hub
- Lazy loading on first inference request
- Option to cache in GPU worker memory

### 7.3 API Design

**Base URL:** `https://api.tumor3d.research/v1`

**Endpoints:**

```
POST   /upload
       Body: multipart/form-data (DICOM files or .mhd+.raw)
       Response: { session_id, volume_id, metadata }

GET    /volume/{volume_id}/slices
       Query: ?index=0&window_center=0&window_width=1600
       Response: Base64-encoded PNG slice image

POST   /volume/{volume_id}/segment
       Body: { 
         mode: "roi" | "full_volume",
         roi: { x, y, z, width, height, depth } | null  // null for full_volume
       }
       Response: { task_id }

POST   /volume/{volume_id}/roi
       Body: { x, y, z, width, height, depth }
       Response: { roi_id, cropped_volume_metadata }

POST   /roi/{roi_id}/segment
       Response: { task_id }

GET    /task/{task_id}/status
       Response: { 
         status, 
         progress, 
         result_url,
         nodule_count?: number  // for full_volume mode
       }

GET    /segment/{segment_id}/nodules
       Query: ?min_volume=100  // mm³
       Response: { 
         nodules: [
           { nodule_id, volume, centroid, bbox, confidence }
         ]
       }

POST   /segment/{segment_id}/reconstruct
       Body: { 
         iso_value, 
         smoothing_iterations,
         nodule_ids?: string[]  // for multi-nodule reconstruction
       }
       Response: { task_id }

GET    /mesh/{mesh_id}/download
       Query: ?format=glb
       Response: Binary mesh file

GET    /mesh/{mesh_id}/metadata
       Response: { 
         triangle_count, 
         volume, 
         surface_area,
         nodule_count?: number  // if multi-nodule mesh
       }

DELETE /session/{session_id}
       Response: { deleted: true }
```

### 7.4 Data Flow

**Detailed Processing Pipeline:**

1. **Upload Phase:**
   - Client uploads DICOM series or MetaImage
   - Backend validates files, parses metadata
   - Extract voxel data, normalize HU values
   - Generate volume ID, store in session cache (Redis)
   - Return volume metadata to client

2. **Mode Selection Phase (NEW):**
   - Client presents user with processing mode options:
     - **ROI Mode**: Faster, focused (proceeds to ROI Selection)
     - **Full Volume Mode**: Comprehensive (skips to Segmentation)
   - User selects mode
   - Client sends mode selection to backend

3a. **ROI Selection Phase (ROI Mode only):**
   - Client requests axial slices for visualization
   - User draws bounding box
   - Client sends ROI coordinates to backend
   - Backend crops volume, validates ROI
   - Store cropped volume, return ROI ID

3b. **Skip ROI Phase (Full Volume Mode):**
   - No user interaction required
   - Backend prepares full volume for segmentation
   - Downsample if necessary (>512³)
   - Store volume metadata

4. **Segmentation Phase:**
   - Client triggers segmentation task
   - Backend queues GPU inference job
   - GPU worker loads appropriate model (lightweight for ROI, full-res for volume)
   - Run inference on cropped ROI or full volume
   - Post-process mask:
     - ROI Mode: keep largest component
     - Full Volume Mode: separate multiple nodules
   - Store segmentation mask(s), return task status

5. **Nodule Selection Phase (Full Volume Mode only):**
   - If multiple nodules detected, present list to user
   - User selects which nodule(s) to reconstruct
   - Default: reconstruct all nodules

6. **Implicit Surface Phase:**
   - Client triggers surface modeling
   - Backend computes SDF or occupancy field from mask
   - Process each selected nodule separately (if multiple)
   - Store implicit field(s) as NumPy array
   - Return field metadata

7. **Mesh Extraction Phase:**
   - Client triggers Marching Cubes
   - Backend runs surface extraction on implicit field(s)
   - Post-process mesh(es): smoothing, decimation
   - Convert to .glb format
   - Store mesh(es), return download URL(s)

8. **Visualization Phase:**
   - Client downloads mesh file(s)
   - Three.js loads and renders mesh(es) in browser
   - ROI Mode: single mesh rendering
   - Full Volume Mode: multi-mesh rendering with different colors
   - User interacts with 3D model(s)
   - Optional: overlay mesh contour on 2D slices

### 7.5 Deployment Architecture

**Modal Labs**

```yaml
Frontend: Vercel (static hosting)
Backend API: Modal (serverless Python)
GPU Inference: Modal GPU functions
# Storage: Modal Volumes (ephemeral) + S3 (model weights)
# Database: Redis (Modal managed or Upstash)
```


---

## 8. Data Flow & Pipeline Description

### 8.1 Stage 1: CT Volume Input

**Input:**
- DICOM series (multiple .dcm files (folder)) OR MetaImage (.mhd + .raw)

**Processing:**
1. Parse DICOM metadata (Patient Position, Pixel Spacing, Slice Thickness)
2. Load voxel data into 3D NumPy array (shape: Z×Y×X)
3. Apply Hounsfield Unit scaling: `HU = pixel_value × RescaleSlope + RescaleIntercept`
4. Validate orientation (prefer LPS or RAS, convert if needed)
5. Resample if necessary (target: isotropic 1mm³ voxels for segmentation)

**Output:**
- Normalized 3D volume tensor (float32)
- Metadata: spacing, origin, orientation, dimensions
- Quality flags: anisotropy warning, missing slice detection

**Edge Cases:**
- Non-contiguous DICOM series → reject or warn
- Missing metadata → use defaults, warn user
- Corrupted files → skip, proceed with available data
- Non-chest CT → warn based on FOV heuristics

### 8.2 Stage 2: ROI Selection (Optional)

**Mode Selection:**

User selects processing mode at pipeline start:
- **ROI Mode**: Faster, focused on single nodule, requires manual selection
- **Full Volume Mode**: Comprehensive, finds all nodules, more computationally intensive

**ROI Mode Processing:**

**Input:** Normalized CT volume, user bounding box coordinates

**Processing:**
1. Validate bounding box is within volume bounds
2. Crop volume: `roi_volume = volume[z:z+d, y:y+h, x:x+w]`
3. Verify ROI contains lung tissue (mean HU in range [-1000, 200])
4. Pad to minimum size if necessary (16³ voxels)
5. Resize to model input size (typically 64³ or 128³)

**Output:**
- Cropped ROI volume
- Transformation matrix (for mapping back to original volume)
- ROI statistics (volume, mean HU, std HU)

**Full Volume Mode Processing:**

**Input:** Normalized CT volume

**Processing:**
1. Downsample volume if exceeds model limits:
   - If any dimension > 512: resize to max 512³ maintaining aspect ratio
   - Store downsampling factor for later upscaling
2. Pad to ensure dimensions divisible by model's stride (typically 16)
3. No cropping - entire volume passed to segmentation

**Output:**
- Full volume (potentially downsampled)
- Downsampling metadata
- Volume statistics

**Automatic Detection Mode (Future):**

**Input:** Full CT volume

**Processing:**
1. Run nodule detection model (e.g., YOLO-3D, RetinaNet-3D)
2. Apply NMS to remove duplicate detections
3. Filter detections by confidence threshold (>0.5)
4. Rank by confidence × size metric
5. Return top-K bounding boxes as ROI proposals

**Output:**
- List of bounding boxes with confidence scores
- Visualization overlay for user selection

**Failure Handling:**
- If detection fails, fall back to manual ROI mode or full volume mode
- Display clear message: "No nodules detected, please select processing mode"
- Detection errors do NOT block pipeline progression

### 8.3 Stage 3: Segmentation

**Input:** ROI volume (ROI Mode) OR Full CT volume (Full Volume Mode)

**Model Architecture:** 3D U-Net
- Encoder: 4 levels with 3D convolutions, batch norm, ReLU
- Decoder: 4 levels with transposed convolutions, skip connections
- Output: Sigmoid activation → probability map ∈ [0,1]

**Model Variants:**
- **Lightweight U-Net** (ROI Mode): Input 64³-128³, ~50M parameters
- **Full-Resolution U-Net** (Full Volume Mode): Input up to 512³, ~100M parameters

**Processing:**
1. Preprocess input: normalize to [0,1] or standardize (zero mean, unit variance)
2. Add batch dimension: (1, C, D, H, W)
3. Run model inference (torch.no_grad())
4. Apply threshold to probability map (default: 0.5)
5. **Branching logic based on mode:**

**ROI Mode Post-Processing:**
- Connected component analysis → keep largest component
- Optional: morphological closing (fill small holes)
- Resize mask back to original ROI dimensions (if downsampled)

**Full Volume Mode Post-Processing:**
- Connected component analysis → identify all components
- Filter components by volume (minimum: 100 mm³ = ~100 voxels at 1mm³)
- Rank components by volume
- Label each component as separate nodule (nodule_1, nodule_2, ...)
- Compute bounding box for each nodule
- Return multi-nodule segmentation

**Output:**

**ROI Mode:**
- Binary segmentation mask (uint8, same size as ROI)
- Probability map (float32, optional)
- Segmentation metrics: volume, bounding box, centroid

**Full Volume Mode:**
- Multi-label segmentation mask (uint8, 0=background, 1-N=nodules)
- Per-nodule statistics:
  - nodule_id, volume, centroid, bounding box, mean probability
- Nodule count

**Quality Checks:**
- Empty segmentation → return error, suggest alternative mode
- ROI Mode: segmentation touches boundary → warn potential truncation
- Full Volume Mode: 
  - >10 nodules → warn possible false positives, show top 10 by volume
  - All nodules <200 mm³ → warn possible over-sensitivity
- Volume validation: flag if volume > 50% of input (likely over-segmentation)

### 8.4 Stage 4: Implicit Surface Modeling

**Input:** Binary segmentation mask

**Method 1: Signed Distance Field (SDF)**

**Processing:**
1. Apply distance transform to mask: `dist = distance_transform_edt(mask)`
2. Compute exterior distance: `dist_ext = distance_transform_edt(~mask)`
3. Combine: `sdf = dist_ext - dist`
4. Normalize: `sdf = sdf / voxel_spacing` (physical units)
5. Optional: Gaussian smoothing for C¹ continuity

**Output:**
- SDF volume (float32, same size as mask)
- Zero-level set corresponds to mask boundary

**Method 2: Occupancy Field**

**Processing:**
1. Convolve mask with Gaussian kernel (σ = 1-2 voxels)
2. Normalize to [0,1] range
3. Optional: apply tanh for sharper boundaries

**Output:**
- Occupancy field (float32, 0=exterior, 1=interior)
- Iso-value 0.5 corresponds to mask boundary

**Rationale for Implicit Modeling:**
- Enables sub-voxel accuracy in mesh extraction
- Facilitates smooth surface reconstruction
- Supports differentiable rendering (future neural implicit work)
- Decouples discrete segmentation from continuous representation

### 8.5 Stage 5: Surface Extraction

**Input:** Implicit field (SDF)

**Algorithm:** Marching Cubes (skimage.measure.marching_cubes)

**Processing:**
1. Select iso-value:
   - SDF: iso_value = 0.0 (zero-level set)
2. Run Marching Cubes: `verts, faces, normals, values = marching_cubes(field, level=iso_value)`
3. Transform vertices to physical coordinates: `verts_mm = verts × spacing + origin`
4. Flip normals if needed (ensure outward-pointing)
5. Remove small disconnected components (threshold: <1% of total triangles)
6. Laplacian smoothing (5-10 iterations, λ=0.5)
7. Decimate if triangle count > 200K (target: 50K-100K)

**Output:**
- Triangle mesh (vertices: Nx3 float, faces: Mx3 int)
- Vertex normals (Nx3 float)
- Mesh statistics (volume, surface area, Euler characteristic)

**Quality Metrics:**
- Triangle quality (aspect ratio distribution)
- Mesh manifoldness check (closed, no self-intersections)
- Volume concordance with segmentation (Dice on rasterized mesh)

**Mesh Export:**
- Convert to glTF 2.0 format (.glb binary)
- Include metadata as glTF extras (spacing, origin, pipeline params)
- Alternative exports: .obj, .stl, .ply

### 8.6 Pipeline Orchestration Logic

**Sequential Dependencies:**

```
CT Volume → [ROI Selection OR Skip] → Segmentation → Implicit Field → Mesh Extraction
    ↓              ↓                        ↓               ↓              ↓
 Validation   Crop/Passthrough          Inference       SDF Calc       Marching Cubes

Decision Point: User chooses processing mode
├─ ROI Mode: Manual bounding box → Crop volume → Segmentation on ROI
└─ Full Volume Mode: Skip ROI → Segmentation on entire volume
```

**Parallel Opportunities:**
- Slice thumbnail generation (parallel to ROI selection)
- Multi-ROI segmentation (if multiple nodules selected)

**Checkpointing:**
- Save after each stage to allow re-running downstream steps
- Store intermediate results with expiration (6 hours)

**Error Propagation:**
- Validation failures → halt pipeline, return user-actionable error
- Segmentation empty → proceed to implicit stage, warn user
- Mesh extraction failure → retry with adjusted parameters, fallback to coarser resolution

---

## 9. Error Handling & Edge Cases

---