# sam3d-body-batch-runner

Batch inference wrapper for Meta’s **SAM-3D-Body** model.  
This script runs **all JPG/JPEG images in a folder** through SAM-3D-Body and saves meshes and visualizations.

This repository **does not vendor** Meta’s code.  
The upstream `sam-3d-body` library is installed as a dependency via `pip`.

Upstream project: https://github.com/facebookresearch/sam-3d-body

---

## Environment

- **Python**: 3.11.14  
- **OS**: Windows, macOS, or Linux  
- **GPU**: Optional (CUDA recommended for performance)

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/ansomin/sam3dbody-batch-runner.git
cd sam3d-body-batch-runner
```

### 2. Follow sam-3d-body repo's INSTALL.md

https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md

### 3. Install dependencies unique to this repo (Optional)

Only follow this step if script doesn't work after step 2.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Note:

This installs Meta’s sam-3d-body library directly from GitHub into your Python environment.

## Hugging Face / Model Setup (Important)

SAM-3D-Body downloads model weights from Hugging Face.

Please follow the official SAM-3D-Body installation instructions regarding:

* Hugging Face access

* Authentication (if required)

* Model downloads

👉 Refer to:
https://github.com/facebookresearch/sam-3d-body

If Hugging Face authentication is required, make sure you are logged in:

```bash
huggingface-cli login
```

## Input / Output Structure

Place images under:

inputs/<folder_name>/


Example:

inputs/example_images/
  image1.jpg
  image2.jpg


Outputs will be written to:

outputs/<folder_name>/


Each image gets its own subfolder with meshes, visualizations, and JSON outputs.

## Running the Script

**Make sure the virtual environment is activated before running the script.**

```bash
python run_sam3d_body.py \
  --input_dir example_images \
  --device cuda \
  --save_mesh \
  --save_2d_vis
```

### Arguments

* --input_dir : Name of a folder inside ./inputs

* --device : cpu or cuda

* --save_mesh : Save 3D mesh outputs (OBJ)

* --save_2d_vis : Save 2D skeleton overlays

* --render_3d_vis : Try OpenGL-based 3D rendering (may fail on headless systems)

## Notes & Known Issues

* OpenGL-based rendering may fail on headless machines or some Windows setups.

* CUDA performance depends on PyTorch + driver compatibility.

* For GPU usage, install a CUDA-enabled PyTorch build following:
https://pytorch.org

## License & Attribution

This project is a lightweight wrapper around Meta’s SAM-3D-Body.
All model code and weights are subject to the upstream project’s license.

Upstream repository:
https://github.com/facebookresearch/sam-3d-body