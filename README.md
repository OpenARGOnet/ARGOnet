# ARGOnet — Angular Resolution Graph Operator

Official inference code and pre-trained weights for **ARGOnet**, a lightweight
dual-branch graph neural network for angular super-resolution of fiber
orientation distributions (FODs) from 12-direction dMRI.

---

## What ARGOnet does

Clinical dMRI acquisitions typically use 12–30 gradient directions — too few
for reliable FOD estimation with constrained spherical deconvolution (CSD).
ARGOnet recovers high-quality FODs from 12-direction data by modelling white
matter as an anatomically constrained spatial graph and applying a dual-branch
GNN (GAT + GraphSAGE) with a residual decoder.

- **Mean angular error reduction**: 6.65° (30.7%) vs 12-direction CSD baseline
- **Zero-shot generalization** to schizophrenia and bipolar disorder cohorts
- **No GPU required**: full-volume inference in ~124s on a standard CPU (8 GB RAM)
- **~100k parameters**: model weights < 5 MB

---

## Installation

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, match your PyTorch version:
https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Usage

```bash
python run_inference.py \
    --nii  sample_data/sub-10347_dwi12.nii.gz \
    --bval sample_data/sub-10347.bval \
    --bvec sample_data/sub-10347.bvec \
    --model ARGO.pt \
    --out   output/
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--nii` | 4D DWI NIfTI file (12 directions) | required |
| `--bval` | .bval file | required |
| `--bvec` | .bvec file | required |
| `--model` | Path to ARGO.pt | `ARGO.pt` |
| `--out` | Output directory | `output/` |
| `--fa_threshold` | FA threshold for WM graph | `0.3` |
| `--subject_id` | Subject ID for output filenames | derived from `--nii` |

### Output

- `{subject_id}_fod_predicted.npz` — predicted SH coefficients (order 8, 45 coefficients per WM voxel) + voxel coordinates
- `{subject_id}_fod_predicted.nii.gz` — full-volume NIfTI (viewable in FSLeyes, 3D Slicer, MRtrix3)

---

## Sample data

`sample_data/` contains subject `sub-10347` from the UCLA Consortium for
Neuropsychiatric Phenomics (CNP) dataset, subsampled to 12 optimally-distributed
directions (electrostatic repulsion, Jones et al. 1999).

Original data: OpenNeuro ds000030 (CC0 license)
https://openneuro.org/datasets/ds000030

---

## Acquisition scheme compatibility

ARGOnet was trained and validated on the CNP acquisition scheme
(64 directions, b=1000 s/mm², single-shell, Siemens TrioTim 3T).
The optimal 12-direction subset was selected via electrostatic repulsion
minimization (mean angular separation 62.0°).

Performance on alternative 12-direction schemes has been partially validated
(retention 71.8%, Supplementary Table S2). Full retraining on a
scheme-specific subset is recommended for optimal results on different protocols.

**If your data uses a different acquisition protocol**, we welcome collaboration
for scheme-specific fine-tuning. Please open a GitHub Issue with your
acquisition parameters (directions, b-value, scanner).

---

## License

MIT License. See LICENSE for details.
