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

- **Mean angular error reduction**: 6.77° (32.5%) vs 12-direction CSD baseline (FA ≥ 0.5)
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
- `{subject_id}_fod_predicted.nii.gz` — full-volume NIfTI (45 SH coefficients per voxel)

All SH coefficients use DIPY's `descoteaux07` basis (the default of
`ConstrainedSphericalDeconvModel`), which is also the basis of the
64-direction reference used for training and evaluation. MRtrix3 uses a
different SH convention: convert the coefficients (e.g. with DIPY's SH basis
conversion utilities) before loading them in MRtrix3 tools such as `mrview`
or `tckgen`.

---

## Sample data

`sample_data/` contains subject `sub-10347` from the UCLA Consortium for
Neuropsychiatric Phenomics (CNP) dataset, subsampled to 12 optimally-distributed
directions (electrostatic repulsion, Jones et al. 1999).

Original data: OpenNeuro ds000030 (CC0 license)
https://openneuro.org/datasets/ds000030

The 64-direction acquisition used as reference is not included because of its
size. It is needed only to run the evaluation (see below).

---

## Evaluation

`run_evaluation.py` compares the predicted FODs with a 64-direction CSD
reference. Download the original DWI of `sub-10347` from OpenNeuro, either from
the dataset page (`sub-10347/dwi/sub-10347_dwi.nii.gz`) or with the AWS CLI:

```bash
aws s3 cp --no-sign-request \
    s3://openneuro.org/ds000030/sub-10347/dwi/sub-10347_dwi.nii.gz \
    sample_data/sub-10347_dwi64.nii.gz
```

Then run:

```bash
python run_evaluation.py \
    --pred  output/sub-10347_fod_predicted.npz \
    --nii64 sample_data/sub-10347_dwi64.nii.gz \
    --bval  sample_data/sub-10347_64.bval \
    --bvec  sample_data/sub-10347_64.bvec
```

The script reports angular error (AE), peak overlap (PO) and the fraction of
voxels below 10° for each white matter FA stratum.

---

## Acquisition scheme compatibility

ARGOnet was trained and validated on the CNP acquisition scheme
(64 directions, b=1000 s/mm², single-shell, Siemens TrioTim 3T).
The optimal 12-direction subset was selected via electrostatic repulsion
minimization (mean angular separation 62.0°).

Performance on alternative 12-direction schemes has been partially validated
(retention 74.5%, Table 2 of the manuscript). Full retraining on a
scheme-specific subset is recommended for optimal results on different protocols.

**If your data uses a different acquisition protocol**, we welcome collaboration
for scheme-specific fine-tuning. Please open a GitHub Issue with your
acquisition parameters (directions, b-value, scanner).

---

## How to cite

If you use ARGOnet (code or pre-trained weights) in your research, please cite:

> [AUTHORS] (2026). *ARGOnet: Angular Resolution Graph Operator* (Version [X.Y]) [Software and model weights]. Zenodo. https://doi.org/[ZENODO_DOI]

The associated manuscript is currently under review at *NeuroImage*.
This section will be updated with the article reference upon publication.
A machine-readable citation is provided in `CITATION.cff`.

---

## License

The code and pre-trained weights in this repository are released under the
**Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)**. See `LICENSE` for the full text.

- **Attribution**: any use must credit the authors (see *How to cite*).
- **NonCommercial**: commercial use of any kind is not permitted.
  For commercial licensing, please contact the authors.

The sample data in `sample_data/` remain under their original CC0 license
(OpenNeuro ds000030).
