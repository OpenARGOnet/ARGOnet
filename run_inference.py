"""
ARGO — Angular Resolution Graph Operator
=========================================
Inference script for angular super-resolution of fiber orientation
distributions (FODs) from 12-direction dMRI.

Usage:
    python run_inference.py \
        --nii  sample_data/sub-10347_dwi12.nii.gz \
        --bval sample_data/sub-10347.bval \
        --bvec sample_data/sub-10347.bvec \
        --model ARGO.pt \
        --out   output/

Dataset:
    Sample data from the UCLA Consortium for Neuropsychiatric Phenomics
    (CNP) dataset, OpenNeuro ds000030 (CC0 license).
    https://openneuro.org/datasets/ds000030
"""

import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import (auto_response_ssst,
                                    ConstrainedSphericalDeconvModel)
from dipy.reconst import dti
from scipy.ndimage import zoom, binary_erosion


# ──────────────────────────────────────────────────────────────────
# Model architecture (must match ARGO.pt)
# ──────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, dim_in=128, dim_hidden=256, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(dim_in, dim_hidden)
        self.bn1  = nn.BatchNorm1d(dim_hidden)
        self.fc2  = nn.Linear(dim_hidden, dim_in)
        self.bn2  = nn.BatchNorm1d(dim_in)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop(F.elu(self.bn1(self.fc1(x))))
        return F.elu(self.bn2(self.fc2(z)) + x)


class _ARGO(nn.Module):
    """
    Dual-branch GNN: GAT + GraphSAGE with residual decoder.
    Input:  47-dim feature vector per WM voxel (45 SH + FA + MD)
    Output: 45 spherical harmonic coefficients (SH order 8)
    ~100k trainable parameters.
    """
    def __init__(self, in_channels=47, out_channels=45,
                 latent_dim=64, heads=2):
        super().__init__()
        self.gat1    = GATConv(in_channels, 32,
                               heads=heads, concat=True, dropout=0.1)
        self.gat2    = GATConv(32*heads,    32,
                               heads=heads, concat=True, dropout=0.1)
        self.bn_gat  = nn.BatchNorm1d(32*heads)
        self.sage1   = SAGEConv(in_channels, 64)
        self.sage2   = SAGEConv(64, 64)
        self.bn_sage = nn.BatchNorm1d(64)
        combined     = (32*heads) + 64          # 128
        self.res     = _ResBlock(combined, 256, 0.1)
        self.dec_out = nn.Sequential(
            nn.Linear(combined, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, out_channels),
        )

    def forward(self, x, edge_index):
        g = F.elu(self.gat1(x, edge_index))
        g = self.bn_gat(F.elu(self.gat2(g, edge_index)))
        s = F.elu(self.sage1(x, edge_index))
        s = self.bn_sage(F.elu(self.sage2(s, edge_index)))
        z = self.res(torch.cat([g, s], dim=1))
        return self.dec_out(z)


# ──────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────

def _preprocess(nii_path, bval_path, bvec_path):
    """
    Load and preprocess a 12-direction DWI volume.
    Returns: data_up, gtab, mask, affine, header
    """
    print("  [1/5] Loading DWI volume...")
    img    = nib.load(nii_path)
    data   = img.get_fdata().astype(np.float32)
    affine = img.affine
    header = img.header

    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)

    if data.ndim == 3:
        raise ValueError("Input must be a 4D DWI volume (x, y, z, directions).")
    if data.shape[3] != len(bvals):
        raise ValueError(
            f"Number of volumes ({data.shape[3]}) does not match "
            f"number of b-values ({len(bvals)})."
        )

    print(f"  [1/5] Volume shape: {data.shape}  |  Directions: {len(bvals)}")

    print("  [2/5] Upsampling 1.5× in-plane...")
    data_up = zoom(data, (1.5, 1.5, 1, 1), order=1)
    print(f"  [2/5] Upsampled shape: {data_up.shape}")

    print("  [2/5] Computing brain mask...")
    _, mask      = median_otsu(data_up, vol_idx=[0], numpass=4, dilate=1)
    mask_refined = binary_erosion(mask, iterations=1).astype(bool)
    print(f"  [2/5] Brain voxels: {mask_refined.sum():,}")

    gtab = gradient_table(bvals, bvecs=bvecs)
    return data_up, gtab, mask_refined, affine, header


def _extract_features(data_up, gtab, mask):
    """
    Run CSD and DTI on 12-direction data.
    Returns: features (N, 47), sh_low (N, 45), coords (N, 3), mask_shape
    """
    print("  [3/5] Running CSD on 12-direction data...")
    response, _ = auto_response_ssst(gtab, data_up,
                                     roi_radii=10, fa_thr=0.7)
    csd_model   = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit     = csd_model.fit(data_up, mask=mask)
    sh_low      = csd_fit.shm_coeff

    print("  [3/5] Running DTI for FA and MD...")
    tenfit = dti.TensorModel(gtab).fit(data_up, mask=mask)
    fa_3d  = tenfit.fa
    md_3d  = tenfit.md

    sh_low_v     = sh_low[mask]
    fa_v         = fa_3d[mask][..., np.newaxis]
    md_v         = md_3d[mask][..., np.newaxis]
    features     = np.hstack([sh_low_v, fa_v, md_v]).astype(np.float32)
    coords       = np.array(np.where(mask)).T.astype(np.int16)

    return features, sh_low_v, coords, mask.shape


def _build_graph(features, coords, mask_shape, fa_threshold, scaler_x):
    """
    Build white matter graph: nodes = WM voxels (FA >= threshold),
    edges = 26-neighbourhood connectivity.
    Returns: PyG Data object, wm_indices (indices into features array)
    """
    print(f"  [4/5] Building WM graph (FA >= {fa_threshold})...")

    fa      = features[:, 45]
    wm_mask = fa >= fa_threshold
    n_wm    = int(wm_mask.sum())

    if n_wm < 100:
        raise ValueError(
            f"Only {n_wm} WM voxels found. "
            "Check FA threshold or input data quality."
        )

    x_wm      = features[wm_mask]
    fa_wm     = fa[wm_mask]
    coords_wm = coords[wm_mask]

    x_norm = scaler_x.transform(x_wm).astype(np.float32)

    node_map = np.full(mask_shape, -1, dtype=np.int32)
    node_map[coords_wm[:, 0],
             coords_wm[:, 1],
             coords_wm[:, 2]] = np.arange(n_wm, dtype=np.int32)

    offsets = np.array([(dx, dy, dz)
                        for dx in [-1, 0, 1]
                        for dy in [-1, 0, 1]
                        for dz in [-1, 0, 1]
                        if not (dx == 0 and dy == 0 and dz == 0)],
                       dtype=np.int32)

    src_list, dst_list = [], []
    for offset in offsets:
        neigh = coords_wm + offset
        valid = (
            (neigh[:, 0] >= 0) & (neigh[:, 0] < mask_shape[0]) &
            (neigh[:, 1] >= 0) & (neigh[:, 1] < mask_shape[1]) &
            (neigh[:, 2] >= 0) & (neigh[:, 2] < mask_shape[2])
        )
        neigh_valid = neigh[valid]
        dst_idx     = node_map[neigh_valid[:, 0],
                               neigh_valid[:, 1],
                               neigh_valid[:, 2]]
        keep = dst_idx != -1
        src_list.append(np.where(valid)[0][keep])
        dst_list.append(dst_idx[keep])

    edge_index = torch.tensor(
        np.vstack([np.concatenate(src_list),
                   np.concatenate(dst_list)]),
        dtype=torch.long)

    print(f"  [4/5] WM nodes: {n_wm:,}  |  Edges: {edge_index.shape[1]:,}")

    graph = Data(
        x          = torch.tensor(x_norm, dtype=torch.float32),
        edge_index = edge_index,
        num_nodes  = n_wm,
        fa         = torch.tensor(fa_wm, dtype=torch.float32),
    )
    return graph, wm_mask, coords_wm


def _run_inference(model, graph):
    print("  [5/5] Running ARGO inference...")
    t0 = time.time()
    with torch.no_grad():
        pred_norm = model(graph.x, graph.edge_index).numpy()
    elapsed = time.time() - t0
    print(f"  [5/5] Inference completed in {elapsed:.1f}s")
    return pred_norm, elapsed


def _save_outputs(pred_norm, scaler_y, wm_mask, coords_wm,
                  mask_shape, affine, header, out_dir, subject_id):
    """
    Inverse-transform predictions and save as .npz and .nii.gz
    """
    os.makedirs(out_dir, exist_ok=True)

    sh_pred = scaler_y.inverse_transform(pred_norm).astype(np.float32)

    # Save .npz
    npz_path = os.path.join(out_dir, f"{subject_id}_fod_predicted.npz")
    np.savez_compressed(npz_path,
                        sh_coefficients = sh_pred,
                        coords          = coords_wm,
                        mask_shape      = np.array(mask_shape),
                        wm_mask         = wm_mask)
    print(f"  Saved: {npz_path}")

    # Reconstruct 4D volume for NIfTI
    # Adjust affine for 1.5x in-plane upsampling
    affine_up          = affine.copy()
    affine_up[0, 0]   /= 1.5
    affine_up[1, 1]   /= 1.5

    vol = np.zeros((*mask_shape, sh_pred.shape[1]), dtype=np.float32)
    # Only WM voxels have predictions
    wm_indices = np.where(wm_mask)[0]
    all_coords = coords_wm  # (N_wm, 3)
    vol[all_coords[:, 0],
        all_coords[:, 1],
        all_coords[:, 2]] = sh_pred

    nii_path = os.path.join(out_dir, f"{subject_id}_fod_predicted.nii.gz")
    nib.save(nib.Nifti1Image(vol, affine_up), nii_path)
    print(f"  Saved: {nii_path}")

    return npz_path, nii_path


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ARGO — Angular FOD super-resolution from 12-direction dMRI"
    )
    parser.add_argument("--nii",   required=True,
                        help="Path to 4D DWI NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--bval",  required=True,
                        help="Path to .bval file")
    parser.add_argument("--bvec",  required=True,
                        help="Path to .bvec file")
    parser.add_argument("--model", default="ARGO.pt",
                        help="Path to ARGO.pt weights file (default: ARGO.pt)")
    parser.add_argument("--out",   default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--fa_threshold", type=float, default=0.3,
                        help="FA threshold for WM graph (default: 0.3)")
    parser.add_argument("--subject_id", default=None,
                        help="Subject ID for output filenames "
                             "(default: derived from --nii filename)")
    args = parser.parse_args()

    # Subject ID from filename if not provided
    subject_id = args.subject_id or \
        os.path.basename(args.nii).replace('.nii.gz', '').replace('.nii', '')

    print()
    print("=" * 60)
    print("  ARGO — Angular Resolution Graph Operator")
    print("  [Author et al.] (2025), NeuroImage (under review)")
    print("=" * 60)
    print(f"  Subject:  {subject_id}")
    print(f"  Input:    {args.nii}")
    print(f"  Model:    {args.model}")
    print(f"  Output:   {args.out}/")
    print()

    t_total = time.time()

    # Load model and scalers
    print("  Loading ARGO model...")
    ckpt     = torch.load(args.model, map_location='cpu', weights_only=False)
    model    = _ARGO()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scaler_x = ckpt['scalers']['scaler_x']
    scaler_y = ckpt['scalers']['scaler_y']
    print(f"  Model loaded  |  Parameters: {ckpt['n_params']:,}  "
          f"|  Version: {ckpt['version']}")
    print()

    # Pipeline
    data_up, gtab, mask, affine, header = _preprocess(
        args.nii, args.bval, args.bvec)

    features, sh_low_v, coords, mask_shape = _extract_features(
        data_up, gtab, mask)

    graph, wm_mask, coords_wm = _build_graph(
        features, coords, mask_shape, args.fa_threshold, scaler_x)

    pred_norm, inference_time = _run_inference(model, graph)

    npz_path, nii_path = _save_outputs(
        pred_norm, scaler_y, wm_mask, coords_wm,
        mask_shape, affine, header, args.out, subject_id)

    total_time = time.time() - t_total
    print()
    print("=" * 60)
    print(f"  Done in {total_time:.1f}s  "
          f"(model inference: {inference_time:.1f}s, "
          f"{inference_time/total_time*100:.1f}% of total)")
    print(f"  WM voxels processed: {wm_mask.sum():,}")
    print()
    print("  Outputs:")
    print(f"    {npz_path}")
    print(f"    {nii_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
