"""
ARGO — run_evaluation.py
=========================
Evaluate ARGO predictions against 64-direction CSD reference.

Computes per-voxel Angular Error (AE), Peak Overlap (PO), and fraction
of voxels below 10° across three white matter FA strata.

Usage:
    python run_evaluation.py \
        --pred  output/sub-10347_fod_predicted.npz \
        --nii64 sample_data/sub-10347_dwi64.nii.gz \
        --bval  sample_data/sub-10347.bval \
        --bvec  sample_data/sub-10347.bvec

The .bval/.bvec files should correspond to the full 64-direction scheme.
The predicted .npz is the output of run_inference.py.
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import nibabel as nib
import torch
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import (auto_response_ssst,
                                    ConstrainedSphericalDeconvModel)
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
from scipy.ndimage import zoom, binary_erosion

# ──────────────────────────────────────────────────────────────────
# Sphere and SH-to-SF matrix (shared across metric functions)
# ──────────────────────────────────────────────────────────────────
_sphere = get_sphere('repulsion724')
_B, _   = sh_to_sf_matrix(_sphere, sh_order=8, basis_type='tournier07')

FA_BANDS = [
    ('Low WM  (FA 0.3–0.5)', 0.3, 0.5),
    ('High WM (FA 0.5–0.7)', 0.5, 0.7),
    ('Core WM (FA ≥ 0.7)',   0.7, 9.9),
    ('All WM  (FA ≥ 0.5)',   0.5, 9.9),
]
PRIMARY_STRATUM = 'All WM  (FA ≥ 0.5)'


# ──────────────────────────────────────────────────────────────────
# Metric functions (identical to paper implementation)
# ──────────────────────────────────────────────────────────────────

def _sf(sh):
    return np.clip(sh @ _B, 0, None)

def _primary_peaks(sh):
    return _sphere.vertices[np.argmax(_sf(sh), axis=1)]

def _angular_error(sh_pred, sh_ref):
    p1 = _primary_peaks(sh_pred)
    p2 = _primary_peaks(sh_ref)
    return np.degrees(
        np.arccos(np.clip(np.abs(np.sum(p1 * p2, axis=1)), 0, 1))
    )

def _peak_overlap(sh_pred, sh_ref, max_vox=2000, angle_thr=25.0):
    """
    Peak Overlap: fraction of reference FOD peaks recovered within
    angle_thr degrees. Up to 3 peaks per voxel. Evaluated on a
    random subsample of max_vox voxels (seed=0).
    """
    thresh = np.cos(np.radians(angle_thr))
    np.random.seed(0)
    idx  = np.random.choice(len(sh_pred),
                            min(max_vox, len(sh_pred)), replace=False)
    sp2  = sh_pred[idx]
    st2  = sh_ref[idx]

    def _get_peaks(row):
        peaks = []
        sc    = _sf(row[None])[0].copy()
        for _ in range(3):
            i = np.argmax(sc)
            if sc[i] < 1e-6:
                break
            peaks.append(_sphere.vertices[i])
            sc[np.abs(_sphere.vertices @ _sphere.vertices[i]) > thresh] = 0
        return peaks

    def _recall(est_peaks, ref_peaks):
        if not ref_peaks:
            return 1.0
        return sum(
            any(
                np.degrees(np.arccos(
                    np.clip(np.abs(np.dot(e, r)), 0, 1)
                )) < angle_thr
                for e in est_peaks
            )
            for r in ref_peaks
        ) / len(ref_peaks)

    return np.mean([
        _recall(_get_peaks(sp2[i]), _get_peaks(st2[i]))
        for i in range(len(sp2))
    ])


# ──────────────────────────────────────────────────────────────────
# Reference CSD computation
# ──────────────────────────────────────────────────────────────────

def _compute_reference(nii64_path, bval_path, bvec_path):
    """
    Compute 64-direction CSD reference FOD (SH order 8).
    Applies the same 1.5x in-plane upsampling as run_inference.py.
    Returns: sh_ref (N_wm_up, 45), fa_ref (N_wm_up,), coords_up (N_wm_up, 3)
    """
    print("  [1/3] Loading 64-direction DWI volume...")
    img  = nib.load(nii64_path)
    data = img.get_fdata().astype(np.float32)
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs=bvecs)

    print(f"  [1/3] Volume shape: {data.shape}  |  Directions: {len(bvals)}")

    print("  [2/3] Upsampling 1.5× in-plane (matching inference preprocessing)...")
    data_up = zoom(data, (1.5, 1.5, 1, 1), order=1)
    _, mask = median_otsu(data_up, vol_idx=[0], numpass=4, dilate=1)
    mask    = binary_erosion(mask, iterations=1).astype(bool)

    print("  [3/3] Running CSD on 64-direction data (this may take a few minutes)...")
    response, _ = auto_response_ssst(gtab, data_up,
                                     roi_radii=10, fa_thr=0.7)
    csd_fit = ConstrainedSphericalDeconvModel(gtab, response).fit(
        data_up, mask=mask)
    sh_ref_vol = csd_fit.shm_coeff

    # Also compute FA from DTI for FA stratification
    from dipy.reconst import dti
    tenfit = dti.TensorModel(gtab).fit(data_up, mask=mask)
    fa_vol = tenfit.fa

    sh_ref = sh_ref_vol[mask].astype(np.float32)
    fa_ref = fa_vol[mask].astype(np.float32)
    coords = np.array(np.where(mask)).T.astype(np.int16)

    print(f"  [3/3] Reference computed: {len(sh_ref):,} brain voxels")
    return sh_ref, fa_ref, coords, mask.shape


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARGO predictions against 64-direction CSD reference"
    )
    parser.add_argument("--pred",  required=True,
                        help="Path to .npz output from run_inference.py")
    parser.add_argument("--nii64", required=True,
                        help="Path to 64-direction DWI NIfTI (.nii or .nii.gz)")
    parser.add_argument("--bval",  required=True,
                        help="Path to .bval file (64-direction scheme)")
    parser.add_argument("--bvec",  required=True,
                        help="Path to .bvec file (64-direction scheme)")
    parser.add_argument("--out",   default=None,
                        help="Optional: save results as .txt (default: print only)")
    args = parser.parse_args()

    print()
    print("=" * 65)
    print("  ARGO — Evaluation against 64-direction CSD reference")
    print("=" * 65)
    print(f"  Predictions: {args.pred}")
    print(f"  Reference:   {args.nii64}")
    print()

    # Load predictions
    print("  Loading ARGO predictions...")
    npz        = np.load(args.pred)
    sh_pred    = npz['sh_coefficients'].astype(np.float32)
    coords_pred = npz['coords']
    mask_shape = tuple(npz['mask_shape'])
    wm_mask    = npz['wm_mask']
    print(f"  Predicted WM voxels: {len(sh_pred):,}")

    # Compute reference
    sh_ref_all, fa_ref_all, coords_ref, _ = _compute_reference(
        args.nii64, args.bval, args.bvec)

    # Align predictions and reference on shared WM voxels
    # Build coordinate lookup for reference
    print("\n  Aligning predictions and reference on shared WM coordinates...")
    ref_map = {}
    for i, c in enumerate(coords_ref):
        ref_map[tuple(c)] = i

    shared_pred_idx = []
    shared_ref_idx  = []
    for i, c in enumerate(coords_pred):
        key = tuple(c)
        if key in ref_map:
            shared_pred_idx.append(i)
            shared_ref_idx.append(ref_map[key])

    shared_pred_idx = np.array(shared_pred_idx)
    shared_ref_idx  = np.array(shared_ref_idx)

    sh_pred_shared = sh_pred[shared_pred_idx]
    sh_ref_shared  = sh_ref_all[shared_ref_idx]
    fa_shared      = fa_ref_all[shared_ref_idx]

    print(f"  Shared WM voxels: {len(sh_pred_shared):,}")

    # Also compute baseline (12-dir CSD = low-order SH from prediction input)
    # Note: baseline is the first 45 dims of the input features (already SH)
    # We load these from the wm_mask info — not available here directly.
    # We report AE for ARGO predictions only; baseline requires run_inference
    # intermediate data. For a full comparison, run both on the same subject.

    # Evaluate per FA stratum
    print()
    print("=" * 65)
    print(f"  {'FA Stratum':<24} {'N vox':>8} {'AE (°)':>9} "
          f"{'PO':>8} {'<10°':>7}")
    print("  " + "-" * 63)

    results = {}
    for label, fa_lo, fa_hi in FA_BANDS:
        m = (fa_shared >= fa_lo) & (fa_shared < fa_hi)
        n = int(m.sum())
        if n == 0:
            print(f"  {label:<24} {'0':>8}  — no voxels")
            continue

        sp = sh_pred_shared[m]
        st = sh_ref_shared[m]

        ae_vals = _angular_error(sp, st)
        po      = _peak_overlap(sp, st)
        p10     = float((ae_vals < 10).mean() * 100)
        ae_mean = float(ae_vals.mean())
        ae_std  = float(ae_vals.std())

        primary = " ◀" if label == PRIMARY_STRATUM else ""
        print(f"  {label:<24} {n:>8,} {ae_mean:>7.2f}°"
              f" {po:>9.3f} {p10:>6.1f}%{primary}")

        results[label] = dict(n=n, ae_mean=ae_mean, ae_std=ae_std,
                               po=po, p10=p10)

    print("=" * 65)
    print()

    # Primary stratum summary
    if PRIMARY_STRATUM in results:
        r = results[PRIMARY_STRATUM]
        print(f"  Primary stratum (FA ≥ 0.5):")
        print(f"    AE  = {r['ae_mean']:.2f}°")
        print(f"    PO  = {r['po']:.3f}")
        print(f"    <10° = {r['p10']:.1f}% of WM voxels")
        print()
        print(f"  Expected for sub-10347: AE ≈ 14.96°, PO ≈ 0.807")
        print()

    # Optional save
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write("ARGO Evaluation Results\n")
            f.write("=" * 65 + "\n")
            f.write(f"Predictions: {args.pred}\n")
            f.write(f"Reference:   {args.nii64}\n\n")
            for label, r in results.items():
                f.write(f"{label}\n")
                f.write(f"  N={r['n']:,}  AE={r['ae_mean']:.2f}±{r['ae_std']:.2f}°  "
                        f"PO={r['po']:.3f}  <10°={r['p10']:.1f}%\n")
        print(f"  Results saved to: {args.out}")


if __name__ == "__main__":
    main()
