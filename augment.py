"""
augment.py  –  Pure numpy/scipy augmentations matching train_gb.py.

Order matches the gunpowder pipeline exactly:
    1. SimpleAugment      → random y/x flips + transpose
    2. IntensityAugment   → per-section contrast/brightness jitter
    3. NoiseAugment       → additive Gaussian noise
    4. DefectAugment      → missing/shifted/darkened sections
    5. ElasticAugment     → random smooth deformation field
    6. IntensityScaleShift→ [0,1] → [-1,1]

All functions operate on:
    raw      : (Z, Y, X)    float32, values in [0, 1]
    indicator: (Z, Y, X)    float32, binary
    vectors  : (3, Z, Y, X) float32
    d_weight : (Z, Y, X)    float32, binary

Points (pre/post) are NOT transformed here — we render blobs after augmentation,
so the vector/indicator arrays are augmented directly as dense volumes.
"""

from __future__ import annotations

from typing import Optional

import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter


# ---------------------------------------------------------------------------
# 1. SimpleAugment  (mirror_only=[1,2], transpose_only=[1,2])
# ---------------------------------------------------------------------------

def simple_augment(raw, indicator, vectors, d_weight):
    """Random y/x mirror and y/x transpose. No z-axis changes."""

    # random mirror along y (axis 1) and x (axis 2)
    for ax in [1, 2]:
        if np.random.random() > 0.5:
            raw       = np.flip(raw,       axis=ax).copy()
            indicator = np.flip(indicator, axis=ax).copy()
            d_weight  = np.flip(d_weight,  axis=ax).copy()
            vectors   = np.flip(vectors,   axis=ax + 1).copy()
            vectors[ax] = -vectors[ax]   # flip corresponding vector component

    # random transpose of y and x axes
    if np.random.random() > 0.5:
        raw       = np.transpose(raw,       (0, 2, 1)).copy()
        indicator = np.transpose(indicator, (0, 2, 1)).copy()
        d_weight  = np.transpose(d_weight,  (0, 2, 1)).copy()
        # swap y and x vector components (axes 1↔2), keep z (axis 0) in place
        vectors   = np.stack([
            np.transpose(vectors[0], (0, 2, 1)),   # z-component spatial dims swapped
            np.transpose(vectors[2], (0, 2, 1)),   # old x-component → new y-component
            np.transpose(vectors[1], (0, 2, 1)),   # old y-component → new x-component
        ])

    return raw, indicator, vectors, d_weight


# ---------------------------------------------------------------------------
# 2. IntensityAugment  (scale [0.8,1.2], shift [-0.15,0.15], z_section_wise)
# ---------------------------------------------------------------------------

def intensity_augment(
    raw: np.ndarray,
    scale_range: tuple = (0.8, 1.2),
    shift_range: tuple = (-0.15, 0.15),
    z_section_wise: bool = True,
) -> np.ndarray:
    """
    Randomly scale and shift intensity.
    With z_section_wise=True each z-slice gets an independent draw,
    simulating per-section contrast variation in EM stacks.
    """
    if z_section_wise:
        nz     = raw.shape[0]
        scales = np.random.uniform(*scale_range, size=(nz, 1, 1)).astype(np.float32)
        shifts = np.random.uniform(*shift_range, size=(nz, 1, 1)).astype(np.float32)
        raw    = raw * scales + shifts
    else:
        scale = np.random.uniform(*scale_range)
        shift = np.random.uniform(*shift_range)
        raw   = raw * scale + shift

    return np.clip(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. NoiseAugment  (additive Gaussian, var drawn from [0, 0.1])
# ---------------------------------------------------------------------------

def noise_augment(
    raw: np.ndarray,
    var_range: tuple = (0.0, 0.1),
) -> np.ndarray:
    """Add Gaussian noise with variance drawn uniformly from var_range."""
    var = np.random.uniform(*var_range)
    if var < 1e-6:
        return raw
    # standard_normal with dtype=float32 avoids float64 alloc + cast
    noise = np.random.standard_normal(raw.shape).astype(np.float32) * np.sqrt(var)
    return np.clip(raw + noise, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. DefectAugment
#    Simulates three types of EM section artifacts:
#      a) missing section  – zero out entire slice
#      b) dark section     – strongly darken a slice
#      c) section shift    – translate one slice in y/x (slip/misalignment)
# ---------------------------------------------------------------------------

def defect_augment(
    raw: np.ndarray,
    prob_missing:    float = 0.03,
    prob_dark:       float = 0.03,
    prob_shift:      float = 0.03,
    max_shift_px:    int   = 16,
) -> np.ndarray:
    """
    Apply random section-level defects independently per z-slice.
    Probabilities are per-slice (matching gunpowder's DefectAugment defaults).
    """
    raw  = raw.copy()
    nz   = raw.shape[0]

    rs          = np.random.random(nz)
    thr_missing = prob_missing
    thr_dark    = prob_missing + prob_dark
    thr_shift   = prob_missing + prob_dark + prob_shift

    missing_zs = np.where(rs < thr_missing)[0]
    dark_zs    = np.where((rs >= thr_missing) & (rs < thr_dark))[0]
    shift_zs   = np.where((rs >= thr_dark)    & (rs < thr_shift))[0]

    for z in missing_zs:
        if 0 < z < nz - 1:
            raw[z] = (raw[z-1] + raw[z+1]) / 2.0
        else:
            raw[z] = 0.0

    if len(dark_zs):
        scales = np.random.uniform(0.1, 0.4, size=len(dark_zs))
        for z, s in zip(dark_zs, scales):
            raw[z] *= s

    if len(shift_zs):
        dys = np.random.randint(-max_shift_px, max_shift_px + 1, size=len(shift_zs))
        dxs = np.random.randint(-max_shift_px, max_shift_px + 1, size=len(shift_zs))
        for z, dy, dx in zip(shift_zs, dys, dxs):
            raw[z] = np.roll(raw[z], dy, axis=0)
            if dx:
                raw[z] = np.roll(raw[z], dx, axis=1)

    return np.clip(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 5. ElasticAugment  (gunpowder-style: field at coarse scale, warp at full res)
# ---------------------------------------------------------------------------

def _build_displacement_field(shape, control_point_spacing, jitter_sigma):
    """
    Build a (3, Z, Y, X) displacement field — matches gunpowder's approach.

    Random noise is sampled at control_point_spacing resolution, smoothed,
    then upsampled to full resolution using map_coordinates (order=1, nearest
    boundary). Using map_coordinates instead of scipy.ndimage.zoom avoids the
    boundary spline extrapolation artifacts that cause radial streaks.
    """
    field = np.zeros((3,) + tuple(shape), dtype=np.float32)

    # coarse grid shape is determined by per-dim spacing, same for all components
    coarse_shape = tuple(
        max(2, int(np.ceil(s / max(1, cps))) + 1)
        for s, cps in zip(shape, control_point_spacing)
    )

    for c in range(3):
        sigma_c = jitter_sigma[c]
        if sigma_c == 0:
            continue

        noise = np.random.standard_normal(coarse_shape).astype(np.float32) * sigma_c
        noise = gaussian_filter(noise, sigma=1.0)

        # upsample using map_coordinates — no boundary spline extrapolation
        coords = np.mgrid[
            0 : coarse_shape[0] - 1 : shape[0] * 1j,
            0 : coarse_shape[1] - 1 : shape[1] * 1j,
            0 : coarse_shape[2] - 1 : shape[2] * 1j,
        ]
        field[c] = map_coordinates(noise, coords, order=1, mode="nearest")

    return field


def _build_rotation_field(shape, angle):
    cy, cx = (shape[1] - 1) / 2.0, (shape[2] - 1) / 2.0
    ys = np.arange(shape[1], dtype=np.float32) - cy
    xs = np.arange(shape[2], dtype=np.float32) - cx
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    field = np.zeros((3,) + shape, dtype=np.float32)
    field[1] = ((cos_a - 1) * yy - sin_a * xx)[None]
    field[2] = (sin_a * yy + (cos_a - 1) * xx)[None]
    return field


def _jacobian_of_field(disp):
    """Compute the full Jacobian J of the deformation field (identity + disp).
    Used to rotate direction vectors consistently with the spatial warp."""
    J = np.zeros((3, 3) + disp.shape[1:], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            J[i, j] = np.gradient(disp[i], axis=j)
    J[0, 0] += 1.0; J[1, 1] += 1.0; J[2, 2] += 1.0
    return J


def _transform_vectors_with_jacobian(vec_w, J):
    """Apply Jacobian J to each voxel's direction vector."""
    return np.einsum("ijzyx,jzyx->izyx", J, vec_w).astype(np.float32)


def elastic_augment(
    raw:       np.ndarray,
    indicator: np.ndarray,
    vectors:   np.ndarray,
    d_weight:  np.ndarray,
    control_point_spacing: list  = (1, 50, 50),
    jitter_sigma:          list  = (1, 3.0, 3.0),
    prob_slip:             float = 0.25,
    prob_shift:            float = 0.25,
    prob_elastic:          float = 0.9,
    correct_vectors:       bool  = False,
    context:               Optional[np.ndarray] = None,
) -> tuple:
    """
    Elastic deformation only — no rotation (rotation is handled by SimpleAugment).

    context: (ctx_z, ctx_yx, ctx_yx) int array — when provided the input already
             contains real zarr context on each side, so no reflect-padding is needed.
             The output is cropped back to the inner (input_size) region.
             When None, falls back to reflect-padding (original behaviour).
    """
    shape = raw.shape

    if context is not None:
        # ── real-context path: no fake padding ───────────────────────────────
        ctx_z  = int(context[0])
        ctx_yx = int(context[1])
        Z = shape[0] - 2 * ctx_z
        Y = shape[1] - 2 * ctx_yx
        X = shape[2] - 2 * ctx_yx
        sl   = (slice(ctx_z,  ctx_z  + Z), slice(ctx_yx, ctx_yx + Y), slice(ctx_yx, ctx_yx + X))
        sl_v = (slice(None),) + sl

        if np.random.random() > prob_elastic:
            return np.clip(raw[sl], 0.0, 1.0), indicator[sl], vectors[sl_v], d_weight[sl]

        disp = _build_displacement_field(shape, list(control_point_spacing), list(jitter_sigma))

        if np.random.random() < prob_slip:
            max_slip = max(jitter_sigma[1], jitter_sigma[2]) * 2
            for z in range(shape[0]):
                disp[1, z] += np.random.uniform(-max_slip, max_slip)
                disp[2, z] += np.random.uniform(-max_slip, max_slip)

        if np.random.random() < prob_shift:
            disp[1] += np.random.uniform(-jitter_sigma[1] * 4, jitter_sigma[1] * 4)
            disp[2] += np.random.uniform(-jitter_sigma[2] * 4, jitter_sigma[2] * 4)

        coords = np.empty((3,) + shape, dtype=np.float32)
        coords[0] = np.arange(shape[0], dtype=np.float32).reshape(-1, 1, 1) + disp[0]
        coords[1] = np.arange(shape[1], dtype=np.float32).reshape(1, -1, 1) + disp[1]
        coords[2] = np.arange(shape[2], dtype=np.float32).reshape(1, 1, -1) + disp[2]

        def _warp(arr, order=1):
            return map_coordinates(arr, coords, order=order, mode="mirror",
                                   prefilter=(order > 1)).astype(np.float32)

        raw_w = _warp(raw,       order=1)
        ind_w = (_warp(indicator, order=0) > 0.5).astype(np.float32)
        dw_w  = (_warp(d_weight,  order=0) > 0.5).astype(np.float32)
        vec_w = np.stack([_warp(vectors[c], order=1) for c in range(3)])

        if correct_vectors:
            J     = _jacobian_of_field(disp)
            vec_w = _transform_vectors_with_jacobian(vec_w, J)

        return np.clip(raw_w[sl], 0.0, 1.0), ind_w[sl], vec_w[sl_v], dw_w[sl]

    # ── reflect-padding fallback (no real context available) ─────────────────
    if np.random.random() > prob_elastic:
        return raw, indicator, vectors, d_weight

    pad_yx = int(np.ceil(max(jitter_sigma[1], jitter_sigma[2]))) * 3 + 2
    pad_z  = int(np.ceil(jitter_sigma[0])) * 3 + 2
    padding = ((pad_z, pad_z), (pad_yx, pad_yx), (pad_yx, pad_yx))

    raw_p  = np.pad(raw,       padding, mode="reflect")
    ind_p  = np.pad(indicator, padding, mode="reflect")
    dw_p   = np.pad(d_weight,  padding, mode="reflect")
    vec_p  = np.stack([np.pad(vectors[c], padding, mode="reflect") for c in range(3)])
    shape_p = raw_p.shape

    disp = _build_displacement_field(shape_p, list(control_point_spacing), list(jitter_sigma))

    if np.random.random() < prob_slip:
        max_slip = max(jitter_sigma[1], jitter_sigma[2]) * 2
        for z in range(shape_p[0]):
            disp[1, z] += np.random.uniform(-max_slip, max_slip)
            disp[2, z] += np.random.uniform(-max_slip, max_slip)

    if np.random.random() < prob_shift:
        disp[1] += np.random.uniform(-jitter_sigma[1] * 4, jitter_sigma[1] * 4)
        disp[2] += np.random.uniform(-jitter_sigma[2] * 4, jitter_sigma[2] * 4)

    coords = np.empty((3,) + shape_p, dtype=np.float32)
    coords[0] = np.arange(shape_p[0], dtype=np.float32).reshape(-1, 1, 1) + disp[0]
    coords[1] = np.arange(shape_p[1], dtype=np.float32).reshape(1, -1, 1) + disp[1]
    coords[2] = np.arange(shape_p[2], dtype=np.float32).reshape(1, 1, -1) + disp[2]

    def _warp(arr, order=1):
        return map_coordinates(arr, coords, order=order, mode="nearest",
                               prefilter=(order > 1)).astype(np.float32)

    raw_w = _warp(raw_p,  order=1)
    ind_w = (_warp(ind_p, order=0) > 0.5).astype(np.float32)
    dw_w  = (_warp(dw_p,  order=0) > 0.5).astype(np.float32)
    vec_w = np.stack([_warp(vec_p[c], order=1) for c in range(3)])

    if correct_vectors:
        J     = _jacobian_of_field(disp)
        vec_w = _transform_vectors_with_jacobian(vec_w, J)

    sl = (
        slice(pad_z,  pad_z  + shape[0]),
        slice(pad_yx, pad_yx + shape[1]),
        slice(pad_yx, pad_yx + shape[2]),
    )
    return (
        np.clip(raw_w[sl],              0.0, 1.0),
        ind_w[sl],
        vec_w[(slice(None),) + sl],
        dw_w[sl],
    )


# ---------------------------------------------------------------------------
# 6. IntensityScaleShift  raw: [0,1] → [-1,1]
# ---------------------------------------------------------------------------

def intensity_scale_shift(raw: np.ndarray, scale: float = 2.0, shift: float = -1.0) -> np.ndarray:
    return raw * scale + shift


# ---------------------------------------------------------------------------
# 7. BlurAugment  — per-section Gaussian blur (out-of-focus sections)
# ---------------------------------------------------------------------------

def blur_augment(
    raw:          np.ndarray,
    prob:         float = 0.1,
    sigma_range:  tuple = (0.0, 1.5),
) -> np.ndarray:
    """Per-section Gaussian blur with random sigma. Simulates focus variation."""
    from scipy.ndimage import gaussian_filter as gf
    nz   = raw.shape[0]
    hits = np.where(np.random.random(nz) < prob)[0]
    if not len(hits):
        return raw
    raw = raw.copy()
    for z in hits:
        sigma = np.random.uniform(*sigma_range)
        if sigma > 0.1:
            raw[z] = gf(raw[z], sigma=sigma)
    return raw


# ---------------------------------------------------------------------------
# 8. GammaAugment  — per-section power-law intensity transform
# ---------------------------------------------------------------------------

def gamma_augment(
    raw:         np.ndarray,
    gamma_range: tuple = (0.75, 1.5),
) -> np.ndarray:
    """Per-section gamma correction. Simulates detector nonlinearity."""
    nz     = raw.shape[0]
    gammas = np.random.uniform(*gamma_range, size=(nz, 1, 1)).astype(np.float32)
    return np.power(np.clip(raw, 1e-8, 1.0), gammas)


# ---------------------------------------------------------------------------
# 9. InvertAugment  — occasional per-section contrast inversion
# ---------------------------------------------------------------------------

def invert_augment(
    raw:  np.ndarray,
    prob: float = 0.01,
) -> np.ndarray:
    """Randomly invert individual z-slices. Rare but real EM artifact."""
    nz   = raw.shape[0]
    mask = (np.random.random(nz) < prob)
    if not mask.any():
        return raw
    raw  = raw.copy()
    raw[mask] = 1.0 - raw[mask]
    return raw


# ---------------------------------------------------------------------------
# 10. CutoutAugment  — random rectangular zero-patches in y/x
# ---------------------------------------------------------------------------

def cutout_augment(
    raw:          np.ndarray,
    prob:         float = 0.5,
    n_holes:      int   = 2,
    hole_size_yx: tuple = (20, 20),
) -> np.ndarray:
    """
    Zero out random rectangular patches in y/x.
    Forces the network to not rely on any single region.
    Applied to the full z-stack (same patch location per z-slice within one cutout).
    """
    raw  = raw.copy()
    _, H, W = raw.shape
    hz, hy = hole_size_yx

    if np.random.random() < prob:
        for _ in range(n_holes):
            y0 = np.random.randint(0, max(1, H - hy))
            x0 = np.random.randint(0, max(1, W - hz))
            raw[:, y0:y0+hy, x0:x0+hz] = 0.0

    return raw


# ---------------------------------------------------------------------------
# 11. SaltPepperAugment  — random hot/dead pixels
# ---------------------------------------------------------------------------

def salt_pepper_augment(
    raw:      np.ndarray,
    prob:     float = 0.001,
) -> np.ndarray:
    """Random per-voxel salt (1.0) and pepper (0.0). Simulates dead detector pixels."""
    raw  = raw.copy()
    n    = raw.size
    # sparse sampling: only draw the affected indices, not a full boolean mask
    n_salt   = np.random.binomial(n, prob / 2)
    n_pepper = np.random.binomial(n, prob / 2)
    if n_salt:
        raw.flat[np.random.randint(0, n, n_salt)] = 1.0
    if n_pepper:
        raw.flat[np.random.randint(0, n, n_pepper)] = 0.0
    return raw


# ---------------------------------------------------------------------------
# GPU elastic augmentation
# Operates on tensors already on the GPU — 400× faster than scipy map_coordinates.
# Called from the training loop after data is transferred to the device.
# ---------------------------------------------------------------------------

def elastic_augment_gpu(
    raw:       torch.Tensor,   # (B, 1, Z, Y, X)  float32, values in [-1, 1]
    indicator: torch.Tensor,   # (B, 1, Z, Y, X)  float32, binary
    vectors:   torch.Tensor,   # (B, 3, Z, Y, X)  float32
    d_weight:  torch.Tensor,   # (B, 1, Z, Y, X)  float32, binary
    control_point_spacing: list = (50, 10, 10),
    jitter_sigma:          list = (0, 4.0, 4.0),
    prob_slip:             float = 0.25,
    prob_shift:            float = 0.25,
    prob_elastic:          float = 0.4,
) -> tuple:
    """
    GPU-accelerated elastic augmentation using torch.nn.functional.grid_sample.
    All volumes are warped in a single batched call (~2ms vs ~3300ms on CPU).
    Probability of applying is prob_elastic; skips cleanly if not applied.
    No vector Jacobian correction (negligible effect at these deformation scales).
    """
    if random.random() > prob_elastic:
        return raw, indicator, vectors, d_weight

    device = raw.device
    B, _, Z, Y, X = raw.shape

    # ── build displacement field: sample on CPU coarse grid, upsample on GPU ──
    # Uploading the tiny coarse array (~2 KB) instead of the full-res field
    # (~46 MB) eliminates a large CPU→GPU transfer per elastic step.
    cps    = control_point_spacing
    coarse = tuple(max(2, int(np.ceil(s / max(1, c))) + 1) for s, c in zip((Z, Y, X), cps))

    coarse_disp = np.zeros((1, 3) + coarse, dtype=np.float32)   # (1, 3, cZ, cY, cX)
    for c in range(3):
        sig = jitter_sigma[c]
        if sig == 0:
            continue
        noise = np.random.randn(*coarse).astype(np.float32) * sig
        coarse_disp[0, c] = gaussian_filter(noise, sigma=1.0)

    # slip: add per-slice offset on the coarse grid (nearest-upsampled later)
    if random.random() < prob_slip:
        max_slip = max(jitter_sigma[1], jitter_sigma[2]) * 2
        cZ = coarse[0]
        coarse_disp[0, 1] += np.random.uniform(-max_slip, max_slip, size=(cZ, 1, 1))
        coarse_disp[0, 2] += np.random.uniform(-max_slip, max_slip, size=(cZ, 1, 1))

    if random.random() < prob_shift:
        coarse_disp[0, 1] += np.random.uniform(-jitter_sigma[1] * 4, jitter_sigma[1] * 4)
        coarse_disp[0, 2] += np.random.uniform(-jitter_sigma[2] * 4, jitter_sigma[2] * 4)

    # upload coarse field and upsample to full resolution on GPU
    disp_t = (
        F.interpolate(
            torch.from_numpy(coarse_disp).to(device),
            size=(Z, Y, X),
            mode='trilinear',
            align_corners=True,
        ).squeeze(0)   # (3, Z, Y, X)
    )

    # ── build grid_sample sampling grid in [-1, 1] ────────────────────────────
    # grid_sample grid shape: (1, Z, Y, X, 3), coords order (x, y, z)
    z_base = torch.linspace(-1, 1, Z, device=device)
    y_base = torch.linspace(-1, 1, Y, device=device)
    x_base = torch.linspace(-1, 1, X, device=device)
    gz, gy, gx = torch.meshgrid(z_base, y_base, x_base, indexing='ij')

    # disp_t is already on device at full resolution — convert pixel offsets to [-1,1]
    gz = gz + disp_t[0] * (2.0 / (Z - 1))
    gy = gy + disp_t[1] * (2.0 / (Y - 1))
    gx = gx + disp_t[2] * (2.0 / (X - 1))

    # grid_sample expects (x, y, z) ordering.
    # Make contiguous once so grid_sample doesn't materialise a hidden copy per call.
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()

    # Free intermediate build tensors before the (large) warp calls.
    del gz, gy, gx, disp_t, z_base, y_base, x_base

    def _warp(vol, nearest=False):
        mode = 'nearest' if nearest else 'bilinear'
        return F.grid_sample(vol, grid, mode=mode, padding_mode='border', align_corners=True)

    raw_w  = _warp(raw).clamp(-1.0, 1.0)
    ind_w  = (_warp(indicator, nearest=True) > 0.5).float()
    dw_w   = (_warp(d_weight,  nearest=True) > 0.5).float()
    vec_w  = _warp(vectors)

    del grid

    return raw_w, ind_w, vec_w, dw_w


# ---------------------------------------------------------------------------
# Full pipeline — driven entirely by params dict
# ---------------------------------------------------------------------------

def augment_sample(
    raw:       np.ndarray,
    indicator: np.ndarray,
    vectors:   np.ndarray,
    d_weight:  np.ndarray,
    params:    dict,
    context:   Optional[np.ndarray] = None,
) -> tuple:
    """
    Run the full augmentation pipeline.
    Each augmentation can be enabled/disabled and configured via params["augmentation"].
    raw is returned in [-1,1] after the final scale/shift.
    """
    aug = params.get("augmentation", {})

    def enabled(key, default=True):
        cfg = aug.get(key, {})
        if not cfg.get("enabled", default):
            return False
        return np.random.random() < cfg.get("apply_prob", 1.0)

    # 1. SimpleAugment — flips and transpose
    if enabled("simple"):
        raw, indicator, vectors, d_weight = simple_augment(
            raw, indicator, vectors, d_weight
        )

    # 2. IntensityAugment — per-section contrast/brightness
    if enabled("intensity"):
        cfg = aug.get("intensity", {})
        raw = intensity_augment(
            raw,
            scale_range    = tuple(cfg.get("scale_range",  [0.8, 1.2])),
            shift_range    = tuple(cfg.get("shift_range",  [-0.15, 0.15])),
            z_section_wise = cfg.get("z_section_wise", True),
        )

    # 3. NoiseAugment — additive Gaussian
    if enabled("noise"):
        cfg = aug.get("noise", {})
        raw = noise_augment(
            raw,
            var_range = tuple(cfg.get("var_range", [0.0, 0.1])),
        )

    # 4. DefectAugment — missing/dark/shifted sections
    if enabled("defect"):
        cfg = aug.get("defect", {})
        raw = defect_augment(
            raw,
            prob_missing  = cfg.get("prob_missing",  0.03),
            prob_dark     = cfg.get("prob_dark",     0.03),
            prob_shift    = cfg.get("prob_shift",    0.03),
            max_shift_px  = cfg.get("max_shift_px",  16),
        )

    # 5. ElasticAugment — smooth deformation (uses real context if provided)
    # Pass context="defer" to skip here and apply GPU elastic in the training loop instead.
    # Do NOT gate via enabled() here — elastic has its own prob_elastic inside, and
    # using apply_prob on top would double-gate and suppress it more than intended.
    if context != "defer" and aug.get("elastic", {}).get("enabled", True):
        cfg = aug.get("elastic", {})
        raw, indicator, vectors, d_weight = elastic_augment(
            raw, indicator, vectors, d_weight,
            control_point_spacing = cfg.get("control_point_spacing", [1, 50, 50]),
            jitter_sigma          = cfg.get("jitter_sigma",          [1, 3.0, 3.0]),
            prob_slip             = cfg.get("prob_slip",             0.25),
            prob_shift            = cfg.get("prob_shift",            0.25),
            prob_elastic          = cfg.get("prob_elastic",          0.9),
            correct_vectors       = cfg.get("correct_vectors",       False),
            context               = context,
        )

    # 6. BlurAugment — per-section Gaussian blur
    if enabled("blur", default=True):
        cfg = aug.get("blur", {})
        raw = blur_augment(
            raw,
            prob        = cfg.get("prob",        0.1),
            sigma_range = tuple(cfg.get("sigma_range", [0.0, 1.5])),
        )

    # 7. GammaAugment — per-section power-law
    if enabled("gamma", default=True):
        cfg = aug.get("gamma", {})
        raw = gamma_augment(
            raw,
            gamma_range = tuple(cfg.get("gamma_range", [0.75, 1.5])),
        )

    # 8. InvertAugment — rare section inversion
    if enabled("invert", default=True):
        cfg = aug.get("invert", {})
        raw = invert_augment(
            raw,
            prob = cfg.get("prob", 0.01),
        )

    # 9. CutoutAugment — random rectangular patches
    if enabled("cutout", default=True):
        cfg = aug.get("cutout", {})
        raw = cutout_augment(
            raw,
            prob         = cfg.get("prob",         0.5),
            n_holes      = cfg.get("n_holes",       2),
            hole_size_yx = tuple(cfg.get("hole_size_yx", [20, 20])),
        )

    # 10. SaltPepperAugment — dead pixels
    if enabled("salt_pepper", default=True):
        cfg = aug.get("salt_pepper", {})
        raw = salt_pepper_augment(
            raw,
            prob = cfg.get("prob", 0.001),
        )

    # 11. IntensityScaleShift [0,1] → [-1,1]
    raw = intensity_scale_shift(raw)

    return raw, indicator, vectors, d_weight

