"""
dataset.py  –  SynfulDataset matching train_gb.py conventions.

CSV layout (train_gb.py uses SafeCsvPointsSource with columns [0,1,2]):
    <zarr_stem>_pre.csv   – one synapse per row, columns: z, y, x  (0-indexed)
    <zarr_stem>_post.csv  – one synapse per row, columns: z, y, x  (0-indexed)

ROI resolution order (matching train_gb.py):
    1. params["rois"]     – explicit list of [[offset], [shape]] pairs
    2. params["json_dir"] – JSON files named <zarr_stem>.json
    3. fallback           – full zarr extent

Species filtering:
    If params["species"] is present, only zarrs whose stem starts with one of
    those species names (first underscore-delimited token) are loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from augment import augment_sample


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class Roi(NamedTuple):
    offset: np.ndarray   # ZYX voxel offset (absolute)
    shape:  np.ndarray   # ZYX voxel shape

    def end(self) -> np.ndarray:
        return self.offset + self.shape


# mirrors train_gb.py:  raw_off = roi.offset - [25,150,150]
RAW_CONTEXT = np.array([25, 150, 150], dtype=int)


# ---------------------------------------------------------------------------
# Blob rendering
# ---------------------------------------------------------------------------

def _ball_se(radius: List[int]) -> np.ndarray:
    rz, ry, rx = radius
    zz, yy, xx = np.mgrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    return (
        (zz / (rz + 0.5))**2 +
        (yy / (ry + 0.5))**2 +
        (xx / (rx + 0.5))**2
    ) <= 1.0


def _paste_blob(canvas, loc, se, half, value=1.0, channel_first=False):
    z, y, x = int(round(loc[0])), int(round(loc[1])), int(round(loc[2]))
    shape = canvas.shape[-3:] if channel_first else canvas.shape

    z0 = max(0, z-half[0]); z1 = max(0, min(shape[0], z+half[0]+1))
    y0 = max(0, y-half[1]); y1 = max(0, min(shape[1], y+half[1]+1))
    x0 = max(0, x-half[2]); x1 = max(0, min(shape[2], x+half[2]+1))
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return

    se_z0 = half[0]-(z-z0); se_z1 = se_z0+(z1-z0)
    se_y0 = half[1]-(y-y0); se_y1 = se_y0+(y1-y0)
    se_x0 = half[2]-(x-x0); se_x1 = se_x0+(x1-x0)

    blob = se[se_z0:se_z1, se_y0:se_y1, se_x0:se_x1]

    if channel_first:
        sub = canvas[:, z0:z1, y0:y1, x0:x1]
        v = (value[:,None,None,None]*blob[None]
             if isinstance(value, np.ndarray) else value*blob[None])
        canvas[:, z0:z1, y0:y1, x0:x1] = np.where(blob[None], v, sub)
    else:
        canvas[z0:z1, y0:y1, x0:x1] = np.maximum(
            canvas[z0:z1, y0:y1, x0:x1],
            blob.astype(canvas.dtype) * value,
        )


def render_syn_indicators(shape, post_locs, blob_radius):
    mask = np.zeros(shape, dtype=np.float32)
    if len(post_locs) == 0:
        return mask
    se = _ball_se(blob_radius)
    for loc in post_locs:
        _paste_blob(mask, loc, se, blob_radius, value=1.0)
    return mask


def render_direction_vectors(shape, post_locs, pre_locs, d_blob_radius, voxel_size=(1,1,1)):
    vectors     = np.zeros((3,)+shape, dtype=np.float32)
    weight_mask = np.zeros(shape,      dtype=np.float32)
    if len(post_locs) == 0:
        return vectors, weight_mask

    rz, ry, rx = d_blob_radius
    zz, yy, xx = np.mgrid[-rz:rz+1, -ry:ry+1, -rx:rx+1]
    norm_dist = np.sqrt(
        (zz / (rz + 0.5))**2 +
        (yy / (ry + 0.5))**2 +
        (xx / (rx + 0.5))**2
    )
    se = norm_dist <= 1.0

    vox = np.array(voxel_size, dtype=np.float32)

    canvas_shape = vectors.shape[1:]

    # Pass 1: collect per-synapse blobs and accumulate an overlap count map,
    # matching synful's AddPartnerVectorMap logic exactly.
    blob_masks  = []   # list of (z0,y0,x0, bool_slice) per synapse
    union_count = np.zeros(shape, dtype=np.int32)

    for post, pre in zip(post_locs, pre_locs):
        post_r = np.round(post).astype(int)

        z0 = max(0, post_r[0]-rz); z1 = min(canvas_shape[0], post_r[0]+rz+1)
        y0 = max(0, post_r[1]-ry); y1 = min(canvas_shape[1], post_r[1]+ry+1)
        x0 = max(0, post_r[2]-rx); x1 = min(canvas_shape[2], post_r[2]+rx+1)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            blob_masks.append(None)
            continue

        se_z0 = rz-(post_r[0]-z0); se_z1 = se_z0+(z1-z0)
        se_y0 = ry-(post_r[1]-y0); se_y1 = se_y0+(y1-y0)
        se_x0 = rx-(post_r[2]-x0); se_x1 = se_x0+(x1-x0)

        blob_sl = se[se_z0:se_z1, se_y0:se_y1, se_x0:se_x1].copy()
        union_count[z0:z1, y0:y1, x0:x1] += blob_sl.astype(np.int32)
        blob_masks.append((z0, y0, x0, blob_sl))

    # Pass 2: remove overlap voxels from all blobs, then reassign them to the
    # nearest post-synaptic centre (KDTree), matching synful's logic.
    overlap_voxels = np.argwhere(union_count > 1)  # (N, 3) canvas coords

    if len(overlap_voxels) and len(post_locs) > 1:
        post_world = np.array([
            np.round(p).astype(int) * vox for p in post_locs
        ], dtype=np.float32)   # (M, 3) world coords of each post centre

        from scipy.spatial import KDTree
        kd = KDTree(post_world)
        overlap_world = overlap_voxels.astype(np.float32) * vox  # (N, 3)
        _, nearest_idx = kd.query(overlap_world)

        for bm in blob_masks:
            if bm is None:
                continue
            z0, y0, x0, bsl = bm
            dz, dy, dx = bsl.shape
            # zero out overlap region in this blob
            local_ovlp = overlap_voxels[
                (overlap_voxels[:,0] >= z0) & (overlap_voxels[:,0] < z0+dz) &
                (overlap_voxels[:,1] >= y0) & (overlap_voxels[:,1] < y0+dy) &
                (overlap_voxels[:,2] >= x0) & (overlap_voxels[:,2] < x0+dx)
            ]
            for v in local_ovlp:
                bsl[v[0]-z0, v[1]-y0, v[2]-x0] = False

        # reassign each overlap voxel to its nearest post centre
        for ov, nidx in zip(overlap_voxels, nearest_idx):
            bm = blob_masks[nidx]
            if bm is None:
                continue
            z0, y0, x0, bsl = bm
            lz, ly, lx = ov[0]-z0, ov[1]-y0, ov[2]-x0
            if 0 <= lz < bsl.shape[0] and 0 <= ly < bsl.shape[1] and 0 <= lx < bsl.shape[2]:
                bsl[lz, ly, lx] = True

    # Pass 3: write vectors and weight mask using the corrected blobs.
    for i, (post, pre) in enumerate(zip(post_locs, pre_locs)):
        bm = blob_masks[i]
        if bm is None:
            continue
        z0, y0, x0, blob_sl = bm
        z1, y1, x1 = z0+blob_sl.shape[0], y0+blob_sl.shape[1], x0+blob_sl.shape[2]

        vox_pos_z = np.arange(z0, z1, dtype=np.float32) * vox[0]
        vox_pos_y = np.arange(y0, y1, dtype=np.float32) * vox[1]
        vox_pos_x = np.arange(x0, x1, dtype=np.float32) * vox[2]
        vox_world = np.stack(np.meshgrid(vox_pos_z, vox_pos_y, vox_pos_x, indexing='ij'), axis=0)

        pre_world     = pre.astype(np.float32) * vox
        per_voxel_vec = pre_world[:, None, None, None] - vox_world

        mask3 = blob_sl[None]
        vectors[:, z0:z1, y0:y1, x0:x1] = np.where(
            mask3, per_voxel_vec, vectors[:, z0:z1, y0:y1, x0:x1]
        )
        weight_mask[z0:z1, y0:y1, x0:x1] = np.maximum(
            weight_mask[z0:z1, y0:y1, x0:x1], blob_sl.astype(np.float32)
        )

    return vectors, weight_mask


# ---------------------------------------------------------------------------
# CSV loading  (index-based: columns 0,1,2 = z,y,x — no named headers)
# ---------------------------------------------------------------------------

def _csv_has_data(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path) as fh:
        lines = fh.readlines()
    return len(lines) > 1


def load_points_csv(path: str, delimiter: str = ",") -> np.ndarray:
    """
    Load ZYX point coords from columns [0,1,2].
    Skips any line whose first token can't be cast to float (header etc.).
    Returns (N,3) float32, or (0,3) if empty.
    """
    if not _csv_has_data(path):
        return np.zeros((0, 3), dtype=np.float32)
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delimiter)
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except (ValueError, IndexError):
                continue   # header or malformed row
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# ROI resolution  (matches train_gb.py priority order)
# ---------------------------------------------------------------------------

def _roi_from_json(json_path: str) -> Roi:
    with open(json_path) as fh:
        j = json.load(fh)
    info   = j["source_info"]
    vox    = np.array(info["vox"],    dtype=float)
    offset = np.array(info["offset"], dtype=float) / vox
    size   = np.array(info["size"],   dtype=float) / vox
    return Roi(offset.astype(int), size.astype(int))


def _roi_from_zarr(zarr_path: str, raw_key: str = "RAW") -> Roi:
    store = zarr.open(zarr_path, mode="r")
    shape = np.array(store[raw_key].shape[-3:], dtype=int)
    return Roi(np.zeros(3, dtype=int), shape)


def resolve_rois(zarr_locs: List[str], params: dict) -> List[Optional[Roi]]:
    """Return the full spatial extent of each zarr as its ROI."""
    rois = []
    for z in zarr_locs:
        try:
            rois.append(_roi_from_zarr(z))
        except Exception as exc:
            print(f"[dataset] WARNING: cannot determine ROI for {z}: {exc}")
            rois.append(None)
    return rois


# ---------------------------------------------------------------------------
# Sample manifest
# ---------------------------------------------------------------------------

class Sample(NamedTuple):
    zarr_path:    str
    roi:          Roi
    pre_csv:      str
    post_csv:     str
    has_synapses: bool
    origin:       np.ndarray   # ZYX voxel-space origin of this zarr in world coords


def _origin_from_zarr_attrs(zarr_path: str, raw_key: str = "RAW") -> np.ndarray:
    """
    Read the world-space origin from the RAW dataset's attrs.
    zarr arrays written by funlib/daisy store 'offset' and 'resolution'
    directly on the dataset, e.g.:
        RAW.attrs = {'offset': [1575, 18350, 19850], 'resolution': [1, 1, 1]}
    Falls back to [0,0,0] if the attribute is absent.
    """
    store = zarr.open(zarr_path, mode="r")
    attrs = store[raw_key].attrs
    offset = np.array(attrs.get("offset", [0, 0, 0]), dtype=np.float32)
    # some writers store resolution separately — divide if present
    res = np.array(attrs.get("resolution", [1, 1, 1]), dtype=np.float32)
    return (offset / res).astype(np.float32)


def build_sample_manifest(params: dict) -> List[Sample]:
    zarr_locs = params["zarr_locs"]
    csv_dir   = params["csv_dir"]
    json_dir  = params.get("json_dir")
    species   = params.get("species", None)

    rois = resolve_rois(zarr_locs, params)
    samples: List[Sample] = []

    for zarr_path, roi in zip(zarr_locs, rois):
        if roi is None:
            continue

        stem  = Path(zarr_path).stem          # e.g. megalopta_FB_1
        parts = stem.split("_", 1)
        if len(parts) < 2:
            print(f"[dataset] WARNING: unexpected stem format '{stem}', skipping")
            continue

        # species filter — first underscore-delimited token
        if species is not None and parts[0] not in species:
            print(f"[dataset] Skipping {stem} (species '{parts[0]}' not in {species})")
            continue

        pre_csv  = os.path.join(csv_dir, f"{stem}_pre.csv")
        post_csv = os.path.join(csv_dir, f"{stem}_post.csv")

        if not os.path.exists(pre_csv) or not os.path.exists(post_csv):
            print(f"[dataset] WARNING: missing CSVs for {stem}, skipping")
            continue

        has_syn = _csv_has_data(pre_csv) and _csv_has_data(post_csv)

        # read world origin directly from the RAW array's stored attributes
        try:
            origin = _origin_from_zarr_attrs(zarr_path)
        except Exception as exc:
            print(f"[dataset] WARNING: could not read origin from {zarr_path}: {exc}, assuming [0,0,0]")
            origin = np.zeros(3, dtype=np.float32)

        samples.append(Sample(zarr_path, roi, pre_csv, post_csv, has_syn, origin))

    print(f"[dataset] {len(samples)} samples loaded "
          f"({sum(s.has_synapses for s in samples)} with synapses)")
    return samples


# ---------------------------------------------------------------------------
# Elastic context helper
# ---------------------------------------------------------------------------

def _elastic_context(params: dict) -> np.ndarray:
    """
    Return the (Z, Y, X) border to load from zarr around the target crop so that
    elastic augmentation has real EM data at its boundaries instead of reflect-padding.
    Returns zeros if elastic augmentation is disabled.
    """
    aug = params.get("augmentation", {})
    if not aug.get("elastic", {}).get("enabled", True):
        return np.zeros(3, dtype=int)

    cfg    = aug.get("elastic", {})
    jitter = cfg.get("jitter_sigma", [1, 3.0, 3.0])

    # minimum padding matching elastic_augment's internal formula
    pad_yx = int(np.ceil(max(jitter[1], jitter[2]))) * 3 + 2
    pad_z  = int(np.ceil(jitter[0])) * 3 + 2

    # extra margin for slip (max_slip = sigma*2) and global shift (sigma*4)
    extra_yx = int(max(jitter[1], jitter[2]) * 4) + 4
    extra_z  = int(jitter[0] * 4) + 4

    return np.array([max(pad_z, extra_z), max(pad_yx, extra_yx), max(pad_yx, extra_yx)], dtype=int)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SynfulDataset(Dataset):

    def __init__(
        self,
        params: dict,
        samples_per_epoch: int = 10_000,
        augment: bool = True,
        raw_dataset: str = "RAW",
    ):
        self.params            = params
        self.samples_per_epoch = samples_per_epoch
        self.augment           = augment
        self.raw_dataset       = raw_dataset

        self.input_size    = np.array(params["input_size"], dtype=int)
        self.output_size   = np.array(params.get("output_size", params["input_size"]), dtype=int)
        self.blob_radius   = list(params["blob_radius"])
        self.d_blob_radius = list(params["d_blob_radius"])
        self.voxel_size    = list(params.get("voxel_size", [1, 1, 1]))
        self.gt_vec_scale  = np.array(params.get("gt_vec_scale", [1, 1, 1]), dtype=np.float32)
        self.delimiter     = params.get("csv_delimiter", ",")
        self.gpu_elastic   = bool(params.get("gpu_elastic", False))

        # p_nonempty mirrors train_gb.py: 0.95 when has_synapses, 0 otherwise
        self.p_nonempty = float(params.get(
            "p_nonempty", params.get("reject_probability", 0.8)
        ))

        # context border for elastic augmentation (zeros if disabled or augment=False)
        self.elastic_ctx = _elastic_context(params) if augment else np.zeros(3, dtype=int)

        self.samples = build_sample_manifest(params)
        if not self.samples:
            raise RuntimeError(
                "[dataset] No valid samples found — check zarr_locs and csv_dir."
            )

        self._stores: List[Optional[zarr.Group]] = []
        for s in self.samples:
            try:
                self._stores.append(zarr.open(s.zarr_path, mode="r"))
            except Exception as exc:
                print(f"[dataset] WARNING: cannot open {s.zarr_path}: {exc}")
                self._stores.append(None)

        # Cache all synapse points up-front — reading CSV per __getitem__ causes
        # thousands of redundant disk reads across workers (one per training sample).
        self._post_points: List[Optional[np.ndarray]] = []
        self._pre_points:  List[Optional[np.ndarray]] = []
        for s in self.samples:
            if s.has_synapses:
                self._post_points.append(load_points_csv(s.post_csv, self.delimiter))
                self._pre_points.append(load_points_csv(s.pre_csv,  self.delimiter))
            else:
                self._post_points.append(np.zeros((0, 3), dtype=np.float32))
                self._pre_points.append(np.zeros((0, 3), dtype=np.float32))

        w = np.array([4.0 if s.has_synapses else 1.0 for s in self.samples])
        self._sample_weights = w / w.sum()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        while True:
            result = self._draw_sample()
            if result is not None:
                return result

    def _draw_sample(self) -> Optional[Dict[str, torch.Tensor]]:
        idx = np.random.choice(len(self.samples), p=self._sample_weights)
        sample = self.samples[idx]
        store  = self._stores[idx]

        if store is None:
            return None

        try:
            raw_arr = store[self.raw_dataset]
        except KeyError:
            return None

        vol_shape = np.array(raw_arr.shape[-3:], dtype=int)

        # ROI is the full volume — crop anywhere within it
        raw_off = sample.roi.offset
        raw_sh  = sample.roi.shape

        # load_size = input_size + 2*context so elastic has real border data
        ctx       = self.elastic_ctx
        load_size = self.input_size + 2 * ctx

        # fall back to no context if the volume is too small
        if np.any(raw_sh < load_size):
            if np.any(raw_sh < self.input_size):
                return None
            ctx       = np.zeros(3, dtype=int)
            load_size = self.input_size

        # random crop (of the larger load_size) within the volume
        max_off      = raw_sh - load_size
        crop_off     = np.array([np.random.randint(0, int(m)+1) for m in max_off])
        abs_crop_off = raw_off + crop_off

        sl = tuple(
            slice(int(abs_crop_off[i]), int(abs_crop_off[i] + load_size[i]))
            for i in range(3)
        )
        raw_crop = (
            raw_arr[sl] if raw_arr.ndim == 3 else raw_arr[0][sl]
        ).astype(np.float32)

        lo, hi   = raw_crop.min(), raw_crop.max()
        raw_crop = (raw_crop - lo) / max(hi - lo, 1e-8)

        # synapses → coords relative to the full load_size crop (including context)
        load_shape = tuple(load_size)
        post_locs  = np.zeros((0, 3), dtype=np.float32)
        pre_locs   = np.zeros((0, 3), dtype=np.float32)

        if sample.has_synapses:
            post_abs = self._post_points[idx]
            pre_abs  = self._pre_points[idx]

            if len(post_abs) > 0 and len(pre_abs) > 0:
                if len(post_abs) != len(pre_abs):
                    print(f"[dataset] WARNING: {Path(sample.post_csv).name} has {len(post_abs)} rows "
                          f"but {Path(sample.pre_csv).name} has {len(pre_abs)} rows — skipping sample")
                    return None
                post_local = post_abs - sample.origin - abs_crop_off
                pre_local  = pre_abs  - sample.origin - abs_crop_off

                in_bounds = (
                    (post_local[:,0] >= 0) & (post_local[:,0] < load_shape[0]) &
                    (post_local[:,1] >= 0) & (post_local[:,1] < load_shape[1]) &
                    (post_local[:,2] >= 0) & (post_local[:,2] < load_shape[2])
                )
                post_locs = post_local[in_bounds]
                pre_locs  = pre_local[in_bounds]

        # rejection sampling — check against the output region, not the full
        # input region. input_size includes elastic context; output_size is what
        # the model actually supervises. Using input_size here caused synapses
        # near output edges to be systematically underrepresented during training
        # because a synapse centre had to land well inside the larger input crop
        # to count, making edge-of-output positions rarely seen.
        inner_sl    = tuple(slice(int(ctx[i]), int(ctx[i] + self.input_size[i])) for i in range(3))
        out_margin  = (self.input_size - self.output_size) // 2
        out_lo      = ctx + out_margin
        out_hi      = out_lo + self.output_size
        inner_post  = post_locs[
            np.all((post_locs >= out_lo) & (post_locs < out_hi), axis=1)
        ] if len(post_locs) else post_locs
        p_req = self.p_nonempty if sample.has_synapses else 0.0
        if len(inner_post) == 0 and np.random.random() < p_req:
            return None

        # render targets at full load_size (context border included)
        indicator = render_syn_indicators(load_shape, post_locs, self.blob_radius)
        vectors, d_weight = render_direction_vectors(
            load_shape, post_locs, pre_locs, self.d_blob_radius, self.voxel_size
        )
        vectors = vectors * self.gt_vec_scale[:, None, None, None]

        # augmentation — elastic step crops back to input_size using real context.
        # When gpu_elastic=True, elastic is deferred to the training loop (faster);
        # the context border is stripped here manually instead.
        if self.augment:
            aug_ctx = "defer" if self.gpu_elastic else ctx
            raw_crop, indicator, vectors, d_weight = augment_sample(
                raw_crop, indicator, vectors, d_weight, self.params, context=aug_ctx
            )
            if self.gpu_elastic:
                # elastic was deferred — strip context border now.
                # augment_sample already applied intensity_scale_shift ([0,1]→[-1,1]),
                # so do NOT re-apply the *2-1 normalisation here.
                raw_crop  = raw_crop[inner_sl]
                indicator = indicator[inner_sl]
                d_weight  = d_weight[inner_sl]
                vectors   = vectors[(slice(None),) + inner_sl]
        else:
            # no augmentation: center-crop to input_size and normalise [0,1]→[-1,1]
            raw_crop  = raw_crop[inner_sl] * 2.0 - 1.0
            indicator = indicator[inner_sl]
            d_weight  = d_weight[inner_sl]
            vectors   = vectors[(slice(None),) + inner_sl]

        return {
            "raw":               torch.from_numpy(raw_crop[None]),
            "indicator_mask":    torch.from_numpy(indicator[None]),
            "direction_vectors": torch.from_numpy(vectors.copy()),
            "d_weight_mask":     torch.from_numpy(d_weight[None]),
        }


def build_dataset(params: dict, samples_per_epoch: int = 10_000) -> SynfulDataset:
    return SynfulDataset(params, samples_per_epoch=samples_per_epoch)