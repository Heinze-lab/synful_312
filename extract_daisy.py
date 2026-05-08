"""
extract_daisy.py  –  Block-wise synapse extraction using daisy.

Usage:
    python extract_daisy.py parameter.json

Reads params["extract_configs"] — same keys as extract.py, plus:
    block_size_zyx   : inner block size in voxels  (default [48, 536, 536])
    context_zyx      : halo on each side in voxels (default [20, 40, 40])
    num_workers      : daisy worker processes       (default 4)

Strategy:
    Each daisy block reads indicators + vectors with a halo (context).
    Threshold → CC → extract detections whose centroid falls inside the
    inner (non-halo) block.  Because the halo is larger than any synapse
    blob, every blob is fully captured in exactly one block.
    After all blocks finish, global NMS is applied and CSV/JSON written.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import (
    distance_transform_edt,
    label as nd_label,
    find_objects,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import daisy
from funlib.geometry import Coordinate, Roi

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS (identical to extract.py)
# ---------------------------------------------------------------------------

def nms(detections: list[dict], radius: float) -> list[dict]:
    if not detections:
        return []
    from scipy.spatial import cKDTree
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    pts = np.array([[d["post_z"], d["post_y"], d["post_x"]] for d in dets])
    tree = cKDTree(pts)
    suppressed = np.zeros(len(dets), dtype=bool)
    kept = []
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(d)
        neighbors = tree.query_ball_point(pts[i], radius)
        for j in neighbors:
            if j > i:
                suppressed[j] = True
    return kept


# ---------------------------------------------------------------------------
# Per-block extraction worker
# ---------------------------------------------------------------------------

def _split_mask_watershed(
    mask: np.ndarray,
    min_peak_dist: int,
    min_sub_size: int,
) -> list[np.ndarray]:
    """
    Given a binary mask, try to split it into sub-regions via EDT + watershed.
    Returns a list of boolean sub-masks (one per synapse).
    If no split is warranted (only one EDT peak found), returns [mask].
    """
    edt = distance_transform_edt(mask)
    edt_max = edt.max()
    if edt_max == 0:
        return [mask]

    # only seed from peaks that are at least 50% of the EDT maximum —
    # this suppresses the many shallow local maxima on flat blob surfaces
    peak_coords = peak_local_max(
        edt,
        min_distance=min_peak_dist,
        threshold_abs=edt_max * 0.5,
        exclude_border=False,
    )
    if len(peak_coords) <= 1:
        return [mask]

    # build seed markers — one unique label per peak
    markers = np.zeros(mask.shape, dtype=np.int32)
    for i, coord in enumerate(peak_coords, start=1):
        markers[tuple(coord)] = i

    # watershed on inverted EDT (fills from peaks outward)
    ws = watershed(-edt, markers, mask=mask)

    sub_masks = []
    for lbl in range(1, len(peak_coords) + 1):
        sub = ws == lbl
        if sub.sum() >= min_sub_size:
            sub_masks.append(sub)

    # if filtering removed all regions somehow, fall back to unsplit
    return sub_masks if sub_masks else [mask]


def _sphere_mask(shape: tuple, center: np.ndarray, radius_vx: np.ndarray) -> np.ndarray:
    """Boolean mask of an axis-aligned ellipsoid with per-axis radii."""
    gz, gy, gx = np.ogrid[
        0:shape[0],
        0:shape[1],
        0:shape[2],
    ]
    dist_sq = (
        ((gz - center[0]) / radius_vx[0]) ** 2 +
        ((gy - center[1]) / radius_vx[1]) ** 2 +
        ((gx - center[2]) / radius_vx[2]) ** 2
    )
    return dist_sq <= 1.0


def extract_block(
    block: daisy.Block,
    ind_ds,
    vec_ds,
    cc_threshold: float,
    loc_type: str,
    score_thr: float,
    score_type: str,
    size_thr: int,
    flipprepost: bool,
    post_offset_scale: float,
    pre_offset_scale: float,
    vec_scale: np.ndarray,
    voxel_size: np.ndarray,
    zarr_offset: np.ndarray,
    read_offset_nm: np.ndarray,
    tmp_dir: str,
    suppression_enabled: bool = False,
    suppression_threshold: float = 0.9,
    suppression_radius_vx: np.ndarray = None,
    suppression_min_separation_vx: float = 5.0,
    suppression_score_thr: float = None,
    suppression_size_thr: int = None,
    splitting_enabled: bool = False,
    splitting_min_size_vx: int = 80,
    splitting_min_elongation: float = 0.0,
    splitting_min_peak_dist: int = 4,
    splitting_min_sub_size: int = 8,
) -> None:
    """Run inside a daisy worker process. Writes per-block results to tmp_dir."""

    # ---- read arrays for the full read_roi (inner + halo) ------------------
    read_roi  = block.read_roi
    write_roi = block.write_roi

    # convert daisy Roi (world/nm) → zarr voxel slices
    # zarr array is indexed from 0, so subtract the world origin (zarr_offset + read_offset_nm)
    world_origin_nm = zarr_offset + read_offset_nm
    vs = voxel_size  # ZYX nm/voxel

    arr_shape = np.array(ind_ds.shape)

    def roi_to_slices(roi):
        offset_vx = ((np.array(roi.offset) - world_origin_nm) / vs).astype(int)
        shape_vx  = (np.array(roi.shape) / vs).astype(int)
        # clamp both start and end to valid array bounds
        start = np.maximum(offset_vx, 0)
        end   = np.minimum(offset_vx + shape_vx, arr_shape)
        return tuple(slice(int(s), int(e)) for s, e in zip(start, end))

    read_sl  = roi_to_slices(read_roi)
    write_sl = roi_to_slices(write_roi)

    # halo offset inside the read crop (write_roi.offset - read_roi.offset)
    halo_vx = ((np.array(write_roi.offset) - np.array(read_roi.offset)) / vs).astype(int)

    block_id = "_".join(str(int(o)) for o in block.write_roi.offset)
    log.info(f"Block {block_id}: read_sl={read_sl}")

    # load only indicators for the full read region — vectors read lazily per detection
    try:
        ind_crop = np.array(ind_ds[read_sl]).astype(np.float32) / 255.0
    except Exception as exc:
        log.warning(f"Block {block_id} read failed: {exc}")
        _write_block_result(tmp_dir, block_id, [])
        return

    # ---- threshold + CC ----------------------------------------------------
    binary = ind_crop >= cc_threshold
    if not binary.any():
        log.info(f"Block {block_id}: no foreground")
        _write_block_result(tmp_dir, block_id, [])
        return

    labeled, n_cc = nd_label(binary)
    if n_cc == 0:
        _write_block_result(tmp_dir, block_id, [])
        return

    bboxes = find_objects(labeled)

    # write_roi shape in voxels (for centroid filtering)
    write_shape_vx = (np.array(write_roi.shape) / vs).astype(int)

    # absolute zarr origin of this read crop (for lazy vec slicing)
    read_origin_vx = np.array([s.start for s in read_sl])

    detections = []
    # occupied_mask tracks all voxels claimed by accepted CCs (used by suppression guard A)
    occupied_mask = np.zeros_like(binary)

    for lbl, bbox in enumerate(bboxes, start=1):
        if bbox is None:
            continue

        lab_crop  = labeled[bbox]
        mask_crop = lab_crop == lbl

        size = int(mask_crop.sum())
        if size < size_thr:
            continue

        ind_sub = ind_crop[bbox]
        if score_type == "mean":
            score = float(ind_sub[mask_crop].mean())
        else:
            score = float(ind_sub[mask_crop].max())

        if score < score_thr:
            continue

        # ---- optional blob splitting ----------------------------------------
        # Decide whether to attempt a watershed split on this CC.
        attempt_split = (
            splitting_enabled
            and size >= splitting_min_size_vx
        )
        if attempt_split and splitting_min_elongation > 0.0:
            # cheap elongation gate: major/minor axis ratio via inertia tensor
            try:
                from skimage.measure import regionprops
                props = regionprops(mask_crop.astype(np.uint8))[0]
                elongation = (props.axis_major_length / max(props.axis_minor_length, 1e-6))
                attempt_split = elongation >= splitting_min_elongation
            except Exception:
                pass  # if regionprops fails for any reason, still attempt split

        if attempt_split:
            sub_masks = _split_mask_watershed(mask_crop, splitting_min_peak_dist, splitting_min_sub_size)
        else:
            sub_masks = [mask_crop]

        bbox_origin = np.array([s.start for s in bbox])

        # lazy vector read once for the whole bbox (shared across sub-masks)
        abs_bbox = tuple(
            slice(int(s.start + read_origin_vx[i]), int(s.stop + read_origin_vx[i]))
            for i, s in enumerate(bbox)
        )
        vec_crop_raw = np.array(vec_ds[(slice(None),) + abs_bbox]).astype(np.float32)
        vec_crop_bb  = vec_crop_raw / vec_scale[:, None, None, None]

        for sub_mask in sub_masks:
            sub_size = int(sub_mask.sum())

            # re-score the sub-region
            ind_sub_local = ind_sub  # same bbox
            if score_type == "mean":
                sub_score = float(ind_sub_local[sub_mask].mean())
            else:
                sub_score = float(ind_sub_local[sub_mask].max())

            if sub_score < score_thr:
                continue

            # location within bbox
            if loc_type == "centroid":
                zz, yy, xx = np.where(sub_mask)
                local_loc = np.array([zz.mean(), yy.mean(), xx.mean()])
            elif loc_type == "edt":
                edt = distance_transform_edt(sub_mask)
                local_loc = np.array(np.unravel_index(edt.argmax(), edt.shape), dtype=float)
            else:  # peak
                masked = ind_sub_local * sub_mask
                local_loc = np.array(np.unravel_index(masked.argmax(), masked.shape), dtype=float)

            post_in_crop = local_loc + bbox_origin

            # check centroid is inside write roi
            pos_in_write = post_in_crop - halo_vx
            if np.any(pos_in_write < 0) or np.any(pos_in_write >= write_shape_vx):
                continue

            vz = float(vec_crop_bb[0][sub_mask].mean())
            vy = float(vec_crop_bb[1][sub_mask].mean())
            vx = float(vec_crop_bb[2][sub_mask].mean())
            vec = np.array([vz, vy, vx])

            post_zyx = post_in_crop
            pre_zyx  = post_zyx + vec

            if post_offset_scale != 0.0:
                post_zyx = post_zyx + vec * post_offset_scale
            if pre_offset_scale != 0.0:
                pre_zyx = pre_zyx + vec * pre_offset_scale

            post_world = (post_zyx + read_origin_vx) * vs + world_origin_nm
            pre_world  = (pre_zyx  + read_origin_vx) * vs + world_origin_nm

            if flipprepost:
                post_world, pre_world = pre_world, post_world

            detections.append({
                "post_z": float(post_world[0]),
                "post_y": float(post_world[1]),
                "post_x": float(post_world[2]),
                "pre_z":  float(pre_world[0]),
                "pre_y":  float(pre_world[1]),
                "pre_x":  float(pre_world[2]),
                "score":  sub_score,
                "size":   sub_size,
            })

        # mark voxels occupied for suppression guard A
        full_mask = labeled == lbl
        occupied_mask |= full_mask

    # ---- local confidence suppression pass ---------------------------------
    if suppression_enabled and suppression_radius_vx is not None:
        sup_score_thr = suppression_score_thr if suppression_score_thr is not None else score_thr
        sup_size_thr  = suppression_size_thr  if suppression_size_thr  is not None else size_thr

        # collect centroids (crop coords) of high-confidence initial detections
        high_conf_centroids = []
        for lbl, bbox in enumerate(bboxes, start=1):
            if bbox is None:
                continue
            lab_crop  = labeled[bbox]
            mask_crop = lab_crop == lbl
            ind_sub   = ind_crop[bbox]
            if score_type == "mean":
                sc = float(ind_sub[mask_crop].mean())
            else:
                sc = float(ind_sub[mask_crop].max())
            if sc >= suppression_threshold:
                if loc_type == "centroid":
                    zz, yy, xx = np.where(mask_crop)
                    cen = np.array([zz.mean(), yy.mean(), xx.mean()])
                elif loc_type == "edt":
                    edt = distance_transform_edt(mask_crop)
                    cen = np.array(np.unravel_index(edt.argmax(), edt.shape), dtype=float)
                else:
                    masked = ind_sub * mask_crop
                    cen = np.array(np.unravel_index(masked.argmax(), masked.shape), dtype=float)
                bbox_origin = np.array([s.start for s in bbox])
                high_conf_centroids.append(cen + bbox_origin)

        if high_conf_centroids:
            # suppress a sphere around each high-conf centroid in a copy of ind_crop
            ind_suppressed = ind_crop.copy()
            for cen in high_conf_centroids:
                smask = _sphere_mask(ind_suppressed.shape, cen, suppression_radius_vx)
                ind_suppressed[smask] = 0.0

            # re-threshold and re-label the suppressed volume
            binary2  = ind_suppressed >= cc_threshold
            # exclude voxels already occupied by initial CCs (guard A)
            binary2 &= ~occupied_mask
            if binary2.any():
                labeled2, n2 = nd_label(binary2)
                bboxes2 = find_objects(labeled2)
                log.info(f"Block {block_id}: suppression found {n2} candidate CCs")

                initial_post_pts = np.array([
                    [(d["post_z"] - world_origin_nm[0]) / vs[0] - read_origin_vx[0],
                     (d["post_y"] - world_origin_nm[1]) / vs[1] - read_origin_vx[1],
                     (d["post_x"] - world_origin_nm[2]) / vs[2] - read_origin_vx[2]]
                    for d in detections
                ]) if detections else None

                n_new = 0
                for lbl2, bbox2 in enumerate(bboxes2, start=1):
                    if bbox2 is None:
                        continue

                    lab_crop2  = labeled2[bbox2]
                    mask_crop2 = lab_crop2 == lbl2

                    size2 = int(mask_crop2.sum())
                    if size2 < sup_size_thr:
                        continue

                    ind_sub2 = ind_suppressed[bbox2]
                    if score_type == "mean":
                        score2 = float(ind_sub2[mask_crop2].mean())
                    else:
                        score2 = float(ind_sub2[mask_crop2].max())

                    if score2 < sup_score_thr:
                        continue

                    if loc_type == "centroid":
                        zz, yy, xx = np.where(mask_crop2)
                        local_loc2 = np.array([zz.mean(), yy.mean(), xx.mean()])
                    elif loc_type == "edt":
                        edt2 = distance_transform_edt(mask_crop2)
                        local_loc2 = np.array(np.unravel_index(edt2.argmax(), edt2.shape), dtype=float)
                    else:
                        masked2 = ind_sub2 * mask_crop2
                        local_loc2 = np.array(np.unravel_index(masked2.argmax(), masked2.shape), dtype=float)

                    bbox_origin2 = np.array([s.start for s in bbox2])
                    post_in_crop2 = local_loc2 + bbox_origin2

                    # check centroid inside write roi
                    pos_in_write2 = post_in_crop2 - halo_vx
                    if np.any(pos_in_write2 < 0) or np.any(pos_in_write2 >= write_shape_vx):
                        continue

                    # guard B: centroid distance from all initial detections
                    if initial_post_pts is not None:
                        dists = np.linalg.norm(initial_post_pts - post_in_crop2, axis=1)
                        if dists.min() < suppression_min_separation_vx:
                            continue

                    abs_bbox2 = tuple(
                        slice(int(s.start + read_origin_vx[i]), int(s.stop + read_origin_vx[i]))
                        for i, s in enumerate(bbox2)
                    )
                    vec_crop_raw2 = np.array(vec_ds[(slice(None),) + abs_bbox2]).astype(np.float32)
                    vec_crop_bb2  = vec_crop_raw2 / vec_scale[:, None, None, None]

                    vz2 = float(vec_crop_bb2[0][mask_crop2].mean())
                    vy2 = float(vec_crop_bb2[1][mask_crop2].mean())
                    vx2 = float(vec_crop_bb2[2][mask_crop2].mean())
                    vec2 = np.array([vz2, vy2, vx2])

                    post_zyx2 = post_in_crop2
                    pre_zyx2  = post_zyx2 + vec2

                    if post_offset_scale != 0.0:
                        post_zyx2 = post_zyx2 + vec2 * post_offset_scale
                    if pre_offset_scale != 0.0:
                        pre_zyx2 = pre_zyx2 + vec2 * pre_offset_scale

                    post_world2 = (post_zyx2 + read_origin_vx) * vs + world_origin_nm
                    pre_world2  = (pre_zyx2  + read_origin_vx) * vs + world_origin_nm

                    if flipprepost:
                        post_world2, pre_world2 = pre_world2, post_world2

                    detections.append({
                        "post_z": float(post_world2[0]),
                        "post_y": float(post_world2[1]),
                        "post_x": float(post_world2[2]),
                        "pre_z":  float(pre_world2[0]),
                        "pre_y":  float(pre_world2[1]),
                        "pre_x":  float(pre_world2[2]),
                        "score":  score2,
                        "size":   size2,
                    })
                    n_new += 1

                log.info(f"Block {block_id}: suppression added {n_new} new detections")

    log.info(f"Block {block_id}: {len(detections)} detections")
    _write_block_result(tmp_dir, block_id, detections)


def _write_block_result(tmp_dir: str, block_id: str, detections: list) -> None:
    import pickle
    path = os.path.join(tmp_dir, f"block_{block_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(detections, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    cfg = params["extract_configs"]

    # ---- parameters --------------------------------------------------------
    cc_threshold      = float(cfg.get("cc_threshold",      0.5))
    loc_type          = cfg.get("loc_type",                "edt")
    score_thr         = float(cfg.get("score_thr",         0.5))
    score_type        = cfg.get("score_type",              "mean")
    size_thr          = int(cfg.get("size_thr",            0))
    nms_radius        = float(cfg.get("nms_radius",        0))
    flipprepost       = bool(cfg.get("flipprepost",        False))
    post_offset_scale = float(cfg.get("post_offset_scale", 0.0))
    pre_offset_scale  = float(cfg.get("pre_offset_scale",  0.0))
    vec_scale_cfg     = np.array(cfg.get("vector_scale",   [1, 1, 1]), dtype=np.float32)

    block_size_zyx = list(cfg.get("block_size_zyx", [48, 536, 536]))
    context_zyx    = list(cfg.get("context_zyx",    [20,  40,  40]))
    num_workers    = int(cfg.get("num_workers",     4))

    # ---- suppression parameters --------------------------------------------
    sup_cfg = cfg.get("suppression", {})
    suppression_enabled   = bool(sup_cfg.get("enabled",            False))
    suppression_threshold = float(sup_cfg.get("threshold",          0.9))
    suppression_radius_vx = np.array(
        sup_cfg.get("radius_vx", [3, 8, 8]), dtype=float
    )
    suppression_min_sep   = float(sup_cfg.get("min_separation_vx",  5.0))
    # optional overrides for score/size thresholds in the suppression pass
    _sup_score_raw = sup_cfg.get("score_thr",  None)
    _sup_size_raw  = sup_cfg.get("size_thr",   None)
    suppression_score_thr = float(_sup_score_raw) if _sup_score_raw is not None else None
    suppression_size_thr  = int(_sup_size_raw)     if _sup_size_raw  is not None else None

    # ---- splitting parameters ----------------------------------------------
    spl_cfg = cfg.get("blob_splitting", {})
    splitting_enabled      = bool(spl_cfg.get("enabled",           False))
    splitting_min_size_vx  = int(spl_cfg.get("min_size_vx",        80))
    splitting_min_elong    = float(spl_cfg.get("elongation_thr",   0.0))
    splitting_min_peak_dist= int(spl_cfg.get("min_peak_distance_vx", 4))
    splitting_min_sub_size = int(spl_cfg.get("min_sub_size_vx",    8))

    log.info(f"cc_threshold={cc_threshold}  loc_type={loc_type}")
    log.info(f"score_thr={score_thr}  size_thr={size_thr}  nms_radius={nms_radius}")
    log.info(f"block_size_zyx={block_size_zyx}  context_zyx={context_zyx}  num_workers={num_workers}")
    if splitting_enabled:
        log.info(
            f"Blob splitting: min_size_vx={splitting_min_size_vx}  "
            f"elongation_thr={splitting_min_elong}  "
            f"min_peak_distance_vx={splitting_min_peak_dist}  "
            f"min_sub_size_vx={splitting_min_sub_size}"
        )
    if suppression_enabled:
        log.info(
            f"Suppression: threshold={suppression_threshold}  "
            f"radius_vx={suppression_radius_vx.tolist()}  "
            f"min_separation_vx={suppression_min_sep}  "
            f"score_thr={suppression_score_thr}  size_thr={suppression_size_thr}"
        )

    # ---- open prediction zarr ----------------------------------------------
    inf_path = os.path.join(cfg.get("inference_dir", "."), cfg.get("inference_file", "pred.zarr"))
    log.info(f"Reading: {inf_path}")

    store = zarr.open(inf_path, mode="r")
    if "pred_syn_indicators" not in store:
        raise KeyError(f"'pred_syn_indicators' not found in {inf_path}")
    if "pred_partner_vectors" not in store:
        raise KeyError(f"'pred_partner_vectors' not found in {inf_path}")

    ind_ds = store["pred_syn_indicators"]
    vec_ds = store["pred_partner_vectors"]

    vol_shape = np.array(ind_ds.shape)   # ZYX
    log.info(f"Volume shape: {vol_shape}  (indicators uint8 {vol_shape.prod()/1e9:.2f} GB)")

    # ---- world offset / voxel size -----------------------------------------
    raw_file = cfg.get("raw_file") or params.get("predict", {}).get("raw_file", "")
    raw_ds   = cfg.get("raw_dataset") or params.get("predict", {}).get("raw_dataset", "RAW")
    zarr_offset = np.zeros(3, dtype=float)
    voxel_size  = np.ones(3,  dtype=float)

    if raw_file and os.path.exists(raw_file):
        try:
            rz = zarr.open(raw_file, mode="r")
            zarr_offset = np.array(rz[raw_ds].attrs.get("offset",     [0,0,0]), dtype=float)
            voxel_size  = np.array(rz[raw_ds].attrs.get("resolution", [1,1,1]), dtype=float)
            log.info(f"Zarr offset:     {zarr_offset.tolist()}")
            log.info(f"Voxel size (nm): {voxel_size.tolist()}")
        except Exception as exc:
            log.warning(f"Could not read zarr attrs: {exc}")

    read_offset = np.array(
        params.get("predict", {}).get("read_offset", [0, 0, 0]), dtype=float
    )
    # -1 is used as a sentinel meaning "start of volume" in predict.py; clamp here too.
    read_offset = np.where(read_offset < 0, 0.0, read_offset)
    read_offset_nm = read_offset * voxel_size

    # ---- build daisy ROIs --------------------------------------------------
    # daisy works in world (nm) units
    vs = voxel_size  # ZYX

    total_roi = Roi(
        offset=tuple((zarr_offset + read_offset_nm).tolist()),
        shape=tuple((vol_shape * vs).tolist()),
    )
    block_size_nm = tuple(float(b * v) for b, v in zip(block_size_zyx, vs))
    context_nm    = tuple(float(c * v) for c, v in zip(context_zyx,    vs))

    write_roi = Roi(offset=(0,)*3, shape=block_size_nm)
    read_roi  = write_roi.grow(context_nm, context_nm)

    log.info(f"Total ROI: {total_roi}")
    log.info(f"Block size (nm): {block_size_nm}  Context (nm): {context_nm}")

    # ---- temp dir for per-block results ------------------------------------
    import tempfile, pickle
    tmp_dir = tempfile.mkdtemp(prefix="extract_daisy_")
    log.info(f"Temp dir for block results: {tmp_dir}")

    # ---- daisy task --------------------------------------------------------
    task = daisy.Task(
        task_id="extract",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda block: extract_block(
            block,
            ind_ds,
            vec_ds,
            cc_threshold,
            loc_type,
            score_thr,
            score_type,
            size_thr,
            flipprepost,
            post_offset_scale,
            pre_offset_scale,
            vec_scale_cfg,
            vs,
            zarr_offset,
            read_offset_nm,
            tmp_dir,
            suppression_enabled=suppression_enabled,
            suppression_threshold=suppression_threshold,
            suppression_radius_vx=suppression_radius_vx,
            suppression_min_separation_vx=suppression_min_sep,
            suppression_score_thr=suppression_score_thr,
            suppression_size_thr=suppression_size_thr,
            splitting_enabled=splitting_enabled,
            splitting_min_size_vx=splitting_min_size_vx,
            splitting_min_elongation=splitting_min_elong,
            splitting_min_peak_dist=splitting_min_peak_dist,
            splitting_min_sub_size=splitting_min_sub_size,
        ),
        num_workers=num_workers,
        fit="shrink",
    )

    log.info("Starting daisy scheduler ...")
    daisy.run_blockwise([task])
    log.info("All blocks done.")

    # ---- collect results ---------------------------------------------------
    import glob
    all_detections = []
    for pkl_path in glob.glob(os.path.join(tmp_dir, "block_*.pkl")):
        with open(pkl_path, "rb") as f:
            all_detections.extend(pickle.load(f))
    import shutil
    shutil.rmtree(tmp_dir)

    log.info(f"After size/score filter: {len(all_detections)} detections")

    # ---- global NMS --------------------------------------------------------
    if nms_radius > 0:
        all_detections = nms(all_detections, nms_radius)
        log.info(f"After NMS (r={nms_radius}): {len(all_detections)} detections")

    # re-index IDs
    for i, d in enumerate(all_detections):
        d["id"] = i

    # ---- write output ------------------------------------------------------
    to_json  = params.get("to_json_config", {})
    out_name = to_json.get("output_name",
               cfg.get("inference_dir", ".") + "/synapses.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)

    with open(out_name, "w") as fh:
        json.dump({"synapses": all_detections, "n": len(all_detections)}, fh, indent=2)
    log.info(f"Wrote {len(all_detections)} synapses → {out_name}")

    csv_path = out_name.replace(".json", ".csv")
    with open(csv_path, "w") as fh:
        fh.write("id,post_z,post_y,post_x,pre_z,pre_y,pre_x,score,size\n")
        for d in all_detections:
            fh.write(
                f"{d['id']},{d['post_z']:.1f},{d['post_y']:.1f},{d['post_x']:.1f},"
                f"{d['pre_z']:.1f},{d['pre_y']:.1f},{d['pre_x']:.1f},"
                f"{d['score']:.4f},{d['size']}\n"
            )
    log.info(f"Wrote CSV → {csv_path}")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter.json"
    extract(params_path)
