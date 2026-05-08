"""
extract.py  –  Synapse extraction from network predictions.

Usage:
    python extract.py parameter_logits_big.json

Implements params["extract_configs"]:
    cc_threshold   : threshold pred_syn_indicators to get binary blobs
    loc_type       : "centroid" → location = unweighted centre of mass of the blob
                     "edt"      → location = peak of Euclidean distance transform
                     "peak"     → location = peak of raw probability map
    score_thr      : discard detections whose mean score < score_thr
    score_type     : "mean" → use mean of pred_syn_indicators in the CC
    size_thr       : discard CCs smaller than this many voxels
    nms_radius     : suppress detections within this radius of a higher-scoring one
    flipprepost        : if True, swap pre/post (swap sign of direction vector)
    post_offset_scale  : shift post site along the direction vector by this fraction
                         of the vector length (e.g. 0.5 moves post halfway toward pre)

Output: JSON file with a list of synapses, each:
    {
        "id":       int,
        "post_z":   float,  "post_y": float,  "post_x": float,
        "pre_z":    float,  "pre_y":  float,  "pre_x":  float,
        "score":    float
    }

Coordinates are in absolute world voxels (zarr offset is added back).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import zarr
from scipy.ndimage import (
    distance_transform_edt,
    label as nd_label,
    find_objects,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def edt_peak(binary_mask: np.ndarray) -> np.ndarray:
    """Return ZYX coordinates of the peak of the EDT inside a binary mask."""
    edt = distance_transform_edt(binary_mask)
    return np.array(np.unravel_index(edt.argmax(), edt.shape), dtype=float)


def prob_peak(prob_map: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Return ZYX coordinates of the maximum probability inside a binary mask."""
    masked = prob_map * binary_mask
    return np.array(np.unravel_index(masked.argmax(), masked.shape), dtype=float)


def nms(detections: list[dict], radius: float) -> list[dict]:
    """
    Non-maximum suppression: keep only the highest-scoring detection
    within `radius` voxels of any other detection.
    Sorted highest→lowest score so the best ones survive.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    suppressed = set()
    for i, d in enumerate(dets):
        if i in suppressed:
            continue
        kept.append(d)
        pi = np.array([d["post_z"], d["post_y"], d["post_x"]])
        for j, d2 in enumerate(dets[i+1:], start=i+1):
            if j in suppressed:
                continue
            pj = np.array([d2["post_z"], d2["post_y"], d2["post_x"]])
            if np.linalg.norm(pi - pj) < radius:
                suppressed.add(j)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    cfg = params["extract_configs"]

    # ---- parameters -------------------------------------------------------
    cc_threshold  = float(cfg.get("cc_threshold",  0.5))
    loc_type      = cfg.get("loc_type",     "edt")
    score_thr     = float(cfg.get("score_thr",     0.5))
    score_type    = cfg.get("score_type",   "mean")
    size_thr      = int(cfg.get("size_thr",       0))
    nms_radius        = float(cfg.get("nms_radius",        0))
    flipprepost       = bool(cfg.get("flipprepost",        False))
    post_offset_scale   = float(cfg.get("post_offset_scale", 0.0))
    pre_offset_scale    = float(cfg.get("pre_offset_scale",  0.0))

    print(f"[extract] cc_threshold={cc_threshold}  loc_type={loc_type}")
    print(f"[extract] score_thr={score_thr}  score_type={score_type}")
    print(f"[extract] size_thr={size_thr}  nms_radius={nms_radius}")
    print(f"[extract] flipprepost={flipprepost}  post_offset_scale={post_offset_scale}  pre_offset_scale={pre_offset_scale}")

    # ---- open prediction zarr ---------------------------------------------
    inf_dir  = cfg.get("inference_dir",  ".")
    inf_file = cfg.get("inference_file", "pred.zarr")
    inf_path = os.path.join(inf_dir, inf_file)
    print(f"[extract] Reading: {inf_path}")

    store = zarr.open(inf_path, mode="r")

    if "pred_syn_indicators" not in store:
        raise KeyError(f"'pred_syn_indicators' not found in {inf_path}")
    if "pred_partner_vectors" not in store:
        raise KeyError(f"'pred_partner_vectors' not found in {inf_path}")

    # pred_syn_indicators is uint8 scaled 0-255 → convert back to [0,1]
    indicators = store["pred_syn_indicators"][:].astype(np.float32) / 255.0
    _vec_raw  = store["pred_partner_vectors"][:].astype(np.float32)   # (3,Z,Y,X) int8-scaled
    _vec_scale = np.array(cfg.get("vector_scale", [1, 1, 1]), dtype=np.float32)[:, None, None, None]
    vectors   = _vec_raw * _vec_scale

    print(f"[extract] Indicator shape: {indicators.shape}  "
          f"range=[{indicators.min():.3f},{indicators.max():.3f}]")
    print(f"[extract] Vector shape   : {vectors.shape}")

    # ---- zarr world offset ------------------------------------------------
    # Try to read offset from the raw zarr so output coords are in world space
    # extract_configs.raw_file overrides predict.raw_file (useful when predict
    # uses a rechunked/local copy that lacks attrs)
    raw_file = cfg.get("raw_file") or params.get("predict", {}).get("raw_file", "")
    raw_ds   = cfg.get("raw_dataset") or params.get("predict", {}).get("raw_dataset", "RAW")
    zarr_offset  = np.zeros(3, dtype=float)
    voxel_size   = np.ones(3,  dtype=float)
    if raw_file and os.path.exists(raw_file):
        try:
            rz = zarr.open(raw_file, mode="r")
            zarr_offset = np.array(rz[raw_ds].attrs.get("offset",     [0,0,0]), dtype=float)
            voxel_size  = np.array(rz[raw_ds].attrs.get("resolution", [1,1,1]), dtype=float)
            print(f"[extract] Zarr offset:     {zarr_offset.tolist()}")
            print(f"[extract] Voxel size (nm): {voxel_size.tolist()}")
        except Exception as exc:
            print(f"[extract] WARNING: could not read zarr attrs: {exc}")

    # read_offset is in local voxels; convert to nm then add zarr world origin
    read_offset = np.array(
        params.get("predict", {}).get("read_offset", [0, 0, 0]),
        dtype=float
    )
    read_offset_nm = read_offset * voxel_size

    # ---- threshold + connected components ---------------------------------
    binary    = indicators >= cc_threshold
    n_fg      = binary.sum()
    print(f"[extract] Foreground voxels at thresh {cc_threshold}: {n_fg}")

    if n_fg == 0:
        print("[extract] No foreground found — lower cc_threshold")
        _write_output([], params, cfg)
        return

    labeled, n_cc = nd_label(binary)
    print(f"[extract] Connected components found: {n_cc}")

    # get bounding boxes once — avoids full-volume mask per CC
    bboxes = find_objects(labeled)

    # ---- process each CC --------------------------------------------------
    detections: list[dict] = []

    for lbl, bbox in enumerate(tqdm(bboxes, desc="Extracting"), start=1):
        if bbox is None:
            continue

        # work only within the bounding box
        lab_crop  = labeled[bbox]
        mask_crop = (lab_crop == lbl)

        size = int(mask_crop.sum())
        if size < size_thr:
            continue

        # score using indicator values inside bbox
        ind_crop = indicators[bbox]
        if score_type == "mean":
            score = float(ind_crop[mask_crop].mean())
        else:
            score = float(ind_crop[mask_crop].max())

        if score < score_thr:
            continue

        # postsynaptic location within bbox
        if loc_type == "centroid":
            zz, yy, xx = np.where(mask_crop)
            local_loc = np.array([zz.mean(), yy.mean(), xx.mean()])
        elif loc_type == "edt":
            edt_crop = distance_transform_edt(mask_crop)
            local_loc = np.array(np.unravel_index(edt_crop.argmax(), edt_crop.shape), dtype=float)
        else:  # peak
            masked = ind_crop * mask_crop
            local_loc = np.array(np.unravel_index(masked.argmax(), masked.shape), dtype=float)

        # convert local bbox coords back to volume coords
        bbox_origin = np.array([s.start for s in bbox])
        post_zyx = local_loc + bbox_origin

        # mean direction vector inside bbox
        vec_crop = vectors[:, bbox[0], bbox[1], bbox[2]]
        vz = float(vec_crop[0][mask_crop].mean())
        vy = float(vec_crop[1][mask_crop].mean())
        vx = float(vec_crop[2][mask_crop].mean())
        vec = np.array([vz, vy, vx])

        pre_zyx = post_zyx + vec
        if post_offset_scale != 0.0:
            post_zyx = post_zyx + vec * post_offset_scale
        if pre_offset_scale != 0.0:
            pre_zyx = pre_zyx + vec * pre_offset_scale

        # world coordinates (nm): local voxel → nm, then add block origin
        post_world = post_zyx * voxel_size + read_offset_nm + zarr_offset
        pre_world  = pre_zyx  * voxel_size + read_offset_nm + zarr_offset

        if flipprepost:
            post_world, pre_world = pre_world, post_world

        detections.append({
            "id":     len(detections),
            "post_z": float(post_world[0]),
            "post_y": float(post_world[1]),
            "post_x": float(post_world[2]),
            "pre_z":  float(pre_world[0]),
            "pre_y":  float(pre_world[1]),
            "pre_x":  float(pre_world[2]),
            "score":  score,
            "size":   size,
        })

    print(f"[extract] After size/score filter: {len(detections)} detections")

    # ---- NMS --------------------------------------------------------------
    if nms_radius > 0:
        detections = nms(detections, nms_radius)
        print(f"[extract] After NMS (r={nms_radius}): {len(detections)} detections")

    # re-index IDs
    for i, d in enumerate(detections):
        d["id"] = i

    _write_output(detections, params, cfg)


def _write_output(detections: list[dict], params: dict, cfg: dict) -> None:
    # JSON output
    to_json = params.get("to_json_config", {})
    out_name = to_json.get("output_name",
               cfg.get("inference_dir", ".") + "/synapses.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)
    with open(out_name, "w") as fh:
        json.dump({"synapses": detections, "n": len(detections)}, fh, indent=2)
    print(f"[extract] Wrote {len(detections)} synapses → {out_name}")

    # also write a simple CSV for easy loading
    csv_path = out_name.replace(".json", ".csv")
    with open(csv_path, "w") as fh:
        fh.write("id,post_z,post_y,post_x,pre_z,pre_y,pre_x,score,size\n")
        for d in detections:
            fh.write(f"{d['id']},{d['post_z']:.1f},{d['post_y']:.1f},"
                     f"{d['post_x']:.1f},{d['pre_z']:.1f},{d['pre_y']:.1f},"
                     f"{d['pre_x']:.1f},{d['score']:.4f},{d['size']}\n")
    print(f"[extract] Wrote CSV → {csv_path}")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter_logits_big.json"
    extract(params_path)