"""
predict.py  –  Fast, memory-bounded blockwise inference over a zarr volume.

Usage:
    python predict.py parameter_logits_big.json

Add to params["predict"]:
    "batch_size":      4          # blocks per GPU forward pass (tune to VRAM)
    "prefetch_blocks": 16         # read-ahead depth
    "overlap":         0.25       # fraction overlap between adjacent blocks
                                  # (scalar or [z,y,x]); 0 = no overlap (default)
                                  # overlap mode uses in-RAM accumulators per tile;
                                  # does not support crash-resume
    "overlap_tile_size": [108, 2048, 2048]
                                  # 3D tile size for in-RAM accumulators; tune so that
                                  # 5 * tile_z * tile_y * tile_x * 4 bytes <= target RAM
    "norm_global":     false      # if true, compute normalisation stats from volume sample
    "norm_p1":         x          # hardcode normalisation p1 (skips sampling)
    "norm_p99":        y          # hardcode normalisation p99
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import zarr
from tqdm import tqdm

from model import build_model


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(params, predict_cfg, device):
    model = build_model(params)
    ckpt_dir   = predict_cfg["checkpoint_dir"]
    ckpt_num   = predict_cfg["checkpoint_num"]
    model_name = predict_cfg.get("model_name", params.get("model_name", "model"))
    ckpt_path  = os.path.join(ckpt_dir, f"{model_name}_checkpoint_{ckpt_num}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    mtime = os.path.getmtime(ckpt_path)
    created = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[predict] Loading checkpoint: {ckpt_path}")
    print(f"[predict] Checkpoint saved:   {created}")
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    import re
    def _migrate(k):
        k = re.sub(r'encoder\.conv_blocks\.(\d+)', r'encoder.conv_\1', k)
        k = re.sub(r'(mask_decoder|vec_decoder)\.upsamples\.(\d+)', r'\1.up_\2', k)
        k = re.sub(r'(mask_decoder|vec_decoder)\.conv_blocks\.(\d+)', r'\1.conv_\2', k)
        return k
    if any("conv_blocks" in k or "upsamples" in k for k in sd):
        sd = {_migrate(k): v for k, v in sd.items()}
        print("[predict] Migrated checkpoint keys from old ModuleList names")
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Output zarr
# ---------------------------------------------------------------------------

def setup_output_zarr(out_path, read_sh, output_size, stride, n_blocks,
                      out_props, overwrite, overlap_mode,
                      offset=None, resolution=None):
    mode  = "w" if overwrite else "a"
    store = zarr.open(out_path, mode=mode)

    for cfg in out_props.values():
        dsname = cfg["dsname"]
        chunks = tuple(output_size.tolist())
        if dsname == "pred_syn_indicators":
            shape = tuple(read_sh.tolist())
        elif dsname == "pred_partner_vectors":
            shape  = (3,) + tuple(read_sh.tolist())
            chunks = (3,) + chunks
        else:
            shape = tuple(read_sh.tolist())
        if dsname not in store or overwrite:
            store.create_dataset(dsname, shape=shape, chunks=chunks,
                                 dtype=cfg["dtype"], overwrite=overwrite,
                                 fill_value=0)
            if offset is not None:
                store[dsname].attrs["offset"] = offset
            if resolution is not None:
                store[dsname].attrs["resolution"] = resolution
            print(f"[predict] Created '{dsname}' shape={shape} offset={offset} resolution={resolution}")

    if "blocks_done" not in store or overwrite or overlap_mode:
        store.create_dataset("blocks_done", shape=tuple(n_blocks.tolist()),
                             dtype=bool, overwrite=True, fill_value=False)
        print(f"[predict] Created 'blocks_done' shape={tuple(n_blocks.tolist())}")
    return store


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def compute_norm_stats(raw_ds, read_off, read_sh, n_samples=50, sample_size=None):
    """Sample n_samples small cubes and return (p1, p99) percentile stats."""
    sample_size = np.minimum(
        np.array(sample_size, dtype=int) if sample_size is not None else np.array([64, 64, 64]),
        read_sh,
    )
    rng = np.random.default_rng(42)
    chunks = []
    for _ in range(n_samples):
        max_off = np.maximum(read_sh - sample_size, 0)
        off = (rng.random(3) * max_off).astype(int) + read_off
        end = off + sample_size
        sl = tuple(slice(int(off[i]), int(end[i])) for i in range(3))
        cube = (raw_ds[sl] if raw_ds.ndim == 3 else raw_ds[0][sl]).astype(np.float32)
        chunks.append(cube.ravel())
    vals = np.concatenate(chunks)
    p1  = float(np.percentile(vals, 1))
    p99 = float(np.percentile(vals, 99))
    print(f"[predict] Norm stats ({n_samples} samples): p1={p1:.2f}  p99={p99:.2f}")
    return p1, p99


def normalise(arr: np.ndarray, p1=None, p99=None) -> np.ndarray:
    if p1 is None:
        lo, hi = arr.min(), arr.max()
        p1, p99 = float(lo), float(hi)
    arr = (arr.astype(np.float32) - p1) / max(p99 - p1, 1e-8)
    return arr.clip(0.0, 1.0) * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Taper window (overlap blending)
# ---------------------------------------------------------------------------

def make_taper(output_size: np.ndarray) -> np.ndarray:
    """3D separable cosine taper, always > 0 (half-sample shift avoids exact zeros)."""
    ws = []
    for n in output_size:
        n = int(n)
        w = np.sin(np.pi * (np.arange(n) + 0.5) / n).astype(np.float32)
        ws.append(w)
    return ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]


# ---------------------------------------------------------------------------
# Block geometry helpers
# ---------------------------------------------------------------------------

def block_input_slice(blk_idx, stride, output_size, read_off, read_sh, context, vol_full):
    out_off  = blk_idx * stride
    out_size = np.minimum(output_size, read_sh - out_off)
    if np.any(out_size <= 0):
        return None

    in_size    = output_size + 2 * context
    in_off_abs = read_off + out_off - context
    in_end_abs = in_off_abs + in_size
    in_off_c   = np.maximum(in_off_abs, 0)
    in_end_c   = np.minimum(in_end_abs, vol_full)

    if np.any(in_end_c <= in_off_c):
        return None

    return out_off, out_size, in_off_c, in_end_c, in_off_abs, in_end_abs


def read_block(raw_ds, geom, input_size, norm_p1, norm_p99):
    out_off, out_size, in_off_c, in_end_c, in_off_abs, in_end_abs = geom
    sl  = tuple(slice(int(in_off_c[i]), int(in_end_c[i])) for i in range(3))
    raw = (raw_ds[sl] if raw_ds.ndim == 3 else raw_ds[0][sl]).astype(np.float32)

    pad_before = (in_off_c - in_off_abs).astype(int)
    pad_after  = np.maximum((in_end_abs - in_end_c).astype(int), 0)
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        raw = np.pad(raw, list(zip(pad_before.tolist(), pad_after.tolist())),
                     mode="reflect")

    raw = raw[:input_size[0], :input_size[1], :input_size[2]]
    if raw.shape != tuple(input_size):
        shortfall = [(0, max(0, input_size[i] - raw.shape[i])) for i in range(3)]
        raw = np.pad(raw, shortfall, mode="reflect")
        raw = raw[:input_size[0], :input_size[1], :input_size[2]]

    raw = normalise(raw, norm_p1, norm_p99)
    return out_off, out_size, raw


# ---------------------------------------------------------------------------
# Reader  (multi-threaded zarr reads)
# ---------------------------------------------------------------------------

_SENTINEL = object()


def reader_worker(raw_ds, vol_full, read_off, read_sh, input_size,
                  output_size, stride, context, block_list,
                  read_q, batch_size, prefetch, norm_p1, norm_p99):
    """
    Reads blocks from block_list and produces batches of
    (out_offs, out_sizes, raw_stack) where raw_stack is (B,1,Z,Y,X) float32 pinned tensor.
    """
    num_read_workers = min(max(batch_size * 2, 4), 8)
    with ThreadPoolExecutor(max_workers=num_read_workers) as pool:
        import time as _time2

        result_q = queue.Queue()

        def _wrapped_read(raw_ds, geom, input_size):
            result = read_block(raw_ds, geom, input_size, norm_p1, norm_p99)
            result_q.put(result)

        idx_iter  = iter(block_list)
        in_flight = 0

        def _try_submit():
            nonlocal in_flight
            for blk_idx in idx_iter:
                geom = block_input_slice(blk_idx, stride, output_size,
                                         read_off, read_sh, context, vol_full)
                if geom is None:
                    continue
                pool.submit(_wrapped_read, raw_ds, geom, input_size)
                in_flight += 1
                return True
            return False

        for _ in range(num_read_workers):
            if not _try_submit():
                break

        batch = []
        while in_flight > 0:
            result = result_q.get()
            in_flight -= 1
            _try_submit()
            batch.append(result)
            if len(batch) >= batch_size:
                while read_q.qsize() >= prefetch:
                    _time2.sleep(0.005)
                read_q.put(([r[0] for r in batch],
                             [r[1] for r in batch],
                             torch.from_numpy(
                                 np.stack([r[2] for r in batch])[:, None]
                             ).pin_memory()))
                batch = []

        if batch:
            while read_q.qsize() >= prefetch:
                _time2.sleep(0.005)
            read_q.put(([r[0] for r in batch],
                         [r[1] for r in batch],
                         torch.from_numpy(
                             np.stack([r[2] for r in batch])[:, None]
                         ).pin_memory()))

    read_q.put(_SENTINEL)


# ---------------------------------------------------------------------------
# No-overlap writer (direct zarr write, supports crash-resume)
# ---------------------------------------------------------------------------

def writer_worker_direct(out_store, out_props, write_q, stride):
    while True:
        item = write_q.get()
        if item is _SENTINEL:
            break

        out_offs, out_sizes, masks, vecs = item

        for out_off, out_size, mask_np, vec_np in zip(out_offs, out_sizes, masks, vecs):
            out_sl = tuple(
                slice(int(out_off[i]), int(out_off[i] + out_size[i]))
                for i in range(3)
            )
            bd_idx = tuple(int(out_off[i]) // int(stride[i]) for i in range(3))

            for cfg in out_props.values():
                dsname = cfg["dsname"]
                scale  = cfg.get("scale", 1)
                if dsname == "pred_syn_indicators":
                    arr = (mask_np * scale).clip(0, 255).astype(cfg["dtype"])
                    out_store[dsname][out_sl] = arr
                elif dsname == "pred_partner_vectors":
                    if isinstance(scale, (list, tuple)):
                        sc = np.array(scale, dtype=np.float32)[:, None, None, None]
                    else:
                        sc = float(scale)
                    arr = (vec_np * sc).clip(-128, 127).astype(cfg["dtype"])
                    out_store[dsname][(slice(None),) + out_sl] = arr

            out_store["blocks_done"][bd_idx] = True


# ---------------------------------------------------------------------------
# Overlap mode: in-RAM tile accumulators
# ---------------------------------------------------------------------------

class TileAccum:
    """In-RAM float32 accumulators for one 3D tile of the output volume."""

    def __init__(self, tile_off, tile_sh):
        self.tile_off = tile_off.copy()
        self.tile_sh  = tile_sh.copy()
        sh = tuple(tile_sh.tolist())
        self.mask    = np.zeros(sh,        dtype=np.float32)
        self.weights = np.zeros(sh,        dtype=np.float32)
        self.vec     = np.zeros((3,) + sh, dtype=np.float32)

    def add(self, out_off, out_size, mask_np, vec_np, taper):
        # coordinates relative to this tile
        local_off = out_off - self.tile_off
        oz, oy, ox = int(out_size[0]), int(out_size[1]), int(out_size[2])
        lo = local_off
        hi = local_off + out_size

        # clip to tile bounds
        cl = np.maximum(lo, 0)
        ch = np.minimum(hi, self.tile_sh)
        if np.any(ch <= cl):
            return

        # corresponding slice into the prediction arrays
        pl = cl - lo
        ph = pl + (ch - cl)

        tsl  = (slice(int(cl[0]), int(ch[0])),
                slice(int(cl[1]), int(ch[1])),
                slice(int(cl[2]), int(ch[2])))
        psl  = (slice(int(pl[0]), int(ph[0])),
                slice(int(pl[1]), int(ph[1])),
                slice(int(pl[2]), int(ph[2])))
        w = taper[:oz, :oy, :ox][psl]

        self.mask[tsl]      += mask_np[psl] * w
        self.weights[tsl]   += w
        self.vec[(slice(None),) + tsl] += vec_np[(slice(None),) + psl] * w[None]

    def flush(self, out_store, out_props):
        safe_w = np.where(self.weights > 0, self.weights, 1.0)
        sl3 = (slice(int(self.tile_off[0]), int(self.tile_off[0] + self.tile_sh[0])),
               slice(int(self.tile_off[1]), int(self.tile_off[1] + self.tile_sh[1])),
               slice(int(self.tile_off[2]), int(self.tile_off[2] + self.tile_sh[2])))

        for cfg in out_props.values():
            dsname = cfg["dsname"]
            scale  = cfg.get("scale", 1)
            if dsname == "pred_syn_indicators":
                result = self.mask / safe_w
                out_store[dsname][sl3] = (result * scale).clip(0, 255).astype(cfg["dtype"])
            elif dsname == "pred_partner_vectors":
                result = self.vec / safe_w[None]
                if isinstance(scale, (list, tuple)):
                    sc = np.array(scale, dtype=np.float32)[:, None, None, None]
                else:
                    sc = float(scale)
                out_store[dsname][(slice(None),) + sl3] = (result * sc).clip(-128, 127).astype(cfg["dtype"])


def run_overlap_tile(tile_off, tile_sh, read_sh, read_off, vol_full,
                     input_size, output_size, stride, context,
                     taper, out_props, out_store,
                     raw_ds, model, device, batch_size, prefetch,
                     norm_p1, norm_p99, use_amp, pbar):
    """Run inference + blending for one 3D tile, keeping accumulators in RAM."""

    # Find all block indices whose output intersects this tile
    blk_lo = np.maximum(np.floor((tile_off) / stride).astype(int) - 1, 0)
    n_blocks_vol = np.ceil(read_sh / stride).astype(int)
    blk_hi = np.minimum(np.ceil((tile_off + tile_sh) / stride).astype(int) + 1,
                         n_blocks_vol)

    block_list = []
    for bz in range(int(blk_lo[0]), int(blk_hi[0])):
        for by in range(int(blk_lo[1]), int(blk_hi[1])):
            for bx in range(int(blk_lo[2]), int(blk_hi[2])):
                blk_idx = np.array([bz, by, bx], dtype=int)
                out_off = blk_idx * stride
                # check actual overlap with tile
                out_end = out_off + output_size
                if np.all(out_end > tile_off) and np.all(out_off < tile_off + tile_sh):
                    block_list.append(blk_idx)

    if not block_list:
        return

    accum = TileAccum(tile_off, tile_sh)

    read_q = queue.Queue(maxsize=prefetch)
    reader = threading.Thread(
        target=reader_worker,
        args=(raw_ds, vol_full, read_off, read_sh, input_size,
              output_size, stride, context, block_list,
              read_q, batch_size, prefetch, norm_p1, norm_p99),
        daemon=True,
    )
    reader.start()

    with torch.no_grad():
        while True:
            item = read_q.get()
            if item is _SENTINEL:
                break

            out_offs, out_sizes, raw_stack = item
            B = raw_stack.shape[0]

            raw_t = raw_stack.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_mask_t, pred_vec_t = model(raw_t)

            actual_out     = np.array(pred_mask_t.shape[2:], dtype=int)
            actual_context = (actual_out - output_size) // 2
            cz = int(actual_context[0])
            cy = int(actual_context[1])
            cx = int(actual_context[2])

            for b in range(B):
                oz = int(out_sizes[b][0])
                oy = int(out_sizes[b][1])
                ox = int(out_sizes[b][2])
                m = torch.sigmoid(
                    pred_mask_t[b, 0, cz:cz+oz, cy:cy+oy, cx:cx+ox]
                ).cpu().float().numpy()
                v = pred_vec_t[b, :, cz:cz+oz, cy:cy+oy, cx:cx+ox
                    ].cpu().float().numpy()
                accum.add(out_offs[b], out_sizes[b], m, v, taper)
                pbar.update(1)

    reader.join()
    accum.flush(out_store, out_props)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict_blockwise(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    cfg = params["predict"]

    device_num = int(cfg.get("device_num", 0))
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device_num)
    print(f"[predict] Device: {device}")

    model = load_model(params, cfg, device)

    if cfg.get("compile", True):
        try:
            compile_mode = cfg.get("compile_mode", "reduce-overhead")
            model = torch.compile(model, mode=compile_mode)
            print(f"[predict] Model compiled with torch.compile (mode={compile_mode})")
        except Exception:
            pass

    input_size  = np.array(cfg["input_size"],  dtype=int)
    output_size = np.array(cfg["output_size"], dtype=int)
    context     = (input_size - output_size) // 2
    assert np.all(context >= 0), "output_size must be ≤ input_size"

    # overlap / stride
    overlap_cfg = cfg.get("overlap", 0.0)
    if isinstance(overlap_cfg, (list, tuple)):
        overlap = np.array(overlap_cfg, dtype=float)
    else:
        overlap = np.array([float(overlap_cfg)] * 3)
    assert np.all((overlap >= 0) & (overlap < 1)), "overlap must be in [0, 1)"
    overlap_mode = np.any(overlap > 0)
    stride = np.maximum(np.round(output_size * (1.0 - overlap)).astype(int), 1)

    raw_file = zarr.open(cfg["raw_file"], mode="r")
    raw_ds   = raw_file[cfg["raw_dataset"]]
    vol_full = np.array(raw_ds.shape[-3:], dtype=int)
    read_off = np.array(cfg.get("read_offset", [0, 0, 0]), dtype=int)
    read_sh  = np.array(cfg.get("read_shape",  list(vol_full)), dtype=int)
    read_off = np.where(read_off < 0, 0,               read_off)
    read_sh  = np.where(read_sh  < 0, vol_full - read_off, read_sh)

    raw_offset     = list(raw_ds.attrs.get("offset",     [0, 0, 0]))
    raw_resolution = list(raw_ds.attrs.get("resolution", [1, 1, 1]))
    out_offset = [raw_offset[i] + read_off[i] * raw_resolution[i] for i in range(3)]

    out_path  = os.path.join(cfg.get("out_directory", "."),
                             cfg.get("out_filename", "pred.zarr"))
    overwrite = cfg.get("overwrite", False)
    out_props = cfg.get("out_properties", {})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    n_blocks     = np.ceil(read_sh / stride).astype(int)
    total_blocks = int(np.prod(n_blocks))

    out_store = setup_output_zarr(out_path, read_sh, output_size, stride,
                                  n_blocks, out_props, overwrite, overlap_mode,
                                  offset=out_offset, resolution=raw_resolution)

    # global norm stats
    if "norm_p1" in cfg and "norm_p99" in cfg:
        norm_p1, norm_p99 = float(cfg["norm_p1"]), float(cfg["norm_p99"])
        print(f"[predict] Norm stats (from config): p1={norm_p1:.2f}  p99={norm_p99:.2f}")
    elif cfg.get("norm_global", False):
        n_samples   = cfg.get("norm_samples", 50)
        sample_size = cfg.get("norm_sample_size", None)
        norm_p1, norm_p99 = compute_norm_stats(raw_ds, read_off, read_sh,
                                               n_samples=n_samples,
                                               sample_size=sample_size)
    else:
        norm_p1, norm_p99 = None, None

    batch_size = cfg.get("batch_size",     4)
    prefetch   = cfg.get("prefetch_blocks", 16)
    use_amp    = (device.type == "cuda")

    print(f"[predict] stride={stride.tolist()}  n_blocks={n_blocks.tolist()}  total={total_blocks}")
    print(f"[predict] batch_size={batch_size}  prefetch={prefetch}  amp={use_amp}")

    # -----------------------------------------------------------------------
    # No-overlap path: direct writes with crash-resume
    # -----------------------------------------------------------------------
    if not overlap_mode:
        blocks_done_arr = out_store["blocks_done"][:]
        n_done    = int(blocks_done_arr.sum())
        n_pending = total_blocks - n_done
        print(f"[predict] {total_blocks} blocks total, {n_done} already done, {n_pending} to process")

        if n_pending == 0:
            print("[predict] All blocks already complete.")
            return

        all_indices = [
            np.array(np.unravel_index(i, n_blocks), dtype=int)
            for i in range(total_blocks)
            if not blocks_done_arr[tuple(np.unravel_index(i, n_blocks))]
        ]

        read_q  = queue.Queue(maxsize=prefetch)
        write_q = queue.Queue()

        reader = threading.Thread(
            target=reader_worker,
            args=(raw_ds, vol_full, read_off, read_sh, input_size,
                  output_size, stride, context, all_indices,
                  read_q, batch_size, prefetch, norm_p1, norm_p99),
            daemon=True,
        )
        writer = threading.Thread(
            target=writer_worker_direct,
            args=(out_store, out_props, write_q, stride),
            daemon=True,
        )
        reader.start()
        writer.start()

        pbar = tqdm(total=n_pending, desc="Predicting", unit="blocks")
        import time as _time

        with torch.no_grad():
            while True:
                item = read_q.get()
                if item is _SENTINEL:
                    break

                out_offs, out_sizes, raw_stack = item
                B = raw_stack.shape[0]
                raw_t = raw_stack.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_mask_t, pred_vec_t = model(raw_t)

                actual_out     = np.array(pred_mask_t.shape[2:], dtype=int)
                actual_context = (actual_out - output_size) // 2
                cz, cy, cx = int(actual_context[0]), int(actual_context[1]), int(actual_context[2])

                masks_np, vecs_np = [], []
                for b in range(B):
                    oz, oy, ox = int(out_sizes[b][0]), int(out_sizes[b][1]), int(out_sizes[b][2])
                    m = torch.sigmoid(
                        pred_mask_t[b, 0, cz:cz+oz, cy:cy+oy, cx:cx+ox]
                    ).cpu().float().numpy()
                    v = pred_vec_t[b, :, cz:cz+oz, cy:cy+oy, cx:cx+ox].cpu().float().numpy()
                    masks_np.append(m)
                    vecs_np.append(v)

                write_q.put((out_offs, out_sizes, masks_np, vecs_np))
                pbar.update(B)

        write_q.put(_SENTINEL)
        writer.join()
        pbar.close()
        print(f"[predict] Done. Output: {out_path}")
        return

    # -----------------------------------------------------------------------
    # Overlap path: process one 3D tile at a time, accumulators in RAM
    # -----------------------------------------------------------------------
    print(f"[predict] Overlap mode: overlap={overlap.tolist()}  stride={stride.tolist()}")
    print(f"[predict] NOTE: crash-resume disabled in overlap mode")

    default_tile = [
        min(108, int(read_sh[0])),
        min(2048, int(read_sh[1])),
        min(2048, int(read_sh[2])),
    ]
    tile_cfg = cfg.get("overlap_tile_size", default_tile)
    tile_size = np.minimum(np.array(tile_cfg, dtype=int), read_sh)

    # estimate RAM
    accum_gb = float(np.prod(tile_size)) * 4 * 5 / 1e9
    print(f"[predict] Tile size: {tile_size.tolist()}  accum RAM per tile ≈ {accum_gb:.1f} GB")

    # Tile the output volume — tiles are non-overlapping regions of the output
    # (blocks that straddle tile boundaries are re-run per tile; cost is small)
    tile_offsets = []
    for tz in range(0, int(read_sh[0]), int(tile_size[0])):
        for ty in range(0, int(read_sh[1]), int(tile_size[1])):
            for tx in range(0, int(read_sh[2]), int(tile_size[2])):
                t_off = np.array([tz, ty, tx], dtype=int)
                t_sh  = np.minimum(tile_size, read_sh - t_off)
                tile_offsets.append((t_off, t_sh))

    n_tiles = len(tile_offsets)
    print(f"[predict] {n_tiles} tiles to process")

    taper = make_taper(output_size)

    pbar = tqdm(total=total_blocks * n_tiles // n_tiles, desc="Predicting", unit="blocks")
    # count blocks per tile for accurate progress
    total_work = sum(
        len([
            1 for bz in range(max(0, int(np.floor(t_off[0]/stride[0]))-1),
                              min(int(n_blocks[0]), int(np.ceil((t_off[0]+t_sh[0])/stride[0]))+1))
            for by in range(max(0, int(np.floor(t_off[1]/stride[1]))-1),
                            min(int(n_blocks[1]), int(np.ceil((t_off[1]+t_sh[1])/stride[1]))+1))
            for bx in range(max(0, int(np.floor(t_off[2]/stride[2]))-1),
                            min(int(n_blocks[2]), int(np.ceil((t_off[2]+t_sh[2])/stride[2]))+1))
            if np.all((np.array([bz,by,bx])*stride + output_size > t_off) &
                      (np.array([bz,by,bx])*stride < t_off + t_sh))
        ])
        for t_off, t_sh in tile_offsets
    )
    pbar.close()
    pbar = tqdm(total=total_work, desc="Predicting", unit="blocks")

    for tile_idx, (t_off, t_sh) in enumerate(tile_offsets):
        print(f"\n[predict] Tile {tile_idx+1}/{n_tiles}  off={t_off.tolist()}  sh={t_sh.tolist()}")
        run_overlap_tile(
            t_off, t_sh, read_sh, read_off, vol_full,
            input_size, output_size, stride, context,
            taper, out_props, out_store,
            raw_ds, model, device, batch_size, prefetch,
            norm_p1, norm_p99, use_amp, pbar,
        )

    pbar.close()
    print(f"[predict] Done. Output: {out_path}")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter_logits_big.json"
    predict_blockwise(params_path)
