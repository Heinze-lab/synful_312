"""
train.py  –  Multi-task training for synaptic partner detection (setup03).

Usage:
    python train.py parameter_logits_big.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Reduce CUDA allocator fragmentation — must be set before any CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Allow TF32 on Ampere+ GPUs — faster matmuls with negligible precision loss.
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from dataset import build_dataset
from model import build_model
from augment import elastic_augment_gpu


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def center_crop(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Center-crop tensor t (B,C,Z,Y,X) to match target_shape's spatial dims."""
    for dim in range(2, t.dim()):
        diff  = t.shape[dim] - target_shape[dim]
        start = diff // 2
        t     = t.narrow(dim, start, target_shape[dim])
    return t


def crop_pred_and_target(pred, target, output_size):
    """Crop both pred and target to output_size from their centers (synful-style supervision)."""
    for dim, sz in enumerate(output_size):
        diff_p = pred.shape[dim + 2] - sz
        diff_t = target.shape[dim + 2] - sz
        pred   = pred.narrow(dim + 2, diff_p // 2, sz)
        target = target.narrow(dim + 2, diff_t // 2, sz)
    return pred, target


def mask_loss(pred, target, gamma=2.0, pos_weight=None, balance=False, balance_scale=1.0,
              output_size=None):
    """Sigmoid focal loss with optional per-crop BalanceLabels-style weighting.

    balance=False (default): use fixed scalar pos_weight (original behaviour).
    balance=True: compute per-crop frequency weights like gp.BalanceLabels —
        w_pos = (n_total / (2 * n_pos)) * balance_scale
        w_neg =  n_total / (2 * n_neg)
        balance_scale=1.0 → equal gradient contribution from pos and neg voxels
        balance_scale>1.0 → bias toward recall (more penalty on false negatives)
        balance_scale<1.0 → bias toward precision
    output_size: if given, crop both pred and target to this size (synful-style);
                 otherwise crop target to pred's shape (default, full-output supervision).
    """
    if output_size is not None:
        pred, target = crop_pred_and_target(pred, target, output_size)
    else:
        target = center_crop(target, pred.shape)
    pred = pred.clamp(-15.0, 15.0)

    if balance:
        # compute weights in float32 to avoid float16 overflow under AMP
        target_f32 = target.float()
        n_pos   = target_f32.sum().item()
        n_neg   = (1.0 - target_f32).sum().item()
        n_total = n_pos + n_neg
        # guard: if crop is empty of positives, use uniform weights
        if n_pos < 1.0:
            weight = torch.ones_like(target)
        else:
            # Clip weights to match synful's BalanceLabels clipmin/clipmax=[7e-4, 0.9993].
            # Converting frequency clips to weight clips: w = total/(2*freq),
            # so freq_min=7e-4 → w_max = 1/(2*7e-4) ≈ 714, freq_max=0.9993 → w_min ≈ 0.5.
            w_max = 1.0 / (2.0 * 7e-4)    # ≈ 714  (matches synful clipmin)
            w_min = 1.0 / (2.0 * 0.9993)  # ≈ 0.50 (matches synful clipmax)
            w_pos = float(np.clip((n_total / (2.0 * n_pos)) * balance_scale, w_min, w_max))
            w_neg = float(np.clip( n_total / (2.0 * max(n_neg, 1.0)),        w_min, w_max))
            weight = target_f32 * w_pos + (1.0 - target_f32) * w_neg
        ce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction='none'
        )
    else:
        pw = torch.tensor([pos_weight], device=pred.device, dtype=pred.dtype) if pos_weight else None
        ce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pw, reduction='none'
        )

    p   = torch.sigmoid(pred)
    p_t = target * p + (1.0 - target) * (1.0 - p)
    return (((1.0 - p_t) ** gamma) * ce).mean()


def direction_loss(pred, target, weight_mask, channel_weights=None, normalize_by_magnitude=False,
                   output_size=None):
    """MSE restricted to synapse blobs — computed in float32 to avoid float16 overflow."""
    pred = pred.float()
    if output_size is not None:
        pred, target = crop_pred_and_target(pred, target.float(), output_size)
        weight_mask  = center_crop(weight_mask.float(), pred.shape)
    else:
        target      = center_crop(target, pred.shape).float()
        weight_mask = center_crop(weight_mask, pred.shape).float()
    diff2 = (pred - target).pow(2) * weight_mask
    if channel_weights is not None:
        diff2 = diff2 * channel_weights.float().view(1, -1, 1, 1, 1)
    if normalize_by_magnitude:
        gt_mag = target.pow(2).sum(dim=1, keepdim=True).clamp(min=0.0).sqrt()
        diff2 = diff2 / (gt_mag + 1.0)
    n = weight_mask.sum() * pred.shape[1] + 1e-7
    return diff2.sum() / n


def combined_loss(pred_mask, pred_vec, t_mask, t_vec, d_weight,
                  m_scale, d_scale, comb_type, focal_gamma, channel_weights=None,
                  normalize_by_magnitude=False, pos_weight=None,
                  balance=False, balance_scale=1.0, output_size=None):
    m_loss = mask_loss(pred_mask.float(), t_mask.float(), focal_gamma, pos_weight, balance, balance_scale,
                       output_size=output_size)
    d_loss = direction_loss(pred_vec, t_vec, d_weight, channel_weights, normalize_by_magnitude,
                            output_size=output_size)
    if comb_type == "sum":
        total = m_scale * m_loss + d_scale * d_loss
    elif comb_type == "mean":
        total = (m_scale * m_loss + d_scale * d_loss) / 2.0
    else:
        raise ValueError(f"Unknown loss_comb_type: {comb_type!r}")
    return total, m_loss, d_loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(directory, model_name, iteration):
    return os.path.join(directory, f"{model_name}_checkpoint_{iteration}.pt")


def save_checkpoint(model, optimizer, scaler, scheduler, iteration, directory, model_name):
    os.makedirs(directory, exist_ok=True)
    path = _ckpt_path(directory, model_name, iteration)
    torch.save({
        "iteration":            iteration,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)
    print(f"[train] Saved checkpoint: {path}")


def load_latest_checkpoint(model, optimizer, scaler, scheduler, directory, model_name):
    if not os.path.isdir(directory):
        return 0
    ckpts = list(Path(directory).glob(f"{model_name}_checkpoint_*.pt"))
    if not ckpts:
        return 0
    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    print(f"[train] Resuming from {latest}")
    state = torch.load(latest, map_location="cpu")
    sd    = state["model_state_dict"]

    # torch.compile wraps parameters under an "_orig_mod." prefix.
    # Strip it when loading into a fresh (uncompiled) model, or add it back
    # if the current model is compiled but the checkpoint was saved uncompiled.
    has_prefix  = any(k.startswith("_orig_mod.") for k in sd)
    model_compiled = any(k.startswith("_orig_mod.") for k in model.state_dict())
    if has_prefix and not model_compiled:
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    elif not has_prefix and model_compiled:
        sd = {"_orig_mod." + k: v for k, v in sd.items()}

    # Migrate old ModuleList key names → new named-submodule key names
    import re
    def _migrate(k):
        k = re.sub(r'encoder\.conv_blocks\.(\d+)', r'encoder.conv_\1', k)
        k = re.sub(r'(mask_decoder|vec_decoder)\.upsamples\.(\d+)', r'\1.up_\2', k)
        k = re.sub(r'(mask_decoder|vec_decoder)\.conv_blocks\.(\d+)', r'\1.conv_\2', k)
        return k
    migrated = any("conv_blocks" in k or "upsamples" in k for k in sd)
    if migrated:
        sd = {_migrate(k): v for k, v in sd.items()}
        print("[train] Migrated checkpoint keys from old ModuleList names")

    model.load_state_dict(sd)
    if not migrated:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if "scaler_state_dict" in state:
                scaler.load_state_dict(state["scaler_state_dict"])
        except (RuntimeError, ValueError) as e:
            print(f"[train] WARNING: optimizer state incompatible ({e}); "
                  f"resuming with fresh optimizer (weights are preserved)")
    else:
        print("[train] Skipping optimizer state (old ModuleList checkpoint — moments reset, weights kept)")
    if "scheduler_state_dict" in state and scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return state["iteration"]


# ---------------------------------------------------------------------------
# Tensorboard image logging
# ---------------------------------------------------------------------------


def _stack_to_rgb(vol: np.ndarray) -> np.ndarray:
    """(Z,Y,X) → (Z,3,Y,X) greyscale-as-RGB, normalised to [0,1]."""
    lo, hi = vol.min(), vol.max()
    v = (vol - lo) / max(hi - lo, 1e-8)
    return np.stack([v, v, v], axis=1).astype(np.float32)


def _overlay_stack(raw: np.ndarray, mask: np.ndarray,
                   color: tuple = (1.0, 0.2, 0.2)) -> np.ndarray:
    """
    (Z,Y,X) raw + (Z,Y,X) mask → (Z,3,Y,X) RGB stack, normalised.
    Mask regions are tinted with `color`.
    """
    lo, hi = raw.min(), raw.max()
    raw_n = (raw - lo) / max(hi - lo, 1e-8)
    m = np.clip(mask, 0.0, 1.0)
    r = raw_n * (1 - m) + color[0] * m
    g = raw_n * (1 - m) + color[1] * m
    b = raw_n * (1 - m) + color[2] * m
    return np.stack([r, g, b], axis=1).astype(np.float32)  # (Z,3,Y,X)


def log_images(
    writer:    "SummaryWriter",
    batch:     dict,
    pred_mask: torch.Tensor,
    pred_vec:  torch.Tensor,
    iteration: int,
) -> None:
    """Write z-scrollable image stacks to tensorboard (add_images with slider)."""
    pred_sh = pred_mask.shape[2:]   # (Z, Y, X)

    def _crop_np(arr):
        slices = [slice(None)] * (arr.ndim - 3)
        for a, t in zip(arr.shape[-3:], pred_sh):
            s = (a - t) // 2
            slices.append(slice(s, s + t))
        return arr[tuple(slices)]

    raw_np  = _crop_np(batch["raw"][0, 0].cpu().numpy())
    gt_mask = _crop_np(batch["indicator_mask"][0, 0].cpu().numpy())
    gt_vec  = _crop_np(batch["direction_vectors"][0].cpu().numpy())

    p_mask = torch.sigmoid(pred_mask[0, 0]).detach().cpu().float().numpy()
    p_vec  = pred_vec[0].detach().cpu().float().numpy()

    # raw EM — greyscale z-stack slider
    writer.add_images("raw",            _stack_to_rgb(raw_np),  iteration)

    # standalone mask sliders
    writer.add_images("gt/indicator",   _stack_to_rgb(gt_mask), iteration)
    writer.add_images("pred/indicator", _stack_to_rgb(p_mask),  iteration)

    # overlays: GT=red, pred=green on raw EM
    writer.add_images("overlay/gt_on_raw",   _overlay_stack(raw_np, gt_mask, color=(1.0, 0.2, 0.2)), iteration)
    writer.add_images("overlay/pred_on_raw", _overlay_stack(raw_np, p_mask,  color=(0.2, 1.0, 0.2)), iteration)

    # vector magnitude sliders
    gt_mag = np.sqrt((gt_vec ** 2).sum(axis=0))   # (Z,Y,X)
    p_mag  = np.sqrt((p_vec  ** 2).sum(axis=0))
    writer.add_images("gt/vec_mag",   _stack_to_rgb(gt_mag), iteration)
    writer.add_images("pred/vec_mag", _stack_to_rgb(p_mag),  iteration)

    # per-axis GT vector components
    for i, ax in enumerate(["z", "y", "x"]):
        writer.add_images(f"gt/vec_{ax}", _stack_to_rgb(gt_vec[i]), iteration)


# ---------------------------------------------------------------------------
# Snapshot  (matches gunpowder Snapshot output format)
# ---------------------------------------------------------------------------

def save_snapshot(
    batch:      dict,
    pred_mask:  torch.Tensor,
    pred_vec:   torch.Tensor,
    iteration:  int,
    directory:  str,
) -> None:
    """
    Save a training snapshot to HDF5, matching gunpowder's Snapshot format.

    Datasets written:
        volumes/raw                  (Z, Y, X)  float32  [-1, 1]
        volumes/gt_post_indicator    (Z, Y, X)  float32  binary
        volumes/gt_postpre_vectors   (3, Z, Y, X) float32 unit vectors
        volumes/pred_post_indicator  (Z, Y, X)  float32  sigmoid probs
        volumes/pred_postpre_vectors (3, Z, Y, X) float32
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"batch_{iteration:08d}.hdf")

    p_mask  = torch.sigmoid(pred_mask[0, 0]).detach().cpu().float().numpy()
    p_vec   = pred_vec[0].detach().cpu().float().numpy()
    pred_sh = p_mask.shape

    def center_crop_np(arr, target):
        """Crop numpy array (or 3D/4D) to target spatial shape from centre."""
        slices = [slice(None)] * (arr.ndim - 3)
        for a, t in zip(arr.shape[-3:], target):
            start = (a - t) // 2
            slices.append(slice(start, start + t))
        return arr[tuple(slices)]

    raw_np  = center_crop_np(batch["raw"][0, 0].cpu().numpy(),               pred_sh)
    gt_mask = center_crop_np(batch["indicator_mask"][0, 0].cpu().numpy(),    pred_sh)
    gt_vec  = center_crop_np(batch["direction_vectors"][0].cpu().numpy(),    pred_sh)

    with h5py.File(path, "w") as f:
        f.create_dataset("volumes/raw",                  data=raw_np,  compression="gzip")
        f.create_dataset("volumes/gt_post_indicator",    data=gt_mask, compression="gzip")
        f.create_dataset("volumes/gt_postpre_vectors",   data=gt_vec,  compression="gzip")
        f.create_dataset("volumes/pred_post_indicator",  data=p_mask,  compression="gzip")
        f.create_dataset("volumes/pred_postpre_vectors", data=p_vec,   compression="gzip")

    print(f"[train] Snapshot saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(params_path: str) -> None:
    with open(params_path) as fh:
        params = json.load(fh)

    # ---- device -----------------------------------------------------------
    device_num = params.get("device_num", 0)
    device = torch.device(
        f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device_num)
    print(f"[train] Device: {device}")

    # ---- model ------------------------------------------------------------
    model = build_model(params).to(device)

    # torch.compile — optional, off by default (inductor can allocate huge intermediate
    # buffers during training that cause OOM after a few thousand iterations)
    if params.get("compile", False):
        compile_mode = params.get("compile_mode", "default")
        print(f"[train] Compiling model with torch.compile (mode={compile_mode}) ...")
        model = torch.compile(model, mode=compile_mode)
    else:
        print("[train] torch.compile disabled")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {n_params:,}")

    # ---- optimiser (Adam beta1=0.95, matching original) -------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(0.95, 0.999),
        eps=1e-7,
    )

    # ---- scheduler — fixed warmup + cosine decay (no total_steps dependency) -----
    max_iter         = params.get("max_iteration", 1_000_000)
    warmup_steps     = int(params.get("warmup_steps",  8_000))
    cosine_period    = int(params.get("cosine_period", 2_000_000))
    final_div_factor = float(params.get("final_div_factor", 100.0))
    lr_min_factor    = 1.0 / final_div_factor

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / cosine_period
        cosine   = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_min_factor + (1.0 - lr_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- AMP --------------------------------------------------------------
    use_amp = params.get("use_amp", True) and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)


    # ---- names / dirs -----------------------------------------------------
    model_name   = params.get("model_name",   "model")
    snapshot_dir = params.get("snapshot_dir", "snapshots")

    # ---- resume -----------------------------------------------------------
    start_iter = load_latest_checkpoint(
        model, optimizer, scaler, scheduler, snapshot_dir, model_name
    )

    # ---- dataset & loader -------------------------------------------------
    # Scale epoch length with total training budget: ~1% of max_iter per epoch,
    # clamped to [1000, 10000] so short and long runs both get reasonable epoch sizes.
    samples_per_epoch = max(1000, min(10_000, max_iter // 100))

    dataset = build_dataset(params, samples_per_epoch=samples_per_epoch)
    loader  = DataLoader(
        dataset,
        batch_size=params.get("batch_size", 1),
        num_workers=params.get("num_data_workers", 8),
        pin_memory=(device.type == "cuda"),
        prefetch_factor=4,
        persistent_workers=True,
    )

    # ---- GPU elastic augmentation -----------------------------------------
    gpu_elastic     = bool(params.get("gpu_elastic", False))
    elastic_cfg     = params.get("augmentation", {}).get("elastic", {})
    elastic_cps     = elastic_cfg.get("control_point_spacing", [50, 10, 10])
    elastic_sigma   = elastic_cfg.get("jitter_sigma",          [0, 4.0, 4.0])
    elastic_pslip   = elastic_cfg.get("prob_slip",             0.25)
    elastic_pshift  = elastic_cfg.get("prob_shift",            0.25)
    elastic_pelastic= elastic_cfg.get("prob_elastic", elastic_cfg.get("apply_prob", 0.2))

    # ---- loss hyper-params ------------------------------------------------
    m_scale     = float(params.get("m_loss_scale", 1.0))
    d_scale     = float(params.get("d_loss_scale", 1.0))
    comb_type   = params.get("loss_comb_type", "sum")
    focal_gamma = float(params.get("focal_gamma", 2.0))
    # Optional gamma schedule: ramp to a new gamma value at a given iteration.
    # Config: "focalgamma_schedule": {"step": 425000, "gamma": 1.0}
    gamma_schedule = params.get("focalgamma_schedule", None)
    _cw = params.get("vec_channel_weights", [1.0, 1.0, 1.0])
    channel_weights = torch.tensor(_cw, dtype=torch.float32).to(device)
    normalize_by_magnitude = bool(params.get("vec_normalize_by_magnitude", False))
    pos_weight    = params.get("mask_pos_weight", None)
    if pos_weight is not None:
        pos_weight = float(pos_weight)
    balance       = bool(params.get("balance_labels", False))
    balance_scale = float(params.get("balance_scale", 1.0))

    # synful-style center-crop supervision: supervise only output_size region
    # disabled by default (full model output supervised); enable with "supervise_output_size": true
    supervise_output_size = None
    if params.get("supervise_output_size", False):
        _out = params.get("output_size", None)
        if _out is None:
            raise ValueError("supervise_output_size requires 'output_size' in params")
        supervise_output_size = tuple(int(x) for x in _out)
        print(f"[train] supervise_output_size={supervise_output_size} (synful-style center crop)")
    else:
        print(f"[train] supervise_output_size=disabled (full model output supervised)")

    # ---- tensorboard ------------------------------------------------------
    tb_dir = params.get("tensorboard_dir", "tensorboard")
    writer = SummaryWriter(tb_dir)

    save_every      = params.get("save_every",      10_000)
    log_every       = params.get("log_every",       100)
    snapshot_every  = params.get("snapshot_every",  25_000)
    hist_every      = params.get("hist_every",      2_000)   # grad/weight histograms

    # ---- tensorboard: model graph -----------------------------------------
    # Logged once at the very start (or resume). Uses a dummy input on CPU to
    # avoid perturbing the training device state.
    try:
        _dummy = torch.zeros(1, 1, *params["input_size"], dtype=torch.float32)
        _graph_model = build_model(params)   # fresh uncompiled copy — add_graph needs hooks
        writer.add_graph(_graph_model, _dummy)
        del _dummy, _graph_model
        print("[train] Model graph written to TensorBoard.")
    except Exception as _e:
        pass  # TensorBoard tracing is flaky with grad_checkpoint; skip silently

    # ---- tensorboard: custom scalars layout --------------------------------
    # Groups scalars into named panels in the TensorBoard UI.
    layout = {
        "Loss": {
            "total + smoothed":  ["Multiline", ["loss/total", "loss/total_ema"]],
            "components":        ["Multiline", ["loss/mask",  "loss/direction"]],
        },
        "Gradient": {
            "global norm":       ["Multiline", ["grad/global_norm"]],
            "max layer norm":    ["Multiline", ["grad/max_layer_norm"]],
        },
        "Training": {
            "learning rate":     ["Multiline", ["lr"]],
            "amp scale":         ["Multiline", ["amp/scale"]],
            "throughput (it/s)": ["Multiline", ["perf/iter_per_sec"]],
        },
        "Data": {
            "positive voxel fraction": ["Multiline", ["data/pos_vox_frac"]],
        },
        "Predictions": {
            "vec magnitude (mean)": ["Multiline",
                                     ["pred/vec_mag_z", "pred/vec_mag_y", "pred/vec_mag_x"]],
        },
    }
    writer.add_custom_scalars(layout)

    # ---- tensorboard: hyperparameters -------------------------------------
    # Flatten the params dict to scalar hparams TensorBoard can display.
    hparam_keys = [
        "learning_rate", "batch_size", "fmap_num", "fmap_inc_factor",
        "m_loss_scale", "d_loss_scale", "focal_gamma", "balance_labels",
        "balance_scale", "mask_pos_weight", "warmup_steps", "cosine_period",
        "final_div_factor", "grad_clip", "use_amp", "gpu_elastic",
        "num_data_workers", "kernel_size", "norm_type",
    ]
    hparams = {k: params[k] for k in hparam_keys if k in params}
    # Booleans must be cast to int for TensorBoard hparam display
    hparams = {k: (int(v) if isinstance(v, bool) else v)
               for k, v in hparams.items() if isinstance(v, (int, float, str, bool))}
    # We'll call writer.add_hparams at the end of training with final metric values.

    # ---- loop -------------------------------------------------------------
    iteration = start_iter
    model.train()
    data_iter = iter(loader)

    # Apply gamma schedule immediately if resuming past the scheduled step.
    gamma_applied = False
    if gamma_schedule and iteration >= gamma_schedule["step"]:
        focal_gamma = float(gamma_schedule["gamma"])
        gamma_applied = True
        print(f"[train] focal_gamma → {focal_gamma} (schedule already passed at iter {gamma_schedule['step']})")

    print(f"[train] Starting from iteration {iteration} / {max_iter}")
    if balance:
        print(f"[train] balance_labels=True  balance_scale={balance_scale}  (pos_weight ignored)")
    else:
        print(f"[train] balance_labels=False  pos_weight={pos_weight}")

    loss_ema         = None   # exponential moving average of total loss
    ema_alpha        = 0.98   # smoothing factor (higher = smoother)
    t_last           = time.perf_counter()
    iters_since_log  = 0

    grad_clip = params.get("grad_clip", 1.0)

    while iteration < max_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        raw      = batch["raw"].to(device, non_blocking=True)
        t_mask   = batch["indicator_mask"].to(device, non_blocking=True)
        t_vec    = batch["direction_vectors"].to(device, non_blocking=True)
        d_weight = batch["d_weight_mask"].to(device, non_blocking=True)

        if gpu_elastic:
            with torch.no_grad():
                raw, t_mask, t_vec, d_weight = elastic_augment_gpu(
                    raw, t_mask, t_vec, d_weight,
                    control_point_spacing = elastic_cps,
                    jitter_sigma          = elastic_sigma,
                    prob_slip             = elastic_pslip,
                    prob_shift            = elastic_pshift,
                    prob_elastic          = elastic_pelastic,
                )

        # Apply gamma schedule if configured (fires exactly once at the step boundary)
        if gamma_schedule and not gamma_applied and iteration >= gamma_schedule["step"]:
            focal_gamma = float(gamma_schedule["gamma"])
            gamma_applied = True
            print(f"[train] focal_gamma → {focal_gamma} at iteration {iteration}")
            writer.add_scalar("focal_gamma", focal_gamma, iteration)

        optimizer.zero_grad(set_to_none=True)

        try:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_mask, pred_vec = model(raw)
                loss, m_loss, d_loss = combined_loss(
                    pred_mask, pred_vec, t_mask, t_vec, d_weight,
                    m_scale, d_scale, comb_type, focal_gamma, channel_weights,
                    normalize_by_magnitude, pos_weight,
                    balance, balance_scale,
                    output_size=supervise_output_size,
                )
        except torch.OutOfMemoryError:
            print(f"[train] WARNING: OOM at iter {iteration} — freeing cache and skipping batch")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            iteration += 1
            iters_since_log += 1
            continue

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"[train] WARNING: non-finite loss={loss_val:.4g} at iter {iteration} "
                  f"(mask={m_loss.item():.4g} vec={d_loss.item():.4g}) — skipping batch")
            print(f"[train]   pred_mask: min={pred_mask.min().item():.3g} max={pred_mask.max().item():.3g} "
                  f"nan={pred_mask.isnan().any().item()} inf={pred_mask.isinf().any().item()}")
            print(f"[train]   raw: min={raw.min().item():.3g} max={raw.max().item():.3g} "
                  f"nan={raw.isnan().any().item()}")
            print(f"[train]   t_mask: min={t_mask.min().item():.3g} max={t_mask.max().item():.3g} "
                  f"nan={t_mask.isnan().any().item()}")
            optimizer.zero_grad(set_to_none=True)
            del loss, m_loss, d_loss, pred_mask, pred_vec
            torch.cuda.empty_cache()
            iteration += 1
            iters_since_log += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Capture grad norm before clipping so we can log the unclipped value.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        iteration += 1
        iters_since_log += 1

        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = ema_alpha * loss_ema + (1.0 - ema_alpha) * loss_val

        if iteration % log_every == 0:
            l, ml, dl = loss_val, m_loss.item(), d_loss.item()
            lr = scheduler.get_last_lr()[0]

            t_now    = time.perf_counter()
            elapsed  = max(t_now - t_last, 1e-6)
            it_per_s = iters_since_log / elapsed
            t_last   = t_now
            iters_since_log = 0

            print(f"[{iteration:>8}/{max_iter}]  loss={l:.5f}  ema={loss_ema:.5f}  "
                  f"mask={ml:.5f}  vec={dl:.5f}  lr={lr:.2e}  "
                  f"gnorm={grad_norm:.3f}  {it_per_s:.1f}it/s")

            # ---- scalar logging ----
            writer.add_scalar("loss/total",        l,          iteration)
            writer.add_scalar("loss/total_ema",    loss_ema,   iteration)
            writer.add_scalar("loss/mask",         ml,         iteration)
            writer.add_scalar("loss/direction",    dl,         iteration)
            writer.add_scalar("lr",                lr,         iteration)
            writer.add_scalar("grad/global_norm",  float(grad_norm),       iteration)
            writer.add_scalar("amp/scale",         scaler.get_scale(),     iteration)
            writer.add_scalar("perf/iter_per_sec", it_per_s,               iteration)

            # positive voxel fraction in GT mask (measures class imbalance)
            pos_frac = center_crop(t_mask, pred_mask.shape).float().mean().item()
            writer.add_scalar("data/pos_vox_frac", pos_frac, iteration)

            # mean predicted vector magnitude per axis
            with torch.no_grad():
                pv = pred_vec.float().detach()
                writer.add_scalar("pred/vec_mag_z", pv[:, 0].abs().mean().item(), iteration)
                writer.add_scalar("pred/vec_mag_y", pv[:, 1].abs().mean().item(), iteration)
                writer.add_scalar("pred/vec_mag_x", pv[:, 2].abs().mean().item(), iteration)

        if iteration % hist_every == 0:
            # Log weight and gradient histograms per layer.
            # max_layer_norm tracks the largest single-layer grad — useful for spotting
            # exploding/vanishing gradients in specific parts of the network.
            max_layer_norm = 0.0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    safe = name.replace(".", "/")
                    writer.add_histogram(f"weights/{safe}", param.detach().float(), iteration)
                    if param.grad is not None:
                        g = param.grad.detach().float()
                        writer.add_histogram(f"grads/{safe}", g, iteration)
                        max_layer_norm = max(max_layer_norm, g.norm().item())
            writer.add_scalar("grad/max_layer_norm", max_layer_norm, iteration)

        if iteration % snapshot_every == 0:
            save_snapshot(batch, pred_mask, pred_vec, iteration, snapshot_dir)
            log_images(writer, batch, pred_mask, pred_vec, iteration)

            # PR curve: precision-recall over this batch's predictions vs GT.
            with torch.no_grad():
                gt_cropped = center_crop(t_mask, pred_mask.shape)
                probs  = torch.sigmoid(pred_mask).float().cpu().flatten()
                labels = gt_cropped.float().cpu().flatten().long()
                writer.add_pr_curve("pr/indicator", labels, probs, iteration)

        if iteration % save_every == 0:
            save_checkpoint(model, optimizer, scaler, scheduler, iteration, snapshot_dir, model_name)

    save_checkpoint(model, optimizer, scaler, scheduler, iteration, snapshot_dir, model_name)

    # ---- final hparams entry -----------------------------------------------
    # Links the run's hyperparameters to its final metrics in the HPARAMS tab.
    # loss_val/m_loss/d_loss are only defined if at least one iteration ran.
    if loss_ema is not None:
        writer.add_hparams(
            hparams,
            {
                "hparam/final_loss":      loss_val,
                "hparam/final_loss_ema":  loss_ema,
                "hparam/final_mask_loss": m_loss.item(),
                "hparam/final_vec_loss":  d_loss.item(),
            },
        )

    writer.close()
    print("[train] Done.")


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "parameter_logits_big.json"
    train(params_path)