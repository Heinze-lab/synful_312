"""
optimize.py  —  Optuna-based hyperparameter search for synapse detection.

Usage:
    pip install optuna
    python optimize.py optimize_config.json

Each trial:
    1. Sample hyperparameters from the search space
    2. Train from scratch to 300k steps (warmup/baseline)
    3. Every 100k steps after 300k: predict on eval cubes → extract → F-score
    4. Stop early if F-score regresses vs previous checkpoint
    5. Report best F-score to Optuna → guides next trial

Results are written to:
    optimization_results.txt  — human-readable trial log
    optuna_study.db           — SQLite database (can resume with --resume)

Resume a crashed study:
    python optimize.py optimize_config.json --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_FILE = "optimization_results.txt"


def log_result(msg: str) -> None:
    """Write to both logger and results file."""
    logger.info(msg)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


# ---------------------------------------------------------------------------
# F-score (placeholder — replace with your implementation)
# ---------------------------------------------------------------------------

def compute_fscore(
    pred_synapses_path: str,   # path to extracted JSON from extract.py
    gt_csv_pre:         str,   # ground truth pre CSV
    gt_csv_post:        str,   # ground truth post CSV
    tolerance_vox:      float = 5.0,
) -> float:
    """
    TODO: replace this stub with your actual F-score computation.

    Should return a float in [0, 1] where 1.0 is perfect.

    The extracted JSON has structure:
        {"synapses": [{"id": 0, "post_z": ..., "post_y": ..., "post_x": ...,
                       "pre_z":  ..., "pre_y":  ..., "pre_x":  ...,
                       "score": ...}, ...]}
    """
    import random
    # PLACEHOLDER — returns a random score between 0.3 and 0.7
    # Replace with real implementation when ready
    logger.warning("compute_fscore: using placeholder implementation!")
    return random.uniform(0.3, 0.7)


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

def sample_params(trial, base_params: dict) -> dict:
    """
    Sample hyperparameters from Optuna trial.
    Returns a modified copy of base_params.
    """
    import optuna

    p = json.loads(json.dumps(base_params))   # deep copy

    # ── Loss balance ────────────────────────────────────────────────────────
    p["learning_rate"]  = trial.suggest_float("learning_rate",  1e-5, 2e-4, log=True)
    p["m_loss_scale"]   = trial.suggest_float("m_loss_scale",   10.0, 500.0, log=True)
    p["d_loss_scale"]   = trial.suggest_float("d_loss_scale",   1e-5, 1e-3, log=True)

    # ── Model architecture ──────────────────────────────────────────────────
    p["fmap_num"]        = trial.suggest_categorical("fmap_num",        [4, 6, 8])
    p["fmap_inc_factor"] = trial.suggest_categorical("fmap_inc_factor", [3, 4, 5])

    # ── Target rendering ────────────────────────────────────────────────────
    blob_rz  = trial.suggest_int("blob_rz",   1, 4)
    blob_ryx = trial.suggest_int("blob_ryx",  3, 10)
    p["blob_radius"] = [blob_rz, blob_ryx, blob_ryx]

    d_rz  = trial.suggest_int("d_blob_rz",  1, 4)
    d_ryx = trial.suggest_int("d_blob_ryx", 5, 20)
    p["d_blob_radius"] = [d_rz, d_ryx, d_ryx]

    # ── Rejection sampling ──────────────────────────────────────────────────
    p["reject_probability"] = trial.suggest_float("reject_probability", 0.7, 0.98)

    # ── Augmentation ────────────────────────────────────────────────────────
    aug = p.setdefault("augmentation", {})

    aug["intensity"] = {
        "enabled":        True,
        "scale_range":    [
            trial.suggest_float("intensity_scale_lo", 0.6, 0.9),
            trial.suggest_float("intensity_scale_hi", 1.1, 1.5),
        ],
        "shift_range":    [-0.15, 0.15],
        "z_section_wise": True,
    }
    aug["noise"] = {
        "enabled":   True,
        "var_range": [0.0, trial.suggest_float("noise_var_max", 0.05, 0.2)],
    }
    aug["defect"] = {
        "enabled":       True,
        "prob_missing":  trial.suggest_float("defect_prob_missing", 0.01, 0.08),
        "prob_dark":     trial.suggest_float("defect_prob_dark",    0.01, 0.08),
        "prob_shift":    trial.suggest_float("defect_prob_shift",   0.01, 0.08),
        "max_shift_px":  trial.suggest_int("max_shift_px", 8, 32),
    }
    aug["cutout"] = {
        "enabled":      True,
        "prob":         trial.suggest_float("cutout_prob", 0.2, 0.8),
        "n_holes":      trial.suggest_int("cutout_holes", 1, 4),
        "hole_size_yx": [
            trial.suggest_int("cutout_size_y", 10, 40),
            trial.suggest_int("cutout_size_x", 10, 40),
        ],
    }

    # ── Focal loss ──────────────────────────────────────────────────────────
    p["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)

    # ── Extraction thresholds ────────────────────────────────────────────────
    cc_thr = trial.suggest_float("cc_threshold", 0.3, 0.8)
    p["extract_configs"]["cc_threshold"] = cc_thr
    p["extract_configs"]["score_thr"]    = trial.suggest_float("score_thr", cc_thr, 0.95)
    p["extract_configs"]["size_thr"]     = trial.suggest_int("size_thr", 10, 80)
    p["extract_configs"]["nms_radius"]   = trial.suggest_int("nms_radius", 10, 50)

    return p


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def run_trial(
    trial_idx:   int,
    params:      dict,
    opt_cfg:     dict,
    trial_dir:   str,
) -> float:
    """
    Run a single trial:
        - train to max_iter, evaluating every eval_interval after warmup_iters
        - stop early if F-score regresses
        - return best F-score achieved
    """
    os.makedirs(trial_dir, exist_ok=True)

    warmup_iters  = opt_cfg.get("warmup_iters",   300_000)
    eval_interval = opt_cfg.get("eval_interval",  100_000)
    max_iters     = opt_cfg.get("max_iters",       700_000)
    device_num    = opt_cfg.get("device_num",      0)
    eval_cubes    = opt_cfg["eval_cubes"]          # list of {zarr, pre_csv, post_csv}
    python_exe    = opt_cfg.get("python", sys.executable)

    # ── patch params for this trial ─────────────────────────────────────────
    params["device_num"]    = device_num
    params["snapshot_dir"]  = os.path.join(trial_dir, "snapshots")
    params["tensorboard_dir"] = os.path.join(trial_dir, "tensorboard")
    params["model_name"]    = f"trial_{trial_idx:03d}"
    params["max_iteration"] = max_iters

    params_path = os.path.join(trial_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    log_result(f"\n{'='*60}")
    log_result(f"TRIAL {trial_idx:03d}  dir={trial_dir}")
    log_result(f"  lr={params['learning_rate']:.2e}  "
               f"m_scale={params['m_loss_scale']:.1f}  "
               f"d_scale={params['d_loss_scale']:.2e}  "
               f"fmap={params['fmap_num']}×{params['fmap_inc_factor']}")

    best_fscore    = 0.0
    prev_fscore    = 0.0
    eval_checkpoints = list(range(
        warmup_iters + eval_interval,
        max_iters + 1,
        eval_interval,
    ))

    # ── train to warmup ──────────────────────────────────────────────────────
    params["max_iteration"] = warmup_iters
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    log_result(f"  Training to warmup ({warmup_iters:,} steps)...")
    ret = subprocess.run(
        [python_exe, "train.py", params_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if ret.returncode != 0:
        log_result(f"  TRAIN FAILED at warmup (returncode={ret.returncode})")
        return 0.0

    # get baseline F-score at warmup
    prev_fscore = _evaluate(
        trial_idx, warmup_iters, params, params_path, eval_cubes, trial_dir, python_exe
    )
    log_result(f"  Baseline F-score at {warmup_iters:,}: {prev_fscore:.4f}")
    best_fscore = prev_fscore

    # ── train and evaluate every eval_interval ───────────────────────────────
    for target_iter in eval_checkpoints:
        params["max_iteration"] = target_iter
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        log_result(f"  Training to {target_iter:,}...")
        ret = subprocess.run(
            [python_exe, "train.py", params_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            log_result(f"  TRAIN FAILED at {target_iter:,} (returncode={ret.returncode})")
            break

        fscore = _evaluate(
            trial_idx, target_iter, params, params_path, eval_cubes, trial_dir, python_exe
        )
        log_result(f"  F-score at {target_iter:,}: {fscore:.4f}  "
                   f"(prev={prev_fscore:.4f}  best={best_fscore:.4f})")

        if fscore > best_fscore:
            best_fscore = fscore

        # early stop if regressed vs previous eval
        if fscore < prev_fscore:
            log_result(f"  ⚠ F-score regressed ({fscore:.4f} < {prev_fscore:.4f}) — stopping early")
            break

        prev_fscore = fscore

    log_result(f"  Trial {trial_idx:03d} complete — best F-score: {best_fscore:.4f}")

    # clean up checkpoints to save disk space (keep only best)
    _cleanup_checkpoints(params["snapshot_dir"], params["model_name"], best_fscore)

    return best_fscore


def _evaluate(
    trial_idx:   int,
    iteration:   int,
    params:      dict,
    params_path: str,
    eval_cubes:  list,
    trial_dir:   str,
    python_exe:  str,
) -> float:
    """Run predict + extract on all eval cubes, return mean F-score."""
    fscores = []

    for cube_idx, cube in enumerate(eval_cubes):
        cube_name   = f"trial_{trial_idx:03d}_iter_{iteration:07d}_cube_{cube_idx}"
        pred_zarr   = os.path.join(trial_dir, f"{cube_name}_pred.zarr")
        extract_json = os.path.join(trial_dir, f"{cube_name}_synapses.json")

        # patch predict config for this cube
        predict_params = json.loads(json.dumps(params))
        predict_params["predict"]["raw_file"]      = cube["zarr"]
        predict_params["predict"]["raw_dataset"]   = cube.get("raw_dataset", "RAW")
        predict_params["predict"]["read_offset"]   = cube.get("read_offset", [0, 0, 0])
        predict_params["predict"]["read_shape"]    = cube.get("read_shape",
            _get_zarr_shape(cube["zarr"], cube.get("raw_dataset", "RAW")))
        predict_params["predict"]["out_directory"] = trial_dir
        predict_params["predict"]["out_filename"]  = f"{cube_name}_pred.zarr"
        predict_params["predict"]["overwrite"]     = True

        # use latest checkpoint
        ckpt_dir  = params["snapshot_dir"]
        ckpt_num  = _find_latest_checkpoint(ckpt_dir, params["model_name"])
        if ckpt_num is None:
            log_result(f"  WARNING: no checkpoint found in {ckpt_dir}")
            return 0.0
        predict_params["predict"]["checkpoint_num"] = ckpt_num
        predict_params["predict"]["checkpoint_dir"] = ckpt_dir

        # patch extract config
        predict_params["extract_configs"]["inference_dir"]  = trial_dir
        predict_params["extract_configs"]["inference_file"] = f"{cube_name}_pred.zarr"
        predict_params["to_json_config"]["output_name"]     = extract_json

        cube_params_path = os.path.join(trial_dir, f"{cube_name}_params.json")
        with open(cube_params_path, "w") as f:
            json.dump(predict_params, f, indent=2)

        # predict
        ret = subprocess.run(
            [python_exe, "predict.py", cube_params_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            log_result(f"  WARNING: predict failed for cube {cube_idx}")
            fscores.append(0.0)
            continue

        # extract
        ret = subprocess.run(
            [python_exe, "extract.py", cube_params_path],
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if ret.returncode != 0:
            log_result(f"  WARNING: extract failed for cube {cube_idx}")
            fscores.append(0.0)
            continue

        # F-score
        fscore = compute_fscore(
            pred_synapses_path = extract_json,
            gt_csv_pre         = cube["pre_csv"],
            gt_csv_post        = cube["post_csv"],
            tolerance_vox      = cube.get("tolerance_vox", 5.0),
        )
        log_result(f"    cube {cube_idx}: F={fscore:.4f}  ({cube['zarr']})")
        fscores.append(fscore)

        # clean up pred zarr to save disk
        if os.path.exists(pred_zarr):
            shutil.rmtree(pred_zarr, ignore_errors=True)

    return float(np.mean(fscores)) if fscores else 0.0


def _get_zarr_shape(zarr_path: str, raw_dataset: str = "RAW") -> list:
    try:
        import zarr
        z = zarr.open(zarr_path, mode="r")
        return list(z[raw_dataset].shape[-3:])
    except Exception:
        return [100, 500, 500]


def _find_latest_checkpoint(ckpt_dir: str, model_name: str) -> int | None:
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = list(Path(ckpt_dir).glob(f"{model_name}_checkpoint_*.pt"))
    if not ckpts:
        return None
    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    return int(latest.stem.split("_")[-1])


def _cleanup_checkpoints(ckpt_dir: str, model_name: str, best_fscore: float) -> None:
    """Keep only the latest checkpoint, delete intermediate ones."""
    if not os.path.isdir(ckpt_dir):
        return
    ckpts = sorted(
        Path(ckpt_dir).glob(f"{model_name}_checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for ckpt in ckpts[:-1]:   # keep only the last one
        try:
            os.remove(ckpt)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def optimize(opt_cfg_path: str, resume: bool = False) -> None:
    try:
        import optuna
    except ImportError:
        print("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    with open(opt_cfg_path) as f:
        opt_cfg = json.load(f)

    base_params_path = opt_cfg["base_params"]
    with open(base_params_path) as f:
        base_params = json.load(f)

    n_trials    = opt_cfg.get("n_trials",     20)
    study_name  = opt_cfg.get("study_name",   "synful_opt")
    output_dir  = opt_cfg.get("output_dir",   "optimization_runs")
    db_path     = os.path.join(output_dir, "optuna_study.db")

    os.makedirs(output_dir, exist_ok=True)

    # write header to results file
    global RESULTS_FILE
    RESULTS_FILE = os.path.join(output_dir, "optimization_results.txt")
    if not resume:
        with open(RESULTS_FILE, "w") as f:
            f.write(f"Synful Hyperparameter Optimization\n")
            f.write(f"Started: {datetime.now()}\n")
            f.write(f"Base params: {base_params_path}\n")
            f.write(f"N trials: {n_trials}\n")
            f.write(f"Eval cubes: {[c['zarr'] for c in opt_cfg['eval_cubes']]}\n")
            f.write("="*60 + "\n\n")

    storage = f"sqlite:///{db_path}"
    if resume:
        study = optuna.load_study(study_name=study_name, storage=storage)
        log_result(f"Resuming study '{study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name    = study_name,
            storage       = storage,
            direction     = "maximize",
            sampler       = optuna.samplers.TPESampler(seed=42),
            load_if_exists= True,
        )

    trial_counter = [len(study.trials)]

    def objective(trial: "optuna.Trial") -> float:
        idx       = trial_counter[0]
        trial_counter[0] += 1
        trial_dir = os.path.join(output_dir, f"trial_{idx:03d}")

        params = sample_params(trial, base_params)

        log_result(f"\nStarting trial {idx:03d} (Optuna trial #{trial.number})")

        t0      = time.time()
        fscore  = run_trial(idx, params, opt_cfg, trial_dir)
        elapsed = time.time() - t0

        log_result(f"Trial {idx:03d} finished in {elapsed/3600:.1f}h — F-score: {fscore:.4f}")

        # write summary after every trial
        _write_summary(study, output_dir)

        return fscore

    log_result(f"Starting optimization: {n_trials} trials, output → {output_dir}")
    study.optimize(objective, n_trials=n_trials)

    # final summary
    log_result("\n" + "="*60)
    log_result("OPTIMIZATION COMPLETE")
    log_result(f"Best trial: #{study.best_trial.number}")
    log_result(f"Best F-score: {study.best_value:.4f}")
    log_result("Best params:")
    for k, v in study.best_params.items():
        log_result(f"  {k}: {v}")

    _write_summary(study, output_dir)

    # write best params as a ready-to-use JSON
    best_params = sample_params(study.best_trial, base_params)
    best_path   = os.path.join(output_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log_result(f"\nBest params saved to: {best_path}")


def _write_summary(study: "optuna.Study", output_dir: str) -> None:
    """Write a clean human-readable summary of all trials so far."""
    path = os.path.join(output_dir, "summary.txt")
    trials = [t for t in study.trials if t.value is not None]
    trials.sort(key=lambda t: t.value, reverse=True)

    with open(path, "w") as f:
        f.write(f"Optimization Summary  ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n")
        f.write(f"{len(trials)} completed trials\n\n")

        f.write(f"{'Rank':<6} {'Trial':<8} {'F-score':<10} "
                f"{'lr':<10} {'m_scale':<10} {'d_scale':<10} "
                f"{'fmap':<6} {'inc':<6} {'blob_ryx':<10} {'d_ryx':<8}\n")
        f.write("-" * 80 + "\n")

        for rank, t in enumerate(trials[:20], 1):
            p = t.params
            f.write(
                f"{rank:<6} {t.number:<8} {t.value:<10.4f} "
                f"{p.get('learning_rate', 0):<10.2e} "
                f"{p.get('m_loss_scale', 0):<10.1f} "
                f"{p.get('d_loss_scale', 0):<10.2e} "
                f"{p.get('fmap_num', 0):<6} "
                f"{p.get('fmap_inc_factor', 0):<6} "
                f"{p.get('blob_ryx', 0):<10} "
                f"{p.get('d_blob_ryx', 0):<8}\n"
            )

        if study.best_trial:
            f.write(f"\nBest params (trial #{study.best_trial.number}):\n")
            for k, v in sorted(study.best_params.items()):
                f.write(f"  {k}: {v}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synful hyperparameter optimization")
    parser.add_argument("config", help="Path to optimize_config.json")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing Optuna study")
    args = parser.parse_args()
    optimize(args.config, resume=args.resume)
