"""
Detailed per-step profiling of the data augmentation pipeline.
Run from the training directory:  python profiling.py
"""
import time, json, numpy as np, zarr
from dataset import (build_dataset, SynfulDataset, load_points_csv,
                     render_syn_indicators, render_direction_vectors,
                     _elastic_context)
from augment import (simple_augment, intensity_augment, noise_augment,
                     defect_augment, elastic_augment, blur_augment,
                     gamma_augment, invert_augment, cutout_augment,
                     salt_pepper_augment, intensity_scale_shift)

with open('param_template.json') as f:
    params = json.load(f)

N = 20   # repetitions per step

# ── pick one sample with synapses ────────────────────────────────────────────
from dataset import build_sample_manifest as build_samples
samples = build_samples(params)
sample  = next(s for s in samples if s.has_synapses)

# ── compute sizes ─────────────────────────────────────────────────────────────
input_size = np.array(params['input_size'])
ctx        = _elastic_context(params)
load_size  = input_size + 2 * ctx

z_vol = zarr.open(sample.zarr_path, mode='r')['RAW']
max_start = np.array(z_vol.shape) - load_size
np.random.seed(0)
origin = np.array([np.random.randint(0, max(1, m)) for m in max_start])
sl     = tuple(slice(int(o), int(o+s)) for o, s in zip(origin, load_size))

aug = params.get('augmentation', {})

def T(fn, *args, **kwargs):
    """Run fn N times, return (result, mean_ms)."""
    fn(*args, **kwargs)   # warmup
    t0 = time.perf_counter()
    for _ in range(N):
        out = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) / N * 1000
    return out, ms

def row(label, ms, note=''):
    bar = '█' * int(ms / 2)
    print(f'  {label:<30s}  {ms:7.1f} ms  {bar}  {note}')

print(f'\nload_size  : {load_size}  (input {input_size} + 2×ctx {ctx})')
print(f'input_size : {input_size}')
print(f'N={N} repetitions each\n')
print('─' * 70)

# ── 1. zarr IO ───────────────────────────────────────────────────────────────
_, ms_io = T(lambda: z_vol[sl].astype(np.float32))
row('zarr IO + cast', ms_io)

raw_ctx = z_vol[sl].astype(np.float32)
raw_ctx = (raw_ctx - raw_ctx.min()) / ((raw_ctx.max() - raw_ctx.min()) + 1e-7)

# ── 2. point loading + coord transform ───────────────────────────────────────
post_abs = load_points_csv(sample.post_csv)
pre_abs  = load_points_csv(sample.pre_csv)
_, ms_pts = T(lambda: (
    (post_abs - sample.origin - origin).astype(int),
    (pre_abs  - sample.origin - origin).astype(int),
))
row('point coord transform', ms_pts)

post_loc = (post_abs - sample.origin - origin).astype(int)
pre_loc  = (pre_abs  - sample.origin - origin).astype(int)
r = np.array(params['d_blob_radius'])
in_crop  = np.all((post_loc >= -r) & (post_loc < load_size + r), axis=1)
post_loc = post_loc[in_crop]; pre_loc = pre_loc[in_crop]
ls_t = tuple(load_size)

# ── 3. label rendering ───────────────────────────────────────────────────────
_, ms_ind = T(render_syn_indicators, ls_t, post_loc, params['blob_radius'])
row('render indicator mask', ms_ind, f'({in_crop.sum()} synapses)')

_, ms_vec = T(render_direction_vectors, ls_t, post_loc, pre_loc, params['d_blob_radius'])
row('render direction vecs', ms_vec)

ind_ctx, _ = render_syn_indicators(ls_t, post_loc, params['blob_radius']), None
ind_ctx    = render_syn_indicators(ls_t, post_loc, params['blob_radius'])
vec_ctx, dw_ctx = render_direction_vectors(ls_t, post_loc, pre_loc, params['d_blob_radius'])

# helpers to get per-aug config
def cfg(key): return aug.get(key, {})

print('─' * 70)

# ── 4. augmentations ─────────────────────────────────────────────────────────
raw_w = raw_ctx.copy()
ind_w = ind_ctx.copy()
vec_w = vec_ctx.copy()
dw_w  = dw_ctx.copy()

_, ms = T(simple_augment, raw_w, ind_w, vec_w, dw_w)
row('simple_augment (flip/transp)', ms)

_, ms = T(intensity_augment, raw_w,
          scale_range=tuple(cfg('intensity').get('scale_range', [0.8,1.2])),
          shift_range=tuple(cfg('intensity').get('shift_range', [-0.15,0.15])))
row('intensity_augment', ms)

_, ms = T(noise_augment, raw_w,
          var_range=tuple(cfg('noise').get('var_range', [0.0,0.1])))
row('noise_augment', ms)

_, ms = T(defect_augment, raw_w,
          prob_missing=cfg('defect').get('prob_missing', 0.03),
          prob_dark   =cfg('defect').get('prob_dark',    0.03),
          prob_shift  =cfg('defect').get('prob_shift',   0.03))
row('defect_augment', ms)

ec = cfg('elastic')
_, ms = T(elastic_augment, raw_ctx, ind_ctx, vec_ctx, dw_ctx,
          control_point_spacing=ec.get('control_point_spacing', [1,50,50]),
          jitter_sigma         =ec.get('jitter_sigma',          [1,3.0,3.0]),
          prob_slip            =ec.get('prob_slip',  0.25),
          prob_shift           =ec.get('prob_shift', 0.25),
          prob_elastic         =1.0,
          correct_vectors      =False,
          context              =ctx)
row('elastic_augment (forced)', ms, '(prob_elastic=1 for timing)')

_, ms = T(blur_augment, raw_w,
          prob=cfg('blur').get('prob', 0.1),
          sigma_range=tuple(cfg('blur').get('sigma_range', [0.0,1.5])))
row('blur_augment', ms)

_, ms = T(gamma_augment, raw_w,
          gamma_range=tuple(cfg('gamma').get('gamma_range', [0.75,1.5])))
row('gamma_augment', ms)

_, ms = T(invert_augment, raw_w,
          prob=cfg('invert').get('prob', 0.01))
row('invert_augment', ms)

_, ms = T(cutout_augment, raw_w,
          prob=cfg('cutout').get('prob', 0.5),
          n_holes=cfg('cutout').get('n_holes', 2),
          hole_size_yx=tuple(cfg('cutout').get('hole_size_yx', [20,20])))
row('cutout_augment', ms)

_, ms = T(salt_pepper_augment, raw_w,
          prob=cfg('salt_pepper').get('prob', 0.001))
row('salt_pepper_augment', ms)

_, ms = T(intensity_scale_shift, raw_w)
row('intensity_scale_shift', ms)

print('─' * 70)

# ── 5. full single-threaded sample ───────────────────────────────────────────
ds = SynfulDataset(params, samples_per_epoch=N, augment=True)
_, ms_full = T(lambda: ds[0])
row('FULL sample (single thread)', ms_full)

# ── 6. GPU forward + backward ─────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader
from model import build_model
from train import combined_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = build_model(params).to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler    = torch.amp.GradScaler('cuda')

loader = DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)
batch  = next(iter(loader))
raw_t  = batch['raw'].to(device, dtype=torch.float32)
t_mask = batch['indicator_mask'].to(device, dtype=torch.float32)
t_vec  = batch['direction_vectors'].to(device, dtype=torch.float32)
dw_t   = batch['d_weight_mask'].to(device, dtype=torch.float32)
focal_gamma = float(params.get('focal_gamma', 2.0))
m_scale     = float(params.get('m_loss_scale', 1.0))
d_scale     = float(params.get('d_loss_scale', 1.0))

def gpu_step():
    with torch.amp.autocast('cuda'):
        pm, pv = model(raw_t)
        loss, _, _ = combined_loss(pm, pv, t_mask, t_vec, dw_t,
                                   m_scale, d_scale, 'sum', focal_gamma)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

# warmup
for _ in range(3):
    gpu_step()
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(N):
    gpu_step()
torch.cuda.synchronize()
ms_gpu = (time.perf_counter() - t0) / N * 1000
row('GPU forward+backward+step', ms_gpu)

# ── 7. projection ─────────────────────────────────────────────────────────────
TARGET = 500_000
num_workers = 8
ms_per_step_data = ms_full / num_workers   # pipelined across workers
ms_per_step      = max(ms_per_step_data, ms_gpu)
bottleneck       = 'DATA' if ms_per_step_data > ms_gpu else 'GPU'

print()
print('─' * 70)
print(f'  Projection ({TARGET//1000}k iterations, {num_workers} data workers)')
print(f'  data/step (÷{num_workers} workers)   {ms_per_step_data:.1f} ms')
print(f'  GPU step                         {ms_gpu:.1f} ms')
print(f'  effective step (bottleneck={bottleneck})  {ms_per_step:.1f} ms')
h = ms_per_step * TARGET / 3_600_000
print(f'  → {TARGET//1000}k steps ≈ {h:.1f} h')
print('─' * 70)
print()
