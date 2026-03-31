# Synaptic Partner Detection - rewrite of [Synful](https://github.com/funkelab/synful)
This is a (mostly) TensorFlow implementation of the dual-headed UNET architecture from Synful. Complete with model architecture, augmentations, and pulling from data. This is so far only tested with zarr files. 

Training, prediction, and extraction pipeline for synapse detection and partner vector prediction.

## Environment Setup

Requires CUDA 12.4 and conda.

```bash
conda create -n synpred python=3.10 -y
conda activate synpred
```

### PyTorch (CUDA 12.4)

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Core dependencies

```bash
pip install \
    numpy==1.24.3 \
    zarr==2.18.4 \
    h5py \
    scipy \
    pandas \
    tqdm \
    tensorboard \
    neuroglancer==2.41.2
```

### funlib stack

```bash
pip install \
    funlib-geometry==0.3.0 \
    funlib-math==0.1 \
    funlib-persistence==0.5.4 \
    funlib.segment \
    daisy==1.0
```

## Pipeline

```
predict.py → extract_daisy.py 
```

## Key files

| File | Purpose |
|---|---|
| `dataset.py` | Data loading + GT rendering |
| `model.py` | U-Net architecture |
| `train.py` | Training loop |
| `predict.py` | Blockwise inference over zarr |
| `extract_daisy.py` | Daisy-based chunked synapse extraction (faster and with daisy) |
| `extract.py` | Single-machine extraction (small volumes, slower but no daisy requirement) |

## Testing files

| File | Purpose |
|---|---|
| `view_snap.ipynb` | Viewer for snapshots produced during training|
| `pred_view.ipynb` | Output viewer for predicted volumes |
| `profiling.py` | Profiler, runs a single test with paramter file defined in it, to check for speed and compiling on GPU |

## Parameter files

- `param_template.json` — Parameter input JSON
