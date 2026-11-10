# RTH-GAN

Core implementation of visible-to-infrared image generation with PINN Retinex
priors, RG-HSG (IG-SAGN and RPA), TAFF, and RTD/HCW consistency. The repository
contains source code, configuration templates, and synthetic tests.

## Install

Use Python 3.10 or newer and a matching PyTorch/torchvision installation.
From the repository root:

```bash
python -m pip install -r requirements.txt
```

For quantitative Landsat raster loading and the synthetic end-to-end test:

```bash
python -m pip install rasterio
```

## Core files

- `models/rghsg.py`: full RGB-to-single-channel generator.
- `models/modules/manuscript_retinex.py`: frozen PINN priors and safe reflectance.
- `models/modules/manuscript_fusion.py`: local/radial adaptive fusion.
- `models/thermal_rtd.py` and `models/intensity_rtd.py`: quantitative thermal
  consistency and the separate dimensionless JPEG intensity proxy.
- `train_rthgan.py` / `infer_rthgan.py`: staged training, strict checkpoints,
  inference, and evaluation.
- `data/`: data loading and preparation.
- `configs/`: editable configuration templates; `tests/`: synthetic validation.

The small shared generator and geometry modules are runtime dependencies of the
model and its controlled ablations.

## Provide your data

Set paths in a copy of the relevant dataset template.
The quantitative mode requires scene-exclusive train/val/test rows, science/QA
raster paths, radiative context, and train-only normalization statistics; see
`data/landsat.py::REQUIRED_COLUMNS` and `configs/rtd/thermal_template.yaml`.

The JPEG mode uses the manifest contract in `data/jpeg_pairs.py`. A prepared
Landsat manifest and its preparation/normalization provenance can be generated
with `scripts/prepare_landsat_jpeg.py` and
`configs/dataset/landsat_jpeg_preparation.yaml`.

JPEG intensity is not calibrated temperature. Its fixed auxiliary proxy must
not be interpreted as measured thermal radiative transfer.

## Train and infer

After pointing `configs/dataset/landsat_jpeg.yaml` to your prepared manifest,
inspect the full-model configuration without starting training:

```bash
python scripts/run_manuscript_ablation.py --dataset-config configs/dataset/landsat_jpeg.yaml --variant full --seed 42 --output-dir runs/rthgan_seed42 --dry-run
```

Run the same command without `--dry-run` to train. The full preset uses separate
PINN pretraining, reconstruction/auxiliary warm-up, then adversarial training;
the PINN remains frozen during generator training.

```bash
python infer_rthgan.py --checkpoint runs/rthgan_seed42/checkpoints/best.pth --manifest /path/to/manifest.csv --output-dir results/rthgan_seed42 --device cuda:0
```

Quantitative training can use `train_rthgan.py` with explicit `--dataset-config`,
`--model-config`, `--train-config`, `--thermal-config`, and `--output-dir`.
Fill physical normalization and sensor-response fields using actual data.

## Validate

```bash
python -m pytest -q
python scripts/smoke_rthgan_manuscript.py --output-dir results/synthetic_smoke
```

The smoke command generates synthetic rasters and validates reduced-model
training, checkpoint resume, and inference. It requires rasterio.

## Profile the generator

The profiling defaults select RTH-GAN. Use explicit random-weight diagnostics,
or supply the same trained checkpoint used for inference:

```bash
python scripts/compute_model_complexity.py --input-size 512 --device cpu --output-root results/profile
python scripts/benchmark_inference.py --untrained --input-size 512 --device cuda:0 --warmup 20 --iterations 100 --output-root results/profile
```

For trained-model profiling add `--checkpoint /path/to/model.pth` to both
commands, replacing `--untrained` in the timing command. Estimated FLOPs cover
major contractions, including PINN and RPA; they are not complete operator FLOPs.
Latency covers the full generator forward pass. FPS is `1000/mean_latency_ms`;
peak allocated GPU memory is reported in MiB. Data loading, export and training
losses are excluded.

The publication allowlist in `.gitignore` prevents local research files from
being added accidentally. Add a deliberate allowlist entry when introducing a
new core source file.
