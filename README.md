# RTH-GAN

### Radiative Transfer Guided Hyperbolic GAN for Visible-to-Infrared Remote Sensing Image Translation

This repository provides the research implementation accompanying the manuscript **RTH-GAN: Radiative Transfer Guided Hyperbolic GAN for Visible-to-Infrared Remote Sensing Image Translation**, submitted to *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.

**Authors:** Yuanfeng Lian, Jintai Li, Guangjun Gao, Xuanhe Liu, and Jing Hua

---

## Abstract

> Visible-to-infrared (VI-to-IR) image translation provides complementary infrared images for remote sensing interpretation and downstream applications. Existing VI-to-IR methods often concentrate on visual appearance, resulting in visually similar infrared images but detail loss and artifacts, which limits the quality of cross-modal translation. To alleviate this issue, we propose Radiative Transfer Guided Hyperbolic Generative Adversarial Network (RTH-GAN) for visible-to-infrared remote sensing image translation. The proposed method aims to preserve structural details and improve the similarity between generated and reference infrared images. Specifically, a physics-informed neural network (PINN) based Retinex decomposition network first estimates illumination and reflectance priors from the visible image. These priors then guide a Retinex-Guided Hyperbolic Space Generator (RG-HSG) to model features in both Euclidean and hyperbolic spaces to preserve local details and represent hierarchical scene relations. In addition, a Radiative Transfer Guided Discriminator (RTD) combines conditional visual discrimination with a frozen radiative transfer extractor, thus providing radiative infrared information. A consistency loss is computed between the extractor outputs obtained from generated and reference infrared images to provide an additional constraint for generator training. Within bounded approximation errors of the extractor, we further derive an upper bound on the ideal radiance discrepancy in terms of the consistency loss and the approximation error terms. Experimental results on three remote sensing datasets demonstrate that RTH-GAN achieves better reconstruction quality, structural similarity, and visual quality than the included state-of-the-art baselines while reducing the discrepancy measured using the frozen radiative transfer extractor.

---

## Installation

The reference environment uses Python 3.11, PyTorch 2.4.0, torchvision 0.19.0, and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate pytorch-rthgan
```

The offline preprocessing tools require several optional packages:

```bash
python -m pip install deepxde opencv-python scikit-learn matplotlib
```

---

## Dataset

### Dataset structure

Organize one dataset at a time as follows:

```text
data/
|-- trainA/   # Visible training images
|-- trainB/   # Reference infrared training images
|-- trainC/   # Illumination priors
|-- trainD/   # Reflectance priors
|-- trainE/   # Radiance-condition maps
|-- testA/    # Visible test images
|-- testB/    # Reference infrared test images
|-- testC/    # Test illumination priors
`-- testD/    # Test reflectance priors
```

All modalities belonging to one sample must use the same base name:

```text
data/trainA/000001.png
data/trainB/000001.png
data/trainC/000001.png
data/trainD/000001.png
data/trainE/000001.png
```

The dataset loader stops when it detects a missing, duplicated, or extra base name.

---

## Prior preprocessing

### PINN-based Retinex decomposition

`pinn_extraction.py` performs per-image PINN optimization and produces:

- `<name>_L.png`: illumination-sensitive prior;
- `<name>_R.png`: reflectance-related structural prior.

Before running the script, replace its input glob and `SAVE_DIR` with your local paths:

```bash
python pinn_extraction.py
```

Copy illumination outputs to `trainC/` or `testC/` and reflectance outputs to `trainD/` or `testD/`. Rename the files to match the corresponding visible-image base names because the loader does not remove `_L` or `_R` automatically.

### Radiance-condition maps

`radiative_process.ipynb` applies K-means scene grouping and converts the scene classes into grayscale condition maps. Edit the notebook input/output paths, run its cells in order, and place the final maps in `data/trainE/` using the same base names as `trainA/`.

---

## Train and test

Run the following commands from the repository root. The current training script reads `./data/trainA` through `./data/trainE`.

### Training

Selected training arguments:

| Argument              | Type  |           Default | Description                                           |
| --------------------- | ----- | ----------------: | ----------------------------------------------------- |
| `--name`              | str   | `experiment_name` | Experiment and checkpoint directory name              |
| `--img_size`          | int   |             `128` | Square training resolution; paper experiments use 256 |
| `--batch_size`        | int   |               `1` | Training batch size                                   |
| `--lr`                | float |          `0.0002` | Generator Adam learning rate                          |
| `--n_epochs`          | int   |             `200` | Epochs before the configured decay stage              |
| `--n_epochs_decay`    | int   |             `200` | Additional decay-stage epochs                         |
| `--lambda_pair`       | float |            `15.0` | Paired L1 reconstruction weight                       |
| `--lambda_rad`        | float |            `0.01` | Radiance-prior supervision weight                     |
| `--val_ratio`         | float |             `0.1` | Fraction reserved for validation                      |
| `--val_seed`          | int   |              `42` | Deterministic validation split seed                   |
| `--best_metric`       | str   |       `composite` | Best-checkpoint criterion: `composite` or `ssim`      |
| `--tadsf_fusion_mode` | str   |            `full` | TA-DSF fusion or ablation mode                        |

The best generator is saved as `checkpoints/<name>/G_best.pth`.

### Inference and evaluation

```bash
python test.py \
  --weights ./checkpoints/rth_gan_256/G_best.pth \
  --input_dir ./data/testA \
  --target_dir ./data/testB \
  --illum_dir ./data/testC \
  --reflect_dir ./data/testD \
  --img_size 256 \
  --output_dir ./results/rth_gan_256
```

Selected inference arguments:

| Argument                | Default                                           | Description                                          |
| ----------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| `--weights`             | `./checkpoints/pinn_paired_exp01/G_epoch_200.pth` | Generator checkpoint                                 |
| `--input_dir`           | `./data/testA`                                    | Visible test images                                  |
| `--target_dir`          | `./data/testB`                                    | Reference infrared images                            |
| `--illum_dir`           | `./data/testC`                                    | Illumination priors                                  |
| `--reflect_dir`         | `./data/testD`                                    | Reflectance priors                                   |
| `--output_dir`          | `./results/pinn_paired_exp01_epoch_200`           | Output directory                                     |
| `--img_size`            | `128`                                             | Inference resolution                                 |
| `--tadsf_hyperagg_mode` | `auto`                                            | Infer spatial or channel aggregation from checkpoint |

---

## Repository structure

```text
RTH-GAN/
|-- checkpoints/             # Weights, logs, and validation records
|-- data/                    # Local datasets and precomputed priors
|-- images/                  # Optional training visualizations
|-- model/                   # Generator, discriminators, and network blocks
|-- options/                 # Command-line options
|-- results/                 # Generated images and evaluation reports
|-- tests/                   # Metric and TA-DSF tests
|-- utils/                   # Data, losses, metrics, and visualization
|-- environment.yml
|-- pinn_extraction.py
|-- radiative_process.ipynb
|-- train.py
|-- test.py
|-- LICENSE
`-- README.md
```

---

## Citation

If this work is useful in your research, please cite the manuscript:

```bibtex
@misc{lian2026rthgan,
  title  = {RTH-GAN: Radiative Transfer Guided Hyperbolic GAN for Visible-to-Infrared Remote Sensing Image Translation},
  author = {Lian, Yuanfeng and Li, Jintai and Gao, Guangjun and Liu, Xuanhe and Hua, Jing},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Transactions on Geoscience and Remote Sensing}
}
```

The citation will be updated after publication.

---

## Acknowledgments

Parts of the training infrastructure evolved from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We thank its authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE). Third-party datasets, pretrained models, and external code remain subject to their original licenses.
