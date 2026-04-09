<p align="center">

  <h1 align="center">3DPR: Single Image 3D Portrait Relighting with Generative Priors
    <a href='https://dl.acm.org/doi/10.1145/3757377.3763962'>
    <img src='https://img.shields.io/badge/Offical-(64 MB)-red' alt='PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/3dpr/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://arxiv.org/abs/2510.15846'>
    <img src='https://img.shields.io/badge/Arxiv-PDF-red' alt='arxiv PDF'>
    </a>
  </h1>
  <h2 align="center">ACM SIGGRAPH-Asia 2025 Conference Proceedings</h2>
  <div align="center">
  </div>
</p>
<p float="center">
  <img src="assets/teaser.jpg" width="98%" />
</p>

This repository contains the official implementation for **3DPR: Single Image 3D Portrait Relighting with Generative Priors**.

## Repository Status

This release folder is being prepared for the public code release. The current tree already contains:

- inversion and relighting code,
- the public paper overview,
- portable relighting inference and training wrappers,
- placeholders for pretrained checkpoints.

## Installation

The project expects a Python environment with PyTorch, torchvision, OpenCV, NumPy, `tqdm`, `natsort`, and the rendering dependencies used by EG3D-style codepaths.

A typical setup is:

```bash
conda create -n 3dpr python=3.10
conda activate 3dpr
pip install torch torchvision
pip install numpy opencv-python tqdm natsort click imageio imageio-ffmpeg scipy
```

Additional packages may still be required depending on the exact training or rendering path you use. This release folder should be treated as the code base; the final public environment file can be added later once the dependency set is frozen.

## Pretrained Models

Create the folder `pretrained_models/` and place the released inference checkpoints there. The expected base filenames are:

- `ffhqrebalanced512-128.pkl`
- `encoder_FFHQ.pt`
- `afa_FFHQ.pt`

Details are listed in [`pretrained_models/README.md`](/CT/VORF_GAN4/work/code/3dpr/pretrained_models/README.md).

Pretrained model download link:

- Google Drive: [https://drive.google.com/drive/folders/1KdS9KlDL70JpSAZilIoyyGZU5-UXwA-l?usp=sharing](Pretrained-models)

Reflectance checkpoints for relighting inference are provided separately as training outputs and are passed explicitly to the inference script.

## Data Layout

The code expects an input identity folder passed via `--data`. In practice this should point to one subject directory containing the input images and camera labels in the format used by the training and evaluation loaders in [`training/dataset_face.py`](/CT/VORF_GAN4/work/code/3dpr/training/dataset_face.py).

Relighting inference also requires:

- an environment-map directory containing `.exr` files,
- an MPI light-direction file for MPI-style relighting,
- optional z-spiral inset maps for visualizing sampled OLAT directions.

Those paths are now exposed through CLI flags instead of being fully hardcoded.

The training loaders still use dataset-root declarations inside the code. These are now grouped at the beginning of:

- [`training/dataset_face.py`](/CT/VORF_GAN4/work/code/3dpr/training/dataset_face.py)
- [`training/dataset_relight.py`](/CT/VORF_GAN4/work/code/3dpr/training/dataset_relight.py)

Update those top-of-file dataset-root declarations to match your local storage layout before training or evaluation.

## Dataset

Below is the expected MPI dataset layout at a high level. This is the structure to mirror when preparing data for inference and training.

```text
MPI_DATASET/
|-- preprocess_root/
|   `-- 10001/
|       |-- ID20000/
|       |   |-- camera/
|       |   |   `-- dataset_dict.json
|       |   |-- crop/
|       |   |   `-- Cam07_ID20000.png
|       |   |-- mask_rmbg2/
|       |   |   `-- Cam07_ID20000.png
|       |   |-- mask_seg/
|       |   |   `-- Cam07_ID20000_EMAP-860.png
|       |   |-- transform/
|       |   |   `-- Cam07_ID20000.txt
|       |   `-- ...
|       `-- IDxxxxx/
|
|-- relighting/
|   |-- indoor-0/
|   |   `-- ID20000/
|   |       `-- images/
|   |           |-- Cam06_ID20000_EMAP-860.png
|   |           `-- Cam06_ID20000_EMAP-861.png
|   |-- indoor-1/
|   |-- indoor-2/
|   |-- outdoor-0/
|   |-- outdoor-1/
|   `-- outdoor-2/
|
|-- FOLAT_c2_align/
|   `-- Cam06/
|       `-- ID20000/
|           |-- 000.exr
|           |-- 001.exr
|           `-- ...
|
`-- vorf_gan_config/
    `-- LSX_light_positions_mpi.txt
```

Notes:

- Inference uses a single subject folder such as `data/scans/ID00600/`.
- Training reads relit images and auxiliary data for training subjects such as `ID20000`.
- FaceOLAT samples are read from the camera-first layout under `FOLAT_c2_align/`.
- If your storage layout differs, update the `DATASET_PATHS` blocks in the dataset loader files above.

## Inference

Two top-level inference entrypoints are included:

- [`infer.py`](/CT/VORF_GAN4/work/code/3dpr/infer.py): inversion / reconstruction and optional multiview or video rendering.
- [`infer_relighting.py`](/CT/VORF_GAN4/work/code/3dpr/infer_relighting.py): relighting inference with environment maps and reflectance checkpoints.

The recommended release wrapper is:

```bash
bash scripts/infer_goae.sh
```

By default it uses:

```bash
DATA_DIR=data/scans/ID00600
ENVMAP_DIR=data/envmaps
OUTDIR=outputs/inference/ID00600
R_CKPT=pretrained_models/network-snapshot-000225.pkl
```

Optional environment variables for the wrapper:

- `DATA_DIR`: input subject folder.
- `ENVMAP_DIR`: directory containing target `.exr` environment maps.
- `OUTDIR`: output directory.
- `R_CKPT`: reflectance checkpoint `.pkl` file or checkpoint directory.
- `PYTHON_BIN`: Python executable to use.
- `G_CKPT`, `E_CKPT`, `AFA_CKPT`: override the default pretrained model paths.
- `LIGHT_DIRS_PATH`: path to the MPI light-direction file.
- `ENV_ZSPIRAL_PATH`: path to inset environment-map masks.
- `CUDA_VISIBLE_DEVICES`: GPU index.

Example with overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTDIR=outputs/inference/custom_run \
bash scripts/infer_goae.sh
```

You can also call the Python entrypoint directly:

```bash
python infer_relighting.py \
  --data /path/to/subject_ID00001 \
  --G_ckpt pretrained_models/ffhqrebalanced512-128.pkl \
  --E_ckpt pretrained_models/encoder_FFHQ.pt \
  --AFA_ckpt pretrained_models/afa_FFHQ.pt \
  --R_ckpt /path/to/checkpoints \
  --envmap_dir /path/to/envmaps \
  --light_dirs_path /path/to/LSX_light_positions_mpi.txt \
  --dataset_name mpi \
  --lightstage_res full \
  --use_viewdirs True \
  --use_sr_encoder True \
  --eval_mode True \
  --multi_view True \
  --input_view True \
  --add_envmap True \
  --use_mask True \
  --gen_mask_mode 2 \
  --outdir outputs/inference
```

## Relighting

Relighting is handled by [`infer_relighting.py`](/CT/VORF_GAN4/work/code/3dpr/infer_relighting.py). The main inputs are:

- one subject folder,
- the released generator / encoder / AFA checkpoints,
- a trained reflectance checkpoint,
- target environment maps.

The script supports multiview rendering, video export, input-view relighting, and mask-aware compositing. The release wrapper currently defaults to MPI-style relighting with:

- `o_dim=32`,
- `arch=resnet`,
- `lightstage_res=full`,
- `gen_mask_mode=2`,
- `use_sr_encoder=True`.

## Training

Training of the relighting / reflectance model is exposed through [`train_lightstage_encoder_relight.py`](/CT/VORF_GAN4/work/code/3dpr/train_lightstage_encoder_relight.py).

The recommended bash wrapper is:

```bash
bash scripts/train_relight_encoder.sh
```

The wrapper defaults to the current release checkpoints and a baseline MPI training configuration:

```bash
OUTDIR=training-MPI-LS/04_runs
DATASET_NAME=mpi
NUM_GPUS=3
BATCH=9
NUM_IDS=450
NUM_VIEWS=1
LIGHTSTAGE_RES=half
LR=0.00015
WT_LPIPS=0.3
O_DIM=32
ARCH=resnet
USE_SR_ENCODER=True
```

Optional environment variables for the training wrapper:

- `OUTDIR`: training output root.
- `R_CKPT`: optional checkpoint path or checkpoint directory for initialization / resume.
- `NUM_GPUS`, `BATCH`, `NUM_IDS`, `NUM_VIEWS`: training scale settings.
- `LIGHTSTAGE_RES`, `LR`, `WT_COL`, `WT_LPIPS`, `LPIPS_WT_MODE`: relighting training hyperparameters.
- `START_STAGE2_AFTER`, `UPDATE_SR_AFTER`, `PORT`, `PRINT_INTERVAL`: runtime and schedule settings.
- `G_CKPT`, `E_CKPT`, `AFA_CKPT`, `PYTHON_BIN`: checkpoint and interpreter overrides.

Example with overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
OUTDIR=training-MPI-LS/05_ablns \
NUM_GPUS=3 \
NUM_VIEWS=1 \
LIGHTSTAGE_RES=half \
WT_LPIPS=0.3 \
bash scripts/train_relight_encoder.sh
```

A minimal example looks like:

```bash
python train_lightstage_encoder_relight.py \
  --G_ckpt pretrained_models/ffhqrebalanced512-128.pkl \
  --E_ckpt pretrained_models/encoder_FFHQ.pt \
  --AFA_ckpt pretrained_models/afa_FFHQ.pt \
  --outdir training-MPI-LS/04_runs \
  --dataset_name mpi \
  --relight True \
  --eval_mode False \
  --reload True \
  --batch 9 \
  --num_gpus 3 \
  --num_ids 450 \
  --num_views 1 \
  --lightstage_res half \
  --lr 0.00015 \
  --wt_col 1 \
  --wt_lpips 0.3 \
  --lpips_wt_mode -1 \
  --decoder_depth 1 \
  --o_dim 32 \
  --use_viewdirs True \
  --arch resnet \
  --act_fn sigmoid \
  --tonemap True \
  --use_sr_encoder True \
  --port 6000
```

Training expects the MPI light-stage data layout used by the dataset loaders already present in this repository. The exact public dataset packaging may still be documented separately once finalized.

## Dataset Repository

- FaceOLAT Dataset: https://github.com/prraoo/FaceOLAT

## Citation

If you find this work useful, please cite:

```bibtex
@article{prao20253dpr,
	title = {3DPR: Single Image 3D Portrait Relighting with Generative Priors},
	author = {Rao, Pramod and Meka, Abhimitra and Zhou, Xilong   and Fox, Gereon and B R, Mallikarjun and Zhan, Fangneng and Weyrich, Tim and Bickel, Bernd and Pfister, Hanspeter and Matusik, Wojciech and Beeler, Thabo and Elgharib, Mohamed and Habermann, Marc and Theobalt, Christian },
	booktitle = {ACM SIGGRAPH ASIA 2025 Conference Proceedings},
	year={2025}
}
```
