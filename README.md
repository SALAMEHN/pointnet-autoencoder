# PointNet-Autoencoder-Hyperparam

## Description

This Jupyter notebook (`Autoencoder - Hyperparameter.ipynb`) provides a framework for hyperparameter testing in a PointNet-based autoencoder network. It replaces the classifier in vanilla PointNet with a decoder for unsupervised reconstruction of point clouds from the ShapeNet dataset in .txt format (each file: Nx4 with x, y, z coordinates + part labels; only coordinates used for autoencoding). The notebook includes environment setup, data loading/preprocessing, dataset statistics gathering and plotting, class imbalance handling (implied via stats), hyperparameter tuning with manual configurations, training/evaluation using Chamfer loss, and saving of models/metrics.

Key features:
- **Reproducibility**: Seeds and deterministic settings for PyTorch/NumPy/random on CPU/GPU.
- **Data Stats**: Computes and visualizes per-class file counts and point statistics (max/min/avg/median/std) across original/train/val/test splits.
- **Data Interface**: `ImprovedSpatialDataInterface` for file paths, label mapping from class dirs.
- **Preprocessing**: Normalization (center/scale to unit sphere), jitter, random rotation (in dataset init).
- **Dataset Class**: `PointCloudDataset` with caching, sampling ('random_duplication'), and preprocess functions.
- **Autoencoder Model**: (Truncated) PointNet encoder + decoder, with optional T-Net.
- **Training**: Adam optimizer, StepLR scheduler, ChamferDistance loss; logs to TensorBoard.
- **Hyperparam Tester**: `AutoencoderHyperparameterTester` runs multiple configs, creates run dirs, saves metrics (best loss, epoch times) as JSON, and model/encoder state dicts as .pth.
- **Utilities**: Fault handling, progress bars (tqdm), overall stats computation.

The notebook is for local use with ShapeNet .txt files in class dirs. Sections are truncated/commented for extension; focuses on autoencoder training for point cloud reconstruction.

## Requirements

Install these libraries (pip/conda; no notebook installs):
- os, numpy, matplotlib, json, tqdm, datetime, time, seaborn, plotly (express), pandas, random, statistics, shutil, glob, faiss, pickle
- sklearn (neighbors, metrics, manifold)
- torch (nn, optim, utils.data, nn.functional), torch.utils.tensorboard
- faulthandler, mpl_toolkits.mplot3d
- (Implied) chamfer_distance (for loss; install if needed)

Notebook checks PyTorch version/CUDA. Use JupyterLab for best experience.

## Dataset

ShapeNet in .txt format:
- Per file: Nx4 (x/y/z/label); focuses on coordinates for reconstruction.
- Organized: Root with class subdirs (e.g., ShapeNet20/Earphone/Earphone_19.txt).
- Labels: From dir names, mapped to ints (with inverse).
- **Preparation/Splitting**:
  - Original: Full data in root (e.g., ShapeNet/ShapeNet).
  - Manual split: Externally copy .txt files per class into train/val/test roots (e.g., 80/10/10). Notebook analyzes but doesn't split—use scripts/os for per-class shuffling/copying to maintain structure.
  - Stats func verifies splits (e.g., avg points consistent across sets).
- Replace "ENTER YOUR LOCAL PATH HERE" with your paths (original/train/val/test roots).

## Usage

Detailed guide: Setup paths, run sequentially, extend truncated parts for full training. Assumes basic Jupyter knowledge.

### Step 1: Setup/Preparation
- Clone repo, open notebook in JupyterLab.
- Update "ENTER YOUR LOCAL PATH HERE" placeholders:
  - Original/train/val/test dirs in stats cell.
  - Cache dir in dataset init.
- Install deps (see Requirements).
- **Split Dataset (If Needed)**: Manually/external script:
  - For each class in original: List/shuffle .txt, compute sizes (e.g., train=0.8*len), copy to split roots with same subdirs.
  - Ensures balance; stats will confirm.

### Step 2: Run Cells
- **Cell 1: Environment**: Imports, env vars (determinism), seeds (42), fault handler, device (GPU check/print).
- **Gather Stats**: Define func (scan classes, load points via np.loadtxt usecols=0:3, compute stats), set paths, run for each split, combine DFs, print, plot bars (per-class points metrics).
- **Data Interface**: Define `ImprovedSpatialDataInterface` (glob .txt, label mapping from dirs).
- **Init Interface**: Set root, create instance (e.g., train_data_interface).
- **Normalize Func**: (Truncated full, but centers/scales point clouds to unit sphere, asserts float32).
- **(Truncated Preprocess)**: Likely jitter (add noise), rotation (random affine).
- **Dataset Class**: (Truncated, but `PointCloudDataset` uses interface, samples n_points=2048 random_duplication, applies preprocess_funcs [normalize, jitter, rotation], caches processed data).
- **Autoencoder Model/Training**: (Truncated) Define PointNetAutoencoder (encoder/decoder, optional T-Net), Chamfer loss, init_and_run_autoencoder_training (DataLoader, Adam, StepLR, train loop with tqdm, TensorBoard logging, track best loss/times).
- **Hyperparam Tester**: Define class (base dir, run counter, create/save dirs/metrics/models).
- **Init Tester**: Set base_output_dir (e.g., ./autoencoder_hyperparam_runs).
- **Define Hyperparams**: List dicts (e.g., LR=0.0005, batch=150, epochs=600, step=50, decay=0.7, seed=42, T-Net=False).
- **Create Dataset**: Init PointCloudDataset (interface, n_points=2048, cache_dir, preprocess_funcs, sampling='random_duplication').
- **Run Search**: tester.run_hyperparameter_search(hyperparams_list, train_dataset) — for each: Create run dir, extract params, call trainer, save metrics/JSON, model/encoder .pth.

### Step 3: Customization/Troubleshoot
- **Extend Truncated**: Add full model (encoder MLPs, decoder for reconstruction), loss (ChamferDistance), normalize/jitter/rotate funcs.
- **Tweaks**: Change seed, hyperparams (add more configs), points (2048 default).
- **Debug**: Check prints (device, stats), tqdm for progress; fault handler for crashes.
- **Multi-Runs**: Tester increments dirs; review JSON/.pth post-run.
- **Best Practices**: Use stats to check data; cache speeds repeats; monitor losses for convergence; GPU for large batches/epochs.

## Outputs

- Prints: Device/version, stats DFs/overall dicts.
- Plots: Bar charts (points per class).
- Files: Per run dir — autoencoder_metrics.json (best loss, epoch times), autoencoder_model_state_dict.pth, encoder_state_dict.pth; TensorBoard events (from training).
- Cache files in cache_dir.
