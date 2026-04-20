"""
Baseline training and evaluation on ~/rst/extracted_data cnvmap files.

Two-phase workflow:
  1. Preprocess: cnvmap files → batched .npy files (run once)
  2. Train: DatasetFromPresaved loads via mmap — fast, memory-efficient

Usage:
  # Step 1 — preprocess (run once per grid_size):
  python run_baseline.py --preprocess --grid_size 120 --max_files 500

  # Step 2 — train:
  python run_baseline.py --grid_size 120 --max_files 500 --epochs 20 [--wandb]

  # Combined (preprocess if needed, then train):
  python run_baseline.py --grid_size 120 --max_files 500 --epochs 20 --wandb
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
import pydarnio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from DataModule import DatasetFromPresaved, load_dataset_from_disk

# IMF fields stored in every cnvmap record — used as solar-wind features.
IMF_FIELDS = ["IMF.Bx", "IMF.By", "IMF.Bz", "IMF.Kp", "IMF.Vx"]
IMF_DIM    = len(IMF_FIELDS)  # 5


# ── grid conversion ───────────────────────────────────────────────────────────

def record_to_grid(record, grid_size):
    """Convert one cnvmap record → (5, G, G) float32 array."""
    mlats = np.asarray(record.get("vector.mlat", []),        dtype=np.float32)
    mlons = np.asarray(record.get("vector.mlon", []),        dtype=np.float32)
    vels  = np.asarray(record.get("vector.vel.median", []),  dtype=np.float32)
    pwrs  = np.asarray(record.get("vector.vel.sd", []),      dtype=np.float32)
    wids  = np.asarray(record.get("vector.kvect", []),       dtype=np.float32)

    n = min(len(mlats), len(mlons), len(vels))
    if n == 0:
        return np.zeros((5, grid_size, grid_size), dtype=np.float32)

    vel_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    vel_cnt = np.zeros((grid_size, grid_size), dtype=np.float32)
    pwr_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    wid_sum = np.zeros((grid_size, grid_size), dtype=np.float32)

    lat_idx = np.clip(
        ((mlats[:n] + 90.0) / 180.0 * (grid_size - 1)).astype(int), 0, grid_size - 1)
    lon_idx = np.clip(
        ((mlons[:n] % 360.0) / 360.0 * (grid_size - 1)).astype(int), 0, grid_size - 1)

    np.add.at(vel_sum, (lat_idx, lon_idx), vels[:n])
    np.add.at(vel_cnt, (lat_idx, lon_idx), 1.0)
    if len(pwrs) >= n:
        np.add.at(pwr_sum, (lat_idx, lon_idx), pwrs[:n])
    if len(wids) >= n:
        np.add.at(wid_sum, (lat_idx, lon_idx), wids[:n])

    denom     = np.maximum(vel_cnt, 1.0)
    mean_vel  = vel_sum / denom
    mean_pwr  = pwr_sum / denom
    mean_wdt  = wid_sum / denom
    occupancy = (vel_cnt > 0).astype(np.float32)
    density   = np.log1p(vel_cnt).astype(np.float32)

    return np.stack([mean_vel, mean_pwr, mean_wdt, occupancy, density], axis=0)


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_to_disk(cnvmap_dir, out_dir, grid_size, max_files=None, chunk_size=200):
    """
    Read cnvmap files, compute consecutive-record pairs, save as
    dataA_NNN.npy / dataB_NNN.npy (same format as DatasetFromPresaved).
    Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(cnvmap_dir) if f.endswith(".cnvmap"))
    if max_files:
        files = files[:max_files]

    print(f"Preprocessing {len(files)} files → {out_dir}")
    t0 = time.time()

    bufA, bufB, bufIMF = [], [], []
    chunk_idx   = 0
    total_pairs = 0

    def flush(buf_a, buf_b, buf_imf, idx):
        np.save(os.path.join(out_dir, f"dataA_{idx:06d}.npy"),
                np.stack(buf_a).astype(np.float32))
        np.save(os.path.join(out_dir, f"dataB_{idx:06d}.npy"),
                np.stack(buf_b).astype(np.float32))
        np.save(os.path.join(out_dir, f"imf_{idx:06d}.npy"),
                np.stack(buf_imf).astype(np.float32))

    def extract_imf(record):
        return np.array([float(record.get(f, 0.0)) for f in IMF_FIELDS],
                        dtype=np.float32)

    for fname in tqdm(files, desc="cnvmap→npy"):
        fpath = os.path.join(cnvmap_dir, fname)
        try:
            records, _ = pydarnio.read_map(fpath, mode="lax")
        except Exception:
            continue
        if len(records) < 2:
            continue

        grids = [record_to_grid(r, grid_size) for r in records]
        imfs  = [extract_imf(r) for r in records]

        for i in range(len(grids) - 1):
            bufA.append(grids[i])
            bufB.append(grids[i + 1])
            bufIMF.append(imfs[i])      # IMF at the input timestep
            total_pairs += 1

            if len(bufA) >= chunk_size:
                flush(bufA, bufB, bufIMF, chunk_idx)
                chunk_idx += 1
                bufA, bufB, bufIMF = [], [], []

    if bufA:
        flush(bufA, bufB, bufIMF, chunk_idx)

    shape = [-1, 5, grid_size, grid_size]
    with open(os.path.join(out_dir, "shape.txt"), "w") as f:
        f.write(str(shape))

    elapsed = time.time() - t0
    print(f"  {total_pairs:,} pairs written in {elapsed:.1f}s  "
          f"({total_pairs/elapsed:.0f} pairs/s)")
    return out_dir


def extract_imf_to_disk(cnvmap_dir, out_dir, max_files=None, chunk_size=200):
    """
    Fast IMF-only extraction: reads cnvmap files, writes imf_NNN.npy that
    are chunk-aligned with existing dataA_NNN.npy / dataB_NNN.npy.
    Skips chunks that already have an imf file.
    """
    files = sorted(f for f in os.listdir(cnvmap_dir) if f.endswith(".cnvmap"))
    if max_files:
        files = files[:max_files]

    existing = {f for f in os.listdir(out_dir) if f.startswith("imf_") and f.endswith(".npy")}
    print(f"IMF extraction: {len(files)} files, {len(existing)} chunks already done")
    t0 = time.time()

    buf, chunk_idx, total = [], 0, 0

    def flush(b, idx):
        path = os.path.join(out_dir, f"imf_{idx:06d}.npy")
        if f"imf_{idx:06d}.npy" not in existing:
            np.save(path, np.stack(b).astype(np.float32))

    def extract_imf(record):
        return np.array([float(record.get(f, 0.0)) for f in IMF_FIELDS],
                        dtype=np.float32)

    for fname in tqdm(files, desc="cnvmap→imf"):
        fpath = os.path.join(cnvmap_dir, fname)
        try:
            records, _ = pydarnio.read_map(fpath, mode="lax")
        except Exception:
            continue
        if len(records) < 2:
            continue

        imfs = [extract_imf(r) for r in records]
        for i in range(len(imfs) - 1):
            buf.append(imfs[i])
            total += 1
            if len(buf) >= chunk_size:
                flush(buf, chunk_idx)
                chunk_idx += 1
                buf = []

    if buf:
        flush(buf, chunk_idx)

    elapsed = time.time() - t0
    print(f"  {total:,} IMF vectors in {elapsed:.1f}s  ({total/elapsed:.0f}/s)")


# ── solar-wind aware dataset ──────────────────────────────────────────────────

class SolarDataset(Dataset):
    """
    Wraps DatasetFromPresaved and adds mmap-loaded IMF vectors.
    Returns (x, solar_vec, solar_mask, y) if IMF files exist,
    else falls back to (x, y) for backward compatibility.
    """
    def __init__(self, base: DatasetFromPresaved, imf_files: list[str]):
        self.base      = base
        self.imf_files = imf_files  # parallel list to base.dataA
        self._imf_avail = bool(imf_files)

        # Pre-compute IMF normalisation (mean/std) from a small sample
        if self._imf_avail:
            sample = []
            for f in imf_files[:20]:
                try:
                    sample.append(np.load(f, mmap_mode='r')[:50])
                except Exception:
                    pass
            if sample:
                arr = np.concatenate(sample, axis=0).astype(np.float64)
                self._imf_mean = arr.mean(axis=0).astype(np.float32)
                self._imf_std  = np.maximum(arr.std(axis=0), 1e-6).astype(np.float32)
            else:
                self._imf_mean = np.zeros(IMF_DIM, dtype=np.float32)
                self._imf_std  = np.ones(IMF_DIM,  dtype=np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not self._imf_avail:
            return x, y

        # Locate which chunk and offset (mirrors DatasetFromPresaved logic)
        idx        = int(idx % len(self.base))
        file_index = int(np.searchsorted(self.base.cumulative_sizes, idx, side='right'))
        prev_total = int(self.base.cumulative_sizes[file_index - 1]) if file_index > 0 else 0
        offset     = idx - prev_total

        try:
            raw = np.load(self.imf_files[file_index], mmap_mode='r')[offset]
            imf_norm = (raw - self._imf_mean) / self._imf_std
            # mask=1 if any field is non-zero (real measurement available)
            mask = float(np.any(raw != 0.0))
        except Exception:
            imf_norm = np.zeros(IMF_DIM, dtype=np.float32)
            mask     = 0.0

        solar_vec  = torch.tensor(imf_norm,  dtype=torch.float32)
        solar_mask = torch.tensor([mask],    dtype=torch.float32)
        return x, solar_vec, solar_mask, y


# ── data module ───────────────────────────────────────────────────────────────

class PresavedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, use_solar=True):
        super().__init__()
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.use_solar  = use_solar

    def setup(self, stage=None):
        dataA, dataB, shape = load_dataset_from_disk(self.data_dir)
        base = DatasetFromPresaved(dataA, dataB, shape)

        # Normalise on a random sample of train split (fast — avoids reading all 26k items)
        n_val   = max(1, int(len(base) * 0.1))
        n_train = len(base) - n_val
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_train, size=min(2000, n_train), replace=False)
        stats = base.compute_stats_from_indices(sample_idx.astype(np.int64))
        base.set_normalization_stats(stats)

        # Load parallel IMF files if they exist
        imf_files = []
        if self.use_solar:
            imf_files = sorted(
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if f.startswith("imf_") and f.endswith(".npy")
            )

        full = SolarDataset(base, imf_files)

        self.train_ds, self.val_ds = random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))

        solar_dim = IMF_DIM if (self.use_solar and imf_files) else 0
        print(f"  train: {n_train:,}  val: {n_val:,}  solar_dim: {solar_dim}")
        print(f"  x_mean: {stats['x_mean'].round(3)}")
        print(f"  x_std:  {stats['x_std'].round(3)}")
        self.solar_dim = solar_dim

    def train_dataloader(self):
        nw = min(6, os.cpu_count() or 1)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
                          prefetch_factor=3 if nw > 0 else None)

    def val_dataloader(self):
        nw = min(4, os.cpu_count() or 1)
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=nw, pin_memory=True, persistent_workers=nw > 0)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cnvmap_dir",    default=os.path.expanduser("~/rst/extracted_data"))
    p.add_argument("--cache_dir",     default=os.path.expanduser("~/rst/preprocessed"))
    p.add_argument("--grid_size",     type=int,   default=120)
    p.add_argument("--embed_dim",     type=int,   default=64)
    p.add_argument("--mlp_ratio",     type=int,   default=2)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--max_files",     type=int,   default=500,
                   help="0 = use all 26k files")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--preprocess",    action="store_true",
                   help="Force re-preprocess even if cache exists")
    p.add_argument("--imf_only",      action="store_true",
                   help="Extract only IMF files (faster; reuses existing grid chunks)")
    p.add_argument("--fast_dev_run",  action="store_true")
    p.add_argument("--ckpt_dir",      default="./checkpoints_baseline")
    p.add_argument("--wandb",         action="store_true",
                   help="Log to Weights & Biases (requires WANDB_API_KEY)")
    p.add_argument("--wandb_project", default="SuperDARN-baseline")
    p.add_argument("--wandb_entity",  default="st7ma784")
    p.add_argument("--run_name",      default=None)
    p.add_argument("--no_solar",      action="store_true",
                   help="Disable solar-wind FiLM conditioning even if IMF files exist")
    args = p.parse_args()
    if args.max_files == 0:
        args.max_files = None

    torch.set_float32_matmul_precision("medium")

    # ── derive cache path (grid_size + file count make it unique) ─────────────
    tag      = f"g{args.grid_size}_f{args.max_files or 'all'}"
    data_dir = os.path.join(args.cache_dir, tag)

    # ── preprocess if needed ──────────────────────────────────────────────────
    shape_file = os.path.join(data_dir, "shape.txt")
    if args.preprocess or not os.path.exists(shape_file):
        preprocess_to_disk(args.cnvmap_dir, data_dir, args.grid_size, args.max_files)
    elif args.imf_only:
        extract_imf_to_disk(args.cnvmap_dir, data_dir, args.max_files)
    else:
        npy_files = [f for f in os.listdir(data_dir) if f.startswith("dataA_")]
        imf_files = [f for f in os.listdir(data_dir) if f.startswith("imf_")]
        if len(imf_files) < len(npy_files):
            print(f"IMF files missing ({len(imf_files)}/{len(npy_files)}); running extraction...")
            extract_imf_to_disk(args.cnvmap_dir, data_dir, args.max_files)
        else:
            print(f"Cache found: {len(npy_files)} chunks, {len(imf_files)} IMF in {data_dir}")

    # ── data module ───────────────────────────────────────────────────────────
    dm = PresavedDataModule(data_dir, batch_size=args.batch_size,
                            use_solar=not args.no_solar)
    dm.setup()

    # ── model ─────────────────────────────────────────────────────────────────
    from model import Pangu
    model = Pangu(
        grid_size=args.grid_size,
        embed_dim=args.embed_dim,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.lr,
        noise_factor=0.05,
        time_step=1,
        use_ema=True,
        solar_wind_dim=dm.solar_dim,
        log_diagnostics=True,
        diagnostics_interval=20,
        log_images_every_n_val_epochs=5,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  parameters: {total_params:,}  "
          f"grid: {args.grid_size}×{args.grid_size}  embed_dim: {args.embed_dim}\n")

    # ── logger ────────────────────────────────────────────────────────────────
    logtool = None
    if args.wandb and not args.fast_dev_run:
        try:
            from pytorch_lightning.loggers import WandbLogger
            run_name = args.run_name or f"baseline-{tag}"
            logtool = WandbLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                save_dir=args.ckpt_dir,
                log_model=False,
                config=vars(args),
            )
            print(f"  W&B: {args.wandb_entity}/{args.wandb_project}/{run_name}")
        except Exception as e:
            print(f"  W&B failed ({e}); using CSVLogger")

    # ── trainer ───────────────────────────────────────────────────────────────
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        EarlyStopping(monitor="val_mse", patience=6, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="pangu-{epoch:02d}-{val_mse:.4f}",
            monitor="val_mse", mode="min", save_top_k=2,
        ),
    ]

    import torch as _torch
    precision = "16-mixed" if _torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        precision=precision,
        max_epochs=args.epochs,
        gradient_clip_val=0.25,
        callbacks=callbacks,
        logger=logtool,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    trainer.fit(model, dm)

    if not args.fast_dev_run:
        results = trainer.validate(model, dm.val_dataloader())
        print("\n=== Final Metrics ===")
        for k, v in sorted(results[0].items()):
            print(f"  {k:45s}: {v:.4f}")
        if logtool is not None:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
