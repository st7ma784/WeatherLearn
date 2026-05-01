"""
Baseline training and evaluation on ~/rst/extracted_data cnvmap files.

Two-phase workflow
------------------
  1. Preprocess: cnvmap files → batched .npy files (run once)
  2. Train: DatasetFromPresaved loads via mmap — fast, memory-efficient

Usage
-----
  # Step 1 — preprocess (run once per grid_size):
  python run_baseline.py --preprocess --grid_size 120 --max_files 500

  # Step 2 — train:
  python run_baseline.py --grid_size 120 --max_files 500 --epochs 20 [--wandb]

  # Combined (preprocess if needed, then train):
  python run_baseline.py --grid_size 120 --max_files 500 --epochs 20 --wandb

Grid channels (6)
-----------------
  0  obs_vel_north  — SH-fitted northward E×B drift in radar-covered cells (m/s); zero elsewhere.
  1  obs_vel_east   — SH-fitted eastward E×B drift in radar-covered cells (m/s); zero elsewhere.
  2  model_vel_north — Weimer/TS96 background-model northward drift, m/s
  3  model_vel_east  — background-model eastward drift, m/s
  4  soft_occ        — radar coverage confidence tanh(weight/median), [0, 1]

Solar-wind / CPCP conditioning (SW_FIELDS, 6 scalars)
------------------------------------------------------
  IMF.Bx, IMF.By, IMF.Bz  — interplanetary magnetic field components (nT)
  IMF.Kp                   — planetary geomagnetic activity index
  IMF.Vx                   — solar-wind bulk velocity (km/s)
  pot.drop                 — cross-polar cap potential drop, CPCP (kV)
  chi.sqr                  — chi-squared of the spherical harmonic fit (goodness of fit)

These are injected via FiLM layers (Feature-wise Linear Modulation) at each
encoder/decoder stage of the Pangu model, allowing the network to condition
the convection forecast on the global geomagnetic driving state.
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

# Solar-wind / geomagnetic conditioning fields stored in every cnvmap record.
# pot.drop is the cross-polar cap potential (CPCP in kV) — the single most
# informative scalar for characterising how strongly the convection is driven.
SW_FIELDS = ["IMF.Bx", "IMF.By", "IMF.Bz", "IMF.Kp", "IMF.Vx", "pot.drop", "chi.sqr"]
SW_DIM    = len(SW_FIELDS)  # 7

# Legacy alias so existing code that imports IMF_FIELDS / IMF_DIM still works.
IMF_FIELDS = SW_FIELDS
IMF_DIM    = SW_DIM


# ── grid conversion ───────────────────────────────────────────────────────────

def record_to_grid(record, grid_size, min_mlat=50.0, max_mlat=90.0, splat_sigma=3.0):
    """Convert one cnvmap record → (6, G, G) float32 array.

    Channels
    --------
    0  obs_vel_north  — SH-fitted northward E×B drift in radar-covered cells (m/s),
                        zero in unobserved cells.
    1  obs_vel_east   — SH-fitted eastward E×B drift in radar-covered cells (m/s),
                        zero in unobserved cells.
    2  model_vel_north — Weimer/TS96 background-model northward drift, m/s
    3  model_vel_east  — background-model eastward drift, m/s
    4  soft_occ        — tanh(obs_splat_weight / median_weight), [0, 1]
    5  boundary_dist   — signed mlat distance from Heppner-Maynard boundary (°)

    NOTE: vector.kvect in cnvmap files is the radar BEAM AZIMUTH (toward the radar),
    not the plasma flow direction.  Using v_LOS * cos(beam_azimuth) gives the LOS
    velocity projected onto geographic axes, not the true 2D E×B velocity.  Obs
    channels are therefore derived from the SH-fitted model velocity (model.*),
    which uses model.kvect as the true flow azimuth, masked to observed cells.
    """
    def _splat_vectors(mlat_arr, mlon_arr, vn_arr, ve_arr):
        """Project and Gaussian-splat (vn, ve) onto the polar grid."""
        n = len(mlat_arr)
        if n == 0:
            z = np.zeros((grid_size, grid_size), dtype=np.float32)
            return z, z.copy(), np.zeros((grid_size, grid_size), dtype=np.float64)

        in_band = (mlat_arr >= min_mlat) & (mlat_arr <= max_mlat) & np.isfinite(mlat_arr)
        if not in_band.any():
            z = np.zeros((grid_size, grid_size), dtype=np.float32)
            return z, z.copy(), np.zeros((grid_size, grid_size), dtype=np.float64)

        mlats_b = mlat_arr[in_band]
        mlons_b = mlon_arr[in_band]
        vns_b   = vn_arr[in_band]
        ves_b   = ve_arr[in_band]

        half  = (grid_size - 1) / 2.0
        r     = (90.0 - mlats_b) / (90.0 - min_mlat)
        theta = np.deg2rad(mlons_b % 360.0)
        xi = np.clip(half + half * r * np.sin(theta), 0, grid_size - 1)
        yi = np.clip(half - half * r * np.cos(theta), 0, grid_size - 1)

        vn_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
        ve_sum = np.zeros((grid_size, grid_size), dtype=np.float64)
        cnt    = np.zeros((grid_size, grid_size), dtype=np.float64)

        radius    = max(1, int(np.ceil(3.0 * splat_sigma)))
        gx        = np.arange(-radius, radius + 1, dtype=np.float32)
        k1d       = np.exp(-0.5 * (gx / splat_sigma) ** 2)
        kernel_2d = np.outer(k1d, k1d)
        xi_int    = np.round(xi).astype(np.int32)
        yi_int    = np.round(yi).astype(np.int32)

        for k in range(len(mlats_b)):
            cx, cy = xi_int[k], yi_int[k]
            x0 = cx - radius;  x1 = cx + radius + 1
            y0 = cy - radius;  y1 = cy + radius + 1
            gx0 = max(0, -x0);  gx1 = gx0 + min(x1, grid_size) - max(x0, 0)
            gy0 = max(0, -y0);  gy1 = gy0 + min(y1, grid_size) - max(y0, 0)
            ax0 = max(x0, 0);   ax1 = min(x1, grid_size)
            ay0 = max(y0, 0);   ay1 = min(y1, grid_size)
            if ax0 >= ax1 or ay0 >= ay1:
                continue
            w = kernel_2d[gy0:gy1, gx0:gx1]
            vn_sum[ay0:ay1, ax0:ax1] += w * vns_b[k]
            ve_sum[ay0:ay1, ax0:ax1] += w * ves_b[k]
            cnt[ay0:ay1, ax0:ax1]    += w

        denom = np.maximum(cnt, 1e-9)
        return (vn_sum / denom).astype(np.float32), (ve_sum / denom).astype(np.float32), cnt

    # ── observed radar locations (for coverage only, not velocity) ────────────
    # vector.kvect is the radar BEAM AZIMUTH (toward the radar), not the plasma
    # flow direction.  v_LOS * cos(beam_azimuth) gives the LOS projected onto
    # geographic N/E — not the true 2D E×B velocity.  Use obs locations only to
    # derive soft_occ; obs velocity comes from the SH-fitted model.* fields.
    obs_mlats = np.asarray(record.get("vector.mlat", []), dtype=np.float32)
    obs_mlons = np.asarray(record.get("vector.mlon", []), dtype=np.float32)
    n_obs = min(len(obs_mlats), len(obs_mlons))
    ones  = np.ones(n_obs, dtype=np.float32)
    _, _, obs_cnt = _splat_vectors(obs_mlats[:n_obs], obs_mlons[:n_obs], ones, ones)

    # ── background statistical model vectors ─────────────────────────────────
    mod_mlats  = np.asarray(record.get("model.mlat", []),        dtype=np.float32)
    mod_mlons  = np.asarray(record.get("model.mlon", []),        dtype=np.float32)
    mod_vels   = np.asarray(record.get("model.vel.median", []),  dtype=np.float32)
    mod_kvects = np.asarray(record.get("model.kvect", []),       dtype=np.float32)
    n_mod = min(len(mod_mlats), len(mod_mlons), len(mod_vels))
    if n_mod > 0:
        kv_rad  = np.deg2rad(mod_kvects[:n_mod] if len(mod_kvects) >= n_mod
                             else np.zeros(n_mod, dtype=np.float32))
        mod_vn  = mod_vels[:n_mod] * np.cos(kv_rad)
        mod_ve  = mod_vels[:n_mod] * np.sin(kv_rad)
    else:
        mod_vn = mod_ve = np.zeros(0, dtype=np.float32)

    mod_vn_grid, mod_ve_grid, _ = _splat_vectors(
        mod_mlats[:n_mod], mod_mlons[:n_mod], mod_vn, mod_ve)

    # ── soft occupancy from observed locations ────────────────────────────────
    median_occ = float(np.median(obs_cnt[obs_cnt > 1e-9])) if (obs_cnt > 1e-9).any() else 1.0
    soft_occ   = np.tanh(obs_cnt / max(median_occ, 1e-9)).astype(np.float32)

    # ── obs velocity channels: model velocity masked to radar coverage ─────────
    # Uses the SH-fitted model.* velocity (physically correct) in covered cells;
    # zero in unobserved cells so the network can distinguish coverage regions.
    occ_mask    = (soft_occ > 0.05).astype(np.float32)
    obs_vn_grid = mod_vn_grid * occ_mask
    obs_ve_grid = mod_ve_grid * occ_mask

    # ── Heppner-Maynard boundary distance ────────────────────────────────────
    _bnd_mlat = record.get("boundary.mlat", [])
    _bnd_mlon = record.get("boundary.mlon", [])
    bnd_mlats = np.asarray(_bnd_mlat if _bnd_mlat is not None else [], dtype=np.float32)
    bnd_mlons = np.asarray(_bnd_mlon if _bnd_mlon is not None else [], dtype=np.float32)
    half_g = (grid_size - 1) / 2.0
    xi_g, yi_g  = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    dx_g = (xi_g - half_g) / half_g
    dy_g = (half_g - yi_g) / half_g
    pixel_mlat = (90.0 - np.sqrt(dx_g**2 + dy_g**2) * (90.0 - min_mlat)).astype(np.float32)
    pixel_mlon = (np.rad2deg(np.arctan2(dx_g, dy_g)) % 360.0).astype(np.float32)

    if len(bnd_mlats) > 0 and len(bnd_mlats) == len(bnd_mlons):
        bnd_mlon_n = bnd_mlons % 360.0
        sort_idx   = np.argsort(bnd_mlon_n)
        bnd_mlon_s = bnd_mlon_n[sort_idx]
        bnd_mlat_s = bnd_mlats[sort_idx]
        mlon_w = np.concatenate([bnd_mlon_s - 360, bnd_mlon_s, bnd_mlon_s + 360])
        mlat_w = np.concatenate([bnd_mlat_s,        bnd_mlat_s, bnd_mlat_s])
        bnd_interp    = np.interp(pixel_mlon.ravel(), mlon_w, mlat_w).reshape(grid_size, grid_size)
        boundary_dist = (pixel_mlat - bnd_interp).astype(np.float32)
    else:
        boundary_dist = np.zeros((grid_size, grid_size), dtype=np.float32)

    return np.stack([obs_vn_grid, obs_ve_grid, mod_vn_grid, mod_ve_grid,
                     soft_occ, boundary_dist], axis=0)


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_to_disk(cnvmap_dir, out_dir, grid_size, max_files=None, chunk_size=200,
                       min_mlat=50.0, max_mlat=90.0):
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
        return np.array([float(record.get(f, 0.0)) for f in SW_FIELDS],
                        dtype=np.float32)

    for fname in tqdm(files, desc="cnvmap→npy"):
        fpath = os.path.join(cnvmap_dir, fname)
        try:
            records, _ = pydarnio.read_map(fpath, mode="lax")
        except Exception:
            continue
        if len(records) < 2:
            continue

        grids = [record_to_grid(r, grid_size, min_mlat, max_mlat) for r in records]
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

    # 6 channels: obs_vel_north, obs_vel_east, model_vel_north, model_vel_east, soft_occ, boundary_dist
    shape = [-1, 6, grid_size, grid_size]
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
        return np.array([float(record.get(f, 0.0)) for f in SW_FIELDS],
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

        # Pre-compute IMF normalisation (mean/std) from a representative sample
        if self._imf_avail:
            sample = []
            for f in imf_files[:100]:
                try:
                    sample.append(np.load(f, mmap_mode='r')[:50])
                except Exception:
                    pass
            if sample:
                arr = np.concatenate(sample, axis=0).astype(np.float64)
                self._imf_mean = arr.mean(axis=0).astype(np.float32)
                # Floor at 1.0 — any IMF field with near-zero variance is likely
                # missing/constant; normalising by 1e-6 would explode rare non-zero
                # readings to ~1e6, overflowing float16 in the FiLM MLP.
                self._imf_std  = np.maximum(arr.std(axis=0), 1.0).astype(np.float32)
            else:
                self._imf_mean = np.zeros(IMF_DIM, dtype=np.float32)
                self._imf_std  = np.ones(IMF_DIM,  dtype=np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base now returns (x_agg, x_last, y) — x_last is the single most-recent
        # raw frame (not temporally averaged), used as residual base in the model.
        x, x_last, y = self.base[idx]

        if not self._imf_avail:
            return (x,
                    x_last,
                    torch.zeros(IMF_DIM, dtype=torch.float32),
                    torch.zeros(1,       dtype=torch.float32),
                    y)

        # Locate which chunk and offset (mirrors DatasetFromPresaved logic)
        idx        = int(idx % len(self.base))
        file_index = int(np.searchsorted(self.base.cumulative_sizes, idx, side='right'))
        prev_total = int(self.base.cumulative_sizes[file_index - 1]) if file_index > 0 else 0
        offset     = idx - prev_total

        imf_norm = np.zeros(IMF_DIM, dtype=np.float32)
        mask     = 0.0
        if file_index < len(self.imf_files):
            arr = np.load(self.imf_files[file_index], mmap_mode='r')
            if offset < len(arr):
                raw      = arr[offset]
                imf_norm = np.clip((raw - self._imf_mean) / self._imf_std, -5.0, 5.0)
                mask     = float(np.any(raw != 0.0))

        return (x,
                x_last,
                torch.tensor(imf_norm, dtype=torch.float32),
                torch.tensor([mask],   dtype=torch.float32),
                y)


# ── data module ───────────────────────────────────────────────────────────────

class PresavedDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, use_solar=True,
                 num_input_frames=1, temporal_agg_frames=1):
        super().__init__()
        self.data_dir            = data_dir
        self.batch_size          = batch_size
        self.use_solar           = use_solar
        self.num_input_frames    = num_input_frames
        self.temporal_agg_frames = temporal_agg_frames

    def setup(self, stage=None):
        dataA, dataB, shape = load_dataset_from_disk(self.data_dir)
        base = DatasetFromPresaved(dataA, dataB, shape,
                                   num_input_frames=self.num_input_frames,
                                   temporal_agg_frames=self.temporal_agg_frames)

        # Normalise on a random sample of train split (fast — avoids reading all 26k items)
        n_val   = max(1, int(len(base) * 0.1))
        n_train = len(base) - n_val
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_train, size=min(2000, n_train), replace=False)
        stats = base.compute_stats_from_indices(sample_idx.astype(np.int64))
        # x_mean/x_std: stats over temporally-aggregated input frames (unit-normalise x_in).
        # y_mean/y_std: stats over single target frames (unit-normalise y and x_last).
        # x_last and y share y_stats so delta = y - x_last has no per-channel bias in
        # empty cells.  Do NOT override y_stats with x_stats when temporal aggregation
        # is active — the two distributions have different variances.
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

        solar_dim = SW_DIM if (self.use_solar and imf_files) else 0
        print(f"  train: {n_train:,}  val: {n_val:,}  solar_dim: {solar_dim}")
        print(f"  x_in mean (agg):  {stats['x_mean'].round(4)}")
        print(f"  x_in std  (agg):  {stats['x_std'].round(4)}")
        print(f"  y/x_last mean:    {stats['y_mean'].round(4)}")
        print(f"  y/x_last std:     {stats['y_std'].round(4)}")
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
    p.add_argument("--grid_size",     type=int,   default=240)
    p.add_argument("--embed_dim",     type=int,   default=128)
    p.add_argument("--mlp_ratio",     type=int,   default=2)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_steps",  type=int,   default=1000)
    p.add_argument("--grad_clip",     type=float, default=5.0)
    p.add_argument("--max_files",     type=int,   default=0,
                   help="0 = use all 26k files")
    p.add_argument("--epochs",        type=int,   default=100)
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
    p.add_argument("--resume_from",   default=None,
                   help="Path to checkpoint to resume from (auto-detects latest if 'auto')")
    p.add_argument("--num_input_frames", type=int, default=1,
                   help="Number of aggregated input slots to stack as channels")
    p.add_argument("--temporal_agg_frames", type=int, default=1,
                   help="Raw frames averaged together per input slot — increases density")
    p.add_argument("--min_mlat", type=float, default=50.0,
                   help="Minimum magnetic latitude for polar grid (degrees)")
    p.add_argument("--max_mlat", type=float, default=90.0,
                   help="Maximum magnetic latitude for polar grid (degrees)")
    args = p.parse_args()
    if args.max_files == 0:
        args.max_files = None

    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"  SDPA backends — flash: {torch.backends.cuda.flash_sdp_enabled()}, "
              f"mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}, "
              f"math: {torch.backends.cuda.math_sdp_enabled()}")

    # ── derive cache path — include polar bounds so different projections don't collide ──
    tag      = f"g{args.grid_size}_f{args.max_files or 'all'}_mlat{int(args.min_mlat)}-{int(args.max_mlat)}"
    data_dir = os.path.join(args.cache_dir, tag)

    # ── preprocess if needed ──────────────────────────────────────────────────
    shape_file = os.path.join(data_dir, "shape.txt")
    if args.preprocess or not os.path.exists(shape_file):
        preprocess_to_disk(args.cnvmap_dir, data_dir, args.grid_size, args.max_files,
                           min_mlat=args.min_mlat, max_mlat=args.max_mlat)
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
                            use_solar=not args.no_solar,
                            num_input_frames=args.num_input_frames,
                            temporal_agg_frames=args.temporal_agg_frames)
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
        num_input_frames=args.num_input_frames,
        use_ema=True,
        solar_wind_dim=dm.solar_dim,
        log_diagnostics=True,
        diagnostics_interval=20,
        log_images_every_n_val_epochs=5,
    )
    model._warmup_steps = args.warmup_steps
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  parameters: {total_params:,}  "
          f"grid: {args.grid_size}×{args.grid_size}  embed_dim: {args.embed_dim}\n")

    if not args.fast_dev_run and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── logger ────────────────────────────────────────────────────────────────
    logtool = None
    if args.wandb and not args.fast_dev_run:
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

    # ── trainer ───────────────────────────────────────────────────────────────
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        EarlyStopping(monitor="val_skill_persistence", patience=10, mode="max", verbose=True),
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            filename="pangu-{epoch:02d}-{val_skill_persistence:.4f}",
            monitor="val_skill_persistence", mode="max", save_top_k=2,
        ),
    ]

    import torch as _torch
    # bf16 has fp32's exponent range — no GradScaler needed, no grad-norm inflation.
    # Fall back to fp16 if the GPU doesn't support bf16 (pre-Ampere).
    if _torch.cuda.is_available():
        precision = "bf16-mixed" if _torch.cuda.is_bf16_supported() else "16-mixed"
    else:
        precision = "32-true"

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        precision=precision,
        max_epochs=args.epochs,
        gradient_clip_val=args.grad_clip,
        callbacks=callbacks,
        logger=logtool,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    ckpt_path = args.resume_from
    if ckpt_path == "auto":
        ckpts = sorted(
            (os.path.getmtime(os.path.join(args.ckpt_dir, f)), f)
            for f in os.listdir(args.ckpt_dir)
            if f.endswith(".ckpt")
        )
        ckpt_path = os.path.join(args.ckpt_dir, ckpts[-1][1]) if ckpts else None
        if ckpt_path:
            print(f"  Resuming from: {ckpt_path}")

    trainer.fit(model, dm, ckpt_path=ckpt_path)

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
