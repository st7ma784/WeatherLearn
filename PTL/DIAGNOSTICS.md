# PTL Diagnostics Guide

This guide explains the Weights & Biases diagnostics produced by the PTL training run, what "healthy" trends look like, what failure patterns look like, and what to do when they appear.

## Scope

These diagnostics are emitted by the PTL model during training and validation.

Current controls:

- `--log_diagnostics` (default: `True`)
- `--diagnostics_interval` (default: `50`)
- `--log_images_every_n_val_epochs` (default: `1`)

## How To Read The Dashboard

Read diagnostics in this order:

1. Training stability (optimization health)
2. Validation skill (does it beat baselines?)
3. Spatial and residual diagnostics (where/why does it fail?)
4. Lead-time diagnostics (rollout drift behavior)

---

## 1) Training Stability Diagnostics

### `diag/grad_norm`

What it is:
- Global gradient norm captured after backward.

Healthy pattern:
- Early spike, then stabilizes within a bounded band.
- No sustained upward trend late in training.

Concerning pattern:
- Repeated extreme spikes or monotonic growth.
- Collapse to near-zero very early (possible dead network / poor LR).

Actions:
- Lower LR, increase gradient clipping, verify normalization stats, inspect per-block activity.

### `diag/step_time_s` and `diag/samples_per_sec`

What they are:
- Batch latency and throughput.

Healthy pattern:
- Fairly stable after warmup.

Concerning pattern:
- Increasing step time across epochs (I/O stalls, memory pressure, plotting too often).

Actions:
- Reduce diagnostics image frequency, tune workers/prefetch, inspect GPU memory pressure.

### `diag/train_pred_mean_c*`, `diag/train_pred_std_c*`, `diag/train_target_mean_c*`, `diag/train_target_std_c*`

What they are:
- Per-channel prediction vs target distribution snapshots.

Healthy pattern:
- Means/standard deviations track target distributions over time.

Concerning pattern:
- Prediction std consistently much smaller than target std (oversmoothing).
- Mean drifts away from target mean (bias).

Actions:
- Rebalance channel weights, tune loss components, inspect train-stat normalization cache.

---

## 2) Validation Skill Diagnostics

### `diag/val_skill_persistence`

What it is:
- Skill score vs persistence baseline, computed as:
  - `1 - model_error / persistence_error`

Healthy pattern:
- Positive and increasing (or stable positive) as training progresses.

Concerning pattern:
- Near zero or negative for long periods.

Interpretation:
- Model is not beating the naive persistence baseline.

Actions:
- Improve temporal modeling, scheduled sampling, horizon curriculum, revisit data split and feature quality.

### `diag/val_skill_climatology`

What it is:
- Skill score vs climatology-like zero baseline.

Healthy pattern:
- Positive and preferably significantly above zero.

Concerning pattern:
- Near zero indicates model not extracting useful signal.

Actions:
- Check data/label alignment, normalize train stats, verify target construction.

### `diag/val_event_mse` and `diag/val_quiet_mse`

What they are:
- MSE split by occupancy-like event mask (active vs quiet regions).

Healthy pattern:
- Event MSE > quiet MSE is expected, but the gap should reduce over training.

Concerning pattern:
- Event MSE plateaus very high while quiet improves strongly.

Interpretation:
- Model learns easy background but misses high-value dynamics.

Actions:
- Increase event weighting, add event-focused sampling, add storm-conditioned eval slices.

---

## 3) Spatial / Behavior Explanation Diagnostics

### `diag/val_rmse_map`

What it is:
- Per-channel spatial RMSE heatmaps aggregated over validation.

Healthy pattern:
- Errors localized to known sparse/noisy regions.

Concerning pattern:
- Large coherent high-error structures across broad areas.

Interpretation:
- Systematic geospatial bias or representational weakness.

Actions:
- Improve geospatial features, revisit patch/window resolution, inspect input coverage maps.

### `diag/val_example_panel`

What it is:
- Side-by-side sample panel (input / target / prediction per channel).

Healthy pattern:
- Predicted structures aligned with target, minimal phase lag, no excessive blur.

Concerning pattern:
- Spatial lag, ringing artifacts, diffuse/washed predictions.

Actions:
- Tune autoregressive rollout strategy, loss weighting, and model resolution hierarchy.

### `diag/val_residual_hist`

What it is:
- Residual distribution for a validation example.

Healthy pattern:
- Centered near zero, moderate spread, limited heavy tails.

Concerning pattern:
- Mean shifted from zero (bias), heavy tails (unstable extremes), multimodal errors.

Actions:
- Address bias with calibration/normalization checks; address tails with robust objectives or uncertainty modeling.

---

## 4) Lead-Time Diagnostics

### `diag/val_lead_time_table`

What it is:
- Proxy lead-time MSE across increasing rollout depth.

Important note:
- This is a proxy curve against a fixed target, useful for drift diagnostics but not a full multi-horizon benchmark.

Healthy pattern:
- MSE increases gradually with horizon.

Concerning pattern:
- Sharp early jump indicates unstable autoregressive propagation.

Actions:
- Use scheduled sampling, horizon curriculum, or reduce rollout noise during early training.

---

## Recommended Alert Thresholds (Starter)

Use these as initial monitoring rules, then calibrate by project history:

- Alert if `diag/val_skill_persistence < 0` for 3+ epochs.
- Alert if `diag/grad_norm` spikes above 5x rolling median.
- Alert if `diag/train_pred_std_c* / diag/train_target_std_c* < 0.5` for multiple channels.
- Alert if `diag/val_event_mse / diag/val_quiet_mse` grows epoch-over-epoch.

---

## Practical Runbook

If training is unstable:
1. Check `diag/grad_norm`, `diag/step_time_s`, and throughput first.
2. Verify normalization stats were loaded from train split only.
3. Reduce LR or increase clipping before architectural changes.

If validation skill is poor:
1. Check persistence/climatology skill first.
2. Inspect RMSE map and residual histogram for systematic bias.
3. Confirm event-region performance is improving, not just quiet-region performance.

If behavior looks over-smoothed:
1. Compare per-channel std diagnostics.
2. Inspect example panels for washed structures.
3. Rebalance loss weights and improve temporal rollout strategy.

---

## Current Limitations

- Lead-time table is proxy-based, not true horizon-labeled evaluation.
- Residual histogram currently uses validation sample(s), not full-distribution accumulation.
- Event split uses occupancy-like channel thresholding and may require domain-specific tuning.

These are still highly useful diagnostics for rapid iteration and regressions.

---

## Recommended W&B Dashboard Layout

Use a consistent dashboard so every run is reviewed the same way.

### One-Command Setup (Auto-Generated Report)

You can auto-create a dashboard report with:

```bash
cd src/weatherlearn/PTL
python setup_wandb_dashboard.py --entity <your-wandb-entity> --project <your-wandb-project>
```

Optional filters and dry run:

```bash
python setup_wandb_dashboard.py \
  --entity <your-wandb-entity> \
  --project <your-wandb-project> \
  --query "config.method = \"grid\"" \
  --dry-run
```

Notes:

- The script creates a report layout with core scalar diagnostics panels.
- Media/table keys are documented in the report text for quick manual pinning in the W&B UI when SDK versions differ.

### Section A: Training Health

Add these charts:

- Line chart: `train_loss`, `train_mse`, `val_loss`, `val_mse`
- Line chart: `diag/grad_norm`
- Line chart: `diag/step_time_s`, `diag/samples_per_sec`
- Multi-line chart: `diag/train_pred_std_c*` and `diag/train_target_std_c*`

What to check first:

1. Optimization is stable (`grad_norm` not diverging).
2. Throughput is stable (no growing stalls).
3. Prediction distributions are not collapsing.

### Section B: Skill vs Baselines

Add these charts:

- Line chart: `diag/val_skill_persistence`
- Line chart: `diag/val_skill_climatology`
- Line chart: `diag/val_event_mse`, `diag/val_quiet_mse`

What to check:

1. Skill metrics are positive and trending up.
2. Event-region error is improving, not just quiet-region error.

### Section C: Spatial Diagnostics

Add these media panels:

- Image panel: `diag/val_rmse_map`
- Image panel: `diag/val_example_panel`
- Histogram panel: `diag/val_residual_hist`

What to check:

1. RMSE hot spots are explainable (coverage/sparsity), not broad unexplained structures.
2. Example panels preserve geometry and avoid blur/lag.
3. Residual histogram remains centered near zero.

### Section D: Rollout Diagnostics

Add these charts:

- Table panel: `diag/val_lead_time_table`

Recommended derived panel:

- Convert `diag/val_lead_time_table` into a line chart (`horizon` vs `mse`) in a custom W&B panel.

What to check:

1. Error growth with horizon is gradual.
2. No sharp early-horizon jumps.

### Section E: EMA / Experiment Controls

Track as config fields in W&B:

- `use_ema`
- `use_ema_eval`
- `ema_decay`
- `ema_warmup_steps`
- `cache_stats`
- `diagnostics_interval`

Why:

- Enables apples-to-apples comparison between runs and prevents misreading metric shifts caused by config changes.

---

## Recommended Review Cadence

Per run:

1. Every epoch: check Sections A and B.
2. Every validation epoch with images: check Section C.
3. Every major config change: compare Section D and EMA config in Section E.

Per experiment sweep:

1. Rank by `diag/val_skill_persistence` and `val_loss`.
2. Tie-break using event-region error (`diag/val_event_mse`).
3. Reject runs with unstable training health even if best single-point metric.
