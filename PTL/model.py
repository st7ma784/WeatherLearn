

"""
MinioToDisk Module
==================

This module provides functionality for downloading all objects from a MinIO bucket to a local folder. It supports 
multithreaded downloads for improved performance and ensures that the folder structure of the bucket is preserved 
locally.

Key Features
------------

- **MinIO Integration**:
  - Uses the `Minio` client to interact with MinIO buckets.
  - Supports authentication using access and secret keys.

- **Multithreaded Downloads**:
  - Utilizes `ThreadPoolExecutor` for concurrent downloads of objects from the bucket.
  - Displays progress using the `tqdm` progress bar.

- **Folder Structure Preservation**:
  - Ensures that the folder structure of the bucket is replicated in the local target folder.

- **Error Handling**:
  - Handles errors during object downloads and logs issues for debugging.

Dependencies
------------

- **minio**:
  Provides the MinIO client for interacting with MinIO buckets.

- **os**:
  Used for file system operations, such as creating directories and saving files.

- **concurrent.futures**:
  Enables multithreaded downloads using `ThreadPoolExecutor`.

- **tqdm**:
  Displays a progress bar for tracking download progress.

Usage
-----

To download all objects from a MinIO bucket to a local folder, use the `download_minio_bucket_to_folder` function. 
You can also run the script directly with command-line arguments to specify the MinIO configuration, bucket name, 
and target folder.

Example Command:
----------------

.. code-block:: bash

   python MinioToDisk.py --host localhost --port 9000 --access_key minioadmin --secret_key minioadmin \
                         --bucket mybucket --target_folder /path/to/local/folder

"""





import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch import nn
import numpy as np
from utils import get_shift_window_mask, WindowReverse, WindowPartition, PatchEmbed2D, PatchRecovery2D, get_pad3d, DownSample, UpSample, get_earth_position_index, Crop3D
import math
import time


def norm_cdf(x):
    """
    Computes the standard normal cumulative distribution function.

    Args:
        x (float): Input value.

    Returns:
        float: The cumulative probability for the input value.
    """
    return (1. + math.erf(x / math.sqrt(2.))) / 2.


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.

    Supports both shifted and non-shifted windows.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution [pressure levels, latitude, longitude].
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value. Default: True.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weights. Default: 0.0.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0.
        attn_mask (torch.Tensor, optional): Attention mask. Default: None.
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_mask=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (input_resolution[1] // window_size[1])

        earth_position_bias_table = torch.zeros((window_size[0] ** 2) * (window_size[1] ** 2) * (window_size[2] * 2 - 1),
                        self.type_of_windows, num_heads)
      # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(window_size)  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)
        self.forward_WMask=self.forward_null
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
            self.attn_mask=attn_mask
            self.forward_WMask=self.forward_mask
            self.nLon = self.attn_mask.shape[0]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        std=.02
        a, b = -2, 2
        mean=0
        l = norm_cdf((-2 - 0) / std)
        u = norm_cdf((2 - 0) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        earth_position_bias_table.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        earth_position_bias_table.erfinv_()

        # Transform to proper mean, std
        earth_position_bias_table.mul_(std * math.sqrt(2.))
        earth_position_bias_table.add_(mean)

        # Clamp to ensure it's in the proper range
        earth_position_bias_table.clamp_(min=a, max=b)
        self.softmax = nn.Softmax(dim=-1)

        earth_position_bias = earth_position_bias_table[earth_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows, -1
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        self.earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous().unsqueeze(0)
        self.earth_position_bias= nn.Parameter(self.earth_position_bias, requires_grad=True)

    def forward_null(self, x: torch.Tensor, B_, nW_, N):
        """
        Forward pass without applying attention mask.

        Args:
            x (torch.Tensor): Input tensor.
            B_ (int): Batch size.
            nW_ (int): Number of windows.
            N (int): Window size.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x

    def forward_mask(self, x: torch.Tensor, B_, nW_, N):
        """
        Forward pass with attention mask applied.

        Args:
            x (torch.Tensor): Input tensor.
            B_ (int): Batch size.
            nW_ (int): Number of windows.
            N (int): Window size.

        Returns:
            torch.Tensor: Output tensor with mask applied.
        """
        x = x.view(-1, self.nLon, self.num_heads, nW_, N, N) + self.attn_mask.unsqueeze(1).unsqueeze(0)
        x = x.view(-1, self.num_heads, nW_, N, N)
        return x

    def forward(self, x: torch.Tensor, mask=None):
        """
        Forward pass for 3D attention with earth position bias.

        Args:
            x (torch.Tensor): Input features with shape (B * num_lon, num_pl*num_lat, N, C).
            mask (torch.Tensor, optional): Attention mask. Default: None.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B_, nW_, N, C = x.shape
        hd = C // self.num_heads
        Bn = B_ * nW_

        # QKV: (B_, nW_, N, 3, nH, hd) → (3, B_, nH, nW_, N, hd) → unbind
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, hd)).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv.unbind(0)  # (B_, nH, nW_, N, hd)

        # Build additive bias. earth_position_bias: (1, nH, nW_, N, N).
        # Expand over B_, apply optional shift mask, then permute nW_ adjacent
        # to B_ so the reshape to (Bn, nH, N, N) is contiguous-safe.
        bias = self.forward_WMask(
            self.earth_position_bias.expand(B_, -1, -1, -1, -1), B_, nW_, N
        )                                                           # (B_, nH, nW_, N, N)
        bias = bias.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, N)

        q = q.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)
        k = k.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)
        v = v.permute(0, 2, 1, 3, 4).reshape(Bn, self.num_heads, N, hd)

        # mem_efficient / Flash-Attention-2 backend: never stores the N² matrix.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )  # (Bn, nH, N, hd)

        # (Bn, nH, N, hd) → (B_, nW_, N, C)
        out = out.permute(0, 2, 1, 3).reshape(B_, nW_, N, C)
        return self.proj_drop(self.proj(out))


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Args:
        drop_prob (float): Probability of dropping a path. Default: 0.0.
        scale_by_keep (bool): Whether to scale by keep probability. Default: True.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        keep_prob = 1.0 - drop_prob
        self.scale_factor = (1.0 / keep_prob) if (keep_prob > 0 and scale_by_keep) else 1.0

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        rand_tensor = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(1 - self.drop_prob) * self.scale_factor
        return x * rand_tensor

    def extra_repr(self):
        """
        Returns a string representation of the DropPath module.

        Returns:
            str: String representation.
        """
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class EarthSpecificBlock(nn.Module):
    """
    3D Transformer Block with earth-specific attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, and value. Default: True.
        qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Default: None.
        drop (float, optional): Dropout rate. Default: 0.0.
        attn_drop (float, optional): Attention dropout rate. Default: 0.0.
        drop_path (float, optional): Stochastic depth rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=None, shift_size=None, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        # F.pad on channel-last (B, Pl, Lat, Lon, C) — avoids two permutes per block per step.
        # F.pad reads padding right-to-left: (C_l, C_r, Lon_l, Lon_r, Lat_t, Lat_b, Pl_f, Pl_b)
        self._fpad = (0, 0) + padding

        pad_resolution = list(input_resolution)
        pad_resolution[0] += (padding[-1] + padding[-2])
        pad_resolution[1] += (padding[2] + padding[3])
        pad_resolution[2] += (padding[0] + padding[1])

        self.pad_resolution = pad_resolution
        self.WindowPartition=WindowPartition((2,*self.pad_resolution,dim),self.window_size)
        self.WindowReverse = WindowReverse(window_size,*self.pad_resolution,dim)
        self.Crop3d = Crop3D(padding)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = bool(shift_pl and shift_lon and shift_lat)
        self.neg_shifts = (-shift_pl, -shift_lat, -shift_lon) if self.roll else (0, 0, 0)
        self.pos_shifts = (shift_pl, shift_lat, shift_lon) if self.roll else (0, 0, 0)
        attn_mask = None
        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, self.shift_size)

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,attn_mask=attn_mask
        )

        # Dispatch once at init — no per-call branch in forward.
        self._roll_neg = self.negrollX if self.roll else self.null_rollX
        self._roll_pos = self.posrollX if self.roll else self.null_rollX

    def negrollX(self, x: torch.Tensor):
        return torch.roll(x, shifts=self.neg_shifts, dims=(1, 2, 3))

    def null_rollX(self, x: torch.Tensor):
        return x

    def posrollX(self, x: torch.Tensor):
        return torch.roll(x, shifts=self.pos_shifts, dims=(1, 2, 3))

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = x.unflatten(1, self.input_resolution)
        x = F.pad(x, self._fpad)
        x = self._roll_neg(x)
        shifted_x = self.WindowReverse(self.attn(self.WindowPartition(x)))
        shifted_x = self._roll_pos(shifted_x)
        shifted_x = self.Crop3d(shifted_x)
        shifted_x = shifted_x.flatten(1, 3)
        shifted_x = shortcut + self.drop_path(shifted_x)
        shifted_x = shifted_x + self.drop_path(self.mlp(self.norm2(shifted_x)))
        return shifted_x
    


class IdentityFiLM(nn.Module):
    """Drop-in no-op for FiLMLayer when solar conditioning is disabled."""
    def forward(self, x: torch.Tensor, solar_vec, solar_mask) -> torch.Tensor:
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on a solar-wind scalar vector.

    Projects `solar_dim` scalars → (γ, β) of size `feature_dim` and applies
        out = (1 + γ * mask) * x + β * mask
    so mask=0 is an exact identity (model is unchanged when data is absent).
    """

    def __init__(self, solar_dim: int, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(solar_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        # Initialise to near-zero so conditioning starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor,
                solar_vec: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, N, C) token sequence
            solar_vec:  (B, solar_dim) conditioning scalars
            mask:       (B, 1) float — 1 if data available, 0 if missing
        """
        gamma, beta = self.mlp(solar_vec).chunk(2, dim=-1)   # each (B, C)
        gamma = gamma.unsqueeze(1) * mask.unsqueeze(-1)       # (B, 1, C)
        beta  = beta.unsqueeze(1)  * mask.unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class Pangu(pl.LightningModule):
    """
    Pangu: A PyTorch implementation of the Pangu-Weather model.

    Args:
        embed_dim (int): Patch embedding dimension. Default: 128.
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
        learning_rate (float): Learning rate for the optimizer. Default: 1e-4.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, embed_dim=128, num_heads=(8, 16, 16, 8), window_size=(2, 8, 16), learning_rate=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate=learning_rate
        self.time_steps= kwargs.get("time_step", 1)
        drop_path = np.linspace(0, 0.2, 8).tolist()
        self.criterion = torch.nn.SmoothL1Loss(beta=0.1, reduction='mean')
        self.metric_mse = torch.nn.MSELoss(reduction='mean')
        self.grid_size=kwargs.get("grid_size",300)
        self.mlp_ratio=kwargs.get("mlp_ratio",4)
        self.noise_factor=kwargs.get("noise_factor",0.1)
        self.weight_decay = kwargs.get("weight_decay", 0.05)
        self.use_ema = kwargs.get("use_ema", True)
        self.use_ema_eval = kwargs.get("use_ema_eval", False)
        self.ema_decay = float(kwargs.get("ema_decay", 0.999))
        self.ema_warmup_steps = int(kwargs.get("ema_warmup_steps", 200))
        self.ema_shadow = {}
        self._ema_backup = None
        self.log_diagnostics = kwargs.get("log_diagnostics", True)
        self.diagnostics_interval = int(kwargs.get("diagnostics_interval", 50))
        self.log_images_every_n_val_epochs = int(kwargs.get("log_images_every_n_val_epochs", 1))
        self._batch_start_time = None
        self._last_grad_norm = None
        self._last_step_time = None
        self._last_throughput = None
        self._lead_time_curve = None
        self._val_example = None
        self.num_input_frames = int(kwargs.get("num_input_frames", 1))
        self.patchembed2d = PatchEmbed2D(
            img_size=(self.grid_size, self.grid_size),
            patch_size=(4, 4),
            in_chans=5 * self.num_input_frames,
            embed_dim=embed_dim,
        )
        reduced_grid=(1, self.grid_size//4,self.grid_size//4)
        self.incremental_step=0
        self.reduced_grid= (self.grid_size//4,self.grid_size//4)
        further_reduced_grid=(1, self.grid_size//6,self.grid_size//6)
        self.layer1 =  nn.Sequential(*[
            EarthSpecificBlock(dim=embed_dim, input_resolution=reduced_grid, num_heads=num_heads[0], window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=self.mlp_ratio,
                               qkv_bias=True,
                               qk_scale=None, drop=0., attn_drop=0.,
                               drop_path=drop_path[2:][i] if isinstance(drop_path, list) else drop_path[2:],
                               norm_layer=nn.LayerNorm)
            for i in range(2)
        ])

        self.downsample = DownSample(in_dim=embed_dim, input_resolution=reduced_grid, output_resolution=further_reduced_grid)
        self.layer2 =nn.Sequential(*[
            EarthSpecificBlock(dim=embed_dim*2, input_resolution=further_reduced_grid, num_heads=num_heads[1], window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=self.mlp_ratio,
                               qkv_bias=True,
                               qk_scale=None, drop=0., attn_drop=0.,
                               drop_path=drop_path[2:][i] if isinstance(drop_path, list) else drop_path[2:],
                               norm_layer=nn.LayerNorm)
            for i in range(6)
        ])
        
        self.layer3 = nn.Sequential(*[
            EarthSpecificBlock(dim=embed_dim * 2, input_resolution=further_reduced_grid, num_heads=num_heads[2], window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=self.mlp_ratio,
                               qkv_bias=True,
                               qk_scale=None, drop=0., attn_drop=0.,
                               drop_path=drop_path[2:][i] if isinstance(drop_path, list) else drop_path[2:],
                               norm_layer=nn.LayerNorm)
            for i in range(6)
        ])
        
        
        self.upsample = UpSample(embed_dim * 2, embed_dim, further_reduced_grid, reduced_grid)
        self.layer4 = nn.Sequential(*[
            EarthSpecificBlock(dim=embed_dim, input_resolution=reduced_grid, num_heads=num_heads[3], window_size=window_size,
                               shift_size=(0, 0, 0) if i % 2 == 0 else None, mlp_ratio=self.mlp_ratio,
                               qkv_bias=True,
                               qk_scale=None, drop=0., attn_drop=0.,
                               drop_path=drop_path[2:][i] if isinstance(drop_path, list) else drop_path[2:],
                               norm_layer=nn.LayerNorm)
            for i in range(2)
        ])
        
        self.patchrecovery2d = PatchRecovery2D((self.grid_size, self.grid_size), (4, 4), (1+self.time_steps) * embed_dim, 5)
        # Zero-init output head: delta_hat = 0 at step 0 = persistence forecast.
        # Gives positive climatology skill from epoch 1 and eliminates the large
        # gradient contribution from random outputs over all inactive pixels.
        nn.init.zeros_(self.patchrecovery2d.conv.weight)
        nn.init.zeros_(self.patchrecovery2d.conv.bias)

        # Place more weight on observed regions/channels that better represent ionospheric structure.
        self.register_buffer("channel_weights",
            torch.tensor([1.2, 1.0, 1.0, 1.5, 1.1], dtype=torch.float32).view(1, -1, 1, 1))

        # ── Optional FiLM solar-wind conditioning ────────────────────────────
        # solar_wind_dim=0 disables conditioning entirely (no extra parameters).
        self.solar_wind_dim = int(kwargs.get("solar_wind_dim", 0))
        self.solar_wind_dropout = float(kwargs.get("solar_wind_dropout", 0.2))
        if self.solar_wind_dim > 0:
            self.film_enc  = FiLMLayer(self.solar_wind_dim, embed_dim)
            self.film_bot  = FiLMLayer(self.solar_wind_dim, embed_dim * 2)
            self.film_bot2 = FiLMLayer(self.solar_wind_dim, embed_dim * 2)
            self.film_dec  = FiLMLayer(self.solar_wind_dim, embed_dim)
        else:
            self.film_enc = self.film_bot = self.film_bot2 = self.film_dec = IdentityFiLM()

    def _wandb_run(self):
        if self.logger is None or not hasattr(self.logger, "experiment"):
            return None
        return self.logger.experiment

    def _wandb_log(self, payload):
        run = self._wandb_run()
        if run is None or not hasattr(run, "log"):
            return
        run.log(payload, step=int(self.global_step))

    def _should_log_diag_step(self, batch_idx):
        return self.log_diagnostics and (self.diagnostics_interval > 0) and (batch_idx % self.diagnostics_interval == 0)

    def on_before_optimizer_step(self, optimizer):
        # Fires after scaler.unscale_() in 16-mixed — grads are true fp32 values here,
        # not the artificially inflated fp16-scaled ones seen in on_after_backward.
        if not self.log_diagnostics:
            return
        sq_norm_sum = None
        for p in self.parameters():
            if p.grad is None:
                continue
            g2 = p.grad.detach().float().pow(2).sum()
            sq_norm_sum = g2 if sq_norm_sum is None else (sq_norm_sum + g2)
        if sq_norm_sum is not None:
            self._last_grad_norm = torch.sqrt(sq_norm_sum).detach()

    def _ema_named_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.dtype.is_floating_point:
                yield name, param

    def _init_ema(self):
        if not self.use_ema:
            return
        self.ema_shadow = {
            name: param.detach().clone()
            for name, param in self._ema_named_params()
        }

    def _current_ema_decay(self):
        if self.ema_warmup_steps <= 0:
            return self.ema_decay
        warmup_ratio = min(1.0, float(self.global_step + 1) / float(self.ema_warmup_steps))
        return self.ema_decay * warmup_ratio

    def _update_ema(self):
        if not self.use_ema:
            return
        if not self.ema_shadow:
            self._init_ema()
        decay = self._current_ema_decay()
        one_minus_decay = 1.0 - decay
        for name, param in self._ema_named_params():
            self.ema_shadow[name].mul_(decay).add_(param.detach(), alpha=one_minus_decay)

    def _swap_to_ema_weights(self):
        if not (self.use_ema and self.use_ema_eval and self.ema_shadow):
            return
        self._ema_backup = {
            name: param.detach().clone()
            for name, param in self._ema_named_params()
        }
        for name, param in self._ema_named_params():
            param.data.copy_(self.ema_shadow[name])

    def _restore_from_ema_weights(self):
        if self._ema_backup is None:
            return
        for name, param in self._ema_named_params():
            param.data.copy_(self._ema_backup[name])
        self._ema_backup = None

    def on_train_start(self):
        self._init_ema()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._update_ema()
        if self._batch_start_time is not None:
            self._last_step_time = float(time.perf_counter() - self._batch_start_time)
            bs = int(batch[0].shape[0]) if isinstance(batch, (tuple, list)) else 0
            if self._last_step_time > 0 and bs > 0:
                self._last_throughput = float(bs / self._last_step_time)

        if self._should_log_diag_step(batch_idx):
            log_payload = {}
            if self._last_grad_norm is not None:
                log_payload["diag/grad_norm"] = float(self._last_grad_norm.item())
            if self._last_step_time is not None:
                log_payload["diag/step_time_s"] = self._last_step_time
            if self._last_throughput is not None:
                log_payload["diag/samples_per_sec"] = self._last_throughput
            if log_payload:
                self._wandb_log(log_payload)

    def on_validation_start(self):
        self._swap_to_ema_weights()

    def on_validation_epoch_start(self):
        dev = self.device
        self._val_sse_map         = torch.zeros(5, self.grid_size, self.grid_size, device=dev)
        self._val_count           = 0
        # Tensor accumulators — no .item() / GPU sync per batch.
        self._val_event_sse       = torch.zeros(1, device=dev)
        self._val_event_count     = torch.zeros(1, device=dev)
        self._val_quiet_sse       = torch.zeros(1, device=dev)
        self._val_quiet_count     = torch.zeros(1, device=dev)
        self._val_model_sse       = torch.zeros(1, device=dev)
        self._val_baseline_sse    = torch.zeros(1, device=dev)
        self._val_baseline_clim_sse = torch.zeros(1, device=dev)
        self._lead_time_curve     = None
        self._val_example         = None
        self._val_example_tensors = None   # raw GPU tensors, converted in on_validation_end

    def on_validation_epoch_end(self):
        # self.log() is only valid here, not in on_validation_end
        self._log_validation_diagnostics_scalars()

    def on_validation_end(self):
        self._log_validation_diagnostics_images()
        self._restore_from_ema_weights()

    def on_test_start(self):
        self._swap_to_ema_weights()

    def on_test_end(self):
        self._restore_from_ema_weights()

    # ── visualisation helpers ────────────────────────────────────────────────

    _CH_NAMES  = ["Velocity", "Vel.SD", "Kvect", "Occupancy", "Density"]
    _CH_CMAPS  = ["RdBu_r",   "viridis", "twilight", "binary",   "plasma"]
    _CH_UNITS  = ["m/s",       "m/s",    "°",        "",         "log(n)"]

    def _plot_channel_grid(self, x, y, y_hat, title_prefix="val", solar_info=None):
        """
        4-row comparison panel per channel:
          Row 0 — Input (current state)
          Row 1 — Target (next state, ground truth)
          Row 2 — Prediction
          Row 3 — |Error| = |pred − target|

        solar_info: optional dict with IMF scalars to annotate.
        """
        n_ch   = x.shape[0]
        n_rows = 4
        fig, axes = plt.subplots(n_rows, n_ch,
                                 figsize=(2.8 * n_ch, 2.8 * n_rows),
                                 squeeze=False)
        row_labels = ["Input", "Target", "Prediction", "|Error|"]

        for c in range(n_ch):
            cmap  = self._CH_CMAPS[c] if c < len(self._CH_CMAPS) else "coolwarm"
            name  = self._CH_NAMES[c] if c < len(self._CH_NAMES) else f"c{c}"
            unit  = self._CH_UNITS[c] if c < len(self._CH_UNITS) else ""
            label = f"{name}" + (f" [{unit}]" if unit else "")

            # shared colour range for input/target/pred
            vmin = float(np.nanmin([x[c], y[c], y_hat[c]]))
            vmax = float(np.nanmax([x[c], y[c], y_hat[c]]))

            for row, data in enumerate([x[c], y[c], y_hat[c]]):
                ax = axes[row, c]
                im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                               interpolation="nearest", origin="lower")
                ax.set_title(f"{row_labels[row]}\n{label}", fontsize=7)
                ax.axis("off")
                if c == n_ch - 1:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # error row
            err = np.abs(y_hat[c] - y[c])
            ax_e = axes[3, c]
            im_e = ax_e.imshow(err, cmap="hot", vmin=0,
                               vmax=max(float(err.max()), 1e-6),
                               interpolation="nearest", origin="lower")
            ax_e.set_title(f"|Error|\n{label}", fontsize=7)
            ax_e.axis("off")
            if c == n_ch - 1:
                fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)

        if solar_info:
            parts = [f"{k}={v:.2f}" for k, v in solar_info.items()]
            fig.suptitle("IMF: " + "  ".join(parts), fontsize=8, y=1.01)

        plt.tight_layout()
        return fig

    def _plot_rmse_map(self, rmse_map):
        channels = rmse_map.shape[0]
        fig, axes = plt.subplots(1, channels, figsize=(3 * channels, 3),
                                 squeeze=False)
        for c in range(channels):
            name = self._CH_NAMES[c] if c < len(self._CH_NAMES) else f"c{c}"
            ax   = axes[0, c]
            im   = ax.imshow(rmse_map[c], cmap="magma", origin="lower")
            ax.set_title(f"RMSE\n{name}", fontsize=8)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    def _log_validation_diagnostics_scalars(self):
        """Called from on_validation_epoch_end — self.log() is allowed here.
        All .item() / CPU transfers happen here, once per epoch."""
        if not self.log_diagnostics or self._val_count <= 0:
            return

        rmse_map = torch.sqrt(self._val_sse_map / max(self._val_count, 1)).detach().cpu().float().numpy()
        self._last_rmse_map = rmse_map  # cache for image logging

        model_sse = self._val_model_sse.item()
        skill_pers = float(1.0 - model_sse / max(self._val_baseline_sse.item(), 1e-8))
        skill_clim = float(1.0 - model_sse / max(self._val_baseline_clim_sse.item(), 1e-8))

        payload = {
            "diag/val_skill_persistence": skill_pers,
            "diag/val_skill_climatology": skill_clim,
            "diag/val_event_mse": float(self._val_event_sse.item() / max(self._val_event_count.item(), 1.0)),
            "diag/val_quiet_mse": float(self._val_quiet_sse.item() / max(self._val_quiet_count.item(), 1.0)),
        }
        for c, name in enumerate(self._CH_NAMES):
            if c < rmse_map.shape[0]:
                payload[f"diag/val_rmse_{name.lower().replace('.','_')}"] = float(rmse_map[c].mean())

        self._wandb_log(payload)
        self.log("val_skill_persistence", skill_pers, on_epoch=True)
        self.log("val_skill_climatology", skill_clim, on_epoch=True)

    def _log_validation_diagnostics_images(self):
        """Called from on_validation_end — image/table logging to W&B experiment directly."""
        if not self.log_diagnostics or self._val_count <= 0:
            return

        run = self._wandb_run()
        if run is None:
            return

        try:
            import wandb
        except Exception:
            return

        rmse_map = getattr(self, "_last_rmse_map", None)
        if rmse_map is None:
            return

        if (self.current_epoch % max(self.log_images_every_n_val_epochs, 1)) == 0:
            rmse_fig = self._plot_rmse_map(rmse_map)
            self._wandb_log({"diag/val_rmse_map": wandb.Image(rmse_fig)})
            plt.close(rmse_fig)

            # Convert stored raw tensors to numpy here — single CPU transfer per epoch.
            if self._val_example_tensors is not None:
                t_x, t_y, t_yhat, t_sv = self._val_example_tensors
                ex_x    = t_x.cpu().float().numpy()
                ex_y    = t_y.cpu().float().numpy()
                ex_yhat = t_yhat.cpu().float().numpy()
                ex_res  = (t_yhat - t_y).cpu().float().numpy()
                sv_list = t_sv.cpu().tolist()
                keys    = ["Bx", "By", "Bz", "Kp", "Vx"][:len(sv_list)]
                ex_solar = dict(zip(keys, sv_list)) if any(v != 0.0 for v in sv_list) else None
                panel_fig = self._plot_channel_grid(
                    ex_x, ex_y, ex_yhat,
                    title_prefix=f"epoch {self.current_epoch}",
                    solar_info=ex_solar,
                )
                self._wandb_log({"diag/val_example_panel": wandb.Image(panel_fig)})
                plt.close(panel_fig)
                self._wandb_log({"diag/val_residual_hist": wandb.Histogram(ex_res.reshape(-1))})

            if self._lead_time_curve is not None:
                lead_table = wandb.Table(columns=["horizon", "mse"])
                for h, mse in self._lead_time_curve:
                    lead_table.add_data(int(h), float(mse))
                self._wandb_log({"diag/val_lead_time_table": lead_table})
        
    def forward(self, x):
        """
        Forward pass for the Pangu model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        layer1_out = self.layer1(x)
        x = self.downsample(layer1_out)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        return x, layer1_out

    def _forward_with_solar(self, x, solar_vec=None, solar_mask=None):
        """Full encoder-decoder pass with FiLM conditioning (IdentityFiLM when disabled)."""
        layer1_out = self.film_enc(self.layer1(x), solar_vec, solar_mask)

        x = self.film_bot(self.layer2(self.downsample(layer1_out)), solar_vec, solar_mask)
        x = self.film_bot2(self.layer3(x), solar_vec, solar_mask)

        x = self.film_dec(self.layer4(self.upsample(x)), solar_vec, solar_mask)
        return x, layer1_out

    def _forecast_latent(self, x, add_noise=False, solar_vec=None, solar_mask=None):
        noise_scale = self.noise_factor if (add_noise and self.noise_factor > 0) else 0.0
        skips = []

        if noise_scale > 0.0:
            for _ in range(self.time_steps):
                x, skip = self._forward_with_solar(
                    x + torch.randn_like(x) * noise_scale, solar_vec, solar_mask)
                skips.append(skip)
        else:
            for _ in range(self.time_steps):
                x, skip = self._forward_with_solar(x, solar_vec, solar_mask)
                skips.append(skip)

        # torch.stack keeps all skips in the autograd graph (no in-place writes).
        skip_cat = torch.stack(skips, dim=2).reshape(x.shape[0], x.shape[1], self.time_steps * x.shape[2])
        return torch.cat((x, skip_cat), dim=-1)

    def _decode_from_latent(self, latent):
        decoded = latent.transpose(1, 2).unflatten(2, self.reduced_grid)
        return self.patchrecovery2d(decoded)

    def _predict(self, x_in, add_noise=False, solar_vec=None, solar_mask=None):
        x = self.patchembed2d(x_in)
        x = x.flatten(2, 3).transpose(1, 2)
        x = self._forecast_latent(x, add_noise=add_noise, solar_vec=solar_vec, solar_mask=solar_mask)
        # residual skip from most-recent frame only (last 5 channels when T > 1)
        return x_in[:, -5:] + self._decode_from_latent(x)

    @torch._dynamo.disable
    def _lead_time_mse_curve(self, x_emb, x_in, y, solar_vec=None, solar_mask=None):
        if self.time_steps <= 1:
            lat = self._forecast_latent(x_emb, add_noise=False, solar_vec=solar_vec, solar_mask=solar_mask)
            y_hat = x_in[:, -5:] + self._decode_from_latent(lat)
            return [(1, float(self.metric_mse(y_hat, y).item()))]

        curve = []
        for h in range(1, min(self.time_steps, 6) + 1):
            old_steps = self.time_steps
            self.time_steps = h
            with torch.no_grad():
                lat   = self._forecast_latent(x_emb, add_noise=False, solar_vec=solar_vec, solar_mask=solar_mask)
                y_hat = x_in[:, -5:] + self._decode_from_latent(lat)
                mse   = self.metric_mse(y_hat, y)
            self.time_steps = old_steps
            curve.append((h, float(mse.item())))
        return curve

    def _weighted_loss(self, y_hat, y):
        error = F.smooth_l1_loss(y_hat, y, reduction='none', beta=0.1)
        weighted = error * self.channel_weights
        # After z-score normalisation, active cells (raw occ=1) have y[:,3]>0;
        # inactive cells (raw occ=0) have y[:,3]<0.
        occ_weight = torch.where(y[:, 3:4] > 0.0,
                                 torch.full_like(y[:, 3:4], 4.0),
                                 torch.ones_like(y[:, 3:4]))
        occ_weight = occ_weight.expand_as(weighted)
        return (weighted * occ_weight).sum() / occ_weight.sum()

    def _unpack_solar(self, batch):
        """Extract solar wind vector and availability mask from a (x, sv, sm, y) batch."""
        return batch[1].float(), batch[2].float()

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        solar_vec, solar_mask = self._unpack_solar(batch)
        x_in, y = batch[0], batch[-1]
        delta_target = y - x_in[:, -5:]
        # Random dropout of solar conditioning during training
        if self.solar_wind_dropout > 0:
            drop = (torch.rand(solar_mask.shape[0], device=solar_mask.device)
                    < self.solar_wind_dropout).float().unsqueeze(1)
            solar_mask = solar_mask * (1.0 - drop)
        x = self.patchembed2d(x_in)
        x = x.flatten(2, 3).transpose(1, 2)
        delta_hat = self._decode_from_latent(
            self._forecast_latent(x, add_noise=True, solar_vec=solar_vec, solar_mask=solar_mask))
        loss = self._weighted_loss(delta_hat, delta_target)
        y_hat = x_in[:, -5:] + delta_hat
        mse = self.metric_mse(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_epoch=True, prog_bar=False)

        if self._should_log_diag_step(batch_idx):
            self._log_train_diag(delta_hat.detach(), delta_target.detach())

        return loss

    @torch._dynamo.disable
    def _log_train_diag(self, x, y):
        pred_mean   = x.mean(dim=(0, 2, 3))
        pred_std    = x.std(dim=(0, 2, 3), unbiased=False)
        target_mean = y.mean(dim=(0, 2, 3))
        target_std  = y.std(dim=(0, 2, 3), unbiased=False)
        payload = {}
        for c in range(x.shape[1]):
            payload[f"diag/train_pred_mean_c{c}"]   = float(pred_mean[c].item())
            payload[f"diag/train_pred_std_c{c}"]    = float(pred_std[c].item())
            payload[f"diag/train_target_mean_c{c}"] = float(target_mean[c].item())
            payload[f"diag/train_target_std_c{c}"]  = float(target_std[c].item())
        self._wandb_log(payload)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        solar_vec, solar_mask = self._unpack_solar(batch)
        x_in, y = batch[0], batch[-1]
        x = self.patchembed2d(x_in)
        x = x.flatten(2, 3).transpose(1, 2)
        latent = self._forecast_latent(x, add_noise=False, solar_vec=solar_vec, solar_mask=solar_mask)
        delta_hat = self._decode_from_latent(latent)
        loss = self._weighted_loss(delta_hat, y - x_in[:, -5:])
        y_hat = x_in[:, -5:] + delta_hat
        mse = self.metric_mse(y_hat, y)

        sq_err = (y_hat.detach() - y.detach()).pow(2)
        self._val_sse_map += sq_err.sum(dim=0)
        self._val_count += y.shape[0]

        occ_mask   = (y[:, 3:4].abs() > 0.05).expand_as(sq_err)
        quiet_mask = ~occ_mask
        self._val_event_sse   += (sq_err * occ_mask).sum()
        self._val_event_count += occ_mask.sum()
        self._val_quiet_sse   += (sq_err * quiet_mask).sum()
        self._val_quiet_count += quiet_mask.sum()
        self._val_model_sse        += sq_err.sum()
        self._val_baseline_sse     += (x_in[:, -5:].detach() - y.detach()).pow(2).sum()
        self._val_baseline_clim_sse += y.detach().pow(2).sum()

        # Store raw tensors for first batch; CPU transfer deferred to on_validation_end.
        if batch_idx == 0:
            self._val_example_tensors = (
                x_in[0].detach(),
                y[0].detach(),
                y_hat[0].detach(),
                solar_vec[0].detach(),
            )
            if self.log_diagnostics:
                self._lead_time_curve = self._lead_time_mse_curve(
                    x, x_in, y, solar_vec=solar_vec, solar_mask=solar_mask)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse, on_epoch=True, prog_bar=True)
        return loss

    def plot_and_log_data(self, x, y, y_hat, batch_idx):
        batch_size = x.shape[0]
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        y_hat = y_hat.cpu().detach().numpy()
        os.makedirs(f"results/{batch_idx}", exist_ok=True)
        for item in range(batch_size):
            fig, ax = plt.subplots(3, 5, figsize=(14, 8))
            for i in range(5):
                ax[0, i].imshow(x[item, i, :, :])
                ax[0, i].axis("off")
                ax[1, i].imshow(y[item, i, :, :])
                ax[1, i].axis("off")
                ax[2, i].imshow(y_hat[item, i, :, :])
                ax[2, i].axis("off")
            plt.tight_layout()
            plt.savefig(f"results/{batch_idx}/{item}.png")
            plt.close()
        if self.logger is not None and hasattr(self.logger, "log_image"):
            self.logger.log_image("examples", [f"results/{batch_idx}/{item}.png" for item in range(batch_size)])

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        solar_vec, solar_mask = self._unpack_solar(batch)
        x_in, y = batch[0], batch[-1]
        x = self.patchembed2d(x_in)
        x = x.flatten(2, 3).transpose(1, 2)
        delta_hat = self._decode_from_latent(
            self._forecast_latent(x, add_noise=False, solar_vec=solar_vec, solar_mask=solar_mask))
        loss = self._weighted_loss(delta_hat, y - x_in)
        mse = self.metric_mse(x_in + delta_hat, y)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_mse', mse, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Split parameters: weight decay harms 1-D params (norm γ/β, biases) and
        # learned position biases — regularising them toward zero kills positional
        # structure and makes LayerNorm ineffective.
        decay, no_decay = [], []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim <= 1 or name.endswith('.bias') or 'position_bias' in name:
                no_decay.append(p)
            else:
                decay.append(p)

        optimizer = torch.optim.AdamW(
            [{'params': decay,    'weight_decay': self.weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08,
        )
        warmup = int(getattr(self, '_warmup_steps', 1000))
        total_steps = self.trainer.estimated_stepping_batches
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup), eta_min=self.learning_rate * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.perf_counter()
