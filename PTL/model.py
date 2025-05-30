

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
from utils import get_shift_window_mask, WindowReverse, WindowPartition, PatchEmbed2D, PatchEmbed3D, PatchRecovery2D, get_pad3d, DownSample, UpSample, get_earth_position_index, Crop3D
import math


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
        qkv = self.qkv(x)
        qkv=qkv.unflatten(3,( 3, self.num_heads, C // self.num_heads))
        qkv=qkv.permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

          # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        attn = attn + self.earth_position_bias
        attn=self.forward_WMask(attn,B_, nW_, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = (attn @ v).permute(0, 2, 3, 1, 4)
        attn=attn.flatten(-2,-1)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


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
        if drop_prob == 0. or not self.training:
            self.forward = self.null_forward
        self.scale_by_keep = scale_by_keep
        scale_factor=1/(1-drop_prob)
        if 1-drop_prob<=0 or not scale_by_keep:
            scale_factor=1
        self.scale_factor=scale_factor

    def null_forward(self, x):
        """
        Forward pass without applying drop path.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return x

    def forward(self, x):
        """
        Forward pass with drop path applied.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with drop path applied.
        """
        rand_tensor=x.new_empty(x.shape[0],*([1] * (x.ndim - 1))).bernoulli_(1 - self.drop_prob) * self.scale_factor
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
        self.pad = nn.ZeroPad3d(padding)

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
        self.roll = shift_pl and shift_lon and shift_lat
        self.posrollX=self.null_rollX
        self.negrollX=self.null_rollX
        attn_mask=None
        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
            self.posrollX=self.posrollX
            self.negrollX=self.negrollX

        self.attn = EarthAttention3D(
            dim=dim, input_resolution=pad_resolution, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,attn_mask=attn_mask
        )
        
    def negrollX(self, x: torch.Tensor):
        shift_pl, shift_lat, shift_lon = self.shift_size
        return torch.roll(x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3))
    
    def null_rollX(self, x: torch.Tensor):
        return x
    
    def posrollX(self, x: torch.Tensor):
        shift_pl, shift_lat, shift_lon = self.shift_size
        return torch.roll(x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3))
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the transformer block.
        """
        shortcut = x
        x = self.norm1(x)
        x=x.unflatten(1, self.input_resolution)
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = self.negrollX(x)
        shifted_x=self.WindowReverse(self.attn(self.WindowPartition(x)))
        shifted_x = self.posrollX(shifted_x)
        shifted_x= self.Crop3d(shifted_x)
        shifted_x = shifted_x.flatten(1, 3)
        shifted_x = shortcut + self.drop_path(shifted_x)
        shifted_x = shifted_x + self.drop_path(self.mlp(self.norm2(shifted_x)))
        return shifted_x
    


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
        self.criterion= torch.nn.MSELoss(reduction='sum')
        self.L1Loss = torch.nn.L1Loss()
        self.grid_size=kwargs.get("grid_size",300)
        self.mlp_ratio=kwargs.get("mlp_ratio",4)
        self.noise_factor=kwargs.get("noise_factor",0.1)
        self.patchembed2d = PatchEmbed2D(
            img_size=(self.grid_size, self.grid_size),
            patch_size=(4, 4),
            in_chans=5,
            embed_dim=embed_dim,
        )
        reduced_grid=(1, self.grid_size//4,self.grid_size//4)
        self.incremental_step=0
        self.skip_steps=[]
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

        def save_skip(module, input, output):
            self.skip_steps.append(output)

        self.skiphook = self.layer1.register_forward_hook(save_skip)
        
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
        
    def forward(self, x):
        """
        Forward pass for the Pangu model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.layer1(x)
        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)
        return x

    
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch
        self.skip_steps=[]

        x = self.patchembed2d(x)
        x = x.flatten(2,3).transpose(1, 2)
        for t in range(self.time_steps):
            x=x+(torch.randn_like(x)*self.noise_factor)
            x = self(x)
            x=x-(torch.randn_like(x)*self.noise_factor)

        x=torch.concat([x, *self.skip_steps], dim=-1)
        x = x.transpose(1, 2).unflatten(2,self.reduced_grid)
        x = self.patchrecovery2d(x)
        loss = self.criterion(x, y)
        self.log('train_loss', loss, on_epoch=True,prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        self.skip_steps=[]

        x, y = batch
        x = self.patchembed2d(x)
        x = x.flatten(2,3).transpose(1, 2)
        for t in range(self.time_steps):
            x = self(x)
        x = torch.concat([x, *self.skip_steps], dim=-1)
        x = x.transpose(1, 2).unflatten(2,self.reduced_grid)

        y_hat = self.patchrecovery2d(x)
        loss = self.criterion(y_hat, y)
        if batch_idx % 100 == 0:
            self.plot_and_log_data(x, y, y_hat, batch_idx)
        self.log('val_loss', loss, on_epoch=True,prog_bar=True)
        return loss

    def plot_and_log_data(self, x, y, y_hat, batch_idx):
        Batch_size= x.shape[0]
        x=x.cpu().detach().numpy()
        y=y.cpu().detach().numpy()
        y_hat=y_hat.cpu().detach().numpy()
        os.makedirs(f"results/{batch_idx}", exist_ok=True)
        for item in range(Batch_size):
            ax, fig = plt.subplots(3, 5)
            for i in range(5):
                fig[0,i].imshow(x[item,i,:,:])
                fig[1,i].imshow(y[item,i,:,:])
                fig[2,i].imshow(y_hat[item,i,:,:])
            plt.savefig(f"results/{batch_idx}/{item}.png")
            plt.close()
        self.logger.log_image("examples",[f"results/{batch_idx}/{item}.png" for item in range(Batch_size)])     

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (tuple): Input batch containing features and labels.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Test loss.
        """
        self.skip_steps=[]

        x, y = batch
        for t in range(self.time_steps):
            x = self(x)
        y_hat = torch.concat([x, *self.skip_steps], dim=-1) 
        y_hat = y_hat.transpose(1, 2).unflatten(2,self.reduced_grid)
        y_hat = self.patchrecovery2d(y_hat)
        loss=F.cross_entropy_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        return optimizer
