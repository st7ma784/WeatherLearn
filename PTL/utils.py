import torch
from torch import nn
"""
Utils Module
============

This module provides utility classes and functions for tensor operations and model components used in weather 
prediction tasks. It includes operations for up-sampling, down-sampling, cropping, patch embedding, patch recovery, 
and window partitioning, as well as helper functions for attention mechanisms and earth-specific computations.

Key Features
------------

- **Up-Sampling and Down-Sampling**:
  - `UpSample`: Performs up-sampling operations on tensors.
  - `DownSample`: Performs down-sampling operations on tensors.

- **Cropping**:
  - `crop2d`: Crops a 2D tensor to a specified resolution.
  - `crop3d`: Crops a 3D tensor to a specified resolution.
  - `Crop3D`: A module for cropping 3D tensors based on padding values.

- **Patch Embedding and Recovery**:
  - `PatchEmbed2D`: Converts 2D images into patch embeddings.
  - `PatchEmbed3D`: Converts 3D images into patch embeddings.
  - `PatchRecovery2D`: Recovers 2D images from patch embeddings.
  - `PatchRecovery3D`: Recovers 3D images from patch embeddings.

- **Window Partitioning and Reversal**:
  - `window_partition`: Partitions tensors into windows.
  - `window_reverse`: Reverses the window partitioning operation.
  - `WindowPartition`: A module for partitioning tensors into windows.
  - `WindowReverse`: A module for reversing window partitioning.

- **Attention Mechanisms**:
  - `get_shift_window_mask`: Generates a shift window mask for shifted window multi-head self-attention (SW-MSA).
  - `get_earth_position_index`: Constructs position indices for earth-specific attention.

Dependencies
------------

- **PyTorch**:
  Provides the core framework for tensor operations and neural network components.

- **Torch.nn**:
  Used for defining layers and modules, such as `Linear`, `LayerNorm`, and `Conv2d`.

"""

class UpSample(nn.Module):
    """
    Up-sampling operation.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude].
        output_resolution (tuple[int]): [pressure levels, latitude, longitude].
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Forward pass for up-sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Up-sampled tensor.
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[:, :out_pl, pad_top: 2 * in_lat - pad_bottom, pad_left: 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample(nn.Module):
    """
    Down-sampling operation.

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude].
        output_resolution (tuple[int]): [pressure levels, latitude, longitude].
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        assert in_pl == out_pl, "the dimension of pressure level shouldn't change"
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = torch.nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for down-sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Down-sampled tensor.
        """
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


def crop2d(x: torch.Tensor, resolution):
    """
    Crops a 2D tensor to the specified resolution.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, Lat, Lon).
        resolution (tuple[int]): Target resolution (Lat, Lon).

    Returns:
        torch.Tensor: Cropped tensor.
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[:, :, padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]


def crop3d(x: torch.Tensor, resolution):
    """
    Crops a 3D tensor to the specified resolution.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, Pl, Lat, Lon).
        resolution (tuple[int]): Target resolution (Pl, Lat, Lon).

    Returns:
        torch.Tensor: Cropped tensor.
    """
    _, Pl, Lat, Lon, _ = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[:, padding_front: Pl - padding_back, padding_top: Lat - padding_bottom,
           padding_left: Lon - padding_right, :]


class Crop3D(nn.Module):
    """
    Crops a 3D tensor based on specified padding.

    Args:
        padding (tuple[int]): Padding values (left, right, top, bottom, front, back).
    """

    def __init__(self, padding):
        super().__init__()
        self.padding_front, self.padding_back = padding[-1], padding[-2]
        self.padding_top, self.padding_bottom = padding[2], padding[3]
        self.padding_left, self.padding_right = padding[0], padding[1]

    def forward(self, x: torch.Tensor):
        """
        Forward pass for cropping a 3D tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Cropped tensor.
        """
        return x[:,
                self.padding_front:,
                self.padding_top: -self.padding_bottom,
                self.padding_left: -self.padding_right,
                :]


def get_pad3d(input_resolution, window_size):
    """
    Calculates padding for a 3D tensor to align with the window size.

    Args:
        input_resolution (tuple[int]): Input resolution (Pl, Lat, Lon).
        window_size (tuple[int]): Window size (Pl, Lat, Lon).

    Returns:
        tuple[int]: Padding values (left, right, top, bottom, front, back).
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Calculates padding for a 2D tensor to align with the window size.

    Args:
        input_resolution (tuple[int]): Input resolution (Lat, Lon).
        window_size (tuple[int]): Window size (Lat, Lon).

    Returns:
        tuple[int]: Padding values (left, right, top, bottom).
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


class PatchEmbed2D(nn.Module):
    """
    Converts a 2D image into patch embeddings.

    Args:
        img_size (tuple[int]): Image size (Lat, Lon).
        patch_size (tuple[int]): Patch size (Lat, Lon).
        in_chans (int): Number of input channels.
        embed_dim (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0
        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)
        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        layerlist = [nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom)),
                     nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                     ]
        if norm_layer is not None:
            layerlist.append(torch.nn.permute(0, 2, 3, 1))
            layerlist.append(norm_layer(embed_dim))
            layerlist.append(torch.nn.permute(0, 3, 1, 2))
        self.model = nn.Sequential(*layerlist)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for patch embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Patch-embedded tensor.
        """
        return self.model(x)


class PatchEmbed3D(nn.Module):
    """
    Converts a 3D image into patch embeddings.

    Args:
        img_size (tuple[int]): Image size (Pl, Lat, Lon).
        patch_size (tuple[int]): Patch size (Pl, Lat, Lon).
        in_chans (int): Number of input channels.
        embed_dim (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % l_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = torch.nn.ZeroPad3d(
            (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        )
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        layerlist = [self.pad, self.proj]

        if norm_layer is not None:
            layerlist.append(torch.nn.permute(0, 2, 3, 4, 1))
            layerlist.append(norm_layer(embed_dim))
            layerlist.append(torch.nn.permute(0, 4, 1, 2, 3))
        self.model = nn.Sequential(*layerlist)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for patch embedding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Patch-embedded tensor.
        """
        return self.model(x)


class PatchRecovery2D(nn.Module):
    """
    Recovers a 2D image from patch embeddings.

    Args:
        img_size (tuple[int]): Image size (Lat, Lon).
        patch_size (tuple[int]): Patch size (Lat, Lon).
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for patch recovery.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Recovered image tensor.
        """
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[:, :, padding_top: H - padding_bottom, padding_left: W - padding_right]


class PatchRecovery3D(nn.Module):
    """
    Recovers a 3D image from patch embeddings.

    Args:
        img_size (tuple[int]): Image size (Pl, Lat, Lon).
        patch_size (tuple[int]): Patch size (Pl, Lat, Lon).
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for patch recovery.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Recovered image tensor.
        """
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[:, :, padding_front: Pl - padding_back,
               padding_top: Lat - padding_bottom, padding_left: Lon - padding_right]


def window_partition(x: torch.Tensor, window_size):
    """
    Partitions a tensor into windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, Pl, Lat, Lon, C).
        window_size (tuple[int]): Window size (Pl, Lat, Lon).

    Returns:
        torch.Tensor: Partitioned windows.
    """
    B, Pl, Lat, Lon, C = x.shape
    win_pl, win_lat, win_lon = window_size
    x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(
        -1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C
    )
    return windows


class WindowPartition(nn.Module):
    """
    Torch module for partitioning a tensor into windows.

    Args:
        input_shape (tuple[int]): Shape of the input tensor (B, Pl, Lat, Lon, C).
        window_size (tuple[int]): Window size (Pl, Lat, Lon).
    """

    def __init__(self, input_shape, window_size):
        super().__init__()
        self.input_shape = input_shape
        self.window_size = window_size
        self.view_shape = (-1, (input_shape[1] // window_size[0]) * (input_shape[2] // window_size[1]),
                           window_size[0] * window_size[1] * window_size[2], input_shape[-1])
        self.xview_shape = (-1, input_shape[1] // window_size[0], window_size[0],
                            input_shape[2] // window_size[1], window_size[1], input_shape[3] // window_size[2],
                            window_size[2], input_shape[-1])

    def forward(self, x: torch.Tensor):
        """
        Forward pass for window partitioning.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Partitioned windows.
        """
        x = x.view(*self.xview_shape)
        windows = x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(*self.view_shape)
        return windows


def window_reverse(windows, window_size, Pl, Lat, Lon, dim):
    """
    Reverses the window partitioning operation.

    Args:
        windows (torch.Tensor): Partitioned windows.
        window_size (tuple[int]): Window size (Pl, Lat, Lon).
        Pl (int): Pressure levels.
        Lat (int): Latitude.
        Lon (int): Longitude.
        dim (int): Number of channels.

    Returns:
        torch.Tensor: Reconstructed tensor.
    """
    win_pl, win_lat, win_lon = window_size
    x = windows.view(-1, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, dim)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(-1, Pl, Lat, Lon, dim)
    return x


class WindowReverse(nn.Module):
    """
    Torch module for reversing window partitioning.

    Args:
        window_size (tuple[int]): Window size (Pl, Lat, Lon).
        Pl (int): Pressure levels.
        Lat (int): Latitude.
        Lon (int): Longitude.
        dim (int): Number of channels.
    """

    def __init__(self, window_size, Pl, Lat, Lon, dim):
        super().__init__()
        self.window_size = window_size
        self.Pl = Pl
        self.Lat = Lat
        self.Lon = Lon
        self.windows_view_shape = (-1, Lon // window_size[2], Pl // window_size[0], Lat // window_size[1],
                                   window_size[0], window_size[1], window_size[2], dim)
        self.xview_shape = (-1, self.Pl, self.Lat, self.Lon, dim)

    def forward(self, windows: torch.Tensor):
        """
        Forward pass for reversing window partitioning.

        Args:
            windows (torch.Tensor): Partitioned windows.

        Returns:
            torch.Tensor: Reconstructed tensor.
        """
        x = windows.unflatten(2, self.window_size).view(*self.windows_view_shape)
        x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(*self.xview_shape)
        return x


def get_shift_window_mask(input_resolution, window_size, shift_size):
    """
    Generates a shift window mask for SW-MSA.

    Args:
        input_resolution (tuple[int]): Input resolution (Pl, Lat, Lon).
        window_size (tuple[int]): Window size (Pl, Lat, Lon).
        shift_size (tuple[int]): Shift size (Pl, Lat, Lon).

    Returns:
        torch.Tensor: Attention mask.
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size
    shift_pl, shift_lat, shift_lon = shift_size

    img_mask = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))

    pl_slices = (slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None))
    lat_slices = (slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None))
    lon_slices = (slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None))

    cnt = 0
    for pl in pl_slices:
        for lat in lat_slices:
            for lon in lon_slices:
                img_mask[:, pl, lat, lon, :] = cnt
                cnt += 1

    img_mask = img_mask[:, :, :, :Lon, :]

    mask_windows = window_partition(img_mask, window_size)  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1
    mask_windows = mask_windows.view(mask_windows.shape[0], mask_windows.shape[1], win_pl * win_lat * win_lon)
    attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def get_earth_position_index(window_size):
    """
    Constructs the position index for earth-specific attention.

    Args:
        window_size (tuple[int]): Window size (Pl, Lat, Lon).

    Returns:
        torch.Tensor: Position index tensor.
    """
    win_pl, win_lat, win_lon = window_size
    coords_zi = torch.arange(win_pl)
    coords_zj = -torch.arange(win_pl) * win_pl

    coords_hi = torch.arange(win_lat)
    coords_hj = -torch.arange(win_lat) * win_lat

    coords_w = torch.arange(win_lon)

    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat

    position_index = coords.sum(-1)

    return position_index
