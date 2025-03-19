import torch
from torch import nn


class UpSample(nn.Module):
    """
    Up-sampling operation.
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
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
        Args:
            x (torch.Tensor): (B, N, C)
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
    Down-sampling operation
    Implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): [pressure levels, latitude, longitude]
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

    def forward(self, x):
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
    Args:
        x (torch.Tensor): B, C, Lat, Lon
        resolution (tuple[int]): Lat, Lon
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
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[:, :, padding_front: Pl - padding_back, padding_top: Lat - padding_bottom,
           padding_left: Lon - padding_right]
class Crop3D(nn.Module):
    def __init__(self, inputshape, resolution):
        super().__init__()
        self.inputshape = inputshape
        self.resolution = resolution
        self.pl_pad = inputshape[-3] - resolution[0]
        self.lat_pad = inputshape[-2] - resolution[1]
        self.lon_pad = inputshape[-1] - resolution[2]
        self.padding_front = self.pl_pad // 2
        self.padding_back = self.pl_pad - self.padding_front
        self.padding_top = self.lat_pad // 2
        self.padding_bottom = self.lat_pad - self.padding_top
        self.padding_left = self.lon_pad // 2
        self.padding_right = self.lon_pad - self.padding_left
    def forward(self, x: torch.Tensor):
        return x[:, :, self.padding_front: self.inputshape[-3] - self.padding_back,
               self.padding_top: self.inputshape[-2] - self.padding_bottom,
               self.padding_left: self.inputshape[-1] - self.padding_right]

def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    print(window_size)
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
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


class PatchEmbed2D(nn.Module):
    """
    2D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
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

        layerlist=[ nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom)),
                 nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
                    ]   
        if norm_layer is not None:
                layerlist.append(torch.nn.permute(0, 2, 3, 1))
                layerlist.append(norm_layer(embed_dim))
                layerlist.append(torch.nn.permute(0, 3, 1, 2))
        self.model=nn.Sequential(*layerlist)
    def forward(self, x: torch.Tensor):
       
        return self.model(x)


class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding.

    Args:
        img_size (tuple[int]): Image size.
        patch_size (tuple[int]): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim(int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
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
        layerlist=[self.pad, self.proj]

        if norm_layer is not None:
            layerlist.append(torch.nn.permute(0, 2, 3, 4, 1))
            layerlist.append(norm_layer(embed_dim))
            layerlist.append(torch.nn.permute(0, 4, 1, 2, 3))
        self.model=nn.Sequential(*layerlist)

    def forward(self, x: torch.Tensor):
        return self.model(x)

class PatchRecovery2D(nn.Module):
    """
    Patch Embedding Recovery to 2D Image.

    Args:
        img_size (tuple[int]): Lat, Lon
        patch_size (tuple[int]): Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)
        


    def forward(self, x):
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
    Patch Embedding Recovery to 3D Image.

    Args:
        img_size (tuple[int]): Pl, Lat, Lon
        patch_size (tuple[int]): Pl, Lat, Lon
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
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
    Args:
        x: (B, Pl, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon]

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
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
        window_size (tuple[int]): Window size [win_pl, win_lat, win_lon].
    """

    def __init__(self, input_shape, window_size):
        super().__init__()
        self.input_shape = input_shape
        self.window_size = window_size
        self.view_shape = (-1, (input_shape[1] // window_size[0]) * (input_shape[2] // window_size[1]),
                           window_size[0], window_size[1], window_size[2], input_shape[-1])
        self.xview_shape = (input_shape[0], input_shape[1] // window_size[0], window_size[0],
                            input_shape[2] // window_size[1], window_size[1], input_shape[3] // window_size[2],
                            window_size[2], input_shape[-1])
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, Pl, Lat, Lon, C)

        Returns:
            windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        """
        # B, Pl, Lat, Lon, C = self.input_shape
        # win_pl, win_lat, win_lon = self.window_size
        x = x.view(*self.xview_shape)
        windows = x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(*self.view_shape)
        return windows
    

def window_reverse(windows, window_size, Pl, Lat, Lon):
    """
    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon]

    Returns:
        x: (B, Pl, Lat, Lon, C)
    """
    win_pl, win_lat, win_lon = window_size
    B = int(windows.shape[0] / (Lon / win_lon))
    x = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return x
# -------------TO DO - Make this into a module with preset sizes
# class WindowReverse(nn.Module):

def get_shift_window_mask(input_resolution, window_size, shift_size):
    """
    Along the longitude dimension, the leftmost and rightmost indices are actually close to each other.
    If half windows apper at both leftmost and rightmost positions, they are dircetly merged into one window.
    Args:
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].

    Returns:
        attn_mask: (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon)
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
    This function construct the position index to reuse symmetrical parameters of the position bias.
    implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [pressure levels, latitude, longitude]

    Returns:
        position_index (torch.Tensor): [win_pl * win_lat * win_lon, win_pl * win_lat * win_lon]
    """
    win_pl, win_lat, win_lon = window_size
    # Index in the pressure level of query matrix
    coords_zi = torch.arange(win_pl)
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(win_pl) * win_pl

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat

    # Sum up the indexes in three dimensions
    position_index = coords.sum(-1)

    return position_index
