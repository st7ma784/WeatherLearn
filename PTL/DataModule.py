"""
DataModule Module
=================

This module defines data handling classes and functions for processing SuperDARN radar data. It provides tools 
for loading, preprocessing, and managing datasets from MinIO buckets or disk storage. The module supports both 
flat and grid-based data representations and includes functionality for caching datasets and saving/loading 
preprocessed data.

Key Features
------------

- **Dataset Management**:
  - `SuperDARNDatasetFolder`: Handles datasets stored on disk.
  - `SuperDARNDataset`: Handles datasets stored in MinIO buckets.
  - `DatasetFromMinioBucket`: A PyTorch Lightning DataModule for managing data loading and preprocessing.
  - `DatasetFromPresaved`: Handles preprocessed datasets saved on disk.

- **Preprocessing and Caching**:
  - Supports preprocessing of radar data into flat or grid-based representations.
  - Provides functionality for saving datasets to disk and loading them for reuse.

- **Utility Functions**:
  - `gaussian`: Computes a 2D Gaussian function.
  - `save_dataset_to_disk`: Saves a dataset to disk in minibatches.
  - `load_dataset_from_disk`: Loads a dataset from disk.

Dependencies
------------

- **PyTorch Lightning**:
  Provides the `LightningDataModule` for managing data loading and preprocessing.

- **MinIO**:
  Used for interacting with MinIO buckets to load radar data.

- **pydarnio**:
  A library for reading and processing SuperDARN radar data.

- **NumPy**:
  Used for numerical operations and data manipulation.

- **Torch**:
  Provides utilities for creating datasets and DataLoaders.

- **TQDM**:
  Used for progress tracking during data processing.

"""


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch
from torch.utils.data import IterableDataset, Dataset
import pydarnio
import numpy as np
import os
try:
    from minio import Minio
except ImportError:
    Minio = None  # optional; only needed for MinIO-backed datasets
import random
from tqdm import tqdm
import hashlib
import ast

def gaussian(x, y, x0, y0, sigma_x, sigma_y):
    """
    Computes a 2D Gaussian function.

    Args:
        x (float): X-coordinate.
        y (float): Y-coordinate.
        x0 (float): X-coordinate of the Gaussian center.
        y0 (float): Y-coordinate of the Gaussian center.
        sigma_x (float): Standard deviation in the X direction.
        sigma_y (float): Standard deviation in the Y direction.

    Returns:
        float: The value of the Gaussian function at (x, y).
    """
    return np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))


class SuperDARNDatasetFolder(Dataset):
    """
    Handles datasets stored on disk.

    This class provides functionality to load and process SuperDARN radar data stored on disk.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Attributes:
        dataset (SuperDARNDataset): Dataset object.
        data (list): List of data files.
        data_dir (str): Directory containing the data files.
        batch_size (int): Batch size for data loading.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the dataset folder.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__()
        self.dataset = SuperDARNDataset.from_disk(*args, **kwargs)
        self.data = [file for file in self.dataset.generator()]
        self.data_dir = args[0]
        self.batch_size = kwargs.get("batch_size", 1)

    def __len__(self):
        """
        Returns the number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves the data at the specified index.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            Any: The data at the specified index.
        """
        filename = self.data[index]
        try:
            output = self.dataset.process_file(open(os.path.join(self.data_dir, filename), 'rb'))
        except Exception as e:
            print("Error: ", e)
            output = None
        if output is None or output[0] is None or output[1] is None:
            output = self.__getitem__(random.randint(0, len(self) - 1))
        return output


class SuperDARNDataset(IterableDataset):
    """
    Handles datasets stored in MinIO buckets.

    This class provides functionality to load and process SuperDARN radar data stored in MinIO buckets.

    Args:
        miniodict (dict): Dictionary containing MinIO connection details.
        minioBucket (str): Name of the MinIO bucket.
        batch_size (int): Batch size for data loading.
        method (str): Data representation method ('flat' or 'grid').
        window_size (int): Size of the time window for data processing.
        **kwargs: Additional keyword arguments.

    Attributes:
        minioClient (Minio): MinIO client object.
        minioBucket (str): Name of the MinIO bucket.
        batch_size (int): Batch size for data loading.
        method (str): Data representation method ('flat' or 'grid').
        window_size (int): Size of the time window for data processing.
    """

    @classmethod
    def from_disk(cls, data_dir, batch_size, method='flat', window_size=10, **kwargs):
        """
        Loads the dataset from disk.

        Args:
            data_dir (str): Directory containing the data files.
            batch_size (int): Batch size for data loading.
            method (str): Data representation method ('flat' or 'grid').
            window_size (int): Size of the time window for data processing.
            **kwargs: Additional keyword arguments.

        Returns:
            SuperDARNDataset: An instance of the dataset.
        """
        self = cls.__new__(cls, {}, "", batch_size, method, window_size, silent=True, **kwargs)

        def generator():
            import os
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(".bin") or file.endswith(".npy") or file.endswith(".txt"):
                        continue
                    yield file

        self.generator = generator

        def __iter__():
            for file in self.generator():
                output = self.process_file(open(os.path.join(data_dir, file), 'rb'))
                if output is None or output[0] is None or output[1] is None:
                    continue
                yield output

        self.__iter__ = __iter__
        return self

    def __new__(cls, miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        """
        Creates a new instance of the dataset.

        Args:
            miniodict (dict): Dictionary containing MinIO connection details.
            minioBucket (str): Name of the MinIO bucket.
            batch_size (int): Batch size for data loading.
            method (str): Data representation method ('flat' or 'grid').
            window_size (int): Size of the time window for data processing.
            **kwargs: Additional keyword arguments.

        Returns:
            SuperDARNDataset: A new instance of the dataset.
        """
        self = super().__new__(cls)
        self.__init__(miniodict, minioBucket, batch_size, method, window_size, **kwargs)
        return self

    def __init__(self, miniodict, minioBucket, batch_size, method='flat', window_size=10, **kwargs):
        """
        Initializes the dataset.

        Args:
            miniodict (dict): Dictionary containing MinIO connection details.
            minioBucket (str): Name of the MinIO bucket.
            batch_size (int): Batch size for data loading.
            method (str): Data representation method ('flat' or 'grid').
            window_size (int): Size of the time window for data processing.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.minioBucket = minioBucket
        self.batch_size = batch_size
        self.grid_size = kwargs.get("grid_size", 300)
        self.time_step = kwargs.get("time_step", 1)
        self.device = "cpu"
        self.max_retries = kwargs.get("max_retries", 3)
        try:
            if Minio is None:
                raise ImportError("minio package not installed")
            self.minioClient = Minio(miniodict.get("host", "localhost") + ":" + str(miniodict.get("port", 9000)),
                                     access_key=miniodict.get("access_key", "minioadmin"),
                                     secret_key=miniodict.get("secret_key", "minioadmin"),
                                     secure=False)
            self.miniodict = miniodict
        except Exception as e:
            if not kwargs.get("silent", False):
                raise e

        self.window_size = window_size
        self.method = method
        if self.method == 'flat':
            self.process_conv_to_tensor = self.process_conv_to_flat
            self.process_fitacf_to_tensor = self.process_fitacf_to_flat
        elif self.method == 'grid':
            self.process_conv_to_tensor = self.process_conv_to_grid
            self.process_fitacf_to_tensor = self.process_fitacf_to_grid
        else:
            raise ValueError("method must be either 'flat' or 'grid'")
        self.location = {"max_vector": 88008, "max_mlat": 0,
                         "max_mlon": 0,
                         "min_mlat": 360,
                         "min_mlon": 360}
        self.first_epoch = True

    def _safe_float(self, value):
        try:
            return float(value)
        except Exception:
            return 0.0

    def _records_to_grid(self, records):
        if not records:
            return np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)

        vel_sum = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        vel_cnt = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        pwr_sum = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        wd_sum = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for record in records:
            mlats = record.get("vector.mlat", []) or []
            mlons = record.get("vector.mlon", []) or []
            vels = record.get("vector.vel.median", []) or []
            pwrs = record.get("vector.pwr.median", []) or []
            wids = record.get("vector.wdt.median", []) or []
            n = min(len(mlats), len(mlons), len(vels))
            if n == 0:
                continue
            for i in range(n):
                mlat = self._safe_float(mlats[i])
                mlon = self._safe_float(mlons[i])
                if not np.isfinite(mlat) or not np.isfinite(mlon):
                    continue
                lat_idx = int(np.clip((mlat + 90.0) / 180.0 * (self.grid_size - 1), 0, self.grid_size - 1))
                lon_idx = int(np.clip((mlon % 360.0) / 360.0 * (self.grid_size - 1), 0, self.grid_size - 1))
                vel_sum[lat_idx, lon_idx] += self._safe_float(vels[i])
                vel_cnt[lat_idx, lon_idx] += 1.0
                if i < len(pwrs):
                    pwr_sum[lat_idx, lon_idx] += self._safe_float(pwrs[i])
                if i < len(wids):
                    wd_sum[lat_idx, lon_idx] += self._safe_float(wids[i])

        mean_vel = np.divide(vel_sum, np.maximum(vel_cnt, 1.0), dtype=np.float32)
        mean_pwr = np.divide(pwr_sum, np.maximum(vel_cnt, 1.0), dtype=np.float32)
        mean_wdt = np.divide(wd_sum, np.maximum(vel_cnt, 1.0), dtype=np.float32)
        occupancy = (vel_cnt > 0).astype(np.float32)
        density = np.log1p(vel_cnt).astype(np.float32)

        return np.stack([mean_vel, mean_pwr, mean_wdt, occupancy, density], axis=0).astype(np.float32)

    def process_conv_to_grid(self, records):
        return self._records_to_grid(records)

    def process_fitacf_to_grid(self, records):
        return self._records_to_grid(records)

    def process_conv_to_flat(self, records):
        return self.process_conv_to_grid(records).reshape(5, -1)

    def process_fitacf_to_flat(self, records):
        return self.process_fitacf_to_grid(records).reshape(5, -1)

    def process_data_conv(self, records):
        if not records:
            return None
        if len(records) < 2:
            grid = self.process_conv_to_grid(records)
            tensor = torch.from_numpy(grid)
            return tensor, tensor.clone()
        split_idx = max(1, len(records) // 2)
        x_grid = self.process_conv_to_grid(records[:split_idx])
        y_grid = self.process_conv_to_grid(records[split_idx:])
        if x_grid is None or y_grid is None:
            return None
        return torch.from_numpy(x_grid), torch.from_numpy(y_grid)

    def generator(self):
        """
        Returns an iterator over the objects in the MinIO bucket.

        Yields:
            Any: The next object in the bucket.
        """
        for obj in self.minioClient.list_objects(self.minioBucket):
            yield obj

    def process_file(self, data1):
        """
        Processes a file and converts it into a usable format.

        Args:
            data1 (file-like object): The file to process.

        Returns:
            tuple: Processed data.
        """
        file_stream = data1.read()
        reader = pydarnio.SDarnRead(file_stream, True)
        output = self.process_data_conv(reader.read_map())
        if output is None or output[0] is None or output[1] is None:
            return None
        else:
            return output

    def __iter__(self):
        """
        Returns an iterator over the dataset.

        Yields:
            tuple: Processed data.
        """
        for obj in self.generator():
            for _ in range(self.max_retries):
                try:
                    data = self.minioClient.get_object(self.minioBucket, obj.object_name)
                    data1 = self.process_file(data)
                    if data1 is None:
                        break
                    yield data1
                    break
                except Exception as e:
                    print("Error processing object {}: {}".format(obj.object_name, e))

    def __getitem__(self, index):
        """
        Retrieves the data at the specified index.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            tuple: The data at the specified index.
        """
        return next(iter(self))


class DatasetFromMinioBucket(LightningDataModule):
    """
    A PyTorch Lightning DataModule for managing data loading and preprocessing.

    This class provides functionality to load, preprocess, and manage SuperDARN radar data from MinIO buckets.

    Args:
        minioClient (dict): Dictionary containing MinIO connection details.
        bucket_name (str): Name of the MinIO bucket.
        data_dir (str): Directory to store cached data.
        batch_size (int): Batch size for data loading.
        method (str): Data representation method ('flat' or 'grid').
        windowMinutes (int): Size of the time window for data processing.
        **kwargs: Additional keyword arguments.

    Attributes:
        minioClient (dict): Dictionary containing MinIO connection details.
        bucket_name (str): Name of the MinIO bucket.
        data_dir (str): Directory to store cached data.
        batch_size (int): Batch size for data loading.
        method (str): Data representation method ('flat' or 'grid').
        window_size (int): Size of the time window for data processing.
    """

    def __init__(self, minioClient, bucket_name, data_dir, batch_size, method='grid', WindowsMinutes=20, **kwargs):
        super().__init__()
        self.minioClient = minioClient
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.method = method
        self.window_size = kwargs.get("window_size", WindowsMinutes)
        self.preProcess = kwargs.get("preProcess", False)
        self.cache_to_disk = kwargs.get("cache_to_disk", kwargs.get("cache_first", False))
        self.HPC = kwargs.get("HPC", False)
        self.cache_stats = kwargs.get("cache_stats", True)
        self.kwargs = kwargs

        hash_payload = {
            "bucket_name": bucket_name,
            "method": method,
            "window_size": self.window_size,
            "grid_size": kwargs.get("grid_size", 300),
            "time_step": kwargs.get("time_step", 1),
        }
        self.dataset_hash = hashlib.md5(str(sorted(hash_payload.items())).encode("utf-8")).hexdigest()[:12]

        self.train_dataset = None
        self.val_dataset = None
        self.dataset = None
        self.stats_path = os.path.join(self.data_dir, "data", str(self.dataset_hash), "train_stats.npz")

    def _dataset_for_stage(self):
        return self.train_dataset if self.train_dataset is not None else self.dataset

    def _subset_indices(self, subset):
        if isinstance(subset, torch.utils.data.Subset):
            return np.asarray(subset.indices, dtype=np.int64)
        return np.arange(len(subset), dtype=np.int64)

    def _load_or_compute_train_stats(self):
        if not isinstance(self.dataset, DatasetFromPresaved):
            return
        if not self.cache_stats:
            return

        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        if os.path.exists(self.stats_path):
            loaded = np.load(self.stats_path)
            stats = {
                "x_mean": loaded["x_mean"],
                "x_std": loaded["x_std"],
                "y_mean": loaded["y_mean"],
                "y_std": loaded["y_std"],
            }
        else:
            train_indices = self._subset_indices(self.train_dataset)
            stats = self.dataset.compute_stats_from_indices(train_indices)
            np.savez(
                self.stats_path,
                x_mean=stats["x_mean"],
                x_std=stats["x_std"],
                y_mean=stats["y_mean"],
                y_std=stats["y_std"],
            )
        self.dataset.set_normalization_stats(stats)

    def prepare_data(self):
        """
        Prepares the data for training by downloading or preprocessing it.

        Returns:
            None
        """
        if self.cache_to_disk and not self.HPC:
            os.makedirs(self.data_dir, exist_ok=True)
            from MinioToDisk import download_minio_bucket_to_folder
            download_minio_bucket_to_folder(self.minioClient, self.bucket_name, self.data_dir)

        if self.preProcess:
            self.cache_to_disk = True
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

            if not os.path.exists(os.path.join(self.data_dir, "data", str(self.dataset_hash))):
                os.makedirs(os.path.join(self.data_dir, "data", str(self.dataset_hash)))
                dataset = SuperDARNDatasetFolder(self.data_dir, self.batch_size, self.method, self.window_size, **self.kwargs)
                Data = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count())
                save_dataset_to_disk(Data, os.path.join(self.data_dir, "data", str(self.dataset_hash)))

    def setup(self, stage=None):
        """
        Sets up the dataset for training, validation, and testing.

        Args:
            stage (str, optional): The stage of the setup process. Defaults to None.

        Returns:
            None
        """
        if not self.cache_to_disk:
            self.dataset = SuperDARNDataset(self.minioClient, self.bucket_name, self.batch_size, self.method, self.window_size, **self.kwargs)
        else:
            if self.preProcess:
                self.dataset = DatasetFromPresaved(*load_dataset_from_disk(os.path.join(self.data_dir, "data", str(self.dataset_hash))))
            else:
                self.dataset = SuperDARNDatasetFolder(self.data_dir, self.batch_size, self.method, self.window_size, **self.kwargs)

        if isinstance(self.dataset, IterableDataset):
            self.train_dataset = self.dataset
            self.val_dataset = self.dataset
            return

        dataset_len = len(self.dataset)
        train_len = int(dataset_len * 0.8)
        val_len = max(1, dataset_len - train_len)
        train_len = dataset_len - val_len
        indices = list(range(dataset_len))
        self.train_dataset = torch.utils.data.Subset(self.dataset, indices[:train_len])
        self.val_dataset = torch.utils.data.Subset(self.dataset, indices[train_len:])

        # Fit normalization on training split only to avoid leakage into validation/test.
        self._load_or_compute_train_stats()

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: The training DataLoader.
        """
        num_workers = min(12, os.cpu_count() or 1)
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 3

        if isinstance(self.train_dataset, IterableDataset):
            return DataLoader(self.train_dataset, shuffle=False, **kwargs)
        return DataLoader(self.train_dataset, shuffle=True, **kwargs)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: The validation DataLoader.
        """
        num_workers = min(8, os.cpu_count() or 1)
        return DataLoader(
            self._dataset_for_stage() if self.val_dataset is None else self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: The test DataLoader.
        """
        num_workers = min(4, os.cpu_count() or 1)
        return DataLoader(
            self._dataset_for_stage() if self.val_dataset is None else self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )


def save_dataset_to_disk(DataLoader, path):
    """
    Saves a dataset to disk in minibatches.

    Args:
        DataLoader (torch.utils.data.DataLoader): DataLoader object containing the dataset.
        path (str): Path to save the dataset.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    tensorA = torch.tensor([])
    tensorB = torch.tensor([])
    tensorshape = None
    idx = -1
    for idx, i in enumerate(tqdm(DataLoader)):
        dataA, dataB = i
        if dataA is None or dataB is None:
            continue
        if tensorA.shape[0] == 0:
            tensorA = dataA
            tensorB = dataB
        else:
            tensorA = torch.cat((tensorA, dataA), dim=0)
            tensorB = torch.cat((tensorB, dataB), dim=0)
        if tensorA.shape[0] >= 200:
            np.save(os.path.join(path, "dataA_{}.npy".format(idx)), tensorA[:200].numpy())
            np.save(os.path.join(path, "dataB_{}.npy".format(idx)), tensorB[:200].numpy())
            tensorA = tensorA[200:]
            tensorB = tensorB[200:]
            tensorshape = tensorA.shape

    if tensorA.shape[0] > 0:
        final_idx = max(idx, 0)
        np.save(os.path.join(path, "dataA_{}.npy".format(final_idx)), tensorA.numpy())
        np.save(os.path.join(path, "dataB_{}.npy".format(final_idx)), tensorB.numpy())
        tensorshape = list([tensorA.shape[i] for i in range(len(tensorA.shape))])
    if tensorshape is None:
        return
    tensorshape[0] = -1
    with open(os.path.join(path, "shape.txt"), 'w') as f:
        f.write(str(tensorshape))


def load_dataset_from_disk(path):
    """
    Loads a dataset from disk.

    Args:
        path (str): Path to the dataset directory.

    Returns:
        tuple: A tuple containing dataA, dataB, and the dataset shape.
    """
    dataA = []
    dataB = []
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if "dataA" in file:
                dataA.append(os.path.join(root, file))
            elif "dataB" in file:
                dataB.append(os.path.join(root, file))
    dataA.sort()
    dataB.sort()
    shape = None
    try:
        with open(os.path.join(path, "shape.txt"), 'r') as f:
            shape = ast.literal_eval(f.read())
            shape[0] = -1
    except Exception as e:
        print("Error: ", e)
        shape = None
    return dataA, dataB, shape


class DatasetFromPresaved(Dataset):
    """
    Handles preprocessed datasets saved on disk.

    This class provides functionality to load and process preprocessed SuperDARN radar data saved on disk.

    Args:
        dataA (list): List of file paths for dataA.
        dataB (list): List of file paths for dataB.
        shape (list): Shape of the dataset.

    Attributes:
        dataA (list): List of file paths for dataA.
        dataB (list): List of file paths for dataB.
        shape (list): Shape of the dataset.
        len (int): Length of the dataset.
    """

    def __init__(self, dataA, dataB, shape):
        """
        Initializes the dataset.

        Args:
            dataA (list): List of file paths for dataA.
            dataB (list): List of file paths for dataB.
            shape (list): Shape of the dataset.
        """
        self.dataA = dataA
        self.dataB = dataB
        if shape is not None:
            self.shape = shape[-3:]
        else:
            self.shape = None
        self.chunk_sizes = []
        for file in self.dataA:
            try:
                self.chunk_sizes.append(int(np.load(file, mmap_mode='r').shape[0]))
            except Exception:
                self.chunk_sizes.append(0)
        self.cumulative_sizes = np.cumsum(self.chunk_sizes) if self.chunk_sizes else np.array([], dtype=np.int64)
        self.len = int(self.cumulative_sizes[-1]) if len(self.cumulative_sizes) > 0 else 0
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def set_normalization_stats(self, stats):
        self.x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32).view(-1, 1, 1)
        self.x_std = torch.tensor(stats["x_std"], dtype=torch.float32).view(-1, 1, 1)
        self.y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32).view(-1, 1, 1)
        self.y_std = torch.tensor(stats["y_std"], dtype=torch.float32).view(-1, 1, 1)

    def compute_stats_from_indices(self, indices):
        if len(indices) == 0:
            channels = int(self.shape[0]) if self.shape is not None else 5
            zeros = np.zeros((channels,), dtype=np.float32)
            ones = np.ones((channels,), dtype=np.float32)
            return {"x_mean": zeros, "x_std": ones, "y_mean": zeros, "y_std": ones}

        indices = np.asarray(indices, dtype=np.int64)
        indices.sort()
        file_ids = np.searchsorted(self.cumulative_sizes, indices, side='right')

        x_sum = x_sq_sum = y_sum = y_sq_sum = None
        count = 0
        eps = 1e-6

        for file_id in np.unique(file_ids):
            local_mask = file_ids == file_id
            idxs = indices[local_mask]
            prev_total = int(self.cumulative_sizes[file_id - 1]) if file_id > 0 else 0
            local_offsets = idxs - prev_total

            dA = np.load(self.dataA[int(file_id)], mmap_mode='r')
            dB = np.load(self.dataB[int(file_id)], mmap_mode='r')
            batchA = np.asarray(dA[local_offsets], dtype=np.float64)
            batchB = np.asarray(dB[local_offsets], dtype=np.float64)

            if batchA.ndim != 4 or batchB.ndim != 4:
                continue

            if x_sum is None:
                x_sum = batchA.sum(axis=(0, 2, 3))
                x_sq_sum = np.square(batchA).sum(axis=(0, 2, 3))
                y_sum = batchB.sum(axis=(0, 2, 3))
                y_sq_sum = np.square(batchB).sum(axis=(0, 2, 3))
            else:
                x_sum += batchA.sum(axis=(0, 2, 3))
                x_sq_sum += np.square(batchA).sum(axis=(0, 2, 3))
                y_sum += batchB.sum(axis=(0, 2, 3))
                y_sq_sum += np.square(batchB).sum(axis=(0, 2, 3))
            count += int(batchA.shape[0] * batchA.shape[2] * batchA.shape[3])

        if count == 0:
            channels = int(self.shape[0]) if self.shape is not None else 5
            zeros = np.zeros((channels,), dtype=np.float32)
            ones = np.ones((channels,), dtype=np.float32)
            return {"x_mean": zeros, "x_std": ones, "y_mean": zeros, "y_std": ones}

        x_mean = (x_sum / count).astype(np.float32)
        y_mean = (y_sum / count).astype(np.float32)
        x_var = np.maximum((x_sq_sum / count) - np.square(x_mean, dtype=np.float32), eps)
        y_var = np.maximum((y_sq_sum / count) - np.square(y_mean, dtype=np.float32), eps)

        return {
            "x_mean": x_mean,
            "x_std": np.sqrt(x_var, dtype=np.float32),
            "y_mean": y_mean,
            "y_std": np.sqrt(y_var, dtype=np.float32),
        }

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.len

    def __getitem__(self, index):
        """
        Retrieves the data at the specified index.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            tuple: The data at the specified index.
        """
        if self.len <= 0:
            raise IndexError("Dataset is empty")
        index = int(index % self.len)
        file_index = int(np.searchsorted(self.cumulative_sizes, index, side='right'))
        prev_total = int(self.cumulative_sizes[file_index - 1]) if file_index > 0 else 0
        file_offset = index - prev_total
        dA = np.load(self.dataA[file_index], mmap_mode='r')
        dB = np.load(self.dataB[file_index], mmap_mode='r')
        if dA.shape[0] <= file_offset or dB.shape[0] <= file_offset:
            return self.__getitem__(random.randint(0, self.len - 1))
        try:
            dataA = dA[file_offset, :, :, :]
            dataB = dB[file_offset, :, :, :]
        except Exception as e:
            print("Error: ", e)
            self.len = self.len - 200 + dA.shape[0]
            return self.__getitem__(random.randint(0, self.len - 1))
        if self.shape is None:
            self.shape = list(dataA.shape)
            self.shape[0] = -1
        dataA = dataA.reshape(self.shape)
        dataB = dataB.reshape(self.shape)
        dataA = torch.tensor(dataA, dtype=torch.float32)
        dataB = torch.tensor(dataB, dtype=torch.float32)
        if self.x_mean is not None and self.x_std is not None:
            dataA = (dataA - self.x_mean) / torch.clamp(self.x_std, min=1e-6)
        else:
            eps = 1e-6
            dataA = dataA / torch.clamp(torch.norm(dataA, dim=[-1, -2], keepdim=True), min=eps)
        if self.y_mean is not None and self.y_std is not None:
            dataB = (dataB - self.y_mean) / torch.clamp(self.y_std, min=1e-6)
        else:
            eps = 1e-6
            dataB = dataB / torch.clamp(torch.norm(dataB, dim=[-1, -2], keepdim=True), min=eps)
        return dataA, dataB

if __name__ == "__main__":
    from minio import Minio
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--MINIOHost", type=str, default="10.45.15.149")
    parser.add_argument("--MINIOPort", type=int, default=9000)
    parser.add_argument("--MINIOAccesskey", type=str, default="minioadmin")
    parser.add_argument("--MINIOSecret", type=str, default="minioadmin")
    parser.add_argument("--bucket_name", type=str, default="convmap")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--method", type=str, default="grid")
    parser.add_argument("--WindowsMinutes", type=int, default=120)
    args = parser.parse_args()

    minioClient = {"host": args.MINIOHost, "port": args.MINIOPort, "access_key": args.MINIOAccesskey, "secret_key": args.MINIOSecret}

    dataModule = DatasetFromMinioBucket(minioClient, args.bucket_name, args.data_dir, args.batch_size, args.method, args.WindowsMinutes)
    dataModule.prepare_data()
    dataModule.setup()
    for idx, batch in enumerate(dataModule.train_dataloader()):
        if idx % 10 == 0:
            fig, axs = plt.subplots(2)
            axs[0].imshow(batch[0].flatten(0, -2).numpy()[:, :300])
            axs[1].imshow(batch[1].flatten(0, -2).numpy()[:, :300])
            plt.savefig("test{}.png".format(idx))
            plt.close()
    print("Test Passed")
    print(dataModule.dataset.location)
