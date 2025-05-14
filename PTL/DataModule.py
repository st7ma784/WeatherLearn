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
from torchvision import datasets, transforms
import torch
from torch.utils.data import IterableDataset, Dataset
import pydarnio
import numpy as np
import datetime
from minio import Minio
import os
import random
from tqdm import tqdm
import hashlib
import time

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
                if output[0] is None or output[1] is None:
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
        try:
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
        while True:
            try:
                obj = next(self.generator())
                try:
                    minioClient = Minio(self.miniodict.get("host", "localhost") + ":" + str(self.miniodict.get("port", 9000)),
                                        access_key=self.miniodict.get("access_key", "minioadmin"),
                                        secret_key=self.miniodict.get("secret_key", "minioadmin"),
                                        secure=False)
                    data = minioClient.get_object(self.minioBucket, obj.object_name)
                    data1 = self.process_file(data)
                    if data1 is None:
                        continue
                    yield data1
                except Exception as e:
                    print("Error: ", e)
                    self.minioClient.remove_object(self.minioBucket, obj.object_name)
            except Exception as e:
                pass

    def __getitem__(self, index):
        """
        Retrieves the data at the specified index.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            tuple: The data at the specified index.
        """
        return next(self.__iter__())


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

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset) * 0.8), len(self.dataset) - int(len(self.dataset) * 0.8)])

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: The training DataLoader.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=3)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: The validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: The test DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


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
        np.save(os.path.join(path, "dataA_{}.npy".format(idx)), tensorA.numpy())
        np.save(os.path.join(path, "dataB_{}.npy".format(idx)), tensorB.numpy())
        tensorshape = list([tensorA.shape[i] for i in range(len(tensorA.shape))])
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
    shape = None
    try:
        with open(os.path.join(path, "shape.txt"), 'r') as f:
            shape = eval(f.read())
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
        self.len = len(dataA) * 200

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
        file_index = index // 200
        file_offset = index % 200
        dA = np.load(self.dataA[file_index], mmap_mode='r')
        dB = np.load(self.dataB[file_index], mmap_mode='r')
        if dA.shape[0] <= file_offset or dB.shape[0] <= file_offset:
            self.len = self.len - 200 + dA.shape[0]
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
        dataA = dataA / torch.norm(dataA, dim=[-1, -2], keepdim=True)
        dataB = dataB / torch.norm(dataB, dim=[-1, -2], keepdim=True)
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
