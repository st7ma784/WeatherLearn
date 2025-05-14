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

from minio import Minio
from minio.error import S3Error
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def download_minio_bucket_to_folder(minio_config, bucket_name, target_folder):
    """
    Downloads all objects from a Minio bucket to a local folder.

    Args:
        minio_config (dict): Configuration for Minio client with keys:
                             - host: Minio server host
                             - port: Minio server port
                             - access_key: Access key for Minio
                             - secret_key: Secret key for Minio
        bucket_name (str): Name of the bucket to download.
        target_folder (str): Path to the local folder where files will be saved.
    """
    # print(f"Downloading from bucket: {bucket_name} to folder: {target_folder}")
    # Create Minio client
    minio_client = Minio(
        f"{minio_config.get('host', 'localhost')}:{minio_config.get('port', 9000)}",
        access_key=minio_config.get("access_key", "minioadmin"),
        secret_key=minio_config.get("secret_key", "minioadmin"),
        secure=False
    )

    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)
    def download_object(obj):
        try:
            object_path = obj.object_name
            local_file_path = os.path.join(target_folder, object_path)
            # Ensure subdirectories exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # Download the object
            minio_client.fget_object(bucket_name, object_path, local_file_path)
            # print(f"Downloaded: {object_path} -> {local_file_path}")
        except S3Error as e:
            print(f"Error occurred while downloading {obj.object_name}: {e}")

    # List and download all objects in the bucket using multithreading
    try:
        # print("Starting download...")
        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust max_workers as needed
            objects = list(minio_client.list_objects(bucket_name, recursive=True))
            with tqdm(total=len(objects), desc="Downloading files", unit="file") as pbar:
                for _ in executor.map(download_object, objects):
                    pbar.update(1)
    except S3Error as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Example configuration
    import argparse
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="Minio server host")
    parser.add_argument("--port", type=int, default=9000, help="Minio server port")
    parser.add_argument("--access_key", type=str, default="minioadmin", help="Access key for Minio")
    parser.add_argument("--secret_key", type=str, default="minioadmin", help="Secret key for Minio")
    parser.add_argument("--bucket", type=str, default="convmap", help="Name of the bucket to download")
    parser.add_argument("--target_folder", type=str, default="/tmp", help="Path to the local folder where files will be saved")
    args = parser.parse_args()
    minio_config = {
        "host": args.host,
        "port": args.port,
        "access_key": args.access_key,
        "secret_key": args.secret_key
    }
    bucket_name = args.bucket
    target_folder = args.target_folder

    download_minio_bucket_to_folder(minio_config, bucket_name, target_folder)