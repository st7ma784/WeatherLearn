from minio import Minio
from minio.error import S3Error
import os
from concurrent.futures import ThreadPoolExecutor

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
            executor.map(download_object, minio_client.list_objects(bucket_name, recursive=True))
    except S3Error as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Example configuration
    import argparse
    from concurrent.futures import ThreadPoolExecutor
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