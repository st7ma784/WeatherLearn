import os
import tempfile
from pathlib import Path
from minio import Minio
from smb.SMBConnection import SMBConnection
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple
from tqdm import tqdm

def create_smb_connection(smb_server: str, smb_user: str, smb_password: str) -> SMBConnection:
    """Create a new SMB connection for thread-safe operations"""
    conn = SMBConnection(smb_user, smb_password, f"client_{threading.current_thread().ident}", smb_server)
    conn.connect(smb_server)
    return conn

def transfer_single_file(file_info: Tuple, smb_config: dict, minio_config: dict, bucket_name: str) -> str:
    """Transfer a single file from SMB to MinIO"""
    file_obj, filename, file_size = file_info
    
    # Create thread-local connections
    minio_client = Minio(**minio_config)
    smb_conn = create_smb_connection(
        smb_config['server'], 
        smb_config['user'], 
        smb_config['password']
    )
    
    try:
        with tempfile.NamedTemporaryFile() as temp_file:
            # Download from SMB to temp file
            smb_conn.retrieveFile(smb_config['share'], filename, temp_file)
            temp_file.seek(0)
            
            # Upload to MinIO
            minio_client.put_object(
                bucket_name,
                filename,
                temp_file,
                file_size
            )
            
            return f"Transferred: {filename}"
    
    finally:
        smb_conn.close()

def transfer_smb_to_minio(user, password, max_workers: int = 4, share: str = "/") -> None:
    # MinIO configuration
    minio_config = {
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "secure": False
    }
    
    # SMB configuration
    smb_config = {
        "server": "luna",  # Remove the //luna/fst format, just use hostname
        "share": "fst",    # Separate share name from server
        "user": user,
        "password": password
    }
    
    bucket_name = "superdarn-data"
    
    # Get file list using main connection
    main_conn = create_smb_connection(smb_config['server'], smb_config['user'], smb_config['password'])
    
    try:
        # List files in SMB share
        file_list = main_conn.listPath(smb_config['share'], share)
        
        # Prepare file info tuples for non-directory files
        files_to_transfer = [
            (file_info, file_info.filename, file_info.file_size)
            for file_info in file_list 
            if not file_info.isDirectory
        ]
        
        print(f"Found {len(files_to_transfer)} files to transfer")
        
        # Transfer files using thread pool with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all transfer tasks
            future_to_file = {
                executor.submit(transfer_single_file, file_info, smb_config, minio_config, bucket_name): file_info[1]
                for file_info in files_to_transfer
            }
            
            # Process completed transfers with progress bar
            with tqdm(total=len(files_to_transfer), desc="Transferring files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        result = future.result()
                        pbar.set_postfix_str(f"Last: {filename}")
                        pbar.update(1)
                    except Exception as exc:
                        pbar.write(f"File {filename} generated an exception: {exc}")
                        pbar.update(1)
    
    finally:
        main_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer files from SMB to MinIO")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("--user", type=str, required=True, help="SMB username")
    parser.add_argument("--password", type=str, required=True, help="SMB password")
    parser.add_argument("--share", type=str, default="/PY/SPP/data/SuperDARN/rawacf/2012/01/", help="SMB share path (default: /)")
    args = parser.parse_args()
    
    transfer_smb_to_minio(args.user, args.password, max_workers=args.workers, share=args.share)