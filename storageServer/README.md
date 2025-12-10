# Storage Server Module

The Storage Server module provides a distributed MinIO object storage solution for WeatherLearn. It implements a high-availability, scalable storage cluster using Docker containers and NGINX load balancing, designed to handle large volumes of SuperDARN radar data and machine learning artifacts.

## Overview

This module sets up a distributed MinIO cluster with 4 storage nodes behind an NGINX load balancer. This configuration provides redundancy, load distribution, and high-performance data access for the WeatherLearn machine learning pipeline.

## Architecture

### Components

1. **MinIO Cluster**: 4 MinIO server instances for distributed object storage
2. **NGINX Load Balancer**: Routes requests across MinIO nodes with health checking
3. **Docker Compose**: Orchestrates the entire storage infrastructure

### Features

- **High Availability**: Multiple MinIO nodes ensure service continuity
- **Load Balancing**: NGINX distributes requests across healthy nodes
- **Data Redundancy**: Distributed storage across multiple drives/nodes
- **Health Monitoring**: Automatic health checks and failover
- **Scalability**: Easy to add more storage nodes as needed

## Configuration

### Storage Layout

The cluster maps to different physical drives for optimal performance:

```
minio1 -> /mnt/drive3
minio2 -> /mnt/drive2  
minio3 -> /mnt/drive4
minio4 -> /mnt/drive1
```

### Network Configuration

- **MinIO API**: Port 9000 (load balanced)
- **MinIO Console**: Port 9001 (load balanced with sticky sessions)
- **Internal Communication**: Custom Docker network `mynetwork`

### Access Credentials

- **Username**: minioadmin
- **Password**: minioadmin

> **Security Note**: Change default credentials in production environments

## Quick Start

### 1. Prerequisites

Ensure you have the required mount points:

```powershell
# Create mount directories (if using Windows with WSL/Docker Desktop)
mkdir C:\minio-data\drive1, C:\minio-data\drive2, C:\minio-data\drive3, C:\minio-data\drive4

# For Linux systems, ensure mount points exist:
# sudo mkdir -p /mnt/drive{1,2,3,4}
```

### 2. Start the Storage Cluster

```powershell
cd storageServer
docker-compose up -d
```

### 3. Verify Cluster Status

```powershell
# Check all services are running
docker-compose ps

# Check MinIO cluster status
docker-compose logs minio1
```

### 4. Access the Services

- **MinIO Console**: http://localhost:9001
- **MinIO API**: http://localhost:9000
- **Health Check**: http://localhost:9000/minio/health/live

## Usage

### Creating Buckets

```powershell
# Using MinIO client (mc)
docker run --rm -it --network storageserver_mynetwork minio/mc `
  mc alias set myminio http://nginx:9000 minioadmin minioadmin

docker run --rm -it --network storageserver_mynetwork minio/mc `
  mc mb myminio/superdarn-data

docker run --rm -it --network storageserver_mynetwork minio/mc `
  mc mb myminio/models

docker run --rm -it --network storageserver_mynetwork minio/mc `
  mc mb myminio/processed-data
```

### Python Integration

```python
from minio import Minio

# Connect to the load-balanced cluster
client = Minio(
    'localhost:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False
)

# List buckets
buckets = client.list_buckets()
for bucket in buckets:
    print(bucket.name)

# Upload a file
client.fput_object(
    'superdarn-data',
    'test-file.dat',
    '/path/to/local/file.dat'
)
```

### Integration with PTL Module

```python
from PTL.DataModule import DatasetFromMinioBucket

# Configure for the storage cluster
config = {
    "MINIOHost": "localhost",
    "MINIOPort": 9000,
    "MINIOAccesskey": "minioadmin", 
    "MINIOSecret": "minioadmin",
    "bucket_name": "superdarn-data"
}

# Use with PyTorch Lightning
datamodule = DatasetFromMinioBucket(**config)
```

## Configuration Files

### docker-compose.yaml

Defines the complete storage infrastructure:

- **4 MinIO services** with shared configuration
- **NGINX load balancer** with custom configuration
- **Custom network** for internal communication
- **Health checks** for automatic failover
- **Volume mounts** for persistent storage

### nginx.conf

NGINX configuration for load balancing:

- **Upstream pools** for API and Console
- **Health checking** with automatic failover
- **Session affinity** for console access
- **Large file support** for data uploads
- **Request buffering optimization**

## Monitoring and Maintenance

### Health Monitoring

```powershell
# Check cluster health
curl http://localhost:9000/minio/health/live

# Check individual node health
docker-compose exec minio1 curl http://localhost:9000/minio/health/live
```

### Log Monitoring

```powershell
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f minio1
docker-compose logs -f nginx
```

### Performance Monitoring

```powershell
# View resource usage
docker stats

# Monitor MinIO metrics
curl http://localhost:9000/minio/v2/metrics/cluster
```

## Scaling and Performance

### Adding Storage Nodes

To add more MinIO nodes:

1. Update `docker-compose.yaml` with new service definitions
2. Add corresponding upstream entries in `nginx.conf`
3. Restart the cluster

### Performance Optimization

- **Drive Selection**: Use separate physical drives for each node
- **Network**: Use dedicated network interfaces for storage traffic
- **Memory**: Allocate sufficient RAM for MinIO caching
- **CPU**: Consider CPU-optimized instances for encryption/compression

### Capacity Planning

- Monitor storage usage across nodes
- Plan for data growth and retention policies
- Consider backup and disaster recovery requirements

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```powershell
   # Check mount points exist
   docker-compose logs minio1
   
   # Verify network connectivity
   docker network ls
   ```

2. **Load Balancer Issues**
   ```powershell
   # Check NGINX configuration
   docker-compose exec nginx nginx -t
   
   # Reload NGINX configuration
   docker-compose exec nginx nginx -s reload
   ```

3. **Storage Issues**
   ```powershell
   # Check disk space
   docker-compose exec minio1 df -h /data
   
   # Verify permissions
   docker-compose exec minio1 ls -la /data
   ```

### Debug Commands

```powershell
# Enter MinIO container
docker-compose exec minio1 /bin/bash

# Check MinIO server info
docker-compose exec minio1 mc admin info local

# Test connectivity between nodes
docker-compose exec minio1 ping minio2
```

## Security Considerations

### Production Deployment

1. **Change Default Credentials**
   ```yaml
   environment:
     MINIO_ROOT_USER: your-admin-user
     MINIO_ROOT_PASSWORD: your-secure-password
   ```

2. **Enable TLS/SSL**
   - Configure SSL certificates in MinIO
   - Update NGINX for HTTPS termination
   - Use secure=True in client connections

3. **Network Security**
   - Use private networks for inter-node communication
   - Implement firewall rules
   - Consider VPN for remote access

4. **Access Control**
   - Implement IAM policies
   - Use service accounts for applications
   - Enable audit logging

## Backup and Recovery

### Data Backup

```powershell
# Backup bucket data
docker run --rm -v ${PWD}/backup:/backup --network storageserver_mynetwork minio/mc `
  mc mirror myminio/superdarn-data /backup/superdarn-data

# Backup configuration
docker-compose exec minio1 mc admin config export local > minio-config-backup.json
```

### Disaster Recovery

1. **Node Failure**: Cluster continues with remaining nodes
2. **Data Recovery**: Restore from backups to new nodes
3. **Complete Failure**: Rebuild cluster and restore from backups

## File Structure

```
storageServer/
├── __init__.py              # Package initialization
├── docker-compose.yaml      # Docker Compose configuration
├── nginx.conf              # NGINX load balancer configuration
└── README.md               # This documentation
```

## Integration Examples

### With Machine Learning Pipeline

```python
# Upload training data
client.fput_object('superdarn-data', 'training/dataset.h5', 'local-dataset.h5')

# Save model checkpoints
client.fput_object('models', 'pangu-model-v1.pth', 'model.pth')

# Store processed results
client.fput_object('processed-data', 'predictions/forecast.nc', 'output.nc')
```

### With Data Processing

```python
from PTL.MinioToDisk import download_minio_bucket_to_folder

# Download data for local processing
download_minio_bucket_to_folder(
    minio_client=client,
    bucket_name='superdarn-data',
    target_folder='./local-data'
)
```

## References

- [MinIO Documentation](https://docs.min.io/)
- [NGINX Load Balancing](https://docs.nginx.com/nginx/admin-guide/load-balancer/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [MinIO Python SDK](https://docs.min.io/docs/python-client-quickstart-guide.html)