# WeatherLearn - for SuperDARN

A machine learning project for SuperDARN. We will take a WeatherLearn architecture and apply the logic to the data from superdarn, to analyse the effectiveness for forecasting ionosphere activity. 

## Overview

WeatherLearn is designed to predict weather patterns using machine learning algorithms. The project utilizes MinIO for scalable file storage and implements various ML models to analyze and forecast weather conditions.

## MinIO File Store Setup

MinIO is used as the object storage backend for managing large weather datasets and model artefacts.

### Quick Start with Docker

1. **Start MinIO using Docker Compose:**
   ```bash
   docker-compose up -d minio
   ```

2. **Access MinIO Console:**
   - URL: http://localhost:9001
   - Default credentials: `minioadmin` / `minioadmin`

3. **Create required buckets:**
   ```bash
   # Using MinIO client (mc)
   mc alias set local http://localhost:9000 minioadmin minioadmin
   mc mb local/weather-data
   mc mb local/models
   mc mb local/processed-data
   ```

### Configuration

Update your environment variables or config files:
```env
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
```

## PTL Module - Machine Learning Pipeline

The PTL (PyTorch Lightning) module is the core machine learning component of WeatherLearn, implementing a modified Pangu-Weather model architecture for SuperDARN ionospheric flow forecasting.

### Key Features

- **Pangu-Weather Model**: Adapted transformer architecture for geospatial data
- **3D Attention Mechanisms**: Specialized for atmospheric/ionospheric data processing
- **PyTorch Lightning**: Efficient training with automatic optimization and scaling
- **HPC Integration**: SLURM support for high-performance computing environments
- **Experiment Tracking**: Weights & Biases and Neptune integration
- **Distributed Storage**: MinIO integration for scalable data management

### Quick Start with PTL

1. **Install dependencies:**
   ```bash
   cd PTL
   pip install -r requirements.txt
   ```

2. **Basic training example:**
   ```python
   from PTL.launch import train
   
   config = {
       "batch_size": 16,
       "learning_rate": 1e-4,
       "MINIOHost": "localhost",
       "MINIOPort": 9000,
       "MINIOAccesskey": "minioadmin",
       "MINIOSecret": "minioadmin",
       "bucket_name": "superdarn-data",
       "grid_size": 300
   }
   
   train(config, accelerator="gpu", devices=1)
   ```

3. **Training with experiment tracking:**
   ```python
   from PTL.launch import wandbtrain
   
   wandbtrain(config, project_name="superdarn-forecast")
   ```

### PTL Module Components

#### 1. Model Architecture (`PTL/model.py`)
- **Pangu**: Main PyTorch Lightning model implementation
- **EarthAttention3D**: 3D window attention with earth position bias
- **EarthSpecificBlock**: Transformer blocks for atmospheric data
- **DropPath**: Stochastic depth regularization

#### 2. Data Management (`PTL/DataModule.py`)
- **DatasetFromMinioBucket**: Load data directly from MinIO storage
- **DatasetFromPresaved**: Load preprocessed data from disk
- Support for SuperDARN FITACF and CONVMAP file formats
- Configurable time windows and data representations

#### 3. Training Framework (`PTL/launch.py`)
- **train()**: Basic training with configurable parameters
- **wandbtrain()**: Training with Weights & Biases logging
- **neptunetrain()**: Training with Neptune logging
- **SlurmRun()**: Generate SLURM scripts for HPC deployment

#### 4. Data Processing (`PTL/generateFitToConv.py`)
- Process SuperDARN FITACF files into organized datasets
- Associate FITACF files with corresponding CONVMAP files
- Handle radar record extraction and time-based grouping

#### 5. Utilities (`PTL/utils.py`)
- Patch embedding and recovery operations
- Up/down sampling for multi-resolution processing
- Window partitioning for attention mechanisms
- Earth-specific helper functions

#### 6. Data Transfer (`PTL/MinioToDisk.py`)
- Efficient multithreaded downloads from MinIO
- Preserve folder structure during transfer
- Progress tracking and error handling

#### 7. Data Exploration (`PTL/viewDataFiles.ipynb`)
- Jupyter notebook for exploring SuperDARN data formats
- Examples of reading RAWACF, FITACF, and GRDMAP files
- Data visualization and analysis workflows

### Configuration Parameters

**Model Parameters:**
- `embed_dim`: Patch embedding dimension (default: 128)
- `num_heads`: Attention heads per layer (default: (8, 16, 16, 8))
- `window_size`: Attention window size (default: (2, 8, 16))
- `learning_rate`: Learning rate (default: 1e-4)
- `grid_size`: Input grid resolution (default: 300)

**Data Parameters:**
- `batch_size`: Training batch size (default: 16)
- `bucket_name`: MinIO bucket name
- `method`: Data representation ('flat' or 'grid')
- `windowMinutes`: Time window size in minutes

**Training Parameters:**
- `max_epochs`: Maximum training epochs (default: 200)
- `accelerator`: Device type ('gpu', 'cpu', 'auto')
- `devices`: Number of devices
- `precision`: Training precision (16, 32)

### HPC Deployment

Generate SLURM scripts for HPC environments:

```python
from PTL.launch import SlurmRun

SlurmRun(
    config,
    time="24:00:00",
    partition="gpu", 
    nodes=1,
    ntasks_per_node=1,
    gres="gpu:1"
)
```

### Performance Optimization

- Use mixed precision training (`precision=16`)
- Enable multiple data loading workers (`num_workers=12`)
- Use MinIO for distributed storage access
- Implement gradient checkpointing for large models

For detailed documentation, see `PTL/README.md`.

## Storage Server Module

The Storage Server module provides a distributed MinIO object storage solution with high availability and load balancing.

### Features

- **Distributed Storage**: 4-node MinIO cluster for redundancy
- **Load Balancing**: NGINX-based request distribution
- **High Availability**: Automatic failover and health monitoring
- **Scalability**: Easy to add more storage nodes

### Quick Setup

1. **Start the storage cluster:**
   ```bash
   cd storageServer
   docker-compose up -d
   ```

2. **Access MinIO Console:**
   - URL: http://localhost:9001
   - Credentials: minioadmin/minioadmin

3. **Create buckets:**
   ```bash
   docker run --rm -it --network storageserver_mynetwork minio/mc \
     mc alias set myminio http://nginx:9000 minioadmin minioadmin
   
   docker run --rm -it --network storageserver_mynetwork minio/mc \
     mc mb myminio/superdarn-data
   ```

For detailed documentation, see `storageServer/README.md`.

## Project Workflow

1. **Data Ingestion**: SuperDARN FITACF files → MinIO storage
2. **Data Processing**: Raw radar data → Processed grid format
3. **Model Training**: Grid data → Pangu model → Trained weights
4. **Evaluation**: Model predictions vs. ground truth
5. **Deployment**: Trained model → Inference pipeline


1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
## Contributing
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
