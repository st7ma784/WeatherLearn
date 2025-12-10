# PTL (PyTorch Lightning) Module

The PTL module is the core machine learning component of WeatherLearn, adapted for SuperDARN ionosphere activity forecasting. It implements a modified Pangu-Weather model architecture using PyTorch Lightning for efficient training and deployment.

## Overview

This module provides an end-to-end machine learning pipeline for analyzing SuperDARN radar data to forecast ionosphere activity. It includes data preprocessing, model training, evaluation, and deployment capabilities with support for High-Performance Computing (HPC) environments.

## Key Components

### 1. Model Architecture (`model.py`)

The core model is based on the Pangu-Weather architecture, adapted for SuperDARN data:

- **Pangu Model**: PyTorch Lightning implementation with earth-specific attention mechanisms
- **EarthAttention3D**: 3D window attention with earth position bias for handling geospatial data
- **EarthSpecificBlock**: Transformer blocks optimized for atmospheric/ionospheric data
- **DropPath**: Stochastic depth implementation for regularization

Key features:
- 3D attention mechanisms for handling pressure levels, latitude, and longitude
- Skip connections for better gradient flow
- Patch embedding for efficient processing of grid data
- Support for multi-step forecasting

### 2. Data Management (`DataModule.py`)

Handles data loading and preprocessing for SuperDARN radar data:

- **DatasetFromMinioBucket**: Loads data directly from MinIO object storage
- **DatasetFromPresaved**: Loads preprocessed data from disk
- Supports both 'flat' and 'grid' data representations
- Configurable time windows for temporal analysis
- Integration with PyTorch Lightning's DataModule interface

### 3. Training Framework (`launch.py`)

Provides comprehensive training capabilities:

- **train()**: Basic training function with configurable parameters
- **wandbtrain()**: Training with Weights & Biases logging
- **neptunetrain()**: Training with Neptune logging
- **SlurmRun()**: SLURM script generation for HPC environments

### 4. Data Processing (`generateFitToConv.py`)

Tools for processing SuperDARJN data files:

- **process_fitacf_to_filelists()**: Process FITACF files into organized datasets
- **find_conv_maps_from_filelists()**: Associate FITACF files with corresponding CONVMAP files
- Handles file loading and radar record extraction

### 5. Utilities (`utils.py`)sudo usermod -aG docker $USER

Supporting components for model operations:

- **Patch operations**: PatchEmbed2D, PatchEmbed3D, PatchRecovery2D
- **Sampling operations**: UpSample, DownSample
- **Window operations**: WindowPartition, WindowReverse
- **Cropping operations**: Crop3D, crop2d, crop3d
- Earth-specific helper functions

### 6. Data Transfer (`MinioToDisk.py`)

Efficient data transfer from MinIO storage:

- Multithreaded downloads for improved performance
- Preserves folder structure during transfer
- Progress tracking with tqdm
- Error handling and logging

### 7. Data Exploration (`viewDataFiles.ipynb`)

Jupyter notebook for exploring SuperDARN data formats:

- Examples of reading RAWACF, FITACF, and GRDMAP files
- Data structure visualization
- Sample data analysis workflows

## Quick Start

### 1. Environment Setup

Install required dependencies:

```bash
cd PTL
pip install -r requirements.txt
```

### 2. Start MinIO Storage (Optional)

If using MinIO for data storage:

```bash
docker-compose up -d minio
```

Access MinIO console at http://localhost:9001 (minioadmin/minioadmin)

### 3. Basic Training

```python
from PTL.launch import train
from PTL.DataModule import DatasetFromMinioBucket

# Configure MinIO connection
config = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "MINIOHost": "localhost",
    "MINIOPort": 9000,
    "MINIOAccesskey": "minioadmin",
    "MINIOSecret": "minioadmin",
    "bucket_name": "superdarn-data",
    "grid_size": 300,
    "time_step": 1
}

# Start training
train(config, accelerator="gpu", devices=1)
```

### 4. Training with Logging

```python
from PTL.launch import wandbtrain

# Train with Weights & Biases logging
wandbtrain(config, project_name="superdarn-forecast")
```

### 5. HPC Deployment

```python
from PTL.launch import SlurmRun

# Generate SLURM script for HPC
SlurmRun(
    config, 
    time="24:00:00",
    partition="gpu",
    nodes=1,
    ntasks_per_node=1,
    gres="gpu:1"
)
```

## Configuration Parameters

### Model Parameters

- `embed_dim` (int): Patch embedding dimension (default: 128)
- `num_heads` (tuple): Number of attention heads per layer (default: (8, 16, 16, 8))
- `window_size` (tuple): Attention window size (default: (2, 8, 16))
- `learning_rate` (float): Learning rate (default: 1e-4)
- `grid_size` (int): Input grid resolution (default: 300)
- `time_step` (int): Number of forecasting steps (default: 1)
- `mlp_ratio` (float): MLP expansion ratio (default: 4)
- `noise_factor` (float): Training noise factor (default: 0.1)

### Data Parameters

- `batch_size` (int): Training batch size (default: 16)
- `bucket_name` (str): MinIO bucket name
- `data_dir` (str): Local data directory
- `method` (str): Data representation ('flat' or 'grid')
- `windowMinutes` (int): Time window size in minutes

### Training Parameters

- `max_epochs` (int): Maximum training epochs (default: 200)
- `accelerator` (str): Device type ('gpu', 'cpu', 'auto')
- `devices` (int/str): Number of devices ('auto' or specific number)
- `precision` (int): Training precision (16, 32)
- `gradient_clip_val` (float): Gradient clipping value (default: 0.25)

## File Structure

```
PTL/
├── __init__.py           # Package initialization
├── DataModule.py         # Data loading and preprocessing
├── model.py             # Model definitions (Pangu, attention mechanisms)
├── utils.py             # Utility functions and classes
├── launch.py            # Training scripts and HPC integration
├── generateFitToConv.py # SuperDARN data processing
├── MinioToDisk.py       # Data transfer utilities
├── viewDataFiles.ipynb  # Data exploration notebook
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # MinIO setup
└── README.md           # This file
```

## Data Flow

1. **Data Ingestion**: SuperDARN FITACF files → MinIO bucket
2. **Data Processing**: FITACF files → Processed grid data
3. **Training**: Grid data → Pangu model → Predictions
4. **Evaluation**: Model predictions vs. ground truth
5. **Deployment**: Trained model → Inference pipeline

## Advanced Usage

### Custom Data Processing

```python
from PTL.generateFitToConv import process_fitacf_to_filelists

# Process FITACF files
file_lists = process_fitacf_to_filelists(
    folder_path="data/fitacf",
    time_window=120,  # minutes
    min_radars=3
)
```

### Model Customization

```python
from PTL.model import Pangu

# Custom model configuration
model = Pangu(
    embed_dim=256,
    num_heads=(16, 32, 32, 16),
    window_size=(4, 12, 24),
    learning_rate=5e-5,
    grid_size=512
)
```

### Distributed Training

```python
# Multi-GPU training
train(config, accelerator="gpu", devices=4)

# Multi-node training (with SLURM)
SlurmRun(config, nodes=2, ntasks_per_node=2, gres="gpu:2")
```

## Performance Optimization

### Memory Usage
- Use mixed precision training (`precision=16`)
- Adjust batch size based on GPU memory
- Enable gradient checkpointing for large models

### Training Speed
- Use multiple workers for data loading (`num_workers=12`)
- Enable pin_memory for GPU training
- Use prefetch_factor for improved throughput

### Storage Optimization
- Use MinIO for distributed storage access
- Preprocess data to reduce I/O overhead
- Use efficient data formats (HDF5, Zarr)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Slow Data Loading**: Increase num_workers or check storage I/O
3. **Training Instability**: Adjust learning rate or enable gradient clipping
4. **MinIO Connection Issues**: Check network connectivity and credentials

### Debug Mode

```python
# Enable fast development run
config["fast_dev_run"] = True
train(config)
```

### Logging and Monitoring

- Use Weights & Biases for experiment tracking
- Monitor GPU utilization and memory usage
- Check training curves for convergence issues
- Save model checkpoints regularly

## Contributing

When modifying the PTL module:

1. Follow PyTorch Lightning best practices
2. Add appropriate documentation and type hints
3. Update requirements.txt for new dependencies
4. Test with both CPU and GPU configurations
5. Ensure HPC compatibility

## References

- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Pangu-Weather Paper](https://arxiv.org/abs/2211.02556)
- [SuperDARN Documentation](https://superdarn.ca/)
- [pydarnio Library](https://github.com/SuperDARN/pydarnio)