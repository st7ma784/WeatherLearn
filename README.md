# WeatherLearn - for SUPERDarn

A machine learning project for SUPERDarn. WE will take a weatherlearn architecture and apply the logic to the data from superdarn, to analyse the effectiveness for forecasting ionosphere activity. 

## Overview

WeatherLearn is designed to predict weather patterns using machine learning algorithms. The project utilizes MinIO for scalable file storage and implements various ML models to analyze and forecast weather conditions.

## MinIO File Store Setup

MinIO is used as the object storage backend for managing large weather datasets and model artifacts.

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

The PTL (PyTorch Lightning) module is the core machine learning component of WeatherLearn, implementing a modified Pangu-Weather model architecture for SUPERDarn ionosphere activity forecasting.

### Key Features

- **Pangu-Weather Model**: Adapted transformer architecture for geospatial data
- **3D Attention Mechanisms**: Specialized for atmospheric/ionospheric data processing
- **PyTorch Lightning**: Efficient training with automatic optimization and scaling
- **HPC Integration**: SLURM support for high-performance computing environments
- **Experiment Tracking**: Weights & Biases and Neptune integration
- **Distributed Storage**: MinIO integration for scalable data management
- **Solar Wind Conditioning**: Optional FiLM-based conditioning on IMF (Bx, By, Bz, Kp, Vx) — model degrades gracefully without it

### Quick Start with PTL

1. **Install dependencies:**
   ```bash
   cd PTL
   pip install -r requirements.txt
   ```

2. **Baseline training on cnvmap files (no MinIO required):**
   ```bash
   # Preprocess cnvmap → .npy chunks (run once)
   python run_baseline.py --preprocess --grid_size 120 --max_files 500

   # Train with solar wind FiLM conditioning + W&B logging
   python run_baseline.py --grid_size 120 --max_files 500 --epochs 20 \
       --wandb --wandb_project SuperDARN-baseline
   ```

3. **Basic training example (MinIO backend):**
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

### PTL Module Components

#### 1. Model Architecture (`PTL/model.py`)
- **Pangu**: Main PyTorch Lightning model implementation
- **EarthAttention3D**: 3D window attention with earth position bias
- **EarthSpecificBlock**: Transformer blocks for atmospheric data
- **FiLMLayer**: Feature-wise Linear Modulation for solar wind conditioning
- **DropPath**: Stochastic depth regularization

#### 2. Data Management (`PTL/DataModule.py`)
- **DatasetFromMinioBucket**: Load data directly from MinIO storage
- **DatasetFromPresaved**: Load preprocessed data from disk (mmap-backed, memory efficient)
- Support for SUPERDarn FITACF and CONVMAP file formats
- Configurable time windows and data representations

#### 3. Baseline Training (`PTL/run_baseline.py`)
- Two-phase pipeline: preprocess cnvmap files to `.npy` chunks, then train via mmap
- IMF solar wind extraction and FiLM conditioning (Bx, By, Bz, Kp, Vx)
- W&B logging with predicted vs actual convection map panels

#### 4. Training Framework (`PTL/launch.py`)
- **train()**: Basic training with configurable parameters
- **wandbtrain()**: Training with Weights & Biases logging
- **neptunetrain()**: Training with Neptune logging
- **SlurmRun()**: Generate SLURM scripts for HPC deployment

#### 5. Data Processing (`PTL/generateFitToConv.py`)
- Process SUPERDarn FITACF files into organized datasets
- Associate FITACF files with corresponding CONVMAP files

#### 6. Utilities (`PTL/utils.py`)
- Patch embedding and recovery operations
- Up/down sampling for multi-resolution processing
- Window partitioning for attention mechanisms

### W&B Diagnostics

Diagnostics are logged to Weights & Biases during training/validation.

Configuration flags (from `PTL/launch.py`):

- `--log_diagnostics` (default: `True`)
- `--diagnostics_interval` (default: `50`)
- `--log_images_every_n_val_epochs` (default: `1`)

EMA and normalization controls:

- `--use_ema`
- `--use_ema_eval`
- `--ema_decay`
- `--ema_warmup_steps`
- `--cache_stats`

Interpretation guide:

- See `PTL/DIAGNOSTICS.md` for expected trends, failure signatures, and corrective actions.
- See `PTL/DIAGNOSTICS.md` section "Recommended W&B Dashboard Layout" for a standard panel setup.

Auto-create the W&B dashboard report:

```bash
cd PTL
python setup_wandb_dashboard.py --entity <your-wandb-entity> --project <your-wandb-project>
```

### Configuration Parameters

**Model Parameters:**
- `embed_dim`: Patch embedding dimension (default: 128)
- `num_heads`: Attention heads per layer (default: (8, 16, 16, 8))
- `window_size`: Attention window size (default: (2, 8, 16))
- `learning_rate`: Learning rate (default: 1e-4)
- `grid_size`: Input grid resolution (default: 300)
- `solar_wind_dim`: IMF feature dimension (default: 0 = disabled)

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

1. **Data Ingestion**: SUPERDarn FITACF/CONVMAP files → MinIO storage or local `.npy` cache
2. **Data Processing**: Raw radar data → Processed grid format (5 channels, configurable resolution)
3. **Model Training**: Grid data + optional IMF solar wind → Pangu model → Trained weights
4. **Evaluation**: Model predictions vs. ground truth; persistence and climatology skill scores
5. **Deployment**: Trained model → Inference pipeline

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
