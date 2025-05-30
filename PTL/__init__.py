"""
PTL Package
===========

This package contains modules and utilities for training machine learning models using PyTorch Lightning.
It includes data handling, model definitions, and training scripts.

Modules
-------

- **DataModule**:
  Handles data loading and preprocessing.

- **generateFitToConv**:
  Processes FITACF files and generates CONVMAP files.

- **launch**:
  Provides training scripts and SLURM integration for HPC environments.

- **model**:
  Defines machine learning models, including the Pangu model.

- **utils**:
  Contains utility functions and classes for tensor operations and model components.
"""
from .DataModule import DatasetFromMinioBucket, DatasetFromPresaved
from .generateFitToConv import process_fitacf_to_filelists, find_conv_maps_from_filelists
from .launch import train, wandbtrain, neptunetrain, SlurmRun
from .model import Pangu, DropPath, EarthSpecificBlock
from .model import norm_cdf, EarthAttention3D
from .utils import (
    UpSample,
    DownSample,
    crop2d,
    crop3d,
)

import os
import sys
#run pip install -r requirements.txt
