#!/usr/bin/env python
"""
Project setup script for Time Series Forecasting Engine.
Run this script to create the entire project structure.
"""

import os
import sys
import shutil
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {path}")

def create_file(path, content=""):
    """Create file with optional content."""
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def setup_project():
    """Set up the complete project directory structure."""
    # Create main project directory
    project_root = "time_series_forecasting_engine"
    create_directory(project_root)
    
    # Create subdirectories
    dirs = [
        f"{project_root}/data/raw",
        f"{project_root}/data/processed",
        f"{project_root}/data/external",
        f"{project_root}/notebooks",
        f"{project_root}/src/data",
        f"{project_root}/src/features",
        f"{project_root}/src/models",
        f"{project_root}/src/visualization",
        f"{project_root}/src/utils",
        f"{project_root}/src/pipeline",
        f"{project_root}/src/config",
        f"{project_root}/src/monitoring",
        f"{project_root}/tests",
        f"{project_root}/docs",
        f"{project_root}/deployment",
    ]
    
    for directory in dirs:
        create_directory(directory)
    
    # Create __init__.py files in all Python package directories
    python_dirs = [
        f"{project_root}/src",
        f"{project_root}/src/data",
        f"{project_root}/src/features",
        f"{project_root}/src/models",
        f"{project_root}/src/visualization",
        f"{project_root}/src/utils",
        f"{project_root}/src/pipeline",
        f"{project_root}/src/config",
        f"{project_root}/src/monitoring",
    ]
    
    for directory in python_dirs:
        create_file(os.path.join(directory, "__init__.py"))
    
    # Create main project files
    create_file(os.path.join(project_root, "README.md"), """# Time Series Forecasting Engine

A production-ready machine learning system for time series analysis and forecasting built for high-frequency trading applications.

## Project Overview

This project implements an end-to-end pipeline for collecting, processing, modeling, and deploying time series forecasting models for market data.

## Features

- Comprehensive data pipeline for time series data
- Multiple deep learning architectures (LSTM, GRU, Transformer)
- Feature engineering optimized for financial time series
- MLOps infrastructure for monitoring and continuous deployment
- Drift detection and automated retraining capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time_series_forecasting_engine.git
cd time_series_forecasting_engine

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

See the notebooks directory for examples of how to use the system.

## License

Internal use only
""")
    
    create_file(os.path.join(project_root, ".gitignore"), """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Data - don't commit data files by default
data/raw/
data/processed/

# Model files
*.h5
*.pkl
*.pt

# Logs
logs/
*.log

# Environment variables
.env

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
""")
    
    create_file(os.path.join(project_root, "requirements.txt"), """# Core libraries
numpy==1.23.5
pandas==2.0.1
scikit-learn==1.2.2

# Deep Learning
torch==2.0.1
torchvision==0.15.2
transformers==4.30.2

# Data Processing
pyarrow==12.0.0
fastparquet==2023.4.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.15.0

# MLOps
mlflow==2.4.1
tensorboard==2.13.0
optuna==3.2.0
ray[tune]==2.5.1

# API and Serving
fastapi==0.97.0
uvicorn==0.22.0
pydantic==1.10.9

# Monitoring
prometheus-client==0.17.0
evidently==0.3.1

# Testing
pytest==7.3.1
pytest-cov==4.1.0

# Utilities
tqdm==4.65.0
python-dotenv==1.0.0
pyyaml==6.0
joblib==1.2.0
""")
    
    create_file(os.path.join(project_root, "setup.py"), """from setuptools import find_packages, setup

setup(
    name='time_series_forecasting_engine',
    packages=find_packages(),
    version='0.1.0',
    description='Time Series Forecasting Engine for High-Frequency Trading',
    author='Your Name',
    license='Internal',
)
""")
    
    # Create core Python module files
    modules = {
        f"{project_root}/src/data/data_collector.py": """\"\"\"
Data Collector Module - Responsible for collecting time series data from various sources.
\"\"\"
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import yaml
import requests

class DataCollector:
    \"\"\"
    A class for collecting time series data from various sources.
    
    This collector is designed to fetch high-frequency trading data from:
    - Internal databases
    - Market data APIs
    - Flat files (CSV, Parquet)
    \"\"\"
    
    def __init__(self, config: Optional[Dict] = None):
        \"\"\"Initialize the DataCollector with configuration settings.\"\"\"
        self.config = config or {}
        self.data_dir = self.config.get('data_dir', 'data/raw')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_market_data(self, symbols, start_date, end_date, interval='1m'):
        \"\"\"
        Collect market data for specified symbols.
        
        Args:
            symbols: List of ticker symbols
            interval: Time interval for data (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            
        Returns:
            DataFrame with market data
        \"\"\"
        # Implementation details will go here
        pass
""",
        
        f"{project_root}/src/data/data_preprocessor.py": """\"\"\"
Data Preprocessor Module - Handles preprocessing of time series data.
\"\"\"
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class DataPreprocessor:
    \"\"\"
    A class for preprocessing time series data for machine learning models.
    
    Handles:
    - Data cleaning
    - Feature normalization
    - Missing value imputation
    - Outlier detection and handling
    - Time series specific preprocessing
    \"\"\"
    
    def __init__(self, config: Optional[Dict] = None):
        \"\"\"Initialize the DataPreprocessor with configuration.\"\"\"
        self.config = config or {}
        
    def process(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        \"\"\"
        Main processing method that applies the complete preprocessing pipeline.
        
        Args:
            data: Raw input data as a DataFrame
            is_training: Whether this is training data (True) or inference data (False)
            
        Returns:
            Processed DataFrame
        \"\"\"
        # Implementation details will go here
        pass
""",
        
        f"{project_root}/src/features/feature_engineering.py": """\"\"\"
Feature Engineering Module - Creates meaningful features from time series data.
\"\"\"
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

class FeatureEngineer:
    \"\"\"
    A class for engineering features from time series data.
    
    Creates:
    - Technical indicators
    - Statistical features
    - Time-based features
    - Domain-specific market features
    \"\"\"
    
    def __init__(self, config: Optional[Dict] = None):
        \"\"\"Initialize the FeatureEngineer with configuration.\"\"\"
        self.config = config or {}
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Create features from the input data.
        
        Args:
            data: Input DataFrame with time series data
            
        Returns:
            DataFrame with additional engineered features
        \"\"\"
        # Implementation details will go here
        pass
""",
        
        f"{project_root}/src/models/lstm_model.py": """\"\"\"
LSTM Model Module - Implements LSTM neural network for time series forecasting.
\"\"\"
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional

class LSTMModel(nn.Module):
    \"\"\"
    LSTM-based model for time series forecasting.
    \"\"\"
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        \"\"\"
        Initialize the LSTM model.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout probability
        \"\"\"
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        \"\"\"
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        \"\"\"
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_len, hidden_dim)
        
        # Get the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out
""",
        
        f"{project_root}/src/models/transformer_model.py": """\"\"\"
Transformer Model Module - Implements a Transformer-based model for time series forecasting.
\"\"\"
import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, List, Optional

class PositionalEncoding(nn.Module):
    \"\"\"
    Positional encoding for the Transformer model.
    \"\"\"
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        \"\"\"
        Forward pass.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        \"\"\"
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    \"\"\"
    Transformer-based model for time series forecasting.
    \"\"\"
    
    def __init__(self, 
                input_dim: int, 
                d_model: int = 64, 
                nhead: int = 8,
                num_encoder_layers: int = 3,
                dim_feedforward: int = 256,
                dropout: float = 0.1,
                output_dim: int = 1):
        \"\"\"
        Initialize the Transformer model.
        
        Args:
            input_dim: Input dimension (number of features)
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            output_dim: Output dimension
        \"\"\"
        super(TransformerModel, self).__init__()
        
        # Feature embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model
        
    def forward(self, src):
        \"\"\"
        Forward pass through the network.
        
        Args:
            src: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        \"\"\"
        # Change shape to (seq_len, batch_size, input_dim)
        src = src.permute(1, 0, 2)
        
        # Embed features to d_model dimensions
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        output = self.transformer_encoder(src)
        
        # Get the last time step
        output = output[-1]  # Shape: (batch_size, d_model)
        
        # Pass through the output layer
        output = self.output_layer(output)  # Shape: (batch_size, output_dim)
        
        return output
""",
        
        f"{project_root}/src/models/model_trainer.py": """\"\"\"
Model Trainer Module - Handles training of deep learning models for time series forecasting.
\"\"\"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import time
import json
import yaml
from pathlib import Path
import logging

from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .gru_model import GRUModel

class ModelTrainer:
    \"\"\"
    A class for training deep learning models for time series forecasting.
    \"\"\"
    
    def __init__(self, config: Optional[Dict] = None):
        \"\"\"
        Initialize the ModelTrainer.
        
        Args:
            config: Configuration dictionary
        \"\"\"
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def create_model(self, model_type: str, model_params: Dict) -> nn.Module:
        \"\"\"
        Create a model of the specified type.
        
        Args:
            model_type: Type of model ('lstm', 'transformer', 'gru')
            model_params: Parameters for the model
            
        Returns:
            PyTorch model
        \"\"\"
        if model_type.lower() == 'lstm':
            model = LSTMModel(**model_params)
        elif model_type.lower() == 'transformer':
            model = TransformerModel(**model_params)
        elif model_type.lower() == 'gru':
            model = GRUModel(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model = model.to(self.device)
        return model
        
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        \"\"\"
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        \"\"\"
        # Implementation details will go here
        pass
""",
        
        f"{project_root}/src/pipeline/training_pipeline.py": """\"\"\"
Training Pipeline Module - Orchestrates the entire training process.
\"\"\"
import os
import pandas as pd
import numpy as np
import yaml
import torch
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import logging
import json

from ..data.data_collector import DataCollector
from ..data.data_preprocessor import DataPreprocessor
from ..features.feature_engineering import FeatureEngineer
from ..models.model_trainer import ModelTrainer
from ..models.model_evaluator import ModelEvaluator
from ..utils.logger import setup_logger

class TrainingPipeline:
    \"\"\"
    Orchestrates the end-to-end training process, from data collection to model evaluation.
    \"\"\"
    
    def __init__(self, config_path: str):
        \"\"\"
        Initialize the training pipeline.
        
        Args:
            config_path: Path to the configuration file
        \"\"\"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.logger = setup_logger(__name__)
        self.model_dir = self.config.get('model_dir', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def run(self):
        \"\"\"
        Execute the complete training pipeline.
        
        Returns:
            Dictionary with pipeline results including model metrics
        \"\"\"
        self.logger.info("Starting training pipeline")
        
        # Step 1: Collect data
        self.logger.info("Step 1: Collecting data")
        collector = DataCollector(config=self.config.get('data_collection', {}))
        # Implementation details will go here
        
        # Step 2: Preprocess data
        self.logger.info("Step 2: Preprocessing data")
        preprocessor = DataPreprocessor(config=self.config.get('preprocessing', {}))
        # Implementation details will go here
        
        # Step 3: Engineer features
        self.logger.info("Step 3: Engineering features")
        feature_engineer = FeatureEngineer(config=self.config.get('feature_engineering', {}))
        # Implementation details will go here
        
        # Step 4: Train model
        self.logger.info("Step 4: Training model")
        trainer = ModelTrainer(config=self.config.get('model_training', {}))
        # Implementation details will go here
        
        # Step 5: Evaluate model
        self.logger.info("Step 5: Evaluating model")
        evaluator = ModelEvaluator()
        # Implementation details will go here
        
        self.logger.info("Training pipeline completed successfully")
        return {}
""",
        
        f"{project_root}/src/monitoring/model_monitor.py": """\"\"\"
Model Monitor Module - Monitors model performance and detects drift.
\"\"\"
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import json
import yaml
import time
from datetime import datetime, timedelta
import os

class ModelMonitor:
    \"\"\"
    Monitors model performance and data drift in production.
    
    Features:
    - Performance tracking
    - Data drift detection
    - Anomaly detection
    - Alerting
    \"\"\"
    
    def __init__(self, config: Optional[Dict] = None):
        \"\"\"
        Initialize the ModelMonitor.
        
        Args:
            config: Configuration dictionary
        \"\"\"
        self.config = config or {}
        
    def monitor_performance(self, model_id: str, predictions: np.ndarray, actual: np.ndarray):
        \"\"\"
        Monitor model performance metrics.
        
        Args:
            model_id: ID of the model to monitor
            predictions: Model predictions
            actual: Actual values
            
        Returns:
            Dictionary with performance metrics
        \"\"\"
        # Implementation details will go here
        pass
""",
        
        f"{project_root}/src/utils/logger.py": """\"\"\"
Logger Module - Provides logging functionality for the application.
\"\"\"
import logging
import os
from datetime import datetime

def setup_logger(name, log_level=logging.INFO):
    \"\"\"
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level
        
    Returns:
        Configured logger
    \"\"\"
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(f'logs/app_{timestamp}.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger
"""
    }
    
    # Create files
    for file_path, content in modules.items():
        create_file(file_path, content)
    
    print(f"\nProject setup complete! Project created at: {os.path.abspath(project_root)}")
    print("Next steps:")
    print("1. Navigate to the project directory: cd time_series_forecasting_engine")
    print("2. Create a virtual environment: python -m venv venv")
    print("3. Activate the environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Start implementing the modules as per the job requirements")

if __name__ == "__main__":
    setup_project()