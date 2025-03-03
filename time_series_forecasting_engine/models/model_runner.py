"""
LSTM Model Training Runner - Simplified Version

This script handles the training and evaluation of LSTM models for time series forecasting.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime

def setup_logger():
    """Set up a logger for the model training runner."""
    logger = logging.getLogger('model_runner')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

def train_models():
    """Train LSTM models for each symbol."""
    logger = setup_logger()
    logger.info("Starting model training")
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Import the LSTM model - using a direct import since we're in the same directory
    try:
        # Direct import from the same directory
        from lstm_model import LSTMForecaster, ModelTrainer, TimeSeriesDataset, prepare_data_for_model
        logger.info("Successfully imported LSTM model")
    except ImportError as e:
        logger.error(f"Error importing LSTM model: {str(e)}")
        logger.info("Trying alternative import...")
        
        try:
            # Add the current directory to the Python path
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            # Try alternative import
            import lstm_model
            from lstm_model import LSTMForecaster, ModelTrainer, TimeSeriesDataset, prepare_data_for_model
            logger.info("Alternative import successful")
        except ImportError as e2:
            logger.error(f"Alternative import also failed: {str(e2)}")
            print("Please ensure lstm_model.py is in the same directory as this script")
            return
    
    # Set up directories
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    plots_dir = os.path.join(project_root, 'plots')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Working with processed_dir: {processed_dir}")
    logger.info(f"Models will be saved to: {models_dir}")
    
    # Find training data files
    try:
        train_files = [f for f in os.listdir(processed_dir) if f.endswith('_train.parquet')]
        logger.info(f"Found {len(train_files)} training data files")
    except Exception as e:
        logger.error(f"Error finding training files: {str(e)}")
        print(f"Could not find training files in {processed_dir}")
        return
    
    if not train_files:
        logger.error("No training data files found. Run the pipeline first.")
        print(f"No training files found in {processed_dir}")
        return
    
    # Model hyperparameters
    sequence_length = 10
    hidden_dim = 64
    num_layers = 2
    dropout = 0.2
    batch_size = 32
    epochs = 30
    learning_rate = 0.001
    patience = 10
    target_column = 'target_next_close'
    
    # Track overall results
    all_results = []
    
    # Train a model for each symbol
    for train_file in train_files:
        symbol = train_file.split('_')[0]
        logger.info(f"Training model for {symbol}")
        
        # Get validation file
        val_file = train_file.replace('_train.parquet', '_val.parquet')
        val_path = os.path.join(processed_dir, val_file)
        
        if not os.path.exists(val_path):
            logger.error(f"Validation file not found for {symbol}")
            continue
        
        # Prepare data
        train_path = os.path.join(processed_dir, train_file)
        
        try:
            # Load and prepare data
            logger.info(f"Loading training data from {train_path}")
            train_data, target_col, feature_cols = prepare_data_for_model(
                train_path, target_col=target_column, sequence_length=sequence_length
            )
            
            logger.info(f"Loading validation data from {val_path}")
            val_data, _, _ = prepare_data_for_model(
                val_path, target_col=target_column, sequence_length=sequence_length
            )
            
            # Create datasets
            train_dataset = TimeSeriesDataset(
                data=train_data, 
                target_col=target_col, 
                feature_cols=feature_cols, 
                sequence_length=sequence_length
            )
            
            val_dataset = TimeSeriesDataset(
                data=val_data, 
                target_col=target_col, 
                feature_cols=feature_cols, 
                sequence_length=sequence_length
            )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset.to_torch_dataset(), 
                batch_size=batch_size, 
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset.to_torch_dataset(), 
                batch_size=batch_size, 
                shuffle=False
            )
            
            # Initialize model
            input_dim = train_dataset.get_feature_dim()
            logger.info(f"Creating LSTM model with input dim: {input_dim}")
            model = LSTMForecaster(
                input_dim=input_dim, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout
            )
            
            # Initialize trainer
            trainer = ModelTrainer(
                model=model, 
                learning_rate=learning_rate
            )
            
            # Train model
            logger.info(f"Starting training for {symbol}")
            history = trainer.train(
                train_loader=train_loader, 
                val_loader=val_loader, 
                epochs=epochs, 
                patience=patience, 
                model_dir=models_dir
            )
            
            # Plot loss curves
            loss_plot_path = os.path.join(plots_dir, f"{symbol}_loss.png")
            trainer.plot_loss(save_path=loss_plot_path)
            
            # Evaluate on validation set
            val_metrics = trainer.evaluate(val_loader)
            
            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Best validation loss: {history['best_val_loss']:.6f}")
            logger.info(f"Model saved to {history['best_model_path']}")
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    logger.info("Model training completed")

if __name__ == "__main__":
    # Train models
    train_models()