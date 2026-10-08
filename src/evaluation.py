"""
Evaluation, Metrics, and Plotting Utilities for GDI Prediction Replication
Stanford CS 229: Automated Identification of Gait Abnormalities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_regression_metrics(y_true, y_pred):
    """
    Computes RMSE, MAE, R2 score, Pearson correlation, and explained variance.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    mean_err = np.mean(y_pred - y_true)
    std_err = np.std(y_pred - y_true)
    
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'Pearson_r': float(corr),
        'Mean_Error': float(mean_err),
        'Std_Error': float(std_err),
        'Count': int(len(y_true))
    }

def plot_learning_curves(history_data, save_path=None):
    """
    Plots the training and validation learning curves (Loss and MAE) across epochs,
    replicating Figure 3 (Top) of CS229 Report 31.
    """
    epochs = history_data['epoch']
    loss = history_data['loss']
    val_loss = history_data['val_loss']
    mae = history_data['mae']
    val_mae = history_data['val_mae']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve (MSE)
    ax1.plot(epochs, loss, label='Training Loss (MSE)', color='#1f77b4', lw=2)
    ax1.plot(epochs, val_loss, label='Validation Loss (MSE)', color='#ff7f0e', lw=2)
    ax1.set_title('GDI-Net Learning Curve (Mean Squared Error)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MSE)', fontsize=11)
    ax1.set_ylim(0, max(val_loss[1:]) * 1.1)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # MAE curve
    ax2.plot(epochs, mae, label='Training MAE', color='#2ca02c', lw=2)
    ax2.plot(epochs, val_mae, label='Validation MAE', color='#d62728', lw=2)
    ax2.set_title('GDI-Net Mean Absolute Error over Epochs', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Mean Absolute Error', fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_regression_performance(y_true, y_pred, title="GDI-Net: True vs. Predicted GDI", save_path=None):
    """
    Plots the True GDI vs. Predicted GDI regression scatter plot with identity line
    and error bounds, replicating Figure 3 (Middle) of CS229 Report 31.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    metrics = compute_regression_metrics(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter points
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors='navy', facecolors='#4b9cd3', s=45, label='Validation Clips (N=733)')
    
    # Ideal line y = x
    min_val = min(y_true.min(), y_pred.min()) - 5
    max_val = max(y_true.max(), y_pred.max()) + 5
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = x)')
    
    # +/- 1 RMSE bounds
    ax.fill_between([min_val, max_val], 
                    [min_val - metrics['RMSE'], max_val - metrics['RMSE']], 
                    [min_val + metrics['RMSE'], max_val + metrics['RMSE']], 
                    color='red', alpha=0.1, label=f'±1 RMSE Band (±{metrics["RMSE"]:.2f})')
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel('True Physician GDI Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted GDI-Net Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nRMSE = {metrics["RMSE"]:.2f} | MAE = {metrics["MAE"]:.2f} | R² = {metrics["R2"]:.2f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_residuals_distribution(y_true, y_pred, save_path=None):
    """
    Plots residual error distribution (y_pred - y_true).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = y_pred - y_true
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=35, color='#348abd', edgecolor='black', alpha=0.8, density=True)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax.axvline(np.mean(residuals), color='black', linestyle='-', lw=1.5, label=f'Mean Error: {np.mean(residuals):.2f}')
    
    ax.set_title('GDI Prediction Residual Error Distribution (Predicted - True)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Error (GDI points)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def generate_paper_comparison_table():
    """
    Generates comparison DataFrame reproducing Table 1 from CS229 Report 31.
    """
    data = [
        {"Model": "Guess Mean (Zero Rule)", "Input Type": "Frame", "Training RMSE": 13.7, "Validation RMSE": 13.7, "Notes": "Baseline predicting mean GDI"},
        {"Model": "Linear Regression", "Input Type": "Frame", "Training RMSE": 0.0, "Validation RMSE": 13.0, "Notes": "Overfit on flattened pixels"},
        {"Model": "VGG-16 (Pretrained)", "Input Type": "Frame", "Training RMSE": 10.1, "Validation RMSE": 9.5, "Notes": "Frozen ImageNet weights, cropped 224x224"},
        {"Model": "VGG-16 (Trained)", "Input Type": "Frame", "Training RMSE": 11.6, "Validation RMSE": 11.4, "Notes": "Trained from scratch, cropped 224x224"},
        {"Model": "Custom 2D CNN (ANet)", "Input Type": "Frame", "Training RMSE": 8.1, "Validation RMSE": 8.2, "Notes": "6-layer CNN on whole 480x640 frame"},
        {"Model": "CNN + 1D-CNN", "Input Type": "Video (10 frames)", "Training RMSE": 4.2, "Validation RMSE": 11.3, "Notes": "Temporal 1D convolution over frames"},
        {"Model": "CNN + LSTM (GDI-Net)", "Input Type": "Video (10 frames)", "Training RMSE": 1.4, "Validation RMSE": 4.4, "Notes": "Best Performer (R² = 0.86)"}
    ]
    return pd.DataFrame(data)
