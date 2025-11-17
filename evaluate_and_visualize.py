#!/usr/bin/env python3
"""
Comprehensive Evaluation and Visualization Script for MSA Models

This script provides detailed analysis and visualization of model predictions including:
- Prediction quality analysis (scatter, residuals, error distribution)
- Sentiment-range specific performance
- Classification metrics (binary and 7-class)
- Model comparison capabilities
- Failure analysis
- Statistical testing

All visualizations are carefully designed to avoid overlapping labels and elements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
)
import h5py
import argparse
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_predictions_from_model(model_path, data_dir, split='test'):
    """
    Load predictions from a saved model.
    
    Args:
        model_path: Path to saved model (.h5 file)
        data_dir: Directory containing data
        split: 'test', 'valid', or 'train'
    
    Returns:
        y_true, y_pred: Arrays of true and predicted values
    """
    import tensorflow as tf
    from utils.data_loader import make_dataset
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading {split} dataset from: {data_dir}")
    dataset = make_dataset(data_dir, split=split, batch_size=32)
    
    print("Generating predictions...")
    y_true_list = []
    y_pred_list = []
    
    for batch_data, batch_labels in dataset:
        predictions = model.predict_on_batch(batch_data)
        y_true_list.append(batch_labels.numpy())
        y_pred_list.append(predictions)
    
    y_true = np.concatenate(y_true_list, axis=0).flatten()
    y_pred = np.concatenate(y_pred_list, axis=0).flatten()
    
    print(f"Loaded {len(y_true)} predictions")
    return y_true, y_pred


def load_predictions_from_arrays(y_true_path, y_pred_path):
    """
    Load predictions from numpy arrays or h5 files.
    
    Args:
        y_true_path: Path to true labels
        y_pred_path: Path to predictions
    
    Returns:
        y_true, y_pred: Arrays of true and predicted values
    """
    # Try different formats
    if y_true_path.endswith('.npy'):
        y_true = np.load(y_true_path).flatten()
        y_pred = np.load(y_pred_path).flatten()
    elif y_true_path.endswith('.h5'):
        with h5py.File(y_true_path, 'r') as f:
            y_true = f['data'][:].flatten()
        with h5py.File(y_pred_path, 'r') as f:
            y_pred = f['data'][:].flatten()
    else:
        raise ValueError("Unsupported file format. Use .npy or .h5")
    
    print(f"Loaded {len(y_true)} predictions")
    return y_true, y_pred


def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Pearson and Spearman correlations
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    
    # Error statistics
    errors = y_pred - y_true
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_ae = np.median(np.abs(errors))
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_ae': median_ae,
        'max_error': np.max(np.abs(errors)),
        'min_error': np.min(np.abs(errors))
    }
    
    return metrics


def get_binary_labels(y, threshold=0.0):
    """Convert continuous sentiment to binary labels."""
    return (y > threshold).astype(int)


def get_7class_labels(y):
    """Convert continuous sentiment to 7-class labels [-3, -2, -1, 0, 1, 2, 3]."""
    # Discretize into 7 bins
    bins = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    labels = [-3, -2, -1, 0, 1, 2, 3]
    return pd.cut(y, bins=bins, labels=labels).astype(int)


def calculate_classification_metrics(y_true, y_pred):
    """Calculate classification metrics for binary and 7-class."""
    metrics = {}
    
    # Binary classification
    y_true_binary = get_binary_labels(y_true)
    y_pred_binary = get_binary_labels(y_pred)
    
    metrics['binary_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    metrics['binary_f1'] = f1_score(y_true_binary, y_pred_binary, average='binary')
    
    # 7-class classification
    y_true_7class = get_7class_labels(y_true)
    y_pred_7class = get_7class_labels(y_pred)
    
    metrics['7class_accuracy'] = accuracy_score(y_true_7class, y_pred_7class)
    metrics['7class_f1_macro'] = f1_score(y_true_7class, y_pred_7class, average='macro')
    metrics['7class_f1_weighted'] = f1_score(y_true_7class, y_pred_7class, average='weighted')
    
    return metrics


def plot_prediction_scatter(y_true, y_pred, metrics, save_path=None):
    """
    Create scatter plot of predictions vs actual values.
    No overlapping labels guaranteed.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate errors for color mapping
    errors = np.abs(y_pred - y_true)
    
    # Create scatter plot with color gradient based on error
    scatter = ax.scatter(y_true, y_pred, c=errors, cmap='YlOrRd', 
                        alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'b--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], p([min_val, max_val]), 
            'r-', linewidth=2, label=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})', alpha=0.7)
    
    # Labels and title with extra padding
    ax.set_xlabel('True Sentiment', fontweight='bold', labelpad=10)
    ax.set_ylabel('Predicted Sentiment', fontweight='bold', labelpad=10)
    ax.set_title('Prediction Quality: True vs Predicted Values', 
                 fontweight='bold', pad=20, fontsize=13)
    
    # Add colorbar with label
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Absolute Error', rotation=270, labelpad=20, fontweight='bold')
    
    # Add metrics text box - positioned carefully to avoid overlap
    textstr = f'MAE: {metrics["mae"]:.4f}\n'
    textstr += f'RMSE: {metrics["rmse"]:.4f}\n'
    textstr += f'R²: {metrics["r2"]:.4f}\n'
    textstr += f'Pearson r: {metrics["pearson_r"]:.4f}\n'
    textstr += f'Spearman ρ: {metrics["spearman_r"]:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Legend positioned to not overlap with data
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='black')
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout(pad=1.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_residual_analysis(y_true, y_pred, save_path=None):
    """
    Create residual plot to identify systematic biases.
    No overlapping elements.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 1. Residuals vs Predicted Values
    ax = axes[0, 0]
    ax.scatter(y_pred, errors, alpha=0.5, s=25, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Predicted Values', fontweight='bold', labelpad=8)
    ax.set_ylabel('Residuals (Pred - True)', fontweight='bold', labelpad=8)
    ax.set_title('Residual Plot', fontweight='bold', pad=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 2. Error Distribution
    ax = axes[0, 1]
    n, bins, patches = ax.hist(errors, bins=50, edgecolor='black', 
                                alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=np.mean(errors), color='orange', linestyle='-', 
               linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    ax.set_xlabel('Prediction Error', fontweight='bold', labelpad=8)
    ax.set_ylabel('Frequency', fontweight='bold', labelpad=8)
    ax.set_title('Error Distribution', fontweight='bold', pad=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Q-Q Plot (Normal distribution check)
    ax = axes[1, 0]
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3)
    # Adjust labels for Q-Q plot
    ax.set_xlabel('Theoretical Quantiles', fontweight='bold', labelpad=8)
    ax.set_ylabel('Sample Quantiles', fontweight='bold', labelpad=8)
    
    # 4. Absolute Error vs True Values
    ax = axes[1, 1]
    ax.scatter(y_true, abs_errors, alpha=0.5, s=25, 
               edgecolors='black', linewidth=0.5, color='coral')
    # Add trend line
    z = np.polyfit(y_true, abs_errors, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(y_true.min(), y_true.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r-', linewidth=2, 
            label='Trend (polynomial)', alpha=0.7)
    ax.set_xlabel('True Values', fontweight='bold', labelpad=8)
    ax.set_ylabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_title('Error Magnitude vs True Values', fontweight='bold', pad=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Analysis', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_sentiment_range_analysis(y_true, y_pred, save_path=None):
    """
    Analyze performance across different sentiment ranges.
    Carefully positioned to avoid overlaps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define sentiment bins
    bin_edges = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    bin_labels = ['Very Neg\n(<-2)', 'Negative\n(-2 to -1)', 'Sl. Neg\n(-1 to 0)', 
                  'Sl. Pos\n(0 to 1)', 'Positive\n(1 to 2)', 'Very Pos\n(>2)']
    
    # Assign bins
    true_bins = pd.cut(y_true, bins=bin_edges, labels=bin_labels)
    
    # Calculate metrics per bin
    bin_data = []
    for bin_label in bin_labels:
        mask = (true_bins == bin_label)
        if mask.sum() > 0:
            bin_true = y_true[mask]
            bin_pred = y_pred[mask]
            bin_data.append({
                'bin': bin_label,
                'count': mask.sum(),
                'mae': mean_absolute_error(bin_true, bin_pred),
                'rmse': np.sqrt(mean_squared_error(bin_true, bin_pred)),
                'r2': r2_score(bin_true, bin_pred) if mask.sum() > 1 else 0
            })
    
    df_bins = pd.DataFrame(bin_data)
    
    # 1. MAE per Sentiment Range
    ax = axes[0, 0]
    bars = ax.bar(range(len(df_bins)), df_bins['mae'], 
                  color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(df_bins)))
    ax.set_xticklabels(df_bins['bin'], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('MAE', fontweight='bold', labelpad=8)
    ax.set_title('MAE by Sentiment Range', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_bins['mae'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Sample Count per Range
    ax = axes[0, 1]
    bars = ax.bar(range(len(df_bins)), df_bins['count'], 
                  color='coral', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(df_bins)))
    ax.set_xticklabels(df_bins['bin'], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Sample Count', fontweight='bold', labelpad=8)
    ax.set_title('Sample Distribution', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, df_bins['count']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(df_bins['count'])*0.01,
                f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    # 3. RMSE per Range
    ax = axes[1, 0]
    bars = ax.bar(range(len(df_bins)), df_bins['rmse'], 
                  color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(df_bins)))
    ax.set_xticklabels(df_bins['bin'], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('RMSE', fontweight='bold', labelpad=8)
    ax.set_title('RMSE by Sentiment Range', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, df_bins['rmse']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. R² per Range
    ax = axes[1, 1]
    bars = ax.bar(range(len(df_bins)), df_bins['r2'], 
                  color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(df_bins)))
    ax.set_xticklabels(df_bins['bin'], rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('R² Score', fontweight='bold', labelpad=8)
    ax.set_title('R² by Sentiment Range', fontweight='bold', pad=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, df_bins['r2']):
        height = bar.get_height()
        y_pos = height + 0.02 if height > 0 else height - 0.05
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=8)
    
    plt.suptitle('Performance Analysis Across Sentiment Ranges', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_classification_metrics(y_true, y_pred, save_path=None):
    """
    Visualize binary and 7-class classification performance.
    All elements carefully positioned.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    
    # Binary classification
    y_true_binary = get_binary_labels(y_true)
    y_pred_binary = get_binary_labels(y_pred)
    
    # 7-class classification
    y_true_7class = get_7class_labels(y_true)
    y_pred_7class = get_7class_labels(y_pred)
    
    # 1. Binary Confusion Matrix
    ax = fig.add_subplot(gs[0, 0])
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'}, ax=ax,
                annot_kws={'size': 12})
    ax.set_xlabel('Predicted', fontweight='bold', labelpad=10)
    ax.set_ylabel('True', fontweight='bold', labelpad=10)
    ax.set_title('Binary Classification\nConfusion Matrix', 
                 fontweight='bold', pad=15)
    
    # 2. Binary Metrics Bar Chart
    ax = fig.add_subplot(gs[0, 1])
    binary_acc = accuracy_score(y_true_binary, y_pred_binary)
    binary_f1 = f1_score(y_true_binary, y_pred_binary)
    prec, rec, _, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, 
                                                        average='binary')
    
    metrics_data = {'Accuracy': binary_acc, 'Precision': prec, 
                   'Recall': rec, 'F1-Score': binary_f1}
    bars = ax.bar(metrics_data.keys(), metrics_data.values(), 
                  color=['steelblue', 'coral', 'lightgreen', 'mediumpurple'],
                  edgecolor='black', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score', fontweight='bold', labelpad=8)
    ax.set_title('Binary Classification Metrics', fontweight='bold', pad=15)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # 3. Binary Class Distribution
    ax = fig.add_subplot(gs[0, 2])
    true_counts = np.bincount(y_true_binary)
    pred_counts = np.bincount(y_pred_binary)
    
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, true_counts, width, label='True', 
                   color='steelblue', edgecolor='black', alpha=0.7)
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', 
                   color='coral', edgecolor='black', alpha=0.7)
    
    ax.set_xlabel('Class', fontweight='bold', labelpad=8)
    ax.set_ylabel('Count', fontweight='bold', labelpad=8)
    ax.set_title('Binary Class Distribution', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(true_counts.max(), pred_counts.max())*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 4. 7-Class Confusion Matrix
    ax = fig.add_subplot(gs[1, :2])
    cm_7class = confusion_matrix(y_true_7class, y_pred_7class)
    sns.heatmap(cm_7class, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=[-3, -2, -1, 0, 1, 2, 3],
                yticklabels=[-3, -2, -1, 0, 1, 2, 3],
                cbar_kws={'label': 'Count'}, ax=ax,
                annot_kws={'size': 10})
    ax.set_xlabel('Predicted Sentiment Class', fontweight='bold', labelpad=10)
    ax.set_ylabel('True Sentiment Class', fontweight='bold', labelpad=10)
    ax.set_title('7-Class Sentiment Classification\nConfusion Matrix', 
                 fontweight='bold', pad=15)
    
    # 5. 7-Class Metrics
    ax = fig.add_subplot(gs[1, 2])
    class_7_acc = accuracy_score(y_true_7class, y_pred_7class)
    class_7_f1_macro = f1_score(y_true_7class, y_pred_7class, average='macro')
    class_7_f1_weighted = f1_score(y_true_7class, y_pred_7class, average='weighted')
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_true_7class, y_pred_7class, average=None)
    
    metrics_7class = {
        'Accuracy': class_7_acc,
        'F1-Macro': class_7_f1_macro,
        'F1-Weighted': class_7_f1_weighted
    }
    
    bars = ax.bar(metrics_7class.keys(), metrics_7class.values(),
                  color=['steelblue', 'coral', 'lightgreen'],
                  edgecolor='black', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Score', fontweight='bold', labelpad=8)
    ax.set_title('7-Class Classification Metrics', fontweight='bold', pad=15)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.suptitle('Classification Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_error_analysis(y_true, y_pred, top_k=10, save_path=None):
    """
    Analyze worst predictions and error patterns.
    Carefully laid out to prevent overlaps.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # 1. Top K Worst Predictions
    ax = axes[0, 0]
    worst_indices = np.argsort(abs_errors)[-top_k:][::-1]
    
    x_pos = np.arange(top_k)
    bars = ax.barh(x_pos, abs_errors[worst_indices], color='crimson', 
                   edgecolor='black', alpha=0.7)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f'#{i}\nT:{y_true[i]:.2f}\nP:{y_pred[i]:.2f}' 
                        for i in worst_indices], fontsize=8)
    ax.set_xlabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_title(f'Top {top_k} Worst Predictions', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add error values
    for i, (bar, idx) in enumerate(zip(bars, worst_indices)):
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                f'{abs_errors[idx]:.3f}', ha='left', va='center', fontsize=8)
    
    # 2. Error Percentiles
    ax = axes[0, 1]
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = [np.percentile(abs_errors, p) for p in percentiles]
    
    bars = ax.bar([f'{p}th' for p in percentiles], percentile_values,
                  color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_ylabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_xlabel('Percentile', fontweight='bold', labelpad=8)
    ax.set_title('Error Percentiles', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, percentile_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Error by Prediction Confidence (binned predictions)
    ax = axes[1, 0]
    pred_bins = pd.cut(y_pred, bins=10, duplicates='drop')
    bin_centers = []
    bin_errors = []
    
    for bin_label in pred_bins.cat.categories:
        mask = (pred_bins == bin_label)
        if mask.sum() > 0:
            bin_centers.append(bin_label.mid)
            bin_errors.append(np.mean(abs_errors[mask]))
    
    ax.plot(bin_centers, bin_errors, marker='o', linewidth=2, 
            markersize=8, color='steelblue', markerfacecolor='coral',
            markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel('Predicted Value (binned)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Mean Absolute Error', fontweight='bold', labelpad=8)
    ax.set_title('Error vs Predicted Value Range', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative Error Distribution
    ax = axes[1, 1]
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    
    ax.plot(sorted_errors, cumulative, linewidth=2, color='steelblue')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, 
               label='50% of samples')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, 
               label='90% of samples')
    
    # Add annotations for key percentiles
    for pct in [50, 90]:
        idx = int(len(sorted_errors) * pct / 100)
        error_val = sorted_errors[idx]
        ax.plot(error_val, pct, 'ro', markersize=8)
        ax.annotate(f'{pct}%: {error_val:.3f}', 
                   xy=(error_val, pct), 
                   xytext=(error_val + 0.3, pct - 5),
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    ax.set_xlabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_ylabel('Cumulative Percentage (%)', fontweight='bold', labelpad=8)
    ax.set_title('Cumulative Error Distribution', fontweight='bold', pad=12)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error Analysis and Failure Patterns', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def compare_models(model1_name, y_pred1, model2_name, y_pred2, y_true, save_path=None):
    """
    Compare two models side by side.
    All elements positioned to avoid overlap.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    errors1 = np.abs(y_pred1 - y_true)
    errors2 = np.abs(y_pred2 - y_true)
    
    # 1. MAE Comparison
    ax = axes[0, 0]
    mae1 = mean_absolute_error(y_true, y_pred1)
    mae2 = mean_absolute_error(y_true, y_pred2)
    
    bars = ax.bar([model1_name, model2_name], [mae1, mae2],
                  color=['steelblue', 'coral'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('MAE', fontweight='bold', labelpad=8)
    ax.set_title('Mean Absolute Error', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values and improvement
    for i, (bar, val) in enumerate(zip(bars, [mae1, mae2])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    improvement = ((mae1 - mae2) / mae1) * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:.2f}%', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightcoral', 
                     alpha=0.7), fontsize=9)
    
    # 2. RMSE Comparison
    ax = axes[0, 1]
    rmse1 = np.sqrt(mean_squared_error(y_true, y_pred1))
    rmse2 = np.sqrt(mean_squared_error(y_true, y_pred2))
    
    bars = ax.bar([model1_name, model2_name], [rmse1, rmse2],
                  color=['lightgreen', 'mediumpurple'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('RMSE', fontweight='bold', labelpad=8)
    ax.set_title('Root Mean Squared Error', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [rmse1, rmse2]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Correlation Comparison
    ax = axes[0, 2]
    r1, _ = stats.pearsonr(y_true, y_pred1)
    r2, _ = stats.pearsonr(y_true, y_pred2)
    
    bars = ax.bar([model1_name, model2_name], [r1, r2],
                  color=['gold', 'lightcoral'], edgecolor='black', alpha=0.7)
    ax.set_ylabel('Pearson Correlation', fontweight='bold', labelpad=8)
    ax.set_title('Correlation with Ground Truth', fontweight='bold', pad=12)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [r1, r2]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Error Distribution Comparison
    ax = axes[1, 0]
    ax.hist(errors1, bins=40, alpha=0.6, label=model1_name, 
            color='steelblue', edgecolor='black')
    ax.hist(errors2, bins=40, alpha=0.6, label=model2_name, 
            color='coral', edgecolor='black')
    ax.set_xlabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_ylabel('Frequency', fontweight='bold', labelpad=8)
    ax.set_title('Error Distribution Comparison', fontweight='bold', pad=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Box Plot Comparison
    ax = axes[1, 1]
    bp = ax.boxplot([errors1, errors2], labels=[model1_name, model2_name],
                    patch_artist=True, widths=0.6,
                    boxprops=dict(alpha=0.7, linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Absolute Error', fontweight='bold', labelpad=8)
    ax.set_title('Error Distribution (Box Plot)', fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistical Test Results
    ax = axes[1, 2]
    ax.axis('off')
    
    # Perform statistical tests
    t_stat, p_value_t = stats.ttest_rel(errors1, errors2)
    w_stat, p_value_w = stats.wilcoxon(errors1, errors2)
    
    # Calculate additional stats
    mean_diff = np.mean(errors1) - np.mean(errors2)
    median_diff = np.median(errors1) - np.median(errors2)
    
    # Create text summary
    text_content = f"Statistical Comparison\n" + "="*40 + "\n\n"
    text_content += f"Mean Error Difference:\n  {mean_diff:.4f}\n\n"
    text_content += f"Median Error Difference:\n  {median_diff:.4f}\n\n"
    text_content += f"Paired T-Test:\n  t-statistic: {t_stat:.4f}\n  p-value: {p_value_t:.4e}\n\n"
    text_content += f"Wilcoxon Test:\n  w-statistic: {w_stat:.1f}\n  p-value: {p_value_w:.4e}\n\n"
    
    if p_value_t < 0.05:
        text_content += "Difference is statistically\nsignificant (p < 0.05)"
        bbox_color = 'lightgreen'
    else:
        text_content += "No significant difference\n(p >= 0.05)"
        bbox_color = 'lightcoral'
    
    ax.text(0.5, 0.5, text_content, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=bbox_color, alpha=0.7, 
                     edgecolor='black', linewidth=2),
            family='monospace')
    
    plt.suptitle(f'Model Comparison: {model1_name} vs {model2_name}', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], pad=2.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def generate_comprehensive_report(y_true, y_pred, model_name, output_dir):
    """
    Generate all visualizations and a summary report.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Name of the model
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*70)
    print(f"Generating Comprehensive Evaluation Report: {model_name}")
    print("="*70)
    
    # Calculate all metrics
    print("\nCalculating metrics...")
    reg_metrics = calculate_regression_metrics(y_true, y_pred)
    class_metrics = calculate_classification_metrics(y_true, y_pred)
    
    # Combine metrics
    all_metrics = {**reg_metrics, **class_metrics}
    
    # Print metrics summary
    print("\n" + "-"*70)
    print("REGRESSION METRICS")
    print("-"*70)
    print(f"MAE:           {reg_metrics['mae']:.4f}")
    print(f"MSE:           {reg_metrics['mse']:.4f}")
    print(f"RMSE:          {reg_metrics['rmse']:.4f}")
    print(f"R²:            {reg_metrics['r2']:.4f}")
    print(f"Pearson r:     {reg_metrics['pearson_r']:.4f} (p={reg_metrics['pearson_p']:.4e})")
    print(f"Spearman ρ:    {reg_metrics['spearman_r']:.4f} (p={reg_metrics['spearman_p']:.4e})")
    print(f"Mean Error:    {reg_metrics['mean_error']:.4f}")
    print(f"Std Error:     {reg_metrics['std_error']:.4f}")
    print(f"Median AE:     {reg_metrics['median_ae']:.4f}")
    
    print("\n" + "-"*70)
    print("CLASSIFICATION METRICS")
    print("-"*70)
    print(f"Binary Accuracy:        {class_metrics['binary_accuracy']:.4f}")
    print(f"Binary F1:              {class_metrics['binary_f1']:.4f}")
    print(f"7-Class Accuracy:       {class_metrics['7class_accuracy']:.4f}")
    print(f"7-Class F1 (Macro):     {class_metrics['7class_f1_macro']:.4f}")
    print(f"7-Class F1 (Weighted):  {class_metrics['7class_f1_weighted']:.4f}")
    
    # Generate visualizations
    print("\n" + "-"*70)
    print("Generating visualizations...")
    print("-"*70)
    
    # 1. Prediction Scatter
    print("1/5: Prediction scatter plot...")
    plot_prediction_scatter(
        y_true, y_pred, reg_metrics,
        save_path=os.path.join(output_dir, f'{model_name}_prediction_scatter_{timestamp}.png')
    )
    plt.close()
    
    # 2. Residual Analysis
    print("2/5: Residual analysis...")
    plot_residual_analysis(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f'{model_name}_residual_analysis_{timestamp}.png')
    )
    plt.close()
    
    # 3. Sentiment Range Analysis
    print("3/5: Sentiment range analysis...")
    plot_sentiment_range_analysis(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f'{model_name}_sentiment_ranges_{timestamp}.png')
    )
    plt.close()
    
    # 4. Classification Metrics
    print("4/5: Classification metrics...")
    plot_classification_metrics(
        y_true, y_pred,
        save_path=os.path.join(output_dir, f'{model_name}_classification_{timestamp}.png')
    )
    plt.close()
    
    # 5. Error Analysis
    print("5/5: Error analysis...")
    plot_error_analysis(
        y_true, y_pred, top_k=10,
        save_path=os.path.join(output_dir, f'{model_name}_error_analysis_{timestamp}.png')
    )
    plt.close()
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, f'{model_name}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, f'{model_name}_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"COMPREHENSIVE EVALUATION REPORT: {model_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write("DATASET STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of samples: {len(y_true)}\n")
        f.write(f"True values - Mean: {np.mean(y_true):.4f}, Std: {np.std(y_true):.4f}\n")
        f.write(f"True values - Min: {np.min(y_true):.4f}, Max: {np.max(y_true):.4f}\n")
        f.write(f"Pred values - Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}\n")
        f.write(f"Pred values - Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}\n\n")
        
        f.write("REGRESSION METRICS\n")
        f.write("-"*70 + "\n")
        for key, value in reg_metrics.items():
            f.write(f"{key:20s}: {value:.6f}\n")
        
        f.write("\nCLASSIFICATION METRICS\n")
        f.write("-"*70 + "\n")
        for key, value in class_metrics.items():
            f.write(f"{key:25s}: {value:.6f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("VISUALIZATIONS GENERATED\n")
        f.write("-"*70 + "\n")
        f.write(f"1. Prediction scatter: {model_name}_prediction_scatter_{timestamp}.png\n")
        f.write(f"2. Residual analysis: {model_name}_residual_analysis_{timestamp}.png\n")
        f.write(f"3. Sentiment ranges: {model_name}_sentiment_ranges_{timestamp}.png\n")
        f.write(f"4. Classification: {model_name}_classification_{timestamp}.png\n")
        f.write(f"5. Error analysis: {model_name}_error_analysis_{timestamp}.png\n")
    
    print(f"Report saved to: {report_path}")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Model Evaluation and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from saved model
  python evaluate_and_visualize.py --model weights/model.h5 --data ./data --name MSA_SeqLevel
  
  # Evaluate from prediction arrays
  python evaluate_and_visualize.py --y_true test_true.npy --y_pred test_pred.npy --name MyModel
  
  # Compare two models
  python evaluate_and_visualize.py --compare --model1 model1.h5 --model2 model2.h5 --data ./data
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model', type=str, 
                            help='Path to saved model (.h5)')
    input_group.add_argument('--y_pred', type=str,
                            help='Path to predictions (.npy or .h5)')
    
    parser.add_argument('--y_true', type=str,
                       help='Path to ground truth (.npy or .h5)')
    parser.add_argument('--data', type=str, default='./data',
                       help='Data directory (required if using --model)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Data split to evaluate (default: test)')
    
    # Model comparison
    parser.add_argument('--compare', action='store_true',
                       help='Compare two models')
    parser.add_argument('--model1', type=str,
                       help='First model for comparison')
    parser.add_argument('--model2', type=str,
                       help='Second model for comparison')
    parser.add_argument('--name1', type=str, default='Model 1',
                       help='Name for first model')
    parser.add_argument('--name2', type=str, default='Model 2',
                       help='Name for second model')
    
    # Output options
    parser.add_argument('--name', type=str, default='Model',
                       help='Model name for reports')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.compare:
        # Compare two models
        if not args.model1 or not args.model2:
            parser.error("--compare requires --model1 and --model2")
        
        print("\n" + "="*70)
        print("MODEL COMPARISON MODE")
        print("="*70)
        
        # Load predictions from both models
        y_true1, y_pred1 = load_predictions_from_model(args.model1, args.data, args.split)
        y_true2, y_pred2 = load_predictions_from_model(args.model2, args.data, args.split)
        
        # Verify they use the same ground truth
        if not np.allclose(y_true1, y_true2):
            print("Warning: Models evaluated on different ground truth!")
        
        # Generate individual reports
        print(f"\nGenerating report for {args.name1}...")
        generate_comprehensive_report(y_true1, y_pred1, args.name1, args.output)
        
        print(f"\nGenerating report for {args.name2}...")
        generate_comprehensive_report(y_true2, y_pred2, args.name2, args.output)
        
        # Generate comparison plot
        print("\nGenerating comparison plot...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compare_models(
            args.name1, y_pred1,
            args.name2, y_pred2,
            y_true1,
            save_path=os.path.join(args.output, f'comparison_{timestamp}.png')
        )
        plt.close()
        
    else:
        # Single model evaluation
        if args.model:
            y_true, y_pred = load_predictions_from_model(args.model, args.data, args.split)
        elif args.y_pred and args.y_true:
            y_true, y_pred = load_predictions_from_arrays(args.y_true, args.y_pred)
        else:
            parser.error("Either --model or both --y_true and --y_pred required")
        
        # Generate comprehensive report
        generate_comprehensive_report(y_true, y_pred, args.name, args.output)


if __name__ == '__main__':
    main()

