import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def find_latest_training_log(log_pattern="improved_training_log_*.csv"):
    """Find the most recent training log file."""
    log_files = glob.glob(f'/Users/rizkimuhammad/Honours/msa-tf2/weights/{log_pattern}')
    if not log_files:
        raise FileNotFoundError(f"No training log files found matching pattern: {log_pattern}")
    
    # Sort by modification time and get the latest
    latest_log = max(log_files, key=os.path.getmtime)
    return latest_log

def visualize_huber_training(csv_path=None, save_path=None):
    """
    Create comprehensive visualization for Huber loss training.
    
    Args:
        csv_path: Path to training log CSV file. If None, finds latest improved training log.
        save_path: Path to save the visualization. If None, saves to current directory.
    """
    
    # Find the training log file
    if csv_path is None:
        csv_path = find_latest_training_log()
    
    print(f"Loading training data from: {csv_path}")
    
    # Read the training log CSV file
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('MSA-TF2 Improved Training Progress (Huber Loss)', fontsize=16, fontweight='bold')
    
    # Define colors
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Purple
    lr_color = '#F18F01'     # Orange
    huber_color = '#E74C3C'  # Red
    
    # Plot 1: Huber Loss (Training vs Validation)
    axes[0, 0].plot(df['epoch'], df['loss'], label='Training Huber Loss', color=train_color, linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Huber Loss', color=val_color, linewidth=2)
    axes[0, 0].set_title('Huber Loss Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Huber Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add annotation for best validation Huber loss
    best_val_loss_idx = df['val_loss'].idxmin()
    best_val_loss = df.loc[best_val_loss_idx, 'val_loss']
    best_epoch = df.loc[best_val_loss_idx, 'epoch']
    axes[0, 0].annotate(f'Best Val Huber Loss: {best_val_loss:.3f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_val_loss), xytext=(best_epoch+5, best_val_loss+0.1),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 2: MAE (Training vs Validation)
    axes[0, 1].plot(df['epoch'], df['mae'], label='Training MAE', color=train_color, linewidth=2)
    axes[0, 1].plot(df['epoch'], df['val_mae'], label='Validation MAE', color=val_color, linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error Over Time', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add annotation for best validation MAE
    best_val_mae_idx = df['val_mae'].idxmin()
    best_val_mae = df.loc[best_val_mae_idx, 'val_mae']
    best_mae_epoch = df.loc[best_val_mae_idx, 'epoch']
    axes[0, 1].annotate(f'Best Val MAE: {best_val_mae:.3f}\n(Epoch {best_mae_epoch})', 
                       xy=(best_mae_epoch, best_val_mae), xytext=(best_mae_epoch+5, best_val_mae+0.2),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 3: MSE (Training vs Validation)
    axes[1, 0].plot(df['epoch'], df['mse'], label='Training MSE', color=train_color, linewidth=2)
    axes[1, 0].plot(df['epoch'], df['val_mse'], label='Validation MSE', color=val_color, linewidth=2)
    axes[1, 0].set_title('Mean Squared Error Over Time', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add annotation for best validation MSE
    best_val_mse_idx = df['val_mse'].idxmin()
    best_val_mse = df.loc[best_val_mse_idx, 'val_mse']
    best_mse_epoch = df.loc[best_val_mse_idx, 'epoch']
    axes[1, 0].annotate(f'Best Val MSE: {best_val_mse:.3f}\n(Epoch {best_mse_epoch})', 
                       xy=(best_mse_epoch, best_val_mse), xytext=(best_mse_epoch+5, best_val_mse+0.5),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 4: Learning Rate and Training Gaps
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    # Calculate training-validation gaps
    huber_gap = df['val_loss'] - df['loss']
    mae_gap = df['val_mae'] - df['mae']
    
    # Plot learning rate
    line1 = ax1.plot(df['epoch'], df['learning_rate'], label='Learning Rate', color=lr_color, linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Learning Rate', color=lr_color)
    ax1.tick_params(axis='y', labelcolor=lr_color)
    
    # Plot training-validation gaps
    line2 = ax2.plot(df['epoch'], huber_gap, label='Val-Train Gap (Huber)', color=huber_color, linewidth=2, linestyle='-')
    line3 = ax2.plot(df['epoch'], mae_gap, label='Val-Train Gap (MAE)', color='green', linewidth=2, linestyle='--')
    ax2.set_ylabel('Validation - Training Gap', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Add horizontal line at gap = 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate & Overfitting Gaps', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add text box with key statistics
    final_huber_gap = df['val_loss'].iloc[-1] - df['loss'].iloc[-1]
    final_mae_gap = df['val_mae'].iloc[-1] - df['mae'].iloc[-1]
    
    # Find learning rate reduction point
    lr_reduction_epoch = None
    if 'learning_rate' in df.columns:
        lr_changes = df['learning_rate'].diff().abs() > 1e-6
        if lr_changes.any():
            lr_reduction_epoch = df[lr_changes].index[0] + 1
    
    textstr = f'''Key Statistics (Huber Loss Training):
• Best Validation Huber Loss: {best_val_loss:.4f} (Epoch {best_epoch})
• Best Validation MAE: {best_val_mae:.4f} (Epoch {best_mae_epoch})
• Final Training Huber Loss: {df['loss'].iloc[-1]:.4f}
• Final Validation Huber Loss: {df['val_loss'].iloc[-1]:.4f}
• Huber Overfitting Gap: {final_huber_gap:.4f}
• MAE Overfitting Gap: {final_mae_gap:.4f}'''
    
    if lr_reduction_epoch:
        textstr += f'\n• LR Reduction: Epoch {lr_reduction_epoch}'
    
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Save the plot
    if save_path is None:
        save_path = '/Users/rizkimuhammad/Honours/msa-tf2/training_visualization_huber.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Training visualization saved as '{save_path}'")
    print(f"\nTraining Summary (Huber Loss):")
    print(f"• Best validation Huber loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"• Best validation MAE: {best_val_mae:.4f} at epoch {best_mae_epoch}")
    print(f"• Final training Huber loss: {df['loss'].iloc[-1]:.4f}")
    print(f"• Final validation Huber loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"• Huber overfitting gap: {final_huber_gap:.4f}")
    print(f"• MAE overfitting gap: {final_mae_gap:.4f}")
    
    # Show the plot
    plt.show()
    
    return df

def compare_metrics(df):
    """Compare different metrics to understand their relationship."""
    print("\n" + "="*60)
    print("METRIC COMPARISON ANALYSIS")
    print("="*60)
    
    # Find best epochs for each metric
    best_huber_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_mae_epoch = df.loc[df['val_mae'].idxmin(), 'epoch']
    best_mse_epoch = df.loc[df['val_mse'].idxmin(), 'epoch']
    
    print(f"Best validation Huber loss: Epoch {best_huber_epoch}")
    print(f"Best validation MAE: Epoch {best_mae_epoch}")
    print(f"Best validation MSE: Epoch {best_mse_epoch}")
    
    # Check if they align
    if best_huber_epoch == best_mae_epoch:
        print("✅ Huber loss and MAE align perfectly!")
    else:
        print(f"⚠️  Huber loss and MAE differ by {abs(best_huber_epoch - best_mae_epoch)} epochs")
    
    # Calculate correlation between metrics
    huber_mae_corr = df['val_loss'].corr(df['val_mae'])
    huber_mse_corr = df['val_loss'].corr(df['val_mse'])
    mae_mse_corr = df['val_mae'].corr(df['val_mse'])
    
    print(f"\nCorrelation Analysis:")
    print(f"• Huber Loss ↔ MAE: {huber_mae_corr:.4f}")
    print(f"• Huber Loss ↔ MSE: {huber_mse_corr:.4f}")
    print(f"• MAE ↔ MSE: {mae_mse_corr:.4f}")
    
    return {
        'best_huber_epoch': best_huber_epoch,
        'best_mae_epoch': best_mae_epoch,
        'best_mse_epoch': best_mse_epoch,
        'huber_mae_corr': huber_mae_corr,
        'huber_mse_corr': huber_mse_corr,
        'mae_mse_corr': mae_mse_corr
    }

if __name__ == "__main__":
    try:
        # Create visualization
        df = visualize_huber_training()
        
        # Perform metric comparison
        comparison_results = compare_metrics(df)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the improved training script first to generate training logs.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that the training log file exists and has the expected format.")








