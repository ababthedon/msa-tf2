import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

def find_latest_training_log(log_pattern="*training_log_*.csv", weights_dir="/Users/rizkimuhammad/Honours/msa-tf2/weights"):
    """Find the most recent training log file."""
    log_files = glob.glob(os.path.join(weights_dir, log_pattern))
    if not log_files:
        raise FileNotFoundError(f"No training log files found matching pattern: {log_pattern}")
    
    # Sort by modification time and get the latest
    latest_log = max(log_files, key=os.path.getmtime)
    return latest_log

def detect_training_type(df):
    """Detect the type of training based on available columns and values."""
    training_type = "unknown"
    
    # Check for Huber loss (delta parameter in loss function)
    if 'loss' in df.columns and 'val_loss' in df.columns:
        # Check if this looks like Huber loss by examining the values
        # Huber loss typically has different scale than MAE
        loss_scale = df['loss'].mean()
        mae_scale = df['mae'].mean() if 'mae' in df.columns else 0
        
        if abs(loss_scale - mae_scale) > 0.1:  # Different scales suggest Huber loss
            training_type = "improved_huber"
        else:
            training_type = "original_mae"
    
    # Check for specific patterns in column names or values
    if 'learning_rate' in df.columns:
        lr_values = df['learning_rate'].unique()
        if len(lr_values) > 1:  # Learning rate schedule suggests improved training
            training_type = "improved_huber"
    
    return training_type

def create_comprehensive_visualization(df, training_type="auto", save_path=None):
    """
    Create comprehensive visualization for any training type.
    
    Args:
        df: Training data DataFrame
        training_type: Type of training ("original_mae", "improved_huber", "balanced", "auto")
        save_path: Path to save the visualization
    """
    
    if training_type == "auto":
        training_type = detect_training_type(df)
    
    print(f"Detected training type: {training_type}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Set title based on training type
    title_map = {
        "original_mae": "MSA-TF2 Original Training Progress (MAE Loss)",
        "improved_huber": "MSA-TF2 Improved Training Progress (Huber Loss)",
        "balanced": "MSA-TF2 Balanced Training Progress",
        "unknown": "MSA-TF2 Training Progress"
    }
    fig.suptitle(title_map.get(training_type, "MSA-TF2 Training Progress"), 
                 fontsize=16, fontweight='bold')
    
    # Define colors
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Purple
    lr_color = '#F18F01'     # Orange
    huber_color = '#E74C3C'  # Red
    
    # Plot 1: Loss (Training vs Validation)
    loss_label = "Huber Loss" if training_type == "improved_huber" else "Loss"
    axes[0, 0].plot(df['epoch'], df['loss'], label=f'Training {loss_label}', color=train_color, linewidth=2)
    axes[0, 0].plot(df['epoch'], df['val_loss'], label=f'Validation {loss_label}', color=val_color, linewidth=2)
    axes[0, 0].set_title(f'{loss_label} Over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel(loss_label)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add annotation for best validation loss
    best_val_loss_idx = df['val_loss'].idxmin()
    best_val_loss = df.loc[best_val_loss_idx, 'val_loss']
    best_epoch = df.loc[best_val_loss_idx, 'epoch']
    axes[0, 0].annotate(f'Best Val {loss_label}: {best_val_loss:.3f}\n(Epoch {best_epoch})', 
                       xy=(best_epoch, best_val_loss), xytext=(best_epoch+5, best_val_loss+0.1),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 2: MAE (Training vs Validation)
    if 'mae' in df.columns and 'val_mae' in df.columns:
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
    else:
        axes[0, 1].text(0.5, 0.5, 'MAE data not available', ha='center', va='center', 
                       transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('MAE Not Available', fontweight='bold')
    
    # Plot 3: MSE (Training vs Validation)
    if 'mse' in df.columns and 'val_mse' in df.columns:
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
    else:
        axes[1, 0].text(0.5, 0.5, 'MSE data not available', ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('MSE Not Available', fontweight='bold')
    
    # Plot 4: Learning Rate and Training Gaps
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    # Plot learning rate if available
    if 'learning_rate' in df.columns:
        line1 = ax1.plot(df['epoch'], df['learning_rate'], label='Learning Rate', color=lr_color, linewidth=2)
        ax1.set_ylabel('Learning Rate', color=lr_color)
        ax1.tick_params(axis='y', labelcolor=lr_color)
    else:
        ax1.text(0.5, 0.5, 'Learning rate data not available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_ylabel('Learning Rate (N/A)', color=lr_color)
    
    # Calculate and plot training-validation gaps
    gaps_plotted = []
    if 'loss' in df.columns and 'val_loss' in df.columns:
        loss_gap = df['val_loss'] - df['loss']
        gap_label = f'Val-Train Gap ({loss_label})'
        line2 = ax2.plot(df['epoch'], loss_gap, label=gap_label, color=huber_color, linewidth=2, linestyle='-')
        gaps_plotted.append(gap_label)
    
    if 'mae' in df.columns and 'val_mae' in df.columns:
        mae_gap = df['val_mae'] - df['mae']
        line3 = ax2.plot(df['epoch'], mae_gap, label='Val-Train Gap (MAE)', color='green', linewidth=2, linestyle='--')
        gaps_plotted.append('Val-Train Gap (MAE)')
    
    ax2.set_ylabel('Validation - Training Gap', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Add horizontal line at gap = 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    axes[1, 1].set_title('Learning Rate & Overfitting Gaps', fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Epoch')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add text box with key statistics
    final_loss_gap = df['val_loss'].iloc[-1] - df['loss'].iloc[-1]
    final_mae_gap = df['val_mae'].iloc[-1] - df['mae'].iloc[-1] if 'mae' in df.columns else 0
    
    # Find learning rate reduction point
    lr_reduction_epoch = None
    if 'learning_rate' in df.columns:
        lr_changes = df['learning_rate'].diff().abs() > 1e-6
        if lr_changes.any():
            lr_reduction_epoch = df[lr_changes].index[0] + 1
    
    textstr = f'''Key Statistics ({training_type.replace('_', ' ').title()}):
• Best Validation {loss_label}: {best_val_loss:.4f} (Epoch {best_epoch})'''
    
    if 'mae' in df.columns:
        textstr += f'\n• Best Validation MAE: {best_val_mae:.4f} (Epoch {best_mae_epoch})'
    
    textstr += f'''
• Final Training {loss_label}: {df['loss'].iloc[-1]:.4f}
• Final Validation {loss_label}: {df['val_loss'].iloc[-1]:.4f}
• {loss_label} Overfitting Gap: {final_loss_gap:.4f}'''
    
    if 'mae' in df.columns:
        textstr += f'\n• MAE Overfitting Gap: {final_mae_gap:.4f}'
    
    if lr_reduction_epoch:
        textstr += f'\n• LR Reduction: Epoch {lr_reduction_epoch}'
    
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Save the plot
    if save_path is None:
        save_path = f'/Users/rizkimuhammad/Honours/msa-tf2/training_visualization_{training_type}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Training visualization saved as '{save_path}'")
    
    return df, training_type

def main():
    parser = argparse.ArgumentParser(description='Visualize MSA-TF2 training progress')
    parser.add_argument('--csv', type=str, help='Path to training log CSV file')
    parser.add_argument('--type', type=str, choices=['original_mae', 'improved_huber', 'balanced', 'auto'], 
                       default='auto', help='Type of training to visualize')
    parser.add_argument('--save', type=str, help='Path to save the visualization')
    parser.add_argument('--pattern', type=str, default='*training_log_*.csv', 
                       help='Pattern to match training log files')
    
    args = parser.parse_args()
    
    try:
        # Find or use specified CSV file
        if args.csv:
            csv_path = args.csv
        else:
            csv_path = find_latest_training_log(args.pattern)
        
        print(f"Loading training data from: {csv_path}")
        
        # Read the training log CSV file
        df = pd.read_csv(csv_path)
        
        # Create visualization
        df, training_type = create_comprehensive_visualization(df, args.type, args.save)
        
        # Print summary
        print(f"\nTraining Summary ({training_type.replace('_', ' ').title()}):")
        print(f"• Best validation loss: {df['val_loss'].min():.4f}")
        if 'mae' in df.columns:
            print(f"• Best validation MAE: {df['val_mae'].min():.4f}")
        print(f"• Final training loss: {df['loss'].iloc[-1]:.4f}")
        print(f"• Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
        
        # Show the plot
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run a training script first to generate training logs.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check that the training log file exists and has the expected format.")

if __name__ == "__main__":
    main()








