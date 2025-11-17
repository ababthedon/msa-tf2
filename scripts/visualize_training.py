import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize MSA training progress from CSV log')
parser.add_argument('csv_path', type=str, help='Path to training log CSV file')
parser.add_argument('--output', '-o', type=str, default=None, 
                    help='Output path for visualization (default: training_visualization.png in same directory as CSV)')
args = parser.parse_args()

# Check if file exists
if not os.path.exists(args.csv_path):
    print(f"Error: File not found: {args.csv_path}")
    sys.exit(1)

# Read the training log CSV file
print(f"Reading training log: {args.csv_path}")
df = pd.read_csv(args.csv_path)

# Set default output path
if args.output is None:
    csv_dir = os.path.dirname(args.csv_path)
    csv_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    args.output = os.path.join(csv_dir, f'{csv_name}_visualization.png')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('MSA Sequence-Level Model - Training Progress', fontsize=16, fontweight='bold', y=0.98)

# Define colors
train_color = '#2E86AB'  # Blue
val_color = '#A23B72'    # Purple
lr_color = '#F18F01'     # Orange

# Plot 1: MAE (Training vs Validation)
axes[0, 0].plot(df['epoch'], df['mae'], label='Training MAE', color=train_color, linewidth=2)
axes[0, 0].plot(df['epoch'], df['val_mae'], label='Validation MAE', color=val_color, linewidth=2)
axes[0, 0].set_title('Mean Absolute Error Over Time', fontweight='bold', pad=15)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].legend(loc='upper right')
axes[0, 0].grid(True, alpha=0.3)

# Add annotation for best validation MAE
best_val_mae_idx = df['val_mae'].idxmin()
best_val_mae = df.loc[best_val_mae_idx, 'val_mae']
best_mae_epoch = df.loc[best_val_mae_idx, 'epoch']

# Calculate adaptive position to keep annotation inside plot
y_range = axes[0, 0].get_ylim()
x_range = axes[0, 0].get_xlim()
y_span = y_range[1] - y_range[0]
x_span = x_range[1] - x_range[0]

# Position annotation in upper-left area to avoid overlap
text_x = x_range[0] + 0.05 * x_span
text_y = y_range[0] + 0.85 * y_span

axes[0, 0].annotate(f'Best: {best_val_mae:.3f}\n(Epoch {best_mae_epoch})', 
                   xy=(best_mae_epoch, best_val_mae), 
                   xytext=(text_x, text_y),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5),
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.8, edgecolor='red'),
                   fontsize=9, ha='left')

# Plot 2: MSE (Training vs Validation)
axes[0, 1].plot(df['epoch'], df['mse'], label='Training MSE', color=train_color, linewidth=2)
axes[0, 1].plot(df['epoch'], df['val_mse'], label='Validation MSE', color=val_color, linewidth=2)
axes[0, 1].set_title('Mean Squared Error Over Time', fontweight='bold', pad=15)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].legend(loc='upper right')
axes[0, 1].grid(True, alpha=0.3)

# Add annotation for best validation MSE
best_val_mse_idx = df['val_mse'].idxmin()
best_val_mse = df.loc[best_val_mse_idx, 'val_mse']
best_mse_epoch = df.loc[best_val_mse_idx, 'epoch']

# Calculate adaptive position to keep annotation inside plot
y_range = axes[0, 1].get_ylim()
x_range = axes[0, 1].get_xlim()
y_span = y_range[1] - y_range[0]
x_span = x_range[1] - x_range[0]

# Position annotation in upper-left area to avoid overlap
text_x = x_range[0] + 0.05 * x_span
text_y = y_range[0] + 0.85 * y_span

axes[0, 1].annotate(f'Best: {best_val_mse:.3f}\n(Epoch {best_mse_epoch})', 
                   xy=(best_mse_epoch, best_val_mse), 
                   xytext=(text_x, text_y),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7, lw=1.5),
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.8, edgecolor='red'),
                   fontsize=9, ha='left')

# Plot 3: Learning Rate and Training Gap
ax1 = axes[1, 0]
ax2 = ax1.twinx()

# Calculate training-validation gap (using MAE)
gap = df['val_mae'] - df['mae']

# Plot learning rate
line1 = ax1.plot(df['epoch'], df['learning_rate'], label='Learning Rate', color=lr_color, linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate', color=lr_color)
ax1.tick_params(axis='y', labelcolor=lr_color)

# Plot training-validation gap
line2 = ax2.plot(df['epoch'], gap, label='Val-Train Gap (MAE)', color='red', linewidth=2, linestyle='--')
ax2.set_ylabel('Validation - Training MAE', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add horizontal line at gap = 0
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

axes[1, 0].set_title('Learning Rate & Overfitting Gap', fontweight='bold', pad=15)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

ax1.grid(True, alpha=0.3)

# Plot 4: Training Progress (Epochs over time)
# Calculate epoch duration if possible
if 'epoch' in df.columns and len(df) > 1:
    epochs = df['epoch'].values
    # Create a simple epoch progress plot
    axes[1, 1].plot(df['epoch'], df['mae'], label='Training MAE', 
                   color=train_color, linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(df['epoch'], df['val_mae'], label='Validation MAE', 
                   color=val_color, linewidth=1.5, alpha=0.7)
    axes[1, 1].plot(df['epoch'], df['mse']/2, label='Training MSE/2', 
                   color=train_color, linewidth=1.5, linestyle='--', alpha=0.5)
    axes[1, 1].plot(df['epoch'], df['val_mse']/2, label='Validation MSE/2', 
                   color=val_color, linewidth=1.5, linestyle='--', alpha=0.5)
    
    axes[1, 1].set_title('Combined Metrics Over Time', fontweight='bold', pad=15)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Metric Value')
    axes[1, 1].legend(loc='upper right', fontsize=8, framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight convergence point (where validation starts increasing)
    if len(df) > 5:
        # Find potential early stopping point
        val_mae_vals = df['val_mae'].values
        rolling_mean = pd.Series(val_mae_vals).rolling(window=3).mean()
        if rolling_mean.iloc[-1] > rolling_mean.min():
            conv_idx = rolling_mean.idxmin()
            axes[1, 1].axvline(x=df.loc[conv_idx, 'epoch'], color='green', 
                             linestyle=':', alpha=0.5, linewidth=2)
            axes[1, 1].text(df.loc[conv_idx, 'epoch'], axes[1, 1].get_ylim()[1]*0.9, 
                          'Convergence', rotation=90, va='top', ha='right', 
                          color='green', fontsize=9)

# Add overall summary in title area (not blocking charts)
final_gap = df['val_mae'].iloc[-1] - df['mae'].iloc[-1]
total_epochs = len(df)

# Check if learning rate was reduced
lr_reduced = df['learning_rate'].iloc[-1] < df['learning_rate'].iloc[0]
if lr_reduced:
    lr_reduction_epoch = df[df['learning_rate'] < df['learning_rate'].iloc[0]].index[0]
    lr_text = f"LR reduced at epoch {lr_reduction_epoch}"
else:
    lr_text = "No LR reduction"

summary_text = (f"Best Val MAE: {best_val_mae:.4f} @ Epoch {best_mae_epoch}  |  "
                f"Best Val MSE: {best_val_mse:.4f} @ Epoch {best_mse_epoch}  |  "
                f"Final Gap: {final_gap:.4f}  |  "
                f"Total Epochs: {total_epochs}  |  "
                f"{lr_text}")

# Adjust layout first to position charts properly
plt.tight_layout(rect=[0, 0, 1, 0.955])

# Add summary text after layout adjustment
fig.text(0.5, 0.975, summary_text, ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.85, edgecolor='navy'))

# Save the plot
plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n{'='*70}")
print(f"Training Visualization Saved")
print(f"{'='*70}")
print(f"Output: {args.output}")
print(f"\nTraining Summary:")
print(f"  Best Validation MAE:  {best_val_mae:.4f} at epoch {best_mae_epoch}")
print(f"  Best Validation MSE:  {best_val_mse:.4f} at epoch {best_mse_epoch}")
print(f"  Final Training MAE:   {df['mae'].iloc[-1]:.4f}")
print(f"  Final Validation MAE: {df['val_mae'].iloc[-1]:.4f}")
print(f"  Final Training MSE:   {df['mse'].iloc[-1]:.4f}")
print(f"  Final Validation MSE: {df['val_mse'].iloc[-1]:.4f}")
print(f"  Overfitting Gap:      {final_gap:.4f}")
print(f"  Total Epochs:         {total_epochs}")
print(f"{'='*70}")

# Show the plot
plt.show()
