# Training Visualization Guide

## Overview

The `visualize_training.py` script creates comprehensive visualizations of your MSA model training progress from CSV log files.

---

## Usage

### Basic Usage

```bash
python visualize_training.py <path_to_csv>
```

**Example:**
```bash
python visualize_training.py weights/seqlevel_training_log_20251019_010827.csv
```

**Output:** Creates a PNG file in the same directory as the CSV:
- `weights/seqlevel_training_log_20251019_010827_visualization.png`

---

### Custom Output Path

```bash
python visualize_training.py <path_to_csv> --output <output_path>
```

**Example:**
```bash
python visualize_training.py weights/seqlevel_training_log_20251019_010827.csv \
    --output my_training_results.png
```

---

### Get Help

```bash
python visualize_training.py --help
```

---

## Visualization Layout

The script creates a **2×2 grid** with 4 charts:

### **Top Left: Mean Absolute Error (MAE)**
- Training MAE (blue line)
- Validation MAE (purple line)
- Annotation showing best validation MAE

### **Top Right: Mean Squared Error (MSE)**
- Training MSE (blue line)
- Validation MSE (purple line)
- Annotation showing best validation MSE

### **Bottom Left: Learning Rate & Overfitting Gap**
- Learning rate over time (orange line, left y-axis)
- Validation-Training gap (red dashed line, right y-axis)
- Shows if model is overfitting (gap > 0)

### **Bottom Right: Combined Metrics**
- All metrics overlaid for comparison
- Shows convergence point
- Helps identify when training stabilizes

### **Top Center: Overall Summary**
- Best validation MAE and MSE
- Final overfitting gap
- Total epochs trained
- Learning rate reductions

---

## Key Improvements from Original

✅ **No Loss plot** (since MAE = Loss for our model)  
✅ **Summary doesn't block charts** (moved to top center)  
✅ **Command-line arguments** (flexible CSV path)  
✅ **Better layout** (MAE and MSE both prominent)  
✅ **Auto-output naming** (based on CSV name)  

---

## Example Output

The visualization includes:

1. **Training Progress:** See how MAE and MSE decrease over epochs
2. **Overfitting Detection:** Red dashed line shows train-val gap
3. **Learning Rate Schedule:** Orange line shows LR adjustments
4. **Best Results:** Yellow boxes highlight best validation scores
5. **Convergence Point:** Green line (if detected) shows optimal stopping

---

## Interpreting the Visualization

### Good Training Signs:
- ✅ Validation curves follow training curves closely
- ✅ Gap between train and validation is small (<0.1)
- ✅ Curves flatten out (convergence reached)
- ✅ Learning rate reduced at right times

### Warning Signs:
- ⚠️ Large gap between train and validation (overfitting)
- ⚠️ Validation curves increase while training decreases
- ⚠️ Unstable/noisy validation curves
- ⚠️ No convergence (keep training)

---

## Quick Commands

### Visualize Latest Training Run

```bash
# Find latest log
latest_log=$(ls -t weights/seqlevel_training_log_*.csv | head -1)

# Visualize it
python visualize_training.py "$latest_log"
```

### Visualize All Training Runs

```bash
# Create visualizations for all logs
for csv in weights/seqlevel_training_log_*.csv; do
    echo "Processing: $csv"
    python visualize_training.py "$csv"
done
```

### Compare Two Models

```bash
# Visualize model 1
python visualize_training.py weights/log1.csv --output comparison_model1.png

# Visualize model 2  
python visualize_training.py weights/log2.csv --output comparison_model2.png

# View side by side
open comparison_model1.png comparison_model2.png
```

---

## Troubleshooting

### Error: "File not found"
```bash
# Check if file exists
ls weights/*.csv

# Use full path
python visualize_training.py /full/path/to/log.csv
```

### Error: "Missing columns"
Ensure your CSV has these columns:
- `epoch`, `mae`, `val_mae`, `mse`, `val_mse`, `learning_rate`

### Plot doesn't show
The script calls `plt.show()` at the end. If running remotely:
```bash
# Just save without showing
# Comment out plt.show() line or redirect display
MPLBACKEND=Agg python visualize_training.py weights/log.csv
```

---

## Tips

1. **Compare experiments:** Generate visualizations for different model configs
2. **Track progress:** Run after each training session
3. **Spot issues early:** Large gaps indicate overfitting
4. **Find best epoch:** Yellow annotations show optimal performance
5. **Share results:** High-res PNG perfect for presentations/thesis

---

## Output Format

- **Resolution:** 300 DPI (publication quality)
- **Size:** 16×10 inches (1920×1200 pixels)
- **Format:** PNG with white background
- **File size:** ~500-800 KB

Perfect for:
- Research papers
- Thesis documents  
- Presentations
- Progress reports

---

## Integration with Training

Your training script automatically creates CSV logs:

```bash
# Train model
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128

# This creates: weights/seqlevel_training_log_TIMESTAMP.csv

# Visualize results
python visualize_training.py weights/seqlevel_training_log_TIMESTAMP.csv
```

---

## Example Workflow

```bash
# 1. Start training
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128 --epochs 100

# 2. Wait for training to complete (or interrupt with Ctrl+C)

# 3. Find the log file
ls -t weights/seqlevel_training_log_*.csv | head -1

# 4. Visualize
python visualize_training.py weights/seqlevel_training_log_20251019_010827.csv

# 5. View the PNG
open weights/seqlevel_training_log_20251019_010827_visualization.png
```

---

## Summary

**Quick start:**
```bash
python visualize_training.py weights/seqlevel_training_log_*.csv
```

**For thesis/paper:**
```bash
python visualize_training.py weights/final_model_log.csv \
    --output thesis_figure_training_progress.png
```

**The visualization will show you everything you need to know about your training run!** 📊





