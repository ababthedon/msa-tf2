# Comprehensive Model Evaluation Guide

This guide explains how to use the new evaluation and visualization tools for MSA Seq Level and DeepHOSeq models.

## Overview

The evaluation suite consists of three main scripts:

1. **`evaluate_and_visualize.py`** - Comprehensive single-model evaluation
2. **`compare_msa_deephoseq.py`** - Specialized MSA vs DeepHOSeq comparison
3. **`visualize_training.py`** - Training progress visualization (existing)

## Features

### Visualizations Generated

All visualizations are carefully designed to **avoid overlapping labels, icons, and charts**:

1. **Prediction Quality Analysis**
   - Scatter plot: Predicted vs Actual values with regression line
   - Error-colored points showing prediction accuracy
   - Correlation metrics (Pearson r, Spearman ρ)

2. **Residual Analysis**
   - Residuals vs Predicted values
   - Error distribution histogram
   - Q-Q plot for normality check
   - Absolute error vs true values

3. **Sentiment-Range Performance**
   - MAE by sentiment range (Very Neg to Very Pos)
   - Sample distribution across ranges
   - RMSE by sentiment range
   - R² by sentiment range

4. **Classification Metrics**
   - Binary confusion matrix (Positive/Negative)
   - Binary metrics (Accuracy, Precision, Recall, F1)
   - 7-class confusion matrix (-3 to +3)
   - 7-class metrics

5. **Error Analysis**
   - Top-K worst predictions
   - Error percentiles
   - Error vs prediction confidence
   - Cumulative error distribution

6. **Model Comparison** (when comparing models)
   - Side-by-side metric comparison
   - Error distribution comparison
   - Box plots
   - Statistical significance tests (t-test, Wilcoxon)

### Metrics Calculated

**Regression Metrics:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Pearson Correlation (r and p-value)
- Spearman Correlation (ρ and p-value)
- Mean Error, Std Error, Median Absolute Error

**Classification Metrics:**
- Binary classification: Accuracy, Precision, Recall, F1
- 7-class classification: Accuracy, F1-Macro, F1-Weighted

## Usage Examples

### 1. Evaluate MSA Seq Level Model

```bash
# Activate environment
conda activate msa-tf2

# Evaluate from saved model
python evaluate_and_visualize.py \
    --model weights/seqlevel_final_20251019_010827.h5 \
    --data ./data \
    --split test \
    --name "MSA_SeqLevel" \
    --output ./evaluation_results
```

**What this does:**
- Loads the saved model
- Generates predictions on test set
- Creates 5 comprehensive visualizations
- Saves metrics to JSON
- Generates detailed text report

**Output files:**
```
evaluation_results/
├── MSA_SeqLevel_prediction_scatter_[timestamp].png
├── MSA_SeqLevel_residual_analysis_[timestamp].png
├── MSA_SeqLevel_sentiment_ranges_[timestamp].png
├── MSA_SeqLevel_classification_[timestamp].png
├── MSA_SeqLevel_error_analysis_[timestamp].png
├── MSA_SeqLevel_metrics_[timestamp].json
└── MSA_SeqLevel_report_[timestamp].txt
```

### 2. Evaluate DeepHOSeq Model

First, generate predictions in the DeepHOSeq environment:

```bash
# Navigate to deephoseq directory
cd ../deephoseq

# Activate deephoseq environment (Python 3.6, TF 1.x)
source venv/bin/activate

# Run evaluation and save predictions
python -c "
import sys
import os
import tensorflow as tf
import numpy as np
import h5py

sys.path.append(os.getcwd())
from Deep_HOSeq import Deep_HOSeq

# Configuration
data_dir = './data/'
checkpoint_path = './data/best_val_epoch_18.ckpt'
batch_size = 256

# Load model
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
run_config.allow_soft_placement = True

with tf.Session(config=run_config) as sess:
    model = Deep_HOSeq(
        sess, data_dir=data_dir, batch_size=batch_size,
        hidden_v=5, hidden_a=5, hidden_t=5,
        LSTM_hid_t=128, text_out=64,
        LSTM_hid_v=5, LSTM_hid_a=10,
        Seq_count=20, Conv_filt=1
    )
    
    # Restore checkpoint
    model.saver.restore(sess, checkpoint_path)
    
    # Load test data
    with h5py.File(os.path.join(data_dir, 'y_test.h5'), 'r') as f:
        y_test = f['data'][:]
    
    with h5py.File(os.path.join(data_dir, 'video_test.h5'), 'r') as f:
        v_test = f['data'][:]
    
    with h5py.File(os.path.join(data_dir, 'audio_test.h5'), 'r') as f:
        a_test = f['data'][:]
    
    with h5py.File(os.path.join(data_dir, 'text_test.h5'), 'r') as f:
        t_test = f['data'][:]
    
    # Generate predictions
    feed_dict = {
        model.video_inputs: v_test,
        model.audio_inputs: a_test,
        model.text_inputs: t_test,
        model.y: y_test,
        model.drop_ratio: 1.0
    }
    
    predictions = sess.run(model.Pred, feed_dict)
    
    # Save predictions
    np.save('deephoseq_pred.npy', predictions)
    np.save('deephoseq_y_test.npy', y_test)
    
print('✓ Predictions saved!')
"
```

Then evaluate:

```bash
# Go back to msa-tf2 directory
cd ../msa-tf2
conda activate msa-tf2

# Evaluate DeepHOSeq using saved predictions
python evaluate_and_visualize.py \
    --y_true ../deephoseq/deephoseq_y_test.npy \
    --y_pred ../deephoseq/deephoseq_pred.npy \
    --name "DeepHOSeq" \
    --output ./evaluation_results
```

### 3. Compare MSA Seq Level vs DeepHOSeq

```bash
# Make sure you have predictions from both models
python compare_msa_deephoseq.py \
    --msa_model weights/seqlevel_final_20251019_010827.h5 \
    --data ./data \
    --deephoseq_pred ../deephoseq/deephoseq_pred.npy \
    --deephoseq_true ../deephoseq/deephoseq_y_test.npy \
    --output ./comparison_results
```

**What this does:**
- Evaluates both models
- Generates individual reports for each
- Creates comparison visualization
- Performs statistical significance tests
- Generates comparison summary

**Output files:**
```
comparison_results/
├── MSA_SeqLevel_* (all evaluation files)
├── DeepHOSeq_* (all evaluation files)
├── MSA_vs_DeepHOSeq_[timestamp].png
├── comparison_metrics_[timestamp].json
└── comparison_summary_[timestamp].txt
```

### 4. Compare Two MSA Models

```bash
# Compare two different MSA configurations
python evaluate_and_visualize.py \
    --compare \
    --model1 weights/seqlevel_final_config1.h5 \
    --model2 weights/seqlevel_final_config2.h5 \
    --name1 "MSA_Config1" \
    --name2 "MSA_Config2" \
    --data ./data \
    --output ./model_comparison
```

### 5. Evaluate with Pre-saved Predictions

If you already have predictions saved:

```bash
python evaluate_and_visualize.py \
    --y_true test_labels.npy \
    --y_pred test_predictions.npy \
    --name "MyModel" \
    --output ./results
```

## Quick Workflow

### For MSA Seq Level Training and Evaluation

```bash
# 1. Train model
python train_seqlevel.py --epochs 100 --batch_size 32

# 2. Visualize training progress
python visualize_training.py weights/seqlevel_training_log_[timestamp].csv

# 3. Comprehensive evaluation
python evaluate_and_visualize.py \
    --model weights/seqlevel_final_[timestamp].h5 \
    --data ./data \
    --name "MSA_SeqLevel" \
    --output ./evaluation_results
```

### For Full Comparison

```bash
# 1. Train MSA model (if not done)
python train_seqlevel.py --epochs 100

# 2. Generate DeepHOSeq predictions (see example above)
cd ../deephoseq
# ... run prediction script ...
cd ../msa-tf2

# 3. Run comparison
python compare_msa_deephoseq.py \
    --msa_model weights/seqlevel_final_[timestamp].h5 \
    --data ./data \
    --deephoseq_pred ../deephoseq/deephoseq_pred.npy \
    --deephoseq_true ../deephoseq/deephoseq_y_test.npy \
    --output ./comparison_results
```

## Understanding the Outputs

### Prediction Scatter Plot
- **Perfect alignment**: Points on the diagonal blue line
- **Color gradient**: Darker red = larger errors
- **Regression line**: Shows systematic bias (should be close to diagonal)
- **Metrics box**: Quick overview of performance

### Residual Analysis
- **Residual plot**: Should be randomly scattered around zero
- **Error distribution**: Should be roughly normal (bell-curved)
- **Q-Q plot**: Points on line = normal distribution
- **Error magnitude**: Shows if errors increase with prediction range

### Sentiment Range Analysis
- **MAE by range**: Identifies which sentiments are harder to predict
- **Sample distribution**: Shows data balance
- **RMSE by range**: Penalizes large errors more
- **R² by range**: Shows prediction quality per range

### Classification Metrics
- **Binary confusion matrix**: True Pos/Neg vs Predicted Pos/Neg
- **7-class confusion matrix**: Shows which classes are confused
- **Diagonal = correct predictions**
- **Off-diagonal = misclassifications**

### Error Analysis
- **Top-K worst**: Individual predictions to investigate
- **Percentiles**: Understanding error distribution
- **Cumulative plot**: X% of samples have error ≤ Y

### Model Comparison
- **Bar charts**: Direct metric comparison
- **Box plots**: Shows error distribution shape
- **Statistical tests**: p < 0.05 = significant difference
- **Improvement %**: Positive = Model 1 better

## Tips for Interpretation

1. **MAE vs RMSE**: RMSE penalizes large errors more heavily
2. **R² Score**: 
   - 1.0 = perfect predictions
   - 0.0 = no better than mean
   - Negative = worse than mean
3. **Pearson vs Spearman**: 
   - Pearson = linear correlation
   - Spearman = monotonic correlation
4. **Binary vs 7-class**: 
   - Binary easier (just positive/negative)
   - 7-class harder but more informative
5. **Look for patterns**:
   - Are extreme sentiments harder to predict?
   - Is there systematic bias (over/under prediction)?
   - Are errors normally distributed?

## Troubleshooting

### Memory Issues
```bash
# Reduce batch size when loading predictions
# Edit evaluate_and_visualize.py line 67:
# Change: batch_size=32
# To: batch_size=16
```

### TensorFlow Version Conflicts
```bash
# MSA Seq Level requires TF 2.x
conda activate msa-tf2

# DeepHOSeq requires TF 1.x
cd ../deephoseq
source venv/bin/activate

# Generate predictions separately in each environment
```

### Missing Dependencies
```bash
conda activate msa-tf2
pip install scipy scikit-learn seaborn
```

### Can't Load Model
```bash
# Make sure you're using the correct model file
ls -lh weights/seqlevel_final_*.h5

# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Advanced Usage

### Custom Sentiment Ranges
Edit `evaluate_and_visualize.py` line 288:
```python
bin_edges = [-np.inf, -2, -1, 0, 1, 2, np.inf]  # Customize these
```

### Different Error Metrics
Add custom metrics in `calculate_regression_metrics()` function

### Save Predictions for Later
```python
import numpy as np
# After generating predictions:
np.save('my_predictions.npy', y_pred)
np.save('ground_truth.npy', y_true)
```

## References

- **MSA Seq Level**: Transformer-based architecture with cross-attention fusion
- **DeepHOSeq**: LSTM-based architecture with high-order sequence modeling
- **CMU-MOSI/MOSEI**: Multimodal sentiment analysis datasets

## Need Help?

Check the example outputs in `evaluation_results/` or `comparison_results/` directories for reference.

All visualizations are designed to be publication-ready with:
- Clear labels and titles
- No overlapping elements
- Consistent color schemes
- Proper spacing and padding
- High-resolution output (300 DPI)

