# Evaluation Quick Start

Quick reference for model evaluation and visualization commands.

## 🚀 Quick Commands

### Evaluate MSA Seq Level Model

```bash
# From trained model
python evaluate_and_visualize.py \
    --model weights/seqlevel_final_20251019_010827.h5 \
    --data ./data \
    --name MSA_SeqLevel
```

### Generate Predictions

```bash
# Generate and save predictions
python generate_predictions.py \
    --model weights/seqlevel_final_20251019_010827.h5 \
    --data ./data \
    --name MSA_SeqLevel \
    --output ./predictions
```

### Compare Two Models

```bash
# MSA vs DeepHOSeq
python compare_msa_deephoseq.py \
    --msa_model weights/seqlevel_final.h5 \
    --deephoseq_pred ../deephoseq/pred.npy \
    --deephoseq_true ../deephoseq/true.npy \
    --data ./data
```

### Visualize Training

```bash
# Visualize training progress
python visualize_training.py weights/seqlevel_training_log_20251019_010827.csv
```

## 📊 What You Get

### Single Model Evaluation (5 visualizations)

1. **Prediction Scatter** - Predicted vs Actual with correlations
2. **Residual Analysis** - 4 plots: residuals, distribution, Q-Q, error magnitude
3. **Sentiment Ranges** - Performance across 6 sentiment ranges
4. **Classification Metrics** - Binary + 7-class confusion matrices
5. **Error Analysis** - Top failures, percentiles, cumulative distribution

### Model Comparison (adds)

6. **Side-by-Side Comparison** - 6 plots comparing all key metrics
7. **Statistical Tests** - T-test and Wilcoxon for significance

## 📁 Output Files

```
evaluation_results/
├── [model_name]_prediction_scatter_[timestamp].png
├── [model_name]_residual_analysis_[timestamp].png
├── [model_name]_sentiment_ranges_[timestamp].png
├── [model_name]_classification_[timestamp].png
├── [model_name]_error_analysis_[timestamp].png
├── [model_name]_metrics_[timestamp].json
└── [model_name]_report_[timestamp].txt
```

## 🎯 Common Workflows

### After Training

```bash
# 1. Visualize training
python visualize_training.py weights/seqlevel_training_log_*.csv

# 2. Comprehensive evaluation
python evaluate_and_visualize.py --model weights/seqlevel_final_*.h5 --data ./data --name MSA

# 3. Check results
open evaluation_results/MSA_prediction_scatter_*.png
```

### Before Publishing

```bash
# Generate all evaluation materials
python evaluate_and_visualize.py --model weights/best_model.h5 --data ./data --name MSA_Final

# Compare with baseline
python compare_msa_deephoseq.py \
    --msa_model weights/best_model.h5 \
    --deephoseq_pred ../deephoseq/pred.npy \
    --deephoseq_true ../deephoseq/true.npy \
    --data ./data \
    --output ./paper_figures
```

### Debugging Poor Performance

```bash
# 1. Generate predictions
python generate_predictions.py --model weights/model.h5 --data ./data --evaluate

# 2. Look at error analysis
open predictions/evaluation/*error_analysis*.png

# 3. Check sentiment ranges
open predictions/evaluation/*sentiment_ranges*.png

# 4. Inspect worst predictions in report
cat predictions/evaluation/*report*.txt
```

## 🔑 Key Metrics Explained

| Metric | Good Value | What It Means |
|--------|------------|---------------|
| **MAE** | < 0.8 | Average absolute error |
| **RMSE** | < 1.0 | Penalizes large errors |
| **R²** | > 0.5 | Variance explained |
| **Pearson r** | > 0.7 | Linear correlation |
| **Binary Acc** | > 75% | Pos/Neg classification |
| **7-Class Acc** | > 35% | Fine-grained sentiment |

## 🛠️ Command Options

### evaluate_and_visualize.py

```bash
# Essential
--model PATH              # Model file (.h5)
--data DIR               # Data directory
--name NAME              # Model name for outputs

# Optional
--split {train,valid,test}  # Data split (default: test)
--output DIR                # Output directory
--compare                   # Compare two models
--model1 / --model2        # Models to compare
```

### compare_msa_deephoseq.py

```bash
# MSA options
--msa_model PATH         # MSA model file
--msa_pred PATH          # Pre-saved predictions
--msa_true PATH          # Pre-saved ground truth

# DeepHOSeq options
--deephoseq_pred PATH    # DeepHOSeq predictions
--deephoseq_true PATH    # Ground truth

# Common
--data DIR              # Data directory
--output DIR            # Output directory
```

### generate_predictions.py

```bash
--model PATH            # Model to evaluate
--data DIR             # Data directory
--name NAME            # Model name
--split {train,valid,test}  # Split to predict
--format {npy,h5}      # Output format
--batch_size N         # Batch size
--evaluate             # Run evaluation after
```

## 💡 Pro Tips

1. **Save predictions** for quick re-evaluation without re-running model
2. **Use --evaluate flag** with generate_predictions.py for one-command workflow
3. **Check comparison_summary.txt** for quick model comparison
4. **All visualizations are 300 DPI** - ready for papers/presentations
5. **JSON metrics files** can be loaded into spreadsheets for tables

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python evaluate_and_visualize.py --model weights/model.h5 --data ./data --batch_size 16
```

### Can't Find Model

```bash
# List available models
ls -lh weights/*.h5

# Use full path
python evaluate_and_visualize.py --model /full/path/to/model.h5 --data ./data
```

### Wrong Environment

```bash
# MSA needs TF2
conda activate msa-tf2

# DeepHOSeq needs TF1
cd ../deephoseq && source venv/bin/activate
```

## 📚 More Help

- Full guide: `EVALUATION_GUIDE.md`
- Training guide: `QUICKSTART.md`
- Architecture: `SEQLEVEL_ARCHITECTURE.md`

## 🎨 Visualization Features

All plots are designed with:
- ✅ No overlapping labels or elements
- ✅ Clear, readable fonts
- ✅ Consistent color schemes
- ✅ Proper spacing and padding
- ✅ Publication-ready quality (300 DPI)
- ✅ Informative titles and legends
- ✅ Grid lines for easy reading
- ✅ Annotated key points

