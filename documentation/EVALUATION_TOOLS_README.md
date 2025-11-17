# MSA Evaluation Tools

**Comprehensive evaluation and visualization suite for Multimodal Sentiment Analysis models**

Version: 1.0  
Created: October 2025  
Author: AI Assistant for Honours Project

---

## 📋 Overview

This evaluation suite provides publication-ready visualizations and comprehensive metrics for evaluating MSA models, with special attention to comparing MSA Seq Level (Transformer-based) and DeepHOSeq (LSTM-based) architectures.

### ✨ Key Features

- **🎨 Clean Visualizations**: All plots carefully designed with NO overlapping labels, icons, or charts
- **📊 Comprehensive Metrics**: Regression + Classification (Binary & 7-class)
- **🔬 Deep Analysis**: Prediction quality, residuals, sentiment ranges, error patterns
- **⚖️ Model Comparison**: Side-by-side with statistical significance tests
- **📄 Detailed Reports**: JSON metrics + human-readable summaries
- **🚀 Easy to Use**: Simple command-line interface with sensible defaults
- **📈 Publication Ready**: 300 DPI, consistent styling, professional appearance

---

## 📦 What's Included

### Core Scripts

| Script | Purpose |
|--------|---------|
| `evaluate_and_visualize.py` | Main evaluation script - single model or comparison |
| `compare_msa_deephoseq.py` | Specialized MSA vs DeepHOSeq comparison |
| `generate_predictions.py` | Generate and save predictions from trained models |
| `visualize_training.py` | Training progress visualization (existing) |
| `demo_evaluation.sh` | Automated demo showing all features |

### Documentation

| File | Content |
|------|---------|
| `EVALUATION_GUIDE.md` | Complete usage guide with examples |
| `EVALUATION_QUICKSTART.md` | Quick reference and cheatsheet |
| `EVALUATION_TOOLS_README.md` | This file - overview and summary |

---

## 🚀 Quick Start

### 1. Run the Demo

```bash
# Activate environment
conda activate msa-tf2

# Run demo (uses latest trained model)
./demo_evaluation.sh
```

This will generate all visualizations and show you what the tools can do.

### 2. Evaluate Your Model

```bash
python evaluate_and_visualize.py \
    --model weights/your_model.h5 \
    --data ./data \
    --name YourModel
```

### 3. Compare Models

```bash
python compare_msa_deephoseq.py \
    --msa_model weights/msa_model.h5 \
    --deephoseq_pred ../deephoseq/predictions.npy \
    --deephoseq_true ../deephoseq/ground_truth.npy \
    --data ./data
```

---

## 📊 Visualizations Generated

### 1. Prediction Scatter Plot
**What it shows:** How well predictions align with ground truth

**Elements:**
- Scatter points colored by error magnitude
- Perfect prediction line (diagonal)
- Regression line showing systematic bias
- Correlation metrics (Pearson r, Spearman ρ)
- R², MAE, RMSE summary

**How to interpret:**
- Points on diagonal = perfect predictions
- Darker red = larger errors
- Regression line close to diagonal = no systematic bias
- High correlation = good model

### 2. Residual Analysis (4 plots)
**What it shows:** Error patterns and distribution

**Elements:**
a) **Residual Plot**: Errors vs predicted values
   - Should be randomly scattered around zero
   - Patterns indicate systematic bias

b) **Error Distribution**: Histogram of prediction errors
   - Should be roughly bell-shaped (normal)
   - Centered at zero = unbiased

c) **Q-Q Plot**: Normality check
   - Points on line = normal distribution
   - Deviations indicate skewness/outliers

d) **Error Magnitude**: Absolute error vs true values
   - Shows if certain ranges are harder to predict
   - Trend line reveals patterns

### 3. Sentiment Range Analysis (4 plots)
**What it shows:** Performance across sentiment intensities

**Elements:**
- MAE by sentiment range (Very Neg to Very Pos)
- Sample distribution (data balance)
- RMSE by sentiment range
- R² by sentiment range

**How to interpret:**
- Higher bars = worse performance in that range
- Check if extreme sentiments are harder
- Correlate with sample counts (low samples = unreliable)

### 4. Classification Metrics (3 plots)
**What it shows:** Performance as classification task

**Elements:**
a) **Binary Confusion Matrix**: Positive/Negative
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications

b) **Binary Metrics Bar Chart**
   - Accuracy, Precision, Recall, F1-Score
   - Should all be > 0.75 for good model

c) **7-Class Confusion Matrix**: Fine-grained sentiment
   - Shows which intensity levels are confused
   - Darker diagonal = better predictions

### 5. Error Analysis (4 plots)
**What it shows:** Failure patterns and worst cases

**Elements:**
a) **Top-K Worst Predictions**
   - Individual samples with largest errors
   - Shows true vs predicted values
   - Useful for debugging

b) **Error Percentiles**
   - Distribution of errors
   - 50th percentile = median error
   - 95th percentile = worst 5%

c) **Error vs Prediction Range**
   - Shows if errors increase with prediction magnitude
   - Line should be relatively flat

d) **Cumulative Error Distribution**
   - X% of samples have error ≤ Y
   - Useful for understanding overall quality

### 6. Model Comparison (6 plots, when comparing)
**What it shows:** Direct model-to-model comparison

**Elements:**
a) **Metric Bar Charts**: MAE, RMSE, Correlation
   - Side-by-side comparison
   - Improvement percentages shown

b) **Error Distributions**: Overlaid histograms
   - Shows if one model has tighter distribution

c) **Box Plots**: Error distribution shape
   - Median, quartiles, outliers
   - More robust than just mean

d) **Statistical Tests**: T-test and Wilcoxon
   - p < 0.05 = significant difference
   - Tells you if improvement is real or noise

---

## 📈 Metrics Explained

### Regression Metrics

| Metric | Formula | Good Value | Interpretation |
|--------|---------|------------|----------------|
| **MAE** | mean(\|y_pred - y_true\|) | < 0.8 | Average error magnitude |
| **MSE** | mean((y_pred - y_true)²) | < 1.0 | Penalizes large errors |
| **RMSE** | √MSE | < 1.0 | Same scale as predictions |
| **R²** | 1 - SS_res/SS_tot | > 0.5 | Variance explained (1.0 = perfect) |
| **Pearson r** | Linear correlation | > 0.7 | Linear relationship strength |
| **Spearman ρ** | Rank correlation | > 0.7 | Monotonic relationship |

### Classification Metrics

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| **Binary Accuracy** | > 75% | Correct pos/neg predictions |
| **Binary F1** | > 0.75 | Harmonic mean of precision/recall |
| **7-Class Accuracy** | > 35% | Correct fine-grained predictions |
| **7-Class F1 Macro** | > 0.35 | Average F1 across classes |

### Typical Values for Sentiment Analysis

**Good Model:**
- MAE: 0.6-0.8
- RMSE: 0.8-1.0
- R²: 0.5-0.7
- Binary Acc: 75-85%
- 7-Class Acc: 35-45%

**State-of-the-art Model:**
- MAE: < 0.6
- RMSE: < 0.8
- R²: > 0.7
- Binary Acc: > 85%
- 7-Class Acc: > 45%

---

## 🔧 Usage Examples

### Basic Evaluation

```bash
# Evaluate test set
python evaluate_and_visualize.py --model weights/model.h5 --data ./data --name MyModel

# Evaluate validation set
python evaluate_and_visualize.py --model weights/model.h5 --data ./data --split valid --name MyModel
```

### Generate Predictions First

```bash
# Generate predictions
python generate_predictions.py --model weights/model.h5 --data ./data --name MyModel

# Evaluate later (faster, no model loading)
python evaluate_and_visualize.py \
    --y_true predictions/MyModel_test_true_*.npy \
    --y_pred predictions/MyModel_test_pred_*.npy \
    --name MyModel
```

### Compare Two Models

```bash
# Direct model comparison
python evaluate_and_visualize.py \
    --compare \
    --model1 weights/model_v1.h5 \
    --model2 weights/model_v2.h5 \
    --name1 "Version 1" \
    --name2 "Version 2" \
    --data ./data
```

### MSA vs DeepHOSeq

```bash
# Step 1: Generate MSA predictions
python generate_predictions.py --model weights/msa.h5 --data ./data --name MSA

# Step 2: Generate DeepHOSeq predictions (in deephoseq environment)
# ... see EVALUATION_GUIDE.md for details ...

# Step 3: Compare
python compare_msa_deephoseq.py \
    --msa_pred predictions/MSA_test_pred.npy \
    --msa_true predictions/MSA_test_true.npy \
    --deephoseq_pred ../deephoseq/pred.npy \
    --deephoseq_true ../deephoseq/true.npy
```

---

## 📁 Output Structure

```
evaluation_results/
├── [model_name]_prediction_scatter_[timestamp].png
├── [model_name]_residual_analysis_[timestamp].png
├── [model_name]_sentiment_ranges_[timestamp].png
├── [model_name]_classification_[timestamp].png
├── [model_name]_error_analysis_[timestamp].png
├── [model_name]_metrics_[timestamp].json
└── [model_name]_report_[timestamp].txt

comparison_results/
├── MSA_SeqLevel_* (all MSA files)
├── DeepHOSeq_* (all DeepHOSeq files)
├── MSA_vs_DeepHOSeq_[timestamp].png
├── comparison_metrics_[timestamp].json
└── comparison_summary_[timestamp].txt
```

---

## 🎯 Design Principles

### No Overlapping Elements

All visualizations are carefully designed to prevent:
- ❌ Overlapping labels
- ❌ Overlapping icons
- ❌ Overlapping chart elements
- ❌ Text outside plot boundaries
- ❌ Unreadable small fonts

We achieve this through:
- ✅ Careful layout spacing (`tight_layout`, `pad` parameters)
- ✅ Dynamic text positioning based on data
- ✅ Proper figure size calculations
- ✅ Strategic legend placement
- ✅ Adjusted label padding
- ✅ Smart annotation positioning with arrows

### Publication Quality

- **Resolution**: 300 DPI (publication standard)
- **Format**: PNG with white background
- **Fonts**: Clear, readable sizes (9-13pt)
- **Colors**: Colorblind-friendly palettes
- **Style**: Professional, consistent across all plots

---

## 🔬 Advanced Features

### Statistical Significance Testing

When comparing models, we perform:
1. **Paired T-Test**: Tests if mean errors differ significantly
2. **Wilcoxon Test**: Non-parametric alternative (robust to outliers)

**Interpretation:**
- p < 0.05: Difference is statistically significant
- p < 0.01: Highly significant
- p ≥ 0.05: Difference could be due to chance

### Sentiment Range Analysis

We bin predictions into 6 ranges:
- Very Negative (< -2)
- Negative (-2 to -1)
- Slightly Negative (-1 to 0)
- Slightly Positive (0 to 1)
- Positive (1 to 2)
- Very Positive (> 2)

This reveals if models struggle with:
- Extreme sentiments
- Neutral sentiments
- Specific intensity ranges

### Error Percentiles

Understanding error distribution:
- **50th percentile**: Half of predictions are better than this
- **90th percentile**: 90% of predictions are better than this
- **95th percentile**: Only 5% of errors are larger
- **99th percentile**: Worst 1% of predictions

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Solution: Reduce batch size
python evaluate_and_visualize.py --model model.h5 --data ./data --batch_size 16
```

**2. TensorFlow Version Mismatch**
```bash
# MSA needs TF 2.x
conda activate msa-tf2
python -c "import tensorflow as tf; print(tf.__version__)"  # Should be 2.x

# DeepHOSeq needs TF 1.x - use separate environment
```

**3. Module Not Found**
```bash
# Install missing dependencies
conda activate msa-tf2
pip install scipy scikit-learn seaborn matplotlib pandas h5py
```

**4. Model Won't Load**
```bash
# Check model file exists
ls -lh weights/*.h5

# Check model format
python -c "import tensorflow as tf; m = tf.keras.models.load_model('weights/model.h5'); print('OK')"
```

**5. Plots Look Weird**
```bash
# Update matplotlib
pip install --upgrade matplotlib seaborn

# Clear matplotlib cache
rm -rf ~/.cache/matplotlib
```

---

## 📚 Additional Resources

### Documentation Files

- **`EVALUATION_GUIDE.md`**: Complete guide with examples
- **`EVALUATION_QUICKSTART.md`**: Quick command reference
- **`SEQLEVEL_ARCHITECTURE.md`**: Model architecture details
- **`QUICKSTART.md`**: General training and usage

### Related Scripts

- **`train_seqlevel.py`**: Train MSA Seq Level model
- **`verify_data.py`**: Check data integrity
- **`visualize_training.py`**: Training progress plots

### External References

- CMU-MOSI Dataset: [Link](https://github.com/A2Zadeh/CMU-MultimodalSDK)
- CMU-MOSEI Dataset: [Link](https://github.com/A2Zadeh/CMU-MultimodalSDK)
- TensorFlow Docs: [Link](https://www.tensorflow.org/)

---

## 🎓 Citation

If you use these evaluation tools in your research, please cite:

```
Honours Project - Multimodal Sentiment Analysis
Evaluation Tools Suite v1.0
October 2025
```

---

## 📝 Changelog

**v1.0 (October 2025)**
- Initial release
- Comprehensive single-model evaluation
- Model comparison capabilities
- MSA vs DeepHOSeq specialized comparison
- Publication-ready visualizations
- Detailed metrics and reports
- Demo script and documentation

---

## 🤝 Contributing

Suggestions for improvements:
1. Additional metrics (e.g., Kendall's tau)
2. Interactive visualizations (Plotly)
3. Video/audio modality-specific analysis
4. Attention weight visualization
5. Cross-dataset evaluation

---

## 📄 License

This evaluation suite is part of the Honours Project codebase and follows the same license as the main project.

---

## 💬 Support

For issues or questions:
1. Check `EVALUATION_GUIDE.md` for detailed examples
2. Run `./demo_evaluation.sh` to see expected behavior
3. Review output examples in `evaluation_results/`

---

**Happy Evaluating! 🎉**

Remember: Good evaluation is as important as good models. These tools help you understand not just *how well* your model performs, but *why* it performs that way.

