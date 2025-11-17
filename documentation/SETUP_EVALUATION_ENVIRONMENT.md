# Setup Guide for MSA Evaluation Environment

This guide helps you install all required packages to run the evaluation tools.

---

## Required Packages

The evaluation tools require the following packages:

### Core Dependencies
- **numpy** (≥1.21.0) - Numerical operations
- **pandas** (≥1.3.0) - Data manipulation
- **scipy** (≥1.7.0) - Statistical functions
- **scikit-learn** (≥1.0.0) - Machine learning metrics
- **matplotlib** (≥3.4.0) - Plotting
- **seaborn** (≥0.11.0) - Statistical visualizations
- **h5py** (≥3.1.0) - HDF5 file handling
- **tensorflow** (≥2.13.0) - Deep learning framework

### Optional
- **transformers** (≥4.0.0) - For text processing (already in your environment)
- **tqdm** (≥4.62.0) - Progress bars

---

## Installation Methods

Choose the method that works best for you:

### Method 1: Update Existing Conda Environment (Recommended)

If you already have the `msa-tf2` conda environment:

```bash
# Activate your environment
conda activate msa-tf2

# Install missing packages
pip install scipy scikit-learn seaborn

# Verify installation
python -c "import scipy, sklearn, seaborn; print('✓ All packages installed!')"
```

### Method 2: Install from Requirements File

```bash
# Activate environment
conda activate msa-tf2

# Install all packages
pip install -r requirements_evaluation.txt

# Verify
python -c "import numpy, pandas, scipy, sklearn, matplotlib, seaborn, h5py; print('✓ Success!')"
```

### Method 3: Create New Conda Environment

If starting fresh:

#### For Apple Silicon (M1/M2/M3 Mac):

```bash
# Create environment
conda create -n msa-tf2 python=3.10 -y

# Activate
conda activate msa-tf2

# Install conda packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn h5py tqdm -y

# Install TensorFlow for Apple Silicon
pip install tensorflow-macos>=2.13.0 tensorflow-metal

# Install additional packages
pip install transformers

# Verify
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import scipy, sklearn, seaborn; print('✓ All packages ready!')"
```

#### For Linux/Windows/Intel Mac:

```bash
# Create environment
conda create -n msa-tf2 python=3.10 -y

# Activate
conda activate msa-tf2

# Install all packages
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib seaborn h5py tqdm -y
pip install tensorflow>=2.13.0 transformers

# Verify
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import scipy, sklearn, seaborn; print('✓ All packages ready!')"
```

#### Using the Environment File:

```bash
# Edit environment_evaluation.yml first to uncomment your platform
# Then create:
conda env create -f environment_evaluation.yml

# Or update existing:
conda env update -f environment_evaluation.yml --prune
```

---

## Quick Install (One Command)

For the impatient (assumes existing msa-tf2 environment):

```bash
conda activate msa-tf2 && pip install scipy scikit-learn seaborn && echo "✓ Ready!"
```

---

## Verification

Run this comprehensive check:

```bash
conda activate msa-tf2

python << 'EOF'
import sys
print("="*70)
print("Package Verification for MSA Evaluation Tools")
print("="*70)

packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'h5py': 'h5py',
    'tensorflow': 'tensorflow',
}

all_good = True
for name, import_name in packages.items():
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {name:20s} {version}")
    except ImportError:
        print(f"✗ {name:20s} NOT INSTALLED")
        all_good = False

print("="*70)
if all_good:
    print("✓ All packages installed successfully!")
    print("\nYou're ready to run:")
    print("  ./demo_evaluation.sh")
    print("  python evaluate_and_visualize.py --help")
else:
    print("✗ Some packages are missing. Please install them.")
    sys.exit(1)
EOF
```

---

## What Each Package Does

| Package | Purpose in Evaluation Tools |
|---------|----------------------------|
| **numpy** | Array operations, numerical computations |
| **pandas** | Data manipulation, CSV reading |
| **scipy** | Statistical tests (t-test, Wilcoxon), correlations |
| **scikit-learn** | Metrics (MAE, RMSE, confusion matrix, F1, etc.) |
| **matplotlib** | Base plotting library |
| **seaborn** | Beautiful statistical visualizations, styling |
| **h5py** | Reading HDF5 data files |
| **tensorflow** | Loading models, generating predictions |

---

## Troubleshooting

### "No module named 'scipy'"

```bash
conda activate msa-tf2
pip install scipy
```

### "No module named 'sklearn'"

```bash
conda activate msa-tf2
pip install scikit-learn
```

### "No module named 'seaborn'"

```bash
conda activate msa-tf2
pip install seaborn
```

### TensorFlow Issues

**Apple Silicon:**
```bash
pip install tensorflow-macos tensorflow-metal
```

**Other platforms:**
```bash
pip install tensorflow
```

### Import Errors After Installation

```bash
# Restart Python kernel or terminal
# Or reinstall in fresh environment
conda deactivate
conda activate msa-tf2
```

### Version Conflicts

```bash
# Update all packages
conda activate msa-tf2
pip install --upgrade numpy pandas scipy scikit-learn matplotlib seaborn h5py
```

---

## Testing the Installation

After installation, test the evaluation tools:

```bash
# Activate environment
conda activate msa-tf2

# Run the demo (if you have a trained model)
./demo_evaluation.sh

# Or test individual scripts
python evaluate_and_visualize.py --help
python generate_predictions.py --help
python compare_msa_deephoseq.py --help
```

---

## Minimal Installation

If you only want to run evaluation (not training):

```bash
conda activate msa-tf2
pip install scipy scikit-learn seaborn
```

This adds only the 3 packages needed beyond what's already in your training environment.

---

## Package Sizes (Approximate)

- scipy: ~50 MB
- scikit-learn: ~30 MB
- seaborn: ~5 MB
- **Total new packages: ~85 MB**

---

## Environment Export

After successful setup, save your environment:

```bash
# Export to file
conda env export > environment_msa_complete.yml

# Or just pip packages
pip freeze > requirements_complete.txt
```

---

## Quick Reference

```bash
# Activate environment
conda activate msa-tf2

# Install evaluation packages
pip install scipy scikit-learn seaborn

# Verify
python -c "import scipy, sklearn, seaborn; print('Ready!')"

# Run demo
./demo_evaluation.sh
```

---

## Already Installed in Your Environment

Based on your existing `requirements.txt`:
- ✓ tensorflow-macos
- ✓ tensorflow-metal
- ✓ h5py
- ✓ numpy
- ✓ pandas
- ✓ tqdm
- ✓ transformers

**You only need to add:**
- scipy
- scikit-learn  
- seaborn
- matplotlib (should be installed, but verify)

---

## Ready to Go!

Once installed, start with:

```bash
# Quick verification
conda activate msa-tf2
python -c "import scipy, sklearn, seaborn, matplotlib; print('✓ Ready!')"

# Run the demo
./demo_evaluation.sh

# Or evaluate directly
python evaluate_and_visualize.py \
    --model weights/seqlevel_final_20251019_010827.h5 \
    --data ./data \
    --name MSA_SeqLevel
```

Happy evaluating! 🎉

