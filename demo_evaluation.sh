#!/bin/bash

################################################################################
# Evaluation Demo Script
#
# This script demonstrates how to use the evaluation tools with an example
# trained model. It generates all visualizations and reports.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================================================="
echo "                   MSA Evaluation Tools - Demo Script"
echo "=============================================================================="
echo -e "${NC}"

# Check if we're in the right directory
if [ ! -f "evaluate_and_visualize.py" ]; then
    echo -e "${RED}Error: Please run this script from the msa-tf2 directory${NC}"
    exit 1
fi

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "msa-tf2" ]]; then
    echo -e "${YELLOW}Warning: msa-tf2 conda environment not activated${NC}"
    echo "Please run: conda activate msa-tf2"
    exit 1
fi

# Find the latest trained model
echo -e "${BLUE}Step 1: Finding trained model...${NC}"
MODEL_FILE=$(ls -t weights/seqlevel_final_*.h5 2>/dev/null | head -1)

if [ -z "$MODEL_FILE" ]; then
    echo -e "${RED}No trained model found in weights/ directory${NC}"
    echo "Please train a model first using: python train_seqlevel.py"
    exit 1
fi

echo -e "${GREEN}✓ Found model: $MODEL_FILE${NC}"

# Check if data exists
echo -e "\n${BLUE}Step 2: Checking data...${NC}"
if [ ! -f "data/y_test.h5" ]; then
    echo -e "${RED}Error: Test data not found in data/ directory${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data found${NC}"

# Create output directory
OUTPUT_DIR="./demo_evaluation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Created output directory: $OUTPUT_DIR${NC}"

################################################################################
# Demo 1: Generate Predictions
################################################################################
echo -e "\n${BLUE}"
echo "=============================================================================="
echo "  Demo 1: Generating Predictions"
echo "=============================================================================="
echo -e "${NC}"

echo "Running: python generate_predictions.py --model $MODEL_FILE --data ./data --name Demo_MSA --output $OUTPUT_DIR/predictions"
echo ""

python generate_predictions.py \
    --model "$MODEL_FILE" \
    --data ./data \
    --name Demo_MSA \
    --output "$OUTPUT_DIR/predictions" \
    --split test

echo -e "\n${GREEN}✓ Predictions generated and saved${NC}"

################################################################################
# Demo 2: Comprehensive Evaluation
################################################################################
echo -e "\n${BLUE}"
echo "=============================================================================="
echo "  Demo 2: Comprehensive Model Evaluation"
echo "=============================================================================="
echo -e "${NC}"

echo "Running: python evaluate_and_visualize.py --model $MODEL_FILE --data ./data --name Demo_MSA --output $OUTPUT_DIR/evaluation"
echo ""

python evaluate_and_visualize.py \
    --model "$MODEL_FILE" \
    --data ./data \
    --name Demo_MSA \
    --output "$OUTPUT_DIR/evaluation"

echo -e "\n${GREEN}✓ Evaluation complete${NC}"

################################################################################
# Demo 3: Training Progress Visualization
################################################################################
echo -e "\n${BLUE}"
echo "=============================================================================="
echo "  Demo 3: Training Progress Visualization"
echo "=============================================================================="
echo -e "${NC}"

# Find corresponding training log
LOG_FILE=$(ls -t weights/seqlevel_training_log_*.csv 2>/dev/null | head -1)

if [ -n "$LOG_FILE" ]; then
    echo "Found training log: $LOG_FILE"
    echo "Running: python visualize_training.py $LOG_FILE --output $OUTPUT_DIR/training_progress.png"
    echo ""
    
    python visualize_training.py "$LOG_FILE" --output "$OUTPUT_DIR/training_progress.png"
    
    echo -e "\n${GREEN}✓ Training visualization complete${NC}"
else
    echo -e "${YELLOW}⚠ No training log found, skipping training visualization${NC}"
fi

################################################################################
# Summary
################################################################################
echo -e "\n${BLUE}"
echo "=============================================================================="
echo "  Demo Complete! 🎉"
echo "=============================================================================="
echo -e "${NC}"

echo -e "\n${GREEN}All results saved to: $OUTPUT_DIR${NC}"
echo ""
echo "Generated files:"
echo "  📊 Predictions:"
find "$OUTPUT_DIR/predictions" -name "*.npy" 2>/dev/null | while read -r file; do
    echo "     - $(basename "$file")"
done

echo ""
echo "  📈 Evaluation Visualizations:"
find "$OUTPUT_DIR/evaluation" -name "*.png" 2>/dev/null | while read -r file; do
    echo "     - $(basename "$file")"
done

echo ""
echo "  📄 Reports:"
find "$OUTPUT_DIR/evaluation" -name "*.txt" -o -name "*.json" 2>/dev/null | while read -r file; do
    echo "     - $(basename "$file")"
done

echo ""
echo -e "${BLUE}Quick view commands:${NC}"
echo "  # View prediction scatter plot"
echo "  open $OUTPUT_DIR/evaluation/Demo_MSA_prediction_scatter_*.png"
echo ""
echo "  # View error analysis"
echo "  open $OUTPUT_DIR/evaluation/Demo_MSA_error_analysis_*.png"
echo ""
echo "  # Read metrics report"
echo "  cat $OUTPUT_DIR/evaluation/Demo_MSA_report_*.txt"
echo ""
echo "  # View metrics JSON"
echo "  cat $OUTPUT_DIR/evaluation/Demo_MSA_metrics_*.json | python -m json.tool"

echo -e "\n${BLUE}Next steps:${NC}"
echo "  • Review the visualizations to understand model performance"
echo "  • Check the text report for detailed metrics"
echo "  • Compare with other models using compare_msa_deephoseq.py"
echo "  • See EVALUATION_GUIDE.md for more details"

echo -e "\n${GREEN}Demo completed successfully!${NC}"
echo ""

