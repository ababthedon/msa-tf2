#!/usr/bin/env python3
"""
MSA Seq Level vs DeepHOSeq Comparison Script

This script facilitates comparison between MSA Seq Level (TF2) and DeepHOSeq (TF1) models.
It handles the different frameworks and generates comparative visualizations.
"""

import numpy as np
import h5py
import argparse
import os
import sys
import json
from datetime import datetime

# Import evaluation functions
from evaluate_and_visualize import (
    generate_comprehensive_report,
    compare_models,
    calculate_regression_metrics,
    calculate_classification_metrics
)

import matplotlib.pyplot as plt


def load_msa_predictions(model_path, data_dir, split='test'):
    """Load predictions from MSA Seq Level model (TF2)."""
    import tensorflow as tf
    from utils.data_loader import make_dataset
    
    print(f"\nLoading MSA Seq Level model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading {split} dataset from: {data_dir}")
    dataset = make_dataset(data_dir, split=split, batch_size=32)
    
    print("Generating MSA predictions...")
    y_true_list = []
    y_pred_list = []
    
    for batch_data, batch_labels in dataset:
        predictions = model.predict_on_batch(batch_data)
        y_true_list.append(batch_labels.numpy())
        y_pred_list.append(predictions)
    
    y_true = np.concatenate(y_true_list, axis=0).flatten()
    y_pred = np.concatenate(y_pred_list, axis=0).flatten()
    
    print(f"✓ Loaded {len(y_true)} MSA predictions")
    return y_true, y_pred


def load_deephoseq_predictions(checkpoint_path, data_dir):
    """Load predictions from DeepHOSeq model (TF1)."""
    # This requires TF1 which might not be compatible with TF2 environment
    # So we provide alternative methods
    
    print(f"\nLoading DeepHOSeq model: {checkpoint_path}")
    print("Note: DeepHOSeq uses TensorFlow 1.x")
    
    # Check if we're in TF2 environment
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        if tf_version.startswith('2'):
            print(f"⚠ Warning: Running TensorFlow {tf_version}")
            print("DeepHOSeq requires TensorFlow 1.x for full compatibility")
            print("\nOptions:")
            print("1. Generate predictions in DeepHOSeq environment and save to .npy")
            print("2. Use pre-saved predictions")
            print("\nTo generate predictions, run in Deephoseq directory:")
            print("  python evaluate.py --checkpoint_path <path> --data_dir <path>")
            print("  # Then save predictions with np.save()")
            return None, None
    except:
        pass
    
    # If TF1 is available, proceed
    # Note: This is a simplified version - actual implementation would be more complex
    print("✗ Cannot load DeepHOSeq in TF2 environment")
    print("Please provide pre-generated predictions using --deephoseq_pred")
    return None, None


def load_predictions_from_file(y_true_path, y_pred_path, model_name):
    """Load pre-saved predictions."""
    print(f"\nLoading {model_name} predictions from files...")
    
    if y_pred_path.endswith('.npy'):
        y_true = np.load(y_true_path).flatten()
        y_pred = np.load(y_pred_path).flatten()
    elif y_pred_path.endswith('.h5'):
        with h5py.File(y_true_path, 'r') as f:
            y_true = f['data'][:].flatten()
        with h5py.File(y_pred_path, 'r') as f:
            y_pred = f['data'][:].flatten()
    else:
        raise ValueError("Unsupported file format. Use .npy or .h5")
    
    print(f"✓ Loaded {len(y_true)} predictions")
    return y_true, y_pred


def save_predictions_helper(y_true, y_pred, output_dir, model_name):
    """Helper to save predictions for later use."""
    os.makedirs(output_dir, exist_ok=True)
    
    true_path = os.path.join(output_dir, f'{model_name}_y_true.npy')
    pred_path = os.path.join(output_dir, f'{model_name}_y_pred.npy')
    
    np.save(true_path, y_true)
    np.save(pred_path, y_pred)
    
    print(f"\nSaved predictions:")
    print(f"  True: {true_path}")
    print(f"  Pred: {pred_path}")
    
    return true_path, pred_path


def print_comparison_summary(metrics_msa, metrics_deephoseq):
    """Print a nice comparison summary."""
    print("\n" + "="*80)
    print(" "*20 + "MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'MSA Seq Level':<20} {'DeepHOSeq':<20} {'Winner':<15}")
    print("-"*80)
    
    comparisons = [
        ('MAE', 'mae', 'lower'),
        ('RMSE', 'rmse', 'lower'),
        ('R²', 'r2', 'higher'),
        ('Pearson r', 'pearson_r', 'higher'),
        ('Binary Accuracy', 'binary_accuracy', 'higher'),
        ('7-Class Accuracy', '7class_accuracy', 'higher'),
    ]
    
    for display_name, key, better in comparisons:
        val_msa = metrics_msa.get(key, 0)
        val_deep = metrics_deephoseq.get(key, 0)
        
        if better == 'lower':
            winner = 'MSA' if val_msa < val_deep else 'DeepHOSeq'
            improvement = ((val_deep - val_msa) / val_deep * 100) if val_deep != 0 else 0
        else:
            winner = 'MSA' if val_msa > val_deep else 'DeepHOSeq'
            improvement = ((val_msa - val_deep) / val_deep * 100) if val_deep != 0 else 0
        
        print(f"{display_name:<25} {val_msa:<20.4f} {val_deep:<20.4f} {winner:<15}")
        if abs(improvement) > 0.1:
            print(f"{'':>25} (Improvement: {abs(improvement):.2f}%)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare MSA Seq Level and DeepHOSeq Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using MSA model and pre-saved DeepHOSeq predictions
  python compare_msa_deephoseq.py --msa_model weights/seqlevel_final.h5 \\
                                   --deephoseq_pred deephoseq_predictions.npy \\
                                   --deephoseq_true y_test.npy \\
                                   --data ./data
  
  # Compare using pre-saved predictions for both
  python compare_msa_deephoseq.py --msa_pred msa_pred.npy \\
                                   --msa_true y_test.npy \\
                                   --deephoseq_pred deep_pred.npy \\
                                   --deephoseq_true y_test.npy
  
  # Generate MSA predictions and save for later
  python compare_msa_deephoseq.py --msa_model weights/seqlevel_final.h5 \\
                                   --data ./data \\
                                   --save_msa_pred
        """
    )
    
    # MSA Seq Level options
    msa_group = parser.add_argument_group('MSA Seq Level Model')
    msa_group.add_argument('--msa_model', type=str,
                          help='Path to MSA Seq Level model (.h5)')
    msa_group.add_argument('--msa_pred', type=str,
                          help='Path to pre-saved MSA predictions (.npy)')
    msa_group.add_argument('--msa_true', type=str,
                          help='Path to ground truth for MSA (.npy)')
    
    # DeepHOSeq options
    deep_group = parser.add_argument_group('DeepHOSeq Model')
    deep_group.add_argument('--deephoseq_pred', type=str,
                           help='Path to pre-saved DeepHOSeq predictions (.npy)')
    deep_group.add_argument('--deephoseq_true', type=str,
                           help='Path to ground truth for DeepHOSeq (.npy)')
    
    # Common options
    parser.add_argument('--data', type=str, default='./data',
                       help='Data directory (for MSA model)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Data split (default: test)')
    parser.add_argument('--output', type=str, default='./comparison_results',
                       help='Output directory')
    parser.add_argument('--save_msa_pred', action='store_true',
                       help='Save MSA predictions to .npy files')
    parser.add_argument('--save_deephoseq_pred', action='store_true',
                       help='Save DeepHOSeq predictions to .npy files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "#"*80)
    print("#" + " "*20 + "MSA Seq Level vs DeepHOSeq Comparison")
    print("#"*80)
    
    # Load MSA predictions
    y_true_msa, y_pred_msa = None, None
    if args.msa_model:
        y_true_msa, y_pred_msa = load_msa_predictions(args.msa_model, args.data, args.split)
        if args.save_msa_pred and y_pred_msa is not None:
            save_predictions_helper(y_true_msa, y_pred_msa, args.output, 'msa_seqlevel')
    elif args.msa_pred and args.msa_true:
        y_true_msa, y_pred_msa = load_predictions_from_file(
            args.msa_true, args.msa_pred, 'MSA Seq Level'
        )
    else:
        print("Error: Provide either --msa_model or (--msa_pred and --msa_true)")
        sys.exit(1)
    
    # Load DeepHOSeq predictions
    y_true_deep, y_pred_deep = None, None
    if args.deephoseq_pred and args.deephoseq_true:
        y_true_deep, y_pred_deep = load_predictions_from_file(
            args.deephoseq_true, args.deephoseq_pred, 'DeepHOSeq'
        )
    else:
        print("\nError: Provide --deephoseq_pred and --deephoseq_true")
        print("\nTo generate DeepHOSeq predictions:")
        print("1. Navigate to deephoseq directory")
        print("2. Run evaluation and save predictions:")
        print("   python -c \"")
        print("   import numpy as np")
        print("   # ... run evaluation ...")
        print("   np.save('deephoseq_pred.npy', predictions)")
        print("   np.save('y_test.npy', y_test)")
        print("   \"")
        sys.exit(1)
    
    # Verify predictions are valid
    if y_pred_msa is None or y_pred_deep is None:
        print("Error: Could not load predictions")
        sys.exit(1)
    
    # Check if ground truths match (they should!)
    if not np.allclose(y_true_msa, y_true_deep, atol=1e-5):
        print("\n⚠ Warning: Ground truth values differ between models!")
        print(f"MSA ground truth: mean={np.mean(y_true_msa):.4f}, samples={len(y_true_msa)}")
        print(f"Deep ground truth: mean={np.mean(y_true_deep):.4f}, samples={len(y_true_deep)}")
        print("Using MSA ground truth for comparison...")
    
    y_true = y_true_msa  # Use MSA ground truth as reference
    
    # Calculate metrics for both models
    print("\n" + "="*80)
    print("Calculating metrics...")
    print("="*80)
    
    print("\nMSA Seq Level:")
    metrics_msa = {
        **calculate_regression_metrics(y_true, y_pred_msa),
        **calculate_classification_metrics(y_true, y_pred_msa)
    }
    
    print("\nDeepHOSeq:")
    metrics_deephoseq = {
        **calculate_regression_metrics(y_true, y_pred_deep),
        **calculate_classification_metrics(y_true, y_pred_deep)
    }
    
    # Print comparison summary
    print_comparison_summary(metrics_msa, metrics_deephoseq)
    
    # Generate individual reports
    print("\n" + "="*80)
    print("Generating individual model reports...")
    print("="*80)
    
    print("\n1. MSA Seq Level Report:")
    generate_comprehensive_report(y_true, y_pred_msa, 'MSA_SeqLevel', args.output)
    
    print("\n2. DeepHOSeq Report:")
    generate_comprehensive_report(y_true, y_pred_deep, 'DeepHOSeq', args.output)
    
    # Generate comparison visualization
    print("\n" + "="*80)
    print("Generating comparison visualization...")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(args.output, f'MSA_vs_DeepHOSeq_{timestamp}.png')
    compare_models(
        'MSA Seq Level', y_pred_msa,
        'DeepHOSeq', y_pred_deep,
        y_true,
        save_path=comparison_path
    )
    plt.close()
    
    # Save comparison metrics
    comparison_data = {
        'timestamp': timestamp,
        'msa_seqlevel': metrics_msa,
        'deephoseq': metrics_deephoseq,
        'improvement': {
            'mae': ((metrics_deephoseq['mae'] - metrics_msa['mae']) / metrics_deephoseq['mae'] * 100),
            'rmse': ((metrics_deephoseq['rmse'] - metrics_msa['rmse']) / metrics_deephoseq['rmse'] * 100),
            'r2': ((metrics_msa['r2'] - metrics_deephoseq['r2']) / abs(metrics_deephoseq['r2']) * 100),
        }
    }
    
    comparison_json = os.path.join(args.output, f'comparison_metrics_{timestamp}.json')
    with open(comparison_json, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison metrics saved: {comparison_json}")
    
    # Create summary report
    summary_path = os.path.join(args.output, f'comparison_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MSA SEQ LEVEL vs DEEPHOSEQ COMPARISON SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODEL CONFIGURATIONS\n")
        f.write("-"*80 + "\n")
        f.write("MSA Seq Level:\n")
        f.write("  - Architecture: Transformer-based with cross-attention fusion\n")
        f.write("  - Framework: TensorFlow 2.x\n")
        if args.msa_model:
            f.write(f"  - Model: {args.msa_model}\n")
        f.write("\nDeepHOSeq:\n")
        f.write("  - Architecture: LSTM-based with tensor fusion\n")
        f.write("  - Framework: TensorFlow 1.x\n")
        if args.deephoseq_pred:
            f.write(f"  - Predictions: {args.deephoseq_pred}\n")
        
        f.write("\n\nPERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<25} {'MSA Seq Level':<20} {'DeepHOSeq':<20} {'Winner'}\n")
        f.write("-"*80 + "\n")
        
        for display_name, key, better in [
            ('MAE', 'mae', 'lower'),
            ('RMSE', 'rmse', 'lower'),
            ('R²', 'r2', 'higher'),
            ('Pearson r', 'pearson_r', 'higher'),
            ('Binary Accuracy', 'binary_accuracy', 'higher'),
            ('Binary F1', 'binary_f1', 'higher'),
            ('7-Class Accuracy', '7class_accuracy', 'higher'),
        ]:
            val_msa = metrics_msa.get(key, 0)
            val_deep = metrics_deephoseq.get(key, 0)
            
            if better == 'lower':
                winner = 'MSA Seq Level' if val_msa < val_deep else 'DeepHOSeq'
            else:
                winner = 'MSA Seq Level' if val_msa > val_deep else 'DeepHOSeq'
            
            f.write(f"{display_name:<25} {val_msa:<20.4f} {val_deep:<20.4f} {winner}\n")
        
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-"*80 + "\n")
        
        mae_improvement = ((metrics_deephoseq['mae'] - metrics_msa['mae']) / metrics_deephoseq['mae'] * 100)
        if mae_improvement > 0:
            f.write(f"✓ MSA Seq Level achieves {mae_improvement:.2f}% better MAE\n")
        else:
            f.write(f"✗ DeepHOSeq achieves {abs(mae_improvement):.2f}% better MAE\n")
        
        r2_diff = metrics_msa['r2'] - metrics_deephoseq['r2']
        if r2_diff > 0:
            f.write(f"✓ MSA Seq Level has {r2_diff:.4f} higher R² score\n")
        else:
            f.write(f"✗ DeepHOSeq has {abs(r2_diff):.4f} higher R² score\n")
        
        binary_acc_diff = (metrics_msa['binary_accuracy'] - metrics_deephoseq['binary_accuracy']) * 100
        f.write(f"{'✓' if binary_acc_diff > 0 else '✗'} Binary accuracy difference: {binary_acc_diff:+.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Summary report saved: {summary_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output}")
    print("\nGenerated files:")
    print("  - Individual model reports (PNG + metrics)")
    print("  - Comparison visualization")
    print("  - Comparison metrics (JSON)")
    print("  - Summary report (TXT)")


if __name__ == '__main__':
    main()

