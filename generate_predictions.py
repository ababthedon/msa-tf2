#!/usr/bin/env python3
"""
Generate and Save Predictions Script

Helper script to generate predictions from a trained model and save them
for later evaluation and comparison.
"""

import numpy as np
import h5py
import tensorflow as tf
import argparse
import os
from datetime import datetime
from utils.data_loader import make_dataset


def generate_predictions(model_path, data_dir, split='test', batch_size=32):
    """
    Generate predictions from a trained model.
    
    Args:
        model_path: Path to saved model
        data_dir: Directory containing data
        split: 'train', 'valid', or 'test'
        batch_size: Batch size for prediction
    
    Returns:
        y_true, y_pred: Arrays of ground truth and predictions
    """
    print("\n" + "="*70)
    print("Generating Predictions")
    print("="*70)
    
    print(f"\nLoading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading {split} dataset from: {data_dir}")
    dataset = make_dataset(data_dir, split=split, batch_size=batch_size)
    
    print(f"Generating predictions (batch_size={batch_size})...")
    
    y_true_list = []
    y_pred_list = []
    batch_count = 0
    
    for batch_data, batch_labels in dataset:
        predictions = model.predict_on_batch(batch_data)
        y_true_list.append(batch_labels.numpy())
        y_pred_list.append(predictions)
        batch_count += 1
        
        # Progress indicator
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches...", end='\r')
    
    print(f"  Processed {batch_count} batches... Done!")
    
    y_true = np.concatenate(y_true_list, axis=0).flatten()
    y_pred = np.concatenate(y_pred_list, axis=0).flatten()
    
    print(f"\n✓ Generated {len(y_true)} predictions")
    
    # Print basic statistics
    print("\nPrediction Statistics:")
    print(f"  Ground Truth - Mean: {np.mean(y_true):.4f}, Std: {np.std(y_true):.4f}")
    print(f"  Ground Truth - Min: {np.min(y_true):.4f}, Max: {np.max(y_true):.4f}")
    print(f"  Predictions  - Mean: {np.mean(y_pred):.4f}, Std: {np.std(y_pred):.4f}")
    print(f"  Predictions  - Min: {np.min(y_pred):.4f}, Max: {np.max(y_pred):.4f}")
    
    # Quick error metrics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"\nQuick Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return y_true, y_pred


def save_predictions(y_true, y_pred, output_dir, model_name, split='test', format='npy'):
    """
    Save predictions to file.
    
    Args:
        y_true: Ground truth values
        y_pred: Predictions
        output_dir: Output directory
        model_name: Model name for file naming
        split: Data split name
        format: 'npy' or 'h5'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'npy':
        true_path = os.path.join(output_dir, f'{model_name}_{split}_true_{timestamp}.npy')
        pred_path = os.path.join(output_dir, f'{model_name}_{split}_pred_{timestamp}.npy')
        
        np.save(true_path, y_true)
        np.save(pred_path, y_pred)
        
    elif format == 'h5':
        true_path = os.path.join(output_dir, f'{model_name}_{split}_true_{timestamp}.h5')
        pred_path = os.path.join(output_dir, f'{model_name}_{split}_pred_{timestamp}.h5')
        
        with h5py.File(true_path, 'w') as f:
            f.create_dataset('data', data=y_true)
        
        with h5py.File(pred_path, 'w') as f:
            f.create_dataset('data', data=y_pred)
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'npy' or 'h5'")
    
    print("\n" + "="*70)
    print("Predictions Saved")
    print("="*70)
    print(f"Ground Truth: {true_path}")
    print(f"Predictions:  {pred_path}")
    
    return true_path, pred_path


def load_predictions_from_h5_data(data_dir, split='test'):
    """
    Load ground truth directly from h5 data files.
    
    Args:
        data_dir: Directory containing h5 files
        split: 'train', 'valid', or 'test'
    
    Returns:
        y_true: Ground truth values
    """
    h5_path = os.path.join(data_dir, f'y_{split}.h5')
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Ground truth file not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        y_true = f['data'][:].flatten()
    
    print(f"Loaded {len(y_true)} ground truth values from {h5_path}")
    return y_true


def main():
    parser = argparse.ArgumentParser(
        description='Generate and save predictions from a trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate test predictions
  python generate_predictions.py --model weights/seqlevel_final.h5 \\
                                  --data ./data \\
                                  --name MSA_SeqLevel
  
  # Generate validation predictions and save as HDF5
  python generate_predictions.py --model weights/model.h5 \\
                                  --data ./data \\
                                  --split valid \\
                                  --format h5
  
  # Generate predictions with custom batch size
  python generate_predictions.py --model weights/model.h5 \\
                                  --data ./data \\
                                  --batch_size 16
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5)')
    parser.add_argument('--data', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'valid', 'test'],
                       help='Data split (default: test)')
    parser.add_argument('--name', type=str, default='model',
                       help='Model name for file naming (default: model)')
    parser.add_argument('--output', type=str, default='./predictions',
                       help='Output directory (default: ./predictions)')
    parser.add_argument('--format', type=str, default='npy',
                       choices=['npy', 'h5'],
                       help='Output format (default: npy)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation after generating predictions')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Check if data directory exists
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found: {args.data}")
        return 1
    
    print("\n" + "#"*70)
    print("#" + " "*20 + "Generate Predictions")
    print("#"*70)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    
    # Generate predictions
    try:
        y_true, y_pred = generate_predictions(
            args.model, args.data, args.split, args.batch_size
        )
    except Exception as e:
        print(f"\nError generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save predictions
    try:
        true_path, pred_path = save_predictions(
            y_true, y_pred, args.output, args.name, args.split, args.format
        )
    except Exception as e:
        print(f"\nError saving predictions: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Optionally run evaluation
    if args.evaluate:
        print("\n" + "="*70)
        print("Running Evaluation")
        print("="*70)
        
        try:
            from evaluate_and_visualize import generate_comprehensive_report
            
            eval_output_dir = os.path.join(args.output, 'evaluation')
            generate_comprehensive_report(y_true, y_pred, args.name, eval_output_dir)
            
            print(f"\n✓ Evaluation complete! Results in: {eval_output_dir}")
        except Exception as e:
            print(f"\nError running evaluation: {e}")
            print("You can run evaluation manually with:")
            print(f"  python evaluate_and_visualize.py --y_true {true_path} --y_pred {pred_path} --name {args.name}")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70)
    print("\nTo evaluate these predictions later, run:")
    print(f"  python evaluate_and_visualize.py --y_true {true_path} --y_pred {pred_path} --name {args.name}")
    
    print("\nTo compare with another model:")
    print(f"  python compare_msa_deephoseq.py --msa_pred {pred_path} --msa_true {true_path} \\")
    print(f"                                   --deephoseq_pred <other_pred> --deephoseq_true <other_true>")
    
    return 0


if __name__ == '__main__':
    exit(main())

