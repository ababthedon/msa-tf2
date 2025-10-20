"""
Enhanced metrics for multimodal sentiment analysis.
Includes correlation metrics, classification accuracy, and robustness measures.
"""

import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report


class SentimentMetrics:
    """Collection of metrics specifically designed for sentiment analysis evaluation."""
    
    @staticmethod
    def pearson_correlation(y_true, y_pred):
        """
        Pearson correlation coefficient between predictions and true values.
        Standard metric in multimodal sentiment analysis literature.
        """
        def _pearson_correlation_numpy(y_true_np, y_pred_np):
            y_true_flat = y_true_np.flatten()
            y_pred_flat = y_pred_np.flatten()
            
            # Handle edge cases
            if len(y_true_flat) < 2:
                return 0.0
            
            try:
                corr, _ = pearsonr(y_true_flat, y_pred_flat)
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0
        
        return tf.py_function(
            func=_pearson_correlation_numpy,
            inp=[y_true, y_pred],
            Tout=tf.float32
        )
    
    @staticmethod
    def sentiment_binary_accuracy(y_true, y_pred):
        """
        Binary classification accuracy (positive vs negative sentiment).
        Useful for practical applications where fine-grained sentiment isn't needed.
        """
        # Convert to binary classes (positive vs negative)
        y_true_binary = tf.cast(tf.greater(y_true, 0), tf.float32)
        y_pred_binary = tf.cast(tf.greater(y_pred, 0), tf.float32)
        
        # Calculate accuracy
        correct_predictions = tf.cast(tf.equal(y_true_binary, y_pred_binary), tf.float32)
        return tf.reduce_mean(correct_predictions)
    
    @staticmethod
    def sentiment_7class_accuracy(y_true, y_pred):
        """
        7-class sentiment accuracy by rounding to nearest integer.
        Converts continuous sentiment [-3, 3] to discrete classes.
        """
        # Round to nearest integer for classification
        y_true_class = tf.round(y_true)
        y_pred_class = tf.round(y_pred)
        
        # Calculate accuracy
        correct_predictions = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
        return tf.reduce_mean(correct_predictions)
    
    @staticmethod
    def sentiment_5class_accuracy(y_true, y_pred):
        """
        5-class sentiment accuracy: Very Negative, Negative, Neutral, Positive, Very Positive
        Maps [-3,-3] to classes [0,1,2,3,4]
        """
        def _to_5class(values):
            # Map [-3, 3] to [0, 4] classes
            # [-3, -1.5) -> 0, [-1.5, -0.5) -> 1, [-0.5, 0.5) -> 2, [0.5, 1.5) -> 3, [1.5, 3] -> 4
            conditions = [
                values < -1.5,
                tf.logical_and(values >= -1.5, values < -0.5),
                tf.logical_and(values >= -0.5, values < 0.5),
                tf.logical_and(values >= 0.5, values < 1.5),
                values >= 1.5
            ]
            choices = [0, 1, 2, 3, 4]
            return tf.cast(tf.stack([tf.where(cond, choice, 0) for cond, choice in zip(conditions, choices)]), tf.float32)
        
        y_true_class = tf.reduce_sum(_to_5class(y_true), axis=0)
        y_pred_class = tf.reduce_sum(_to_5class(y_pred), axis=0)
        
        correct_predictions = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
        return tf.reduce_mean(correct_predictions)
    
    @staticmethod
    def explained_variance_score(y_true, y_pred):
        """
        Explained variance score: 1 - Var(y_true - y_pred) / Var(y_true)
        Measures how well the model explains the variance in the data.
        """
        y_true_mean = tf.reduce_mean(y_true)
        total_variance = tf.reduce_mean(tf.square(y_true - y_true_mean))
        residual_variance = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Avoid division by zero
        total_variance = tf.maximum(total_variance, 1e-8)
        
        return 1.0 - (residual_variance / total_variance)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        """
        MAPE: Mean Absolute Percentage Error
        Useful for understanding relative error magnitude.
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        percentage_error = tf.abs((y_true - y_pred) / (tf.abs(y_true) + epsilon))
        return tf.reduce_mean(percentage_error) * 100.0
    
    @staticmethod
    def prediction_range_coverage(y_true, y_pred):
        """
        Ratio of prediction range to true value range.
        Indicates if model is making predictions across the full sentiment spectrum.
        """
        true_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
        pred_range = tf.reduce_max(y_pred) - tf.reduce_min(y_pred)
        
        # Avoid division by zero
        true_range = tf.maximum(true_range, 1e-8)
        
        return pred_range / true_range
    
    @staticmethod
    def sentiment_direction_accuracy(y_true, y_pred):
        """
        Accuracy of sentiment direction relative to neutral (0).
        Checks if model correctly identifies stronger vs weaker sentiment.
        """
        # For each pair, check if the relative ordering is correct
        # This is more complex in TensorFlow, simplified version:
        y_true_centered = y_true - tf.reduce_mean(y_true)
        y_pred_centered = y_pred - tf.reduce_mean(y_pred)
        
        # Check if signs match after centering
        same_direction = tf.cast(
            tf.equal(tf.sign(y_true_centered), tf.sign(y_pred_centered)), 
            tf.float32
        )
        
        return tf.reduce_mean(same_direction)


def get_all_sentiment_metrics():
    """
    Returns a list of all sentiment analysis metrics for model compilation.
    """
    return [
        'mae',                                              # Standard L1 loss
        'mse',                                              # Standard L2 loss
        SentimentMetrics.pearson_correlation,               # Literature standard
        SentimentMetrics.sentiment_binary_accuracy,         # Positive/negative accuracy
        SentimentMetrics.sentiment_7class_accuracy,         # Fine-grained accuracy
        SentimentMetrics.sentiment_5class_accuracy,         # Coarse-grained accuracy
        SentimentMetrics.explained_variance_score,          # Model fit quality
        SentimentMetrics.mean_absolute_percentage_error,    # Relative error
        SentimentMetrics.prediction_range_coverage,         # Range coverage
        SentimentMetrics.sentiment_direction_accuracy       # Direction accuracy
    ]


def get_core_sentiment_metrics():
    """
    Returns core metrics recommended for MSA-TF2 model.
    Balances comprehensiveness with computational efficiency.
    """
    return [
        'mae',                                              # Primary optimization metric
        'mse',                                              # Secondary regression metric
        SentimentMetrics.pearson_correlation,               # Literature comparison
        SentimentMetrics.sentiment_binary_accuracy,         # Practical classification
        SentimentMetrics.explained_variance_score           # Model quality
    ]


def comprehensive_evaluation(model, test_dataset, verbose=True):
    """
    Perform comprehensive evaluation of the model on test data.
    
    Args:
        model: Trained TensorFlow model
        test_dataset: TensorFlow dataset for testing
        verbose: Whether to print detailed results
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    # Collect all predictions and true values
    y_true_list = []
    y_pred_list = []
    
    for batch in test_dataset:
        inputs, y_true_batch = batch
        y_pred_batch = model(inputs, training=False)
        
        y_true_list.append(y_true_batch.numpy())
        y_pred_list.append(y_pred_batch.numpy())
    
    # Concatenate all batches
    y_true = np.concatenate(y_true_list, axis=0).flatten()
    y_pred = np.concatenate(y_pred_list, axis=0).flatten()
    
    # Calculate all metrics
    metrics = {}
    
    # Regression metrics
    metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    metrics['mse'] = np.mean((y_true - y_pred)**2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Correlation metrics
    if len(y_true) > 1:
        try:
            metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true, y_pred)
        except:
            metrics['pearson_r'], metrics['pearson_p'] = 0.0, 1.0
    else:
        metrics['pearson_r'], metrics['pearson_p'] = 0.0, 1.0
    
    # Classification metrics
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    metrics['binary_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    
    y_true_7class = np.round(y_true).astype(int)
    y_pred_7class = np.round(y_pred).astype(int)
    metrics['7class_accuracy'] = accuracy_score(y_true_7class, y_pred_7class)
    
    # Additional metrics
    if np.var(y_true) > 1e-8:
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
    else:
        metrics['explained_variance'] = 0.0
    
    metrics['mean_absolute_percentage_error'] = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    true_range = np.max(y_true) - np.min(y_true)
    pred_range = np.max(y_pred) - np.min(y_pred)
    metrics['range_coverage'] = pred_range / max(true_range, 1e-8)
    
    # Distribution statistics
    metrics['prediction_stats'] = {
        'mean': np.mean(y_pred),
        'std': np.std(y_pred),
        'min': np.min(y_pred),
        'max': np.max(y_pred),
        'range': pred_range
    }
    
    metrics['true_stats'] = {
        'mean': np.mean(y_true),
        'std': np.std(y_true),
        'min': np.min(y_true),
        'max': np.max(y_true),
        'range': true_range
    }
    
    if verbose:
        print("="*60)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Regression Metrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mean_absolute_percentage_error']:.2f}%")
        
        print(f"\nCorrelation Metrics:")
        print(f"  Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
        print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
        
        print(f"\nClassification Metrics:")
        print(f"  Binary Accuracy: {metrics['binary_accuracy']:.4f}")
        print(f"  7-Class Accuracy: {metrics['7class_accuracy']:.4f}")
        
        print(f"\nDistribution Analysis:")
        print(f"  Prediction Range: {pred_range:.3f}")
        print(f"  True Range: {true_range:.3f}")
        print(f"  Range Coverage: {metrics['range_coverage']:.4f}")
        
        print(f"\nPrediction vs True Statistics:")
        print(f"  Pred Mean: {metrics['prediction_stats']['mean']:.3f} | True Mean: {metrics['true_stats']['mean']:.3f}")
        print(f"  Pred Std:  {metrics['prediction_stats']['std']:.3f} | True Std:  {metrics['true_stats']['std']:.3f}")
    
    return metrics


# Example usage function
def create_enhanced_model(base_model_fn, **model_kwargs):
    """
    Create a model with enhanced metrics for sentiment analysis.
    
    Args:
        base_model_fn: Function that creates the base model
        **model_kwargs: Arguments for the base model
    
    Returns:
        Compiled model with enhanced metrics
    """
    model = base_model_fn(**model_kwargs)
    
    # Compile with enhanced metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',
        metrics=get_core_sentiment_metrics()
    )
    
    return model






