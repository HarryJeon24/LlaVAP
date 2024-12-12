import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import joblib
from typing import Dict, Tuple, List
from collections import defaultdict


class EnsembleModel:
    def __init__(self, window_size: int = 75):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = LogisticRegression(class_weight='balanced', max_iter=1000)

    def prepare_features(self, vap_probs: np.ndarray, llama_probs: np.ndarray) -> np.ndarray:
        """Prepare features with additional engineered features."""
        # Basic probabilities
        features = np.column_stack([vap_probs, llama_probs])

        # Add rolling statistics
        window_sizes = [5, 10, 20]  # Multiple window sizes for different temporal scales
        for w in window_sizes:
            for probs in [vap_probs, llama_probs]:
                # Rolling mean
                rolling_mean = pd.Series(probs).rolling(window=w, center=True).mean()
                # Rolling std
                rolling_std = pd.Series(probs).rolling(window=w, center=True).std()
                # Rolling max
                rolling_max = pd.Series(probs).rolling(window=w, center=True).max()
                # Rolling min
                rolling_min = pd.Series(probs).rolling(window=w, center=True).min()

                new_features = np.column_stack([
                    rolling_mean, rolling_std, rolling_max, rolling_min
                ])
                features = np.column_stack([features, new_features])

        # Add interaction terms between VAP and LLaMA
        features = np.column_stack([
            features,
            vap_probs * llama_probs,  # Multiplication interaction
            np.maximum(vap_probs, llama_probs),  # Max voting
            np.minimum(vap_probs, llama_probs),  # Min voting
            (vap_probs + llama_probs) / 2  # Average voting
        ])

        # Fill NaN values from rolling operations
        features = np.nan_to_num(features, nan=0)

        return features

    def evaluate_predictions(self, ground_truth: np.ndarray, predictions: np.ndarray) -> dict:
        """Evaluate predictions using windowed approach."""
        true_windows = np.zeros_like(ground_truth)
        for i in np.where(ground_truth == 1)[0]:
            start = max(0, i - self.window_size)
            end = min(len(ground_truth), i + self.window_size + 1)
            true_windows[start:end] = 1

        tp = np.sum((predictions == 1) & (true_windows == 1))
        tn = np.sum((predictions == 0) & (true_windows == 0))
        fp = np.sum((predictions == 1) & (true_windows == 0))
        fn = np.sum((predictions == 0) & (true_windows == 1))

        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        metrics['f1'] = (2 * metrics['precision'] * metrics['sensitivity'] /
                         (metrics['precision'] + metrics['sensitivity'])
                         if (metrics['precision'] + metrics['sensitivity']) > 0 else 0)

        return metrics


def analyze_thresholds_across_files(ensemble_model, X_all, y_all, scaler, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Analyze different thresholds across all files to find optimal configuration.
    """
    metrics_by_threshold = {
        'normal': [],
        'flipped': []
    }

    # Scale features and get probabilities
    X_scaled = scaler.transform(X_all)
    probabilities = ensemble_model.predict_proba(X_scaled)[:, 1]

    for threshold in thresholds:
        # Evaluate normal predictions
        normal_preds = (probabilities > threshold).astype(int)
        normal_metrics = evaluate_predictions(y_all, normal_preds)
        normal_metrics['threshold'] = threshold
        metrics_by_threshold['normal'].append(normal_metrics)

        # Evaluate flipped predictions
        flipped_preds = (probabilities <= threshold).astype(int)
        flipped_metrics = evaluate_predictions(y_all, flipped_preds)
        flipped_metrics['threshold'] = threshold
        metrics_by_threshold['flipped'].append(flipped_metrics)

    # Convert to DataFrames for easier analysis
    normal_df = pd.DataFrame(metrics_by_threshold['normal'])
    flipped_df = pd.DataFrame(metrics_by_threshold['flipped'])

    # Find best configurations
    optimal_results = {
        'normal': {
            'accuracy': {
                'threshold': float(normal_df.loc[normal_df['accuracy'].idxmax(), 'threshold']),
                'value': float(normal_df['accuracy'].max()),
                'metrics': normal_df.iloc[normal_df['accuracy'].idxmax()].to_dict()
            },
            'balanced_accuracy': {
                'threshold': float(normal_df.loc[normal_df['balanced_accuracy'].idxmax(), 'threshold']),
                'value': float(normal_df['balanced_accuracy'].max()),
                'metrics': normal_df.iloc[normal_df['balanced_accuracy'].idxmax()].to_dict()
            }
        },
        'flipped': {
            'accuracy': {
                'threshold': float(flipped_df.loc[flipped_df['accuracy'].idxmax(), 'threshold']),
                'value': float(flipped_df['accuracy'].max()),
                'metrics': flipped_df.iloc[flipped_df['accuracy'].idxmax()].to_dict()
            },
            'balanced_accuracy': {
                'threshold': float(flipped_df.loc[flipped_df['balanced_accuracy'].idxmax(), 'threshold']),
                'value': float(flipped_df['balanced_accuracy'].max()),
                'metrics': flipped_df.iloc[flipped_df['balanced_accuracy'].idxmax()].to_dict()
            }
        }
    }

    return optimal_results, metrics_by_threshold


def evaluate_predictions(ground_truth: np.ndarray, predictions: np.ndarray, window_size: int = 75) -> dict:
    """Evaluate predictions using windowed approach."""
    true_windows = np.zeros_like(ground_truth)
    for i in np.where(ground_truth == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(ground_truth), i + window_size + 1)
        true_windows[start:end] = 1

    tp = np.sum((predictions == 1) & (true_windows == 1))
    tn = np.sum((predictions == 0) & (true_windows == 0))
    fp = np.sum((predictions == 1) & (true_windows == 0))
    fn = np.sum((predictions == 0) & (true_windows == 1))

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
    metrics['f1'] = (2 * metrics['precision'] * metrics['sensitivity'] /
                     (metrics['precision'] + metrics['sensitivity'])
                     if (metrics['precision'] + metrics['sensitivity']) > 0 else 0)

    return metrics


def plot_threshold_analysis(metrics_by_threshold: Dict, save_path: Path):
    """Plot accuracy and balanced accuracy curves for different thresholds."""
    normal_df = pd.DataFrame(metrics_by_threshold['normal'])
    flipped_df = pd.DataFrame(metrics_by_threshold['flipped'])

    plt.figure(figsize=(12, 6))

    # Plot normal predictions
    plt.plot(normal_df['threshold'], normal_df['accuracy'],
             'b-', label='Normal - Accuracy')
    plt.plot(normal_df['threshold'], normal_df['balanced_accuracy'],
             'b--', label='Normal - Balanced Accuracy')

    # Plot flipped predictions
    plt.plot(flipped_df['threshold'], flipped_df['accuracy'],
             'r-', label='Flipped - Accuracy')
    plt.plot(flipped_df['threshold'], flipped_df['balanced_accuracy'],
             'r--', label='Flipped - Balanced Accuracy')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Ensemble Model Threshold Analysis')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_analysis(ground_truth: np.ndarray, predictions: np.ndarray,
                             probabilities: np.ndarray, save_path: Path, window_size: int = 75):
    """Plot analysis of ensemble results."""
    time_axis = np.arange(len(ground_truth)) / 50  # 50Hz to seconds

    plt.figure(figsize=(15, 12))

    # Plot 1: Ground Truth vs Predictions
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, ground_truth, 'g-', label='Ground Truth')
    plt.plot(time_axis, predictions, 'r--', alpha=0.5, label='Ensemble Predictions')
    plt.title('Ground Truth vs Ensemble Predictions')
    plt.ylabel('Turn Shift Present')
    plt.legend()

    # Plot 2: Evaluation Windows
    plt.subplot(3, 1, 2)
    windows = np.zeros_like(ground_truth, dtype=float)
    for i in np.where(ground_truth == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(ground_truth), i + window_size + 1)
        windows[start:end] = 0.5
    plt.fill_between(time_axis, 0, windows, color='g', alpha=0.2, label='Windows')
    plt.plot(time_axis, predictions, 'r-', alpha=0.7, label='Predictions')
    plt.title('Evaluation Windows')
    plt.ylabel('Window/Prediction')
    plt.legend()

    # Plot 3: Ensemble Probabilities
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, probabilities, 'b-', label='Ensemble Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Default Threshold')
    plt.title('Ensemble Probabilities')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_and_validate_data(vap_path: Path, llama_path: Path) -> Tuple[Dict, Dict]:
    """
    Load and validate data from both models ensuring matching files and lengths.
    Returns validated data dictionaries.
    """
    print("\nLoading and validating data...")

    # Load all files
    vap_files = list(vap_path.glob('*_probabilities.csv'))
    llama_files = list(llama_path.glob('*_probabilities.csv'))

    # Create dictionaries with clean filenames
    vap_data = {}
    llama_data = {}

    for vf in vap_files:
        base_name = vf.stem.replace('_probabilities', '')
        vap_data[base_name] = pd.read_csv(vf)

    for lf in llama_files:
        base_name = lf.stem.replace('_probabilities', '')
        llama_data[base_name] = pd.read_csv(lf)

    # Find common files
    common_files = set(vap_data.keys()) & set(llama_data.keys())

    if not common_files:
        raise ValueError("No matching files found between VAP and LLaMA results!")

    print(f"Found {len(common_files)} matching files")

    # Validate data lengths and features
    validated_vap = {}
    validated_llama = {}
    mismatched_files = []

    for filename in common_files:
        vap_df = vap_data[filename]
        llama_df = llama_data[filename]

        # Check if lengths match
        if len(vap_df) != len(llama_df):
            print(f"Warning: Length mismatch in {filename}")
            print(f"VAP length: {len(vap_df)}, LLaMA length: {len(llama_df)}")
            mismatched_files.append(filename)
            continue

        # Verify required columns exist
        required_cols = ['probability', 'ground_truth']
        if not all(col in vap_df.columns for col in required_cols) or \
                not all(col in llama_df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {filename}")
            continue

        # If all validations pass, add to validated data
        validated_vap[filename] = vap_df
        validated_llama[filename] = llama_df

    if mismatched_files:
        print("\nFiles with length mismatches (skipped):")
        for fname in mismatched_files:
            print(f"- {fname}")

    print(f"\nSuccessfully validated {len(validated_vap)} files")

    if not validated_vap:
        raise ValueError("No valid files remaining after validation!")

    return validated_vap, validated_llama


def main():
    try:
        # Set up paths
        vap_results_path = Path(
            "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_CCPE_output/probs")
        llama_results_path = Path(
            "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_CCPE_output/3b/prob")
        output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lr_ensemble_results_CCPE")
        output_dir.mkdir(exist_ok=True)

        # Load and validate data
        vap_data, llama_data = load_and_validate_data(vap_results_path, llama_results_path)
        file_names = list(vap_data.keys())
        print(f"Total number of files to process: {len(file_names)}")

        # Initialize ensemble model
        ensemble = EnsembleModel()

        # Prepare cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Store results for each fold
        cv_results = []
        all_fold_threshold_results = []

        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(file_names)):
            print(f"\nProcessing fold {fold + 1}/5...")

            # Prepare training data
            X_train = []
            y_train = []

            for idx in train_idx:
                file_name = file_names[idx]
                print(f"Processing training file: {file_name}")

                vap_probs = vap_data[file_name]['probability'].values
                llama_probs = llama_data[file_name]['probability'].values
                ground_truth = vap_data[file_name]['ground_truth'].values

                features = ensemble.prepare_features(vap_probs, llama_probs)
                X_train.append(features)
                y_train.append(ground_truth)

            X_train = np.vstack(X_train)
            y_train = np.concatenate(y_train)

            print(f"Training data shape: {X_train.shape}")

            # Scale features
            X_train_scaled = ensemble.scaler.fit_transform(X_train)

            # Train model
            print("Training model...")
            ensemble.model.fit(X_train_scaled, y_train)

            # Analyze thresholds on training data
            print("\nAnalyzing thresholds across training data...")
            optimal_thresholds, threshold_metrics = analyze_thresholds_across_files(
                ensemble.model, X_train, y_train, ensemble.scaler
            )

            # Store threshold results
            all_fold_threshold_results.append({
                'fold': fold + 1,
                'optimal_thresholds': optimal_thresholds,
                'threshold_metrics': threshold_metrics
            })

            # Plot threshold analysis
            plot_threshold_analysis(
                threshold_metrics,
                output_dir / f"fold{fold + 1}_threshold_analysis.png"
            )

            # Print optimal threshold results
            print("\nOptimal Threshold Configurations:")
            for pred_type in ['normal', 'flipped']:
                print(f"\n{pred_type.title()} Predictions:")
                for metric in ['accuracy', 'balanced_accuracy']:
                    result = optimal_thresholds[pred_type][metric]
                    print(f"Best {metric}: {result['value']:.3f} "
                          f"(threshold: {result['threshold']:.3f})")

            # Find best configuration based on balanced accuracy
            best_config = max(
                [('normal', optimal_thresholds['normal']['balanced_accuracy']),
                 ('flipped', optimal_thresholds['flipped']['balanced_accuracy'])],
                key=lambda x: x[1]['value']
            )
            pred_type, threshold_info = best_config
            optimal_threshold = threshold_info['threshold']

            # Evaluate on test set using optimal threshold
            fold_results = {}
            for idx in test_idx:
                file_name = file_names[idx]
                print(f"Processing test file: {file_name}")

                vap_probs = vap_data[file_name]['probability'].values
                llama_probs = llama_data[file_name]['probability'].values
                ground_truth = vap_data[file_name]['ground_truth'].values

                # Prepare and scale features
                features = ensemble.prepare_features(vap_probs, llama_probs)
                features_scaled = ensemble.scaler.transform(features)

                # Get predictions
                probabilities = ensemble.model.predict_proba(features_scaled)[:, 1]

                # Apply optimal threshold
                predictions = (probabilities <= optimal_threshold) if pred_type == 'flipped' \
                    else (probabilities > optimal_threshold)

                # Calculate metrics
                metrics = evaluate_predictions(ground_truth, predictions)

                # Save results
                fold_results[file_name] = {
                    'metrics': metrics,
                    'optimal_threshold': optimal_threshold,
                    'prediction_type': pred_type,
                    'probabilities': probabilities.tolist(),
                    'predictions': predictions.tolist()
                }

                # Create visualization
                plot_prediction_analysis(
                    ground_truth,
                    predictions,
                    probabilities,
                    output_dir / f"fold{fold + 1}_{file_name}_analysis.png"
                )

                # Print current file metrics
                print(f"\nMetrics for {file_name}:")
                print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
                print(f"Sensitivity: {metrics['sensitivity']:.3f}")
                print(f"Specificity: {metrics['specificity']:.3f}")
                print(f"F1 Score: {metrics['f1']:.3f}")

            cv_results.append(fold_results)

        # Save final results
        final_results = {
            'cross_validation_results': cv_results,
            'threshold_analysis': [
                {
                    'fold': res['fold'],
                    'optimal_thresholds': res['optimal_thresholds']
                }
                for res in all_fold_threshold_results
            ],
            'model_params': {
                'window_size': ensemble.window_size,
                'logistic_regression_params': ensemble.model.get_params()
            }
        }

        # Save results to JSON
        with open(output_dir / "ensemble_results.json", "w") as f:
            json.dump(final_results, f, indent=4)

        # Save the final model
        joblib.dump({
            'scaler': ensemble.scaler,
            'model': ensemble.model
        }, output_dir / "ensemble_model.joblib")

        # Print summary results
        print("\nEnsemble Model Results Summary:")
        metrics_summary = defaultdict(list)
        for fold_results in cv_results:
            for file_results in fold_results.values():
                for metric, value in file_results['metrics'].items():
                    metrics_summary[metric].append(value)

        print("\nOverall Performance Metrics:")
        for metric in ['balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1']:
            values = metrics_summary[metric]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric.title()}: {mean_value:.3f} (±{std_value:.3f})")

        # Calculate overall confusion matrix
        total_tp = sum(metrics_summary['true_positives'])
        total_tn = sum(metrics_summary['true_negatives'])
        total_fp = sum(metrics_summary['false_positives'])
        total_fn = sum(metrics_summary['false_negatives'])

        print("\nOverall Confusion Matrix:")
        print(f"True Positives: {total_tp}")
        print(f"True Negatives: {total_tn}")
        print(f"False Positives: {total_fp}")
        print(f"False Negatives: {total_fn}")

    except Exception as e:
        print(f"Error in execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
    print("\nProcessing completed successfully!")