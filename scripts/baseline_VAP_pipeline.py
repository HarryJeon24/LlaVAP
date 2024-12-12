#baseline_VAP_pipeline.py
import time
import os
import torch
import torchaudio
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from traceback import print_exc
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import json

from ICC_Ground_Truth_Processing import TRPGroundTruth
from VAP import load_vap_model, get_vap_predictions, process_dataset

def evaluate_vap_predictions(ground_truth, predictions, window_size=75):
    """
    Evaluate predictions using balanced metrics with window.
    Window size should be based on average response duration.
    """
    metrics = {}

    # Debug prints
    print(f"\nGround truth stats before window creation:")
    print(f"Ground truth ones: {np.sum(ground_truth == 1)}")
    print(f"Ground truth zeros: {np.sum(ground_truth == 0)}")
    print(f"Window size being used: {window_size} frames ({window_size/50:.3f} seconds)")

    # Create windows around ground truth positives (where enough participants responded)
    true_windows = np.zeros_like(ground_truth)
    for i in np.where(ground_truth == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(ground_truth), i + window_size + 1)
        true_windows[start:end] = 1

    # Print window coverage statistics
    total_frames = len(ground_truth)
    positive_frames = np.sum(true_windows == 1)
    print(f"\nWindow Coverage Analysis:")
    print(f"Total frames: {total_frames} ({total_frames/50:.2f} seconds)")
    print(f"Frames in positive windows: {positive_frames} ({positive_frames/50:.2f} seconds)")
    print(f"Percentage of frames in windows: {100*positive_frames/total_frames:.1f}%")
    print(f"Number of TRPs: {np.sum(ground_truth == 1)}")
    if np.sum(ground_truth == 1) > 1:
        print(f"Average gap between TRPs: {50/np.mean(np.diff(np.where(ground_truth == 1)[0])):.3f} per second")

    # Calculate metrics
    tp = np.sum((predictions == 1) & (true_windows == 1))
    tn = np.sum((predictions == 0) & (true_windows == 0))
    fp = np.sum((predictions == 1) & (true_windows == 0))
    fn = np.sum((predictions == 0) & (true_windows == 1))

    print("\nPrediction distribution in windows:")
    print(f"Predictions=1 in windows=1 (TP): {tp}")
    print(f"Predictions=1 in windows=0 (FP): {fp}")
    print(f"Predictions=0 in windows=1 (FN): {fn}")
    print(f"Predictions=0 in windows=0 (TN): {tn}")

    # Store confusion matrix values
    metrics['true_positives'] = tp
    metrics['false_positives'] = fp
    metrics['true_negatives'] = tn
    metrics['false_negatives'] = fn

    # Calculate regular accuracy
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics['balanced_accuracy'] = (sensitivity + specificity) / 2
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity

    # Calculate precision, recall, and F1
    total_pred_pos = tp + fp
    total_true_pos = tp + fn

    metrics['precision'] = tp / total_pred_pos if total_pred_pos > 0 else 0
    metrics['recall'] = sensitivity  # Same as sensitivity

    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0

    # Calculate additional metrics for analysis
    metrics['trp_density'] = np.sum(ground_truth == 1) / len(ground_truth)  # TRPs per frame
    metrics['prediction_density'] = np.sum(predictions == 1) / len(predictions)  # Predictions per frame
    metrics['window_coverage'] = positive_frames / total_frames  # Proportion of frames in windows
    metrics['window_size_used'] = window_size  # Store window size used for reference

    return metrics

def plot_comparison(stimulus_name, ground_truth, predictions, response_proportions, save_path=None, window_size=75):
    plt.figure(figsize=(15, 12))

    time_axis = np.arange(len(ground_truth)) / 50  # 50Hz to seconds

    # Plot 1: Response proportions
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, response_proportions, 'b-', alpha=0.5, label='Response Proportions')
    plt.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold (0.3)')
    plt.title(f'Response Proportions - {stimulus_name}')
    plt.ylabel('Proportion')
    plt.legend()

    # Plot 2: Ground Truth vs Predictions
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, ground_truth.astype(int), 'g-', label='Ground Truth')
    plt.plot(time_axis, predictions.astype(int), 'r--', alpha=0.5, label='Predictions')
    plt.title('Ground Truth vs Predictions')
    plt.ylabel('TRP Present')
    plt.legend()

    # Plot 3: 3-Second Windows
    plt.subplot(4, 1, 3)
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

    # Plot 4: Local Window-based Accuracy
    plt.subplot(4, 1, 4)
    local_acc = np.zeros_like(ground_truth, dtype=float)
    for i in range(len(ground_truth)):
        window_start = max(0, i - window_size)
        window_end = min(len(ground_truth), i + window_size + 1)

        # Get ground truth and predictions for this window
        gt_window = ground_truth[window_start:window_end]
        pred_window = predictions[window_start:window_end]

        # True positive: Both have 1s in the window
        tp = (np.any(gt_window == 1) and np.any(pred_window == 1))
        # True negative: Both have all 0s in the window
        tn = (not np.any(gt_window == 1) and not np.any(pred_window == 1))

        if np.any(gt_window == 1):
            # For positive ground truth windows
            local_acc[i] = 1.0 if tp else 0.0
        else:
            # For negative ground truth windows
            local_acc[i] = 1.0 if tn else 0.0

    plt.plot(time_axis, local_acc, 'b-', label='Window-based Accuracy')
    plt.title('Local Window-based Accuracy')
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    state_dict_path = "C:/Users/Harry/PycharmProjects/LLA-VAP/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        model = load_vap_model(state_dict_path, device)
        print("VAP model loaded successfully")
    except Exception as e:
        print(f"Error loading VAP model: {e}")
        return

    stimulus_groups = process_dataset(base_path)
    print(f"Found {len(stimulus_groups)} unique stimuli")

    trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50, window_size_sec=1.5)

    vap_predictions_cache = {}
    all_results = {}
    ground_truth_stats = {}

    for stim_key, group in stimulus_groups.items():
        try:
            print(f"\nProcessing stimulus: {stim_key}")

            ground_truth, response_proportions, duration_stats = trp_processor.load_responses(
                str(group['stimulus_path']),
                [str(path) for path in group['response_paths']]
            )

            ground_truth_stats[stim_key] = {
                'num_responses': int(duration_stats['valid_responses']),
                'avg_response_duration': float(duration_stats['avg_duration']),
                'std_response_duration': float(duration_stats['std_duration']),
                'min_duration': float(duration_stats['min_duration']),
                'max_duration': float(duration_stats['max_duration']),
                'total_responses_found': int(duration_stats['num_responses'])
            }

            stimulus_path = str(group['stimulus_path'])
            if stimulus_path not in vap_predictions_cache:
                predictions, probabilities, rtf = get_vap_predictions(model, stimulus_path, device)
                vap_predictions_cache[stimulus_path] = (predictions, probabilities, rtf)
            else:
                predictions, probabilities, rtf = vap_predictions_cache[stimulus_path]
                print(f"Using cached VAP predictions")

            total_frames = len(predictions)
            vap_positives = np.sum(predictions == 1)
            vap_negatives = np.sum(predictions == 0)

            print("\nVAP Prediction Analysis:")
            print(f"Total frames: {total_frames} ({total_frames / 50:.2f} seconds)")
            print(f"VAP positive predictions: {vap_positives} ({vap_positives / total_frames * 100:.1f}%)")
            print(f"VAP negative predictions: {vap_negatives} ({vap_negatives / total_frames * 100:.1f}%)")

            min_len = min(len(ground_truth), len(predictions))
            ground_truth = ground_truth[:min_len]
            predictions = predictions[:min_len]
            probabilities = probabilities[:min_len]
            response_proportions = response_proportions[:min_len]

            window_size = max(25, int(duration_stats['avg_duration']))
            metrics = evaluate_vap_predictions(ground_truth, predictions, window_size=window_size)
            metrics['real_time_factor'] = float(rtf)
            metrics['window_size_used'] = int(window_size)

            all_results[stim_key] = {
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in metrics.items()},
                'ground_truth': ground_truth.tolist(),
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'response_proportions': response_proportions.tolist(),
                'duration_stats': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in duration_stats.items()}
            }

            plot_comparison(
                stim_key,
                ground_truth,
                predictions,
                response_proportions,
                save_path=output_dir / f"{stim_key}_analysis.png",
                window_size=window_size
            )

        except Exception as e:
            print(f"Error processing {stim_key}: {str(e)}")
            continue

    if all_results:
        with open(output_dir / "ground_truth_stats.json", 'w') as f:
            json.dump(ground_truth_stats, f, indent=4)

        metrics_df = pd.DataFrame([r['metrics'] for r in all_results.values()])

        total_tp = metrics_df['true_positives'].sum()
        total_fp = metrics_df['false_positives'].sum()
        total_tn = metrics_df['true_negatives'].sum()
        total_fn = metrics_df['false_negatives'].sum()

        total_predictions = (total_tp + total_tn + total_fp + total_fn)
        overall_accuracy = (total_tp + total_tn) / total_predictions if total_predictions > 0 else 0

        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        final_results = {
            'summary': {
                'num_stimuli': len(all_results),
                'total_frames': int(total_predictions),
                'overall_accuracy': float(overall_accuracy),
                'confusion_matrix': {
                    'true_positives': int(total_tp),
                    'false_positives': int(total_fp),
                    'true_negatives': int(total_tn),
                    'false_negatives': int(total_fn)
                }
            },
            'mean_metrics': {k: float(v) for k, v in mean_metrics.to_dict().items()},
            'std_metrics': {k: float(v) for k, v in std_metrics.to_dict().items()},
            'per_stimulus_results': all_results
        }

        with open(output_dir / "full_results.json", 'w') as f:
            json.dump(final_results, f, indent=4)

        metrics_df.to_csv(output_dir / "evaluation_metrics.csv")

        print(f"\nResults saved to {output_dir}")
        print(f"Number of stimuli processed: {len(metrics_df)}")
        print("\nMean Metrics (± Standard Deviation):")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Mean Accuracy: {mean_metrics['accuracy']:.3f} (±{std_metrics['accuracy']:.3f})")
        print(
            f"Mean Balanced Accuracy: {mean_metrics['balanced_accuracy']:.3f} (±{std_metrics['balanced_accuracy']:.3f})")
        print(f"Mean Sensitivity/Recall: {mean_metrics['sensitivity']:.3f} (±{std_metrics['sensitivity']:.3f})")
        print(f"Mean Specificity: {mean_metrics['specificity']:.3f} (±{std_metrics['specificity']:.3f})")
        print(f"Mean Precision: {mean_metrics['precision']:.3f} (±{std_metrics['precision']:.3f})")
        print(f"Mean F1 Score: {mean_metrics['f1']:.3f} (±{std_metrics['f1']:.3f})")
        print(f"Mean Real-time Factor: {mean_metrics['real_time_factor']:.3f} (±{std_metrics['real_time_factor']:.3f})")
        print(f"Average Window Size: {mean_metrics['window_size_used']:.1f} frames")

if __name__ == "__main__":
    try:
        main()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print_exc()