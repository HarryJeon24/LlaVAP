# baseline_VAP_pipeline_CCPE.py
import time
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_silence
from vap.model import VapGPT, VapConfig
from vap.audio import load_waveform


def load_vap_model(state_dict_path, device="cuda"):
    """Load and initialize the VAP model"""
    conf = VapConfig()
    if not hasattr(conf, 'frame_hz'):
        conf.frame_hz = 50
    model = VapGPT(conf)
    # Use weights_only=True to avoid warning
    sd = torch.load(state_dict_path, weights_only=True)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess_waveform(audio_path, sample_rate=16000, is_stimulus=True):
    """
    Load and preprocess waveform ensuring stereo format required by VAP.
    Only trim 2 seconds if it's a stimulus file.
    """
    # Load with pydub first to ensure proper length
    audio = AudioSegment.from_wav(audio_path)

    # Convert to waveform format needed by VAP
    waveform, sr = load_waveform(audio_path, sample_rate=sample_rate)

    # For mono input, create stereo by duplicating channel
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Ensure batch dimension
    waveform = waveform.unsqueeze(0)

    # Normalize audio
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    max_val = torch.abs(waveform).max()
    if max_val > 1.0:
        waveform = waveform / max_val

    return waveform


def get_vap_predictions(model, audio_path, device="cuda", chunk_duration=1.0, threshold=0.1, flip_predictions=True):
    """
    Get VAP predictions with adjustable threshold and optional prediction flipping
    Args:
        threshold: Value between 0 and 1 to determine positive predictions
        flip_predictions: If True, flips the prediction logic
    Returns:
        Tuple of (binary predictions, raw probabilities, real-time factor)
    """
    print(f"\nGetting VAP predictions for: {audio_path}")
    waveform = load_and_preprocess_waveform(audio_path, sample_rate=model.sample_rate, is_stimulus=False)

    samples_per_frame = model.sample_rate // 50
    chunk_frames = int(chunk_duration * 50)
    chunk_samples = chunk_frames * samples_per_frame

    predictions_list = []
    probabilities_list = []
    total_frames = 0
    total_inference_time = 0

    def process_chunk(chunk):
        if chunk.shape[-1] < chunk_samples:
            pad_size = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        elif chunk.shape[-1] > chunk_samples:
            chunk = chunk[..., :chunk_samples]
        if device == "cuda":
            chunk = chunk.cuda()
        if chunk.dtype != torch.float32:
            chunk = chunk.float()
        return chunk

    with torch.no_grad():
        if device == "cuda":
            waveform = waveform.cuda()

        print(f"Processing audio in chunks...")
        for i in range(0, waveform.shape[-1], chunk_samples):
            try:
                chunk_end = min(i + chunk_samples, waveform.shape[-1])
                chunk = waveform[..., i:chunk_end].contiguous()
                chunk = process_chunk(chunk)

                torch.cuda.synchronize()
                start_time = time.time()
                out = model(chunk)
                torch.cuda.synchronize()
                inference_time = time.time() - start_time

                num_frames = out['vad'].shape[1]
                total_frames += num_frames
                total_inference_time += inference_time

                # Get probabilities and predictions
                probabilities = out['vad'].sigmoid().squeeze().cpu().numpy()[:, 0]
                if flip_predictions:
                    pred_chunk = probabilities <= threshold
                else:
                    pred_chunk = probabilities > threshold

                if chunk_end - i < chunk_samples:
                    actual_frames = int((chunk_end - i) / samples_per_frame)
                    pred_chunk = pred_chunk[:actual_frames]
                    probabilities = probabilities[:actual_frames]

                predictions_list.append(pred_chunk)
                probabilities_list.append(probabilities)

            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue

    if predictions_list:
        predictions = np.concatenate(predictions_list)
        probabilities = np.concatenate(probabilities_list)
        avg_time_per_frame = total_inference_time * 1000 / total_frames
        rtf = avg_time_per_frame / 20
        print(f"Processed {len(predictions)} frames ({len(predictions)/50:.2f} seconds)")
        print(f"Real-time factor: {rtf:.3f}")
        return predictions, probabilities, rtf
    else:
        raise RuntimeError("Failed to process any chunks")


class TurnShiftGroundTruth:
    def __init__(self, silence_threshold_db=-40, min_silence_len=1800, frame_rate=50):
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_len = min_silence_len
        self.frame_rate = frame_rate

    def detect_silences(self, audio_segment):
        """Detect silence periods in audio that indicate turn shifts."""
        silence_ranges = detect_silence(
            audio_segment,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_threshold_db
        )
        return [(start / 1000, end / 1000) for start, end in silence_ranges]

    def create_frame_level_ground_truth(self, turn_shifts, total_frames):
        """Convert turn shift points to frame-level ground truth."""
        ground_truth = np.zeros(total_frames)
        for shift in turn_shifts:
            frame_idx = int(shift * self.frame_rate)
            if 0 <= frame_idx < total_frames:
                ground_truth[frame_idx] = 1
        return ground_truth

    def process_audio_file(self, audio_path):
        """Process a single audio file to find turn shifts."""
        try:
            print(f"Processing: {audio_path}")
            audio = AudioSegment.from_wav(audio_path)
            silences = self.detect_silences(audio)

            turn_shifts = [
                (start + (end - start) / 2)
                for start, end in silences
            ]

            total_frames = int(len(audio) / 1000 * self.frame_rate)
            ground_truth = self.create_frame_level_ground_truth(turn_shifts, total_frames)

            result = {
                'file_path': str(audio_path),
                'duration': len(audio) / 1000,
                'turn_shifts': turn_shifts,
                'ground_truth': ground_truth,
                'num_turns': len(turn_shifts)
            }

            print(f"Found {len(turn_shifts)} turn shifts")
            return result

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_directory(self, directory_path):
        """Process all WAV files in a directory."""
        directory = Path(directory_path)
        results = {}
        print(f"\nProcessing directory: {directory}")

        for audio_file in directory.glob('*.wav'):
            result = self.process_audio_file(audio_file)
            if result:
                results[audio_file.name] = result
                print(f"Successfully processed {audio_file.name}\n")

        print(f"Processed {len(results)} files successfully")
        return results


def evaluate_predictions(ground_truth, predictions, window_size=75):
    """Evaluate predictions using balanced metrics with window."""
    metrics = {}

    # Create windows around ground truth positives
    true_windows = np.zeros_like(ground_truth)
    for i in np.where(ground_truth == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(ground_truth), i + window_size + 1)
        true_windows[start:end] = 1

    # Calculate basic counts
    tp = np.sum((predictions == 1) & (true_windows == 1))
    tn = np.sum((predictions == 0) & (true_windows == 0))
    fp = np.sum((predictions == 1) & (true_windows == 0))
    fn = np.sum((predictions == 0) & (true_windows == 1))

    # Store confusion matrix values
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)

    # Calculate metrics
    metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics['balanced_accuracy'] = float((sensitivity + specificity) / 2)
    metrics['sensitivity'] = float(sensitivity)
    metrics['specificity'] = float(specificity)

    metrics['precision'] = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
    metrics['recall'] = float(sensitivity)

    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = float(2 * (metrics['precision'] * metrics['recall']) /
                              (metrics['precision'] + metrics['recall']))
    else:
        metrics['f1'] = 0.0

    total_frames = len(ground_truth)
    positive_frames = np.sum(true_windows == 1)

    metrics['trp_density'] = float(np.sum(ground_truth == 1) / total_frames)
    metrics['prediction_density'] = float(np.sum(predictions == 1) / total_frames)
    metrics['window_coverage'] = float(positive_frames / total_frames)
    metrics['window_size_used'] = int(window_size)

    return metrics


class TurnShiftEvaluator:
    def __init__(self, vap_model_path, device="cuda"):
        self.device = device
        self.model = load_vap_model(vap_model_path, device)

    def evaluate_file(self, audio_path, ground_truth_data, flip_predictions=True):
        """Evaluate VAP predictions against ground truth."""
        # Get VAP predictions and probabilities
        predictions, probabilities, rtf = get_vap_predictions(
            self.model, audio_path, self.device, flip_predictions=flip_predictions
        )

        # Ensure predictions and ground truth have same length
        min_len = min(len(predictions), len(ground_truth_data['ground_truth']))
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        ground_truth = ground_truth_data['ground_truth'][:min_len]

        # Calculate metrics
        metrics = evaluate_predictions(ground_truth, predictions)
        metrics['real_time_factor'] = float(rtf)

        return metrics, predictions, probabilities

    def evaluate_directory(self, audio_dir, ground_truth_data, output_dir, flip_predictions=True):
        """Evaluate all audio files and save results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        all_results = {}

        for audio_file, gt_data in ground_truth_data.items():
            audio_path = Path(audio_dir) / audio_file
            if audio_path.exists():
                print(f"\nProcessing: {audio_file}")
                metrics, predictions, probabilities = self.evaluate_file(
                    str(audio_path),
                    gt_data,
                    flip_predictions=flip_predictions
                )

                all_results[audio_file] = {
                    'metrics': metrics,
                    'ground_truth': gt_data['ground_truth'].tolist(),
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist()  # Save raw probabilities
                }

                self.plot_comparison(
                    audio_file,
                    gt_data['ground_truth'],
                    predictions,
                    output_dir / f"{audio_file}_analysis.png"
                )

                # Save probabilities to a separate CSV file
                prob_df = pd.DataFrame({
                    'time': np.arange(len(probabilities)) / 50,  # Convert frames to seconds
                    'probability': probabilities,
                    'prediction': predictions,
                    'ground_truth': gt_data['ground_truth'][:len(predictions)]
                })
                prob_df.to_csv(output_dir / f"{audio_file}_probabilities.csv", index=False)

        # Calculate and save summary statistics
        if all_results:
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
                    'num_files': len(all_results),
                    'total_frames': int(total_predictions),
                    'overall_accuracy': float(overall_accuracy),
                    'confusion_matrix': {
                        'true_positives': int(total_tp),
                        'false_positives': int(total_fp),
                        'true_negatives': int(total_tn),
                        'false_negatives': int(total_fn)
                    },
                    'flip_predictions': flip_predictions
                },
                'mean_metrics': {k: float(v) for k, v in mean_metrics.to_dict().items()},
                'std_metrics': {k: float(v) for k, v in std_metrics.to_dict().items()},
                'per_file_results': all_results
            }

            # Save results
            with open(output_dir / "evaluation_results.json", "w") as f:
                json.dump(final_results, f, indent=4)

            metrics_df.to_csv(output_dir / "evaluation_metrics.csv")

            return final_results

        return None

    def plot_comparison(self, audio_name, ground_truth, predictions, save_path, window_size=75):
        """Create visualization matching the example format."""
        plt.figure(figsize=(15, 12))

        # Ensure all arrays have the same length
        min_len = min(len(ground_truth), len(predictions))
        ground_truth = ground_truth[:min_len]
        predictions = predictions[:min_len]
        time_axis = np.arange(min_len) / 50  # 50Hz to seconds

        # Plot 1: Ground Truth vs Predictions
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, ground_truth.astype(int), 'g-', label='Ground Truth')
        plt.plot(time_axis, predictions.astype(int), 'r--', alpha=0.5, label='Predictions')
        plt.title(f'Ground Truth vs Predictions - {audio_name}')
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

        # Plot 3: Local Window-based Accuracy
        plt.subplot(3, 1, 3)
        local_acc = np.zeros_like(ground_truth, dtype=float)
        for i in range(len(ground_truth)):
            window_start = max(0, i - window_size)
            window_end = min(len(ground_truth), i + window_size + 1)
            gt_window = ground_truth[window_start:window_end]
            pred_window = predictions[window_start:window_end]

            tp = (np.any(gt_window == 1) and np.any(pred_window == 1))
            tn = (not np.any(gt_window == 1) and not np.any(pred_window == 1))

            if np.any(gt_window == 1):
                local_acc[i] = 1.0 if tp else 0.0
            else:
                local_acc[i] = 1.0 if tn else 0.0

        plt.plot(time_axis, local_acc, 'b-', label='Window-based Accuracy')
        plt.title('Local Window-based Accuracy')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Example usage with absolute paths
    audio_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/CCPE/generated_audio")
    vap_model_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/output")
    flip_predictions = True  # Set this to True if needed

    # Step 1: Extract ground truth
    print("Extracting ground truth...")
    extractor = TurnShiftGroundTruth()
    ground_truth = extractor.process_directory(audio_dir)

    # Step 2: Evaluate with VAP
    print("\nEvaluating with VAP...")
    evaluator = TurnShiftEvaluator(vap_model_path)
    results = evaluator.evaluate_directory(audio_dir, ground_truth, output_dir, flip_predictions=flip_predictions)

    if results:
        print("\nEvaluation Summary:")
        mean_metrics = results['mean_metrics']
        std_metrics = results['std_metrics']

        print(f"Prediction Mode: {'Flipped' if flip_predictions else 'Normal'}")
        print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.3f}")
        print(f"Mean Accuracy: {mean_metrics['accuracy']:.3f} (±{std_metrics['accuracy']:.3f})")
        print(
            f"Mean Balanced Accuracy: {mean_metrics['balanced_accuracy']:.3f} (±{std_metrics['balanced_accuracy']:.3f})")
        print(f"Mean Sensitivity/Recall: {mean_metrics['sensitivity']:.3f} (±{std_metrics['sensitivity']:.3f})")
        print(f"Mean Specificity: {mean_metrics['specificity']:.3f} (±{std_metrics['specificity']:.3f})")
        print(f"Mean Precision: {mean_metrics['precision']:.3f} (±{std_metrics['precision']:.3f})")
        print(f"Mean F1 Score: {mean_metrics['f1']:.3f} (±{std_metrics['f1']:.3f})")
        print(f"Mean Real-time Factor: {mean_metrics['real_time_factor']:.3f} (±{std_metrics['real_time_factor']:.3f})")


if __name__ == "__main__":
    try:
        main()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"Error in main execution: {e}")