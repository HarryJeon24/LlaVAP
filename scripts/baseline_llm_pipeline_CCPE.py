# baseline_llm_pipeline_CCPE.py
import time
import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydub import AudioSegment
from faster_whisper import WhisperModel
from baseline_VAP_pipeline_CCPE import TurnShiftGroundTruth, TurnShiftEvaluator, evaluate_predictions
import io
import wave
import tempfile
import gc
import matplotlib.pyplot as plt


class LlamaTurnPredictor:
    def __init__(self, threshold=0.4, flip_predictions=False, device="cuda"):
        self.device = device
        self.threshold = threshold
        self.flip_predictions = flip_predictions

        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

        compute_type = "float16" if device == "cuda" else "int8"
        self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
        print("Models loaded successfully")

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> tuple[str, float]:
        start_time = time.time()

        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk / np.abs(audio_chunk).max() * 32767).astype(np.int16)

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_chunk.tobytes())

            wav_io.seek(0)
            audio_segment = AudioSegment.from_wav(wav_io)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                segments, info = self.whisper_model.transcribe(
                    temp_file.name,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                text = " ".join(segment.text for segment in segments)

        return text, time.time() - start_time

    def predict_trp(self, text: str) -> tuple[bool, float, float, tuple[float, float]]:
        start_time = time.time()

        messages = [
            {"role": "system",
             "content": """You are having a conversation with a user. You will get a speech segment of the user's turn. If you think the user has finished their turn or it is appropriate for you to take your turn, answer with 'yes'. If you think the user is not done with their turn, answer with 'no'. Answer with only 'yes' or 'no'.

        Here are some examples:

        Speech segment: "I was walking in the park"
        You: no

        Speech segment: "I was walking in the park last night"
        You: yes

        Speech segment: "I was walking in the park last night, and I saw a squirrel."
        You: yes

        Speech segment: "what kind of movie do you generally like"
        You: yes

        Speech segment: "I recently actually went to see a movie"
        You: no"""
             },
            {"role": "user",
             "content": "Speech segment: {text}"}
        ]

        input_text = f"{messages[0]['content']}\n{messages[1]['content']}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=1,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

            logits = outputs.scores[0][0]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            yes_token = self.tokenizer.encode(" yes")[-1]
            no_token = self.tokenizer.encode(" no")[-1]

            yes_prob = float(probabilities[yes_token].cpu())
            no_prob = float(probabilities[no_token].cpu())

            total = yes_prob + no_prob
            if total > 0:
                yes_prob /= total
                no_prob /= total
            else:
                yes_prob = no_prob = 0.5

            if self.flip_predictions:
                has_trp = yes_prob <= self.threshold
                confidence = no_prob if has_trp else yes_prob
            else:
                has_trp = yes_prob > self.threshold
                confidence = yes_prob if has_trp else no_prob

        return has_trp, confidence, time.time() - start_time, (yes_prob, no_prob)

    def process_stream(self, audio_path: str, chunk_size=1.0):
        print(f"\nProcessing: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)

        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        audio_array = np.array(audio.get_array_of_samples())
        chunk_samples = int(chunk_size * 16000)

        predictions = []
        probabilities = []
        total_asr_time = 0
        total_llm_time = 0
        total_chunks = 0

        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            try:
                text, asr_time = self.transcribe_chunk(chunk)
                has_trp, confidence, llm_time, (yes_prob, no_prob) = self.predict_trp(text)

                total_asr_time += asr_time
                total_llm_time += llm_time
                total_chunks += 1

                predictions.append(has_trp)
                probabilities.append(yes_prob)

            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue

        total_time = total_asr_time + total_llm_time
        total_audio_duration = len(audio_array) / 16000
        rtf = total_time / total_audio_duration

        frame_predictions = np.repeat(predictions, int(50 * chunk_size))
        frame_probabilities = np.repeat(probabilities, int(50 * chunk_size))

        return frame_predictions, frame_probabilities, rtf

    def __del__(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()


class LlamaTurnEvaluator(TurnShiftEvaluator):
    def __init__(self, threshold=0.4, flip_predictions=False, device="cuda"):
        self.device = device
        self.model = LlamaTurnPredictor(threshold=threshold, flip_predictions=flip_predictions, device=device)

    def evaluate_file(self, audio_path, ground_truth_data, flip_predictions=False):
        predictions, probabilities, rtf = self.model.process_stream(audio_path)

        min_len = min(len(predictions), len(ground_truth_data['ground_truth']))
        predictions = predictions[:min_len]
        probabilities = probabilities[:min_len]
        ground_truth = ground_truth_data['ground_truth'][:min_len]

        metrics = evaluate_predictions(ground_truth, predictions, window_size=75)
        metrics['real_time_factor'] = float(rtf)

        return metrics, predictions, probabilities

    def evaluate_directory(self, audio_dir, ground_truth_data, output_dir, flip_predictions=False):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        all_results = {}

        for audio_file, gt_data in ground_truth_data.items():
            audio_path = Path(audio_dir) / audio_file
            if audio_path.exists():
                print(f"\nProcessing: {audio_file}")
                try:
                    metrics, predictions, probabilities = self.evaluate_file(
                        str(audio_path),
                        gt_data,
                        flip_predictions=flip_predictions
                    )

                    all_results[audio_file] = {
                        'metrics': metrics,
                        'ground_truth': gt_data['ground_truth'].tolist(),
                        'predictions': predictions.tolist()
                    }

                    # Save probabilities to CSV
                    prob_df = pd.DataFrame({
                        'time': np.arange(len(probabilities)) / 50,  # Convert frames to seconds
                        'probability': probabilities,
                        'prediction': predictions,
                        'ground_truth': gt_data['ground_truth'][:len(predictions)]
                    })
                    prob_df.to_csv(output_dir / f"{audio_file}_probabilities.csv", index=False)

                    self.plot_comparison(
                        audio_file,
                        gt_data['ground_truth'],
                        predictions,
                        output_dir / f"{audio_file}_analysis.png"
                    )

                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
                    continue

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

            with open(output_dir / "llama_evaluation_results.json", "w") as f:
                json.dump(final_results, f, indent=4)

            metrics_df.to_csv(output_dir / "llama_evaluation_metrics.csv")

            return final_results

        return None

    def plot_comparison(self, audio_name, ground_truth, predictions, save_path, window_size=75):
        plt.figure(figsize=(15, 12))

        min_len = min(len(ground_truth), len(predictions))
        ground_truth = ground_truth[:min_len]
        predictions = predictions[:min_len]
        time_axis = np.arange(min_len) / 50

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
    audio_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/CCPE/generated_audio")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/output")
    flip_predictions = False

    print("Extracting ground truth...")
    extractor = TurnShiftGroundTruth()
    ground_truth = extractor.process_directory(audio_dir)

    print("\nEvaluating with LLAMA...")
    evaluator = LlamaTurnEvaluator(threshold=0.4, flip_predictions=flip_predictions)
    results = evaluator.evaluate_directory(audio_dir, ground_truth, output_dir, flip_predictions=flip_predictions)

    if results:
        print("\nEvaluation Summary:")
        mean_metrics = results['mean_metrics']
        std_metrics = results['std_metrics']

        print(f"Prediction Mode: {'Flipped' if flip_predictions else 'Normal'}")
        print(f"Overall Accuracy: {results['summary']['overall_accuracy']:.3f}")
        print(f"Mean Accuracy: {mean_metrics['accuracy']:.3f} (±{std_metrics['accuracy']:.3f})")
        print(f"Mean Balanced Accuracy: {mean_metrics['balanced_accuracy']:.3f} (±{std_metrics['balanced_accuracy']:.3f})")
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