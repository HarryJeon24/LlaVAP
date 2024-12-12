# baseline_llm_pipeline_prob_ICC.py
import numpy as np
import whisperx
import pandas as pd
from pathlib import Path
import json
import warnings
from typing import List, Tuple, Dict
import time
import torch
import gc
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel, BatchedInferencePipeline
import soundfile as sf
import tempfile
from scipy import signal
from pydub import AudioSegment
import io
import wave
from ICC_Ground_Truth_Processing import TRPGroundTruth
from baseline_VAP_pipeline import evaluate_vap_predictions, plot_comparison
from VAP import process_dataset

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_SYMLINKS"] = "0"


class RealTimeTRPPredictor:
    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 chunk_size: float = 1.0,
                 max_seq_len: int = 4096,
                 threshold: float = 0.5,
                 flip_predictions: bool = False):

        self.threshold = threshold
        self.flip_predictions = flip_predictions

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.whisper_model = WhisperModel(
            "small",
            device=self.device,
            compute_type=compute_type
        )
        self.whisper_model = BatchedInferencePipeline(model=self.whisper_model)

        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.audio_context = []

        print(f"Models loaded successfully")

    def predict_trp(self, current: str) -> Tuple[bool, float, float, Tuple[float, float]]:
        start_time = time.time()

        # Format the input
        messages = [
            {"role": "system",
             "content": "You are having a conversation with a user. you will get a speech segment of the user's turn. If you think the user has finished their turn or it is appropriate for you to take your turn, answer with 'yes'. If you think the user is not done with their turn, answer with 'no'. Answer with only 'yes' or 'no'."},
            {"role": "user",
             "content": f"Speech segment: {current}"}
        ]

        # Convert messages to model input format
        input_text = f"{messages[0]['content']}\n{messages[1]['content']}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get model outputs
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

            # Get the logits from the last token
            logits = outputs.scores[0][0]  # Get logits for the generated token
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Get token IDs for yes and no
            yes_tokens = self.tokenizer.encode(" yes")[-1]  # Get the last token ID
            no_tokens = self.tokenizer.encode(" no")[-1]  # Get the last token ID

            # Get probabilities
            yes_prob = float(probabilities[yes_tokens].cpu())
            no_prob = float(probabilities[no_tokens].cpu())

            # Normalize
            total = yes_prob + no_prob
            if total > 0:
                yes_prob /= total
                no_prob /= total
            else:
                yes_prob = no_prob = 0.5

            print(f"Debug - Yes prob: {yes_prob:.3f}, No prob: {no_prob:.3f}")

            # Get prediction
            if self.flip_predictions:
                has_trp = yes_prob <= self.threshold
                confidence = no_prob if has_trp else yes_prob
            else:
                has_trp = yes_prob > self.threshold
                confidence = yes_prob if has_trp else no_prob

        inference_time = time.time() - start_time
        return has_trp, confidence, inference_time, (yes_prob, no_prob)

    def process_stream(self, audio_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, float]]:
        print(f"\nLoading audio file: {audio_path}")
        audio = AudioSegment.from_wav(audio_path)

        print(f"Initial audio info:")
        print(f"Duration: {len(audio)}ms")
        print(f"Channels: {audio.channels}")
        print(f"Sample width: {audio.sample_width}")
        print(f"Frame rate: {audio.frame_rate}")

        if audio.sample_width != 2:
            print("Converting to 16-bit audio")
            audio = audio.set_sample_width(2)

        if audio.channels > 1:
            print("Converting stereo to mono")
            audio = audio.set_channels(1)

        if audio.frame_rate != 16000:
            print(f"Resampling from {audio.frame_rate}Hz to 16000Hz")
            audio = audio.set_frame_rate(16000)

        audio_array = np.array(audio.get_array_of_samples())

        chunk_samples = int(self.chunk_size * 16000)
        predictions = []
        probabilities = []
        raw_probabilities = []
        transcripts = []

        total_asr_time = 0
        total_llm_time = 0
        total_chunks = 0

        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            try:
                self.audio_context.append(chunk)
                self.audio_context = self.audio_context[-10:] if len(self.audio_context) > 10 else self.audio_context
                context_audio = np.concatenate(self.audio_context)

                print(f"\n{'=' * 80}")
                print(f"Chunk {total_chunks + 1}:")
                print(
                    f"Audio context length: {len(self.audio_context)} chunks ({len(context_audio) / 16000:.2f} seconds)")

                current_text, asr_time = self.transcribe_chunk(context_audio)
                print(f"Context transcription: \"{current_text}\"")

                has_trp, confidence, llm_time, (yes_prob, no_prob) = self.predict_trp(current_text)

                total_asr_time += asr_time
                total_llm_time += llm_time
                total_chunks += 1

                predictions.append(has_trp)
                probabilities.append(confidence)
                raw_probabilities.append((yes_prob, no_prob))
                transcripts.append(current_text)

                print(f"\nTRP Prediction: {has_trp} (confidence: {confidence:.3f})")
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
                print(f"{'=' * 80}\n")

            except Exception as e:
                print(f"Error processing chunk {total_chunks}: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        total_time = total_asr_time + total_llm_time
        total_audio_duration = len(audio_array) / 16000

        rtf_metrics = {
            'rtf_whisper': total_asr_time / total_audio_duration,
            'rtf_llm': total_llm_time / total_audio_duration,
            'rtf_combined': total_time / total_audio_duration,
            'avg_asr_time': total_asr_time / total_chunks,
            'avg_llm_time': total_llm_time / total_chunks,
            'total_audio_duration': total_audio_duration,
            'total_processing_time': total_time
        }

        frame_predictions = np.repeat(predictions, int(50 * self.chunk_size))
        frame_probabilities = np.repeat(probabilities, int(50 * self.chunk_size))
        frame_raw_probs = np.array([np.repeat(p, int(50 * self.chunk_size)) for p in zip(*raw_probabilities)]).T

        return frame_predictions, frame_probabilities, frame_raw_probs, transcripts, rtf_metrics

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> Tuple[str, float]:
        start_time = time.time()

        if audio_chunk.dtype != np.int16:
            audio_chunk = audio_chunk.astype(np.float32)
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
                    batch_size=16,
                    language="en",
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )

                segments = list(segments)
                text = " ".join(segment.text for segment in segments)

        inference_time = time.time() - start_time
        return text, inference_time

    def __del__(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()


def save_predictions(output_dir: Path, predictions: Dict):
    output_dir.mkdir(exist_ok=True)

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    converted_predictions = convert_numpy_types(predictions)

    with open(output_dir / "llama_realtime_results.json", 'w') as f:
        json.dump(converted_predictions, f, indent=4)

    metrics_df = pd.DataFrame([convert_numpy_types(r['metrics']) for r in predictions.values()])
    rtf_df = pd.DataFrame([convert_numpy_types(r['rtf_metrics']) for r in predictions.values()])

    metrics_df.to_csv(output_dir / "metrics.csv")
    rtf_df.to_csv(output_dir / "rtf_metrics.csv")

    return metrics_df, rtf_df

def main():
    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_ICC_output/3b/new_prompt")

    print(f"\nLooking for files in: {base_path.absolute()}")
    output_dir.mkdir(exist_ok=True)

    try:
        print("Initializing predictor...")
        predictor = RealTimeTRPPredictor(threshold=0.7, flip_predictions=False)
        print("Initializing TRP processor...")
        trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50, window_size_sec=1.5)

        stimulus_groups = process_dataset(base_path)

        all_results = {}
        for stim_key, group in stimulus_groups.items():
            print(f"\nProcessing {stim_key}")
            if not group['response_paths']:
                print(f"No response files found for stimulus {stim_key}")
                continue

            try:
                ground_truth, response_proportions, duration_stats = trp_processor.load_responses(
                    str(group['stimulus_path']),
                    [str(path) for path in group['response_paths']]
                )

                frame_predictions, frame_probabilities, frame_raw_probs, transcripts, rtf_metrics = predictor.process_stream(
                    str(group['stimulus_path'])
                )

                print("\nReal-time Factors:")
                print(f"Whisper RTF: {rtf_metrics['rtf_whisper']:.3f}")
                print(f"LLM RTF: {rtf_metrics['rtf_llm']:.3f}")
                print(f"Combined RTF: {rtf_metrics['rtf_combined']:.3f}")

                min_len = min(len(ground_truth), len(frame_predictions))
                ground_truth = ground_truth[:min_len]
                frame_predictions = frame_predictions[:min_len]
                frame_probabilities = frame_probabilities[:min_len]
                frame_raw_probs = frame_raw_probs[:min_len]
                response_proportions = response_proportions[:min_len]

                window_size = max(25, int(duration_stats['avg_duration']))
                metrics = evaluate_vap_predictions(ground_truth, frame_predictions, window_size=window_size)

                all_results[stim_key] = {
                    'metrics': metrics,
                    'rtf_metrics': rtf_metrics,
                    'ground_truth': ground_truth.tolist(),
                    'predictions': frame_predictions.tolist(),
                    'probabilities': frame_probabilities.tolist(),
                    'raw_probabilities': frame_raw_probs.tolist(),
                    'response_proportions': response_proportions.tolist(),
                    'duration_stats': duration_stats,
                    'transcripts': transcripts
                }

                plot_comparison(
                    stim_key,
                    ground_truth,
                    frame_predictions,
                    response_proportions,
                    save_path=output_dir / f"{stim_key}_analysis.png",
                    window_size=window_size
                )

            except Exception as e:
                print(f"Error processing {stim_key}: {str(e)}")
                continue

        if all_results:
            metrics_df, rtf_df = save_predictions(output_dir, all_results)

            print("\nResults Summary:")
            for metric in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'sensitivity']:
                mean = metrics_df[metric].mean()
                std = metrics_df[metric].std()
                print(f"{metric}: {mean:.3f} (±{std:.3f})")

            print("\nAverage RTF Metrics:")
            for metric in ['rtf_whisper', 'rtf_llm', 'rtf_combined']:
                mean = rtf_df[metric].mean()
                std = rtf_df[metric].std()
                print(f"{metric}: {mean:.3f} (±{std:.3f})")
        else:
            print("\nNo results were generated. Please check your input files and paths.")

    except Exception as e:
        print(f"Critical error: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    finally:
        print("\nProcessing complete.")

if __name__ == "__main__":
    main()