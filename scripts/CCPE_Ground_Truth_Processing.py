# CCPE_Ground_Truth_Processing.py
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence


class TurnShiftGroundTruth:
    def __init__(self, silence_threshold_db=-40, min_silence_len=1800, frame_rate=50):  # 1800ms = 1.8s
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
        return [(start / 1000, end / 1000) for start, end in silence_ranges]  # Convert to seconds

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

            # Convert silence periods to turn shift points (take middle of each silence)
            turn_shifts = [
                (start + (end - start) / 2)  # Middle point of silence
                for start, end in silences
            ]

            # Create frame-level ground truth
            total_frames = int(len(audio) / 1000 * self.frame_rate)
            ground_truth = self.create_frame_level_ground_truth(turn_shifts, total_frames)

            result = {
                'file_path': str(audio_path),
                'duration': len(audio) / 1000,  # Convert to seconds
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