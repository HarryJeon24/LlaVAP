# ICC_Ground_Truth_Processing.py
from pydub import AudioSegment
import numpy as np
import warnings


class TRPGroundTruth:
    def __init__(self, response_threshold=0.3, frame_rate=50, window_size_sec=1.5):
        self.threshold = response_threshold
        self.frame_rate = frame_rate
        self.window_frames = int(window_size_sec * frame_rate)
        self.initial_trim_sec = 2.0  # Only for stimulus

    def load_audio(self, file_path):
        """Load audio using pydub and convert to numpy array"""
        audio = AudioSegment.from_wav(file_path)
        samples = np.array(audio.get_array_of_samples())

        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        return samples, audio.frame_rate

    def get_stimulus_info(self, stimulus_path):
        """Get trimmed stimulus duration to use as reference"""
        stimulus, sample_rate = self.load_audio(stimulus_path)
        # Trim first 2 seconds from stimulus
        trim_samples = int(self.initial_trim_sec * sample_rate)
        stimulus = stimulus[trim_samples:]
        stimulus_duration = len(stimulus) / sample_rate
        return stimulus_duration, sample_rate

    def load_responses(self, stimulus_path, response_paths):
        """Load stimulus and aggregate responses for TRP detection"""
        # First get trimmed stimulus duration as reference
        stimulus_duration, _ = self.get_stimulus_info(stimulus_path)
        expected_frames = int(stimulus_duration * self.frame_rate)

        print(f"Trimmed stimulus duration: {stimulus_duration:.2f}s")
        print(f"Expected frames: {expected_frames}")

        # Initialize arrays for aggregating responses
        all_response_arrays = []  # Store each participant's response pattern
        response_durations = []  # Store durations of individual responses
        valid_responses = 0

        # Process each response file
        for response_path in response_paths:
            try:
                print(f"\nProcessing: {response_path}")
                samples, sr = self.load_audio(response_path)

                # Calculate required samples for matching stimulus duration
                required_samples = int(stimulus_duration * sr)

                # Check if response is long enough
                if len(samples) >= required_samples:
                    # Trim from front to match stimulus length
                    samples = samples[:required_samples]
                    print(f"Trimmed response from front to {len(samples) / sr:.2f}s")

                    # Find response moments using a threshold
                    threshold = np.max(np.abs(samples)) * 0.1
                    response_indices = np.where(np.abs(samples) > threshold)[0]

                    # Convert to frames (50Hz)
                    response_frames = (response_indices / sr * self.frame_rate).astype(int)
                    response_frames = response_frames[response_frames < expected_frames]

                    # Create binary array for this participant's responses
                    participant_responses = np.zeros(expected_frames)
                    participant_responses[response_frames] = 1

                    # Find response durations
                    if len(response_indices) > 0:
                        # Group consecutive indices to find response segments
                        segments = np.split(response_indices, np.where(np.diff(response_indices) > 1)[0] + 1)
                        for segment in segments:
                            if len(segment) > 0:
                                duration_frames = (segment[-1] - segment[0]) / sr * self.frame_rate
                                response_durations.append(duration_frames)

                    all_response_arrays.append(participant_responses)
                    valid_responses += 1
                    print(f"Successfully processed")
                else:
                    print(f"Response too short: {len(samples) / sr:.2f}s vs required {stimulus_duration:.2f}s")
                    continue

            except Exception as e:
                print(f"Error processing {response_path}: {str(e)}")
                continue

        if valid_responses == 0:
            raise ValueError("No valid responses found")

        # Stack and calculate proportions
        response_array = np.stack(all_response_arrays)
        response_proportions = np.mean(response_array, axis=0)

        # TRP occurs where proportion exceeds threshold
        ground_truth = response_proportions >= self.threshold

        # Calculate response duration statistics
        duration_stats = {
            'avg_duration': np.mean(response_durations) if response_durations else 0,
            'std_duration': np.std(response_durations) if response_durations else 0,
            'min_duration': np.min(response_durations) if response_durations else 0,
            'max_duration': np.max(response_durations) if response_durations else 0,
            'num_responses': len(response_durations),
            'valid_responses': valid_responses
        }

        print(f"\nResponse Statistics:")
        print(f"Valid responses processed: {valid_responses}")
        print(f"Total response segments found: {len(response_durations)}")
        print(f"Average response duration: {duration_stats['avg_duration']:.2f} frames")
        print(f"Response duration std dev: {duration_stats['std_duration']:.2f} frames")
        print(f"Total TRPs found: {np.sum(ground_truth)} (where >{self.threshold * 100}% participants responded)")

        return ground_truth, response_proportions, duration_stats