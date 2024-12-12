import torch
import torchaudio
from vap.audio import load_waveform
import numpy as np


def analyze_audio_file(file_path):
    """Analyze an audio file and print its properties"""
    print(f"\nAnalyzing: {file_path}")
    print("-" * 50)

    # Try different loading methods
    try:
        # Method 1: Direct torchaudio load
        waveform_torch, sample_rate = torchaudio.load(file_path)
        print("\nTorchaudio direct load:")
        print(f"Shape: {waveform_torch.shape}")
        print(f"Sample rate: {sample_rate}")
        print(f"Data type: {waveform_torch.dtype}")
        print(f"Value range: [{waveform_torch.min():.3f}, {waveform_torch.max():.3f}]")
        print(f"Channels: {waveform_torch.shape[0]}")

        # Method 2: VAP's load_waveform
        waveform_vap, sr_vap = load_waveform(file_path, sample_rate=16000)
        print("\nVAP load_waveform:")
        print(f"Shape: {waveform_vap.shape}")
        print(f"Sample rate: {sr_vap}")
        print(f"Data type: {waveform_vap.dtype}")
        print(f"Value range: [{waveform_vap.min():.3f}, {waveform_vap.max():.3f}]")
        print(f"Channels: {waveform_vap.shape[0]}")

        # Check signal properties
        print("\nSignal properties:")
        # RMS level
        rms = torch.sqrt(torch.mean(waveform_vap ** 2, dim=1))
        print(f"RMS levels per channel: {rms.tolist()}")

        # Channel correlation
        if waveform_vap.shape[0] == 2:
            correlation = torch.corrcoef(waveform_vap)[0, 1].item()
            print(f"Channel correlation: {correlation:.3f}")

        # Check for silence/padding
        silence_threshold = 0.001
        silent_ratio = torch.mean((torch.abs(waveform_vap) < silence_threshold).float(), dim=1)
        print(f"Silent ratio per channel: {silent_ratio.tolist()}")

    except Exception as e:
        print(f"Error analyzing file: {str(e)}")
        import traceback
        traceback.print_exc()


def prepare_chunk_test(file_path, chunk_duration=0.5, sample_rate=16000):
    """Test chunk preparation for a given file"""
    print(f"\nTesting chunk preparation for: {file_path}")
    print("-" * 50)

    try:
        # Load audio
        waveform, sr = load_waveform(file_path, sample_rate=sample_rate)

        # Add batch dimension if needed
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Calculate chunk parameters
        chunk_samples = int(chunk_duration * sample_rate)

        # Get first chunk
        first_chunk = waveform[..., :chunk_samples]

        print(f"Original waveform shape: {waveform.shape}")
        print(f"Chunk shape: {first_chunk.shape}")
        print(f"Chunk duration: {chunk_duration}s ({chunk_samples} samples)")
        print(f"Chunk data type: {first_chunk.dtype}")
        print(f"Chunk value range: [{first_chunk.min():.3f}, {first_chunk.max():.3f}]")

    except Exception as e:
        print(f"Error in chunk test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test working file
    working_file = "C:/Users/Harry/PycharmProjects/LLA-VAP/VoiceActivityProjection/example/student_long_female_en-US-Wavenet-G.wav"
    analyze_audio_file(working_file)
    prepare_chunk_test(working_file)

    # Test problematic file
    problem_file = "C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios/stimulus_lists/stimulus lists/stimulus_0.wav"
    analyze_audio_file(problem_file)
    prepare_chunk_test(problem_file)