import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
from lstm_ensemble_CCPE import LSTMEnsemble, TurnPredictionDataset

def measure_inference_rtf(model, test_loader, device, frame_shift_ms=10):
    """
    Measure the Real-Time Factor (RTF) of model inference.

    Args:
        model: The LSTM ensemble model
        test_loader: DataLoader containing test data
        device: torch device (cuda/cpu)
        frame_shift_ms: Frame shift in milliseconds (default=10ms)

    Returns:
        float: Average RTF
        float: Average inference time per sequence in seconds
        float: Throughput (sequences per second)
    """
    model.eval()
    total_time = 0
    total_sequences = 0

    # Get sequence length from the model
    sequence_length = model.sequence_length

    # Calculate audio duration for each sequence
    sequence_duration = (sequence_length * frame_shift_ms) / 1000  # in seconds

    with torch.no_grad():
        for features, _ in tqdm(test_loader, desc="Measuring RTF"):
            features = features.to(device)
            batch_size = features.size(0)

            # Measure inference time
            start_time = time.perf_counter()
            _ = model(features)
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            total_time += inference_time
            total_sequences += batch_size

    # Calculate metrics
    average_time_per_sequence = total_time / total_sequences
    rtf = average_time_per_sequence / sequence_duration
    throughput = total_sequences / total_time

    return rtf, average_time_per_sequence, throughput


def print_rtf_stats(rtf, avg_time, throughput):
    """Print RTF measurement statistics."""
    print("\nRTF Measurement Results:")
    print(f"Real-Time Factor (RTF): {rtf:.4f}")
    print(f"Average inference time per sequence: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} sequences/second")

    if rtf < 1:
        print("\nModel can run in real-time ✓")
        print(f"Processing is {1 / rtf:.2f}x faster than real-time")
    else:
        print("\nModel cannot run in real-time ✗")
        print(f"Processing is {rtf:.2f}x slower than real-time")

def load_test_data(vap_path: Path, llama_path: Path):
    vap_files = list(vap_path.glob('*_probabilities.csv'))
    all_features = []
    all_labels = []

    for vap_file in vap_files:
        base_name = vap_file.stem.replace('_probabilities', '')
        llama_file = llama_path / f"{base_name}_probabilities.csv"

        if llama_file.exists():
            vap_data = pd.read_csv(vap_file)
            llama_data = pd.read_csv(llama_file)

            if len(vap_data) == len(llama_data):
                features = np.column_stack([
                    vap_data['probability'].values,
                    llama_data['probability'].values
                ])
                labels = vap_data['ground_truth'].values

                all_features.append(features)
                all_labels.append(labels)

    return np.vstack(all_features), np.concatenate(all_labels)


def measure_rtf():
    # Paths
    vap_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_CCPE_output/probs")
    llama_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_CCPE_output/3b/prob")
    model_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lstm_ensemble_results_CCPE")

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    features, labels = load_test_data(vap_path, llama_path)

    # Create test dataset
    test_dataset = TurnPredictionDataset(
        features=features,
        labels=labels,
        sequence_length=100
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Load model
    model = LSTMEnsemble(
        sequence_length=100,
        hidden_size=128,
        num_layers=2
    ).to(device)

    # Load best model from each fold and measure RTF
    for fold in range(5):
        model_path = model_dir / f"fold{fold + 1}_model.pt"
        if model_path.exists():
            print(f"\nTesting fold {fold + 1} model")
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            rtf, avg_time, throughput = measure_inference_rtf(model, test_loader, device)
            print_rtf_stats(rtf, avg_time, throughput)


if __name__ == "__main__":
    measure_rtf()