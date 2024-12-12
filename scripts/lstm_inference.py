import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings


class TurnPredictionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.features, self.labels = self.prepare_data(features, labels)
        self.length = len(self.features)

    def prepare_data(self, features: np.ndarray, labels: np.ndarray):
        # Calculate rolling statistics
        vap_probs = features[:, 0]
        llama_probs = features[:, 1]

        vap_rolling = pd.Series(vap_probs)
        llama_rolling = pd.Series(llama_probs)

        # Multiple window sizes for different temporal scales
        window_sizes = [5, 10, 20]
        additional_features = []

        for window in window_sizes:
            window_features = np.column_stack([
                vap_rolling.rolling(window=window, center=True).mean().fillna(0),
                llama_rolling.rolling(window=window, center=True).std().fillna(0),
                vap_rolling.rolling(window=window, center=True).max().fillna(0),
                vap_rolling.rolling(window=window, center=True).min().fillna(0)
            ])
            additional_features.append(window_features)

        # Combine all features
        combined_features = np.column_stack([
            features,
            np.hstack(additional_features),
            vap_probs * llama_probs,  # Interaction
            np.abs(vap_probs - llama_probs)  # Difference
        ])

        # Scale features
        combined_features = self.scaler.fit_transform(combined_features)

        # Create sequences
        sequences = []
        sequence_labels = []

        for i in range(self.sequence_length - 1, len(combined_features)):
            seq = combined_features[i - self.sequence_length + 1:i + 1]
            if len(seq) == self.sequence_length:
                sequences.append(seq)
                sequence_labels.append(labels[i])

        return (
            np.array(sequences, dtype=np.float32),
            np.array(sequence_labels, dtype=np.float32).reshape(-1, 1)
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )


def plot_prediction_analysis(ground_truth, predictions, probabilities, save_path, window_size=75):
    """Plot analysis of ensemble results."""
    time_axis = np.arange(len(ground_truth)) / 50  # 50Hz to seconds

    plt.figure(figsize=(15, 12))

    # Plot 1: Ground Truth vs Predictions
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, ground_truth, 'g-', label='Ground Truth')
    plt.plot(time_axis, predictions, 'r--', alpha=0.5, label='LSTM Predictions')
    plt.title('Ground Truth vs LSTM Predictions')
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

    # Plot 3: LSTM Probabilities
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, probabilities, 'b-', label='LSTM Probability')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Default Threshold')
    plt.title('LSTM Probabilities')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.legend()

    # Adjust layout with specific margins
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths
    vap_file = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_CCPE_output/probs/CCPE-0a55a.wav_probabilities.csv")
    llama_file = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_CCPE_output/3b/0.4_no_flip_new_prompt/CCPE-0a55a.wav_probabilities.csv")
    model_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lstm_ensemble_results_CCPE/fold1_model.pt")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lstm_ensemble_results_CCPE")

    # Load data
    vap_data = pd.read_csv(vap_file)
    llama_data = pd.read_csv(llama_file)

    # Prepare features
    features = np.column_stack([
        vap_data['probability'].values,
        llama_data['probability'].values
    ])
    labels = vap_data['ground_truth'].values

    # Create dataset
    dataset = TurnPredictionDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint['model_state_dict']

    # Initialize model with same parameters as training
    from lstm_ensemble_CCPE import LSTMEnsemble  # Import the model class
    model = LSTMEnsemble(
        sequence_length=100,
        hidden_size=128,
        num_layers=2
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    # Perform inference
    all_probabilities = []

    with torch.no_grad():
        for batch_features, _ in dataloader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            probabilities = torch.sigmoid(outputs)
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to numpy array and flatten
    all_probabilities = np.array(all_probabilities).flatten()

    # Pad the beginning to match original length
    pad_length = len(labels) - len(all_probabilities)
    padded_probabilities = np.pad(all_probabilities, (pad_length, 0), mode='edge')

    # Generate predictions using threshold
    predictions = (padded_probabilities > 0.5).astype(int)

    # Create visualization
    plot_prediction_analysis(
        labels,
        predictions,
        padded_probabilities,
        output_dir / "CCPE-0a55a_analysis.png"
    )

    # Save predictions to CSV (ensure all arrays are 1D)
    results_df = pd.DataFrame({
        'ground_truth': labels.flatten(),
        'probability': padded_probabilities.flatten(),
        'prediction': predictions.flatten()
    })
    results_df.to_csv(output_dir / "CCPE-0a55a_predictions.csv", index=False)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()