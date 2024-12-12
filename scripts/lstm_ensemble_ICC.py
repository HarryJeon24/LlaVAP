# lstm_ensemble_ICC.py - Part 1: Core Classes
import numpy as np
import pandas as pd
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm
import logging
from ICC_Ground_Truth_Processing import TRPGroundTruth
from VAP import process_dataset


def plot_comparison(audio_name, ground_truth, predictions, save_path, window_size=75):
    """Plot detailed comparison of ground truth and predictions."""
    plt.figure(figsize=(15, 12))

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
    plt.grid(True)

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
    plt.grid(True)

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
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        else:
            if self.mode == 'max':
                if score <= (self.best_score + self.delta):
                    self.counter += 1
                else:
                    self.best_score = score
                    self.best_state = model.state_dict()
                    self.counter = 0
            else:  # mode == 'min'
                if score >= (self.best_score - self.delta):
                    self.counter += 1
                else:
                    self.best_score = score
                    self.best_state = model.state_dict()
                    self.counter = 0

        if self.counter >= self.patience:
            self.early_stop = True


class WeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=3.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        inputs = inputs.view(-1, 1)
        targets = targets.view(-1, 1)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class LSTMEnsemble(nn.Module):
    def __init__(self, sequence_length=100, hidden_size=128, num_layers=2):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input features: VAP prob, LLaMA prob + engineered features
        # 2 base + (3 windows * 4 features) + 2 additional = 16 features
        self.input_size = 16

        # Deep feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(4)
        ])

        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Feature extraction
        features = self.feature_extraction(x)

        # LSTM processing
        lstm_out, _ = self.lstm(features)

        # Multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_weights = attention_head(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            attended = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
            attention_outputs.append(attended.squeeze(1))

        # Combine attention outputs
        attended = torch.cat(attention_outputs, dim=1)

        # Final processing
        out = self.fc1(attended)
        out = self.ln1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out


class TurnPredictionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 100):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.features, self.labels = self.prepare_data(features, labels)

    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]

        print(f"\nClass distribution before balancing:")
        print(f"Positive samples: {len(positive_indices)}")
        print(f"Negative samples: {len(negative_indices)}")

        # Calculate features
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
                llama_rolling.rolling(window=window, center=True).mean().fillna(0),
                vap_rolling.rolling(window=window, center=True).std().fillna(0),
                llama_rolling.rolling(window=window, center=True).std().fillna(0)
            ])
            additional_features.append(window_features)

        # Combine all features
        combined_features = np.column_stack([
            features,  # 2 features (vap, llama)
            np.hstack(additional_features),  # 12 features (3 windows * 4 features each)
            vap_probs * llama_probs,  # 1 feature (interaction)
            np.abs(vap_probs - llama_probs)  # 1 feature (difference)
        ])

        combined_features = self.scaler.fit_transform(combined_features)

        # Create balanced sequences
        sequences = []
        sequence_labels = []

        # Process positive samples
        for idx in positive_indices:
            if idx >= self.sequence_length - 1:
                seq = combined_features[idx - self.sequence_length + 1:idx + 1]
                if len(seq) == self.sequence_length:
                    sequences.append(seq)
                    sequence_labels.append(1)

        # Sample equal number of negative sequences
        neg_samples_needed = len(sequences)
        if len(negative_indices) >= neg_samples_needed:
            sampled_neg_indices = np.random.choice(
                negative_indices[negative_indices >= self.sequence_length - 1],
                size=neg_samples_needed,
                replace=False
            )
        else:
            sampled_neg_indices = np.random.choice(
                negative_indices[negative_indices >= self.sequence_length - 1],
                size=neg_samples_needed,
                replace=True
            )

        for idx in sampled_neg_indices:
            seq = combined_features[idx - self.sequence_length + 1:idx + 1]
            if len(seq) == self.sequence_length:
                sequences.append(seq)
                sequence_labels.append(0)

        # Shuffle the balanced dataset
        shuffle_idx = np.random.permutation(len(sequences))
        sequences = np.array(sequences, dtype=np.float32)[shuffle_idx]
        sequence_labels = np.array(sequence_labels, dtype=np.float32)[shuffle_idx].reshape(-1, 1)

        print(f"\nClass distribution after balancing:")
        print(f"Total sequences: {len(sequences)}")
        print(f"Positive sequences: {np.sum(sequence_labels == 1)}")
        print(f"Negative sequences: {np.sum(sequence_labels == 0)}")

        return sequences, sequence_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor(self.labels[idx])


def convert_to_serializable(obj):
    """Convert numpy and torch types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    return obj


def calculate_metrics(true_labels, predictions, probabilities, loss=None):
    """Calculate comprehensive metrics for balanced evaluation."""
    metrics = {
        'loss': loss,
        'balanced_accuracy': balanced_accuracy_score(true_labels, predictions),
        'sensitivity': recall_score(true_labels, predictions, zero_division=0),
        'specificity': recall_score(true_labels, predictions, pos_label=0, zero_division=0),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'pos_ratio': np.mean(predictions),
        'true_pos_ratio': np.mean(true_labels)
    }

    # Add probability distribution statistics
    metrics.update({
        'prob_mean': np.mean(probabilities),
        'prob_std': np.std(probabilities),
        'prob_median': np.median(probabilities)
    })

    return metrics


def evaluate_vap_predictions(ground_truth, predictions, window_size, duration_stats):
    """Evaluate predictions using window-based approach for unbalanced evaluation."""
    metrics = {}

    # Debug prints
    print(f"\nGround truth stats before window creation:")
    print(f"Ground truth ones: {np.sum(ground_truth == 1)}")
    print(f"Ground truth zeros: {np.sum(ground_truth == 0)}")
    print(f"Window size being used: {window_size} frames ({window_size / 50:.3f} seconds)")

    # Create windows around ground truth positives
    true_windows = np.zeros_like(ground_truth)
    for i in np.where(ground_truth == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(ground_truth), i + window_size + 1)
        true_windows[start:end] = 1

    # Print window coverage statistics
    total_frames = len(ground_truth)
    positive_frames = np.sum(true_windows == 1)
    print(f"\nWindow Coverage Analysis:")
    print(f"Total frames: {total_frames} ({total_frames / 50:.2f} seconds)")
    print(f"Frames in positive windows: {positive_frames} ({positive_frames / 50:.2f} seconds)")
    print(f"Percentage of frames in windows: {100 * positive_frames / total_frames:.1f}%")
    print(f"Number of TRPs: {np.sum(ground_truth == 1)}")
    if np.sum(ground_truth == 1) > 1:
        print(f"Average gap between TRPs: {50 / np.mean(np.diff(np.where(ground_truth == 1)[0])):.3f} per second")

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
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = sensitivity  # Same as sensitivity
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) \
        if (metrics['precision'] + metrics['recall']) > 0 else 0

    # Calculate additional metrics for analysis
    metrics['trp_density'] = np.sum(ground_truth == 1) / len(ground_truth)  # TRPs per frame
    metrics['prediction_density'] = np.sum(predictions == 1) / len(predictions)  # Predictions per frame
    metrics['window_coverage'] = positive_frames / total_frames  # Proportion of frames in windows
    metrics['window_size_used'] = window_size  # Store window size used for reference

    # Store duration statistics
    metrics.update(duration_stats)

    return metrics


def plot_training_progress(history: dict, save_path: Path):
    """Plot detailed training progress."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss During Training')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot balanced accuracy
    ax2.plot(history['train_bacc'], label='Train Balanced Acc')
    ax2.plot(history['val_bacc'], label='Val Balanced Acc')
    ax2.set_title('Balanced Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)

    # Plot sensitivity/specificity
    ax3.plot(history['train_sensitivity'], label='Train Sensitivity')
    ax3.plot(history['train_specificity'], label='Train Specificity')
    ax3.plot(history['val_sensitivity'], label='Val Sensitivity')
    ax3.plot(history['val_specificity'], label='Val Specificity')
    ax3.set_title('Sensitivity/Specificity')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)

    # Plot F1 scores
    ax4.plot(history['train_f1'], label='Train F1')
    ax4.plot(history['val_f1'], label='Val F1')
    ax4.set_title('F1 Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train the model with comprehensive monitoring."""
    # Calculate positive weight from training data
    all_labels = torch.cat([labels for _, labels in train_loader.dataset])
    pos_samples = torch.sum(all_labels).item()
    neg_samples = len(all_labels) - pos_samples
    pos_weight = torch.tensor([neg_samples / pos_samples]).to(device)

    criterion = WeightedFocalLoss(pos_weight=pos_weight, gamma=3.0, alpha=0.75)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    early_stopping = EarlyStopping(patience=10, verbose=True, mode='max')
    history = defaultdict(list)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_outputs = defaultdict(list)

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for features, labels in train_pbar:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Store batch results
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                train_outputs['loss'].append(loss.item())
                train_outputs['preds'].extend(preds.cpu().numpy())
                train_outputs['probs'].extend(probs.cpu().numpy())
                train_outputs['labels'].extend(labels.cpu().numpy())

        # Validation phase
        model.eval()
        val_outputs = defaultdict(list)

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for features, labels in val_pbar:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_outputs['loss'].append(loss.item())
                val_outputs['preds'].extend(preds.cpu().numpy())
                val_outputs['probs'].extend(probs.cpu().numpy())
                val_outputs['labels'].extend(labels.cpu().numpy())

        # Calculate metrics for balanced evaluation
        train_metrics = calculate_metrics(
            train_outputs['labels'],
            train_outputs['preds'],
            train_outputs['probs'],
            np.mean(train_outputs['loss'])
        )

        val_metrics = calculate_metrics(
            val_outputs['labels'],
            val_outputs['preds'],
            val_outputs['probs'],
            np.mean(val_outputs['loss'])
        )

        # Store metrics in history
        history['train_loss'].append(np.mean(train_outputs['loss']))
        history['val_loss'].append(np.mean(val_outputs['loss']))
        history['train_bacc'].append(train_metrics['balanced_accuracy'])
        history['val_bacc'].append(val_metrics['balanced_accuracy'])
        history['train_sensitivity'].append(train_metrics['sensitivity'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['train_specificity'].append(train_metrics['specificity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])

        # Print epoch results
        print(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Balanced Acc: {train_metrics['balanced_accuracy']:.4f}")
        print(
            f"Train - Sensitivity: {train_metrics['sensitivity']:.4f}, Specificity: {train_metrics['specificity']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"Val - Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")

        # Check early stopping
        early_stopping(val_metrics['balanced_accuracy'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return early_stopping.best_state, history


def prepare_full_sequence_evaluation(features, model_params, device, scaler=None):
    """Prepare features for full sequence evaluation."""
    vap_probs = features[:, 0]
    llama_probs = features[:, 1]

    vap_rolling = pd.Series(vap_probs)
    llama_rolling = pd.Series(llama_probs)

    window_sizes = [5, 10, 20]
    additional_features = []

    for window in window_sizes:
        window_features = np.column_stack([
            vap_rolling.rolling(window=window, center=True).mean().fillna(0),
            llama_rolling.rolling(window=window, center=True).mean().fillna(0),
            vap_rolling.rolling(window=window, center=True).std().fillna(0),
            llama_rolling.rolling(window=window, center=True).std().fillna(0)
        ])
        additional_features.append(window_features)

    combined_features = np.column_stack([
        features,
        np.hstack(additional_features),
        vap_probs * llama_probs,
        np.abs(vap_probs - llama_probs)
    ])

    if scaler is None:
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
    else:
        combined_features = scaler.transform(combined_features)

    # Create sequences
    all_sequences = []
    for j in range(model_params['sequence_length'] - 1, len(combined_features)):
        seq = combined_features[j - model_params['sequence_length'] + 1:j + 1]
        all_sequences.append(seq)

    if all_sequences:
        sequences_tensor = torch.FloatTensor(np.array(all_sequences)).to(device)
        return sequences_tensor, scaler

    return None, scaler


def get_predictions_for_sequence(model, sequences_tensor, batch_size=512, threshold=0.5):
    predictions_list = []
    probabilities_list = []

    with torch.no_grad():
        for batch_start in range(0, len(sequences_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(sequences_tensor))
            batch = sequences_tensor[batch_start:batch_end]

            outputs = model(batch)
            batch_probs = torch.sigmoid(outputs).cpu().numpy()

            # Add threshold logging
            print(f"Probability distribution: min={batch_probs.min():.3f}, max={batch_probs.max():.3f}, "
                  f"mean={batch_probs.mean():.3f}, median={np.median(batch_probs):.3f}")

            batch_preds = (batch_probs > threshold).astype(float)

            predictions_list.extend(batch_preds)
            probabilities_list.extend(batch_probs)

    return np.array(predictions_list), np.array(probabilities_list)


def save_results(output_dir: Path, all_results: Dict, cv_results: List, metrics_summary: Dict):
    """Save all results with proper serialization."""
    output_dir.mkdir(exist_ok=True)

    final_results = {
        'balanced_evaluation': {
            'per_stimulus_results': {},
            'cross_validation_metrics': metrics_summary['balanced_metrics']
        },
        'unbalanced_evaluation': {
            'per_stimulus_results': all_results,
            'cross_validation_metrics': metrics_summary['unbalanced_metrics']
        },
        'model_params': metrics_summary['model_params'],
        'dataset_info': metrics_summary['dataset_info']
    }

    # Save full results
    with open(output_dir / "full_results.json", "w") as f:
        json.dump(convert_to_serializable(final_results), f, indent=4)

    # Save CV results
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(convert_to_serializable(cv_results), f, indent=4)

    # Save metrics summary
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(convert_to_serializable(metrics_summary), f, indent=4)

    print(f"\nResults saved to {output_dir}")


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set paths for ICC dataset
    base_path = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/datasets/ICC/Stimulus_List_and_Response_Audios")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lstm_ensemble_results_ICC")
    vap_probs_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_ICC_output/0.999_no_flip_best/full_results.json")
    llama_probs_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_ICC_output/3b/new_prompt/llama_realtime_results.json")

    output_dir.mkdir(exist_ok=True)

    try:
        # Process ICC dataset structure
        stimulus_groups = process_dataset(base_path)
        logger.info(f"Found {len(stimulus_groups)} stimulus groups")

        # Load model predictions from JSON files
        with open(vap_probs_path) as f:
            vap_results = json.load(f)
        with open(llama_probs_path) as f:
            llama_results = json.load(f)

        # Initialize TRP processor
        trp_processor = TRPGroundTruth(response_threshold=0.3, frame_rate=50, window_size_sec=1.5)

        # Prepare data for cross-validation
        all_features = []
        all_labels = []
        all_duration_stats = []
        stim_keys = []

        for stim_key, group in stimulus_groups.items():
            try:
                # Load ground truth and response proportions
                ground_truth, response_proportions, duration_stats = trp_processor.load_responses(
                    str(group['stimulus_path']),
                    [str(path) for path in group['response_paths']]
                )

                # Get predictions from JSON results
                if stim_key in vap_results['per_stimulus_results'] and stim_key in llama_results:
                    vap_probs = vap_results['per_stimulus_results'][stim_key]['probabilities']
                    llama_probs = llama_results[stim_key]['probabilities']

                    features = np.column_stack([
                        vap_probs,
                        llama_probs
                    ])

                    min_len = min(len(ground_truth), len(features))
                    features = features[:min_len]
                    ground_truth = ground_truth[:min_len]

                    all_features.append(features)
                    all_labels.append(ground_truth)
                    all_duration_stats.append(duration_stats)
                    stim_keys.append(stim_key)

            except Exception as e:
                logger.error(f"Error processing {stim_key}: {str(e)}")
                continue

        logger.info(f"Successfully loaded data for {len(stim_keys)} stimuli")

        # Model parameters
        model_params = {
            'sequence_length': 100,
            'hidden_size': 128,
            'num_layers': 2
        }

        train_params = {
            'batch_size': 32,
            'num_epochs': 50
        }

        # Prepare for Leave-One-Out Cross Validation
        kf = LeaveOneOut()
        cv_results = []
        all_results = {}
        balanced_results = {
            'per_stimulus_results': {},
            'cross_validation_metrics': {}
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_features)):
            logger.info(f"\nProcessing fold {fold + 1}/{len(all_features)}")

            # Prepare train/val data
            train_features = [all_features[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]

            val_features = [all_features[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]

            # Create datasets for balanced training
            train_dataset = TurnPredictionDataset(
                np.vstack(train_features),
                np.concatenate(train_labels),
                model_params['sequence_length']
            )

            val_dataset = TurnPredictionDataset(
                np.vstack(val_features),
                np.concatenate(val_labels),
                model_params['sequence_length']
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=train_params['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=train_params['batch_size'],
                num_workers=0,
                pin_memory=True
            )

            # Train model
            model = LSTMEnsemble(**model_params).to(device)
            best_model_state, history = train_model(
                model,
                train_loader,
                val_loader,
                device,
                num_epochs=train_params['num_epochs']
            )

            # Save balanced validation metrics from best epoch
            stim_key = stim_keys[val_idx[0]]
            best_epoch_idx = np.argmax(history['val_bacc'])
            balanced_metrics = {
                'loss': history['val_loss'][best_epoch_idx],
                'balanced_accuracy': history['val_bacc'][best_epoch_idx],
                'sensitivity': history['val_sensitivity'][best_epoch_idx],
                'specificity': history['val_specificity'][best_epoch_idx],
                'f1': history['val_f1'][best_epoch_idx]
            }
            balanced_results['per_stimulus_results'][stim_key] = balanced_metrics

            # Plot training progress
            plot_training_progress(history, output_dir / f"fold{fold + 1}_training_progress.png")

            # Save model
            torch.save({
                'model_state_dict': best_model_state,
                'history': history,
                'model_params': model_params,
                'train_params': train_params
            }, output_dir / f"fold{fold + 1}_model.pt")

            # Evaluate on full stimulus (unbalanced)
            model.load_state_dict(best_model_state)
            model.eval()

            # Process validation stimulus
            i = val_idx[0]  # LOOCV has one validation index
            stim_key = stim_keys[i]
            features = all_features[i]
            ground_truth = all_labels[i]
            duration_stats = all_duration_stats[i]

            # Prepare features for full sequence evaluation
            sequences_tensor, _ = prepare_full_sequence_evaluation(features, model_params, device)
            if sequences_tensor is not None:
                # Get predictions
                predictions, probabilities = get_predictions_for_sequence(model, sequences_tensor)

                # Pad the beginning where we don't have predictions
                pad_length = model_params['sequence_length'] - 1
                predictions = np.pad(predictions.flatten(), (pad_length, 0), 'edge')
                probabilities = np.pad(probabilities.flatten(), (pad_length, 0), 'edge')

                # Get window size from duration stats and evaluate
                window_size = int(duration_stats['avg_duration'])
                print(f"Using window size of {window_size} frames ({window_size / 50:.3f} seconds) "
                      f"based on average response duration")

                # Evaluate using window-based approach
                metrics = evaluate_vap_predictions(ground_truth, predictions, window_size, duration_stats)

                # Store unbalanced evaluation results
                all_results[f"{stim_key}_fold{fold + 1}"] = {
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'ground_truth': ground_truth.tolist(),
                    'metrics': metrics,
                    'duration_stats': duration_stats
                }

                # Plot comparison
                plot_comparison(
                    f"{stim_key}_fold{fold + 1}",
                    ground_truth,
                    predictions,
                    output_dir / f"{stim_key}_fold{fold + 1}_analysis.png",
                    window_size=window_size
                )

            # Store fold results
            cv_results.append({
                'fold': fold + 1,
                'history': history,
                'train_files': [stim_keys[i] for i in train_idx],
                'val_files': [stim_keys[i] for i in val_idx],
                'balanced_metrics': balanced_metrics,
                'unbalanced_metrics': metrics if sequences_tensor is not None else None
            })

        # Calculate final metrics summaries
        balanced_summary = {}
        for metric in ['loss', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1']:
            values = [res[metric] for res in balanced_results['per_stimulus_results'].values()]
            balanced_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        unbalanced_summary = {}
        for metric in next(iter(all_results.values()))['metrics'].keys():
            values = [result['metrics'][metric] for result in all_results.values()]
            unbalanced_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        metrics_summary = {
            'balanced_metrics': balanced_summary,
            'unbalanced_metrics': unbalanced_summary,
            'model_params': model_params,
            'train_params': train_params,
            'dataset_info': {
                'num_stimuli': len(stim_keys),
                'total_samples': sum(len(f) for f in all_features),
                'positive_ratio': float(np.mean([np.mean(l) for l in all_labels]))
            }
        }

        # Save results
        save_results(output_dir, all_results, cv_results, metrics_summary)

        # Print final results
        print("\nBalanced Evaluation Results (Training Phase):")
        print("-" * 50)
        for metric, values in balanced_summary.items():
            print(f"{metric}: {values['mean']:.3f} (±{values['std']:.3f})")

        print("\nUnbalanced Evaluation Results (Full Stimulus):")
        print("-" * 50)
        for metric, values in unbalanced_summary.items():
            print(f"{metric}: {values['mean']:.3f} (±{values['std']:.3f})")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()