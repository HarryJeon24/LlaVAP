# lstm_ensemble_CCPE.py
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
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from collections import defaultdict
from tqdm import tqdm
import logging


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

        # BCE with positive weights
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )

        # Focal loss component
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
        self.input_size = 16  # 2 base + 6 engineered features

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
            ) for _ in range(4)  # 4 attention heads
        ])

        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 8, hidden_size)  # *8 for 4 attention heads
        self.ln1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)

        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        batch_size = x.size(0)

        # Feature extraction
        features = self.feature_extraction(x)

        # LSTM
        lstm_out, _ = self.lstm(features)

        # Multi-head attention
        attention_outputs = []
        for attention_head in self.attention_heads:
            attention_weights = attention_head(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            attended = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
            attention_outputs.append(attended.squeeze(1))

        # Concatenate attention outputs
        attended = torch.cat(attention_outputs, dim=1)

        # Fully connected layers with residual connections
        out = self.fc1(attended)
        out = self.ln1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = torch.relu(out)
        out = self.dropout(out)

        # Output logits (no sigmoid)
        out = self.fc3(out)

        return out


class TurnPredictionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 100, balance_data: bool = True):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.balance_data = balance_data
        self.features, self.labels = self.prepare_data(features, labels)
        self.length = len(self.features)  # Store the length after preparing data

    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and augment features, create sequences with optional balancing."""
        # Get indices of positive and negative samples
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]

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

        sequences = []
        sequence_labels = []

        # Process positive samples
        for idx in positive_indices:
            if idx >= self.sequence_length - 1:
                seq = combined_features[idx - self.sequence_length + 1:idx + 1]
                if len(seq) == self.sequence_length:
                    sequences.append(seq)
                    sequence_labels.append(1)

        # Process negative samples
        if self.balance_data:
            neg_samples_needed = len(sequences)  # Balance dataset
            sampled_neg_indices = np.random.choice(
                negative_indices[negative_indices >= self.sequence_length - 1],
                size=neg_samples_needed,
                replace=False
            )
        else:
            sampled_neg_indices = negative_indices[negative_indices >= self.sequence_length - 1]

        for idx in sampled_neg_indices:
            seq = combined_features[idx - self.sequence_length + 1:idx + 1]
            if len(seq) == self.sequence_length:
                sequences.append(seq)
                sequence_labels.append(0)

        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(sequences))
        sequences = np.array(sequences, dtype=np.float32)[shuffle_idx]
        sequence_labels = np.array(sequence_labels, dtype=np.float32)[shuffle_idx].reshape(-1, 1)

        return sequences, sequence_labels

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Return a single sequence and its label."""
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )


def analyze_predictions(true_labels, predictions, probabilities, loss=None):
    """Analyze prediction patterns and distribution."""
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(true_labels, predictions),
        'sensitivity': recall_score(true_labels, predictions, zero_division=0),
        'specificity': recall_score(true_labels, predictions, pos_label=0, zero_division=0),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'pos_ratio': np.mean(predictions),
        'true_pos_ratio': np.mean(true_labels)
    }

    # Only add loss if it's provided
    if loss is not None:
        metrics['loss'] = loss

    # Analyze probability distribution
    prob_stats = {
        'prob_mean': np.mean(probabilities),
        'prob_std': np.std(probabilities),
        'prob_min': np.min(probabilities),
        'prob_max': np.max(probabilities),
        'prob_median': np.median(probabilities)
    }

    return metrics, prob_stats


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

    # Plot positive prediction ratio
    ax4.plot(history['train_pos_ratio'], label='Train Pos Ratio')
    ax4.plot(history['val_pos_ratio'], label='Val Pos Ratio')
    ax4.plot(history['true_pos_ratio'], label='True Pos Ratio', linestyle='--')
    ax4.set_title('Positive Prediction Ratio')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Ratio')
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
    best_model_state = None

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

        # Calculate metrics
        train_metrics, train_prob_stats = analyze_predictions(
            train_outputs['labels'],
            train_outputs['preds'],
            train_outputs['probs'],
            np.mean(train_outputs['loss'])  # Add loss parameter
        )

        val_metrics, val_prob_stats = analyze_predictions(
            val_outputs['labels'],
            val_outputs['preds'],
            val_outputs['probs'],
            np.mean(val_outputs['loss'])  # Add loss parameter
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
        history['train_pos_ratio'].append(train_metrics['pos_ratio'])
        history['val_pos_ratio'].append(val_metrics['pos_ratio'])
        history['true_pos_ratio'].append(train_metrics['true_pos_ratio'])

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

def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def evaluate_model(model, data_loader, device, criterion=None):
    """Evaluate model on given dataset."""
    model.eval()
    outputs = defaultdict(list)
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)

            model_outputs = model(features)
            if criterion is not None:
                loss = criterion(model_outputs, labels)
                total_loss += loss.item()
                n_batches += 1

            probs = torch.sigmoid(model_outputs)
            preds = (probs > 0.5).float()

            outputs['preds'].extend(preds.cpu().numpy())
            outputs['probs'].extend(probs.cpu().numpy())
            outputs['labels'].extend(labels.cpu().numpy())

    # Calculate average loss if criterion was provided
    avg_loss = total_loss / n_batches if n_batches > 0 else None

    metrics, prob_stats = analyze_predictions(
        outputs['labels'],
        outputs['preds'],
        outputs['probs'],
        avg_loss
    )

    return metrics, prob_stats

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # Set paths
        vap_results_path = Path(
            "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_CCPE_output/probs")
        llama_results_path = Path(
            "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_CCPE_output/3b/0.4_no_flip_new_prompt")
        output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lstm_ensemble_results_CCPE")
        output_dir.mkdir(exist_ok=True)

        # Load and prepare data
        logger.info("Loading and preparing data...")
        vap_files = list(vap_results_path.glob('*_probabilities.csv'))

        all_features = []
        all_labels = []
        file_names = []

        for vap_file in tqdm(vap_files, desc="Processing files"):
            base_name = vap_file.stem.replace('_probabilities', '')
            llama_file = llama_results_path / f"{base_name}_probabilities.csv"

            if llama_file.exists():
                vap_data = pd.read_csv(vap_file)
                llama_data = pd.read_csv(llama_file)

                if len(vap_data) == len(llama_data):
                    features = np.column_stack([
                        vap_data['probability'].values,
                        llama_data['probability'].values
                    ])
                    labels = np.array(vap_data['ground_truth'].values)

                    all_features.append(features)
                    all_labels.append(labels)
                    file_names.append(base_name)

        logger.info(f"Successfully loaded {len(file_names)} files")

        # Model parameters
        params = {
            'sequence_length': 100,
            'hidden_size': 128,
            'num_layers': 2,
            'batch_size': 32,
            'num_epochs': 50
        }

        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(all_features)):
            logger.info(f"\nProcessing fold {fold + 1}/5")

            # Prepare train/val data
            train_features = [all_features[i] for i in train_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_features = [all_features[i] for i in val_idx]
            val_labels = [all_labels[i] for i in val_idx]

            # Create datasets
            train_dataset = TurnPredictionDataset(
                np.vstack(train_features),
                np.concatenate(train_labels),
                params['sequence_length'],
                balance_data=True
            )

            # Create both balanced and imbalanced validation datasets
            val_dataset_balanced = TurnPredictionDataset(
                np.vstack(val_features),
                np.concatenate(val_labels),
                params['sequence_length'],
                balance_data=True
            )

            val_dataset_imbalanced = TurnPredictionDataset(
                np.vstack(val_features),
                np.concatenate(val_labels),
                params['sequence_length'],
                balance_data=False
            )

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            val_loader_balanced = DataLoader(
                val_dataset_balanced,
                batch_size=params['batch_size'],
                num_workers=0,
                pin_memory=True
            )

            val_loader_imbalanced = DataLoader(
                val_dataset_imbalanced,
                batch_size=params['batch_size'],
                num_workers=0,
                pin_memory=True
            )

            # Initialize and train model
            model = LSTMEnsemble(
                sequence_length=params['sequence_length'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers']
            ).to(device)

            best_model_state, history = train_model(
                model,
                train_loader,
                val_loader_balanced,  # Use balanced validation during training
                device,
                params['num_epochs']
            )

            # Plot training progress
            plot_training_progress(
                history,
                output_dir / f"fold{fold + 1}_training_progress.png"
            )

            # Save model
            torch.save({
                'model_state_dict': best_model_state,
                'history': history,
                'params': params
            }, output_dir / f"fold{fold + 1}_model.pt")

            # Load best model for final evaluation
            model.load_state_dict(best_model_state)

            # Evaluate on both balanced and imbalanced validation sets
            balanced_metrics, balanced_prob_stats = evaluate_model(
                model, val_loader_balanced, device
            )

            imbalanced_metrics, imbalanced_prob_stats = evaluate_model(
                model, val_loader_imbalanced, device
            )

            # Store results
            fold_results = {
                'fold': fold + 1,
                'history': history,
                'final_balanced_metrics': balanced_metrics,
                'final_balanced_prob_stats': balanced_prob_stats,
                'final_imbalanced_metrics': imbalanced_metrics,
                'final_imbalanced_prob_stats': imbalanced_prob_stats,
                'train_files': [file_names[i] for i in train_idx],
                'val_files': [file_names[i] for i in val_idx]
            }

            cv_results.append(fold_results)

            # Print final results for this fold
            logger.info(f"\nFold {fold + 1} Final Results:")
            logger.info("\nBalanced Validation Metrics:")
            for k, v in balanced_metrics.items():
                if v is not None:  # Only print if value exists
                    logger.info(f"{k}: {v:.4f}")

            logger.info("\nImbalanced Validation Metrics:")
            for k, v in imbalanced_metrics.items():
                if v is not None:  # Only print if value exists
                    logger.info(f"{k}: {v:.4f}")

        # Calculate and log average metrics across folds
        # Calculate and log average metrics across folds
        avg_balanced_metrics = defaultdict(list)
        avg_imbalanced_metrics = defaultdict(list)

        for result in cv_results:
            for k, v in result['final_balanced_metrics'].items():
                if v is not None:  # Only include non-None values
                    avg_balanced_metrics[k].append(v)
            for k, v in result['final_imbalanced_metrics'].items():
                if v is not None:  # Only include non-None values
                    avg_imbalanced_metrics[k].append(v)

        logger.info("\nAverage Metrics Across Folds:")
        logger.info("\nBalanced Validation:")
        for k, v in avg_balanced_metrics.items():
            if v:  # Only print if list is not empty
                logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

        logger.info("\nImbalanced Validation:")
        for k, v in avg_imbalanced_metrics.items():
            if v:  # Only print if list is not empty
                logger.info(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

        # Save final results
        cv_results = convert_to_serializable(cv_results)
        with open(output_dir / "cv_results.json", "w") as f:
            json.dump(cv_results, f, indent=4)

        logger.info("Training and evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error in execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()