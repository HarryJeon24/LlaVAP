
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Tuple, List, Any
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray, window_size: int = 75) -> Dict:
    """Comprehensive evaluation including window-based metrics"""
    metrics = {}

    # Create windows around ground truth positives
    true_windows = np.zeros_like(y_true)
    for i in np.where(y_true == 1)[0]:
        start = max(0, i - window_size)
        end = min(len(y_true), i + window_size + 1)
        true_windows[start:end] = 1

    # Calculate basic metrics
    tn, fp, fn, tp = confusion_matrix(true_windows, y_pred).ravel()

    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'balanced_accuracy': balanced_accuracy_score(true_windows, y_pred),
        'precision': precision_score(true_windows, y_pred),
        'recall': recall_score(true_windows, y_pred),
        'f1': f1_score(true_windows, y_pred),
        'window_size': window_size
    })

    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(true_windows, y_prob)
    precision, recall, _ = precision_recall_curve(true_windows, y_prob)

    metrics.update({
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': auc(fpr, tpr)
        },
        'pr_curve': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': auc(recall, precision)
        }
    })

    return metrics


def extract_advanced_features(data: np.ndarray, window_sizes=[5, 10, 20]) -> np.ndarray:
    """Extract comprehensive temporal and frequency domain features"""
    # Convert boolean to float if needed
    if data.dtype == bool:
        data = data.astype(float)

    features_list = []

    for window_size in window_sizes:
        # Basic temporal features
        rolling_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        rolling_std = pd.Series(data).rolling(window_size).std().fillna(0).values

        # Gradient features
        gradient = np.gradient(data)
        gradient_mean = np.convolve(gradient, np.ones(window_size) / window_size, mode='same')

        # Peak detection
        peaks, _ = find_peaks(data, distance=window_size)
        valleys, _ = find_peaks(-data, distance=window_size)
        peak_indicator = np.zeros_like(data)
        valley_indicator = np.zeros_like(data)
        peak_indicator[peaks] = 1
        valley_indicator[valleys] = 1

        # Skip change point detection
        change_point_indicator = np.zeros_like(data)

        # Frequency domain features
        n_fft = min(window_size, len(data))
        fft_features = np.abs(fft(data))[:n_fft]
        # Pad FFT features to match data length
        fft_features = np.pad(fft_features, (0, len(data) - len(fft_features)), 'constant')

        # Statistical features
        skewness = pd.Series(data).rolling(window_size).apply(skew, raw=True).fillna(0).values
        kurt = pd.Series(data).rolling(window_size).apply(kurtosis, raw=True).fillna(0).values

        all_features = [
            rolling_mean,
            rolling_std,
            gradient,
            gradient_mean,
            peak_indicator,
            valley_indicator,
            change_point_indicator,
            skewness,
            kurt,
            fft_features
        ]

        features_list.extend([np.array(f).reshape(len(data)) for f in all_features])

    return np.column_stack(features_list)


def create_enhanced_features(llama_results: Dict, vap_results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create comprehensive feature set using both probabilities and predictions"""
    X = []
    y = []
    file_mapping = []

    vap_data = vap_results['per_stimulus_results']

    for stim_key in llama_results.keys():
        if stim_key not in vap_data:
            continue

        print(f"Processing stimulus: {stim_key}")
        llama_data = llama_results[stim_key]
        vap_stimulus_data = vap_data[stim_key]

        # Get all signals
        llama_probs = np.array(llama_data['probabilities'])
        llama_preds = (llama_probs > 0.5).astype(float)
        vap_probs = np.array(vap_stimulus_data['probabilities'])
        vap_preds = np.array(vap_stimulus_data['predictions']).astype(float)
        ground_truth = np.array(llama_data['ground_truth'])

        min_len = min(len(llama_probs), len(vap_preds), len(ground_truth))
        print(f"Sequence length: {min_len}")

        try:
            # Extract features for both probability signals
            llama_prob_features = extract_advanced_features(llama_probs[:min_len])
            vap_prob_features = extract_advanced_features(vap_probs[:min_len])

            # Combine all features
            features = np.column_stack([
                # Raw signals
                llama_probs[:min_len],
                llama_preds[:min_len],
                vap_probs[:min_len],
                vap_preds[:min_len],
                # Advanced features from probabilities
                llama_prob_features,
                vap_prob_features
            ])

            # Remove any remaining inf or nan values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

            X.append(features)
            y.append(ground_truth[:min_len])
            file_mapping.extend([stim_key] * min_len)

            print(f"Features shape for {stim_key}: {features.shape}")

        except Exception as e:
            import traceback
            print(f"Error processing {stim_key}: {str(e)}")
            print(traceback.format_exc())
            continue

    if not X:
        raise ValueError("No features were successfully extracted")

    return np.vstack(X), np.concatenate(y), np.array(file_mapping)

class TRPDataset(Dataset):
    def __init__(self, X, y, sequence_length=50):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.X) - self.sequence_length + 1)

    def __getitem__(self, idx):
        return (self.X[idx:idx + self.sequence_length],
                self.y[idx:idx + self.sequence_length])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return self.sigmoid(output)


class MLPModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class StackingModel(nn.Module):
    def __init__(self, num_models=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_models, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_lstm_model(X_train, y_train, X_val, y_val, device="cuda"):
    model = LSTMModel(X_train.shape[-1]).to(device)

    # Calculate class weights
    pos_weight = torch.FloatTensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create balanced sampler
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # Oversample positive class
    pos_indices = np.tile(pos_indices, len(neg_indices) // len(pos_indices))
    indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(indices)

    # Create datasets with balanced sampling
    train_dataset = TRPDataset(X_train[indices], y_train[indices])
    val_dataset = TRPDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    best_val_loss = float('inf')
    patience = 5
    patience_count = 0

    for epoch in range(50):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X).squeeze()
                val_loss += criterion(output, batch_y).item()

                probs = torch.sigmoid(output)
                preds = (probs > 0.3).float()  # Lower threshold due to imbalance

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Monitor balanced accuracy
        val_bacc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val Balanced Acc = {val_bacc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_model = model.state_dict()
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_model)
    return model


def train_mlp_model(X_train, y_train, X_val, y_val, device="cuda"):
    model = MLPModel(X_train.shape[1]).to(device)

    # Calculate class weights
    pos_weight = torch.FloatTensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Balance training data
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    np.random.shuffle(neg_indices)
    neg_indices = neg_indices[:len(pos_indices) * 3]  # 3:1 ratio
    indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(indices)

    X_train_balanced = X_train[indices]
    y_train_balanced = y_train[indices]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_balanced).to(device)
    y_train_tensor = torch.FloatTensor(y_train_balanced).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    best_val_loss = float('inf')
    patience = 5
    patience_count = 0

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor).squeeze()
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor).squeeze()
            val_loss = criterion(val_output, y_val_tensor).item()

            # Use lower threshold for predictions
            val_probs = torch.sigmoid(val_output)
            val_preds = (val_probs > 0.3).float()
            val_bacc = balanced_accuracy_score(y_val_tensor.cpu().numpy(),
                                               val_preds.cpu().numpy())

        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val Balanced Acc = {val_bacc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_model = model.state_dict()
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_model)
    return model


def evaluate_model(y_true, y_pred, y_prob, model_name: str,
                   output_dir: Path, fold: int):
    """Evaluate individual model performance"""
    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'],
             label=f'ROC (AUC = {metrics["roc_curve"]["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name.lower()}_fold{fold}_roc.png")
    plt.close()

    # Plot PR curve
    plt.figure(figsize=(8, 8))
    plt.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'],
             label=f'PR (AUC = {metrics["pr_curve"]["auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve - Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"{model_name.lower()}_fold{fold}_pr.png")
    plt.close()

    return metrics


def train_and_evaluate_fold(X_train, X_test, y_train, y_test,
                            fold: int, output_dir: Path):
    """Train and evaluate all models for one fold"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    base_predictions = {}

    # Train and evaluate XGBoost
    print(f"\nTraining XGBoost - Fold {fold}")
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'aucpr'],
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1)
    }

    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                          early_stopping_rounds=20,
                          evals=[(dtrain, 'train'), (dtest, 'test')],
                          verbose_eval=False)

    xgb_prob = xgb_model.predict(dtest)
    xgb_pred = (xgb_prob > 0.5).astype(int)
    results['xgboost'] = evaluate_model(y_test, xgb_pred, xgb_prob,
                                        'XGBoost', output_dir, fold)
    base_predictions['xgboost'] = xgb_prob

    # Train and evaluate LSTM
    print(f"Training LSTM - Fold {fold}")
    lstm_model = train_lstm_model(X_train_scaled, y_train,
                                  X_test_scaled, y_test, device)

    with torch.no_grad():
        lstm_prob = lstm_model(torch.FloatTensor(X_test_scaled).to(device))
        lstm_prob = lstm_prob.cpu().numpy().squeeze()
    lstm_pred = (lstm_prob > 0.5).astype(int)
    results['lstm'] = evaluate_model(y_test, lstm_pred, lstm_prob,
                                     'LSTM', output_dir, fold)
    base_predictions['lstm'] = lstm_prob

    # Train and evaluate MLP
    print(f"Training MLP - Fold {fold}")
    mlp_model = train_mlp_model(X_train_scaled, y_train,
                                X_test_scaled, y_test, device)

    with torch.no_grad():
        mlp_prob = mlp_model(torch.FloatTensor(X_test_scaled).to(device))
        mlp_prob = mlp_prob.cpu().numpy().squeeze()
    mlp_pred = (mlp_prob > 0.5).astype(int)
    results['mlp'] = evaluate_model(y_test, mlp_pred, mlp_prob,
                                    'MLP', output_dir, fold)
    base_predictions['mlp'] = mlp_prob

    # Train and evaluate stacking model
    print(f"Training Stacking Model - Fold {fold}")
    stacking_features = np.column_stack([xgb_prob, lstm_prob, mlp_prob])
    stacking_model = StackingModel().to(device)
    stacking_optimizer = optim.Adam(stacking_model.parameters())
    criterion = nn.BCELoss()

    X_stack_tensor = torch.FloatTensor(stacking_features).to(device)
    y_stack_tensor = torch.FloatTensor(y_test).to(device)

    for epoch in range(100):
        stacking_model.train()
        stacking_optimizer.zero_grad()
        output = stacking_model(X_stack_tensor).squeeze()
        loss = criterion(output, y_stack_tensor)
        loss.backward()
        stacking_optimizer.step()

    stacking_model.eval()
    with torch.no_grad():
        stack_prob = stacking_model(X_stack_tensor).cpu().numpy().squeeze()
    stack_pred = (stack_prob > 0.5).astype(int)
    results['stacking'] = evaluate_model(y_test, stack_pred, stack_prob,
                                         'Stacking', output_dir, fold)

    return results, base_predictions


def main():
    # Setup paths
    llama_results_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_ICC_output/3b/0.6_no_flip/llama_realtime_results.json")
    vap_results_path = Path(
        "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_ICC_output/0.999_no_flip_best/full_results.json")
    ensemble_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/stacking_ensemble_ICC_output")
    ensemble_dir.mkdir(exist_ok=True)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data and create features
    print("\nLoading result files...")
    with open(llama_results_path) as f:
        llama_results = json.load(f)
    with open(vap_results_path) as f:
        vap_results = json.load(f)

    print("\nCreating enhanced features...")
    X, y, file_mapping = create_enhanced_features(llama_results, vap_results)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of unique files: {len(np.unique(file_mapping))}")
    print(f"Class distribution: {np.bincount(y)}")

    # Prepare for k-fold cross validation
    unique_files = np.unique(file_mapping)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    all_results = {
        'fold_results': [],
        'model_metrics': {
            'xgboost': [], 'lstm': [], 'mlp': [], 'stacking': []
        }
    }

    # Train and evaluate models
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_files)):
        print(f"\nProcessing fold {fold + 1}/4")
        fold_dir = ensemble_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        train_files = unique_files[train_idx]
        test_files = unique_files[test_idx]

        train_mask = np.isin(file_mapping, train_files)
        test_mask = np.isin(file_mapping, test_files)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Train and evaluate all models for this fold
        fold_results, fold_predictions = train_and_evaluate_fold(
            X_train, X_test, y_train, y_test, fold, fold_dir
        )

        # Store results
        all_results['fold_results'].append({
            'fold': fold,
            'train_files': train_files.tolist(),
            'test_files': test_files.tolist(),
            'metrics': fold_results
        })

        for model_name, metrics in fold_results.items():
            all_results['model_metrics'][model_name].append(metrics)

    # Calculate and display average metrics
    print("\nAverage Metrics Across Folds:")
    for model_name in ['xgboost', 'lstm', 'mlp', 'stacking']:
        metrics = all_results['model_metrics'][model_name]
        avg_metrics = {
            'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in metrics]),
            'f1': np.mean([m['f1'] for m in metrics]),
            'precision': np.mean([m['precision'] for m in metrics]),
            'recall': np.mean([m['recall'] for m in metrics]),
            'auc_roc': np.mean([m['roc_curve']['auc'] for m in metrics])
        }
        std_metrics = {
            'balanced_accuracy': np.std([m['balanced_accuracy'] for m in metrics]),
            'f1': np.std([m['f1'] for m in metrics]),
            'precision': np.std([m['precision'] for m in metrics]),
            'recall': np.std([m['recall'] for m in metrics]),
            'auc_roc': np.std([m['roc_curve']['auc'] for m in metrics])
        }

        print(f"\n{model_name.upper()} Performance:")
        for metric in avg_metrics:
            print(f"{metric}: {avg_metrics[metric]:.3f} (±{std_metrics[metric]:.3f})")

    # Save detailed results
    final_results = {
        'fold_results': all_results['fold_results'],
        'average_metrics': {
            model_name: {
                'metrics': {
                    metric: {
                        'mean': float(np.mean([m[metric] for m in metrics])),
                        'std': float(np.std([m[metric] for m in metrics]))
                    }
                    for metric in ['balanced_accuracy', 'f1', 'precision', 'recall']
                },
                'auc_roc': {
                    'mean': float(np.mean([m['roc_curve']['auc'] for m in metrics])),
                    'std': float(np.std([m['roc_curve']['auc'] for m in metrics]))
                }
            }
            for model_name, metrics in all_results['model_metrics'].items()
        },
        'training_info': {
            'feature_matrix_shape': X.shape,
            'total_samples': len(y),
            'positive_samples': int(np.sum(y == 1)),
            'negative_samples': int(np.sum(y == 0)),
            'class_ratio': float(np.sum(y == 0) / np.sum(y == 1))
        }
    }

    # Save results
    with open(ensemble_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=4)

    # Create comparative plots
    plt.figure(figsize=(12, 6))
    models = ['XGBoost', 'LSTM', 'MLP', 'Stacking']
    metrics = ['balanced_accuracy', 'f1', 'precision', 'recall']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        data = []
        for model in models:
            model_metrics = all_results['model_metrics'][model.lower()]
            data.append([m[metric] for m in model_metrics])

        plt.boxplot(data, labels=models)
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.grid(True)
        plt.savefig(ensemble_dir / f"model_comparison_{metric}.png")
        plt.close()

    print(f"\nAll results saved to {ensemble_dir}")
    print("\nProcess completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error in execution: {str(e)}")
        print(traceback.format_exc())