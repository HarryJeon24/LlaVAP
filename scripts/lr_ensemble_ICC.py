import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             balanced_accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from typing import Dict, Tuple, List

# Feature Extraction Function
def extract_advanced_features(data: np.ndarray, window_sizes=[5, 10, 20]) -> np.ndarray:
    if data.dtype == bool:
        data = data.astype(float)
    features_list = []
    for window_size in window_sizes:
        rolling_mean = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        rolling_std = pd.Series(data).rolling(window_size).std().fillna(0).values
        gradient = np.gradient(data)
        gradient_mean = np.convolve(gradient, np.ones(window_size) / window_size, mode='same')
        peaks, _ = find_peaks(data, distance=window_size)
        valleys, _ = find_peaks(-data, distance=window_size)
        peak_indicator = np.zeros_like(data)
        valley_indicator = np.zeros_like(data)
        peak_indicator[peaks] = 1
        valley_indicator[valleys] = 1
        n_fft = min(window_size, len(data))
        fft_features = np.abs(fft(data))[:n_fft]
        fft_features = np.pad(fft_features, (0, len(data) - len(fft_features)), 'constant')
        skewness = pd.Series(data).rolling(window_size).apply(skew, raw=True).fillna(0).values
        kurt = pd.Series(data).rolling(window_size).apply(kurtosis, raw=True).fillna(0).values
        all_features = [rolling_mean, rolling_std, gradient, gradient_mean,
                        peak_indicator, valley_indicator, skewness, kurt, fft_features]
        features_list.extend([np.array(f).reshape(len(data)) for f in all_features])
    return np.column_stack(features_list)

# Create Feature Matrix and Labels
def create_enhanced_features(llama_results: Dict, vap_results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, file_mapping = [], [], []
    vap_data = vap_results['per_stimulus_results']
    for stim_key in llama_results.keys():
        if stim_key not in vap_data:
            continue
        llama_data = llama_results[stim_key]
        vap_stimulus_data = vap_data[stim_key]
        llama_probs = np.array(llama_data['probabilities'])
        llama_preds = (llama_probs > 0.5).astype(float)
        vap_probs = np.array(vap_stimulus_data['probabilities'])
        vap_preds = np.array(vap_stimulus_data['predictions']).astype(float)
        ground_truth = np.array(llama_data['ground_truth'])
        min_len = min(len(llama_probs), len(vap_preds), len(ground_truth))
        try:
            llama_prob_features = extract_advanced_features(llama_probs[:min_len])
            vap_prob_features = extract_advanced_features(vap_probs[:min_len])
            features = np.column_stack([llama_probs[:min_len], llama_preds[:min_len],
                                         vap_probs[:min_len], vap_preds[:min_len],
                                         llama_prob_features, vap_prob_features])
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            X.append(features)
            y.append(ground_truth[:min_len])
            file_mapping.extend([stim_key] * min_len)
        except Exception as e:
            print(f"Error processing {stim_key}: {str(e)}")
            continue
    if not X:
        raise ValueError("No features were successfully extracted")
    return np.vstack(X), np.concatenate(y), np.array(file_mapping)

# Evaluation Function
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    })
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics.update({
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc(fpr, tpr)},
        'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': auc(recall, precision)}
    })
    return metrics

# Training and Cross-Validation
def train_logistic_regression_kfold(X: np.ndarray, y: np.ndarray, file_mapping: np.ndarray, output_dir: Path, n_splits: int = 4) -> Dict:
    unique_files = np.unique(file_mapping)
    kf = KFold(n_splits=min(n_splits, len(unique_files)), shuffle=True, random_state=42)
    scaler = StandardScaler()
    results = {'fold_metrics': []}
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_files)):
        train_files = unique_files[train_idx]
        test_files = unique_files[test_idx]
        train_mask = np.isin(file_mapping, train_files)
        test_mask = np.isin(file_mapping, test_files)
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        results['fold_metrics'].append(metrics)
        ax_roc.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'], label=f'Fold {fold} (AUC = {metrics["roc_curve"]["auc"]:.2f})')
        ax_pr.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'], label=f'Fold {fold} (AUC = {metrics["pr_curve"]["auc"]:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_title("ROC Curves")
    ax_pr.set_title("Precision-Recall Curves")
    fig_roc.savefig(output_dir / "roc_curves.png")
    fig_pr.savefig(output_dir / "pr_curves.png")
    return results

# Main Function
def main():
    llama_results_path = Path( "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_llm_ICC_output/3b/0.6_no_flip/llama_realtime_results.json")
    vap_results_path = Path( "C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/baseline_VAP_ICC_output/0.999_no_flip_best/full_results.json")
    output_dir = Path("C:/Users/Harry/PycharmProjects/LLA-VAP/prediction_results/lr_ensemble_ICC_output")
    output_dir.mkdir(exist_ok=True)
    with open(llama_results_path) as f:
        llama_results = json.load(f)
    with open(vap_results_path) as f:
        vap_results = json.load(f)
    X, y, file_mapping = create_enhanced_features(llama_results, vap_results)
    results = train_logistic_regression_kfold(X, y, file_mapping, output_dir)
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
