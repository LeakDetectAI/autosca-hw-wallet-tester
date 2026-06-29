import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def compute_top_k_accuracy(y_true, y_pred_scores, k=3):
    """Computes Top-K accuracy.
    y_true: 1D array of true class labels (integer)
    y_pred_scores: 2D array of class scores/probabilities
    """
    if len(y_pred_scores.shape) < 2 or y_pred_scores.shape[1] < k:
        k = max(1, y_pred_scores.shape[1])
        
    top_k_preds = np.argsort(y_pred_scores, axis=1)[:, -k:]
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(correct))

def evaluate_classification_performance(y_true, y_pred_scores, dataset_name, model_type, output_dir=None):
    """Computes and saves standard classification metrics.
    y_true: array-like, can be one-hot encoded (2D) or class labels (1D)
    y_pred_scores: 2D array of class scores/probabilities
    """
    if len(np.shape(y_true)) > 1 and np.shape(y_true)[1] > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = np.array(y_true, dtype=int)

    y_pred_labels = np.argmax(y_pred_scores, axis=1)

    accuracy = float(accuracy_score(y_true_labels, y_pred_labels))
    top_3_acc = float(compute_top_k_accuracy(y_true_labels, y_pred_scores, k=3))
    macro_f1 = float(f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0))
    precision = float(precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0))
    recall = float(recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0))
    
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    report_dict = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
    report_text = classification_report(y_true_labels, y_pred_labels, zero_division=0)

    metrics = {
        "dataset": dataset_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "top_3_accuracy": top_3_acc,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()
    }

    output_dir = output_dir or os.path.join("results", "behavioral")
    os.makedirs(output_dir, exist_ok=True)

    base_filename = f"{dataset_name.lower()}_{model_type.lower()}"
    json_path = os.path.join(output_dir, f"{base_filename}_metrics.json")
    report_path = os.path.join(output_dir, f"{base_filename}_report.txt")

    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(report_path, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("="*40 + "\n")
        f.write(report_text)

    print(f"Metrics saved to {json_path}")
    print(f"Classification report saved to {report_path}")

    return metrics
