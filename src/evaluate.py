import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from .config import FEATURES

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = {feature: [] for feature in FEATURES}
    all_targets = {feature: [] for feature in FEATURES}
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            
            for i, feature in enumerate(FEATURES):
                loss = loss_fn(outputs[i], targets[:, i])
                total_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs[i], dim=1)
                all_preds[feature].extend(preds.cpu().numpy())
                all_targets[feature].extend(targets[:, i].cpu().numpy())
    
    total_loss /= len(val_loader.dataset)
    return all_preds, all_targets, total_loss

def compute_metrics(all_preds, all_targets, feature_classes):
    metrics = {}
    for feature in FEATURES:
        preds = np.array(all_preds[feature])
        targets = np.array(all_targets[feature])
        classes = feature_classes[feature]
        num_classes = len(classes)
        labels = list(range(num_classes))

        acc = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, labels=labels, average="weighted", zero_division=0)
        recall = recall_score(targets, preds, labels=labels, average="weighted", zero_division=0)
        f1 = f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)
        cm = confusion_matrix(targets, preds, labels=labels)
        
        metrics[feature] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "classes": classes,
        }
    return metrics

def print_metrics(metrics):
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    for feature in FEATURES:
        print(f"\n【{feature}】")
        print(f"  准确率: {metrics[feature]['accuracy']:.4f}")
        print(f"  精确率: {metrics[feature]['precision']:.4f}")
        print(f"  召回率: {metrics[feature]['recall']:.4f}")
        print(f"  F1值: {metrics[feature]['f1']:.4f}")
    print("=" * 60)

def save_metrics(metrics, output_dir):
    import json
    save_metrics = {}
    for feature in FEATURES:
        save_metrics[feature] = {
            "accuracy": float(metrics[feature]["accuracy"]),
            "precision": float(metrics[feature]["precision"]),
            "recall": float(metrics[feature]["recall"]),
            "f1": float(metrics[feature]["f1"]),
            "confusion_matrix": metrics[feature]["confusion_matrix"].tolist(),
            "classes": metrics[feature]["classes"],
        }
    with open(f"{output_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, ensure_ascii=False, indent=2)