import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .config import FEATURES, RESULT_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def plot_loss_curves(train_losses, val_losses, output_dir=RESULT_DIR):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="训练损失", color="blue")
    plt.plot(val_losses, label="验证损失", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练与验证损失曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(metrics, output_dir=RESULT_DIR):
    for feature in FEATURES:
        cm = metrics[feature]["confusion_matrix"]
        classes = metrics[feature]["classes"]
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title(f"{feature} - 混淆矩阵")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{feature}.png"), dpi=300)
        plt.close()

def plot_metrics_bar(metrics, output_dir=RESULT_DIR):
    features = FEATURES
    accs = [metrics[f]["accuracy"] for f in features]
    precisions = [metrics[f]["precision"] for f in features]
    recalls = [metrics[f]["recall"] for f in features]
    f1s = [metrics[f]["f1"] for f in features]

    x = np.arange(len(features))
    width = 0.2

    plt.figure(figsize=(14, 8))
    plt.bar(x - 1.5*width, accs, width, label="准确率", color="blue")
    plt.bar(x - 0.5*width, precisions, width, label="精确率", color="green")
    plt.bar(x + 0.5*width, recalls, width, label="召回率", color="orange")
    plt.bar(x + 1.5*width, f1s, width, label="F1值", color="red")

    plt.xlabel("特征")
    plt.ylabel("分数")
    plt.title("各特征分类指标对比")
    plt.xticks(x, features, rotation=30, ha="right")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=300)
    plt.close()

def plot_feature_accuracy(metrics, output_dir=RESULT_DIR):
    features = FEATURES
    accs = [metrics[f]["accuracy"] * 100 for f in features]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, accs, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1f}%", ha="center", va="bottom")

    plt.xlabel("特征")
    plt.ylabel("准确率 (%)")
    plt.title("各特征分类准确率")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_accuracy.png"), dpi=300)
    plt.close()
