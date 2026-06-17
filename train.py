import os
import sys
import shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import FEATURES, TRAIN_PARAMS, LOG_DIR, CHECKPOINT_DIR, RESULT_DIR, SEED
from src.data_loader import get_data_loaders, save_class_mappings
from src.model import build_model, get_loss_fn, get_optimizer, save_model, load_model
from src.evaluate import evaluate_model, compute_metrics, print_metrics, save_metrics
from src.visualize import plot_loss_curves, plot_confusion_matrix, plot_metrics_bar, plot_feature_accuracy

BEST_MODEL_DIR = r"D:\tea_flower\best_model"

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = 0.0
        for i in range(len(FEATURES)):
            loss += loss_fn(outputs[i], targets[:, i])
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    
    total_loss /= len(train_loader.dataset)
    return total_loss

def main():
    print("=" * 60)
    print("茶花特征分类模型训练")
    print("=" * 60)
    
    torch.manual_seed(SEED)
    
    print("\n1. 加载数据...")
    train_loader, val_loader, feature_classes, feature_labels = get_data_loaders()
    save_class_mappings(feature_classes, feature_labels, RESULT_DIR)
    
    print("\n特征类别统计:")
    for feature in FEATURES:
        print(f"  {feature}: {len(feature_classes[feature])} 个类别")
    
    print("\n2. 构建模型...")
    model, device = build_model(feature_classes)
    print(f"  使用设备: {device}")
    
    loss_fn = get_loss_fn()
    optimizer, scheduler = get_optimizer(model)
    
    print("\n3. 开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    
    for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{TRAIN_PARAMS['epochs']}]")
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        
        print(f"  训练损失: {train_loss:.4f}")
        
        all_preds, all_targets, val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"  验证损失: {val_loss:.4f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_model(model, epoch, val_loss, CHECKPOINT_DIR)
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            print(f"  保存最佳模型 (Epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_PARAMS["patience"]:
                print(f"  早停触发，训练终止于 Epoch {epoch}")
                break
    
    print("\n4. 评估模型...")
    all_preds, all_targets, final_val_loss = evaluate_model(model, val_loader, device)
    metrics = compute_metrics(all_preds, all_targets, feature_classes)
    print_metrics(metrics)
    save_metrics(metrics, RESULT_DIR)
    
    print("\n5. 生成可视化结果...")
    plot_loss_curves(train_losses, val_losses, RESULT_DIR)
    plot_confusion_matrix(metrics, RESULT_DIR)
    plot_metrics_bar(metrics, RESULT_DIR)
    plot_feature_accuracy(metrics, RESULT_DIR)
    
    # 保存最佳模型到指定目录
    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        best_model_dest = os.path.join(BEST_MODEL_DIR, "best_model.pth")
        shutil.copy2(best_checkpoint_path, best_model_dest)
        print(f"\n最佳模型已保存至: {best_model_dest}")
    
    print(f"\n训练完成！结果已保存至 {RESULT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()