import os
import sys
import shutil
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import FEATURES, TRAIN_PARAMS, LOG_DIR, CHECKPOINT_DIR, RESULT_DIR, SEED
from src.data_loader import get_data_loaders, save_class_mappings
from src.model import build_model, get_loss_fn, get_optimizer, save_model, load_model, count_parameters, set_seed
from src.evaluate import evaluate_model, compute_metrics, print_metrics, save_metrics
from src.visualize import plot_loss_curves, plot_confusion_matrix, plot_metrics_bar, plot_feature_accuracy

BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "best_model")

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, class_weights=None, use_mixup=False, mixup_alpha=0.2, grad_clip=0.0):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        targets = targets.to(device)
        
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, alpha=mixup_alpha)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = 0.0
        for i, feature in enumerate(FEATURES):
            if class_weights and class_weights[feature] is not None:
                weights = class_weights[feature].to(device)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
                if use_mixup:
                    loss += mixup_criterion(ce_loss, outputs[i], targets_a[:, i], targets_b[:, i], lam)
                else:
                    loss += ce_loss(outputs[i], targets[:, i])
            else:
                if use_mixup:
                    loss += mixup_criterion(loss_fn, outputs[i], targets_a[:, i], targets_b[:, i], lam)
                else:
                    loss += loss_fn(outputs[i], targets[:, i])
        
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    
    total_loss /= len(train_loader.dataset)
    return total_loss


class WarmupScheduler:
    def __init__(self, optimizer, base_scheduler, warmup_epochs=5, warmup_lr_start=1e-6):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.current_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
    
    def step(self, val_loss=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.warmup_lr_start + (self.base_lrs[i] - self.warmup_lr_start) * warmup_factor
        else:
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.base_scheduler.step(val_loss)
                else:
                    self.base_scheduler.step()
    
    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

def parse_args():
    parser = argparse.ArgumentParser(description="茶花特征分类模型训练")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数 (默认使用配置文件中的值)")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小 (默认使用配置文件中的值)")
    parser.add_argument("--lr", type=float, default=None, help="学习率 (默认使用配置文件中的值)")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心值 (默认使用配置文件中的值)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"随机种子 (默认: {SEED})，设置为 -1 表示完全随机")
    parser.add_argument("--use_merge", action="store_true", help="启用标签归并，将相似类别合并")
    parser.add_argument("--finetune", action="store_true", help="启用微调，解冻部分 backbone 进行训练")
    parser.add_argument("--finetune_lr", type=float, default=0.0001, help="backbone 微调的学习率 (默认: 0.0001)")
    parser.add_argument("--unfreeze_layers", type=int, default=3, help="解冻 backbone 最后 N 层 (默认: 3，-1 表示全部解冻)")
    parser.add_argument("--weight_decay", type=float, default=None, help="L2 正则化权重衰减系数 (默认使用配置文件值 0.0001)")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑系数 (默认: 0.0，推荐 0.1)")
    parser.add_argument("--aug_level", type=str, default="medium", choices=["light", "medium", "strong", "randaugment"], help="数据增强强度 (默认: medium)")
    parser.add_argument("--min_samples", type=int, default=1, help="每个标签类别的最少样本数，少于该数目的类别将被过滤 (默认: 1，即不过滤)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="分类头隐藏层维度 (默认: 512)")
    parser.add_argument("--num_layers", type=int, default=1, help="分类头隐藏层数 (默认: 1)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout 比率 (默认: 0.5)")
    parser.add_argument("--no_bn", action="store_true", help="禁用 BatchNorm")
    parser.add_argument("--use_mixup", action="store_true", help="启用 MixUp 数据增强")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="MixUp 的 Beta 分布 alpha 参数 (默认: 0.2)")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="优化器类型 (默认: adamw)")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "step", "none"], help="学习率调度器类型 (默认: cosine)")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="学习率预热轮数 (默认: 0，即不预热)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="梯度裁剪阈值 (默认: 0.0，即不裁剪)")
    parser.add_argument("--use_tta", action="store_true", help="评估时启用测试时增强(TTA)")
    parser.add_argument("--use_swa", action="store_true", help="启用随机权重平均(SWA)")
    parser.add_argument("--swa_start", type=int, default=20, help="SWA 开始的 epoch (默认: 20)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.epochs is not None:
        TRAIN_PARAMS["epochs"] = args.epochs
    if args.batch_size is not None:
        TRAIN_PARAMS["batch_size"] = args.batch_size
    if args.lr is not None:
        TRAIN_PARAMS["lr"] = args.lr
    if args.patience is not None:
        TRAIN_PARAMS["patience"] = args.patience
    if args.weight_decay is not None:
        TRAIN_PARAMS["weight_decay"] = args.weight_decay

    seed = None if args.seed == -1 else args.seed

    print("=" * 60)
    print("茶花特征分类模型训练")
    print("=" * 60)

    set_seed(seed)
    if seed is None:
        print(f"\n  随机种子: 完全随机模式")
    else:
        print(f"\n  随机种子: {seed} (结果可复现)")

    print("\n1. 加载数据...")
    train_loader, val_loader, feature_classes, feature_labels, class_weights = get_data_loaders(
        seed=seed, use_merge=args.use_merge, aug_level=args.aug_level,
        min_samples_per_class=args.min_samples)
    save_class_mappings(feature_classes, feature_labels, RESULT_DIR)

    print("\n特征类别统计:")
    for feature in FEATURES:
        print(f"  {feature}: {len(feature_classes[feature])} 个类别")

    if args.use_merge:
        print(f"\n  已启用标签归并")
    
    print("\n2. 构建模型...")
    model, device = build_model(feature_classes, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                 dropout=args.dropout, use_bn=not args.no_bn)
    print(f"  使用设备: {device}")
    
    if args.finetune:
        model.unfreeze_backbone(args.unfreeze_layers)
    
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {frozen_params:,}")
    
    print(f"\n  分类头配置:")
    print(f"    隐藏层维度: {args.hidden_dim}")
    print(f"    隐藏层数: {args.num_layers}")
    print(f"    Dropout: {args.dropout}")
    print(f"    BatchNorm: {'禁用' if args.no_bn else '启用'}")
    
    if args.use_mixup:
        print(f"\n  MixUp: 已启用 (alpha={args.mixup_alpha})")
    
    if args.finetune:
        print(f"\n  微调模式: 已启用")
        print(f"  解冻层数: {'全部' if args.unfreeze_layers == -1 else args.unfreeze_layers}")
        print(f"  分类头学习率: {TRAIN_PARAMS['lr']}")
        print(f"  Backbone 学习率: {args.finetune_lr}")
    
    print(f"\n  优化器: {args.optimizer}")
    print(f"  学习率调度: {args.scheduler}")
    if args.warmup_epochs > 0:
        print(f"  学习率预热: {args.warmup_epochs} 轮")
    if args.grad_clip > 0:
        print(f"  梯度裁剪: {args.grad_clip}")
    
    loss_fn = get_loss_fn(label_smoothing=args.label_smoothing)
    scheduler_type = None if args.scheduler == "none" else args.scheduler
    optimizer, base_scheduler = get_optimizer(
        model, lr=TRAIN_PARAMS["lr"],
        finetune_lr=args.finetune_lr if args.finetune else None,
        weight_decay=TRAIN_PARAMS["weight_decay"],
        optimizer_type=args.optimizer,
        scheduler_type=scheduler_type,
        epochs=TRAIN_PARAMS["epochs"]
    )
    
    if args.warmup_epochs > 0 and base_scheduler is not None:
        scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_epochs=args.warmup_epochs)
    else:
        scheduler = base_scheduler
    
    swa_model = None
    swa_start = args.swa_start
    if args.use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        print(f"\n  SWA: 已启用 (从 epoch {swa_start} 开始)")
    
    print("\n3. 开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_checkpoint_path = None
    
    for epoch in range(1, TRAIN_PARAMS["epochs"] + 1):
        print(f"\nEpoch [{epoch}/{TRAIN_PARAMS['epochs']}]")
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device,
                                     class_weights=class_weights,
                                     use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                                     grad_clip=args.grad_clip)
        train_losses.append(train_loss)
        
        print(f"  训练损失: {train_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  当前学习率: {current_lr:.6f}")
        
        all_preds, all_targets, val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"  验证损失: {val_loss:.4f}")
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) or \
               isinstance(scheduler, WarmupScheduler):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if args.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
        
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
    
    if args.use_swa and swa_model is not None:
        print("\n  更新 SWA 模型的 BatchNorm 统计量...")
        from torch.optim.swa_utils import update_bn
        update_bn(train_loader, swa_model, device=device)
    
    print("\n4. 评估模型...")
    eval_model = swa_model if (args.use_swa and swa_model is not None) else model
    all_preds, all_targets, final_val_loss = evaluate_model(eval_model, val_loader, device)
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