import os
import random
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from .config import YOLO_MODEL_PATH, FEATURES, TRAIN_PARAMS


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiFeatureClassifier(nn.Module):
    def __init__(self, feature_classes, hidden_dim=512, num_layers=1, dropout=0.5, use_bn=True):
        super(MultiFeatureClassifier, self).__init__()
        yolo_model = YOLO(YOLO_MODEL_PATH)
        self.backbone = yolo_model.model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.fc_layers = nn.ModuleList()
        for feature in FEATURES:
            num_classes = len(feature_classes[feature])
            layers = []
            prev_dim = 1000
            for i in range(num_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.fc_layers.append(nn.Sequential(*layers))

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.view(features.size(0), -1)
        outputs = []
        for fc in self.fc_layers:
            outputs.append(fc(features))
        return outputs
    
    def unfreeze_backbone(self, num_layers=-1):
        backbone_layers = list(self.backbone.children())
        if num_layers == -1 or num_layers >= len(backbone_layers):
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for layer in backbone_layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def train(self, mode=True):
        self.training = mode
        for fc in self.fc_layers:
            fc.train(mode)
        has_unfrozen = any(p.requires_grad for p in self.backbone.parameters())
        if has_unfrozen:
            self.backbone.train(mode)
        else:
            self.backbone.eval()
        return self
    
    def eval(self):
        return self.train(False)

def build_model(feature_classes, hidden_dim=512, num_layers=1, dropout=0.5, use_bn=True):
    model = MultiFeatureClassifier(feature_classes, hidden_dim=hidden_dim, num_layers=num_layers,
                                    dropout=dropout, use_bn=use_bn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_loss_fn(label_smoothing=0.0):
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

def get_optimizer(model, lr=TRAIN_PARAMS["lr"], finetune_lr=None, weight_decay=TRAIN_PARAMS["weight_decay"],
                  optimizer_type="adamw", scheduler_type="cosine", scheduler_params=None, epochs=100):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr})
    if backbone_params and finetune_lr is not None:
        param_groups.append({"params": backbone_params, "lr": finetune_lr})
    elif backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr})
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    
    if scheduler_params is None:
        scheduler_params = {}
    
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=scheduler_params.get("eta_min", lr * 0.01)
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 5)
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1)
        )
    else:
        scheduler = None
    
    return optimizer, scheduler

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def save_model(model, epoch, loss, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth"))

def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["epoch"]
