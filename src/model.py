import os
import torch
import torch.nn as nn
from ultralytics import YOLO
from .config import YOLO_MODEL_PATH, FEATURES, TRAIN_PARAMS

class MultiFeatureClassifier(nn.Module):
    def __init__(self, feature_classes):
        super(MultiFeatureClassifier, self).__init__()
        # 加载YOLOv8分类模型并提取backbone
        yolo_model = YOLO(YOLO_MODEL_PATH)
        self.backbone = yolo_model.model.model  # 获取实际的神经网络模块
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.fc_layers = nn.ModuleList()
        for feature in FEATURES:
            num_classes = len(feature_classes[feature])
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(1000, 512),  # YOLOv8n-cls backbone输出1000维特征
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes),
                )
            )

    def forward(self, x):
        # 通过backbone提取特征
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.view(features.size(0), -1)
        outputs = []
        for fc in self.fc_layers:
            outputs.append(fc(features))
        return outputs
    
    def train(self, mode=True):
        # 重写train方法，只对fc_layers设置训练模式
        # backbone保持eval模式（冻结参数）
        self.training = mode
        for fc in self.fc_layers:
            fc.train(mode)
        self.backbone.eval()  # backbone始终保持eval模式
        return self
    
    def eval(self):
        return self.train(False)

def build_model(feature_classes):
    model = MultiFeatureClassifier(feature_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

def get_loss_fn():
    return nn.CrossEntropyLoss()

def get_optimizer(model, lr=TRAIN_PARAMS["lr"], weight_decay=TRAIN_PARAMS["weight_decay"]):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return optimizer, scheduler

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