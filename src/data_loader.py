import os
import json
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, FEATURES, SEED, TRAIN_PARAMS

class TeaFlowerDataset(Dataset):
    def __init__(self, labels_dir, images_dir, transform=None):
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transform = transform
        self.data = []
        self.feature_labels = {feature: {} for feature in FEATURES}
        self.feature_classes = {feature: [] for feature in FEATURES}
        self._load_data()
        self._build_mappings()

    def _load_data(self):
        json_files = glob.glob(os.path.join(self.labels_dir, "*.json"))
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    label = json.load(f)
                basename = os.path.splitext(os.path.basename(json_file))[0]
                img_path = os.path.join(self.images_dir, basename + ".jpg")
                if os.path.exists(img_path):
                    self.data.append({"img_path": img_path, "label": label})
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    def _build_mappings(self):
        for item in self.data:
            label = item["label"]
            for feature in FEATURES:
                value = label.get(feature, "")
                if value not in self.feature_labels[feature]:
                    self.feature_labels[feature][value] = len(self.feature_labels[feature])
        for feature in FEATURES:
            self.feature_classes[feature] = sorted(self.feature_labels[feature].keys(), 
                                                   key=lambda x: self.feature_labels[feature][x])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = item["label"]
        targets = []
        for feature in FEATURES:
            value = label.get(feature, "")
            targets.append(self.feature_labels[feature].get(value, 0))
        return img, torch.tensor(targets, dtype=torch.long)

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def get_data_loaders():
    train_transform, val_transform = get_transforms(TRAIN_PARAMS["img_size"])
    train_dataset = TeaFlowerDataset(TRAIN_LABELS_DIR, IMAGES_DIR, train_transform)
    val_dataset = TeaFlowerDataset(VAL_LABELS_DIR, IMAGES_DIR, val_transform)

    torch.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=TRAIN_PARAMS["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.feature_classes, train_dataset.feature_labels

def save_class_mappings(feature_classes, feature_labels, output_dir):
    mapping = {
        "feature_classes": feature_classes,
        "feature_labels": feature_labels,
    }
    with open(os.path.join(output_dir, "class_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)