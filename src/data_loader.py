import os
import json
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, FEATURES, TRAIN_PARAMS
from .label_merge import merge_label

class TeaFlowerDataset(Dataset):
    def __init__(self, labels_dir, images_dir, transform=None, feature_labels=None, use_merge=False):
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.transform = transform
        self.use_merge = use_merge
        self.data = []
        self.feature_labels = {feature: {} for feature in FEATURES}
        self.feature_classes = {feature: [] for feature in FEATURES}
        self.original_to_merged = {feature: {} for feature in FEATURES}
        self._load_data()
        if feature_labels is not None:
            self.feature_labels = feature_labels
            self._build_classes_from_labels()
        else:
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
                if value and value.strip():
                    if self.use_merge:
                        merged = merge_label(feature, value)
                        if merged:
                            self.original_to_merged[feature][value] = merged
                            if merged not in self.feature_labels[feature]:
                                self.feature_labels[feature][merged] = len(self.feature_labels[feature])
                    else:
                        if value not in self.feature_labels[feature]:
                            self.feature_labels[feature][value] = len(self.feature_labels[feature])
        for feature in FEATURES:
            self.feature_classes[feature] = sorted(self.feature_labels[feature].keys(),
                                                   key=lambda x: self.feature_labels[feature][x])

    def _build_classes_from_labels(self):
        for feature in FEATURES:
            self.feature_classes[feature] = sorted(self.feature_labels[feature].keys(),
                                                   key=lambda x: self.feature_labels[feature][x])
        valid_data = []
        for item in self.data:
            label = item["label"]
            valid = True
            for feature in FEATURES:
                value = label.get(feature, "")
                if value:
                    if self.use_merge:
                        value = merge_label(feature, value)
                    if value and value not in self.feature_labels[feature]:
                        valid = False
                        break
            if valid:
                valid_data.append(item)
        skipped = len(self.data) - len(valid_data)
        if skipped > 0:
            print(f"  警告: {skipped} 个样本因标签不在训练集类别中被跳过")
        self.data = valid_data

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
            if self.use_merge and value:
                value = merge_label(feature, value)
            targets.append(self.feature_labels[feature].get(value, 0))
        return img, torch.tensor(targets, dtype=torch.long)

def get_transforms(img_size=224, aug_level="medium"):
    use_randaugment = False
    if aug_level == "light":
        color_jitter = {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05}
        rotation = 10
        erasing_p = 0.0
        affine = False
        perspective = False
        grayscale_p = 0.0
        blur_p = 0.0
    elif aug_level == "strong":
        color_jitter = {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.2}
        rotation = 30
        erasing_p = 0.3
        affine = True
        perspective = True
        grayscale_p = 0.1
        blur_p = 0.1
    elif aug_level == "randaugment":
        use_randaugment = True
        color_jitter = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
        rotation = 15
        erasing_p = 0.2
        affine = False
        perspective = False
        grayscale_p = 0.0
        blur_p = 0.0
    else:
        color_jitter = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
        rotation = 15
        erasing_p = 0.1
        affine = False
        perspective = False
        grayscale_p = 0.0
        blur_p = 0.0

    train_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    
    if use_randaugment:
        train_transform_list.append(
            transforms.RandAugment(num_ops=2, magnitude=9)
        )
    else:
        train_transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(rotation),
        ])
        if affine:
            train_transform_list.append(
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
            )
        if perspective:
            train_transform_list.append(transforms.RandomPerspective(distortion_scale=0.2, p=0.3))
        train_transform_list.append(transforms.ColorJitter(**color_jitter))
        if grayscale_p > 0:
            train_transform_list.append(transforms.RandomGrayscale(p=grayscale_p))
        if blur_p > 0:
            train_transform_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
    
    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    if erasing_p > 0:
        train_transform_list.append(transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.2)))
    
    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def get_data_loaders(seed=None, use_merge=False, aug_level="medium"):
    train_transform, val_transform = get_transforms(TRAIN_PARAMS["img_size"], aug_level=aug_level)
    train_dataset = TeaFlowerDataset(TRAIN_LABELS_DIR, IMAGES_DIR, train_transform, use_merge=use_merge)
    val_dataset = TeaFlowerDataset(VAL_LABELS_DIR, IMAGES_DIR, val_transform,
                                   feature_labels=train_dataset.feature_labels, use_merge=use_merge)

    if seed is not None:
        torch.manual_seed(seed)
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
