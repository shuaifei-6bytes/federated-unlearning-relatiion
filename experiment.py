#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验3：验证 Feature-level Unlearning 的局限性
================================================

核心目标：
证明 Ferrari 式 feature-level unlearning 无法处理"同一 feature 多关系选择性删除"问题。

实验场景：
- 使用 Waterbirds 数据集
- 双头模型：bird classification + environment classification
- 共享 feature：background 同时影响两个任务
- 遗忘请求：只删除 background→bird，保留 background→environment

对比方法：
1. M_global：基线模型
2. M_feature：Ferrari 式 feature-level unlearning
3. M_relation：冻结 backbone + head_env，重训 head_bird

预期结果：
- M_feature：Bird ↓ 且 Environment ↓（过度遗忘）
- M_relation：Bird ↓ 但 Environment 保持（精准遗忘）
"""

import argparse
import csv
import json
import os
import random
import tarfile
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ----------------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------------
DATA_URL = "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz"
DATA_TAR = "waterbird_complete95_forest2water2.tar.gz"
DATA_DIR_NAME = "waterbird_complete95_forest2water2"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# 设备 / 工具
# ----------------------------------------------------------------------------
def get_device():
    """返回可用设备，优先 CUDA。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[CUDA] 使用 GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[CUDA] 警告：未检测到 GPU，回退到 CPU（会很慢）")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------------
def _download(url, dest):
    print("[DATA] 下载数据集 ...")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc="下载")
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()


def prepare_data(data_dir):
    """确保数据存在，返回解压后的根目录。"""
    os.makedirs(data_dir, exist_ok=True)
    root = os.path.join(data_dir, DATA_DIR_NAME)
    if os.path.exists(os.path.join(root, "metadata.csv")):
        print(f"[DATA] 数据已就绪: {root}")
        return root

    tar_path = os.path.join(data_dir, DATA_TAR)
    if not os.path.exists(tar_path):
        _download(DATA_URL, tar_path)

    print(f"[DATA] 解压 {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print(f"[DATA] 数据就绪: {root}")
    return root


def load_metadata(root):
    """解析 metadata.csv，返回 [(img_filename, y, split, place), ...]。"""
    path = os.path.join(root, "metadata.csv")
    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        cols = {c.strip(): i for i, c in enumerate(header)}
        idx_img, idx_y, idx_split, idx_place = (
            cols["img_filename"], cols["y"], cols["split"], cols["place"])
        for line in f:
            p = line.strip().split(",")
            rows.append((p[idx_img], int(p[idx_y]), int(p[idx_split]), int(p[idx_place])))
    return rows


# ----------------------------------------------------------------------------
# 数据集
# ----------------------------------------------------------------------------
class WaterbirdsDataset(Dataset):
    """Waterbirds 数据集：返回 (图片, 鸟标签 y, 环境标签 place)。"""

    def __init__(self, root, split, transform=None):
        self.root = root
        self.transform = transform
        split_map = {"train": 0, "val": 1, "test": 2}
        target = split_map[split]
        self.samples = [
            (img, y, place) for (img, y, s, place) in load_metadata(root) if s == target
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_fn, y, place = self.samples[idx]
        img = Image.open(os.path.join(self.root, img_fn)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y, place


def get_transform(train):
    t = [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        t.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)


# ----------------------------------------------------------------------------
# 模型：共享 backbone + 双 head
# ----------------------------------------------------------------------------
class DualHeadResNet(nn.Module):
    """双头模型：bird classification + environment classification。"""
    
    def __init__(self, num_bird=2, num_env=2, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head_bird = nn.Linear(512, num_bird)   # 任务1：鸟分类
        self.head_env = nn.Linear(512, num_env)     # 任务2：环境分类

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.head_bird(feat), self.head_env(feat)


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    """返回 (bird_acc, env_acc)。"""
    model.eval()
    bird_correct = env_correct = total = 0

    with torch.no_grad():
        for x, y, place in loader:
            x, y, place = x.to(device), y.to(device), place.to(device)
            logit_bird, logit_env = model(x)
            pred_bird = logit_bird.argmax(1)
            pred_env = logit_env.argmax(1)

            bird_correct += (pred_bird == y).sum().item()
            env_correct += (pred_env == place).sum().item()
            total += y.size(0)

    bird_acc = bird_correct / total if total else 0.0
    env_acc = env_correct / total if total else 0.0
    return bird_acc, env_acc


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train_model(model, train_loader, device, epochs, lr):
    """训练双头模型。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, place in train_loader:
            x, y, place = x.to(device), y.to(device), place.to(device)
            optimizer.zero_grad()
            logit_bird, logit_env = model(x)
            loss = nn.functional.cross_entropy(logit_bird, y) + nn.functional.cross_entropy(logit_env, place)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  [Train] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(train_loader)):.4f}")
    
    return model


def feature_unlearn(model, train_loader, device, epochs, lr, sigma):
    """Ferrari 式 feature-level unlearning：最小化 feature sensitivity。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, place in train_loader:
            x = x.to(device)
            delta = torch.randn_like(x) * sigma
            optimizer.zero_grad()
            lb1, le1 = model(x)
            lb2, le2 = model(x + delta)
            
            norm_delta = delta.flatten(1).norm(2, dim=1).clamp(min=1e-8)
            loss_bird = ((lb1 - lb2).pow(2).sum(1).sqrt() / norm_delta).mean()
            loss_env = ((le1 - le2).pow(2).sum(1).sqrt() / norm_delta).mean()
            loss = loss_bird + loss_env
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  [Feature-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(train_loader)):.4f}")
    
    return model


def freeze_except_head_bird(model):
    """冻结 backbone 和 head_env，只允许 head_bird 更新。"""
    for name, p in model.named_parameters():
        p.requires_grad = ("head_bird" in name)


def relation_unlearn(model, train_loader, device, epochs, lr):
    """Relation-level unlearning：冻结 backbone + head_env，重训 head_bird。"""
    freeze_except_head_bird(model)
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if "head_bird" in name], lr=lr)
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y, place in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit_bird, _ = model(x)
            loss = nn.functional.cross_entropy(logit_bird, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  [Relation-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(train_loader)):.4f}")
    
    return model


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--unlearn_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=0.001)
    parser.add_argument("--unlearn_lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=0.1, help="Ferrari 高斯扰动标准差")
    parser.add_argument("--max_train_samples", type=int, default=None, help="限制训练样本数（冒烟用）")
    parser.add_argument("--max_test_samples", type=int, default=None, help="限制测试样本数（冒烟用）")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--quick", action="store_true", help="快速冒烟：小样本+少轮次")
    args = parser.parse_args()

    if args.quick:
        args.train_epochs = 2
        args.unlearn_epochs = 1
        args.max_train_samples = 400
        args.max_test_samples = 200
        print("[QUICK] 冒烟模式：小样本 + 少轮次，仅验证流程")

    set_seed(args.seed)
    device = get_device()
    num_workers = args.num_workers if device.type == "cuda" else 0

    # ---- 数据 ----
    root = prepare_data(args.data_dir)
    train_ds = WaterbirdsDataset(root, "train", get_transform(train=True))
    test_ds = WaterbirdsDataset(root, "test", get_transform(train=False))

    train_indices = list(range(len(train_ds)))
    if args.max_train_samples:
        train_indices = train_indices[:args.max_train_samples]
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers)

    test_indices = list(range(len(test_ds)))
    if args.max_test_samples:
        test_indices = test_indices[:args.max_test_samples]
    test_loader = DataLoader(Subset(test_ds, test_indices), batch_size=args.batch_size,
                             shuffle=False, num_workers=num_workers)

    results = {}

    # ---- Step 1: M_global (训练基线模型) ----
    print("\n===== Step 1: 训练 M_global =====")
    model_global = DualHeadResNet().to(device)
    train_model(model_global, train_loader, device, args.train_epochs, args.train_lr)
    
    bird_acc, env_acc = evaluate(model_global, test_loader, device)
    results["M_global"] = {"bird_acc": bird_acc, "env_acc": env_acc}
    print(f"  M_global -> Bird={bird_acc:.4f}, Environment={env_acc:.4f}")

    # ---- Step 2: M_feature (Ferrari 式 feature-level unlearning) ----
    print("\n===== Step 2: Feature-level Unlearning (Ferrari 式) =====")
    model_feature = DualHeadResNet().to(device)
    model_feature.load_state_dict(model_global.state_dict())
    feature_unlearn(model_feature, train_loader, device, args.unlearn_epochs, args.unlearn_lr, args.sigma)
    
    bird_acc_f, env_acc_f = evaluate(model_feature, test_loader, device)
    results["M_feature"] = {"bird_acc": bird_acc_f, "env_acc": env_acc_f}
    print(f"  M_feature -> Bird={bird_acc_f:.4f}, Environment={env_acc_f:.4f}")

    # ---- Step 3: M_relation (Relation-level unlearning) ----
    print("\n===== Step 3: Relation-level Unlearning (本文方法) =====")
    model_relation = DualHeadResNet().to(device)
    model_relation.load_state_dict(model_global.state_dict())
    relation_unlearn(model_relation, train_loader, device, args.unlearn_epochs, args.unlearn_lr)
    
    bird_acc_r, env_acc_r = evaluate(model_relation, test_loader, device)
    results["M_relation"] = {"bird_acc": bird_acc_r, "env_acc": env_acc_r}
    print(f"  M_relation -> Bird={bird_acc_r:.4f}, Environment={env_acc_r:.4f}")

    # ---- 输出 ----
    os.makedirs(args.out_dir, exist_ok=True)
    for name, d in results.items():
        with open(os.path.join(args.out_dir, f"{name}_results.json"), "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    # 计算 delta
    delta_bird_f = results["M_global"]["bird_acc"] - results["M_feature"]["bird_acc"]
    delta_env_f = results["M_global"]["env_acc"] - results["M_feature"]["env_acc"]
    delta_bird_r = results["M_global"]["bird_acc"] - results["M_relation"]["bird_acc"]
    delta_env_r = results["M_global"]["env_acc"] - results["M_relation"]["env_acc"]

    with open(os.path.join(args.out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["方法", "Bird Accuracy", "Environment Accuracy", "ΔBird", "ΔEnvironment"])
        w.writerow(["M_global", f"{results['M_global']['bird_acc']:.4f}", 
                    f"{results['M_global']['env_acc']:.4f}", "-", "-"])
        w.writerow(["M_feature", f"{results['M_feature']['bird_acc']:.4f}", 
                    f"{results['M_feature']['env_acc']:.4f}", 
                    f"{delta_bird_f:.4f}", f"{delta_env_f:.4f}"])
        w.writerow(["M_relation", f"{results['M_relation']['bird_acc']:.4f}", 
                    f"{results['M_relation']['env_acc']:.4f}", 
                    f"{delta_bird_r:.4f}", f"{delta_env_r:.4f}"])

    # ---- 结果汇总 ----
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'方法':<15} {'Bird Acc':<12} {'Env Acc':<12} {'ΔBird':<10} {'ΔEnv':<10}")
    print("-" * 70)
    print(f"{'M_global':<15} {results['M_global']['bird_acc']:<12.4f} {results['M_global']['env_acc']:<12.4f} {'-':<10} {'-':<10}")
    print(f"{'M_feature':<15} {results['M_feature']['bird_acc']:<12.4f} {results['M_feature']['env_acc']:<12.4f} {delta_bird_f:<10.4f} {delta_env_f:<10.4f}")
    print(f"{'M_relation':<15} {results['M_relation']['bird_acc']:<12.4f} {results['M_relation']['env_acc']:<12.4f} {delta_bird_r:<10.4f} {delta_env_r:<10.4f}")
    print("=" * 70)

    # ---- 结论判读 ----
    print("\n结论判读：")
    if delta_bird_f > 0.05 and delta_env_f > 0.05:
        print("✓ Feature-level unlearning 产生过度遗忘（ΔBird > 0.05 且 ΔEnv > 0.05）")
    else:
        print("✗ Feature-level unlearning 未能证明过度遗忘")
    
    if delta_bird_r > 0.05 and delta_env_r < 0.05:
        print("✓ Relation-level unlearning 实现精准遗忘（ΔBird > 0.05 且 ΔEnv < 0.05）")
    else:
        print("✗ Relation-level unlearning 未能证明精准遗忘")
    
    if (delta_bird_f > 0.05 and delta_env_f > 0.05) and (delta_bird_r > 0.05 and delta_env_r < 0.05):
        print("\n✓✓✓ 实验成功！证明 Feature-level unlearning 无法解决同一 feature 多关系选择性删除问题")
    else:
        print("\n✗✗✗ 实验未完全达到预期，需要调整参数或分析原因")

    print(f"\n结果已写入 {args.out_dir}/ 目录")


if __name__ == "__main__":
    main()
