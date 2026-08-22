#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2：验证 Feature-level Unlearning 无法实现客户端关系级精准遗忘
================================================================

核心假设：同一个 Feature（水背景）参与多条关系时，
  Feature-level Unlearning 只能整体删除该 Feature 的影响（过度遗忘），
  Relation-level Unlearning 才能精准删除"客户端 A 贡献的 水背景→水鸟类别"这一条关系。

与实验1 的区别（本实验新增"客户端贡献"语义）：
  - Client A（遗忘客户端）：持 aligned 样本（水鸟+水 / 陆鸟+陆），
    其贡献是"背景→鸟类别"伪关联；目标遗忘关系 = 水背景→水鸟类别。
  - Client B（环境保留客户端）：持 conflicting 样本（背景与鸟矛盾），
    其贡献是"背景→环境"关系（无鸟捷径、环境标签仍可直接学），需保留。
  - Client C/D/E（正常客户端）：均分剩余 aligned 样本。

三种对比方法：
  M_global   : FedAvg 联邦训练（遗忘前基线）
  M_feature  : Ferrari 式 feature-level unlearning（对输入加高斯噪声，
               最小化输出对扰动的敏感度 → 整体削弱背景 Feature）
  M_relation : relation-level unlearning（冻结 backbone 与环境 head，
               只用去关联的均衡数据重训鸟分类 head → 只删"背景→鸟类别"关系）

评价指标（设计单2 第九节）：
  Bird Accuracy         —— 鸟主体→鸟类别（正常分类保持）
  Environment Accuracy  —— 水背景→水环境（非目标关系保持，核心）
  Background Gap        —— 目标关系遗忘（aligned-conflicting，越低越好）
  Target Relation Score —— 目标关系强度（=aligned 组鸟分类 acc，越低=遗忘越彻底）
  Retention Score       —— 非目标知识保持（=环境 acc，越高越好）

用法（Colab/Kaggle GPU，CUDA 自动适配）：
  python experiment.py --data_dir ./data --global_epochs 20 --unlearn_epochs 5

快速冒烟（CPU 可跑，验证流程）：
  python experiment.py --data_dir ./data --quick

依赖：torch, torchvision, numpy, pillow, tqdm
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

NUM_CLIENTS = 5  # A(遗忘) + B(环境) + C/D/E(正常)


# ----------------------------------------------------------------------------
# 设备 / 工具
# ----------------------------------------------------------------------------
def get_device():
    """返回可用设备，优先 CUDA（适配 Colab / Kaggle GPU）。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # 固定输入尺寸，加速卷积
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
    """确保数据存在，返回解压后的根目录（内含 metadata.csv 与图片）。"""
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
    """按列名解析 metadata.csv，返回 [(img_filename, y, split, place), ...]。

    Waterbirds 官方语义：
      y     : 1=水鸟, 0=陆鸟
      place : 1=水背景, 0=陆背景
      split : 0=train, 1=val, 2=test
    本实验只用 y==place 判断 aligned / conflicting 组。
    """
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
        # 实验3 标签重映射：
        # label=0 (Bird): waterbird + water background (y=1, place=1)
        # label=1 (Boat): landbird + water background (y=0, place=1)
        # 其他样本（land background）不参与核心关系，用于 C/D/E
        if y == 1 and place == 1:  # waterbird + water
            new_label = 0  # Bird
        elif y == 0 and place == 1:  # landbird + water
            new_label = 1  # Boat
        else:  # land background 样本
            new_label = -1  # 标记为非核心样本
        if self.transform:
            img = self.transform(img)
        return img, new_label, y  # 返回 img, new_label, original_y（用于统计）


def get_transform(train):
    t = [transforms.Resize((224, 224)), transforms.ToTensor(),
         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        t.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)


# ----------------------------------------------------------------------------
# 模型：共享 ResNet18 backbone + 单 head（Bird vs Boat 二分类）
# ----------------------------------------------------------------------------
class SingleHeadResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head_relation = nn.Linear(512, num_classes)  # 单任务：Bird(0) vs Boat(1)

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.head_relation(feat)


def freeze_except_head_relation(model):
    """冻结 backbone，只允许 head_relation 更新（relation-level 用）。"""
    for name, p in model.named_parameters():
        p.requires_grad = ("head_relation" in name)


# ----------------------------------------------------------------------------
# 评估（实验3语义：water→Bird 目标关系强度 / water→Boat 保留关系保持度）
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    """返回 (overall_acc, water_bird_acc, water_boat_acc, gap)。

    - water_bird_acc: new_label=0 样本的准确率，即 water→Bird 关系强度
    - water_boat_acc: new_label=1 样本的准确率，即 water→Boat 关系保持度
    - gap = water_bird_acc - water_boat_acc（目标关系遗忘程度）
    """
    model.eval()
    correct = total = 0
    group_correct = {"water_bird": 0, "water_boat": 0}
    group_total = {"water_bird": 0, "water_boat": 0}

    with torch.no_grad():
        for x, label, original_y in loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(1)

            correct += (pred == label).sum().item()
            total += label.size(0)

            # 按 new_label 分组：label=0 是 water→Bird，label=1 是 water→Boat
            # 只统计有效样本（label != -1）
            for i in range(label.size(0)):
                lbl = label[i].item()
                if lbl == 0:
                    group_total["water_bird"] += 1
                    group_correct["water_bird"] += (pred[i] == label[i]).item()
                elif lbl == 1:
                    group_total["water_boat"] += 1
                    group_correct["water_boat"] += (pred[i] == label[i]).item()

    overall_acc = correct / total if total else 0.0
    water_bird_acc = group_correct["water_bird"] / group_total["water_bird"] if group_total["water_bird"] else 0.0
    water_boat_acc = group_correct["water_boat"] / group_total["water_boat"] if group_total["water_boat"] else 0.0
    gap = water_bird_acc - water_boat_acc
    return overall_acc, water_bird_acc, water_boat_acc, gap


# ----------------------------------------------------------------------------
# 训练（实验3：单任务 Bird vs Boat 二分类）
# ----------------------------------------------------------------------------
def train_fedavg(model, client_loaders, device, global_epochs, local_epochs, lr, frac, seed):
    """FedAvg 联邦训练：单任务二分类（Bird vs Boat），得到 M_global。"""
    num_clients = len(client_loaders)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for rnd in range(global_epochs):
        selected = random.sample(range(num_clients), max(1, int(frac * num_clients)))
        local_states = []

        for cid in selected:
            model.train()
            for _ in range(local_epochs):
                for x, label, original_y in client_loaders[cid]:
                    x, label = x.to(device), label.to(device)
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, label)
                    loss.backward()
                    optimizer.step()
            local_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})

        avg_state = {k: sum(s[k] for s in local_states) / len(local_states) for k in local_states[0]}
        model.load_state_dict(avg_state)

        if (rnd + 1) % max(1, global_epochs // 5) == 0 or rnd == global_epochs - 1:
            print(f"  [FedAvg] round {rnd + 1}/{global_epochs} 完成")
    return model


def feature_unlearn(model, unlearn_loader, device, epochs, lr, sigma):
    """Ferrari 式 feature-level unlearning。

    对输入 x 加高斯噪声扰动 δ，最小化 feature sensitivity
        s = E[ ||f(x) - f(x+δ)||_2 / ||δ||_2 ]
    使模型对"背景"Feature 的扰动不再敏感 —— 等价于整体削弱背景 Feature。
    作用于整个模型（包括 head_relation），因此会同时影响 water→Bird 和 water→Boat 两条关系（过度遗忘）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, label, original_y in unlearn_loader:
            x = x.to(device)
            delta = torch.randn_like(x) * sigma
            optimizer.zero_grad()
            logits1 = model(x)
            logits2 = model(x + delta)

            norm_delta = delta.flatten(1).norm(2, dim=1).clamp(min=1e-8)
            loss = ((logits1 - logits2).pow(2).sum(1).sqrt() / norm_delta).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [Feature-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(unlearn_loader)):.4f}")
    return model


def build_balanced_loader(dataset, indices, batch_size, seed, num_workers=0):
    """从给定 indices 构造均衡子集（water_bird/water_land 各半），
    使"水背景"与"类别"不再相关，用于 relation-level 去关联重训。"""
    water_bird = []  # waterbird + water (y=1, place=1) → new_label=0 (Bird)
    water_land = []  # landbird + water (y=0, place=1) → new_label=1 (Boat)
    for i in indices:
        _, y, place = dataset.samples[i]
        if place == 1:  # 只处理 water background
            if y == 1:  # waterbird
                water_bird.append(i)
            else:  # landbird
                water_land.append(i)
    random.seed(seed)
    random.shuffle(water_bird)
    random.shuffle(water_land)
    n = min(len(water_bird), len(water_land))
    balanced = water_bird[:n] + water_land[:n]
    random.shuffle(balanced)
    sub = Subset(dataset, balanced)
    return DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def relation_unlearn(model, dataset, all_train_indices, device, epochs, lr, batch_size, seed, num_workers=0):
    """relation-level unlearning（本文方法）。

    冻结 backbone，只用均衡数据重训 head_relation：
    让分类头学会"不看背景、只看主体"，从而只删除"背景→Bird"关系，
    同时完整保留"背景→Boat"关系。

    注意：均衡数据必须来自全训练集（而非仅客户端 A），
    因为客户端 A 只含 aligned 样本，单独无法提供 conflicting 样本去关联。
    """
    freeze_except_head_relation(model)
    loader = build_balanced_loader(dataset, all_train_indices, batch_size, seed, num_workers)
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if "head_relation" in name], lr=lr)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, label, original_y in loader:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [Relation-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(loader)):.4f}")
    return model


# ----------------------------------------------------------------------------
# 联邦数据划分（实验3语义：Client A 贡献 water→Bird，Client B 贡献 water→Boat）
# ----------------------------------------------------------------------------
def build_client_loaders(train_dataset, indices, batch_size, seed, num_workers=0, max_samples_per_client=400):
    """按"客户端贡献关系"划分训练集（实验3语义，3客户端平衡版本）。

    train_dataset 必须是完整 WaterbirdsDataset（含 .samples），indices 指定参与划分的下标。
    返回 (client_loaders, client_a_indices)：
      client_a_indices = 遗忘客户端 A 的数据下标。

    实验3客户端划分（3客户端等量平衡）：
      - Client A = waterbird+water 样本（water→Bird，遗忘目标）
      - Client B = landbird+water 样本（water→Boat，保留关系）
      - Client C = land background 样本下采样（普通保持，量级与 A/B 一致）
    """
    water_bird = []  # waterbird + water (y=1, place=1) → Bird
    water_land = []  # landbird + water (y=0, place=1) → Boat
    land_samples = []  # land background (place=0) → 普通保持

    for i in indices:
        _, y, place = train_dataset.samples[i]
        if place == 1:
            if y == 1:
                water_bird.append(i)
            else:
                water_land.append(i)
        else:
            land_samples.append(i)

    random.seed(seed)
    random.shuffle(water_bird)
    random.shuffle(water_land)
    random.shuffle(land_samples)

    # 确定平衡目标数量（取三者最小值，不超过 max_samples_per_client）
    target_size = min(len(water_bird), len(water_land), len(land_samples), max_samples_per_client)
    
    client_a_data = water_bird[:target_size]
    client_b_data = water_land[:target_size]
    client_c_data = land_samples[:target_size]

    groups = [
        client_a_data,   # A: 遗忘客户端（water→Bird）
        client_b_data,   # B: 保留客户端（water→Boat）
        client_c_data,   # C: 普通保持客户端
    ]

    # 创建 DataLoader（必须过滤掉 new_label=-1 的样本，否则 CE loss 会崩溃）
    loaders = []
    actual_sizes = []
    for cid, g in enumerate(groups):
        valid_indices = []
        for idx in g:
            _, y, place = train_dataset.samples[idx]
            # 只保留属于二分类任务（Bird/Boat）的样本
            if (y == 1 and place == 1) or (y == 0 and place == 1):
                valid_indices.append(idx)
        
        if valid_indices:
            loaders.append(DataLoader(Subset(train_dataset, valid_indices), 
                                     batch_size=batch_size, shuffle=True, num_workers=num_workers))
            actual_sizes.append(len(valid_indices))
        else:
            # 该客户端没有有效训练样本（如 Client C 的 land background）
            actual_sizes.append(0)

    # 详细统计输出
    print(f"\n{'='*60}")
    print(f"[DATA] 客户端划分统计（3客户端等量平衡）")
    print(f"{'='*60}")
    print(f"  原始数据池: water_bird={len(water_bird)}, water_land={len(water_land)}, land={len(land_samples)}")
    print(f"  平衡目标数量: {target_size}")
    
    print(f"\n  --- Client A (water→Bird 遗忘目标) ---")
    if client_a_data:
        n = len(client_a_data)
        bg_w = sum(1 for i in client_a_data if train_dataset.samples[i][2] == 1)
        bird_w = sum(1 for i in client_a_data if train_dataset.samples[i][1] == 1)
        print(f"    样本数: {n}")
        print(f"    background: water={bg_w/n*100:.1f}%, land={(n-bg_w)/n*100:.1f}%")
        print(f"    bird类别: waterbird={bird_w/n*100:.1f}%, landbird={(n-bird_w)/n*100:.1f}%")
        print(f"    新label: Bird(0)=100%")
        print(f"    ✓ 核心关系: water background → Bird")
    
    print(f"\n  --- Client B (water→Boat 保留关系) ---")
    if client_b_data:
        n = len(client_b_data)
        bg_w = sum(1 for i in client_b_data if train_dataset.samples[i][2] == 1)
        bird_l = sum(1 for i in client_b_data if train_dataset.samples[i][1] == 0)
        print(f"    样本数: {n}")
        print(f"    background: water={bg_w/n*100:.1f}%, land={(n-bg_w)/n*100:.1f}%")
        print(f"    bird类别: waterbird={(n-bird_l)/n*100:.1f}%, landbird={bird_l/n*100:.1f}%")
        print(f"    新label: Boat(1)=100%")
        print(f"    ✓ 核心关系: water background → Boat")
    
    print(f"\n  --- Client C (普通保持) ---")
    if client_c_data:
        n = len(client_c_data)
        bg_l = sum(1 for i in client_c_data if train_dataset.samples[i][2] == 0)
        print(f"    样本数: {n}")
        print(f"    background: land={bg_l/n*100:.1f}%, water={(n-bg_l)/n*100:.1f}%")
    
    # 平衡验证
    print(f"\n  --- 平衡验证 ---")
    sizes = [len(g) for g in groups if g]
    if len(set(sizes)) == 1:
        print(f"  ✓ 三客户端样本数完全一致: {sizes[0]}")
    else:
        print(f"  △ 客户端样本数: A={len(client_a_data)}, B={len(client_b_data)}, C={len(client_c_data)}")
    
    print(f"{'='*60}\n")

    return loaders, client_a_data


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--frac", type=float, default=1.0, help="每轮参与训练的客户端比例")
    parser.add_argument("--global_epochs", type=int, default=50)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=0.001)
    parser.add_argument("--unlearn_epochs", type=int, default=20, help="feature/relation 遗忘阶段轮数")
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
        args.global_epochs = 2
        args.unlearn_epochs = 1
        args.max_train_samples = 400
        args.max_test_samples = 200
        print("[QUICK] 冒烟模式：小样本 + 少轮次，仅验证流程")

    set_seed(args.seed)
    device = get_device()
    num_workers = args.num_workers if device.type == "cuda" else 0  # CPU 下避免多进程开销

    # ---- 数据 ----
    root = prepare_data(args.data_dir)
    train_ds = WaterbirdsDataset(root, "train", get_transform(train=True))
    test_ds = WaterbirdsDataset(root, "test", get_transform(train=False))

    # 客户端划分（A=aligned 遗忘, B=conflicting 环境, C/D/E=正常）
    train_indices = list(range(len(train_ds)))
    if args.max_train_samples:
        train_indices = train_indices[:args.max_train_samples]
    client_loaders, client_a_indices = build_client_loaders(train_ds, train_indices, args.batch_size, args.seed, num_workers)

    all_train_indices = train_indices
    if args.max_test_samples:
        test_ds = Subset(test_ds, list(range(args.max_test_samples)))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # 遗忘客户端 A 的数据（feature-level 的遗忘对象）
    unlearn_loader = DataLoader(Subset(train_ds, client_a_indices), batch_size=args.batch_size,
                                shuffle=True, num_workers=num_workers)

    results = {}

    # ---- Step 1: M_global (FedAvg 训练) ----
    print("\n===== Step 1: FedAvg 训练 M_global =====")
    model_global = SingleHeadResNet().to(device)
    train_fedavg(model_global, client_loaders, device, args.global_epochs, args.local_epochs,
                 args.train_lr, args.frac, args.seed)
    r = evaluate(model_global, test_loader, device)
    results["M_global"] = _pack(r)
    _print_result("M_global", results["M_global"])

    # ---- 训练后检查：验证 M_global 是否同时学习两个关系 ----
    print("\n===== M_global 训练后检查 =====")
    global_results = results["M_global"]
    water_bird_acc = global_results["water_bird_acc"]
    water_boat_acc = global_results["water_boat_acc"]
    
    print(f"water→Bird accuracy: {water_bird_acc:.4f}")
    print(f"water→Boat accuracy: {water_boat_acc:.4f}")
    
    if water_boat_acc < 0.7:
        print(f"\n[WARNING] Global model failed to learn retained relation (water→Boat)")
        print(f"water→Boat accuracy ({water_boat_acc:.4f}) < 0.7")
        print(f"实验终止：模型未能同时学习两个核心关系")
        print(f"请调整训练配置（增加 epochs、调整学习率或客户端数量）后重试")
        return
    
    if water_bird_acc < 0.7:
        print(f"\n[WARNING] Global model failed to learn target relation (water→Bird)")
        print(f"water→Bird accuracy ({water_bird_acc:.4f}) < 0.7")
        print(f"实验终止：模型未能同时学习两个核心关系")
        print(f"请调整训练配置后重试")
        return
    
    print(f"[OK] Global model successfully learned both relations")
    print(f"继续执行 unlearning 步骤...\n")

    # ---- Step 2: M_feature (Ferrari 式 feature-level unlearning) ----
    print("\n===== Step 2: Feature-level Unlearning (Ferrari 式) =====")
    model_feature = SingleHeadResNet().to(device)
    model_feature.load_state_dict(model_global.state_dict())
    feature_unlearn(model_feature, unlearn_loader, device, args.unlearn_epochs, args.unlearn_lr, args.sigma)
    r = evaluate(model_feature, test_loader, device)
    results["M_feature"] = _pack(r)
    _print_result("M_feature", results["M_feature"])

    # ---- Step 3: M_relation (relation-level unlearning) ----
    print("\n===== Step 3: Relation-level Unlearning (本文方法) =====")
    model_relation = SingleHeadResNet().to(device)
    model_relation.load_state_dict(model_global.state_dict())
    relation_unlearn(model_relation, train_ds, all_train_indices, device,
                     args.unlearn_epochs, args.unlearn_lr, args.batch_size, args.seed, num_workers)
    r = evaluate(model_relation, test_loader, device)
    results["M_relation"] = _pack(r)
    _print_result("M_relation", results["M_relation"])

    # ---- 输出 ----
    os.makedirs(args.out_dir, exist_ok=True)
    filename_map = {"M_global": "M_global.json", "M_feature": "feature_unlearning.json",
                    "M_relation": "relation_unlearning.json"}
    for name, d in results.items():
        with open(os.path.join(args.out_dir, filename_map[name]), "w") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["方法", "Overall Accuracy(总准确率)", "Water→Bird Accuracy(目标关系强度,越低越好)",
                    "Water→Boat Accuracy(非目标关系保持,越高越好)", "Gap(目标关系遗忘程度)"])
        for name, d in results.items():
            w.writerow([name, f"{d['overall_acc']:.4f}", f"{d['water_bird_acc']:.4f}",
                        f"{d['water_boat_acc']:.4f}", f"{d['gap']:.4f}"])

    print("\n===== 结果汇总 =====")
    print(f"{'方法':<12} {'总acc':<10} {'water→Bird':<12} {'water→Boat':<12} {'gap':<10}")
    for name, d in results.items():
        print(f"{name:<12} {d['overall_acc']:.4f}      {d['water_bird_acc']:.4f}        {d['water_boat_acc']:.4f}        {d['gap']:.4f}")

    print("\n结论判读（预期）：")
    print("  - 若 M_feature 的 water→Bird 下降但 water→Boat 也明显下降 -> feature-level 过度遗忘，成立")
    print("  - 若 M_relation 的 water→Bird 下降且 water→Boat 保持    -> relation-level 精准遗忘，成立")
    print(f"\n结果已写入 {args.out_dir}/ 目录")


def _pack(r):
    """把 evaluate 的返回值打包成实验3要求的指标。"""
    overall_acc, water_bird_acc, water_boat_acc, gap = r
    return {
        "overall_acc": overall_acc,
        "water_bird_acc": water_bird_acc,           # 目标关系"water→Bird"强度（越低=遗忘越彻底）
        "water_boat_acc": water_boat_acc,           # 非目标关系"water→Boat"保持度（越高越好）
        "gap": gap,                                  # water_bird_acc - water_boat_acc
    }


def _print_result(name, d):
    print(f"  {name} -> overall={d['overall_acc']:.4f}  water→Bird={d['water_bird_acc']:.4f}  "
          f"water→Boat={d['water_boat_acc']:.4f}  gap={d['gap']:.4f}")


if __name__ == "__main__":
    main()
