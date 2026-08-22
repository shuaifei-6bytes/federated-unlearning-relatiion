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
# 模型：共享 ResNet18 backbone + 双 head（鸟分类 / 环境分类）
# ----------------------------------------------------------------------------
class DualHeadResNet(nn.Module):
    def __init__(self, num_bird=2, num_env=2, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head_bird = nn.Linear(512, num_bird)   # 任务1：鸟类别
        self.head_env = nn.Linear(512, num_env)     # 任务2：环境类别

    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.head_bird(feat), self.head_env(feat)


def freeze_except_head_bird(model):
    """冻结 backbone 与 head_env，只允许 head_bird 更新（relation-level 用）。"""
    for name, p in model.named_parameters():
        p.requires_grad = ("head_bird" in name)


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    """返回 (bird_acc, env_acc, gap, {"aligned": .., "conflicting": ..})。

    gap = aligned 组鸟分类 acc − conflicting 组鸟分类 acc。
    模型越依赖背景（"背景→鸟类别"伪关联越强），gap 越大。
    """
    model.eval()
    bird_correct = env_correct = total = 0
    group_correct = {"aligned": 0, "conflicting": 0}
    group_total = {"aligned": 0, "conflicting": 0}

    with torch.no_grad():
        for x, y, place in loader:
            x, y, place = x.to(device), y.to(device), place.to(device)
            logit_bird, logit_env = model(x)
            pred_bird = logit_bird.argmax(1)
            pred_env = logit_env.argmax(1)

            bird_correct += (pred_bird == y).sum().item()
            env_correct += (pred_env == place).sum().item()
            total += y.size(0)

            aligned = (y == place)
            for i in range(y.size(0)):
                key = "aligned" if aligned[i].item() else "conflicting"
                group_total[key] += 1
                group_correct[key] += (pred_bird[i] == y[i]).item()

    bird_acc = bird_correct / total if total else 0.0
    env_acc = env_correct / total if total else 0.0
    aligned_acc = group_correct["aligned"] / group_total["aligned"] if group_total["aligned"] else 0.0
    conflict_acc = group_correct["conflicting"] / group_total["conflicting"] if group_total["conflicting"] else 0.0
    gap = aligned_acc - conflict_acc
    return bird_acc, env_acc, gap, {"aligned": aligned_acc, "conflicting": conflict_acc}


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def _multi_task_loss(logit_bird, logit_env, y, place):
    return nn.functional.cross_entropy(logit_bird, y) + nn.functional.cross_entropy(logit_env, place)


def train_fedavg(model, client_loaders, device, global_epochs, local_epochs, lr, frac, seed):
    """FedAvg 联邦训练（多任务：鸟分类 + 环境分类），得到 M_global。"""
    num_clients = len(client_loaders)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for rnd in range(global_epochs):
        selected = random.sample(range(num_clients), max(1, int(frac * num_clients)))
        local_states = []

        for cid in selected:
            model.train()
            for _ in range(local_epochs):
                for x, y, place in client_loaders[cid]:
                    x, y, place = x.to(device), y.to(device), place.to(device)
                    optimizer.zero_grad()
                    lb, le = model(x)
                    loss = _multi_task_loss(lb, le, y, place)
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
    同时作用于两个 head，因此会连同"背景→环境"关系一起遗忘（过度遗忘）。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, y, place in unlearn_loader:
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
        print(f"  [Feature-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(unlearn_loader)):.4f}")
    return model


def build_balanced_loader(dataset, indices, batch_size, seed, num_workers=0):
    """从给定 indices 构造四组均衡子集（aligned/conflicting 各半），
    使"背景"与"鸟类别"不再相关，用于 relation-level 去关联重训。"""
    groups = {"aligned": [], "conflicting": []}
    for i in indices:
        _, y, place = dataset.samples[i]
        groups["aligned" if y == place else "conflicting"].append(i)
    random.seed(seed)
    for k in groups:
        random.shuffle(groups[k])
    n = min(len(groups["aligned"]), len(groups["conflicting"]))
    balanced = groups["aligned"][:n] + groups["conflicting"][:n]
    random.shuffle(balanced)
    sub = Subset(dataset, balanced)
    return DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def relation_unlearn(model, dataset, all_train_indices, device, epochs, lr, batch_size, seed, num_workers=0):
    """relation-level unlearning（本文方法）。

    冻结 backbone 与 head_env，只用均衡数据重训 head_bird：
    让鸟分类 head 学会"不看背景、只看鸟主体"，只删"背景→鸟类别"关系，
    同时完整保留"背景→环境"关系。

    注意：均衡数据必须来自全训练集（而非仅客户端 A），
    因为客户端 A 只含 aligned 样本，单独无法提供 conflicting 样本去关联。
    """
    freeze_except_head_bird(model)
    loader = build_balanced_loader(dataset, all_train_indices, batch_size, seed, num_workers)
    optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if "head_bird" in name], lr=lr)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, y, place in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            lb, _ = model(x)
            loss = nn.functional.cross_entropy(lb, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  [Relation-Unlearn] epoch {ep + 1}/{epochs}  loss={total_loss / max(1, len(loader)):.4f}")
    return model


# ----------------------------------------------------------------------------
# 联邦数据划分（客户端贡献语义）
# ----------------------------------------------------------------------------
def build_client_loaders(train_dataset, indices, batch_size, seed, num_workers=0):
    """按"客户端贡献关系"划分训练集（非随机均分）。

    train_dataset 必须是完整 WaterbirdsDataset（含 .samples），indices 指定参与划分的下标。
    返回 (client_loaders, client_a_indices)：
      client_a_indices = 遗忘客户端 A 的数据下标（aligned，即"背景→鸟类别"关系来源）。
    """
    aligned, conflict = [], []
    for i in indices:
        _, y, place = train_dataset.samples[i]
        (aligned if y == place else conflict).append(i)

    random.seed(seed)
    random.shuffle(aligned)
    random.shuffle(conflict)

    # A 取 1/4 aligned 作遗忘客户端，剩余 3/4 分给 C/D/E
    n_a = len(aligned) // 4
    client_a = aligned[:n_a]
    rest = aligned[n_a:]
    per = len(rest) // 3
    groups = [
        client_a,                          # A: 遗忘客户端（背景→鸟类别关系来源）
        conflict,                          # B: 环境保留客户端（背景→环境关系）
        rest[:per],                        # C
        rest[per:2 * per],                 # D
        rest[2 * per:],                    # E
    ]

    loaders = [
        DataLoader(Subset(train_dataset, g), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for g in groups
    ]
    print(f"[DATA] 客户端划分: A={len(client_a)} aligned | B={len(conflict)} conflicting | "
          f"C/D/E={[len(g) for g in groups[2:]]}")
    return loaders, client_a


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--frac", type=float, default=1.0, help="每轮参与训练的客户端比例")
    parser.add_argument("--global_epochs", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_lr", type=float, default=0.001)
    parser.add_argument("--unlearn_epochs", type=int, default=5, help="feature/relation 遗忘阶段轮数")
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
    model_global = DualHeadResNet().to(device)
    train_fedavg(model_global, client_loaders, device, args.global_epochs, args.local_epochs,
                 args.train_lr, args.frac, args.seed)
    r = evaluate(model_global, test_loader, device)
    results["M_global"] = _pack(r)
    _print_result("M_global", results["M_global"])

    # ---- Step 2: M_feature (Ferrari 式 feature-level unlearning) ----
    print("\n===== Step 2: Feature-level Unlearning (Ferrari 式) =====")
    model_feature = DualHeadResNet().to(device)
    model_feature.load_state_dict(model_global.state_dict())
    feature_unlearn(model_feature, unlearn_loader, device, args.unlearn_epochs, args.unlearn_lr, args.sigma)
    r = evaluate(model_feature, test_loader, device)
    results["M_feature"] = _pack(r)
    _print_result("M_feature", results["M_feature"])

    # ---- Step 3: M_relation (relation-level unlearning) ----
    print("\n===== Step 3: Relation-level Unlearning (本文方法) =====")
    model_relation = DualHeadResNet().to(device)
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

    with open(os.path.join(args.out_dir, "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["方法", "Bird Accuracy(鸟分类保持)", "Environment Accuracy(环境关系保持)",
                    "Background Gap(目标关系遗忘,越低越好)", "Target Relation Score(目标关系强度,越低越好)",
                    "Retention Score(非目标知识保持,越高越好)"])
        for name, d in results.items():
            w.writerow([name, f"{d['bird_acc']:.4f}", f"{d['env_acc']:.4f}", f"{d['gap']:.4f}",
                        f"{d['target_relation_score']:.4f}", f"{d['retention_score']:.4f}"])

    print("\n===== 结果汇总 =====")
    print(f"{'方法':<12} {'鸟分类acc':<10} {'环境acc':<10} {'背景gap':<10} {'目标关系':<10} {'保持得分':<10}")
    for name, d in results.items():
        print(f"{name:<12} {d['bird_acc']:.4f}      {d['env_acc']:.4f}     {d['gap']:.4f}     "
              f"{d['target_relation_score']:.4f}     {d['retention_score']:.4f}")

    print("\n结论判读（预期）：")
    print("  - 若 M_feature 的 gap 下降但 env_acc 明显下降 -> feature-level 过度遗忘，成立")
    print("  - 若 M_relation 的 gap 下降且 env_acc 保持    -> relation-level 精准遗忘，成立")
    print(f"\n结果已写入 {args.out_dir}/ 目录")


def _pack(r):
    """把 evaluate 的返回值打包成设计单2 要求的 5 项指标（+ 透明度字段）。"""
    bird_acc, env_acc, gap, group = r
    return {
        "bird_acc": bird_acc,
        "env_acc": env_acc,
        "gap": gap,
        "target_relation_score": group["aligned"],   # 目标关系"背景→鸟类别"强度（越低=遗忘越彻底）
        "retention_score": env_acc,                  # 非目标关系"背景→环境"保持度（越高越好）
        "aligned_acc": group["aligned"],
        "conflicting_acc": group["conflicting"],
    }


def _print_result(name, d):
    print(f"  {name} -> bird_acc={d['bird_acc']:.4f}  env_acc={d['env_acc']:.4f}  "
          f"gap={d['gap']:.4f}  target_rel={d['target_relation_score']:.4f}  retention={d['retention_score']:.4f}")


if __name__ == "__main__":
    main()
