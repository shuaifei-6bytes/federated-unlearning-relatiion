"""
关系可观测性验证实验（Relation Observability Experiment）

实验目的：验证模型中不同的背景-类别关系是否可以被单独定义、区分和量化。

三个模型：
- M_global: 基础联邦学习模型
- M_target_enhance: 增强 Waterbird+Water 样本比例
- M_target_reduce: 减少 Waterbird+Water 样本，增加 Waterbird+Land 样本

三个指标：
- Target Relation Score = Acc(Waterbird + Water)
- Retention Score = Acc(Landbird + Land)
- Background Gap = |Aligned Acc - Conflict Acc|

配置：
- 数据集: Waterbirds
- 模型: ResNet18
- 联邦: FedAvg, 3客户端, 20轮
- 训练: local_epochs=2, batch_size=64, SGD, lr=0.01
- 种子: 2个 (42, 123)

适配 Colab CUDA 环境
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

# ============== 配置 ==============

class Config:
    # 数据集路径（Colab 挂载 Google Drive 后的路径）
    DATA_DIR = "/content/drive/MyDrive/datasets/waterbird_complete95_forest2water2"
    
    # 模型配置
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 2  # 0: Landbird, 1: Waterbird
    
    # 联邦学习配置
    NUM_CLIENTS = 3
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    
    # 实验配置
    SEEDS = [42, 123]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 数据增强
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ============== 数据集 ==============

class WaterbirdsDataset(Dataset):
    """
    Waterbirds 数据集
    
    类别映射：
    - y=0: Landbird (陆鸟)
    - y=1: Waterbird (水鸟)
    
    环境映射：
    - place=0: Land (陆地)
    - place=1: Water (水)
    """
    
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加载 metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """加载 metadata.csv"""
        csv_path = os.path.join(self.data_dir, "metadata.csv")
        df = pd.read_csv(csv_path)
        
        # 按 split 过滤: 0=train, 1=val, 2=test
        split_map = {"train": 0, "val": 1, "test": 2}
        split_id = split_map[self.split]
        df = df[df["split"] == split_id].reset_index(drop=True)
        
        return df
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 加载图片
        img_path = os.path.join(self.data_dir, row["img_filename"])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # 标签
        label = int(row["y"])  # 0: Landbird, 1: Waterbird
        place = int(row["place"])  # 0: Land, 1: Water
        
        return image, label, place


def get_dataset_stats(dataset):
    """获取数据集统计信息"""
    stats = defaultdict(int)
    for _, label, place in dataset:
        key = f"label{label}_place{place}"
        stats[key] += 1
    return dict(stats)


# ============== 数据采样策略 ==============

def create_enhanced_dataset(train_dataset, enhance_ratio=3.0):
    """
    创建增强目标关系的数据集
    
    增强 Waterbird+Water (label=1, place=1) 样本比例
    
    Args:
        train_dataset: 原始训练集
        enhance_ratio: 增强倍数，默认3倍
    
    Returns:
        增强后的样本索引列表
    """
    metadata = train_dataset.metadata
    
    # 找到所有 Waterbird+Water 样本
    target_indices = metadata[
        (metadata["y"] == 1) & (metadata["place"] == 1)
    ].index.tolist()
    
    # 其他样本保持不变
    other_indices = metadata[
        ~((metadata["y"] == 1) & (metadata["place"] == 1))
    ].index.tolist()
    
    # 重复目标样本
    enhanced_indices = target_indices * int(enhance_ratio)
    
    # 合并并打乱
    all_indices = other_indices + enhanced_indices
    random.shuffle(all_indices)
    
    return all_indices


def create_reduced_dataset(train_dataset):
    """
    创建削弱目标关系的数据集
    
    减少 Waterbird+Water (label=1, place=1) 样本
    增加 Waterbird+Land (label=1, place=0) 样本
    
    Args:
        train_dataset: 原始训练集
    
    Returns:
        削弱后的样本索引列表
    """
    metadata = train_dataset.metadata
    
    # 分类统计
    ww_indices = metadata[
        (metadata["y"] == 1) & (metadata["place"] == 1)
    ].index.tolist()  # Waterbird+Water
    
    wl_indices = metadata[
        (metadata["y"] == 1) & (metadata["place"] == 0)
    ].index.tolist()  # Waterbird+Land
    
    ll_indices = metadata[
        (metadata["y"] == 0) & (metadata["place"] == 0)
    ].index.tolist()  # Landbird+Land
    
    lw_indices = metadata[
        (metadata["y"] == 0) & (metadata["place"] == 1)
    ].index.tolist()  # Landbird+Water
    
    # 减少 Waterbird+Water：只保留 30%
    random.shuffle(ww_indices)
    ww_keep = ww_indices[:int(len(ww_indices) * 0.3)]
    
    # 增加 Waterbird+Land：重复 3 倍
    wl_enhanced = wl_indices * 3
    
    # 其他类别保持不变
    other_indices = ll_indices + lw_indices
    
    # 合并并打乱
    all_indices = ww_keep + wl_enhanced + other_indices
    random.shuffle(all_indices)
    
    return all_indices


def create_client_datasets(dataset, indices, num_clients):
    """
    将数据分配给多个客户端（IID 划分）
    
    Args:
        dataset: 完整数据集
        indices: 样本索引列表
        num_clients: 客户端数量
    
    Returns:
        每个客户端的 DataLoader 列表
    """
    # 打乱索引
    random.shuffle(indices)
    
    # 均分给各客户端
    chunk_size = len(indices) // num_clients
    client_loaders = []
    
    for i in range(num_clients):
        start = i * chunk_size
        end = start + chunk_size if i < num_clients - 1 else len(indices)
        client_indices = indices[start:end]
        
        client_subset = Subset(dataset, client_indices)
        client_loader = DataLoader(
            client_subset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        client_loaders.append(client_loader)
    
    return client_loaders


# ============== 模型 ==============

def create_model():
    """创建 ResNet18 模型"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # 修改最后一层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, Config.NUM_CLASSES)
    return model


# ============== 联邦学习 ==============

def local_train(model, dataloader, epochs, lr, device):
    """
    本地训练（客户端）
    
    Args:
        model: 模型
        dataloader: 数据加载器
        epochs: 本地训练轮数
        lr: 学习率
        device: 设备
    
    Returns:
        训练后的模型
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=Config.MOMENTUM)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return model


def fed_avg(global_model, client_models):
    """
    FedAvg 聚合
    
    Args:
        global_model: 全局模型
        client_models: 客户端模型列表
    
    Returns:
        更新后的全局模型
    """
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        # 平均聚合
        global_dict[key] = torch.stack([
            client_models[i].state_dict()[key].float()
            for i in range(len(client_models))
        ], 0).mean(0)
    
    global_model.load_state_dict(global_dict)
    return global_model


def federated_training(train_dataset, indices, num_rounds, device, verbose=True):
    """
    联邦学习训练主流程
    
    Args:
        train_dataset: 训练数据集
        indices: 样本索引
        num_rounds: 联邦轮数
        device: 设备
        verbose: 是否打印进度
    
    Returns:
        训练好的全局模型
    """
    # 创建客户端数据
    client_loaders = create_client_datasets(
        train_dataset, indices, Config.NUM_CLIENTS
    )
    
    # 初始化全局模型
    global_model = create_model().to(device)
    
    for round_idx in range(num_rounds):
        if verbose and (round_idx + 1) % 5 == 0:
            print(f"    Round {round_idx + 1}/{num_rounds}")
        
        # 客户端训练
        client_models = []
        for client_idx in range(Config.NUM_CLIENTS):
            # 复制全局模型
            client_model = copy.deepcopy(global_model)
            
            # 本地训练
            client_model = local_train(
                client_model,
                client_loaders[client_idx],
                Config.LOCAL_EPOCHS,
                Config.LEARNING_RATE,
                device
            )
            
            client_models.append(client_model)
        
        # FedAvg 聚合
        global_model = fed_avg(global_model, client_models)
    
    return global_model


# ============== 评估 ==============

def evaluate_model(model, test_loader, device):
    """
    评估模型，计算三个指标
    
    Args:
        model: 模型
        test_loader: 测试集加载器
        device: 设备
    
    Returns:
        dict: 包含三个指标的字典
    """
    model.eval()
    
    # 统计各组的正确/总数
    # 组定义: (label, place)
    # - (1, 1): Waterbird + Water (Aligned)
    # - (0, 0): Landbird + Land (Aligned)
    # - (1, 0): Waterbird + Land (Conflict)
    # - (0, 1): Landbird + Water (Conflict)
    
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    
    with torch.no_grad():
        for images, labels, places in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                place = places[i].item()
                pred = predicted[i].item()
                
                group = (label, place)
                group_total[group] += 1
                if pred == label:
                    group_correct[group] += 1
    
    # 计算各组的准确率
    def calc_acc(group):
        if group_total[group] == 0:
            return 0.0
        return group_correct[group] / group_total[group]
    
    # 三个指标
    # Target Relation Score = Acc(Waterbird + Water) = Acc(1, 1)
    target_relation_score = calc_acc((1, 1))
    
    # Retention Score = Acc(Landbird + Land) = Acc(0, 0)
    retention_score = calc_acc((0, 0))
    
    # Background Gap = |Aligned Acc - Conflict Acc|
    aligned_acc = (calc_acc((1, 1)) + calc_acc((0, 0))) / 2
    conflict_acc = (calc_acc((1, 0)) + calc_acc((0, 1))) / 2
    background_gap = abs(aligned_acc - conflict_acc)
    
    return {
        "Target Relation Score": target_relation_score,
        "Retention Score": retention_score,
        "Background Gap": background_gap,
        "group_stats": {
            "Waterbird+Water": (group_correct[(1,1)], group_total[(1,1)]),
            "Landbird+Land": (group_correct[(0,0)], group_total[(0,0)]),
            "Waterbird+Land": (group_correct[(1,0)], group_total[(1,0)]),
            "Landbird+Water": (group_correct[(0,1)], group_total[(0,1)]),
        }
    }


# ============== 主实验 ==============

def run_single_seed(seed, train_dataset, test_dataset, device):
    """
    运行单个种子的完整实验
    
    Args:
        seed: 随机种子
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        device: 设备
    
    Returns:
        dict: 三个模型的结果
    """
    set_seed(seed)
    print(f"\n{'='*50}")
    print(f"Seed: {seed}")
    print(f"{'='*50}")
    
    # 测试集 DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    results = {}
    
    # ============ Model A: M_global (原始模型) ============
    print("\n[1/3] Training M_global (Original Model)...")
    original_indices = list(range(len(train_dataset)))
    
    m_global = federated_training(
        train_dataset,
        original_indices,
        Config.NUM_ROUNDS,
        device,
        verbose=True
    )
    
    results["M_global"] = evaluate_model(m_global, test_loader, device)
    print(f"    Target Relation Score: {results['M_global']['Target Relation Score']:.4f}")
    print(f"    Retention Score: {results['M_global']['Retention Score']:.4f}")
    print(f"    Background Gap: {results['M_global']['Background Gap']:.4f}")
    
    # ============ Model B: M_target_enhance (增强目标关系) ============
    print("\n[2/3] Training M_target_enhance (Enhanced Target Relation)...")
    enhanced_indices = create_enhanced_dataset(train_dataset, enhance_ratio=3.0)
    
    m_enhance = federated_training(
        train_dataset,
        enhanced_indices,
        Config.NUM_ROUNDS,
        device,
        verbose=True
    )
    
    results["M_target_enhance"] = evaluate_model(m_enhance, test_loader, device)
    print(f"    Target Relation Score: {results['M_target_enhance']['Target Relation Score']:.4f}")
    print(f"    Retention Score: {results['M_target_enhance']['Retention Score']:.4f}")
    print(f"    Background Gap: {results['M_target_enhance']['Background Gap']:.4f}")
    
    # ============ Model C: M_target_reduce (削弱目标关系) ============
    print("\n[3/3] Training M_target_reduce (Reduced Target Relation)...")
    reduced_indices = create_reduced_dataset(train_dataset)
    
    m_reduce = federated_training(
        train_dataset,
        reduced_indices,
        Config.NUM_ROUNDS,
        device,
        verbose=True
    )
    
    results["M_target_reduce"] = evaluate_model(m_reduce, test_loader, device)
    print(f"    Target Relation Score: {results['M_target_reduce']['Target Relation Score']:.4f}")
    print(f"    Retention Score: {results['M_target_reduce']['Retention Score']:.4f}")
    print(f"    Background Gap: {results['M_target_reduce']['Background Gap']:.4f}")
    
    return results


def print_summary(all_results):
    """打印汇总结果"""
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    for seed, results in all_results.items():
        print(f"\n{'─'*70}")
        print(f"Seed: {seed}")
        print(f"{'─'*70}")
        print(f"{'模型':<25} {'Target Score':<15} {'Retention':<15} {'BG Gap':<15}")
        print(f"{'─'*70}")
        
        for model_name, metrics in results.items():
            print(f"{model_name:<25} {metrics['Target Relation Score']:<15.4f} "
                  f"{metrics['Retention Score']:<15.4f} {metrics['Background Gap']:<15.4f}")
    
    # 跨种子平均
    print(f"\n{'='*70}")
    print("跨种子平均")
    print(f"{'='*70}")
    
    avg_results = {}
    for model_name in ["M_global", "M_target_enhance", "M_target_reduce"]:
        avg_target = np.mean([r[model_name]["Target Relation Score"] 
                             for r in all_results.values()])
        avg_retention = np.mean([r[model_name]["Retention Score"] 
                                for r in all_results.values()])
        avg_gap = np.mean([r[model_name]["Background Gap"] 
                          for r in all_results.values()])
        avg_results[model_name] = {
            "Target Relation Score": avg_target,
            "Retention Score": avg_retention,
            "Background Gap": avg_gap
        }
    
    print(f"{'模型':<25} {'Target Score':<15} {'Retention':<15} {'BG Gap':<15}")
    print(f"{'─'*70}")
    for model_name, metrics in avg_results.items():
        print(f"{model_name:<25} {metrics['Target Relation Score']:<15.4f} "
              f"{metrics['Retention Score']:<15.4f} {metrics['Background Gap']:<15.4f}")
    
    # 验证假设
    print(f"\n{'='*70}")
    print("假设验证")
    print(f"{'='*70}")
    
    # 检查 Target Relation Score 是否满足: 增强 > 原始 > 削弱
    target_scores = [avg_results[m]["Target Relation Score"] 
                    for m in ["M_target_enhance", "M_global", "M_target_reduce"]]
    
    if target_scores[0] > target_scores[1] > target_scores[2]:
        print("✓ Target Relation Score 满足假设: 增强 > 原始 > 削弱")
        print(f"  {target_scores[0]:.4f} > {target_scores[1]:.4f} > {target_scores[2]:.4f}")
    else:
        print("✗ Target Relation Score 不满足假设")
        print(f"  增强={target_scores[0]:.4f}, 原始={target_scores[1]:.4f}, 削弱={target_scores[2]:.4f}")
    
    # 检查 Retention Score 是否稳定
    retention_stds = [np.std([r[m]["Retention Score"] for r in all_results.values()]) 
                     for m in ["M_global", "M_target_enhance", "M_target_reduce"]]
    avg_retention_std = np.mean(retention_stds)
    
    if avg_retention_std < 0.05:
        print(f"✓ Retention Score 稳定 (跨模型标准差={avg_retention_std:.4f} < 0.05)")
    else:
        print(f"✗ Retention Score 不够稳定 (跨模型标准差={avg_retention_std:.4f})")
    
    return avg_results


def main():
    """主函数"""
    print("="*70)
    print("关系可观测性验证实验 (Relation Observability Experiment)")
    print("="*70)
    print(f"设备: {Config.DEVICE}")
    print(f"模型: {Config.MODEL_NAME}")
    print(f"联邦配置: {Config.NUM_CLIENTS} clients, {Config.NUM_ROUNDS} rounds")
    print(f"种子: {Config.SEEDS}")
    
    # 检查数据路径
    if not os.path.exists(Config.DATA_DIR):
        print(f"\n错误: 数据路径不存在: {Config.DATA_DIR}")
        print("请确保 Google Drive 已挂载，且数据集路径正确。")
        print("Colab 中挂载 Drive:")
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        return
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = WaterbirdsDataset(
        Config.DATA_DIR,
        split="train",
        transform=Config.TRAIN_TRANSFORM
    )
    test_dataset = WaterbirdsDataset(
        Config.DATA_DIR,
        split="test",
        transform=Config.EVAL_TRANSFORM
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 打印数据集统计
    print("\n训练集统计:")
    train_stats = get_dataset_stats(train_dataset)
    for k, v in sorted(train_stats.items()):
        print(f"  {k}: {v}")
    
    print("\n测试集统计:")
    test_stats = get_dataset_stats(test_dataset)
    for k, v in sorted(test_stats.items()):
        print(f"  {k}: {v}")
    
    # 运行所有种子
    all_results = {}
    for seed in Config.SEEDS:
        results = run_single_seed(seed, train_dataset, test_dataset, Config.DEVICE)
        all_results[seed] = results
    
    # 打印汇总
    avg_results = print_summary(all_results)
    
    # 保存结果
    output_file = "/content/drive/MyDrive/experiment_results/relation_observability_results.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("关系可观测性验证实验结果\n")
        f.write("="*70 + "\n\n")
        
        for seed, results in all_results.items():
            f.write(f"Seed: {seed}\n")
            f.write("-"*70 + "\n")
            for model_name, metrics in results.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Target Relation Score: {metrics['Target Relation Score']:.4f}\n")
                f.write(f"  Retention Score: {metrics['Retention Score']:.4f}\n")
                f.write(f"  Background Gap: {metrics['Background Gap']:.4f}\n")
                f.write(f"  Group Stats: {metrics['group_stats']}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("跨种子平均\n")
        f.write("="*70 + "\n")
        for model_name, metrics in avg_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Target Relation Score: {metrics['Target Relation Score']:.4f}\n")
            f.write(f"  Retention Score: {metrics['Retention Score']:.4f}\n")
            f.write(f"  Background Gap: {metrics['Background Gap']:.4f}\n\n")
    
    print(f"\n结果已保存: {output_file}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
