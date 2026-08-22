"""
Colab 启动脚本

使用方法：
1. 在 Colab 中运行此脚本
2. 先挂载 Google Drive
3. 将 experiment.py 上传到 Colab 或从 Drive 加载
"""

# ============== Step 1: 挂载 Google Drive ==============
from google.colab import drive
drive.mount('/content/drive')
print("✓ Google Drive 已挂载")

# ============== Step 2: 检查 GPU ==============
import torch
print(f"\n✓ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU 数量: {torch.cuda.device_count()}")

# ============== Step 3: 安装依赖（如需要）=============
# 通常 Colab 已预装，如需安装取消注释
# !pip install pandas torchvision pillow

# ============== Step 4: 准备实验代码 ==============
import os

# 方式 A: 如果 experiment.py 在 Google Drive 中
EXPERIMENT_FILE = "/content/drive/MyDrive/code/验证关系可测量可量化实验/experiment.py"

if os.path.exists(EXPERIMENT_FILE):
    print(f"\n✓ 从 Drive 加载实验代码: {EXPERIMENT_FILE}")
    # 读取并执行
    with open(EXPERIMENT_FILE, 'r', encoding='utf-8') as f:
        code = f.read()
    exec(code)
else:
    # 方式 B: 手动上传 experiment.py 到 /content/
    UPLOAD_FILE = "/content/experiment.py"
    if os.path.exists(UPLOAD_FILE):
        print(f"\n✓ 从本地加载实验代码: {UPLOAD_FILE}")
        with open(UPLOAD_FILE, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code)
    else:
        print(f"\n✗ 未找到实验代码")
        print(f"请上传 experiment.py 到以下路径之一:")
        print(f"  1. Google Drive: {EXPERIMENT_FILE}")
        print(f"  2. Colab 本地: {UPLOAD_FILE}")

# ============== Step 5: 检查数据集 ==============
DATA_DIR = "/content/drive/MyDrive/datasets/waterbird_complete95_forest2water2"
if not os.path.exists(DATA_DIR):
    print(f"\n⚠ 数据集路径不存在: {DATA_DIR}")
    print("请确保:")
    print("  1. 数据集已上传到 Google Drive")
    print("  2. 路径正确: MyDrive/datasets/waterbird_complete95_forest2water2")
    print("  3. 包含 metadata.csv 和图片文件")
else:
    print(f"\n✓ 数据集路径正确: {DATA_DIR}")
    # 检查 metadata.csv
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    if os.path.exists(metadata_path):
        print(f"✓ metadata.csv 存在")
    else:
        print(f"✗ metadata.csv 不存在")

# ============== Step 6: 运行实验 ==============
print("\n" + "="*70)
print("准备运行实验...")
print("="*70)
print("\n如果以上检查都通过，实验将自动开始。")
print("预计运行时间: 30-60 分钟（取决于 GPU）")
print("\n结果将保存到:")
print("  /content/drive/MyDrive/experiment_results/relation_observability_results.txt")
