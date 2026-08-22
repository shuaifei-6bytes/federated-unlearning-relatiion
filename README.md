# Relation-level vs Feature-level Unlearning 实验

验证 **Feature-level Unlearning（Ferrari 式）无法解决 Relation-level Unlearning 场景**，会产生过度遗忘（Over-unlearning）。

## 核心假设

```
同一个特征（水背景）同时参与两条语义关系：

              ┌──────────────┐
              │    水背景      │
              └──────────────┘
                    |
      ┌─────────────┴─────────────┐
      ↓                           ↓
 水背景 → 水鸟类别            水背景 → 环境类别
 ❌ 错误关系（需删除）         ✅ 合理关系（需保留）
```

**Feature ≠ Relation**：删除整个 Feature 的影响 ≠ 删除某一条错误 Relation。
Feature-level 方法只能整体削弱"水背景"，无法区分两条关系 → 过度遗忘。

## 三种方法

| 方法 | 说明 |
|------|------|
| `M_global` | FedAvg 正常联邦训练（遗忘前基线） |
| `M_feature` | Ferrari 式 feature-level unlearning：最小化 feature sensitivity（对输入加高斯噪声扰动，最小化输出对扰动的敏感度） |
| `M_relation` | relation-level unlearning：冻结 backbone + 环境 head，只用均衡数据重训鸟分类 head |

## 模型与数据

- 数据集：Waterbirds（waterbird_complete95_forest2water2，官方自动下载）
- 模型：共享 ResNet18 backbone + 双 head（`head_bird` 鸟分类 / `head_env` 环境分类）
- 联邦：5 客户端 FedAvg

## 评价指标

1. **Background Gap**：目标关系遗忘能力（越低越好）= aligned 组鸟分类 acc − conflicting 组鸟分类 acc
2. **Environment Accuracy**：非目标关系保持能力（`M_relation` 应保持，`M_feature` 应明显下降）
3. **Bird Accuracy**：正常分类能力保持

## 运行

### Google Colab（推荐，有免费 GPU）

直接打开 `colab_experiment.ipynb`，Run all 即可。数据自动下载，CUDA 自动适配。

### 本地

```bash
pip install -r requirements.txt
python experiment.py --data_dir ./data --global_epochs 20 --unlearn_epochs 5
```

快速冒烟（小样本少轮次，验证流程）：

```bash
python experiment.py --data_dir ./data --quick
```

## 输出

```
results/
├── M_global_results.json
├── M_feature_results.json
├── M_relation_results.json
└── comparison_table.csv
```

## 预期结论

- 若 `M_feature` 的 gap 下降但 env_acc 明显下降 → feature-level 过度遗忘成立
- 若 `M_relation` 的 gap 下降且 env_acc 保持 → relation-level 精准遗忘成立
