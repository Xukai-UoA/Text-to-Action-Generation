# 训练可视化和监控工具使用指南

本指南介绍如何使用增强版Seq2Seq模型的训练可视化和监控工具。

---

## 概述

我们提供了三个工具来帮助您监控和分析训练过程：

| 工具 | 文件 | 功能 | 使用场景 |
|------|------|------|----------|
| **可视化工具** | `visualize_training.py` | 绘制损失曲线和epsilon变化 | 训练完成后分析，或实时监控 |
| **命令行监控** | `monitor_training.py` | 终端显示训练进度和统计 | 训练过程中实时查看状态 |
| **最优模型保存** | 训练器内置 | 自动保存loss最低的模型 | 自动运行，无需手动操作 |

---

## 功能1: 训练损失可视化

### 基本用法

训练完成后生成可视化图表：

```bash
# 基本用法（从默认路径加载）
python visualize_training.py

# 指定训练历史文件路径
python visualize_training.py --history_path ./seq2seq_enhanced_model/training_history.npz

# 保存图片到指定目录
python visualize_training.py --output ./plots/

# 只保存不显示
python visualize_training.py --output ./plots/ --no_show
```

### 高级用法

#### 1. 实时监控模式

在训练过程中自动刷新可视化：

```bash
# 每10秒刷新一次（默认）
python visualize_training.py --live

# 自定义刷新间隔（5秒）
python visualize_training.py --live --refresh_interval 5

# 指定模型目录
python visualize_training.py --live --model_dir ./seq2seq_enhanced_model
```

**效果**：
- 自动加载最新的训练数据
- 图表实时更新
- 显示最优模型位置（如果有）

#### 2. 仅绘制损失对比图

生成简洁的损失对比图（不含epsilon和统计信息）：

```bash
python visualize_training.py --comparison_only --output ./plots/
```

### 输出说明

运行后会生成以下图表：

**完整版 (`training_curves.png`)**:
- 总损失曲线（带最优模型标注）
- 动作重建损失
- 文本重建损失
- Scheduled Sampling epsilon变化
- 训练统计信息框

**简洁版 (`loss_comparison.png`)**:
- 三条损失曲线对比（总损失、动作损失、文本损失）

---

## 功能2: 命令行实时监控

### 基本用法

在训练过程中查看实时进度：

```bash
# 监控默认目录
python monitor_training.py

# 指定模型目录
python monitor_training.py --model_dir ./seq2seq_enhanced_model

# 自定义刷新间隔（默认5秒）
python monitor_training.py --refresh 3

# 指定最大epoch数（用于进度条）
python monitor_training.py --max_epoch 500
```

### 显示内容

监控工具会显示：

```
================================================================================
                    训练监控 - Enhanced Seq2Seq Model
================================================================================
更新时间: 2025-11-20 15:30:45
================================================================================

📊 训练进度
--------------------------------------------------------------------------------
   [███████████████████░░░░░░░░░░] 65.4%
   Epoch: 327 / 500
   预估剩余时间: 2小时15分
   平均epoch时间: 23秒

📉 当前损失 (Epoch 327)
--------------------------------------------------------------------------------
   总损失:     0.023456
   动作损失:   0.015678
   文本损失:   0.001556
   Teacher Forcing Ratio (ε): 0.500

⭐ 最优模型
--------------------------------------------------------------------------------
   Epoch:      285
   总损失:     0.021234
   动作损失:   0.014567
   文本损失:   0.001334
   📈 当前loss比最优高 10.47%

📈 损失趋势 (最近10个epoch)
--------------------------------------------------------------------------------
   ✅ 下降趋势: 0.001234

   -10: ████████████████████████████ 0.0245
   - 9: ███████████████████████████ 0.0243
   - 8: ██████████████████████████ 0.0241
   - 7: █████████████████████████ 0.0239
   - 6: ████████████████████████ 0.0237
   - 5: ███████████████████████ 0.0235
   - 4: ██████████████████████ 0.0236
   - 3: █████████████████████ 0.0234
   - 2: ████████████████████ 0.0236
   - 1: ███████████████████ 0.0235

💾 已保存的模型
--------------------------------------------------------------------------------
   检查点数量: 65
   ✓ best_model.pth (Epoch 285)

================================================================================
下次更新: 5秒后 | 按 Ctrl+C 退出
================================================================================
```

---

## 功能3: 最优模型自动保存

### 工作原理

增强版训练器会自动：

1. **追踪最优loss**: 在每个epoch结束时比较当前loss和历史最优loss
2. **保存最优模型**: 当发现更低的loss时，自动保存为 `best_model.pth`
3. **保留检查点**: 定期检查点（每5个epoch）仍然保存为 `model_epoch_*.pth`

### 文件结构

训练完成后，模型目录包含：

```
seq2seq_enhanced_model/
├── best_model.pth                 # 最优模型（loss最低）⭐
├── model_epoch_5.pth              # 定期检查点
├── model_epoch_10.pth
├── model_epoch_15.pth
├── ...
├── model_epoch_500.pth            # 最后一个epoch
└── training_history.npz           # 训练历史数据
```

### 加载最优模型

```python
import torch
from model.seq2seq_enhanced_structure import EnhancedSeq2SeqModel

# 创建模型
model = EnhancedSeq2SeqModel(...)

# 加载最优模型
checkpoint = torch.load('./seq2seq_enhanced_model/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 查看最优模型信息
print(f"最优模型来自 Epoch {checkpoint['epoch']}")
print(f"总损失: {checkpoint['loss']:.6f}")
print(f"动作损失: {checkpoint['action_loss']:.6f}")
print(f"文本损失: {checkpoint['char_loss']:.6f}")
```

### 训练输出示例

训练过程中会显示：

```
Epoch 150 完成:
  平均损失: 0.034567 (Action: 0.023456, Char: 0.002222)
  Teacher forcing ratio: 0.750
  🌟 新的最优模型! Loss: 0.034567
  ✓ 定期检查点已保存: ./seq2seq_enhanced_model/model_epoch_150.pth
  ✓ 最优模型已保存: ./seq2seq_enhanced_model/best_model.pth
```

---

## 完整工作流示例

### 场景1: 开始新训练并监控

**终端1** - 启动训练:
```bash
python seq2seq_enhanced_pretrain.py
```

**终端2** - 命令行监控:
```bash
python monitor_training.py --model_dir ./seq2seq_enhanced_model
```

**终端3** - 可视化监控（可选）:
```bash
python visualize_training.py --live --model_dir ./seq2seq_enhanced_model
```

### 场景2: 训练完成后分析

```bash
# 生成完整可视化并保存
python visualize_training.py \
    --history_path ./seq2seq_enhanced_model/training_history.npz \
    --output ./analysis/

# 查看生成的图片
ls ./analysis/
# training_curves.png

# 如果想要简洁版
python visualize_training.py --comparison_only --output ./analysis/
```

### 场景3: 从检查点恢复训练

修改 `seq2seq_enhanced_pretrain.py`:

```python
restore = 1
restore_path = './seq2seq_enhanced_model/model_epoch_250.pth'
restore_step = 250
```

运行：
```bash
python seq2seq_enhanced_pretrain.py
```

训练器会自动：
- 加载模型权重
- 恢复优化器状态
- **恢复最优loss记录**（不会被重置）

---

## 可视化图表详解

### 1. 总损失曲线（主图）

- **蓝色曲线**: 总损失随epoch的变化
- **填充区域**: 曲线下方阴影，便于观察趋势
- **红色星标**: 最优模型位置
- **红色虚线**: 最优模型所在epoch的垂直线
- **标注框**: 显示最优loss值和epoch

### 2. 动作损失曲线

- **紫色曲线**: 动作重建损失（MSE）
- 较低的值表示生成的动作更接近真实动作

### 3. 文本损失曲线

- **橙色曲线**: 文本重建损失（MSE × 5）
- 用于cycle-consistency约束
- 较低的值表示模型能够从动作反推回原始文本

### 4. Epsilon曲线（Scheduled Sampling）

- **绿色曲线**: Teacher forcing ratio
- **阶段标注**: 显示衰减的开始和结束点
- **参考线**: 1.0和0.5的水平虚线

**解读**:
- ε = 1.0: 完全teacher forcing（使用真实输入）
- ε = 0.5: 50%混合（一半真实输入，一半模型预测）
- ε = 0.0: 完全自回归（仅使用模型预测）

### 5. 统计信息框

左下角显示：
- 总epoch数
- 最终loss
- 最小loss
- 最优模型详细信息

---

## 高级技巧

### 1. 并行运行多个实验

```bash
# 实验1: 标准配置
python seq2seq_enhanced_pretrain.py

# 实验2: 不使用Scheduled Sampling（修改脚本后）
# 在seq2seq_enhanced_pretrain.py中设置: use_scheduled_sampling = False
python seq2seq_enhanced_pretrain.py

# 分别可视化
python visualize_training.py --history_path ./exp1/training_history.npz --output ./plots/exp1/
python visualize_training.py --history_path ./exp2/training_history.npz --output ./plots/exp2/
```

### 2. 对比多个实验

创建自定义脚本：

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载多个实验的数据
exp1 = np.load('./exp1/training_history.npz')
exp2 = np.load('./exp2/training_history.npz')

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(exp1['epochs'], exp1['total_losses'], label='With Scheduled Sampling')
plt.plot(exp2['epochs'], exp2['total_losses'], label='Without Scheduled Sampling')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.legend()
plt.title('Ablation Study: Scheduled Sampling')
plt.savefig('./comparison.png', dpi=300)
```

### 3. 导出训练数据到CSV

```python
import numpy as np
import pandas as pd

# 加载数据
data = np.load('./seq2seq_enhanced_model/training_history.npz')

# 转换为DataFrame
df = pd.DataFrame({
    'epoch': data['epochs'],
    'total_loss': data['total_losses'],
    'action_loss': data['action_losses'],
    'char_loss': data['char_losses'],
    'epsilon': data['epsilons']
})

# 保存为CSV
df.to_csv('./training_history.csv', index=False)
```

### 4. 使用Jupyter Notebook

```python
# notebook中使用
from visualize_training import load_training_history, plot_training_curves, find_best_model

# 加载数据
history = load_training_history('./seq2seq_enhanced_model/training_history.npz')
best_model = find_best_model('./seq2seq_enhanced_model')

# 绘制（在notebook中显示）
plot_training_curves(history, best_model, output_dir=None, show=True)
```

---

## 常见问题

### Q1: 可视化工具提示"找不到matplotlib"

**A**: 安装matplotlib：
```bash
pip install matplotlib numpy
```

### Q2: 实时监控模式刷新很慢

**A**: 减少刷新间隔：
```bash
python visualize_training.py --live --refresh_interval 5
```

但注意过于频繁的刷新可能影响训练性能。

### Q3: 如何只查看最近的训练数据？

**A**: 修改代码或使用切片：

```python
# 在visualize_training.py中修改
history = load_training_history(args.history_path)

# 只取最后100个epoch
for key in history:
    history[key] = history[key][-100:]

plot_training_curves(history, ...)
```

### Q4: 最优模型没有保存

**A**: 检查：
1. 是否使用了增强版训练器（`seq2seq_enhanced_pretrain.py`）
2. 训练是否已经开始（至少1个epoch完成）
3. 模型目录权限是否正确

### Q5: 训练中断后如何继续监控？

**A**: 可视化工具会自动读取最新数据，直接运行即可：
```bash
python visualize_training.py
```

如果从检查点恢复训练，最优模型记录会自动恢复。

---

## 性能建议

### 1. 减少磁盘IO

训练过程中频繁保存会影响性能，可以调整：

```python
# 在seq2seq_enhanced_pretrain.py中
save_stride = 10  # 从5改为10，减少保存频率
```

### 2. 实时监控的刷新间隔

推荐设置：
- 快速测试: 5秒
- 正式训练: 10-30秒
- 长时间训练: 60秒

### 3. 图片分辨率

如果生成图片文件太大，可以在 `visualize_training.py` 中修改：

```python
# 找到这行
plt.savefig(output_path, dpi=300, ...)  # 默认300 dpi

# 改为
plt.savefig(output_path, dpi=150, ...)  # 降低到150 dpi
```

---

## 最佳实践

### ✅ 推荐做法

1. **训练前**: 准备好监控工具
   ```bash
   # 启动监控（另一个终端）
   python monitor_training.py
   ```

2. **训练中**: 定期检查可视化
   ```bash
   # 每隔1-2小时生成一次可视化
   python visualize_training.py --output ./checkpoints/
   ```

3. **训练后**: 完整分析
   ```bash
   # 生成完整报告
   python visualize_training.py --output ./final_report/
   ```

4. **对比实验**: 保留所有训练历史
   ```bash
   # 复制训练历史到实验目录
   cp ./seq2seq_enhanced_model/training_history.npz ./experiments/exp1/
   ```

### ❌ 避免做法

1. ❌ 训练过程中频繁运行 `--live` 模式（影响性能）
2. ❌ 删除 `training_history.npz`（无法恢复）
3. ❌ 修改 `best_model.pth` 文件名（工具无法识别）
4. ❌ 在训练脚本中修改 `best_loss` 逻辑（可能破坏追踪）

---

## 故障排除

### 问题1: ModuleNotFoundError: No module named 'matplotlib'

**解决**:
```bash
pip install matplotlib numpy scipy
```

### 问题2: 可视化图表显示乱码（中文）

**解决**: 在 `visualize_training.py` 开头添加：
```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 或其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False
```

### 问题3: 实时监控终端无法清屏

**解决**: 某些终端不支持清屏命令，在 `monitor_training.py` 中注释掉：
```python
# os.system('clear' if os.name == 'posix' else 'cls')
```

### 问题4: 训练历史文件损坏

**解决**: 从最近的检查点恢复训练：
```python
restore = 1
restore_path = './seq2seq_enhanced_model/model_epoch_495.pth'
restore_step = 495
```

训练器会重新生成训练历史。

---

## 总结

| 任务 | 使用工具 | 命令 |
|------|----------|------|
| 训练完成后查看曲线 | `visualize_training.py` | `python visualize_training.py` |
| 训练中实时监控状态 | `monitor_training.py` | `python monitor_training.py` |
| 训练中实时查看曲线 | `visualize_training.py --live` | `python visualize_training.py --live` |
| 保存可视化图片 | `--output` | `python visualize_training.py --output ./plots/` |
| 查找最优模型 | - | 直接使用 `best_model.pth` |

---

**文档版本**: v1.0
**最后更新**: 2025-11-20
**维护者**: Claude Code
