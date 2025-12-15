# Detection Order 功能更新

## 📋 更新说明

在 `totaltext_eval_FINAL.py` 中新增了**Detection Order（检测顺序）**功能，用于记录每个单词在每张图中被human检测出来的顺序。

## 🎯 功能描述

### 什么是Detection Order？

Detection Order 是指每个单词在输入JSON文件中出现的顺序（从0开始计数）。这个顺序反映了：
- Human标注员标记单词的顺序
- 可能反映视觉扫描的顺序
- 可以用于分析是否存在位置偏见（position bias）

### 示例

假设一张图片的JSON数据如下：

```json
{
  "image_id": 123,
  "polys": [
    [[10, 20], [30, 20], [30, 40], [10, 40]],  // Detection Order = 0
    [[50, 60], [80, 60], [80, 90], [50, 90]],  // Detection Order = 1
    [[100, 30], [130, 30], [130, 50], [100, 50]] // Detection Order = 2
  ],
  "rec_texts": ["STORE", "OPEN", "NOW"]
}
```

那么：
- "STORE" 的 Detection Order = 0（第一个被检测）
- "OPEN" 的 Detection Order = 1（第二个被检测）
- "NOW" 的 Detection Order = 2（第三个被检测）

## 📊 Excel输出变化

### Sheet 1: Word Pairs

新增了 **"Detection Order"** 列，位于第二列（Image Name之后）：

| 列名 | 说明 | 示例 |
|------|------|------|
| Image Name | 图像文件路径 | `16/img_001.jpg` |
| **Detection Order** | **检测顺序（从0开始）** | **0, 1, 2, ...** |
| GT Text | Ground Truth文本 | `STORE` |
| Pred Text | 预测文本 | `STORE` |
| IoU | Polygon IoU | `0.8523` |
| Edit Distance | 编辑距离 | `0` |
| Word Match | 是否完全匹配 | `Yes` |
| Char F1 | 字符F1 | `1.0000` |

### 示例数据

```
Image Name      Detection Order   GT Text   Pred Text   IoU     Edit Dist   Word Match   Char F1
16/img_001.jpg  0                 STORE     STORE       0.8523  0           Yes          1.0000
16/img_001.jpg  1                 OPEN      OPEN        0.7891  0           Yes          1.0000
16/img_001.jpg  2                 NOW       NOW         0.8234  0           Yes          1.0000
16/img_002.jpg  0                 SALE      SALE        0.7654  0           Yes          1.0000
16/img_002.jpg  1                 TODAY     TODAY       0.8123  0           Yes          1.0000
```

## 🔍 应用场景

### 1. 位置偏见分析

检查识别准确率是否与检测顺序相关：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel
df = pd.read_excel('results/excel_outputs/Sub002_evaluation.xlsx', 
                   sheet_name='Word Pairs')

# 按Detection Order分组统计准确率
accuracy_by_order = df.groupby('Detection Order')['Word Match'].apply(
    lambda x: (x == 'Yes').sum() / len(x)
)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(accuracy_by_order.index, accuracy_by_order.values, marker='o')
plt.xlabel('Detection Order')
plt.ylabel('Accuracy')
plt.title('Recognition Accuracy vs Detection Order')
plt.grid(True, alpha=0.3)
plt.show()

print("Accuracy by Detection Order:")
print(accuracy_by_order)
```

### 2. 早期 vs 晚期检测对比

```python
# 将检测顺序分为早期和晚期
df['detection_phase'] = df['Detection Order'].apply(
    lambda x: 'Early' if x < 5 else 'Late'
)

# 对比准确率
phase_accuracy = df.groupby('detection_phase')['Word Match'].apply(
    lambda x: (x == 'Yes').sum() / len(x)
)

print("Early detections accuracy:", phase_accuracy['Early'])
print("Late detections accuracy:", phase_accuracy['Late'])
```

### 3. 疲劳效应分析

检查是否随着检测顺序增加，准确率下降（可能表示标注员疲劳）：

```python
from scipy.stats import spearmanr

# 将Word Match转换为数值
df['match_numeric'] = (df['Word Match'] == 'Yes').astype(int)

# 计算相关性
correlation, p_value = spearmanr(df['Detection Order'], df['match_numeric'])

print(f"Spearman correlation: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")

if correlation < -0.1 and p_value < 0.05:
    print("⚠️ 发现显著的负相关：检测顺序越后，准确率越低")
elif correlation > 0.1 and p_value < 0.05:
    print("✅ 发现显著的正相关：检测顺序越后，准确率越高（可能因为容易的词先被检测）")
else:
    print("✓ 无显著相关性：检测顺序不影响准确率")
```

### 4. 每张图的检测分布

```python
# 查看每张图有多少个检测
detections_per_image = df.groupby('Image Name')['Detection Order'].agg(['count', 'max'])
detections_per_image.columns = ['Num_Detections', 'Max_Order']
detections_per_image['Max_Order'] += 1  # 因为从0开始

print("Images with most detections:")
print(detections_per_image.nlargest(10, 'Num_Detections'))

# 统计
print(f"\nAverage detections per image: {detections_per_image['Num_Detections'].mean():.1f}")
print(f"Max detections in one image: {detections_per_image['Num_Detections'].max()}")
```

### 5. 特定顺序的错误分析

```python
# 找出在特定顺序位置上错误率最高的情况
df['is_error'] = (df['Word Match'] == 'No')

error_by_order = df.groupby('Detection Order').agg({
    'is_error': 'mean',
    'Image Name': 'count'
})
error_by_order.columns = ['Error_Rate', 'Count']

# 只看有足够样本的顺序位置
significant_orders = error_by_order[error_by_order['Count'] >= 10]
problematic_orders = significant_orders[significant_orders['Error_Rate'] > 0.3]

print("Problematic detection orders (error rate > 30%):")
print(problematic_orders)
```

## 🔬 研究价值

### 1. 标注质量评估
- **早期检测**: 通常是最显著的文本，应该有高准确率
- **晚期检测**: 可能是较小或难识别的文本，准确率可能降低
- **异常模式**: 如果早期检测准确率低，可能表示标注质量问题

### 2. 视觉扫描模式
- 分析人类视觉注意力的顺序
- 检测是否存在从左到右、从上到下的扫描模式
- 识别视觉显著性与检测顺序的关系

### 3. 模型性能评估
- 检查模型是否对某些位置的文本有偏见
- 评估模型是否能一致地处理所有位置的文本
- 发现可能的位置相关性能问题

### 4. 数据平衡
- 确保训练/测试集在不同检测顺序上的分布平衡
- 避免模型学习到与顺序相关的虚假相关性

## 📈 统计分析示例

### 完整分析脚本

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency

# 读取所有subjects的数据
all_subjects = []
for subject_file in Path('results/excel_outputs').glob('Sub*_evaluation.xlsx'):
    df = pd.read_excel(subject_file, sheet_name='Word Pairs')
    df['Subject'] = subject_file.stem.split('_')[0]
    all_subjects.append(df)

df_all = pd.concat(all_subjects, ignore_index=True)

print("="*80)
print("DETECTION ORDER ANALYSIS")
print("="*80)

# 1. 基本统计
print("\n【基本统计】")
print(f"总样本数: {len(df_all)}")
print(f"最大检测顺序: {df_all['Detection Order'].max()}")
print(f"平均每张图检测数: {df_all.groupby(['Subject', 'Image Name']).size().mean():.1f}")

# 2. 准确率 vs 检测顺序
print("\n【准确率 vs 检测顺序】")
df_all['is_correct'] = (df_all['Word Match'] == 'Yes')
accuracy_by_order = df_all.groupby('Detection Order')['is_correct'].mean()

# 只看有足够样本的顺序
valid_orders = df_all.groupby('Detection Order').size() >= 10
valid_accuracy = accuracy_by_order[valid_orders]

print(f"Detection Order 0-9的准确率:")
for i in range(min(10, len(valid_accuracy))):
    if i in valid_accuracy.index:
        print(f"  Order {i}: {valid_accuracy[i]:.1%} ({df_all[df_all['Detection Order']==i]['is_correct'].sum()}/{len(df_all[df_all['Detection Order']==i])})")

# 3. 相关性分析
correlation, p_value = spearmanr(df_all['Detection Order'], df_all['is_correct'])
print(f"\n【相关性分析】")
print(f"Spearman相关系数: {correlation:.3f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"{'⚠️ 显著负相关' if correlation < 0 else '✓ 显著正相关'}")
else:
    print("✓ 无显著相关性")

# 4. 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 4.1 准确率 vs 检测顺序
ax1 = axes[0, 0]
order_counts = df_all.groupby('Detection Order').size()
valid_orders = order_counts[order_counts >= 10].index
plot_data = accuracy_by_order[valid_orders]
ax1.plot(plot_data.index, plot_data.values, marker='o', linewidth=2)
ax1.set_xlabel('Detection Order')
ax1.set_ylabel('Accuracy')
ax1.set_title('Recognition Accuracy vs Detection Order')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=df_all['is_correct'].mean(), color='r', linestyle='--', 
            label=f'Overall Avg: {df_all["is_correct"].mean():.1%}')
ax1.legend()

# 4.2 检测顺序分布
ax2 = axes[0, 1]
order_dist = df_all['Detection Order'].value_counts().sort_index()
ax2.bar(order_dist.index[:20], order_dist.values[:20])
ax2.set_xlabel('Detection Order')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Detection Orders')
ax2.grid(True, alpha=0.3, axis='y')

# 4.3 Char F1 vs 检测顺序
ax3 = axes[1, 0]
char_f1_by_order = df_all.groupby('Detection Order')['Char F1'].mean()
plot_data = char_f1_by_order[valid_orders]
ax3.plot(plot_data.index, plot_data.values, marker='s', color='green', linewidth=2)
ax3.set_xlabel('Detection Order')
ax3.set_ylabel('Character F1')
ax3.set_title('Character F1 vs Detection Order')
ax3.grid(True, alpha=0.3)

# 4.4 早期 vs 晚期检测对比
ax4 = axes[1, 1]
df_all['phase'] = pd.cut(df_all['Detection Order'], 
                         bins=[0, 3, 7, 100], 
                         labels=['Early (0-2)', 'Mid (3-6)', 'Late (7+)'])
phase_stats = df_all.groupby('phase')['is_correct'].agg(['mean', 'count'])
ax4.bar(range(len(phase_stats)), phase_stats['mean'], 
        color=['green', 'orange', 'red'], alpha=0.7)
ax4.set_xticks(range(len(phase_stats)))
ax4.set_xticklabels(phase_stats.index)
ax4.set_ylabel('Accuracy')
ax4.set_title('Accuracy by Detection Phase')
ax4.grid(True, alpha=0.3, axis='y')

# 添加样本数标注
for i, (acc, count) in enumerate(zip(phase_stats['mean'], phase_stats['count'])):
    ax4.text(i, acc + 0.02, f'n={count}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/detection_order_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 可视化图表已保存: results/detection_order_analysis.png")

plt.show()
```

## ⚙️ 技术实现

### 代码变更

#### 1. 数据组织（Line 338-343）
```python
text = rec_texts[i] if i < len(rec_texts) else ''
det_by_image[unique_key].append({
    'bbox': polygon_points,
    'text': text,
    'detection_order': i  # 记录检测顺序
})
```

#### 2. 评估函数（Line 157, 201）
```python
detection_order = det.get('detection_order', -1)  # 获取检测顺序

matched_word_pairs.append({
    # ... 其他字段 ...
    'detection_order': detection_order  # 添加到结果中
})
```

#### 3. Excel输出（Line 465）
```python
word_pairs_data.append({
    'Image Name': pair['image_name'],
    'Detection Order': pair['detection_order'],  # 新增列
    'GT Text': pair['gt'],
    # ... 其他列 ...
})
```

## 📝 注意事项

1. **顺序从0开始**: Detection Order从0开始计数，第一个检测的单词order=0
2. **JSON顺序**: Order反映JSON文件中`polys`和`rec_texts`数组的顺序
3. **缺失值**: 如果某个detection没有order信息，将显示为-1
4. **Per-image**: 每张图的order独立计数，不同图的order=0表示各自的第一个检测

## ✅ 验证

### 检查更新是否成功

运行脚本后，打开任意Excel文件：

```bash
python untils/eval/totaltext_eval_FINAL.py
```

检查 `results/excel_outputs/Sub002_evaluation.xlsx` 的 Sheet 1:
- ✅ 应该看到第二列是 "Detection Order"
- ✅ 值应该是整数（0, 1, 2, ...）
- ✅ 同一张图的不同单词有不同的order
- ✅ 新图的order从0重新开始

## 🎉 总结

Detection Order功能让您能够：
- ✅ 追踪每个单词的检测顺序
- ✅ 分析位置偏见和疲劳效应
- ✅ 研究视觉扫描模式
- ✅ 评估标注质量
- ✅ 发现与顺序相关的性能问题

这个功能为深入分析人类标注行为和模型性能提供了新的维度！









