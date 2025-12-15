# 特征提取工具更新总结

## ✅ 更新完成

### 新增特征

#### 1. **Average Curvature (平均曲率)**
- **含义**: Polygon的形状复杂度
- **单位**: 度 (°)
- **范围**: 0-180°
- **计算**: 连续边之间的角度变化
- **用途**: 量化curved text的弯曲程度

#### 2. **Gaussian RMS Contrast (高斯RMS对比度)**
- **含义**: 基于Gaussian-weighted的局部对比度能量
- **范围**: 0-1
- **理论**: Zuiderbaan et al., 2017
- **参数**: σ = 19.4 pixels
- **用途**: 更准确地反映视觉感知的对比度

### 特征重命名

为了更清晰的命名：
- `avg_local_contrast` → `luminance_std` (亮度标准差)
- `avg_luminance` → `luminance_mean` (平均亮度)

---

## 📊 完整特征列表

现在CSV包含以下列：

```csv
image_name,text,polygon_size,edge_density,luminance_std,luminance_mean,avg_curvature,gaussian_rms_contrast
```

| 特征 | 含义 | 单位/范围 |
|------|------|-----------|
| image_name | 图像路径 | - |
| text | 文本内容 | - |
| polygon_size | 区域面积 | px² |
| edge_density | 边缘密度 | 0-1 |
| luminance_std | 亮度标准差 | 0-255 |
| luminance_mean | 平均亮度 | 0-255 |
| avg_curvature | 平均曲率 🆕 | 0-180° |
| gaussian_rms_contrast | 高斯RMS对比度 🆕 | 0-1 |

---

## 🔧 更新的文件

### 主要脚本
1. ✅ `extract_text_region_features.py`
   - 新增 `compute_polygon_curvature()` 函数
   - 新增 `compute_local_contrast_energy()` 函数
   - 更新特征提取和输出

2. ✅ `analyze_features.py`
   - 更新所有特征分析
   - 更新难度分级算法（考虑曲率）
   - 更新统计输出

### 文档
3. ✅ `NEW_FEATURES_GUIDE.md` - 详细的特征说明和使用指南
4. ✅ `UPDATE_SUMMARY.md` - 本文档

---

## 🚀 使用方法

### 提取特征
```bash
cd /cis/home/qgao14/my_documents/VIOCR_infer_models
python untils/eval/extract_text_region_features.py
```

**输出**: `results/totaltext_16_text_region_features.csv`

### 分析特征
```bash
python untils/eval/analyze_features.py
```

**输出**:
- `results/totaltext_16_features_with_difficulty.csv` (带难度标记)
- `results/totaltext_16_image_level_features.csv` (图像级统计)
- 终端输出详细分析报告

---

## 📈 特征解读示例

### Curvature (曲率)

```python
if avg_curvature < 20:
    # 近似矩形，规则文本
    difficulty = "Easy"
elif avg_curvature < 60:
    # 中等弯曲，典型curved text
    difficulty = "Medium"
else:
    # 高度弯曲，极具挑战
    difficulty = "Hard"
```

### Gaussian RMS Contrast (对比度)

```python
if gaussian_rms_contrast < 0.1:
    # 极低对比度，难以识别
    quality = "Poor"
elif gaussian_rms_contrast < 0.2:
    # 中等对比度，可识别
    quality = "Fair"
else:
    # 高对比度，清晰可辨
    quality = "Good"
```

---

## 🎯 应用场景

### 1. 识别困难样本
```python
df = pd.read_csv('results/totaltext_16_features_with_difficulty.csv')

# 困难样本：低对比度 + 高曲率
hard = df[
    (df['gaussian_rms_contrast'] < 0.1) & 
    (df['avg_curvature'] > 60)
]

print(f"Found {len(hard)} hard samples")
print(hard[['text', 'image_name', 'gaussian_rms_contrast', 'avg_curvature']])
```

### 2. 曲率分布分析
```python
# 按曲率分组
low_curv = df[df['avg_curvature'] < 20]
med_curv = df[(df['avg_curvature'] >= 20) & (df['avg_curvature'] < 60)]
high_curv = df[df['avg_curvature'] >= 60]

print(f"Straight text: {len(low_curv)} ({len(low_curv)/len(df)*100:.1f}%)")
print(f"Curved text: {len(med_curv)} ({len(med_curv)/len(df)*100:.1f}%)")
print(f"Highly curved: {len(high_curv)} ({len(high_curv)/len(df)*100:.1f}%)")
```

### 3. 对比度与识别性能
```python
# 合并特征和识别结果
features = pd.read_csv('results/totaltext_16_features_with_difficulty.csv')
eval_results = pd.read_excel('results/excel_outputs/Sub002_evaluation.xlsx', 
                             sheet_name='Word Pairs')

merged = features.merge(eval_results, 
                       left_on=['image_name', 'text'],
                       right_on=['Image Name', 'GT Text'])

# 分析对比度 vs 识别准确率
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(merged['gaussian_rms_contrast'], merged['Char F1'], alpha=0.5)
plt.xlabel('Gaussian RMS Contrast')
plt.ylabel('Character F1 Score')
plt.title('Contrast vs Recognition Accuracy')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 💡 关键洞察

### Curvature的重要性
- TotalText数据集的核心挑战是curved text
- 曲率直接影响检测和识别难度
- 高曲率样本需要specialized models

### Gaussian RMS Contrast的优势
- 比简单的标准差更符合人类视觉感知
- 考虑了空间局部性
- 基于视觉科学研究（Zuiderbaan et al., 2017）

### 特征组合
最难识别的样本通常具有：
- ✅ 低Gaussian RMS contrast (< 0.1)
- ✅ 高curvature (> 60°)
- ✅ 低edge density (< 0.2)
- ✅ 极端luminance (< 60 或 > 200)

---

## 📚 依赖项

确保安装：
```bash
pip install scipy  # 新增依赖，用于gaussian_filter
```

其他依赖保持不变：
```bash
pip install numpy pandas opencv-python shapely pillow tqdm
```

---

## ✅ 检查清单

运行前确认：
- ✅ 已安装 `scipy`
- ✅ `data/totaltext/anno.json` 存在
- ✅ `data/totaltext/16/` 目录存在且包含图像
- ✅ 有足够的磁盘空间存储结果

运行后验证：
- ✅ CSV包含新的 `avg_curvature` 列
- ✅ CSV包含新的 `gaussian_rms_contrast` 列
- ✅ 列名改为 `luminance_std` 和 `luminance_mean`
- ✅ 数值在合理范围内

---

## 🎉 完成！

所有更新已完成，您现在可以：

1. ✅ 提取更丰富的文本区域特征
2. ✅ 量化curved text的弯曲程度
3. ✅ 使用更准确的对比度度量
4. ✅ 更精确地评估样本难度
5. ✅ 进行更深入的数据分析

详细说明请查看：`NEW_FEATURES_GUIDE.md`

开始您的特征分析之旅！🚀









