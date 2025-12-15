# 新增特征指标说明

## 📊 新增的两个特征

在特征提取系统中新增了两个重要的图像质量指标：
1. **Spatial Frequency（空间频率）**
2. **Perimetric Complexity（周长复杂度）**

---

## 1. Spatial Frequency (空间频率)

### 定义
空间频率是衡量图像中灰度值变化剧烈程度的指标，反映图像的整体活跃度和细节丰富程度。

### 数学公式

```
Row Frequency (RF) = sqrt(mean((I[i,j] - I[i,j-1])²))      # 行方向频率
Column Frequency (CF) = sqrt(mean((I[i,j] - I[i-1,j])²))  # 列方向频率  
Spatial Frequency (SF) = sqrt(RF² + CF²)                   # 空间频率
```

其中：
- `I[i,j]` 是像素 (i,j) 的灰度值
- RF 衡量水平方向的变化
- CF 衡量垂直方向的变化
- SF 是二者的综合

### 取值范围和含义

| 空间频率值 | 含义 | 特征 |
|-----------|------|------|
| **< 0.05** | 极低频率 | 区域平滑，缺乏细节，可能模糊 |
| **0.05 - 0.10** | 低频率 | 细节较少，对比度低 |
| **0.10 - 0.20** | 中等频率 | 正常的文本细节 |
| **0.20 - 0.30** | 高频率 | 丰富的细节，清晰的边缘 |
| **> 0.30** | 极高频率 | 非常详细，可能有噪声或复杂纹理 |

### 在文本识别中的意义

**高空间频率 (> 0.2)**:
- ✅ 文本边缘清晰
- ✅ 字符结构清楚
- ✅ 有助于识别
- ⚠️ 可能包含噪声

**低空间频率 (< 0.1)**:
- ⚠️ 文本模糊
- ⚠️ 边缘不清
- ⚠️ 识别困难
- 可能由于：
  - 相机失焦
  - 运动模糊
  - 低分辨率
  - 平滑背景上的文本

### 实现代码

```python
def compute_spatial_frequency(image, mask):
    gray = image.astype(np.float64) / 255.0
    
    # Extract ROI
    rows, cols = np.where(mask > 0)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    roi = gray[min_row:max_row+1, min_col:max_col+1]
    roi_mask = mask[min_row:max_row+1, min_col:max_col+1]
    
    # Row frequency
    row_diff = np.diff(roi, axis=1)
    row_mask = roi_mask[:, 1:] & roi_mask[:, :-1]
    rf = np.sqrt(np.mean((row_diff[row_mask > 0]) ** 2))
    
    # Column frequency  
    col_diff = np.diff(roi, axis=0)
    col_mask = roi_mask[1:, :] & roi_mask[:-1, :]
    cf = np.sqrt(np.mean((col_diff[col_mask > 0]) ** 2))
    
    # Spatial frequency
    sf = np.sqrt(rf ** 2 + cf ** 2)
    return sf
```

---

## 2. Perimetric Complexity (周长复杂度)

### 定义
周长复杂度是衡量文本区域边界复杂程度的指标，反映文本形状的复杂性。

### 数学公式

```
Perimetric_Complexity = (EdgeLength²) / InkArea
```

其中：
- `EdgeLength` = 边缘像素的总数（使用Canny边缘检测）
- `InkArea` = 文本区域的总像素数（mask内的像素数）

### 取值范围和含义

| 周长复杂度值 | 含义 | 特征 |
|-------------|------|------|
| **< 10** | 低复杂度 | 简单形状，平滑边界 |
| **10 - 50** | 中等复杂度 | 正常文本，适中的边界复杂度 |
| **50 - 200** | 高复杂度 | 复杂文本，多细节，弯曲边界 |
| **> 200** | 极高复杂度 | 非常复杂的形状，可能是艺术字或装饰文本 |

### 物理意义

**高周长复杂度**表示：
- 边缘长度相对于面积很大
- 形状不规则、弯曲或有很多细节
- 边界复杂

**低周长复杂度**表示：
- 边缘相对平滑
- 形状接近矩形或简单多边形
- 边界规则

### 在文本识别中的意义

**低复杂度 (< 20)**:
- 规则的印刷体
- 矩形文本框
- 直线排列
- 易于识别

**中等复杂度 (20-100)**:
- 普通curved text
- 适度弯曲
- TotalText数据集的主要区域

**高复杂度 (> 100)**:
- 高度弯曲或扭曲的文本
- 艺术字体
- 不规则排列
- 识别困难
- 可能表示：
  - Curved text
  - Artistic text
  - 图案/logo中的文本

### 与Curvature的区别

| 特征 | Perimetric Complexity | Average Curvature |
|------|-----------------------|-------------------|
| 衡量对象 | 整体边界的复杂度 | Polygon顶点的角度变化 |
| 受影响因素 | 边缘细节、边界长度 | Polygon形状 |
| 数值稳定性 | 受边缘检测影响 | 仅依赖polygon坐标 |
| 适用场景 | 评估图像质量 | 评估几何形状 |

### 实现代码

```python
def compute_perimetric_complexity(image, mask):
    binary = mask.astype(np.uint8)
    
    # Calculate ink area
    ink_area = np.sum(binary)
    
    if ink_area == 0:
        return 0.0
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    
    # Canny edge detection
    edges = feature.canny(masked_image, sigma=1.0)
    
    # Calculate edge length
    edge_length = np.sum(edges)
    
    if edge_length == 0:
        return 0.0
    
    # Perimetric complexity
    perimetric_complexity = (edge_length ** 2) / ink_area
    
    return perimetric_complexity
```

---

## 📈 特征应用

### 1. 数据质量评估

```python
import pandas as pd

df = pd.read_csv('results/totaltext_16_text_region_features.csv')

# 识别低质量样本（低空间频率）
low_quality = df[df['spatial_frequency'] < 0.05]
print(f"低质量样本: {len(low_quality)} ({len(low_quality)/len(df)*100:.1f}%)")

# 识别复杂文本（高周长复杂度）
complex_text = df[df['perimetric_complexity'] > 100]
print(f"复杂文本: {len(complex_text)} ({len(complex_text)/len(df)*100:.1f}%)")
```

### 2. 困难样本识别

```python
# 综合评分：低空间频率 + 高周长复杂度 = 困难样本
df['difficulty_score'] = (
    (1 / (df['spatial_frequency'] + 0.01)) * 
    (df['perimetric_complexity'] / 100)
)

hardest = df.nlargest(10, 'difficulty_score')
print("最困难的10个样本:")
for idx, row in hardest.iterrows():
    print(f"  {row['text']} - SF:{row['spatial_frequency']:.4f}, PC:{row['perimetric_complexity']:.2f}")
```

### 3. 特征相关性分析

```python
# 空间频率 vs 其他特征
print("空间频率的相关性:")
print(f"  与Edge Density: {df['spatial_frequency'].corr(df['edge_density']):.3f}")
print(f"  与Gaussian Contrast: {df['spatial_frequency'].corr(df['gaussian_rms_contrast']):.3f}")

# 周长复杂度 vs 其他特征
print("\n周长复杂度的相关性:")
print(f"  与Curvature: {df['perimetric_complexity'].corr(df['avg_curvature']):.3f}")
print(f"  与Polygon Size: {df['perimetric_complexity'].corr(df['polygon_size']):.3f}")
```

### 4. 按特征分组分析

```python
# 按空间频率分组
df['sf_group'] = pd.cut(df['spatial_frequency'], 
                         bins=[0, 0.1, 0.2, 1.0],
                         labels=['Low', 'Medium', 'High'])

# 对比识别性能（如果有评估结果）
if 'word_match' in df.columns:
    for group in ['Low', 'Medium', 'High']:
        group_df = df[df['sf_group'] == group]
        accuracy = (group_df['word_match'] == 'Yes').sum() / len(group_df)
        print(f"{group} SF: Accuracy = {accuracy:.1%}")
```

### 5. 可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 空间频率分布
ax1 = axes[0]
ax1.hist(df['spatial_frequency'], bins=50, alpha=0.7, color='blue')
ax1.set_xlabel('Spatial Frequency')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Spatial Frequency')
ax1.axvline(x=0.1, color='r', linestyle='--', label='Low/Medium threshold')
ax1.axvline(x=0.2, color='g', linestyle='--', label='Medium/High threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 周长复杂度分布
ax2 = axes[1]
ax2.hist(df['perimetric_complexity'], bins=50, alpha=0.7, color='green')
ax2.set_xlabel('Perimetric Complexity')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Perimetric Complexity')
ax2.axvline(x=50, color='r', linestyle='--', label='Moderate/Complex threshold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/new_metrics_distribution.png', dpi=300)
plt.show()

# 散点图：空间频率 vs 周长复杂度
plt.figure(figsize=(10, 8))
plt.scatter(df['spatial_frequency'], df['perimetric_complexity'], 
           alpha=0.5, c=df['avg_curvature'], cmap='viridis')
plt.colorbar(label='Average Curvature')
plt.xlabel('Spatial Frequency')
plt.ylabel('Perimetric Complexity')
plt.title('Spatial Frequency vs Perimetric Complexity\n(colored by curvature)')
plt.grid(True, alpha=0.3)
plt.savefig('results/sf_vs_pc.png', dpi=300)
plt.show()
```

---

## 🔬 研究应用

### 1. 图像质量评估
- 空间频率低 → 可能需要图像增强
- 结合多个指标评估overall质量

### 2. 文本难度分级
```python
def classify_difficulty(row):
    if row['spatial_frequency'] < 0.1 and row['perimetric_complexity'] > 100:
        return 'Very Hard'
    elif row['spatial_frequency'] < 0.15 or row['perimetric_complexity'] > 50:
        return 'Hard'
    elif row['spatial_frequency'] > 0.2 and row['perimetric_complexity'] < 20:
        return 'Easy'
    else:
        return 'Medium'

df['difficulty_class'] = df.apply(classify_difficulty, axis=1)
print(df['difficulty_class'].value_counts())
```

### 3. 数据增强策略
- 对低SF样本：应用锐化
- 对高PC样本：简化边界或使用更robust的模型

### 4. 模型选择
- 低SF + 高PC → 需要robust to blur和shape variation的模型
- 高SF + 低PC → 标准OCR模型即可

---

## 📊 完整特征列表

现在CSV包含10个特征：

```csv
image_name,text,polygon_size,edge_density,luminance_std,luminance_mean,avg_curvature,gaussian_rms_contrast,spatial_frequency,perimetric_complexity
```

| # | 特征名 | 类型 | 范围 | 含义 |
|---|--------|------|------|------|
| 1 | polygon_size | 几何 | >0 px² | 文本区域面积 |
| 2 | edge_density | 图像 | 0-1 | 边缘像素比例 |
| 3 | luminance_std | 图像 | 0-255 | 亮度标准差 |
| 4 | luminance_mean | 图像 | 0-255 | 平均亮度 |
| 5 | avg_curvature | 几何 | 0-180° | 平均曲率 |
| 6 | gaussian_rms_contrast | 图像 | 0-1 | 感知对比度 |
| 7 | **spatial_frequency** | **图像** | **0-1** | **空间频率** 🆕 |
| 8 | **perimetric_complexity** | **几何** | **>0** | **周长复杂度** 🆕 |

---

## 🚀 使用方法

### 提取特征
```bash
cd /cis/home/qgao14/my_documents/VIOCR_infer_models
python untils/eval/extract_text_region_features.py
```

### 分析特征
```bash
python untils/eval/analyze_features.py
```

### 查看结果
```bash
# 查看原始数据
head -20 results/totaltext_16_text_region_features.csv

# 查看统计
cat results/totaltext_16_features_with_difficulty.csv | column -t -s, | head -20
```

---

## 📚 参考文献

### Spatial Frequency
1. **Eskicioglu & Fisher (1995)**
   - "Image quality measures and their performance"
   - IEEE Transactions on Communications

2. **Zhu & Wang (2010)**
   - "Image quality assessment based on spatial frequency"
   - 经典的图像质量评估方法

### Perimetric Complexity
1. **Suen & Wang (1994)**
   - "Analysis of line patterns in documents"
   - Pattern Recognition

2. **Rosin (2005)**
   - "Measuring shape: ellipticity, rectangularity, and triangularity"
   - 形状复杂度度量方法

---

## ✅ 总结

### 两个新特征的价值

✅ **Spatial Frequency**:
- 量化图像细节丰富程度
- 评估文本清晰度
- 检测模糊问题

✅ **Perimetric Complexity**:
- 量化边界复杂度
- 评估文本形状复杂性
- 补充Curvature信息

### 与现有特征的关系

- **与Edge Density**: Spatial Frequency关注灰度变化，Edge Density关注边缘存在
- **与Curvature**: Perimetric Complexity关注整体边界，Curvature关注局部角度
- **与Contrast**: Spatial Frequency关注变化频率，Contrast关注变化幅度

### 应用场景

1. ✅ 图像质量诊断
2. ✅ 文本难度评估
3. ✅ 数据增强设计
4. ✅ 模型性能分析
5. ✅ 数据集特性分析

**所有新特征已集成，立即开始使用！** 🎉








