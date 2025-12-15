# 新增特征说明文档

## 📊 更新概览

在 `extract_text_region_features.py` 中新增了两个重要特征：
1. **Average Curvature (平均曲率)** - 衡量polygon的形状复杂度
2. **Gaussian RMS Contrast (高斯RMS对比度)** - 基于Zuiderbaan et al. 2017的局部对比度能量

同时重命名了原有特征以提高清晰度：
- `avg_local_contrast` → `luminance_std` (亮度标准差)
- `avg_luminance` → `luminance_mean` (平均亮度)

---

## 🆕 新增特征详解

### 1. Average Curvature (平均曲率)

#### 定义
衡量polygon边界的弯曲程度，反映文本区域的形状复杂度。

#### 计算方法
```
对于polygon的每个顶点:
  1. 计算前后两条边的向量 v1, v2
  2. 计算两向量之间的夹角 θ
  3. 曲率 = |180° - θ|  (偏离直线的程度)
  
平均曲率 = mean(所有顶点的曲率)
```

#### 数学公式
```
curvature_i = |180° - arccos(v1·v2 / (|v1||v2|))|
avg_curvature = Σ curvature_i / N
```

#### 取值范围和含义

| 曲率值 | 含义 | 示例 |
|--------|------|------|
| **0° - 20°** | 近似矩形，边缘平直 | 规则印刷体文本框 |
| **20° - 40°** | 轻微弯曲 | 略有旋转的文本 |
| **40° - 90°** | 明显弯曲 | 曲线文本，TotalText典型特征 |
| **> 90°** | 高度弯曲/复杂形状 | 螺旋文本，艺术字 |

#### 在识别中的意义
- **低曲率** (< 20°): 
  - 文本排列规则
  - 易于识别
  - 传统OCR模型表现好
  
- **中等曲率** (20-60°):
  - TotalText数据集的核心挑战
  - 需要curved text检测
  - 考验模型对形变的适应性

- **高曲率** (> 60°):
  - 极具挑战性
  - 可能导致字符扭曲
  - 需要专门的curved/arbitrary-shape模型

#### 代码实现
```python
def compute_polygon_curvature(polygon_coords):
    points = np.array(polygon_coords)
    n = len(points)
    curvatures = []
    
    for i in range(n):
        p1 = points[(i - 1) % n]
        p2 = points[i]
        p3 = points[(i + 1) % n]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 > 0 and len_v2 > 0:
            cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            curvature = abs(180.0 - angle)
            curvatures.append(curvature)
    
    return np.mean(curvatures)
```

---

### 2. Gaussian RMS Contrast (高斯RMS对比度)

#### 定义
基于Gaussian-weighted RMS (Root Mean Square) contrast的局部对比度能量度量。

#### 理论基础
参考文献: Zuiderbaan et al., 2017
- 人类视觉系统对局部对比度敏感
- 使用高斯窗口模拟视觉感受野
- 比简单的标准差更符合视觉感知

#### 计算方法
```
1. 对图像应用高斯滤波，得到局部均值 μ(x,y)
2. 计算每个像素与局部均值的差值平方: (I(i) - μ)²
3. 对差值平方再次应用高斯滤波，得到局部方差
4. 对比度图 = sqrt(局部方差)
5. 在mask区域内取平均
```

#### 数学公式
```
local_mean(x,y) = Gaussian_σ * I(x,y)

local_var(x,y) = Gaussian_σ * (I(x,y) - local_mean(x,y))²

contrast(x,y) = sqrt(local_var(x,y))

RMS_contrast = mean(contrast(x,y) for (x,y) in mask)
```

其中 `Gaussian_σ` 表示标准差为 σ 的高斯滤波器。

#### 参数设置
- **σ = 19.4 pixels**: 默认值，基于Zuiderbaan et al. 2017
  - 对应约 1° 的视角（在典型观看距离）
  - 适合捕捉文本级别的对比度变化

#### 取值范围和含义

| 对比度值 | 含义 | 示例场景 |
|----------|------|----------|
| **< 0.05** | 极低对比度 | 文本与背景几乎同色 |
| **0.05 - 0.10** | 低对比度 | 浅色文本+浅色背景，或深色+深色 |
| **0.10 - 0.20** | 中等对比度 | 可识别但不够清晰 |
| **0.20 - 0.30** | 良好对比度 | 大多数正常场景文本 |
| **> 0.30** | 高对比度 | 黑白文本，清晰边界 |

#### 与传统对比度度量的对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Luminance Std** | 简单快速 | 全局度量，忽略空间结构 | 快速评估 |
| **Michelson** | 经典定义 | 需要明确的前景/背景 | 高对比度场景 |
| **Gaussian RMS** | 符合视觉感知，局部度量 | 计算稍慢 | 精确的感知评估 |

#### 在识别中的意义

- **高 Gaussian RMS contrast (> 0.2)**:
  - 文本边界清晰
  - 字符结构可辨
  - 高识别率

- **中等对比度 (0.1 - 0.2)**:
  - 可识别但有挑战
  - 可能需要预处理增强
  - 识别率中等

- **低对比度 (< 0.1)**:
  - 严重挑战
  - 需要对比度自适应算法
  - 低识别率

#### 代码实现
```python
def compute_local_contrast_energy(image, mask, sigma=19.4):
    # Normalize to [0, 1]
    gray = image.astype(np.float64) / 255.0
    
    # Compute local mean with Gaussian filter
    local_mean = gaussian_filter(gray, sigma=sigma)
    
    # Compute squared differences
    squared_diff = (gray - local_mean) ** 2
    
    # Apply Gaussian filter to squared differences
    local_var = gaussian_filter(squared_diff, sigma=sigma)
    
    # RMS contrast (root of local variance)
    contrast_map = np.sqrt(local_var)
    
    # Apply mask and compute mean
    masked_contrast = contrast_map[mask > 0]
    return masked_contrast.mean()
```

---

## 📋 完整特征列表

### 更新后的CSV格式

```csv
image_name,text,polygon_size,edge_density,luminance_std,luminance_mean,avg_curvature,gaussian_rms_contrast
16/img_001.jpg,STORE,2543.67,0.3456,45.23,128.45,23.45,0.2134
```

### 特征总览

| 特征名 | 单位 | 范围 | 含义 |
|--------|------|------|------|
| **polygon_size** | px² | > 0 | 文本区域面积 |
| **edge_density** | - | 0-1 | 边缘像素比例 |
| **luminance_std** | - | 0-255 | 亮度标准差（简单对比度） |
| **luminance_mean** | - | 0-255 | 平均亮度 |
| **avg_curvature** | 度(°) | 0-180 | 平均曲率（形状复杂度） |
| **gaussian_rms_contrast** | - | 0-1 | 高斯RMS对比度（感知对比度） |

---

## 🎯 使用建议

### 1. 难度评估
综合多个特征判断样本难度：

```python
# 困难样本特征
hard_samples = df[
    (df['gaussian_rms_contrast'] < 0.1) &      # 低对比度
    (df['edge_density'] < 0.2) &               # 模糊边缘
    (df['avg_curvature'] > 60)                 # 高曲率
]
```

### 2. 数据集分析
按曲率分组分析：

```python
df['curvature_group'] = pd.cut(
    df['avg_curvature'], 
    bins=[0, 20, 60, 180],
    labels=['Straight', 'Curved', 'High-Curved']
)

for group in ['Straight', 'Curved', 'High-Curved']:
    print(f"{group}:")
    print(f"  Avg Gaussian Contrast: {df[df['curvature_group']==group]['gaussian_rms_contrast'].mean():.4f}")
```

### 3. 模型性能分析
结合识别结果分析：

```python
import pandas as pd

# 读取特征和识别结果
features = pd.read_csv('results/totaltext_16_text_region_features.csv')
eval_results = pd.read_excel('results/excel_outputs/Sub002_evaluation.xlsx', sheet_name='Word Pairs')

# 合并
merged = features.merge(eval_results, left_on=['image_name', 'text'], right_on=['Image Name', 'GT Text'])

# 分析：高曲率样本的识别准确率
high_curvature = merged[merged['avg_curvature'] > 60]
print(f"High curvature accuracy: {high_curvature['Word Match'].value_counts()}")

# 分析：对比度与识别性能的关系
print(merged.groupby(pd.cut(merged['gaussian_rms_contrast'], bins=5))['Char F1'].mean())
```

### 4. 可视化特征关系

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 散点图：曲率 vs 对比度
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='avg_curvature', y='gaussian_rms_contrast', 
                hue='difficulty', alpha=0.6)
plt.xlabel('Average Curvature (degrees)')
plt.ylabel('Gaussian RMS Contrast')
plt.title('Text Region Difficulty Analysis')
plt.show()

# 热力图：特征相关性
features = ['polygon_size', 'edge_density', 'luminance_std', 'luminance_mean', 
            'avg_curvature', 'gaussian_rms_contrast']
corr = df[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

---

## 🔬 研究应用

### 1. 数据集难度量化
- 使用曲率和对比度定义难度级别
- 对比不同数据集的特征分布
- 识别"未覆盖"的难度区间

### 2. 模型鲁棒性测试
- 创建不同曲率范围的测试集
- 评估模型在不同对比度下的性能
- 设计targeted adversarial examples

### 3. 数据增强策略
- 针对低对比度样本进行增强
- 为高曲率样本生成变换
- 平衡训练集的特征分布

### 4. 错误分析
- 识别哪些特征组合导致失败
- 找出模型的"弱点区域"
- 指导模型改进方向

---

## 🚀 快速开始

### 运行特征提取
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
# 查看原始特征
cat results/totaltext_16_text_region_features.csv | head

# 查看难度标记
cat results/totaltext_16_features_with_difficulty.csv | head

# 统计信息
wc -l results/totaltext_16_text_region_features.csv
```

---

## 📚 参考文献

1. **Zuiderbaan et al., 2017**
   - "Modeling center-surround configurations in population receptive field using fMRI"
   - 提出Gaussian-weighted RMS contrast度量

2. **Peli, 1990**
   - "Contrast in complex images"
   - RMS contrast的早期定义

3. **TotalText Dataset (Ch'ng & Chan, 2017)**
   - Curved text detection benchmark
   - 曲率是核心挑战

---

## ✅ 总结

### 新增特征的价值

✅ **Average Curvature**: 
- 量化文本形状复杂度
- TotalText数据集的核心特征
- 直接关联curved text detection难度

✅ **Gaussian RMS Contrast**:
- 更准确的感知对比度度量
- 基于视觉科学研究
- 比简单标准差更有意义

### 应用场景

1. ✅ 数据集质量分析
2. ✅ 样本难度评估
3. ✅ 模型性能诊断
4. ✅ 错误模式识别
5. ✅ 数据增强设计

---

**所有工具已更新，立即开始您的特征分析之旅！** 🎉









