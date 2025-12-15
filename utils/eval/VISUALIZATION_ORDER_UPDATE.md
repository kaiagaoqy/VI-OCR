# 可视化工具 - Detection Order 显示更新

## 📋 更新说明

在 `visualize_detections_and_gt.py` 中更新了可视化功能，现在左侧的Human Detections图会同时显示：
1. **Detection Order（检测顺序）** - 用方括号标记，如 `[0]`, `[1]`, `[2]`
2. **Recognition Text（识别文本）** - 实际识别出的单词

## 🎨 可视化效果

### 更新前
```
左侧图（Detections）显示：
  [红色边框] STORE
  [红色边框] OPEN
  [红色边框] NOW
```

### 更新后
```
左侧图（Detections）显示：
  [红色边框] [0] STORE
  [红色边框] [1] OPEN
  [红色边框] [2] NOW
```

## 🔍 显示格式

### 左侧图（Human Detections）
- **边框颜色**: 红色
- **文本标签格式**: `[Order] Text`
  - 例如: `[0] STORE`（第0个检测到的单词是"STORE"）
  - 例如: `[3] OPEN`（第3个检测到的单词是"OPEN"）
- **Order含义**: 该单词在JSON文件中出现的顺序（从0开始）

### 右侧图（Ground Truth）
- **边框颜色**: 
  - 绿色（lime）- 正常文本
  - 灰色（gray）- Don't care区域（"###"）
- **文本标签**: 仅显示GT文本，不显示order

## 🚀 使用方法

### 运行可视化
```bash
cd /cis/home/qgao14/my_documents/VIOCR_infer_models
python untils/eval/visualize_detections_and_gt.py
```

### 输出位置
```
results/visualizations/
├── Sub002_image_0.png
├── Sub002_image_16.png
├── Sub003_image_0.png
└── ...
```

### 配置选项

在脚本中可以修改：

```python
# Line 191-194
MODEL_OUTPUT_PATH = 'data/human/lowviz/converted'  # 模型输出路径
GT_PATH = 'data/totaltext/anno.json'               # GT标注路径
IMAGE_BASE_PATH = 'data/totaltext'                 # 图像基础路径
OUTPUT_DIR = 'results/visualizations'              # 输出目录

# Line 213
max_visualize = 20  # 每个subject最多可视化多少张图
```

## 📊 应用场景

### 1. 验证Detection Order
直观查看每个单词的检测顺序是否合理：
- Order=0的通常应该是最显著的文本
- 检查是否存在空间规律（从左到右、从上到下）

### 2. 分析标注模式
- 观察标注员的扫描路径
- 识别是否存在一致的标注顺序
- 发现可能的标注偏差

### 3. 调试Order记录
- 验证order是否正确记录
- 检查order与实际位置的对应关系
- 确认order在不同图像间独立计数

### 4. 质量控制
- 快速识别order异常的情况
- 检查order=0的单词是否确实显著
- 验证order连续性

## 🎯 解读示例

### 正常情况
```
图像显示：
[0] STORE   [1] OPEN   [2] 24/7

解读：
- STORE最先被检测（最显著）
- OPEN第二个被检测
- 24/7最后被检测
- Order连续，无跳跃
```

### 异常情况1：Order跳跃
```
图像显示：
[0] STORE   [2] OPEN   [5] NOW

解读：
⚠️ Order不连续（缺少1, 3, 4）
可能原因：
- 某些检测被过滤掉了（polygon太小等）
- 数据处理中有问题
```

### 异常情况2：Order与显著性不符
```
图像显示：
[5] STORE （大且清晰的文本）
[0] sale  （小且模糊的文本）

解读：
⚠️ 最显著的文本order=5（较晚检测）
可能原因：
- 标注员扫描顺序不规律
- 可能的标注质量问题
```

## 🔧 技术实现

### 代码变更

#### 1. 记录Order（Line 90）
```python
det_by_image[unique_key].append({
    'bbox': polygon_points,
    'text': text,
    'order': i  # 记录检测顺序
})
```

#### 2. 显示Order（Line 142-157）
```python
for i, det in enumerate(detections):
    bbox = det['bbox']
    text = det['text']
    order = det.get('order', i)  # 获取order
    
    # 显示格式: [Order] Text
    label = f'[{order}] {text}'
    ax1.text(center_x, center_y, label, 
            fontsize=11, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
            ha='center', va='center')
```

### 关键特性

1. **Fallback机制**: 如果order不存在，使用enumerate的索引
2. **格式一致**: 统一使用 `[Order] Text` 格式
3. **可读性**: 11号字体，白色粗体，红色背景
4. **居中对齐**: `ha='center', va='center'`

## 📈 可视化分析工作流

### Step 1: 运行可视化
```bash
python untils/eval/visualize_detections_and_gt.py
```

### Step 2: 查看结果
```bash
cd results/visualizations
ls -la *.png
```

### Step 3: 选择性查看
打开几个图像，观察：
- Order编号是否清晰可见
- Order顺序是否符合预期
- Detection与GT的对应关系

### Step 4: 结合评估结果
```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 读取评估结果
df = pd.read_excel('results/excel_outputs/Sub002_evaluation.xlsx', 
                   sheet_name='Word Pairs')

# 找出Order=0但识别错误的情况
first_detections = df[df['Detection Order'] == 0]
errors = first_detections[first_detections['Word Match'] == 'No']

print(f"第一个检测(Order=0)的错误率: {len(errors)/len(first_detections)*100:.1f}%")

# 查看对应的可视化
for img_name in errors['Image Name'].unique()[:5]:
    # 查找对应的可视化图像
    img_path = f"results/visualizations/Sub002_image_{img_name.split('/')[-1].split('.')[0]}.png"
    print(f"查看: {img_path}")
```

### Step 5: 模式识别
观察多个图像后，寻找：
- **位置模式**: Order是否从左到右、从上到下递增？
- **大小模式**: 大文本是否更早被检测（Order更小）？
- **对比度模式**: 高对比度文本是否Order更小？
- **错误模式**: 特定Order位置是否更容易出错？

## 📝 使用技巧

### 技巧1: 批量查看特定subject
```bash
# 只查看Sub002的可视化
ls results/visualizations/Sub002_*.png | head -10
```

### 技巧2: 制作动画（可选）
```python
from PIL import Image
import imageio

# 将一个subject的所有图像制作成GIF
images = []
for img_path in sorted(Path('results/visualizations').glob('Sub002_*.png')):
    images.append(imageio.imread(img_path))

imageio.mimsave('results/Sub002_animation.gif', images, duration=1.0)
print("动画已保存: results/Sub002_animation.gif")
```

### 技巧3: 筛选异常Order
在可视化后，可以用脚本筛选需要重点查看的图像：

```python
import pandas as pd
from pathlib import Path

df = pd.read_excel('results/excel_outputs/Sub002_evaluation.xlsx', 
                   sheet_name='Word Pairs')

# 找出Order>10的情况（表示很多检测）
many_detections = df[df['Detection Order'] > 10]
print(f"Found {len(many_detections)} words with order > 10")

# 找出这些图像
images_to_check = many_detections['Image Name'].unique()
print(f"\n需要查看的图像:")
for img in images_to_check[:5]:
    print(f"  - {img}")
```

## ⚠️ 注意事项

1. **Order从0开始**: 第一个检测的order=0，不是1
2. **Per-image独立**: 每张图的order独立计数
3. **红色标签**: Detection的标签用红色背景，容易与GT（绿色）区分
4. **字体大小**: 如果文本重叠，可以调整fontsize参数（Line 154）
5. **图像数量**: 默认每个subject最多可视化20张，可在Line 213修改

## ✅ 验证清单

运行后检查：
- ✅ 左侧图显示 `[Order] Text` 格式
- ✅ Order从0开始
- ✅ 每张图的Order独立计数
- ✅ Order与polygon位置对应
- ✅ 右侧GT图不显示Order（保持原样）
- ✅ 图像清晰可读

## 🎉 总结

更新后的可视化工具让您能够：
- ✅ 直观查看每个单词的检测顺序
- ✅ 验证Order记录的正确性
- ✅ 分析标注员的扫描模式
- ✅ 识别顺序相关的性能问题
- ✅ 结合评估结果进行深度分析

这个功能与 `totaltext_eval_FINAL.py` 中的Detection Order功能完美配合，提供完整的顺序分析能力！🚀









