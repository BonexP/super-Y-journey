# 5. 模型配置文件详解

本文档详细解释YOLO模型的YAML配置文件格式和修改方法。

## 📄 YAML配置文件结构

YAML配置文件是定义YOLO模型架构的核心，位于 `ultralytics/cfg/models/` 目录下。

### 基本结构

```yaml
# 模型元信息
nc: 80 # number of classes（类别数）
scales: # 模型缩放参数（可选）

# 网络架构
backbone: # 主干网络
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]]

head: # 检测头
    # [from, repeats, module, args]
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
```

---

## 🔍 详细参数说明

### 1. 全局参数

**示例** (`ultralytics/cfg/models/v8/yolov8.yaml` 第1-16行):

```yaml
# 类别数
nc: 80 # COCO数据集有80个类别

# 缩放参数（用于n/s/m/l/x不同大小的模型）
scales:
    # [depth_multiple, width_multiple, max_channels]
    n: [0.33, 0.25, 1024] # YOLOv8n - 轻量级
    s: [0.33, 0.50, 1024] # YOLOv8s - 小型
    m: [0.67, 0.75, 768] # YOLOv8m - 中型
    l: [1.00, 1.00, 512] # YOLOv8l - 大型
    x: [1.00, 1.25, 512] # YOLOv8x - 超大型
```

**参数含义**:

- `depth_multiple`: 深度倍数，控制网络层数
    - 例如: `n=3, depth=0.33` → 实际层数 = `max(round(3 * 0.33), 1) = 1`
- `width_multiple`: 宽度倍数，控制通道数
    - 例如: `c2=256, width=0.5` → 实际通道 = `256 * 0.5 = 128`
- `max_channels`: 最大通道数限制

### 2. 层定义格式

每一层使用以下格式定义：

```yaml
[from, repeats, module, args]
```

**参数详解**:

#### `from` - 输入来源

- `-1`: 来自上一层
- `6`: 来自第6层（索引从0开始）
- `[-1, 6]`: 来自多个层（用于Concat）
- `[4, 6, 9]`: 来自多个层（用于Detect等）

#### `repeats` - 重复次数

- `1`: 该模块只有1个
- `3`: 该模块重复3次
- 会受到 `depth_multiple` 影响

#### `module` - 模块名称

- `Conv`: 标准卷积
- `C2f`: YOLOv8的核心模块
- `SPPF`: 空间金字塔池化
- `nn.Upsample`: PyTorch内置模块
- 等等...

#### `args` - 模块参数

根据不同模块有不同的参数，常见格式：

- `[c2, k, s]`: 输出通道, 卷积核, 步长
- `[c2, shortcut]`: 输出通道, 是否使用shortcut
- 详见各模块定义

---

## 📐 完整示例：YOLOv8配置解析

### Backbone部分

```yaml
backbone:
    # [from, repeats, module, args]

    # Stem: 快速下采样
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
      # from=-1: 输入图像
      # repeats=1: 不重复
      # module=Conv: 卷积层
      # args=[64, 3, 2]: 输出64通道, 3x3卷积核, stride=2

    # Stage 1
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
      # 下采样到1/4大小

    - [-1, 3, C2f, [128, True]] # 2
      # repeats=3: 重复3次（受depth_multiple影响）
      # args=[128, True]: 输出128通道, shortcut=True

    # Stage 2
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]] # 4

    # Stage 3
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]] # 6

    # Stage 4
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]] # 8

    # SPP
    - [-1, 1, SPPF, [1024, 5]] # 9
      # args=[1024, 5]: 输出1024通道, 池化核大小5
```

### Head部分（FPN + PAN结构）

```yaml
head:
    # FPN (自顶向下)
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 10
      # 上采样2倍

    - [[-1, 6], 1, Concat, [1]] # 11
      # 拼接层10和层6
      # args=[1]: 在维度1（通道维度）拼接

    - [-1, 3, C2f, [512]] # 12
      # 不需要shortcut参数，使用默认False

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13
    - [[-1, 4], 1, Concat, [1]] # 14
    - [-1, 3, C2f, [256]] # 15 (P3/8-small)

    # PAN (自底向上)
    - [-1, 1, Conv, [256, 3, 2]] # 16
    - [[-1, 12], 1, Concat, [1]] # 17
    - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

    - [-1, 1, Conv, [512, 3, 2]] # 19
    - [[-1, 9], 1, Concat, [1]] # 20
    - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

    # Detect
    - [[15, 18, 21], 1, Detect, [nc]] # 22
      # 从P3, P4, P5三个尺度检测
      # args=[nc]: 类别数
```

---

## 🔧 修改配置文件

### 修改1: 改变网络深度

**原始**:

```yaml
- [-1, 3, C2f, [256, True]] # 重复3次
```

**加深网络**:

```yaml
- [-1, 6, C2f, [256, True]] # 重复6次
```

**说明**: 增加repeats可以加深网络，但会增加参数量和计算量。

### 修改2: 改变通道数

**原始**:

```yaml
- [-1, 1, Conv, [256, 3, 2]]
```

**增加通道**:

```yaml
- [-1, 1, Conv, [512, 3, 2]] # 256 → 512
```

**说明**: 增加通道数可以提升模型容量，但显著增加计算量。

### 修改3: 添加新模块

**在backbone末尾添加注意力**:

```yaml
backbone:
    # ... 原有层
    - [-1, 1, SPPF, [1024, 5]] # 9
    - [-1, 1, CBAM, [1024]] # 10 新增CBAM注意力
```

**注意**: 添加新层后，后续层的索引会改变，需要相应调整 `from` 参数。

### 修改4: 替换模块

**将C2f替换为C3**:

```yaml
# 原始
- [-1, 3, C2f, [256, True]]

# 替换
- [-1, 3, C3, [256, True]]
```

### 修改5: 修改下采样策略

**使用不同的下采样方法**:

**原始 - Conv stride=2**:

```yaml
- [-1, 1, Conv, [256, 3, 2]]
```

**改为 - MaxPool**:

```yaml
- [-1, 1, nn.MaxPool2d, [2, 2, 0]] # [kernel, stride, padding]
```

**改为 - 平均池化**:

```yaml
- [-1, 1, nn.AvgPool2d, [2, 2, 0]]
```

---

## 🎨 创建自定义配置

### 示例1: 轻量级YOLO配置

创建文件: `ultralytics/cfg/models/v8/yolov8-nano.yaml`

```yaml
# YOLOv8-nano - 超轻量级模型
nc: 80
scales:
    nano: [0.25, 0.20, 512] # 更小的depth和width

backbone:
    # 使用DWConv减少参数
    - [-1, 1, DWConv, [32, 3, 2]] # 0-P1/2
    - [-1, 1, DWConv, [64, 3, 2]] # 1-P2/4
    - [-1, 2, C2f, [64, True]] # 2
    - [-1, 1, DWConv, [128, 3, 2]] # 3-P3/8
    - [-1, 3, C2f, [128, True]] # 4
    - [-1, 1, DWConv, [256, 3, 2]] # 5-P4/16
    - [-1, 3, C2f, [256, True]] # 6
    - [-1, 1, DWConv, [512, 3, 2]] # 7-P5/32
    - [-1, 2, C2f, [512, True]] # 8
    - [-1, 1, SPPF, [512, 5]] # 9

head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 2, C2f, [256]] # 12

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]]
    - [-1, 2, C2f, [128]] # 15 (P3/8-small)

    - [-1, 1, Conv, [128, 3, 2]]
    - [[-1, 12], 1, Concat, [1]]
    - [-1, 2, C2f, [256]] # 18 (P4/16-medium)

    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 9], 1, Concat, [1]]
    - [-1, 2, C2f, [512]] # 21 (P5/32-large)

    - [[15, 18, 21], 1, Detect, [nc]]
```

### 示例2: 添加Transformer的YOLO

创建文件: `ultralytics/cfg/models/v8/yolov8-transformer.yaml`

```yaml
# YOLOv8 with Transformer blocks
nc: 80

backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]
    # 添加Transformer
    - [-1, 1, TransformerBlock, [1024, 8, 2]] # [c2, num_heads, num_layers]

head:
    # ... 标准head配置
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 3, C2f, [512]]

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]]
    - [-1, 3, C2f, [256]]

    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 12], 1, Concat, [1]]
    - [-1, 3, C2f, [512]]

    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 10], 1, Concat, [1]] # 注意索引变化！
    - [-1, 3, C2f, [1024]]

    - [[15, 18, 21], 1, Detect, [nc]]
```

### 示例3: 多尺度注意力YOLO

```yaml
# YOLOv8 with multi-scale attention
nc: 80

backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, CBAM, [128]] # P2后加注意力

    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, CBAM, [256]] # P3后加注意力

    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, CBAM, [512]] # P4后加注意力

    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]
    - [-1, 1, CBAM, [1024]] # P5后加注意力

head:
    # 标准head配置...
```

---

## 📊 配置文件对比

### 不同版本YOLO的主要区别

| 版本    | 核心模块            | 特点                     |
| ------- | ------------------- | ------------------------ |
| YOLOv5  | C3, SPPF            | CSP结构，成熟稳定        |
| YOLOv8  | C2f, SPPF           | 改进的CSP，更快的推理    |
| YOLOv9  | RepNCSPELAN4, ADown | 更深的网络，GELAN架构    |
| YOLOv10 | C2fCIB, PSA         | 双分支检测头，注意力增强 |

---

## ✅ 验证配置文件

### 方法1: 使用info命令

```python
from ultralytics import YOLO

model = YOLO("path/to/your/custom.yaml")
model.info()  # 打印模型信息
```

**输出示例**:

```
Model summary: 225 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPs
```

### 方法2: 可视化模型

```python

from ultralytics import YOLO

model = YOLO("custom.yaml")

# 导出为ONNX以可视化
model.export(format="onnx")

# 使用Netron查看: https://netron.app/
```

### 方法3: 测试前向传播

```python
import torch

from ultralytics import YOLO

model = YOLO("custom.yaml")

# 创建随机输入
x = torch.randn(1, 3, 640, 640)

# 前向传播
with torch.no_grad():
    outputs = model(x)

print(f"Output shapes: {[o.shape for o in outputs]}")
```

---

## 🚨 常见错误

### 错误1: 通道数不匹配

```
RuntimeError: size mismatch, expected 256, got 512
```

**原因**: Concat或其他连接操作时通道数不对应

**解决**: 检查 `from` 参数，确保连接的层通道数正确

### 错误2: 索引超出范围

```
IndexError: list index out of range
```

**原因**: `from` 参数引用了不存在的层

**解决**: 添加或删除层后，重新计算索引

### 错误3: 模块未注册

```
KeyError: 'CustomModule'
```

**原因**: 自定义模块未在 `tasks.py` 中注册

**解决**: 参考第3、4章，完成模块注册

---

## 📝 最佳实践

1. **从现有配置修改**: 不要从零开始，基于官方配置修改
2. **渐进式修改**: 一次只改一处，便于调试
3. **保持索引正确**: 使用注释标注层的索引
4. **验证通道数**: 确保Concat等操作的通道数匹配
5. **测试构建**: 修改后立即测试能否正常构建模型

---

下一步，请阅读 [实战示例](./06-practical-examples.md) 查看完整的修改案例。
