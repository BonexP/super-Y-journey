# 6. 实战示例

本文档提供完整的YOLO模型修改实战示例，从定义新模块到训练测试的完整流程。

## 🎯 示例1: 添加SE注意力到YOLOv8

这是一个完整的端到端示例，展示如何添加SE注意力机制到YOLOv8模型。

### 步骤1: 定义SE模块

**文件**: `ultralytics/nn/modules/block.py`

在文件末尾添加（约2000行左右）:

```python
class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention module.

    This module applies channel-wise attention to enhance important features and suppress less useful ones.

    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for the bottleneck

    Examples:
        >>> se = SEAttention(256, reduction=16)
        >>> x = torch.randn(1, 256, 20, 20)
        >>> y = se(x)
        >>> print(y.shape)
        torch.Size([1, 256, 20, 20])
    """

    def __init__(self, channels, reduction=16):
        """Initialize SE attention with squeeze and excitation operations."""
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply SE attention: squeeze global information and excite channel-wise."""
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers with sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: multiply input by attention weights
        return x * y.expand_as(x)
```

### 步骤2: 导出模块

**文件**: `ultralytics/nn/modules/block.py`

在 `__all__` 元组中添加（约15-55行）:

```python
__all__ = (
    "DFL",
    "HGBlock",
    # ... 其他模块
    "SEAttention",  # 添加这一行
)
```

**文件**: `ultralytics/nn/modules/__init__.py`

导入和导出模块:

```python
from .block import (
    # ... 其他导入
    SEAttention,  # 添加这一行
)

__all__ = (
    # ... 其他
    "SEAttention",  # 添加这一行
)
```

### 步骤3: 注册到模型解析器

**文件**: `ultralytics/nn/tasks.py`

在 `parse_model` 函数的 `base_modules` 中添加（约1613-1654行）:

```python
base_modules = frozenset(
    {
        Classify,
        Conv,
        # ... 其他模块
        SEAttention,  # 添加这一行
    }
)
```

### 步骤4: 创建配置文件

**文件**: `ultralytics/cfg/models/v8/yolov8-se.yaml`

```yaml
# YOLOv8 with SE Attention
# Adds SE attention after each C2f block for better feature representation

nc: 80 # number of classes
scales:
    # [depth, width, max_channels]
    n: [0.33, 0.25, 1024]
    s: [0.33, 0.50, 1024]
    m: [0.67, 0.75, 768]
    l: [1.00, 1.00, 512]
    x: [1.00, 1.25, 512]

# YOLOv8 backbone with SE
backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, SEAttention, [128, 16]] # 3 - 添加SE注意力

    - [-1, 1, Conv, [256, 3, 2]] # 4-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, SEAttention, [256, 16]] # 6 - 添加SE注意力

    - [-1, 1, Conv, [512, 3, 2]] # 7-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, SEAttention, [512, 16]] # 9 - 添加SE注意力

    - [-1, 1, Conv, [1024, 3, 2]] # 10-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]] # 12
    - [-1, 1, SEAttention, [1024, 16]] # 13 - 添加SE注意力

# YOLOv8 head
head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 9], 1, Concat, [1]] # cat backbone P4 (注意索引变化)
    - [-1, 3, C2f, [512]] # 16

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]] # cat backbone P3 (注意索引变化)
    - [-1, 3, C2f, [256]] # 19 (P3/8-small)

    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 16], 1, Concat, [1]] # cat head P4
    - [-1, 3, C2f, [512]] # 22 (P4/16-medium)

    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 13], 1, Concat, [1]] # cat head P5 (注意索引变化)
    - [-1, 3, C2f, [1024]] # 25 (P5/32-large)

    - [[19, 22, 25], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

### 步骤5: 测试模型构建

```python
from ultralytics import YOLO

# 加载自定义配置
model = YOLO("ultralytics/cfg/models/v8/yolov8-se.yaml")

# 查看模型信息
model.info()

# 打印模型结构
print(model.model)
```

**预期输出**:

```
Model summary: 268 layers, 3500000 parameters, 3500000 gradients, 9.5 GFLOPs
```

### 步骤6: 训练模型

```python
from ultralytics import YOLO

# 加载模型
model = YOLO("ultralytics/cfg/models/v8/yolov8-se.yaml")

# 训练
results = model.train(
    data="coco8.yaml",  # 数据集配置
    epochs=100,  # 训练轮数
    imgsz=640,  # 图像大小
    batch=16,  # 批量大小
    name="yolov8n-se",  # 实验名称
    device=0,  # GPU设备
)

# 验证
metrics = model.val()

# 推理
results = model("path/to/image.jpg")
```

---

## 🎯 示例2: 创建轻量级Ghost-YOLO

使用GhostConv替换普通卷积以减少参数量。

### 步骤1: 创建Ghost-YOLO配置

**文件**: `ultralytics/cfg/models/v8/yolov8-ghost-custom.yaml`

```yaml
# Custom Ghost-YOLO - Ultra lightweight
nc: 80

backbone:
    # 使用GhostConv替代Conv进行下采样
    - [-1, 1, GhostConv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, GhostConv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]] # 2

    - [-1, 1, GhostConv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]] # 4

    - [-1, 1, GhostConv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]] # 6

    - [-1, 1, GhostConv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]] # 8
    - [-1, 1, SPPF, [1024, 5]] # 9

head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 3, C2f, [512]] # 12

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]]
    - [-1, 3, C2f, [256]] # 15 (P3/8-small)

    - [-1, 1, GhostConv, [256, 3, 2]] # 使用Ghost下采样
    - [[-1, 12], 1, Concat, [1]]
    - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

    - [-1, 1, GhostConv, [512, 3, 2]] # 使用Ghost下采样
    - [[-1, 9], 1, Concat, [1]]
    - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

    - [[15, 18, 21], 1, Detect, [nc]]
```

### 步骤2: 对比测试

```python
from ultralytics import YOLO

# 标准YOLOv8n
model_standard = YOLO("yolov8n.yaml")
print("Standard YOLOv8n:")
model_standard.info()

# Ghost-YOLO
model_ghost = YOLO("ultralytics/cfg/models/v8/yolov8-ghost-custom.yaml")
print("\nGhost-YOLO:")
model_ghost.info()

# 对比参数量和FLOPs
```

---

## 🎯 示例3: 添加CoordConv坐标卷积

### 步骤1: 实现CoordConv

**文件**: `ultralytics/nn/modules/conv.py`

在文件末尾添加:

```python
class CoordConv(nn.Module):
    """Coordinate Convolution adds position information to regular convolution.

    Reference: https://arxiv.org/abs/1807.03247

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size
        s (int): Stride
        p (int, optional): Padding
        g (int): Groups
        d (int): Dilation
        act (bool | nn.Module): Activation function
        with_r (bool): Whether to include radius coordinate

    Examples:
        >>> coord_conv = CoordConv(3, 64, k=3, s=2)
        >>> x = torch.randn(1, 3, 640, 640)
        >>> y = coord_conv(x)
        >>> print(y.shape)
        torch.Size([1, 64, 320, 320])
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, with_r=False):
        """Initialize CoordConv with coordinate information."""
        super().__init__()
        self.with_r = with_r
        # 坐标通道: x, y, 可选的 r (radius)
        extra_channels = 3 if with_r else 2
        self.conv = Conv(c1 + extra_channels, c2, k, s, p, g, d, act)

    def add_coords(self, x):
        """Add x, y (and optionally r) coordinate channels to input."""
        batch_size, _, height, width = x.size()
        device = x.device
        dtype = x.dtype

        # X坐标
        xx_channel = torch.arange(width, dtype=dtype, device=device)
        xx_channel = xx_channel.repeat(1, height, 1)
        xx_channel = xx_channel / (width - 1) * 2 - 1  # 归一化到[-1, 1]
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)

        # Y坐标
        yy_channel = torch.arange(height, dtype=dtype, device=device)
        yy_channel = yy_channel.repeat(1, width, 1).transpose(1, 2)
        yy_channel = yy_channel / (height - 1) * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([x, xx_channel, yy_channel], dim=1)

        if self.with_r:
            # 半径坐标
            rr = torch.sqrt(xx_channel**2 + yy_channel**2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

    def forward(self, x):
        """Forward pass with coordinate augmentation."""
        x = self.add_coords(x)
        return self.conv(x)
```

### 步骤2: 导出和注册

**在 `conv.py` 的 `__all__` 中**:

```python
__all__ = (
    "Conv",
    # ... 其他
    "CoordConv",
)
```

**在 `modules/__init__.py` 中**:

```python
from .conv import (
    # ... 其他
    CoordConv,
)

__all__ = (
    # ... 其他
    "CoordConv",
)
```

**在 `tasks.py` 中**:

```python
base_modules = frozenset(
    {
        # ... 其他
        CoordConv,
    }
)
```

### 步骤3: 创建配置并测试

**文件**: `ultralytics/cfg/models/v8/yolov8-coord.yaml`

```yaml
# YOLOv8 with CoordConv
nc: 80

backbone:
    - [-1, 1, CoordConv, [64, 3, 2, None, 1, 1, True, False]] # 使用CoordConv
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]

# 标准head配置...
head:
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
    - [[-1, 9], 1, Concat, [1]]
    - [-1, 3, C2f, [1024]]

    - [[15, 18, 21], 1, Detect, [nc]]
```

**测试**:

```python
import torch

from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8-coord.yaml")
model.info()

# 测试前向传播
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    y = model(x)
print(f"Output: {len(y)} tensors")
```

---

## 🎯 示例4: 组合多种改进

创建一个结合SE注意力、Ghost卷积和改进FPN的高级模型。

### 配置文件

**文件**: `ultralytics/cfg/models/v8/yolov8-advanced.yaml`

```yaml
# Advanced YOLOv8 with multiple enhancements
nc: 80

backbone:
    # 使用Ghost减少参数
    - [-1, 1, GhostConv, [64, 3, 2]]
    - [-1, 1, GhostConv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, SEAttention, [128, 16]] # SE注意力

    - [-1, 1, GhostConv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, CBAM, [256]] # CBAM注意力

    - [-1, 1, Conv, [512, 3, 2]]
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, SEAttention, [512, 16]]

    - [-1, 1, Conv, [1024, 3, 2]]
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]
    - [-1, 1, CBAM, [1024]] # CBAM注意力

head:
    # BiFPN风格的特征融合
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 9], 1, Concat, [1]]
    - [-1, 3, C2f, [512]]
    - [-1, 1, SEAttention, [512, 16]] # head中也加注意力

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 3, C2f, [256]]
    - [-1, 1, SEAttention, [256, 16]]

    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 16], 1, Concat, [1]]
    - [-1, 3, C2f, [512]]

    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 13], 1, Concat, [1]]
    - [-1, 3, C2f, [1024]]

    - [[19, 22, 25], 1, Detect, [nc]]
```

### 训练脚本

```python
import torch

from ultralytics import YOLO

# 检查CUDA可用性
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载模型
model = YOLO("ultralytics/cfg/models/v8/yolov8-advanced.yaml")

# 打印模型信息
print("\n=== Model Information ===")
model.info()

# 训练配置
train_config = {
    "data": "coco128.yaml",  # 使用COCO128进行快速测试
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "name": "yolov8-advanced",
    "device": device,
    "workers": 8,
    "optimizer": "Adam",
    "lr0": 0.001,
    "patience": 10,
    "save": True,
    "plots": True,
}

# 开始训练
print("\n=== Starting Training ===")
results = model.train(**train_config)

# 验证
print("\n=== Validation ===")
metrics = model.val()

print(f"\nmAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# 导出模型
print("\n=== Exporting Model ===")
model.export(format="onnx", dynamic=True, simplify=True)
```

---

## 📊 性能对比脚本

```python
import time

import torch

from ultralytics import YOLO


def benchmark_model(model_path, name, imgsz=640):
    """Benchmark a YOLO model."""
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {name}")
    print(f"{'=' * 50}")

    # 加载模型
    model = YOLO(model_path)
    model.info(verbose=False)

    # 准备输入
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    model.model.to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model.model(dummy_input)

    # 测速
    num_iterations = 100
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model.model(dummy_input)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / num_iterations * 1000  # ms
    fps = 1000 / avg_time

    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.1f}")

    return avg_time, fps


# 对比不同模型
models = {
    "YOLOv8n": "yolov8n.yaml",
    "YOLOv8n-SE": "ultralytics/cfg/models/v8/yolov8-se.yaml",
    "YOLOv8n-Ghost": "ultralytics/cfg/models/v8/yolov8-ghost-custom.yaml",
    "YOLOv8n-Advanced": "ultralytics/cfg/models/v8/yolov8-advanced.yaml",
}

results = {}
for name, path in models.items():
    try:
        avg_time, fps = benchmark_model(path, name)
        results[name] = {"time": avg_time, "fps": fps}
    except Exception as e:
        print(f"Error benchmarking {name}: {e}")

# 打印对比表
print(f"\n{'=' * 60}")
print(f"{'Model':<25} {'Time (ms)':<15} {'FPS':<10}")
print(f"{'=' * 60}")
for name, metrics in results.items():
    print(f"{name:<25} {metrics['time']:<15.2f} {metrics['fps']:<10.1f}")
```

---

## ✅ 验证清单

在完成修改后，使用以下清单验证：

- [ ] 模块能成功导入
- [ ] 模型能正常构建
- [ ] 前向传播无错误
- [ ] 参数量和FLOPs符合预期
- [ ] 能正常训练（至少1个epoch）
- [ ] 能正常验证
- [ ] 能导出为ONNX/TorchScript
- [ ] 推理速度可接受

---

## 📝 总结

通过这些实战示例，你应该能够：

1. ✅ 定义新的卷积和注意力模块
2. ✅ 正确注册模块到解析器
3. ✅ 创建自定义YAML配置
4. ✅ 训练和测试修改后的模型
5. ✅ 进行性能对比和优化

继续探索和实验，创建最适合你任务的YOLO模型！
