# 4. 添加注意力层指南

本文档详细说明如何在YOLO模型中添加和自定义注意力机制。

## 🎯 注意力机制概览

YOLO模型已经内置了多种注意力机制，我们将学习如何使用它们以及如何创建自定义注意力层。

### 现有的注意力机制

| 注意力类型       | 文件位置               | 主要用途                     |
| ---------------- | ---------------------- | ---------------------------- |
| ChannelAttention | conv.py 第261行        | 通道注意力（类似SENet）      |
| SpatialAttention | conv.py 第291行        | 空间注意力                   |
| CBAM             | conv.py 第330行        | 通道+空间注意力              |
| PSA              | block.py 第1854行      | Position-Sensitive Attention |
| C2fAttn          | block.py 第305行       | 带注意力的C2f模块            |
| ImagePoolingAttn | block.py 第346行       | 图像池化注意力               |
| TransformerBlock | transformer.py 第142行 | Self-attention机制           |
| AIFI             | transformer.py 第181行 | 注意力特征融合               |

---

## 📦 使用现有注意力机制

### 示例1: 在backbone中添加CBAM

**CBAM (Convolutional Block Attention Module)** 是一个轻量级注意力模块，包含通道和空间注意力。

**步骤1**: 查看CBAM的定义 (`ultralytics/nn/modules/conv.py` 第330行):

```python
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
```

**步骤2**: 在YAML配置中使用:

**原始YOLOv8配置**:

```yaml
backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
```

**添加CBAM后**:

```yaml
backbone:
    - [-1, 1, Conv, [64, 3, 2]]
    - [-1, 1, Conv, [128, 3, 2]]
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, CBAM, [128]] # 在C2f后添加CBAM
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, CBAM, [256]] # 在C2f后添加CBAM
```

**注意**: CBAM不改变通道数，所以可以直接插入。

### 示例2: 使用C2fAttn替代C2f

**C2fAttn** 是C2f的增强版本，内置了注意力机制。

**原始**:

```yaml
- [-1, 6, C2f, [512, True]]
```

**替换为C2fAttn**:

```yaml
# C2fAttn参数: [c2, n, ec, nh, gc, shortcut]
# ec: embedding channels (128)
# nh: number of heads (1)
# gc: global context channels (512)
- [-1, 6, C2fAttn, [512, True, 128, 1, 512]]
```

### 示例3: 添加PSA（Position-Sensitive Attention）

**PSA** 适用于需要位置敏感的注意力场景。

```yaml
backbone:
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, PSA, [256]] # 添加PSA
```

**PSA参数说明**:

```python
PSA(c1, c2=None, e=0.5)
# c1: 输入通道数
# c2: 输出通道数（默认等于c1）
# e: expansion ratio
```

---

## 🆕 创建自定义注意力机制

### 自定义注意力1: Squeeze-and-Excitation (SE) 注意力

**步骤1**: 在 `ultralytics/nn/modules/block.py` 中添加SE模块:

```python
class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention module. Paper: https://arxiv.org/abs/1709.01507.
    """

    def __init__(self, channels, reduction=16):
        """Initialize SE attention.

        Args:
            channels (int): Number of input channels
            reduction (int): Reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply SE attention to input tensor."""
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Element-wise multiplication
        return x * y.expand_as(x)
```

**步骤2**: 在 `ultralytics/nn/modules/block.py` 的 `__all__` 中添加:

```python
__all__ = (
    # ... 其他模块
    "SEAttention",  # 新增
)
```

**步骤3**: 在 `ultralytics/nn/modules/__init__.py` 中导入:

```python
from .block import (
    # ... 其他导入
    SEAttention,  # 新增
)

__all__ = (
    # ... 其他
    "SEAttention",  # 新增
)
```

**步骤4**: 在YAML中使用:

```yaml
backbone:
    - [-1, 1, Conv, [256, 3, 2]]
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, SEAttention, [256, 16]] # [channels, reduction]
```

### 自定义注意力2: Efficient Channel Attention (ECA)

**ECA** 是SE的改进版本，使用1D卷积替代全连接层。

**步骤1**: 在 `ultralytics/nn/modules/block.py` 中添加:

```python
class ECAAttention(nn.Module):
    """Efficient Channel Attention. Paper: https://arxiv.org/abs/1910.03151.
    """

    def __init__(self, channels, gamma=2, b=1):
        """Initialize ECA attention.

        Args:
            channels (int): Number of input channels
            gamma (int): Parameter for adaptive kernel size
            b (int): Parameter for adaptive kernel size
        """
        super().__init__()
        # 自适应计算卷积核大小
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Apply ECA attention to input tensor."""
        # Feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
```

**需要在文件开头导入**:

```python

```

### 自定义注意力3: Coordinate Attention (CA)

**CA** 同时考虑通道和空间信息，特别适合移动网络。

**步骤1**: 在 `ultralytics/nn/modules/block.py` 中添加:

```python
class CoordAttention(nn.Module):
    """Coordinate Attention for efficient mobile network design. Paper: https://arxiv.org/abs/2103.02907.
    """

    def __init__(self, inp, oup, reduction=32):
        """Initialize Coordinate Attention.

        Args:
            inp (int): Input channels
            oup (int): Output channels
            reduction (int): Reduction ratio
        """
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Apply Coordinate Attention."""
        identity = x

        _n, _c, h, w = x.size()
        # X方向池化
        x_h = self.pool_h(x)
        # Y方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接并编码
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分割并解码
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 应用注意力
        out = identity * a_w * a_h

        return out
```

---

## 🔄 在不同位置添加注意力

### 位置1: 在卷积层之后

```yaml
- [-1, 1, Conv, [256, 3, 2]]
- [-1, 1, SEAttention, [256]] # 卷积后添加注意力
```

### 位置2: 在C2f模块之后

```yaml
- [-1, 6, C2f, [512, True]]
- [-1, 1, CBAM, [512]] # C2f后添加注意力
```

### 位置3: 在SPPF之后（backbone末尾）

```yaml
- [-1, 1, SPPF, [1024, 5]]
- [-1, 1, CoordAttention, [1024, 1024]] # SPPF后添加注意力
```

### 位置4: 在head中的特征融合处

```yaml
head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]]
    - [-1, 3, C2f, [512]]
    - [-1, 1, ECAAttention, [512]] # 融合后添加注意力
```

---

## 🏗️ 创建带注意力的复合模块

### 示例: C2fSE - 带SE注意力的C2f

**步骤1**: 在 `ultralytics/nn/modules/block.py` 中定义:

```python
class C2fSE(C2f):
    """C2f module with Squeeze-and-Excitation attention."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        """Initialize C2f with SE attention."""
        super().__init__(c1, c2, n, shortcut, g, e)
        # 在输出后添加SE注意力
        self.se = SEAttention(c2, reduction)

    def forward(self, x):
        """Forward pass with SE attention."""
        y = super().forward(x)  # 调用C2f的forward
        return self.se(y)  # 应用SE注意力
```

**步骤2**: 注册并在YAML中使用:

```yaml
backbone:
    - [-1, 3, C2fSE, [256, True, 1, False, 1, 0.5, 16]]
    # 参数: [c2, shortcut, n, shortcut, g, e, reduction]
```

### 示例: ConvCBAM - 带CBAM的卷积

```python
class ConvCBAM(nn.Module):
    """Convolution followed by CBAM attention."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, kernel_size=7):
        """Initialize Conv + CBAM."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.cbam = CBAM(c2, kernel_size)

    def forward(self, x):
        """Forward pass through Conv and CBAM."""
        return self.cbam(self.conv(x))
```

---

## 📊 注意力机制性能对比

### 计算复杂度对比

| 注意力类型     | 参数量 | 计算量 | 推理速度 | 精度提升 |
| -------------- | ------ | ------ | -------- | -------- |
| SE             | 低     | 低     | 快       | 中等     |
| CBAM           | 低     | 低     | 快       | 中等     |
| ECA            | 极低   | 极低   | 极快     | 中等     |
| CoordAttention | 低     | 中等   | 中等     | 高       |
| PSA            | 中等   | 中等   | 中等     | 高       |
| Transformer    | 高     | 高     | 慢       | 高       |

### 适用场景建议

1. **移动端/边缘设备**:
    - 优先选择: ECA, SE
    - 避免: Transformer, PSA

2. **服务器端/高精度要求**:
    - 推荐: CoordAttention, PSA, Transformer
3. **平衡性能和速度**:
    - 推荐: CBAM, C2fAttn

---

## 🎨 组合多种注意力

### 串联注意力

```python
class MultiAttention(nn.Module):
    """Combine multiple attention mechanisms."""

    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
        self.se_attn = SEAttention(channels)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.se_attn(x)
        return x
```

### 并联注意力（加权融合）

```python
class ParallelAttention(nn.Module):
    """Parallel attention with weighted fusion."""

    def __init__(self, channels):
        super().__init__()
        self.cbam = CBAM(channels)
        self.se = SEAttention(channels)
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        cbam_out = self.cbam(x)
        se_out = self.se(x)
        return self.alpha * cbam_out + (1 - self.alpha) * se_out
```

---

## ✅ 验证注意力模块

### 测试代码

```python
import torch

from ultralytics.nn.modules import CoordAttention, ECAAttention, SEAttention

# 创建测试输入
x = torch.randn(2, 256, 40, 40)  # [batch, channels, height, width]

# 测试SE注意力
se = SEAttention(256, reduction=16)
y_se = se(x)
print(f"SE output shape: {y_se.shape}")  # 应该和输入相同

# 测试ECA注意力
eca = ECAAttention(256)
y_eca = eca(x)
print(f"ECA output shape: {y_eca.shape}")

# 测试Coord注意力
coord = CoordAttention(256, 256)
y_coord = coord(x)
print(f"Coord output shape: {y_coord.shape}")

# 验证输出范围
print(f"SE output range: [{y_se.min().item():.2f}, {y_se.max().item():.2f}]")
```

---

## 📝 最佳实践

1. **渐进式添加**: 先添加一个注意力模块，训练测试后再考虑添加更多
2. **位置选择**: 通常在特征提取块之后添加效果最好
3. **参数调优**: 注意力的reduction参数需要根据通道数调整
4. **避免过度**: 过多注意力模块会降低速度而不一定提升精度
5. **对比实验**: 始终与baseline对比，确保改进有效

---

## 🚨 常见问题

### Q1: 添加注意力后精度下降？

**A**: 可能是注意力参数设置不当，尝试调整reduction ratio或kernel size

### Q2: 推理速度明显变慢？

**A**: 避免使用过多复杂注意力，考虑使用ECA等轻量级方案

### Q3: 注意力模块不起作用？

**A**: 确保注意力的输出被正确使用，检查forward函数的实现

---

下一步，请阅读 [模型配置文件详解](./05-yaml-configuration.md) 学习如何通过YAML配置文件灵活定义模型。
