# 3. 修改卷积层实现指南

本文档详细说明如何修改YOLO模型中的卷积层实现。

## 🎯 修改场景

### 场景1: 修改现有Conv类的行为

### 场景2: 创建新的卷积变体

### 场景3: 替换YAML中的卷积层

---

## 📝 场景1: 修改现有Conv类

### 示例1.1: 修改默认激活函数

**目标**: 将默认的SiLU激活函数改为Mish

**修改文件**: `ultralytics/nn/modules/conv.py`

**步骤**:

1. 找到Conv类定义（第38行）:

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # 原始代码
```

2. 修改为:

```python
class Conv(nn.Module):
    default_act = nn.Mish()  # 修改后
```

**影响范围**: 所有使用Conv的地方都会使用Mish激活函数

### 示例1.2: 修改padding计算方式

**目标**: 使用自定义的padding策略

**修改文件**: `ultralytics/nn/modules/conv.py`

**原始的autopad函数**（第29-35行）:

```python
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
```

**修改示例 - 添加额外的padding**:

```python
def autopad(k, p=None, d=1, extra_pad=0):
    """Pad to 'same' shape outputs with optional extra padding."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    # 添加额外的padding
    if isinstance(p, int):
        p += extra_pad
    else:
        p = [x + extra_pad for x in p]
    return p
```

**然后修改Conv.**init****:

```python
def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, extra_pad=0):
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d, extra_pad), groups=g, dilation=d, bias=False)
    # ... 其余代码
```

### 示例1.3: 添加Dropout

**目标**: 在Conv后添加Dropout层

**修改Conv类**:

```python
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()  # 新增

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))  # 修改
```

---

## 🆕 场景2: 创建新的卷积变体

### 示例2.1: 创建CoordConv（坐标卷积）

**目标**: 添加坐标信息到卷积

**步骤1**: 在 `ultralytics/nn/modules/conv.py` 末尾添加新类:

```python
class CoordConv(nn.Module):
    """CoordConv adds coordinate information to convolution. Paper: https://arxiv.org/abs/1807.03247.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, with_r=False):
        """
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Kernel size
            s (int): Stride
            p (int): Padding
            g (int): Groups
            d (int): Dilation
            act (bool|nn.Module): Activation
            with_r (bool): Whether to add radius channel.
        """
        super().__init__()
        # 额外的坐标通道: x, y, (可选)r
        extra_channels = 3 if with_r else 2
        self.with_r = with_r

        # 卷积层的输入通道数需要加上坐标通道
        self.conv = Conv(c1 + extra_channels, c2, k, s, p, g, d, act)

    def add_coords(self, x):
        """Add coordinate channels to input tensor."""
        batch_size, _, height, width = x.size()

        # 生成x坐标
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device)
        xx_channel = xx_channel.repeat(1, height, 1)
        xx_channel = xx_channel / (width - 1)  # 归一化到[0, 1]
        xx_channel = xx_channel * 2 - 1  # 归一化到[-1, 1]
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)

        # 生成y坐标
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device)
        yy_channel = yy_channel.repeat(1, width, 1).transpose(1, 2)
        yy_channel = yy_channel / (height - 1)
        yy_channel = yy_channel * 2 - 1
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([x, xx_channel, yy_channel], dim=1)

        if self.with_r:
            # 生成半径通道
            rr = torch.sqrt(xx_channel**2 + yy_channel**2)
            ret = torch.cat([ret, rr], dim=1)

        return ret

    def forward(self, x):
        """Forward pass with coordinate information."""
        x = self.add_coords(x)
        return self.conv(x)
```

**步骤2**: 在 `ultralytics/nn/modules/conv.py` 的 `__all__` 中添加:

```python
__all__ = (
    "Conv",
    "Conv2",
    # ... 其他
    "CoordConv",  # 新增
    "DWConv",
    "LightConv",
)
```

**步骤3**: 在 `ultralytics/nn/modules/__init__.py` 中导入:

```python
from .conv import (
    # ... 其他导入
    CoordConv,  # 新增
)

__all__ = (
    # ... 其他
    "CoordConv",  # 新增
)
```

**步骤4**: 在 `ultralytics/nn/tasks.py` 的 `parse_model` 函数中注册（如果需要特殊处理）:

在 `base_modules` frozenset 中添加（第1613-1654行）:

```python
base_modules = frozenset(
    {
        Classify,
        Conv,
        # ... 其他
        CoordConv,  # 新增
    }
)
```

**步骤5**: 在YAML配置中使用:

```yaml
backbone:
    - [-1, 1, CoordConv, [64, 3, 2]] # 使用CoordConv替代Conv
    - [-1, 1, Conv, [128, 3, 2]]
    # ...
```

### 示例2.2: 创建OctaveConv（八度卷积）

**定义**: 在 `ultralytics/nn/modules/conv.py` 中添加:

```python
class OctaveConv(nn.Module):
    """Octave Convolution splits features into high and low frequency. Paper: https://arxiv.org/abs/1904.05049.
    """

    def __init__(self, c1, c2, k=3, s=1, alpha_in=0.5, alpha_out=0.5, act=True):
        """
        Args:
            c1 (int): Input channels
            c2 (int): Output channels
            k (int): Kernel size
            s (int): Stride
            alpha_in (float): Ratio of low-freq input channels
            alpha_out (float): Ratio of low-freq output channels
            act (bool|nn.Module): Activation.
        """
        super().__init__()

        # 计算高低频通道数
        self.h_in = int(c1 * (1 - alpha_in))
        self.l_in = c1 - self.h_in
        self.h_out = int(c2 * (1 - alpha_out))
        self.l_out = c2 - self.h_out

        # 四个卷积分支: H->H, H->L, L->H, L->L
        self.conv_h2h = Conv(self.h_in, self.h_out, k, s, act=act) if self.h_out > 0 else None
        self.conv_h2l = Conv(self.h_in, self.l_out, k, s, act=act) if self.l_out > 0 else None
        self.conv_l2h = Conv(self.l_in, self.h_out, k, s, act=act) if self.h_out > 0 else None
        self.conv_l2l = Conv(self.l_in, self.l_out, k, s, act=act) if self.l_out > 0 else None

        self.pool = nn.AvgPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        """Forward pass with high and low frequency separation."""
        # 如果输入是元组（高频，低频），否则分割
        if isinstance(x, tuple):
            x_h, x_l = x
        else:
            x_h, x_l = x.split([self.h_in, self.l_in], dim=1)

        # H -> H
        h2h = self.conv_h2h(x_h) if self.conv_h2h is not None else None

        # H -> L (需要下采样)
        h2l = self.conv_h2l(self.pool(x_h)) if self.conv_h2l is not None else None

        # L -> H (需要上采样)
        l2h = self.upsample(self.conv_l2h(x_l)) if self.conv_l2h is not None else None

        # L -> L
        l2l = self.conv_l2l(x_l) if self.conv_l2l is not None else None

        # 合并高低频特征
        out_h = h2h + l2h if (h2h is not None and l2h is not None) else (h2h if h2h is not None else l2h)
        out_l = h2l + l2l if (h2l is not None and l2l is not None) else (h2l if h2l is not None else l2l)

        # 返回元组或拼接
        if out_h is not None and out_l is not None:
            return torch.cat([out_h, out_l], dim=1)
        return out_h if out_h is not None else out_l
```

---

## 🔄 场景3: 在YAML中替换卷积层

### 示例3.1: 替换backbone中的所有Conv

**原始YAML** (`ultralytics/cfg/models/v8/yolov8.yaml`):

```yaml
backbone:
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
```

**修改后 - 使用CoordConv**:

```yaml
backbone:
    - [-1, 1, CoordConv, [64, 3, 2]] # 使用CoordConv
    - [-1, 1, CoordConv, [128, 3, 2]] # 使用CoordConv
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, CoordConv, [256, 3, 2]] # 使用CoordConv
```

### 示例3.2: 只替换下采样层

**策略**: 仅在stride>1的地方使用特殊卷积

```yaml
backbone:
    - [-1, 1, CoordConv, [64, 3, 2]] # 下采样 - 使用CoordConv
    - [-1, 1, Conv, [128, 3, 2]] # 下采样 - 普通Conv
    - [-1, 3, C2f, [128, True]] # 特征提取
    - [-1, 1, Conv, [256, 3, 2]] # 下采样
```

---

## 🔧 高级修改技巧

### 技巧1: 混合使用多种卷积

创建一个自适应选择卷积类型的包装器:

```python
class AdaptiveConv(nn.Module):
    """Adaptively choose convolution type based on input size."""

    def __init__(self, c1, c2, k=1, s=1, conv_type="auto", **kwargs):
        super().__init__()

        if conv_type == "auto":
            # 小通道数用普通卷积，大通道数用DW卷积
            if c1 < 64:
                self.conv = Conv(c1, c2, k, s, **kwargs)
            else:
                self.conv = DWConv(c1, c2, k, s, **kwargs)
        elif conv_type == "coord":
            self.conv = CoordConv(c1, c2, k, s, **kwargs)
        elif conv_type == "ghost":
            self.conv = GhostConv(c1, c2, k, s, **kwargs)
        else:
            self.conv = Conv(c1, c2, k, s, **kwargs)

    def forward(self, x):
        return self.conv(x)
```

### 技巧2: 可切换的卷积实现

创建配置开关来选择卷积类型:

```python
# 在 ultralytics/nn/modules/conv.py 顶部添加
CONV_BACKEND = "standard"  # 'standard', 'coord', 'octave', etc.


class FlexibleConv(nn.Module):
    """Flexible convolution that can switch backend."""

    def __init__(self, c1, c2, k=1, s=1, **kwargs):
        super().__init__()

        if CONV_BACKEND == "coord":
            self.conv = CoordConv(c1, c2, k, s, **kwargs)
        elif CONV_BACKEND == "octave":
            self.conv = OctaveConv(c1, c2, k, s, **kwargs)
        else:
            self.conv = Conv(c1, c2, k, s, **kwargs)

    def forward(self, x):
        return self.conv(x)
```

使用时:

```python
# 在训练脚本开头设置
from ultralytics.nn.modules import conv

conv.CONV_BACKEND = "coord"
```

---

## ✅ 验证修改

### 步骤1: 测试模块能否正确导入

```python
import torch

from ultralytics.nn.modules import CoordConv

# 创建测试输入
x = torch.randn(1, 3, 640, 640)

# 实例化模块
conv = CoordConv(3, 64, k=3, s=2)

# 前向传播
y = conv(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
# 预期: Input shape: torch.Size([1, 3, 640, 640])
#       Output shape: torch.Size([1, 64, 320, 320])
```

### 步骤2: 测试YAML模型能否构建

```python
from ultralytics import YOLO

# 创建自定义YAML
model = YOLO("path/to/your/custom.yaml")
model.info()  # 查看模型信息
```

### 步骤3: 测试训练

```python
# 小规模测试
model.train(data="coco8.yaml", epochs=1, imgsz=640)
```

---

## 📝 最佳实践

1. **保持向后兼容**: 添加新参数时设置默认值
2. **详细注释**: 说明新卷积的原理和用法
3. **单元测试**: 为新模块编写测试
4. **性能对比**: 对比新旧卷积的速度和精度
5. **文档更新**: 更新README或文档说明新功能

## 🚨 常见问题

### Q1: 修改后模型无法加载

**A**: 确保在 `__init__.py` 和 `tasks.py` 中正确注册新模块

### Q2: YAML中使用新模块报错

**A**: 检查模块名是否在 `__all__` 中导出

### Q3: 通道数不匹配

**A**: 注意某些卷积（如CoordConv）会改变输入通道数

---

下一步，请阅读 [添加注意力层指南](./04-adding-attention.md) 学习如何添加注意力机制。
