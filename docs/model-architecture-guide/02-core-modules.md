# 2. 核心模块详解

本文档详细解释YOLO模型中各个核心模块的实现和作用。

## 📦 卷积模块 (conv.py)

### Conv - 标准卷积块

**位置**: `ultralytics/nn/modules/conv.py` 第38-93行

**结构**:

```
Conv = Conv2d + BatchNorm2d + Activation
```

**源码解析**:

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        # c1: 输入通道数
        # c2: 输出通道数
        # k: 卷积核大小
        # s: 步长
        # p: padding（自动计算如果为None）
        # g: 分组数
        # d: 膨胀率
        # act: 激活函数
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

**使用场景**:

- YOLO模型中最基础的卷积单元
- 几乎所有层都基于此构建
- 下采样、特征提取等

**修改示例**:

```python
# 修改默认激活函数为Mish
Conv.default_act = nn.Mish()

# 或者在初始化时指定
conv = Conv(64, 128, k=3, s=2, act=nn.Mish())
```

### DWConv - 深度可分离卷积

**位置**: `ultralytics/nn/modules/conv.py` 第139-152行

**特点**:

- 使用分组卷积（groups=输入通道数）
- 减少参数量和计算量
- 常用于轻量级模型

**源码**:

```python
class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
```

### GhostConv - Ghost卷积

**位置**: `ultralytics/nn/modules/conv.py` 第170-192行

**原理**:

- 先用少量卷积生成特征
- 再用cheap操作（如DW卷积）生成更多特征
- 显著减少计算量

**应用**: YOLOv8-ghost模型

### ChannelAttention & SpatialAttention

**位置**: `ultralytics/nn/modules/conv.py` 第261-326行

**ChannelAttention** - 通道注意力:

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)  # 1x1卷积
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))  # 加权原始特征
```

**SpatialAttention** - 空间注意力:

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
```

**CBAM** - 结合通道和空间注意力:

```python
class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))
```

---

## 🧱 构建块模块 (block.py)

### C2f - YOLOv8的核心模块

**位置**: `ultralytics/nn/modules/block.py` 第250-302行

**结构**:

```
C2f = Conv + n * Bottleneck + Conv + Concat
```

**详细解析**:

```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        # c1: 输入通道
        # c2: 输出通道
        # n: Bottleneck重复次数
        # shortcut: 是否使用残差连接
        # g: 分组数
        # e: 扩展比率（中间通道 = c2 * e）
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 1x1卷积扩展通道
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 1x1卷积压缩通道
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))  # 分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 依次通过Bottleneck
        return self.cv2(torch.cat(y, 1))  # 拼接并压缩
```

**特点**:

- 梯度分流设计，改善梯度流
- 比YOLOv5的C3模块更快
- 更适合大模型

**在YAML中使用**:

```yaml
- [-1, 3, C2f, [256, True]] # [from, n, module, [c2, shortcut]]
```

### Bottleneck - 瓶颈块

**位置**: `ultralytics/nn/modules/block.py` 第206-233行

**结构**:

```
Bottleneck = Conv(1x1) + Conv(3x3) + (可选的残差连接)
```

**源码**:

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道
        self.cv1 = Conv(c1, c_, k[0], 1)  # 降维
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 升维
        self.add = shortcut and c1 == c2  # 是否使用残差

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

### SPPF - 快速空间金字塔池化

**位置**: `ultralytics/nn/modules/block.py` 第152-178行

**原理**:

- 连续使用相同的池化核，而非并行多个不同尺寸
- 达到类似SPP的效果但更快

**源码**:

```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)  # 降维
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 升维
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
```

**效果**: 三次5x5池化 ≈ 一次5x5、9x9、13x13并行池化

### C2fAttn - 带注意力的C2f

**位置**: `ultralytics/nn/modules/block.py` 第305-342行

**增强**:

- 在C2f基础上添加注意力机制
- 可以使用通道注意力或其他注意力变体

**结构**:

```python
class C2fAttn(nn.Module):
    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        # ec: 嵌入通道数
        # nh: 注意力头数
        # gc: 全局上下文通道数
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = Attention(self.c, ec, nh, gc)  # 注意力层

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1]))  # 添加注意力输出
        return self.cv2(torch.cat(y, 1))
```

---

## 🔍 Transformer和注意力模块 (transformer.py)

### TransformerBlock

**位置**: `ultralytics/nn/modules/transformer.py` 第142-178行

**用途**:

- 在YOLO中引入self-attention机制
- 捕获长距离依赖

**源码简化**:

```python
class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None if c1 == c2 else Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)  # (w*h, b, c)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)
```

### AIFI - 注意力融合

**位置**: `ultralytics/nn/modules/transformer.py` 第181-210行

**特点**:

- 用于特征融合
- 可学习的注意力权重

---

## 🎯 检测头模块 (head.py)

### Detect - YOLO检测头

**位置**: `ultralytics/nn/modules/head.py` 第24-233行

**核心组件**:

```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=()):
        # nc: 类别数
        # ch: 输入通道元组（来自不同尺度）
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # 检测层数量
        self.reg_max = 16  # DFL通道数
        self.no = nc + self.reg_max * 4  # 每个anchor的输出数

        # 边界框回归分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )

        # 分类分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max)  # Distribution Focal Loss
```

**输出**:

- 边界框坐标 (4个值)
- 类别置信度 (nc个值)
- 多尺度预测（通常3个尺度：P3, P4, P5）

---

## 📊 各模块对比

| 模块             | 主要用途   | 参数量 | 计算量 | 特点       |
| ---------------- | ---------- | ------ | ------ | ---------- |
| Conv             | 基础卷积   | 中等   | 中等   | 标准构建块 |
| DWConv           | 轻量卷积   | 低     | 低     | 移动端优化 |
| C2f              | 特征提取   | 高     | 高     | YOLOv8核心 |
| C3               | 特征提取   | 高     | 高     | YOLOv5风格 |
| SPPF             | 多尺度池化 | 低     | 低     | 感受野增强 |
| TransformerBlock | 全局建模   | 高     | 极高   | 长距离依赖 |
| CBAM             | 注意力     | 低     | 低     | 特征增强   |

---

## 💡 模块选择建议

### 1. 需要轻量化模型

```yaml
# 使用DWConv替代Conv
- [-1, 1, DWConv, [256, 3, 2]]
# 使用GhostConv
- [-1, 1, GhostConv, [256, 3, 2]]
```

### 2. 需要提升性能

```yaml
# 添加注意力
- [-1, 1, CBAM, [256]]
# 使用C2fAttn
- [-1, 3, C2fAttn, [256]]
```

### 3. 需要大感受野

```yaml
# 使用SPPF
- [-1, 1, SPPF, [1024, 5]]
# 或者添加Transformer
- [-1, 1, TransformerBlock, [256, 4, 2]] # [c2, num_heads, num_layers]
```

---

## 📝 小结

1. **Conv系列**: 基础卷积操作的各种变体
2. **Block系列**: 复杂的特征提取模块，如C2f、C3
3. **Attention系列**: 各种注意力机制，增强特征表达
4. **Head系列**: 任务相关的输出头

下一步，请阅读 [修改卷积层指南](./03-modifying-conv-layers.md) 学习如何修改和自定义卷积层。
