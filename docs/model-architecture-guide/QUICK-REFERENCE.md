# 快速参考：YOLO模型修改要点

本文档提供快速查找的要点总结。

## 🗺️ 关键文件速查表

| 文件路径 | 主要作用 | 修改频率 |
|---------|---------|---------|
| `ultralytics/nn/modules/conv.py` | 卷积层实现 | ⭐⭐⭐⭐ |
| `ultralytics/nn/modules/block.py` | 构建块实现 | ⭐⭐⭐⭐⭐ |
| `ultralytics/nn/modules/transformer.py` | 注意力机制 | ⭐⭐⭐ |
| `ultralytics/nn/modules/head.py` | 检测头 | ⭐⭐ |
| `ultralytics/nn/modules/__init__.py` | 模块导出 | ⭐⭐⭐⭐⭐ |
| `ultralytics/nn/tasks.py` | 模型解析和构建 | ⭐⭐⭐⭐⭐ |
| `ultralytics/cfg/models/v8/*.yaml` | 模型配置 | ⭐⭐⭐⭐⭐ |

## 📋 修改流程清单

### 添加新卷积层

- [ ] 1. 在 `conv.py` 中定义新类
- [ ] 2. 在 `conv.py` 的 `__all__` 中添加
- [ ] 3. 在 `modules/__init__.py` 中导入和导出
- [ ] 4. 在 `tasks.py` 的 `base_modules` 中注册
- [ ] 5. 在YAML配置中使用
- [ ] 6. 测试模型构建和前向传播

### 添加新注意力机制

- [ ] 1. 在 `block.py` 或 `transformer.py` 中定义
- [ ] 2. 在对应文件的 `__all__` 中添加
- [ ] 3. 在 `modules/__init__.py` 中导入和导出
- [ ] 4. 在 `tasks.py` 中注册（如需要）
- [ ] 5. 在YAML配置中使用
- [ ] 6. 测试并验证效果

### 修改YAML配置

- [ ] 1. 复制现有配置文件
- [ ] 2. 修改层定义
- [ ] 3. 更新索引引用
- [ ] 4. 验证通道数匹配
- [ ] 5. 测试模型构建
- [ ] 6. 训练验证

## 🔧 常用代码片段

### 创建简单卷积模块

```python
class MyConv(nn.Module):
    """自定义卷积模块."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
    
    def forward(self, x):
        return self.conv(x)
```

### 创建注意力模块

```python
class MyAttention(nn.Module):
    """自定义注意力模块."""
    
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

### 测试模块

```python
import torch
from ultralytics.nn.modules import MyModule

x = torch.randn(1, 256, 20, 20)
module = MyModule(256)
y = module(x)
print(f"Input: {x.shape}, Output: {y.shape}")
```

### 测试配置文件

```python
from ultralytics import YOLO

model = YOLO('path/to/config.yaml')
model.info()
```

## 📍 YAML配置语法速查

### 基本层定义

```yaml
# [from, repeats, module, args]
- [-1, 1, Conv, [64, 3, 2]]           # 标准卷积
- [-1, 3, C2f, [256, True]]           # C2f模块
- [-1, 1, SPPF, [1024, 5]]            # SPPF模块
- [-1, 1, CBAM, [256]]                # CBAM注意力
```

### 特殊操作

```yaml
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 上采样
- [[-1, 6], 1, Concat, [1]]                   # 拼接多层
- [[15, 18, 21], 1, Detect, [nc]]             # 检测头
```

### 索引规则

- `-1`: 上一层
- `6`: 第6层（从0开始）
- `[-1, 6]`: 多个输入
- 添加层后记得更新索引！

## 🎯 性能优化技巧

### 减少参数量

```yaml
# 使用DWConv或GhostConv替代Conv
- [-1, 1, DWConv, [256, 3, 2]]
- [-1, 1, GhostConv, [256, 3, 2]]
```

### 提升精度

```yaml
# 添加注意力机制
- [-1, 1, CBAM, [256]]
- [-1, 1, SEAttention, [256, 16]]
```

### 增加感受野

```yaml
# 使用SPPF或SPP
- [-1, 1, SPPF, [1024, 5]]
# 或增加空洞卷积
- [-1, 1, Conv, [256, 3, 1, None, 1, 2]]  # d=2
```

## 🐛 调试技巧

### 打印中间层输出

```python
def forward_hook(module, input, output):
    print(f"{module.__class__.__name__}: {output.shape}")

model = YOLO('config.yaml')
for layer in model.model.model:
    layer.register_forward_hook(forward_hook)

x = torch.randn(1, 3, 640, 640)
model(x)
```

### 检查梯度流

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm():.4f}")
        else:
            print(f"{name}: No gradient!")
```

### 可视化模型

```python
from ultralytics import YOLO

model = YOLO('config.yaml')
model.export(format='onnx')
# 使用 https://netron.app/ 查看
```

## ⚠️ 常见错误及解决

### 错误1: 模块未找到

```
NameError: name 'MyModule' is not defined
```

**解决**: 检查是否在 `__init__.py` 中导入

### 错误2: 通道数不匹配

```
RuntimeError: size mismatch
```

**解决**: 检查Concat操作的通道数是否正确

### 错误3: 索引错误

```
IndexError: list index out of range
```

**解决**: 添加/删除层后更新YAML中的索引

### 错误4: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决**:
- 减小batch size
- 减小图像尺寸
- 使用梯度累积

## 📊 模块参数对比

| 模块 | 输入 | 输出 | 参数 | 说明 |
|------|-----|------|------|------|
| Conv | [c1] | [c2, k, s] | c1, c2, k, s, p, g, d, act | 标准卷积 |
| DWConv | [c1] | [c2, k, s] | c1, c2, k, s | 深度卷积 |
| C2f | [c1] | [c2, shortcut] | c1, c2, n, shortcut | YOLOv8核心 |
| SPPF | [c1] | [c2, k] | c1, c2, k | 空间池化 |
| CBAM | [c1] | [kernel_size] | c1, kernel_size | 注意力 |
| Detect | [ch] | [nc] | nc, ch | 检测头 |

## 🔗 有用资源

### 官方文档
- Ultralytics文档: https://docs.ultralytics.com/
- YOLOv8模型: https://docs.ultralytics.com/models/yolov8/

### 论文参考
- YOLOv8: https://github.com/ultralytics/ultralytics
- CBAM: https://arxiv.org/abs/1807.06521
- SE-Net: https://arxiv.org/abs/1709.01507
- GhostNet: https://arxiv.org/abs/1911.11907
- CoordConv: https://arxiv.org/abs/1807.03247

### 工具
- Netron (模型可视化): https://netron.app/
- ONNX Runtime: https://onnxruntime.ai/
- TensorBoard: https://www.tensorflow.org/tensorboard

## 💡 最佳实践

1. **渐进式修改**: 一次只改一处
2. **保留备份**: 修改前备份原始文件
3. **详细注释**: 说明修改的目的和原理
4. **测试驱动**: 先写测试，再修改
5. **性能对比**: 与baseline对比效果
6. **版本控制**: 使用git跟踪所有改动
7. **文档更新**: 记录所有重要修改

## 🎓 学习路径建议

### 初学者
1. 阅读 [模型组织结构概览](./01-model-organization.md)
2. 运行官方示例，理解基本用法
3. 修改简单的YAML配置（如改通道数）
4. 测试不同的预训练模型

### 中级
1. 阅读 [核心模块详解](./02-core-modules.md)
2. 尝试添加现有的注意力机制
3. 修改backbone结构
4. 创建自定义配置文件

### 高级
1. 阅读 [修改卷积层指南](./03-modifying-conv-layers.md)
2. 阅读 [添加注意力层指南](./04-adding-attention.md)
3. 实现自定义模块
4. 创建新的网络架构
5. 优化训练和推理性能

## 📞 获取帮助

遇到问题时：

1. **检查文档**: 先查看本指南相关章节
2. **搜索Issues**: 在GitHub仓库搜索类似问题
3. **查看源码**: 阅读相关模块的源代码
4. **社区求助**: 在论坛或Discord提问
5. **调试工具**: 使用pdb或IDE调试器

---

**祝你在YOLO模型修改和优化的道路上取得成功！🚀**
