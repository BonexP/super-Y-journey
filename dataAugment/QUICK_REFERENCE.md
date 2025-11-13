# 数据增强快速参考指南

## 📋 快速检查清单

使用此清单确保你的数据增强pipeline符合Albumentations最佳实践：

### ✅ 性能优化
- [ ] 在脚本开头添加 `cv2.setNumThreads(0)`
- [ ] 使用OpenCV (`cv2.imread`) 或torchvision读取图像（避免PIL）
- [ ] 图像保持`uint8`格式直到最后标准化
- [ ] 使用组合Transform（如`Affine`而非多个单独Transform）

### ✅ Pipeline结构（7步法）
- [ ] **Step 1**: 裁剪/尺寸标准化在第一步（`RandomCrop`或`SmallestMaxSize`）
- [ ] **Step 2**: 基础几何不变性（`HorizontalFlip`, `RandomRotate90`）
- [ ] **Step 3**: Dropout/遮挡增强（`CoarseDropout`, `RandomErasing`）
- [ ] **Step 4**: （可选）降低颜色依赖（`ToGray`, `ChannelDropout`）
- [ ] **Step 5**: 仿射变换（`Affine`用于scale+rotate）
- [ ] **Step 6**: 领域特定增强（颜色、模糊、噪声）
- [ ] **Step 7**: 最终标准化（`Normalize`，如果需要）

### ✅ 参数设置
- [ ] 概率参数合理（常用：0.3-0.7）
- [ ] 参数范围适中（不要过度变形）
- [ ] 使用`pad_if_needed=True`在裁剪Transform中

### ✅ 可视化验证
- [ ] 在[explore.albumentations.ai](https://explore.albumentations.ai/)测试你的图像
- [ ] 检查增强后的图像是否真实
- [ ] 确认bbox没有大量丢失

---

## 🎯 常见场景推荐

### 场景1: 目标检测（YOLO/Faster R-CNN）

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomCrop(height=640, width=640, p=1.0, pad_if_needed=True),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=5, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

### 场景2: 图像分类（ResNet/EfficientNet）

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=5, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

---

### 场景3: 语义分割（U-Net/DeepLab）

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomCrop(height=512, width=512, p=1.0, pad_if_needed=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=5, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

---

### 场景4: 医学图像分析

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomCrop(height=256, width=256, p=1.0, pad_if_needed=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.SquareSymmetry(p=0.5),  # 旋转对称
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # 组织变形
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussNoise(var_limit=(10, 30), p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),  # 单通道标准化
])
```

---

### 场景5: 航拍/卫星图像

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomCrop(height=512, width=512, p=1.0, pad_if_needed=True),
    A.SquareSymmetry(p=0.5),  # 8种对称变换
    A.CoarseDropout(max_holes=10, max_height=64, max_width=64, p=0.5),
    A.Affine(scale=(0.8, 1.2), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

### 场景6: 工业检测/缺陷检测

```python
import cv2
cv2.setNumThreads(0)

transform = A.Compose([
    A.RandomCrop(height=640, width=640, p=1.0, pad_if_needed=True),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),  # 光照变化大
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),  # 传感器噪声
    A.ISONoise(p=0.2),  # 相机噪声
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

---

## 📊 Transform效果速查表

| Transform | 影响目标 | 典型用途 | 推荐概率 | 影响力 |
|-----------|---------|---------|---------|--------|
| `RandomCrop` | 空间 | 裁剪+性能优化 | 1.0 | ⭐⭐⭐⭐⭐ |
| `HorizontalFlip` | 空间 | 基础几何 | 0.5 | ⭐⭐⭐⭐ |
| `CoarseDropout` | 像素 | 遮挡鲁棒性 | 0.3-0.5 | ⭐⭐⭐⭐⭐ |
| `Affine` | 空间 | 缩放+旋转 | 0.5-0.7 | ⭐⭐⭐⭐ |
| `RandomBrightnessContrast` | 像素 | 光照鲁棒性 | 0.5 | ⭐⭐⭐⭐ |
| `HueSaturationValue` | 像素 | 颜色变化 | 0.3 | ⭐⭐⭐ |
| `GaussianBlur` | 像素 | 模糊鲁棒性 | 0.2 | ⭐⭐⭐ |
| `GaussNoise` | 像素 | 噪声鲁棒性 | 0.2-0.3 | ⭐⭐⭐ |
| `RandomRotate90` | 空间 | 旋转对称 | 0.3-0.5 | ⭐⭐⭐⭐ |
| `ToGray` | 像素 | 颜色不变性 | 0.1 | ⭐⭐⭐ |
| `ElasticTransform` | 空间 | 非线性变形 | 0.2-0.3 | ⭐⭐⭐⭐ |

---

## ⚡ 性能优化速查

### 优化技巧及其影响

| 优化技巧 | 性能提升 | 实现难度 | 优先级 |
|---------|---------|---------|--------|
| 优先裁剪（小图处理） | 5-16x | 简单 | 🔥🔥🔥🔥🔥 |
| `cv2.setNumThreads(0)` | 2-3x | 极简单 | 🔥🔥🔥🔥🔥 |
| OpenCV读图（vs PIL） | 2-3x | 简单 | 🔥🔥🔥🔥 |
| 保持uint8格式 | 1.2-1.5x | 简单 | 🔥🔥🔥 |
| 组合Transform | 1.2-1.3x | 中等 | 🔥🔥 |
| GPU批量标准化 | 1.5-2x | 中等 | 🔥🔥 |

---

## 🔍 调试技巧

### 问题：增强后bbox大量丢失

**原因**:
- 旋转/缩放角度太大
- 裁剪区域太小
- 目标本来就在图像边缘

**解决方案**:
1. 降低旋转角度：`rotate=(-10, 10)` → `rotate=(-5, 5)`
2. 降低缩放范围：`scale=(0.8, 1.2)` → `scale=(0.9, 1.1)`
3. 使用`BBoxSafeRandomCrop`替代`RandomCrop`
4. 检查原始标注质量

---

### 问题：增强速度很慢

**诊断步骤**:
1. 检查是否优先裁剪 → 如果没有，**立即添加**
2. 检查`cv2.setNumThreads(0)` → 如果没有，**立即添加**
3. 检查图像读取方式 → 使用OpenCV而非PIL
4. 减少Transform数量 → 移除低影响力Transform

---

### 问题：模型没有提升

**检查项**:
1. **可视化检查** - 增强是否过度/不足？
2. **bbox保留率** - 是否大量bbox丢失？
3. **增强多样性** - 是否缺少关键Transform（如dropout）？
4. **参数范围** - 是否过于保守？
5. **训练epochs** - 增强后需要更多epochs

---

### 问题：训练不稳定/发散

**可能原因**:
1. 增强过于激进 → 降低概率和参数范围
2. 学习率过高 → 降低学习率
3. 某些Transform不适合数据 → 逐个移除测试

---

## 📚 推荐阅读顺序

### 初学者
1. [Transforms基础](https://albumentations.ai/docs/2-core-concepts/transforms/)
2. [可视化工具](https://explore.albumentations.ai/) - **上传你的图像测试！**
3. 本快速参考指南中的常见场景

### 进阶用户
1. [选择增强方法](https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/)
2. [性能优化](https://albumentations.ai/docs/3-basic-usage/performance-tuning/)
3. [自定义Transform](https://albumentations.ai/docs/4-advanced-guides/custom-targets/)

### 专家
1. [源码阅读](https://github.com/albumentations-team/albumentations)
2. [性能基准测试](https://albumentations.ai/docs/benchmarks/)
3. 贡献到社区

---

## 🎓 最佳实践总结

### 黄金法则
1. **先裁剪，后增强** - 性能提升16倍
2. **渐进式添加** - 一次一个，测试验证
3. **可视化检查** - 眼见为实
4. **监控指标** - 验证集性能是王道

### 常见错误
❌ 在大图上应用所有Transform  
❌ 跳过dropout增强  
❌ 参数范围过于激进或保守  
❌ 不可视化检查  
❌ 忘记`cv2.setNumThreads(0)`  
❌ 使用PIL读取图像  

### 正确做法
✅ 优先裁剪到目标尺寸  
✅ 包含高影响力dropout增强  
✅ 使用适中的参数范围  
✅ 上传真实图像到explore.albumentations.ai测试  
✅ 在脚本开头设置`cv2.setNumThreads(0)`  
✅ 使用OpenCV或torchvision读图  

---

## 🔗 有用的链接

- **可视化测试**: https://explore.albumentations.ai/
- **官方文档**: https://albumentations.ai/docs/
- **GitHub仓库**: https://github.com/albumentations-team/albumentations
- **Transform列表**: https://albumentations.ai/docs/api-reference/augmentations/
- **性能基准**: https://albumentations.ai/docs/benchmarks/
- **图像读取基准**: https://github.com/ternaus/imread_benchmark

---

## 💡 提示

> **记住**: 数据增强是一门艺术，也是一门科学。没有完美的配置，只有最适合你数据的配置。实验、可视化、测试、迭代！

> **性能优先**: 如果你的训练瓶颈在CPU（GPU利用率低），优先解决性能问题再考虑增加更多Transform。

> **效果优先**: 如果训练速度够快但模型效果不好，优先添加高影响力Transform（如dropout）。

---

**最后更新**: 2025-11-13  
**作者**: AI Assistant  
**基于**: Albumentations官方文档最佳实践

