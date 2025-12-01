# 数据增强 - 边界框丢失问题快速修复指南

## 🚀 快速开始

### 1️⃣ 选择增强模式

在 `data_augment_optimized.py` 的第 114 行左右：

```python
# 方式1：安全模式（推荐，100%保留bbox）
transform = transform_safe

# 方式2：激进模式（更多样化，可能丢失少量bbox）
# transform = transform_aggressive
```

### 2️⃣ 运行增强脚本

```bash
cd /home/user/projects/YOLO11/dataAugment
python data_augment_optimized.py
```

### 3️⃣ 查看统计信息

增强完成后会显示：
```
📊 数据统计:
  - 原始图像: 1000
  - 成功增强: 3000
  - 总图像数: 4000

📦 Bbox保留统计:
  - 原始bbox总数: 5000
  - 保留bbox总数: 4925
  - 总体保留率: 98.50%
  - 完全丢失bbox的增强: 5
  - 部分丢失bbox的增强: 70
```

### 4️⃣ 验证增强结果

```bash
# 分析统计信息
python visualize_augmented.py --dataset_path /home/user/MERGE/FSW-MERGE_augmented_double --analyze_only

# 可视化对比（随机10张）
python visualize_augmented.py --dataset_path /home/user/MERGE/FSW-MERGE_augmented_double --num_samples 10

# 保存可视化结果
python visualize_augmented.py \
    --dataset_path /home/user/MERGE/FSW-MERGE_augmented_double \
    --num_samples 20 \
    --save_dir ./visualization_results
```

## 📊 两种模式对比

| 特性 | 安全模式 (transform_safe) | 激进模式 (transform_aggressive) |
|------|---------------------------|--------------------------------|
| Bbox保留率 | ~100% | ~95-98% |
| 增强多样性 | 中等 | 高 |
| 适用场景 | 小目标、少bbox | 大目标、多bbox |
| 裁剪策略 | Resize（无裁剪） | 先放大再裁剪 |
| 推荐用途 | 焊接缺陷检测、医疗影像 | 通用目标检测 |

## 🔧 关键修改点

### 修改1: Bbox过滤参数

**安全模式：**
```python
bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=0,         # 不过滤任何bbox
    min_visibility=0,   # 保留所有可见度的bbox
)
```

**激进模式：**
```python
bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=16,          # 过滤面积<16像素²的bbox
    min_visibility=0.3,   # 过滤可见度<30%的bbox
)
```

### 修改2: 裁剪策略

**安全模式：**
```python
A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0)
# 无裁剪，仅缩放
```

**激进模式：**
```python
A.Resize(height=int(TARGET_SIZE * 1.2), width=int(TARGET_SIZE * 1.2))  # 先放大
A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=0.5)             # 再裁剪
A.Resize(height=TARGET_SIZE, width=TARGET_SIZE)                        # 兜底
```

### 修改3: 重试机制

```python
max_retries = 5  # 从3次增加到5次

# 如果所有bbox丢失，重试
if retained_bbox_count == 0 and retry < max_retries - 1:
    continue  # 重新增强
```

## 🎯 预期效果

### ✅ 安全模式预期结果
- Bbox保留率: **99-100%**
- 完全丢失: **0-1%**
- 适合对bbox完整性要求严格的任务

### ⚡ 激进模式预期结果
- Bbox保留率: **95-98%**
- 完全丢失: **1-3%**
- 部分丢失: **5-10%**
- 增强效果更明显

## 🛠️ 故障排查

### 问题1: 保留率仍然很低 (<90%)

**解决方案:**
1. 确认使用的是 `transform_safe` 模式
2. 检查原始数据标注是否正确（bbox是否越界）
3. 查看具体哪些图像丢失了bbox：
   ```python
   python visualize_augmented.py --dataset_path <path> --num_samples 50
   ```

### 问题2: 增强图像质量不佳

**解决方案:**
调整增强强度：
```python
# 减弱颜色变化
A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3)

# 减弱模糊
A.OneOf([...], p=0.1)  # 从0.2降到0.1

# 减弱噪声
A.GaussNoise(var_limit=(5.0, 20.0), p=0.1)  # 降低强度和概率
```

### 问题3: 部分图像完全没有bbox

**解决方案:**
```bash
# 运行分析找出问题图像
python visualize_augmented.py --dataset_path <path> --analyze_only

# 查看输出中的警告信息
# 如果警告数量 > 5%，建议切换到 transform_safe
```

## 📝 自定义增强

### 只保留特定增强

```python
transform_custom = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    # 其他增强注释掉或删除
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=0,
    min_visibility=0,
))

transform = transform_custom
```

### 添加新的增强

```python
transform_custom = A.Compose([
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    
    # 新增：垂直翻转
    A.VerticalFlip(p=0.3),
    
    # 新增：透视变换
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    
    # ...其他增强
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

## 📚 更多信息

- 详细说明: [BBOX_FIX_README.md](./BBOX_FIX_README.md)
- Albumentations文档: https://albumentations.ai/docs/
- YOLO数据格式: https://docs.ultralytics.com/datasets/

## 🆘 需要帮助？

如果遇到问题：
1. 查看 [BBOX_FIX_README.md](./BBOX_FIX_README.md) 的FAQ部分
2. 运行可视化脚本检查具体问题
3. 检查日志输出中的警告信息

