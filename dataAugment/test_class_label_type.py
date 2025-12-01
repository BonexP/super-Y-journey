#!/usr/bin/env python3
"""
演示Albumentations类别标签浮点数问题及修复
"""
import albumentations as A
import numpy as np

print("="*70)
print("Albumentations 类别标签类型测试")
print("="*70)

# 创建简单的增强管道
transform = A.Compose([
    A.HorizontalFlip(p=1.0),
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
))

# 模拟输入
image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
bboxes = [[0.5, 0.5, 0.2, 0.2]]
class_labels = [0]  # 整数类别标签

print(f"\n输入:")
print(f"  - class_labels类型: {type(class_labels)}")
print(f"  - class_labels[0]类型: {type(class_labels[0])}")
print(f"  - class_labels值: {class_labels}")

# 应用增强
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_class_labels = transformed['class_labels']

print(f"\n输出:")
print(f"  - transformed_class_labels类型: {type(transformed_class_labels)}")
print(f"  - transformed_class_labels[0]类型: {type(transformed_class_labels[0])}")
print(f"  - transformed_class_labels值: {transformed_class_labels}")

# 模拟错误的保存方式
print(f"\n保存方式对比:")
class_id = transformed_class_labels[0]
print(f"  - 错误方式: f\"{{class_id}} ...\" = \"{class_id} ...\"")
print(f"  - 正确方式: f\"{{int(class_id)}} ...\" = \"{int(class_id)} ...\"")

# 详细分析
print(f"\n详细分析:")
print(f"  - 原始值: {class_id}")
print(f"  - 类型: {type(class_id)}")
print(f"  - 是否为浮点数: {isinstance(class_id, float)}")
print(f"  - 转换为int: {int(class_id)}")
print(f"  - 转换后类型: {type(int(class_id))}")

print(f"\n结论:")
if isinstance(class_id, float):
    print(f"  ⚠️  Albumentations返回的class_labels是浮点数类型！")
    print(f"  ✅ 必须使用 int(class_id) 转换后再保存到YOLO标注文件")
else:
    print(f"  ✅ class_labels是整数类型，但建议仍使用 int() 确保安全")

print("="*70)

