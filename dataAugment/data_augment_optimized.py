import os
import cv2
import yaml
import shutil
import albumentations as A

# 性能优化：防止OpenCV线程竞争（在多worker DataLoader中至关重要）
cv2.setNumThreads(0)

# 配置路径
base_path = "/home/user/PROJECT/FSWD/FSW-MERGE"
output_double = "/home/user/PROJECT/FSWD/FSW-MERGE_augmented_double"
output_quadruple = "/home/user/PROJECT/FSWD/FSW-MERGE_augmented_quadruple"

# 从data.yaml读取类别信息
yaml_path = os.path.join(base_path, "data.yaml")
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# YOLO训练的目标尺寸（根据你的配置调整）
TARGET_SIZE = 640

# 优化后的增强管道 - 遵循Albumentations最佳实践
# 参考：https://albumentations.ai/docs/3-basic-usage/choosing-augmentations/
transform = A.Compose([
    # Step 1: 裁剪优先！（性能提升16倍）
    # 如果原图可能小于目标尺寸，pad_if_needed=True会自动填充
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0, pad_if_needed=True),

    # Step 2: 基础几何不变性
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),  # 如果你的焊接数据有旋转对称性

    # Step 3: Dropout/遮挡增强 遮挡增强：使用 CoarseDropout 统一替代 CutOut/Random Erasing
    A.OneOf([
        A.CoarseDropout(
            num_holes_range=(3, 6),
            hole_height_range=(10, 32),    # 像素范围
            hole_width_range=(10, 32),
            fill="random_uniform",
            p=1.0
        ),
        # 单孔 + 固定黑色填充，模拟经典 CutOut（大小按比例）
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.08, 0.12),  # 按图像尺寸的比例
            hole_width_range=(0.08, 0.12),
            fill=0,
            p=1.0
        ),
    ], p=0.5),

    # Step 5: 仿射变换（组合旋转和缩放更高效）
    A.Affine(
        scale=(0.8, 1.2),      # 80%-120%缩放
        rotate=(-15, 15),      # ±15度旋转
        p=0.5
    ),

    # Step 6: 领域特定增强 - 颜色/光照变化
    A.RandomBrightnessContrast(
        brightness_limit=0.2,   # 增加到0.2以应对更多光照变化
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,     # 增加色调变化
        sat_shift_limit=20,     # 增加饱和度变化
        val_shift_limit=10,
        p=0.3
    ),

    # Step 6: 领域特定增强 - 模糊效果（工业检测常见）
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),

    # Step 6: 领域特定增强 - 噪声（传感器噪声模拟）
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

    # Step 7: 标准化（如果需要）
    # 注意：YOLO通常有自己的标准化方式，可能不需要这一步
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def normalize_bbox(bbox):
    """
    标准化边界框坐标，确保所有值都在[0.0, 1.0]范围内。
    处理由于浮点精度问题导致的微小越界值。
    """
    x_center, y_center, w, h = bbox
    # 将负的极小值修正为0，大于1的极小越界值修正为1
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # 确保边界框不会超出图像边界
    # x_center - w/2 >= 0 and x_center + w/2 <= 1
    if x_center - w/2 < 0:
        x_center = w/2
    if x_center + w/2 > 1:
        x_center = 1 - w/2

    # y_center - h/2 >= 0 and y_center + h/2 <= 1
    if y_center - h/2 < 0:
        y_center = h/2
    if y_center + h/2 > 1:
        y_center = 1 - h/2

    return [x_center, y_center, w, h]


def augment_dataset(original_train_img_dir, original_train_label_dir, output_img_dir, output_label_dir, multiplier):
    """
    增强数据集：对每个原始图像生成多个增强版本。
    multiplier: 1表示双倍（生成1个新图像），3表示四倍（生成3个新图像）
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(original_train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"找到 {len(image_files)} 张图像待增强")
    total_augmented = 0
    failed_count = 0

    for idx, image_file in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            print(f"处理进度: {idx + 1}/{len(image_files)}")

        image_path = os.path.join(original_train_img_dir, image_file)
        label_path = os.path.join(original_train_label_dir, os.path.splitext(image_file)[0] + '.txt')

        # 使用OpenCV读取图像（最佳性能）
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：无法读取图像 {image_file}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        # 标准化边界框坐标（修复浮点精度问题）
                        normalized_bbox = normalize_bbox([x_center, y_center, w, h])
                        bboxes.append(normalized_bbox)
                        class_labels.append(class_id)

        # 保存原始图像和标注到输出目录（作为基础）
        base_name = os.path.splitext(image_file)[0]
        shutil.copy2(image_path, os.path.join(output_img_dir, image_file))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'))

        # 生成增强版本
        for i in range(multiplier):
            max_retries = 3
            success = False

            for retry in range(max_retries):
                try:
                    # 应用增强
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_class_labels = transformed['class_labels']

                    # 检查是否所有bbox都被保留
                    if len(transformed_bboxes) == 0 and len(bboxes) > 0:
                        # 如果所有bbox都丢失了，跳过这次增强
                        if retry < max_retries - 1:
                            continue
                        else:
                            print(f"警告：{image_file} 增强 {i} 所有bbox丢失")
                            break

                    # 保存增强图像
                    aug_image_name = f"{base_name}_aug_{i}.jpg"
                    aug_image_path = os.path.join(output_img_dir, aug_image_name)
                    cv2.imwrite(aug_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                    # 保存增强标注
                    aug_label_name = f"{base_name}_aug_{i}.txt"
                    aug_label_path = os.path.join(output_label_dir, aug_label_name)
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                    total_augmented += 1
                    success = True
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        continue
                    else:
                        print(f"增强失败 {image_file} 尝试 {i}: {e}")
                        failed_count += 1
                        break

    print(f"\n增强完成！")
    print(f"- 原始图像: {len(image_files)}")
    print(f"- 成功增强: {total_augmented}")
    print(f"- 失败次数: {failed_count}")
    print(f"- 总图像数: {len(image_files) + total_augmented}")


def copy_validation_set(original_val_img_dir, original_val_label_dir, output_val_img_dir, output_val_label_dir):
    """复制验证集到输出目录"""
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_label_dir, exist_ok=True)

    img_count = 0
    for f in os.listdir(original_val_img_dir):
        shutil.copy2(os.path.join(original_val_img_dir, f), output_val_img_dir)
        img_count += 1

    label_count = 0
    for f in os.listdir(original_val_label_dir):
        shutil.copy2(os.path.join(original_val_label_dir, f), output_val_label_dir)
        label_count += 1

    print(f"验证集复制完成: {img_count} 图像, {label_count} 标签")


def update_yaml_file(original_yaml_path, output_yaml_path, output_path):
    """更新data.yaml文件以指向新路径"""
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['path'] = output_path
    data['train'] = 'images/Train'
    data['val'] = 'images/Val'
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"YAML配置已更新: {output_yaml_path}")


# 主执行流程
def main():
    original_train_img_dir = os.path.join(base_path, "images/Train")
    original_train_label_dir = os.path.join(base_path, "labels/Train")
    original_val_img_dir = os.path.join(base_path, "images/Val")
    original_val_label_dir = os.path.join(base_path, "labels/Val")

    # 为双倍变体增强
    print("=" * 60)
    print("正在创建双倍增强数据集...")
    print("=" * 60)
    augment_dataset(original_train_img_dir, original_train_label_dir,
                    os.path.join(output_double, "images/Train"),
                    os.path.join(output_double, "labels/Train"), multiplier=1)
    copy_validation_set(original_val_img_dir, original_val_label_dir,
                        os.path.join(output_double, "images/Val"),
                        os.path.join(output_double, "labels/Val"))
    update_yaml_file(yaml_path, os.path.join(output_double, "data.yaml"), output_double)
    print(f"\n✅ 双倍增强完成！输出目录: {output_double}\n")

    # 为四倍变体增强
    print("=" * 60)
    print("正在创建四倍增强数据集...")
    print("=" * 60)
    augment_dataset(original_train_img_dir, original_train_label_dir,
                    os.path.join(output_quadruple, "images/Train"),
                    os.path.join(output_quadruple, "labels/Train"), multiplier=3)
    copy_validation_set(original_val_img_dir, original_val_label_dir,
                        os.path.join(output_quadruple, "images/Val"),
                        os.path.join(output_quadruple, "labels/Val"))
    update_yaml_file(yaml_path, os.path.join(output_quadruple, "data.yaml"), output_quadruple)
    print(f"\n✅ 四倍增强完成！输出目录: {output_quadruple}\n")


if __name__ == "__main__":
    main()

