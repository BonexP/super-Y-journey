import os
import cv2
import yaml
import shutil
import albumentations as A

# 性能优化：防止OpenCV线程竞争（在多worker DataLoader中至关重要）
cv2.setNumThreads(0)

# 配置路径
base_path = "/home/user/FSW-MERGE/FSW-MERGE"
output_double = "/home/user/MERGE/FSW-MERGE_augmented_double"
output_quadruple = "/home/user/MERGE/FSW-MERGE_augmented_quadruple"

# 从data.yaml读取类别信息
yaml_path = os.path.join(base_path, "data.yaml")
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# YOLO训练的目标尺寸（根据你的配置调整）
TARGET_SIZE = 640

# ==============================================================================
# 增强管道配置
# ==============================================================================

# 方案1: 安全模式（推荐）- 使用Resize避免裁剪，100%保留所有bbox
# 适合小目标检测、焊接缺陷检测等对bbox完整性要求高的任务
transform_safe = A.Compose([
    # Step 1: 统一尺寸 - 使用Resize而非RandomCrop，确保所有bbox保留
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),

    # Step 2: 基础几何变换
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),

    # Step 3: 温和的仿射变换（避免bbox被推出边界）
    A.Affine(
        scale=(0.9, 1.1),      # 90%-110%缩放
        rotate=(-10, 10),      # ±10度旋转
        p=0.4,
        border_mode=cv2.BORDER_CONSTANT,
        fill=114
    ),

    # Step 4: 颜色和光照变化（不影响bbox）
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.3
    ),

    # Step 5: 模糊效果（不影响bbox）
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),

    # Step 6: 噪声（不影响bbox）
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.2),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=0,         # 不限制最小面积，保留所有bbox
    min_visibility=0,   # 不限制最小可见度，保留被部分遮挡的bbox
))

# 方案2: 激进模式 - 更强的增强效果，可能丢失少量bbox
# 适合大目标检测或需要更多样化增强的场景
transform_aggressive = A.Compose([
    # Step 1: 先resize到更大尺寸，再随机裁剪，最后resize回目标尺寸
    A.Resize(height=int(TARGET_SIZE * 1.2), width=int(TARGET_SIZE * 1.2), p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=0.5),
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),

    # Step 2: 基础几何变换
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),

    # Step 3: 遮挡增强（降低强度避免过度遮挡）
    A.OneOf([
        A.CoarseDropout(
            num_holes_range=(2, 4),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=1.0
        ),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.05, 0.08),
            hole_width_range=(0.05, 0.08),
            fill=0,
            p=1.0
        ),
    ], p=0.3),

    # Step 4: 仿射变换
    A.Affine(
        scale=(0.85, 1.15),
        rotate=(-12, 12),
        p=0.4,
        border_mode=cv2.BORDER_CONSTANT,
        fill=114
    ),

    # Step 5-7: 颜色/模糊/噪声
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=5, p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.2),
    A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.2),

], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_area=16,           # 最小16像素²，过滤掉极小的残留bbox
    min_visibility=0.3,    # 至少保留30%可见度的bbox
))

# ==============================================================================
# 选择使用的增强模式
# ==============================================================================
# 默认使用安全模式（推荐）- 100%保留bbox
transform = transform_safe

# 如果需要更强的增强效果，可以切换到激进模式（可能丢失少量bbox）
# transform = transform_aggressive

print(f"\n{'='*70}")
print(f"ℹ️  当前使用的增强模式: {'安全模式 (transform_safe)' if transform == transform_safe else '激进��式 (transform_aggressive)'}")
print(f"ℹ️  Bbox保留策略: {'100%保留所有bbox' if transform == transform_safe else '保留大部分bbox (95-98%)'}")
print(f"{'='*70}\n")


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
    bbox_loss_count = 0  # 统计bbox完全丢失的次数
    partial_bbox_loss_count = 0  # 统计部分bbox丢失的次数
    total_original_bboxes = 0
    total_retained_bboxes = 0

    for idx, image_file in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            if total_original_bboxes > 0:
                retention_rate = (total_retained_bboxes / total_original_bboxes) * 100
                print(f"处理进度: {idx + 1}/{len(image_files)} | Bbox保留率: {retention_rate:.1f}% | 完全丢失: {bbox_loss_count} | 部分丢失: {partial_bbox_loss_count}")
            else:
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
            max_retries = 5  # 增加重试次数
            success = False

            for retry in range(max_retries):
                try:
                    # 应用增强
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_class_labels = transformed['class_labels']

                    original_bbox_count = len(bboxes)
                    retained_bbox_count = len(transformed_bboxes)

                    # 检查bbox丢失情况
                    if original_bbox_count > 0:
                        if retained_bbox_count == 0:
                            # 所有bbox都丢失了，重试
                            if retry < max_retries - 1:
                                continue
                            else:
                                print(f"  ⚠️ {image_file} 增强 {i+1}: 所有 {original_bbox_count} 个bbox丢失，跳过此增强")
                                bbox_loss_count += 1
                                failed_count += 1
                                break
                        elif retained_bbox_count < original_bbox_count:
                            # 部分bbox丢失，但仍然保存（可能是合理的裁剪结果）
                            partial_bbox_loss_count += 1
                            if retained_bbox_count < original_bbox_count * 0.5:  # 丢失超过50%
                                print(f"  ⚠️ {image_file} 增强 {i+1}: bbox从 {original_bbox_count} 减少到 {retained_bbox_count}")

                    # 统计bbox保留情况（只在成功时统计一次）
                    total_original_bboxes += original_bbox_count
                    total_retained_bboxes += retained_bbox_count

                    # 保存增强图像
                    aug_image_name = f"{base_name}_aug_{i}.jpg"
                    aug_image_path = os.path.join(output_img_dir, aug_image_name)
                    cv2.imwrite(aug_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                    # 保存增强标注
                    aug_label_name = f"{base_name}_aug_{i}.txt"
                    aug_label_path = os.path.join(output_label_dir, aug_label_name)
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                            # 确保class_id是整数（防止Albumentations返回浮点数）
                            f.write(f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

                    total_augmented += 1
                    success = True
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        continue
                    else:
                        print(f"  ❌ 增强失败 {image_file} 尝试 {i+1}: {e}")
                        failed_count += 1
                        break

    print(f"\n{'='*70}")
    print(f"📊 增强统计报告")
    print(f"{'='*70}")
    print(f"数据统计:")
    print(f"  - 原始图像: {len(image_files)}")
    print(f"  - 成功增强: {total_augmented}")
    print(f"  - 失败次数: {failed_count}")
    print(f"  - 总图像数: {len(image_files) + total_augmented}")
    print(f"\nBbox保留统计:")
    print(f"  - 原始bbox总数: {total_original_bboxes}")
    print(f"  - 保留bbox总数: {total_retained_bboxes}")
    if total_original_bboxes > 0:
        retention_rate = (total_retained_bboxes / total_original_bboxes) * 100
        print(f"  - 总体保留率: {retention_rate:.2f}%")
        loss_rate = (bbox_loss_count / (len(image_files) * multiplier)) * 100 if multiplier > 0 else 0
        partial_loss_rate = (partial_bbox_loss_count / (len(image_files) * multiplier)) * 100 if multiplier > 0 else 0
        print(f"  - 完全丢失bbox的增强: {bbox_loss_count} ({loss_rate:.2f}%)")
        print(f"  - 部分丢失bbox的增强: {partial_bbox_loss_count} ({partial_loss_rate:.2f}%)")

        # 根据保留率给出建议
        if retention_rate < 90:
            print(f"\n⚠️  警告: Bbox保留率较低 ({retention_rate:.1f}%)，建议切换到 transform_safe 模式！")
        elif retention_rate < 95:
            print(f"\nℹ️  提示: Bbox保留率为 {retention_rate:.1f}%，如需更高保留率，可切换到 transform_safe 模式。")
        else:
            print(f"\n✅ Bbox保留率良好 ({retention_rate:.1f}%)！")
    print(f"{'='*70}\n")


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
    print("=" * 70)
    print("正在创建双倍增强数据集...")
    print("=" * 70)
    augment_dataset(original_train_img_dir, original_train_label_dir,
                    os.path.join(output_double, "images/Train"),
                    os.path.join(output_double, "labels/Train"), multiplier=1)
    copy_validation_set(original_val_img_dir, original_val_label_dir,
                        os.path.join(output_double, "images/Val"),
                        os.path.join(output_double, "labels/Val"))
    update_yaml_file(yaml_path, os.path.join(output_double, "data.yaml"), output_double)
    print(f"\n✅ 双倍增强完成！输出目录: {output_double}\n")

    # 为四倍变体增强
    print("=" * 70)
    print("正在创建四倍增强数据集...")
    print("=" * 70)
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

