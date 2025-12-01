"""
可视化增强后的数据集，检查边界框是否正确保留

使用方法：
    python visualize_augmented.py --dataset_path /home/user/MERGE/FSW-MERGE_augmented_double --num_samples 10
"""

import os
import cv2
import argparse
import random


def draw_yolo_bbox(image, bbox_line, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制YOLO格式的边界框

    Args:
        image: 图像数组
        bbox_line: YOLO格式标注行 "class_id x_center y_center width height"
        color: 边界框颜色 (B, G, R)
        thickness: 线条粗细
    """
    h, w = image.shape[:2]
    parts = bbox_line.strip().split()

    if len(parts) != 5:
        return

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width_norm = float(parts[3])
    height_norm = float(parts[4])

    # 转换为像素坐标
    x1 = int((x_center - width_norm / 2) * w)
    y1 = int((y_center - height_norm / 2) * h)
    x2 = int((x_center + width_norm / 2) * w)
    y2 = int((y_center + height_norm / 2) * h)

    # 绘制矩形
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # 添加类别标签
    label = f"Class {class_id}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def visualize_sample(image_path, label_path, output_path=None):
    """
    可视化单张图像及其标注

    Args:
        image_path: 图像路径
        label_path: 标签路径
        output_path: 保存路径（可选）

    Returns:
        带标注的图像数组
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return None

    # 统计bbox数量
    bbox_count = 0

    # 读取并绘制标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            bbox_count = len(lines)

            # 为不同类别使用不同颜色
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 品红
                (0, 255, 255),  # 黄色
            ]

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    color = colors[class_id % len(colors)]
                    draw_yolo_bbox(image, line, color=color)

    # 添加图像信息
    h, w = image.shape[:2]
    info_text = f"Size: {w}x{h} | Bboxes: {bbox_count}"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    # 添加文件名
    filename = os.path.basename(image_path)
    cv2.putText(image, filename, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(image, filename, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # 保存图像
    if output_path:
        cv2.imwrite(str(output_path), image)
        print(f"✅ 保存到: {output_path}")

    return image


def compare_original_and_augmented(dataset_path, num_samples=10, save_dir=None):
    """
    比较原始图像和增强图像

    Args:
        dataset_path: 数据集路径
        num_samples: 采样数量
        save_dir: 保存目录（可选）
    """
    img_dir = os.path.join(dataset_path, "images/Train")
    label_dir = os.path.join(dataset_path, "labels/Train")

    if not os.path.exists(img_dir):
        print(f"❌ 图像目录不存在: {img_dir}")
        return

    # 获取所有图像文件
    all_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 分离原始图像和增强图像
    original_images = [f for f in all_images if '_aug_' not in f]
    augmented_images = [f for f in all_images if '_aug_' in f]

    print(f"\n{'='*70}")
    print(f"📂 数据集路径: {dataset_path}")
    print(f"📊 统计信息:")
    print(f"  - 原始图像: {len(original_images)}")
    print(f"  - 增强图像: {len(augmented_images)}")
    print(f"  - 总计: {len(all_images)}")
    print(f"{'='*70}\n")

    # 随机采样
    sample_originals = random.sample(original_images, min(num_samples, len(original_images)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"💾 可视化结果将保存到: {save_dir}\n")

    # 处理每个样本
    for idx, orig_img in enumerate(sample_originals, 1):
        base_name = os.path.splitext(orig_img)[0]

        print(f"[{idx}/{len(sample_originals)}] 处理: {orig_img}")

        # 原始图像
        orig_img_path = os.path.join(img_dir, orig_img)
        orig_label_path = os.path.join(label_dir, base_name + '.txt')

        orig_output = os.path.join(save_dir, f"compare_{idx}_original.jpg") if save_dir else None
        orig_vis = visualize_sample(orig_img_path, orig_label_path, orig_output)

        # 查找对应的增强图像
        aug_imgs = [f for f in augmented_images if f.startswith(base_name + '_aug_')]

        if aug_imgs:
            # 随机选择一个增强版本
            aug_img = random.choice(aug_imgs)
            aug_img_path = os.path.join(img_dir, aug_img)
            aug_label_path = os.path.join(label_dir, os.path.splitext(aug_img)[0] + '.txt')

            aug_output = os.path.join(save_dir, f"compare_{idx}_augmented.jpg") if save_dir else None
            aug_vis = visualize_sample(aug_img_path, aug_label_path, aug_output)

            # 统计bbox数量
            orig_bbox_count = len(open(orig_label_path).readlines()) if os.path.exists(orig_label_path) else 0
            aug_bbox_count = len(open(aug_label_path).readlines()) if os.path.exists(aug_label_path) else 0

            if orig_bbox_count != aug_bbox_count:
                print(f"  ⚠️ Bbox数量变化: {orig_bbox_count} → {aug_bbox_count}")
            else:
                print(f"  ✅ Bbox数量保持: {orig_bbox_count}")

            # 如果不保存，则显示
            if not save_dir and orig_vis is not None and aug_vis is not None:
                # 并排显示
                combined = cv2.hconcat([orig_vis, aug_vis])
                cv2.imshow('Original (left) vs Augmented (right)', combined)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

        print()

    if not save_dir:
        cv2.destroyAllWindows()

    print(f"✅ 完成！共处理 {len(sample_originals)} 个样本")


def analyze_dataset_statistics(dataset_path):
    """
    分析数据集的bbox统计信息

    Args:
        dataset_path: 数据集路径
    """
    label_dir = os.path.join(dataset_path, "labels/Train")

    if not os.path.exists(label_dir):
        print(f"❌ 标签目录不存在: {label_dir}")
        return

    # 统计信息
    total_images = 0
    total_bboxes = 0
    images_without_bbox = 0
    bbox_counts = []

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            bbox_count = len(lines)

        total_images += 1
        total_bboxes += bbox_count
        bbox_counts.append(bbox_count)

        if bbox_count == 0:
            images_without_bbox += 1

    # 输出统计
    print(f"\n{'='*70}")
    print(f"📊 数据集统计分析")
    print(f"{'='*70}")
    print(f"📂 路径: {label_dir}")
    print(f"\n图像统计:")
    print(f"  - 总图像数: {total_images}")
    print(f"  - 无bbox图像: {images_without_bbox} ({images_without_bbox/total_images*100:.2f}%)")
    print(f"\nBbox统计:")
    print(f"  - 总bbox数: {total_bboxes}")
    print(f"  - 平均每张: {total_bboxes/total_images:.2f}")
    print(f"  - 最小数量: {min(bbox_counts) if bbox_counts else 0}")
    print(f"  - 最大数量: {max(bbox_counts) if bbox_counts else 0}")
    print(f"{'='*70}\n")

    if images_without_bbox > 0:
        print(f"⚠️ 警告: 发现 {images_without_bbox} 张图像没有任何bbox！")
        print(f"   这可能是增强时bbox完全丢失导致的。")
        print(f"   建议切换到 transform_safe 模式。\n")


def main():
    parser = argparse.ArgumentParser(description="可视化增强后的YOLO数据集")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集路径，例如: /home/user/MERGE/FSW-MERGE_augmented_double")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="随机采样数量 (默认: 10)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="保存可视化结果的目录 (可选)")
    parser.add_argument("--analyze_only", action="store_true",
                        help="仅分析统计信息，不进行可视化")

    args = parser.parse_args()

    # 分析数据集统计
    analyze_dataset_statistics(args.dataset_path)

    # 可视化样本
    if not args.analyze_only:
        compare_original_and_augmented(args.dataset_path, args.num_samples, args.save_dir)


if __name__ == "__main__":
    main()

