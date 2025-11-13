import os
import cv2
import yaml
import shutil
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 配置路径
base_path = "/home/user/PROJECT/FSWD/FSW-MERGE"
output_double = "/home/user/PROJECT/FSWD/FSW-MERGE_augmented_double"
output_quadruple = "/home/user/PROJECT/FSWD/FSW-MERGE_augmented_quadruple"

# 从data.yaml读取类别信息
yaml_path = os.path.join(base_path, "data.yaml")
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
class_names = data_yaml['names']

# 定义增强管道 - 使用适中参数以避免极端变化
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_dataset(original_train_img_dir, original_train_label_dir, output_img_dir, output_label_dir, multiplier):
    """
    增强数据集：对每个原始图像生成多个增强版本。
    multiplier: 1表示双倍（生成1个新图像），3表示四倍（生成3个新图像）
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(original_train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(original_train_img_dir, image_file)
        label_path = os.path.join(original_train_label_dir, os.path.splitext(image_file)[0] + '.txt')

        # 读取原始图像和标注
        image = cv2.imread(image_path)
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
                        bboxes.append([x_center, y_center, w, h])
                        class_labels.append(class_id)

        # 保存原始图像和标注到输出目录（作为基础）
        base_name = os.path.splitext(image_file)[0]
        shutil.copy2(image_path, os.path.join(output_img_dir, image_file))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(output_label_dir, os.path.splitext(image_file)[0] + '.txt'))

        # 生成增强版本
        for i in range(multiplier):
            try:
                # 应用增强
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

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
            except Exception as e:
                print(f"增强失败 {image_file} 尝试 {i}: {e}")
                continue


def copy_validation_set(original_val_img_dir, original_val_label_dir, output_val_img_dir, output_val_label_dir):
    """复制验证集到输出目录"""
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_label_dir, exist_ok=True)
    for f in os.listdir(original_val_img_dir):
        shutil.copy2(os.path.join(original_val_img_dir, f), output_val_img_dir)
    for f in os.listdir(original_val_label_dir):
        shutil.copy2(os.path.join(original_val_label_dir, f), output_val_label_dir)


def update_yaml_file(original_yaml_path, output_yaml_path, output_path):
    """更新data.yaml文件以指向新路径"""
    with open(original_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    data['path'] = output_path
    data['train'] = 'images/Train'
    data['val'] = 'images/Val'
    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


# 主执行流程
def main():
    original_train_img_dir = os.path.join(base_path, "images/Train")
    original_train_label_dir = os.path.join(base_path, "labels/Train")
    original_val_img_dir = os.path.join(base_path, "images/Val")
    original_val_label_dir = os.path.join(base_path, "labels/Val")

    # 为双倍变体增强
    print("正在创建双倍增强数据集...")
    augment_dataset(original_train_img_dir, original_train_label_dir,
                    os.path.join(output_double, "images/Train"),
                    os.path.join(output_double, "labels/Train"), multiplier=1)
    copy_validation_set(original_val_img_dir, original_val_label_dir,
                        os.path.join(output_double, "images/Val"),
                        os.path.join(output_double, "labels/Val"))
    update_yaml_file(yaml_path, os.path.join(output_double, "data.yaml"), output_double)
    print(f"双倍增强完成！输出目录: {output_double}")

    # 为四倍变体增强
    print("正在创建四倍增强数据集...")
    augment_dataset(original_train_img_dir, original_train_label_dir,
                    os.path.join(output_quadruple, "images/Train"),
                    os.path.join(output_quadruple, "labels/Train"), multiplier=3)
    copy_validation_set(original_val_img_dir, original_val_label_dir,
                        os.path.join(output_quadruple, "images/Val"),
                        os.path.join(output_quadruple, "labels/Val"))
    update_yaml_file(yaml_path, os.path.join(output_quadruple, "data.yaml"), output_quadruple)
    print(f"四倍增强完成！输出目录: {output_quadruple}")


if __name__ == "__main__":
    main()
