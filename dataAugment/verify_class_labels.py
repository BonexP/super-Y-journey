#!/usr/bin/env python3
"""
验证YOLO标注文件中的类别标签是否为整数
检查是否存在0.0这样的浮点数类别标签
"""
import os

def verify_label_file(label_path):
    """验证单个标注文件"""
    issues = []

    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.append(f"  行 {line_num}: 格式错误，应该有5个字段，实际有 {len(parts)} 个")
                continue

            class_label = parts[0]

            # 检查是否包含小数点（浮点数类别标签）
            if '.' in class_label:
                issues.append(f"  行 {line_num}: 类别标签是浮点数 '{class_label}'，应该是整数")

            # 尝试转换为整数
            try:
                class_id = int(float(class_label))  # 先转float再转int，兼容0.0的情况
                if str(class_id) != class_label:
                    issues.append(f"  行 {line_num}: 类别标签 '{class_label}' 不是标准整数格式")
            except ValueError:
                issues.append(f"  行 {line_num}: 类别标签 '{class_label}' 无法转换为整数")

    return issues


def verify_dataset(dataset_path):
    """验证整个数据集"""
    label_dir = os.path.join(dataset_path, "labels/Train")

    if not os.path.exists(label_dir):
        print(f"❌ 错误: 找不到标签目录 {label_dir}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    print(f"\n{'='*70}")
    print(f"正在验证数据集: {dataset_path}")
    print(f"标签文件数量: {len(label_files)}")
    print(f"{'='*70}\n")

    total_issues = 0
    files_with_issues = 0

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        issues = verify_label_file(label_path)

        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"⚠️  {label_file}:")
            for issue in issues:
                print(issue)
            print()

    print(f"\n{'='*70}")
    print(f"📊 验证结果")
    print(f"{'='*70}")
    print(f"总文件数: {len(label_files)}")
    print(f"有问题的文件数: {files_with_issues}")
    print(f"问题总数: {total_issues}")

    if total_issues == 0:
        print(f"\n✅ 所有标注文件的类别标签格式正确！")
    else:
        print(f"\n❌ 发现 {total_issues} 个问题需要修复！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 验证增强后的数据集
    datasets = [
        "/home/user/MERGE/FSW-MERGE_augmented_double",
        "/home/user/MERGE/FSW-MERGE_augmented_quadruple"
    ]

    for dataset in datasets:
        if os.path.exists(dataset):
            verify_dataset(dataset)
        else:
            print(f"⚠️  跳过: 数据集不存在 {dataset}\n")

