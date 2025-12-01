#!/usr/bin/env python3
"""
修复YOLO标注文件中的浮点数类别标签
将0.0、1.0等浮点数类别标签转换为0、1等整数格式
"""
import os
import shutil

def fix_label_file(label_path, backup=True):
    """修复单个标注文件"""
    fixed_lines = []
    has_issue = False

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            fixed_lines.append(line)  # 保持原样
            continue

        class_label = parts[0]

        # 检查是否需要修复
        if '.' in class_label:
            has_issue = True
            # 转换为整数
            class_id = int(float(class_label))
            # 重建行
            fixed_line = f"{class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)

    # 如果发现问题，进行修复
    if has_issue:
        # 备份原文件
        if backup:
            backup_path = label_path + '.backup'
            shutil.copy2(label_path, backup_path)

        # 写入修复后的内容
        with open(label_path, 'w') as f:
            f.writelines(fixed_lines)

        return True

    return False


def fix_dataset(dataset_path, backup=True):
    """修复整个数据集"""
    label_dir = os.path.join(dataset_path, "labels/Train")

    if not os.path.exists(label_dir):
        print(f"❌ 错误: 找不到标签目录 {label_dir}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    print(f"\n{'='*70}")
    print(f"正在修复数据集: {dataset_path}")
    print(f"标签文件数量: {len(label_files)}")
    print(f"备份模式: {'启用' if backup else '禁用'}")
    print(f"{'='*70}\n")

    fixed_count = 0

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        if fix_label_file(label_path, backup):
            fixed_count += 1
            print(f"✅ 已修复: {label_file}")

    print(f"\n{'='*70}")
    print(f"📊 修复结果")
    print(f"{'='*70}")
    print(f"总文件数: {len(label_files)}")
    print(f"修复文件数: {fixed_count}")

    if fixed_count > 0:
        print(f"\n✅ 成功修复 {fixed_count} 个文件！")
        if backup:
            print(f"ℹ️  原文件已备份为 .backup 后缀")
    else:
        print(f"\n✅ 所有文件格式正确，无需修复！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 修复增强后的数据集
    datasets = [
        "/home/user/MERGE/FSW-MERGE_augmented_double",
        "/home/user/MERGE/FSW-MERGE_augmented_quadruple"
    ]

    print("\n⚠️  警告: 此脚本将修改标注文件！")
    print("建议先使用 verify_class_labels.py 检查问题。")

    response = input("\n是否继续修复？(y/n): ")

    if response.lower() == 'y':
        for dataset in datasets:
            if os.path.exists(dataset):
                fix_dataset(dataset, backup=True)
            else:
                print(f"⚠️  跳过: 数据集不存在 {dataset}\n")
    else:
        print("\n已取消操作。")

