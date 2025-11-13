#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA Script for YOLO Dataset (FSW-MERGE)
Author: ChatGPT (GPT-5)
Date: 2025-11-13
Description:
  Automatically performs exploratory data analysis on a YOLO-format dataset,
  saves static figures to 'EDA_Figures/', optionally generates interactive plots,
  and builds a full HTML report for academic use.
"""

import os
import yaml
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from jinja2 import Template
import plotly.express as px

# ------------------------------
# Configurable Paths
# ------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font_scale=1.2)

# ------------------------------
# Helper functions
# ------------------------------
def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_labels(label_dir):
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    data = []
    for f in label_files:
        with open(f, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    data.append([f, int(cls), x, y, w, h])
    return pd.DataFrame(data, columns=["file", "class", "x", "y", "w", "h"])

def check_dataset_structure(root):
    subdirs = ["images/Train", "images/Val", "labels/Train", "labels/Val"]
    for sub in subdirs:
        if not os.path.exists(os.path.join(root, sub)):
            raise FileNotFoundError(f"Missing directory: {os.path.join(root, sub)}")

def draw_bbox_on_image(img_path, label_path, names, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    with open(label_path, "r") as f:
        for line in f:
            c, x, y, bw, bh = map(float, line.strip().split())
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, names[int(c)], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(save_path, img)
    return True

def save_plot(fig, filename):
    os.makedirs("EDA_Figures", exist_ok=True)
    fig.savefig(os.path.join("EDA_Figures", filename), bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# Main EDA logic
# ------------------------------
def perform_eda(dataset_root, interactive=False):
    yaml_path = os.path.join(dataset_root, "data.yaml")
    cfg = load_yaml(yaml_path)
    names = cfg["names"]

    check_dataset_structure(dataset_root)

    print("🔍 Loading YOLO labels...")
    train_labels = load_labels(os.path.join(dataset_root, "labels/Train"))
    val_labels = load_labels(os.path.join(dataset_root, "labels/Val"))
    all_labels = pd.concat([train_labels.assign(split='Train'),
                            val_labels.assign(split='Val')], ignore_index=True)

    print(f"✅ Loaded {len(all_labels)} total annotations.")

    # Category statistics
    cls_counts = all_labels['class'].value_counts().sort_index()
    cls_names = [names[int(i)] for i in cls_counts.index]
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=cls_names, y=cls_counts.values, palette="viridis", ax=ax)
    ax.set_title("类别目标数量分布")
    ax.set_xlabel("类别")
    ax.set_ylabel("数量")
    save_plot(fig, "class_distribution.png")

    # Bounding box geometry
    all_labels["area"] = all_labels["w"] * all_labels["h"]
    all_labels["aspect_ratio"] = all_labels["w"] / all_labels["h"]

    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(all_labels["area"], bins=50, kde=True, ax=ax)
    ax.set_title("目标框面积分布")
    save_plot(fig, "bbox_area_distribution.png")

    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(all_labels["aspect_ratio"], bins=50, kde=True, ax=ax)
    ax.set_title("宽高比分布")
    save_plot(fig, "bbox_aspect_ratio.png")

    # Heatmap of bbox centers
    fig, ax = plt.subplots(figsize=(6,6))
    sns.kdeplot(x=all_labels["x"], y=all_labels["y"], fill=True, cmap="Reds", ax=ax)
    ax.set_title("目标中心点热力图")
    save_plot(fig, "bbox_heatmap.png")

    # Per-image box counts
    img_obj_counts = all_labels.groupby("file").size()
    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(img_obj_counts, bins=30, kde=False, ax=ax)
    ax.set_title("每张图片的目标数量分布")
    save_plot(fig, "objects_per_image.png")

    # Example images per class
    print("🖼️ Generating example images with bounding boxes...")
    example_dir = os.path.join("EDA_Figures", "examples")
    os.makedirs(example_dir, exist_ok=True)
    img_root = os.path.join(dataset_root, "images/Train")
    label_root = os.path.join(dataset_root, "labels/Train")

    for cls_id, cls_name in names.items():
        found = False
        for label_file in glob.glob(os.path.join(label_root, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
            if any(line.startswith(str(cls_id)) for line in lines):
                img_file = os.path.join(img_root, os.path.basename(label_file).replace(".txt", ".jpg"))
                save_path = os.path.join(example_dir, f"{cls_name}.jpg")
                if os.path.exists(img_file):
                    draw_bbox_on_image(img_file, label_file, names, save_path)
                found = True
                break
        if not found:
            print(f"⚠️ No example found for class {cls_name}")

    # Interactive plots
    interactive_figs = []
    if interactive:
        print("⚡ Generating interactive visualizations...")
        fig1 = px.histogram(all_labels, x="area", color=all_labels["class"].map(names), nbins=40,
                            title="交互式：不同类别的面积分布")
        interactive_figs.append(fig1.to_html(full_html=False, include_plotlyjs='cdn'))

        fig2 = px.scatter(all_labels, x="x", y="y", color=all_labels["class"].map(names),
                          title="交互式：目标中心点分布")
        interactive_figs.append(fig2.to_html(full_html=False, include_plotlyjs='cdn'))

    # ------------------------------
    # Generate HTML Report
    # ------------------------------
    print("🧾 Generating HTML report...")
    html_template = """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>EDA 报告 - FSWD 数据集</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
            h1, h2 { color: #333; }
            img { max-width: 90%; border-radius: 8px; margin: 10px 0; box-shadow: 0 0 5px rgba(0,0,0,0.2); }
            .section { margin-bottom: 40px; background: white; padding: 20px; border-radius: 10px; }
        </style>
    </head>
    <body>
        <h1>搅拌摩擦焊表面缺陷检测数据集 EDA 报告</h1>
        <p>数据路径：{{ dataset_root }}</p>

        <div class="section">
            <h2>1. 类别分布</h2>
            <img src="EDA_Figures/class_distribution.png" alt="Class Distribution">
        </div>

        <div class="section">
            <h2>2. 目标框几何特征分析</h2>
            <img src="EDA_Figures/bbox_area_distribution.png" alt="Area Distribution">
            <img src="EDA_Figures/bbox_aspect_ratio.png" alt="Aspect Ratio">
        </div>

        <div class="section">
            <h2>3. 目标位置热力图</h2>
            <img src="EDA_Figures/bbox_heatmap.png" alt="BBox Heatmap">
        </div>

        <div class="section">
            <h2>4. 每图目标数量分布</h2>
            <img src="EDA_Figures/objects_per_image.png" alt="Objects per Image">
        </div>

        <div class="section">
            <h2>5. 各类别示例图片</h2>
            {% for name in names.values() %}
                <h3>{{ name }}</h3>
                <img src="EDA_Figures/examples/{{ name }}.jpg" alt="{{ name }}">
            {% endfor %}
        </div>

        {% if interactive_figs %}
        <div class="section">
            <h2>6. 交互式可视化</h2>
            {% for fig in interactive_figs %}
                {{ fig | safe }}
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>7. 总结</h2>
            <ul>
                <li>总标注数：{{ total_annotations }}</li>
                <li>训练集：{{ train_count }}，验证集：{{ val_count }}</li>
                <li>类别数量：{{ num_classes }}</li>
            </ul>
        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    html = template.render(
        dataset_root=dataset_root,
        names=names,
        total_annotations=len(all_labels),
        train_count=len(train_labels),
        val_count=len(val_labels),
        num_classes=len(names),
        interactive_figs=interactive_figs
    )

    with open("EDA_Report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 EDA 完成！报告已生成：EDA_Report.html")

# ------------------------------
# CLI Entry
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on YOLO-format dataset.")
    parser.add_argument("--dataset", type=str, default="/home/user/PROJECT/FSWD/FSW-MERGE",
                        help="Dataset root directory containing images/, labels/, and data.yaml")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive visualizations")
    args = parser.parse_args()

    interactive_flag = args.interactive or bool(os.getenv("INTERACTIVE", False))
    perform_eda(args.dataset, interactive=interactive_flag)
