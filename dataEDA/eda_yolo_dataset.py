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
import warnings
from PIL import Image
# ------------------------------
# Configurable Paths
# ------------------------------
# Use default font settings (removed Chinese font configuration)
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
def analyze_image_resolution_and_bbox_pixels(dataset_root, all_labels):
    """
    Comprehensive image-level analysis:
    1. Image resolution distribution
    2. Pixel-level bbox size distribution
    3. Small object detection
    4. Aspect ratio analysis for images
    5. Input size recommendation
    """
    img_root = os.path.join(dataset_root, "images/Train")
    print("🖼️ Analyzing image resolutions and pixel-level bbox sizes...")

    # Collect image sizes
    img_sizes = []
    image_files = glob.glob(os.path.join(img_root, "*.jpg")) + \
                  glob.glob(os.path.join(img_root, "*.png")) + \
                  glob.glob(os.path.join(img_root, "*.jpeg"))
    
    for img_path in tqdm(image_files, desc="Reading image dimensions"):
        try:
            # Use PIL for faster reading (only reads header, not full image)
            with Image.open(img_path) as img:
                w, h = img.size
            img_sizes.append({
                "file": img_path,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else 1.0,
                "resolution": w * h,
                "max_dim": max(w, h),
                "min_dim": min(w, h)
            })
        except Exception as e:
            print(f"Warning: Could not read {img_path}: {e}")
            continue

    if not img_sizes:
        print("⚠️ No training images found, skipping image analysis.")
        return None

    img_sizes_df = pd.DataFrame(img_sizes)

    # Print image statistics
    print(f"\n📊 Image Resolution Statistics:")
    print(f"   Total images: {len(img_sizes_df)}")
    print(f"   Width  - Mean: {img_sizes_df['width'].mean():.1f}, Median: {img_sizes_df['width'].median():.1f}, Range: [{img_sizes_df['width'].min()}, {img_sizes_df['width'].max()}]")
    print(f"   Height - Mean: {img_sizes_df['height'].mean():.1f}, Median: {img_sizes_df['height'].median():.1f}, Range: [{img_sizes_df['height'].min()}, {img_sizes_df['height'].max()}]")
    print(f"   Aspect Ratio - Mean: {img_sizes_df['aspect_ratio'].mean():.2f}, Range: [{img_sizes_df['aspect_ratio'].min():.2f}, {img_sizes_df['aspect_ratio'].max():.2f}]")

    # Detect extreme aspect ratios
    extreme_ar = img_sizes_df[(img_sizes_df['aspect_ratio'] > 2.0) | (img_sizes_df['aspect_ratio'] < 0.5)]
    if len(extreme_ar) > 0:
        print(f"   ⚠️ Found {len(extreme_ar)} images with extreme aspect ratios (>2:1 or <1:2)")

    # Merge with labels to get pixel-level bbox sizes
    all_labels["basename"] = all_labels["file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    img_sizes_df["basename"] = img_sizes_df["file"].apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    merged = pd.merge(all_labels, img_sizes_df[["basename", "width", "height"]], on="basename", how="left")

    # Calculate pixel-level bbox dimensions
    merged["bbox_w_px"] = merged["w"] * merged["width"]
    merged["bbox_h_px"] = merged["h"] * merged["height"]
    merged["bbox_area_px"] = merged["bbox_w_px"] * merged["bbox_h_px"]
    merged["bbox_min_side_px"] = merged[["bbox_w_px", "bbox_h_px"]].min(axis=1)
    merged["bbox_max_side_px"] = merged[["bbox_w_px", "bbox_h_px"]].max(axis=1)

    # Print bbox pixel statistics
    print(f"\n📦 Bounding Box Pixel Statistics:")
    print(f"   Width (px)  - Mean: {merged['bbox_w_px'].mean():.1f}, Median: {merged['bbox_w_px'].median():.1f}, Range: [{merged['bbox_w_px'].min():.1f}, {merged['bbox_w_px'].max():.1f}]")
    print(f"   Height (px) - Mean: {merged['bbox_h_px'].mean():.1f}, Median: {merged['bbox_h_px'].median():.1f}, Range: [{merged['bbox_h_px'].min():.1f}, {merged['bbox_h_px'].max():.1f}]")
    print(f"   Area (px²)  - Mean: {merged['bbox_area_px'].mean():.1f}, Median: {merged['bbox_area_px'].median():.1f}")

    # Analyze small objects at different thresholds
    thresholds = [10, 20, 30, 50]
    print(f"\n🔍 Small Object Analysis:")
    for thresh in thresholds:
        tiny = merged[(merged["bbox_w_px"] < thresh) | (merged["bbox_h_px"] < thresh)]
        pct = (len(tiny) / len(merged)) * 100
        print(f"   Width or Height < {thresh}px: {len(tiny)} boxes ({pct:.2f}%)")

    # Simulate bbox sizes at different input resolutions
    print(f"\n🎯 Simulated Bbox Sizes at Different Input Resolutions:")
    input_sizes = [320, 640, 800, 1024, 1280]

    # Calculate average scaling factor (assuming letterbox resize preserves aspect ratio)
    avg_img_max_dim = img_sizes_df['max_dim'].mean()

    for input_size in input_sizes:
        scale_factor = input_size / avg_img_max_dim
        simulated_w = merged['bbox_w_px'] * scale_factor
        simulated_h = merged['bbox_h_px'] * scale_factor
        simulated_min = np.minimum(simulated_w, simulated_h)

        # Count very small objects (< 8px on shorter side after resize)
        very_small = (simulated_min < 8).sum()
        very_small_pct = (very_small / len(merged)) * 100

        mean_w = simulated_w.mean()
        mean_h = simulated_h.mean()

        print(f"   Input {input_size}x{input_size}: Avg bbox = {mean_w:.1f}x{mean_h:.1f}px, <8px objects: {very_small} ({very_small_pct:.2f}%)")

    # --- Visualization ---

    # 1. Image resolution scatter plot with color by aspect ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(img_sizes_df["width"], img_sizes_df["height"],
                         c=img_sizes_df["aspect_ratio"], cmap="coolwarm",
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("Image Width (pixels)", fontsize=12)
    ax.set_ylabel("Image Height (pixels)", fontsize=12)
    ax.set_title("Image Resolution Distribution (colored by aspect ratio)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Aspect Ratio (W/H)", fontsize=10)
    save_plot(fig, "image_resolution_scatter.png")

    # 2. Image resolution histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(img_sizes_df["width"], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel("Width (pixels)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Image Width Distribution", fontsize=13, fontweight='bold')
    ax1.axvline(img_sizes_df["width"].mean(), color='red', linestyle='--', label=f'Mean: {img_sizes_df["width"].mean():.0f}')
    ax1.legend()

    ax2.hist(img_sizes_df["height"], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Height (pixels)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Image Height Distribution", fontsize=13, fontweight='bold')
    ax2.axvline(img_sizes_df["height"].mean(), color='red', linestyle='--', label=f'Mean: {img_sizes_df["height"].mean():.0f}')
    ax2.legend()
    plt.tight_layout()
    save_plot(fig, "image_resolution_histograms.png")

    # 3. Bbox pixel width distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(merged["bbox_w_px"], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Bbox Width (pixels)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bounding Box Pixel Width Distribution", fontsize=14, fontweight='bold')
    ax.axvline(merged["bbox_w_px"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {merged["bbox_w_px"].mean():.1f}')
    ax.axvline(merged["bbox_w_px"].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {merged["bbox_w_px"].median():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "bbox_width_pixels.png")

    # 4. Bbox pixel height distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(merged["bbox_h_px"], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Bbox Height (pixels)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bounding Box Pixel Height Distribution", fontsize=14, fontweight='bold')
    ax.axvline(merged["bbox_h_px"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {merged["bbox_h_px"].mean():.1f}')
    ax.axvline(merged["bbox_h_px"].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {merged["bbox_h_px"].median():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "bbox_height_pixels.png")

    # 5. Bbox pixel area distribution (log scale for better visualization)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(merged["bbox_area_px"], bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Bbox Area (square pixels)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Bounding Box Pixel Area Distribution", fontsize=14, fontweight='bold')
    ax.axvline(merged["bbox_area_px"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {merged["bbox_area_px"].mean():.1f}')
    ax.axvline(merged["bbox_area_px"].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {merged["bbox_area_px"].median():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, "bbox_area_pixels.png")

    # 6. Small object analysis visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    small_counts = []
    small_pcts = []
    for thresh in thresholds:
        tiny = merged[(merged["bbox_w_px"] < thresh) | (merged["bbox_h_px"] < thresh)]
        small_counts.append(len(tiny))
        small_pcts.append((len(tiny) / len(merged)) * 100)

    x_pos = np.arange(len(thresholds))
    bars = ax.bar(x_pos, small_pcts, color='orangered', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Threshold (pixels)", fontsize=12)
    ax.set_ylabel("Percentage of Objects (%)", fontsize=12)
    ax.set_title("Small Object Detection: % of Objects Below Threshold", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"<{t}px" for t in thresholds])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, small_counts, small_pcts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    save_plot(fig, "small_object_analysis.png")

    # 7. Input size recommendation chart
    fig, ax = plt.subplots(figsize=(12, 6))

    sim_data = []
    for input_size in input_sizes:
        scale_factor = input_size / avg_img_max_dim
        simulated_min = np.minimum(merged['bbox_w_px'], merged['bbox_h_px']) * scale_factor
        very_small = (simulated_min < 8).sum()
        very_small_pct = (very_small / len(merged)) * 100
        sim_data.append(very_small_pct)

    bars = ax.bar(range(len(input_sizes)), sim_data, color='teal', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Input Size", fontsize=12)
    ax.set_ylabel("% Objects with Min Side < 8px", fontsize=12)
    ax.set_title("Input Size Impact on Small Object Detection", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(input_sizes)))
    ax.set_xticklabels([f"{s}x{s}" for s in input_sizes])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='5% threshold (acceptable)')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='10% threshold (warning)')
    ax.legend()

    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, sim_data)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    save_plot(fig, "input_size_recommendation.png")

    # 8. Bbox size scatter plot (width vs height in pixels)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(merged["bbox_w_px"], merged["bbox_h_px"],
                         alpha=0.4, s=20, c=merged["bbox_area_px"],
                         cmap='viridis', edgecolors='none')
    ax.set_xlabel("Bbox Width (pixels)", fontsize=12)
    ax.set_ylabel("Bbox Height (pixels)", fontsize=12)
    ax.set_title("Bbox Size Distribution (Width vs Height in Pixels)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bbox Area (px²)", fontsize=10)

    # Add reference lines for small object thresholds
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='30px threshold')
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    save_plot(fig, "bbox_size_scatter.png")

    print(f"\n✅ Image-level analysis complete!")

    # Return statistics for HTML report
    return {
        'img_stats': img_sizes_df.describe().to_dict(),
        'bbox_stats': merged[['bbox_w_px', 'bbox_h_px', 'bbox_area_px']].describe().to_dict(),
        'small_object_counts': dict(zip([f"<{t}px" for t in thresholds], small_counts)),
        'small_object_pcts': dict(zip([f"<{t}px" for t in thresholds], [f"{p:.2f}%" for p in small_pcts])),
        'input_size_analysis': dict(zip([f"{s}x{s}" for s in input_sizes], [f"{p:.2f}%" for p in sim_data])),
        'extreme_ar_count': len(extreme_ar),
        'recommended_input_size': input_sizes[np.argmin(sim_data)]
    }
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
    # Create dataframe for proper seaborn plotting
    cls_df = pd.DataFrame({'Class': cls_names, 'Count': cls_counts.values})
    sns.barplot(data=cls_df, x='Class', y='Count', hue='Class', palette="viridis", ax=ax, legend=False)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    save_plot(fig, "class_distribution.png")

    # Bounding box geometry
    all_labels["area"] = all_labels["w"] * all_labels["h"]
    all_labels["aspect_ratio"] = all_labels["w"] / all_labels["h"]

    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(all_labels["area"], bins=50, kde=True, ax=ax)
    ax.set_title("Bounding Box Area Distribution")
    save_plot(fig, "bbox_area_distribution.png")

    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(all_labels["aspect_ratio"], bins=50, kde=True, ax=ax)
    ax.set_title("Aspect Ratio Distribution")
    save_plot(fig, "bbox_aspect_ratio.png")

    # Heatmap of bbox centers
    fig, ax = plt.subplots(figsize=(6,6))
    sns.kdeplot(x=all_labels["x"], y=all_labels["y"], fill=True, cmap="Reds", ax=ax)
    ax.set_title("Bounding Box Center Heatmap")
    save_plot(fig, "bbox_heatmap.png")

    # Per-image box counts
    img_obj_counts = all_labels.groupby("file").size()
    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(img_obj_counts, bins=30, kde=False, ax=ax)
    ax.set_title("Objects per Image Distribution")
    save_plot(fig, "objects_per_image.png")

    # Perform comprehensive image-level analysis
    image_stats = analyze_image_resolution_and_bbox_pixels(dataset_root, all_labels)
    
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
                            title="Interactive: Area Distribution by Class")
        interactive_figs.append(fig1.to_html(full_html=False, include_plotlyjs='cdn'))

        fig2 = px.scatter(all_labels, x="x", y="y", color=all_labels["class"].map(names),
                          title="Interactive: Object Center Distribution")
        interactive_figs.append(fig2.to_html(full_html=False, include_plotlyjs='cdn'))

    # ------------------------------
    # Generate HTML Report
    # ------------------------------
    print("🧾 Generating HTML report...")
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>EDA Report - FSW Defect Detection Dataset</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                line-height: 1.6;
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                padding: 20px; 
                background: white; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h2 { 
                color: #34495e; 
                border-left: 5px solid #3498db; 
                padding-left: 15px;
                margin-top: 30px;
            }
            h3 { 
                color: #555; 
                margin-top: 20px;
            }
            img { 
                max-width: 90%; 
                border-radius: 8px; 
                margin: 15px auto; 
                display: block;
                box-shadow: 0 4px 15px rgba(0,0,0,0.15); 
            }
            .section { 
                margin-bottom: 40px; 
                background: white; 
                padding: 30px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }
            .stats-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .stats-table th {
                background: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }
            .stats-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }
            .stats-table tr:hover {
                background: #f5f5f5;
            }
            .highlight {
                background: #fff3cd;
                padding: 15px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
                border-radius: 5px;
            }
            .recommendation {
                background: #d4edda;
                padding: 15px;
                border-left: 4px solid #28a745;
                margin: 15px 0;
                border-radius: 5px;
            }
            .warning {
                background: #f8d7da;
                padding: 15px;
                border-left: 4px solid #dc3545;
                margin: 15px 0;
                border-radius: 5px;
            }
            ul {
                line-height: 2;
            }
            .metric {
                display: inline-block;
                background: #e8f4f8;
                padding: 5px 10px;
                margin: 3px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
        </style>
    </head>
    <body>
        <h1>🔍 Friction Stir Welding Defect Detection Dataset - EDA Report</h1>
        <p style="text-align: center; font-size: 0.9em; color: #666;">Dataset Path: <code>{{ dataset_root }}</code></p>

        <div class="section">
            <h2>📋 Dataset Overview</h2>
            <ul>
                <li><strong>Total Annotations:</strong> <span class="metric">{{ total_annotations }}</span></li>
                <li><strong>Training Set:</strong> <span class="metric">{{ train_count }}</span> annotations</li>
                <li><strong>Validation Set:</strong> <span class="metric">{{ val_count }}</span> annotations</li>
                <li><strong>Number of Classes:</strong> <span class="metric">{{ num_classes }}</span></li>
            </ul>
        </div>

        <div class="section">
            <h2>1. Class Distribution</h2>
            <img src="EDA_Figures/class_distribution.png" alt="Class Distribution">
        </div>

        <div class="section">
            <h2>2. Bounding Box Geometry Analysis (Normalized Coordinates)</h2>
            <img src="EDA_Figures/bbox_area_distribution.png" alt="Area Distribution">
            <img src="EDA_Figures/bbox_aspect_ratio.png" alt="Aspect Ratio">
        </div>

        <div class="section">
            <h2>3. Object Location Heatmap</h2>
            <img src="EDA_Figures/bbox_heatmap.png" alt="BBox Heatmap">
        </div>

        <div class="section">
            <h2>4. Objects per Image Distribution</h2>
            <img src="EDA_Figures/objects_per_image.png" alt="Objects per Image">
        </div>

        {% if image_stats %}
        <div class="section">
            <h2>5. 📐 Image Resolution Analysis</h2>
            
            <h3>5.1 Image Dimension Statistics</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Width (pixels)</th>
                    <th>Height (pixels)</th>
                </tr>
                <tr>
                    <td><strong>Mean</strong></td>
                    <td>{{ "%.1f"|format(image_stats.img_stats.width.mean) }}</td>
                    <td>{{ "%.1f"|format(image_stats.img_stats.height.mean) }}</td>
                </tr>
                <tr>
                    <td><strong>Median</strong></td>
                    <td>{{ "%.1f"|format(image_stats.img_stats.width['50%']) }}</td>
                    <td>{{ "%.1f"|format(image_stats.img_stats.height['50%']) }}</td>
                </tr>
                <tr>
                    <td><strong>Min</strong></td>
                    <td>{{ "%.0f"|format(image_stats.img_stats.width.min) }}</td>
                    <td>{{ "%.0f"|format(image_stats.img_stats.height.min) }}</td>
                </tr>
                <tr>
                    <td><strong>Max</strong></td>
                    <td>{{ "%.0f"|format(image_stats.img_stats.width.max) }}</td>
                    <td>{{ "%.0f"|format(image_stats.img_stats.height.max) }}</td>
                </tr>
            </table>

            {% if image_stats.extreme_ar_count > 0 %}
            <div class="warning">
                <strong>⚠️ Warning:</strong> Found {{ image_stats.extreme_ar_count }} images with extreme aspect ratios (>2:1 or <1:2). 
                This may cause significant padding/distortion when resized to square input sizes.
            </div>
            {% endif %}

            <img src="EDA_Figures/image_resolution_scatter.png" alt="Image Resolution Scatter">
            <img src="EDA_Figures/image_resolution_histograms.png" alt="Image Resolution Histograms">
        </div>

        <div class="section">
            <h2>6. 📦 Pixel-Level Bounding Box Analysis</h2>
            
            <h3>6.1 Bbox Dimension Statistics (in pixels)</h3>
            <table class="stats-table">
                <tr>
                    <th>Metric</th>
                    <th>Width (px)</th>
                    <th>Height (px)</th>
                    <th>Area (px²)</th>
                </tr>
                <tr>
                    <td><strong>Mean</strong></td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_w_px.mean) }}</td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_h_px.mean) }}</td>
                    <td>{{ "%.0f"|format(image_stats.bbox_stats.bbox_area_px.mean) }}</td>
                </tr>
                <tr>
                    <td><strong>Median</strong></td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_w_px['50%']) }}</td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_h_px['50%']) }}</td>
                    <td>{{ "%.0f"|format(image_stats.bbox_stats.bbox_area_px['50%']) }}</td>
                </tr>
                <tr>
                    <td><strong>Min</strong></td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_w_px.min) }}</td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_h_px.min) }}</td>
                    <td>{{ "%.0f"|format(image_stats.bbox_stats.bbox_area_px.min) }}</td>
                </tr>
                <tr>
                    <td><strong>Max</strong></td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_w_px.max) }}</td>
                    <td>{{ "%.1f"|format(image_stats.bbox_stats.bbox_h_px.max) }}</td>
                    <td>{{ "%.0f"|format(image_stats.bbox_stats.bbox_area_px.max) }}</td>
                </tr>
            </table>

            <img src="EDA_Figures/bbox_width_pixels.png" alt="Bbox Width Pixels">
            <img src="EDA_Figures/bbox_height_pixels.png" alt="Bbox Height Pixels">
            <img src="EDA_Figures/bbox_area_pixels.png" alt="Bbox Area Pixels">
            <img src="EDA_Figures/bbox_size_scatter.png" alt="Bbox Size Scatter">

            <h3>6.2 Small Object Analysis</h3>
            <table class="stats-table">
                <tr>
                    <th>Threshold</th>
                    <th>Object Count</th>
                    <th>Percentage</th>
                </tr>
                {% for threshold, count in image_stats.small_object_counts.items() %}
                <tr>
                    <td><strong>{{ threshold }}</strong></td>
                    <td>{{ count }}</td>
                    <td>{{ image_stats.small_object_pcts[threshold] }}</td>
                </tr>
                {% endfor %}
            </table>
            <img src="EDA_Figures/small_object_analysis.png" alt="Small Object Analysis">
        </div>

        <div class="section">
            <h2>7. 🎯 Input Size Recommendation</h2>
            
            <div class="highlight">
                <strong>📊 Analysis:</strong> The following chart shows the percentage of objects that would have a minimum side 
                &lt; 8 pixels after resizing to different input sizes. Objects smaller than 8px are very difficult to detect.
            </div>

            <img src="EDA_Figures/input_size_recommendation.png" alt="Input Size Recommendation">

            <h3>Detailed Input Size Analysis</h3>
            <table class="stats-table">
                <tr>
                    <th>Input Size</th>
                    <th>% Objects with min side &lt; 8px</th>
                </tr>
                {% for size, pct in image_stats.input_size_analysis.items() %}
                <tr>
                    <td><strong>{{ size }}</strong></td>
                    <td>{{ pct }}</td>
                </tr>
                {% endfor %}
            </table>

            <div class="recommendation">
                <strong>✅ Recommendation:</strong> Based on the analysis, <strong>{{ image_stats.recommended_input_size }}x{{ image_stats.recommended_input_size }}</strong> 
                is recommended as the input size, as it minimizes the number of objects that become too small after resizing.
                <br><br>
                <strong>Guidelines:</strong>
                <ul>
                    <li>If &lt; 5% of objects are too small: <strong>Excellent</strong> - the input size is suitable</li>
                    <li>If 5-10% of objects are too small: <strong>Acceptable</strong> - minor impact on small object detection</li>
                    <li>If &gt; 10% of objects are too small: <strong>Warning</strong> - consider larger input size or tiling strategy</li>
                </ul>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>8. Class-Specific Examples with Annotations</h2>
            {% for name in names.values() %}
                <h3>{{ name }}</h3>
                <img src="EDA_Figures/examples/{{ name }}.jpg" alt="{{ name }}">
            {% endfor %}
        </div>

        {% if interactive_figs %}
        <div class="section">
            <h2>9. Interactive Visualizations</h2>
            {% for fig in interactive_figs %}
                {{ fig | safe }}
            {% endfor %}
        </div>
        {% endif %}

        <div class="section">
            <h2>10. 📝 Summary and Conclusions</h2>
            <ul>
                <li><strong>Total Annotations:</strong> {{ total_annotations }}</li>
                <li><strong>Training Set:</strong> {{ train_count }}, <strong>Validation Set:</strong> {{ val_count }}</li>
                <li><strong>Number of Classes:</strong> {{ num_classes }}</li>
                {% if image_stats %}
                <li><strong>Average Image Size:</strong> {{ "%.0f"|format(image_stats.img_stats.width.mean) }} x {{ "%.0f"|format(image_stats.img_stats.height.mean) }} pixels</li>
                <li><strong>Average Bbox Size:</strong> {{ "%.1f"|format(image_stats.bbox_stats.bbox_w_px.mean) }} x {{ "%.1f"|format(image_stats.bbox_stats.bbox_h_px.mean) }} pixels</li>
                <li><strong>Recommended Input Size:</strong> {{ image_stats.recommended_input_size }}x{{ image_stats.recommended_input_size }}</li>
                {% endif %}
            </ul>

            <div class="highlight">
                <strong>💡 Key Insights:</strong>
                <ul>
                    <li>The dataset contains varied object sizes and aspect ratios</li>
                    <li>Small object detection capability depends heavily on input resolution</li>
                    <li>Choose input size based on the balance between computational cost and detection accuracy</li>
                    <li>Consider data augmentation strategies to improve robustness</li>
                </ul>
            </div>
        </div>

        <footer style="text-align: center; padding: 20px; color: #666; font-size: 0.9em;">
            <p>Generated by EDA Script | Date: {{ generation_date }}</p>
        </footer>
    </body>
    </html>
    """

    from datetime import datetime
    
    template = Template(html_template)
    html = template.render(
        dataset_root=dataset_root,
        names=names,
        total_annotations=len(all_labels),
        train_count=len(train_labels),
        val_count=len(val_labels),
        num_classes=len(names),
        interactive_figs=interactive_figs,
        image_stats=image_stats,
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    with open("EDA_Report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 EDA Complete! Report generated: EDA_Report.html")

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
