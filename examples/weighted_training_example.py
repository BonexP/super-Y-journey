#!/usr/bin/env python3
"""
Example script demonstrating how to use YOLOWeightedDataset for training with imbalanced data.

This example shows:
1. How to monkey-patch YOLODataset to use weighted sampling
2. How to train a YOLO model with the weighted dataset
3. How to verify that weighted sampling is working

Usage:
    python examples/weighted_training_example.py --data path/to/data.yaml --model yolo11n.pt
"""

import argparse
from pathlib import Path

import numpy as np

# Import required components
from ultralytics import YOLO
import ultralytics.data.build as build
from ultralytics.data.weighted_dataset import YOLOWeightedDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO with weighted dataset for class imbalance")
    parser.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Path to dataset YAML file (default: coco8.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to model weights or model name (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for training (default: 0)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/weighted_training",
        help="Project directory (default: runs/weighted_training)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name (default: exp)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify weighted sampling, don't train",
    )
    return parser.parse_args()


def verify_weighted_sampling(data_yaml: str, imgsz: int = 640):
    """
    Verify that weighted sampling is working correctly.
    
    Args:
        data_yaml (str): Path to dataset YAML file.
        imgsz (int): Image size.
    """
    print("\n" + "=" * 70)
    print("VERIFYING WEIGHTED SAMPLING")
    print("=" * 70)
    
    # Load dataset configuration
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    
    # Get training images path
    train_path = data_dict.get('train', '')
    if not train_path:
        print("❌ No training data found in dataset YAML")
        return
    
    print(f"\n📁 Dataset: {data_yaml}")
    print(f"📂 Training path: {train_path}")
    
    # Create weighted dataset
    try:
        dataset = YOLOWeightedDataset(
            img_path=train_path,
            data=data_dict,
            imgsz=imgsz,
            augment=False,
            cache=False,
            prefix="train: ",
        )
    except Exception as e:
        print(f"❌ Error creating weighted dataset: {e}")
        return
    
    # Display statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total images: {len(dataset.labels)}")
    print(f"   Number of classes: {len(dataset.counts)}")
    print(f"   Training mode: {dataset.train_mode}")
    
    print(f"\n📈 Class Distribution:")
    class_names = data_dict.get('names', {})
    for i, (count, weight) in enumerate(zip(dataset.counts, dataset.class_weights)):
        class_name = class_names.get(i, f"Class {i}")
        print(f"   {class_name:20s}: {count:5d} instances, weight: {weight:.4f}")
    
    # Check if dataset is balanced
    max_count = np.max(dataset.counts)
    min_count = np.min(dataset.counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n⚖️  Imbalance Analysis:")
    print(f"   Most common class: {max_count} instances")
    print(f"   Least common class: {min_count} instances")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2.0:
        print(f"   ⚠️  Dataset is imbalanced (ratio > 2:1)")
        print(f"   ✅ Weighted sampling will help balance training")
    else:
        print(f"   ℹ️  Dataset is relatively balanced")
        print(f"   ℹ️  Weighted sampling may not be necessary but won't hurt")
    
    # Sample verification
    print(f"\n🔍 Sampling Verification:")
    print(f"   Image weights (first 5): {[f'{w:.3f}' for w in dataset.weights[:5]]}")
    print(f"   Image probabilities (first 5): {[f'{p:.6f}' for p in dataset.probabilities[:5]]}")
    print(f"   Sum of probabilities: {sum(dataset.probabilities):.6f} (should be ~1.0)")
    
    # Simulate sampling
    sample_size = min(1000, len(dataset.labels) * 10)
    print(f"\n🎲 Simulating {sample_size} samples...")
    sampled_classes = []
    for _ in range(sample_size):
        idx = np.random.choice(len(dataset.labels), p=dataset.probabilities)
        label = dataset.labels[idx]
        cls = label["cls"].reshape(-1).astype(int)
        sampled_classes.extend(cls.tolist())
    
    print(f"\n📊 Sampled Class Distribution (from {sample_size} images):")
    unique, counts = np.unique(sampled_classes, return_counts=True)
    for c, count in zip(unique, counts):
        class_name = class_names.get(c, f"Class {c}")
        percentage = (count / len(sampled_classes)) * 100
        print(f"   {class_name:20s}: {count:5d} instances ({percentage:.1f}%)")
    
    print("\n✅ Weighted sampling verification complete!")
    print("=" * 70 + "\n")


def main():
    """Main function to run weighted training example."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("YOLO WEIGHTED TRAINING EXAMPLE")
    print("=" * 70)
    print(f"\n⚙️  Configuration:")
    print(f"   Data: {args.data}")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Device: {args.device}")
    print(f"   Project: {args.project}")
    print(f"   Name: {args.name}")
    
    # Verify weighted sampling first
    verify_weighted_sampling(args.data, args.imgsz)
    
    if args.verify_only:
        print("ℹ️  --verify-only flag set, skipping training")
        return
    
    # Apply monkey-patch to use weighted dataset
    print("\n🔧 Applying monkey-patch to use YOLOWeightedDataset...")
    original_dataset = build.YOLODataset
    build.YOLODataset = YOLOWeightedDataset
    print("✅ YOLODataset replaced with YOLOWeightedDataset")
    
    try:
        # Load model
        print(f"\n🤖 Loading model: {args.model}")
        model = YOLO(args.model)
        
        # Train with weighted dataset
        print(f"\n🚀 Starting training with weighted sampling...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            verbose=True,
        )
        
        print("\n✅ Training complete!")
        print(f"📊 Results: {results}")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
    
    finally:
        # Restore original dataset (optional, for cleanliness)
        build.YOLODataset = original_dataset
        print("\n🔄 Restored original YOLODataset")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
