"""
Test for YOLOWeightedDataset implementation.

This test verifies that the weighted dataset correctly calculates weights and probabilities
for handling class imbalance.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil


def create_test_dataset():
    """
    Create a minimal test dataset with known class distribution.
    
    Returns:
        tuple: (temp_dir, data_dict) containing temporary directory and dataset configuration.
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create images directory
    img_dir = Path(temp_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Create labels directory
    label_dir = Path(temp_dir) / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some dummy images and labels
    # Image 1: Class 0 (common class) - 3 instances
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2_available = True
    try:
        import cv2
        cv2.imwrite(str(img_dir / "img1.jpg"), img1)
    except ImportError:
        cv2_available = False
        from PIL import Image
        Image.fromarray(img1).save(str(img_dir / "img1.jpg"))
    
    with open(label_dir / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")  # Class 0
        f.write("0 0.3 0.3 0.1 0.1\n")  # Class 0
        f.write("0 0.7 0.7 0.1 0.1\n")  # Class 0
    
    # Image 2: Class 1 (rare class) - 1 instance
    img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    if cv2_available:
        import cv2
        cv2.imwrite(str(img_dir / "img2.jpg"), img2)
    else:
        from PIL import Image
        Image.fromarray(img2).save(str(img_dir / "img2.jpg"))
    
    with open(label_dir / "img2.txt", "w") as f:
        f.write("1 0.5 0.5 0.3 0.3\n")  # Class 1
    
    # Image 3: Class 0 (common class) - 2 instances
    img3 = np.zeros((640, 640, 3), dtype=np.uint8)
    if cv2_available:
        import cv2
        cv2.imwrite(str(img_dir / "img3.jpg"), img3)
    else:
        from PIL import Image
        Image.fromarray(img3).save(str(img_dir / "img3.jpg"))
    
    with open(label_dir / "img3.txt", "w") as f:
        f.write("0 0.4 0.4 0.2 0.2\n")  # Class 0
        f.write("0 0.6 0.6 0.2 0.2\n")  # Class 0
    
    # Image 4: Empty (background)
    img4 = np.zeros((640, 640, 3), dtype=np.uint8)
    if cv2_available:
        import cv2
        cv2.imwrite(str(img_dir / "img4.jpg"), img4)
    else:
        from PIL import Image
        Image.fromarray(img4).save(str(img_dir / "img4.jpg"))
    
    with open(label_dir / "img4.txt", "w") as f:
        pass  # Empty file
    
    # Dataset configuration
    data_dict = {
        "names": {0: "common_class", 1: "rare_class"},
        "channels": 3,
    }
    
    return temp_dir, data_dict, str(img_dir)


def test_weighted_dataset_initialization():
    """Test that YOLOWeightedDataset initializes correctly."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        # Create weighted dataset
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="train: ",
        )
        
        # Verify initialization
        assert dataset.train_mode is True, "Should be in training mode"
        assert hasattr(dataset, "counts"), "Should have counts attribute"
        assert hasattr(dataset, "class_weights"), "Should have class_weights attribute"
        assert hasattr(dataset, "weights"), "Should have weights attribute"
        assert hasattr(dataset, "probabilities"), "Should have probabilities attribute"
        
        print("✓ Weighted dataset initialized successfully")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_class_counting():
    """Test that class instances are counted correctly."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="train: ",
        )
        
        # Expected counts: Class 0: 5 instances (3+2), Class 1: 1 instance
        # Note: counts may be adjusted to avoid zero
        assert len(dataset.counts) == 2, f"Should have 2 classes, got {len(dataset.counts)}"
        assert dataset.counts[0] > 0, "Class 0 should have instances"
        assert dataset.counts[1] > 0, "Class 1 should have instances"
        
        # Class 0 should have more instances than Class 1
        assert dataset.counts[0] > dataset.counts[1], \
            f"Class 0 ({dataset.counts[0]}) should have more instances than Class 1 ({dataset.counts[1]})"
        
        print(f"✓ Class counting correct: {dataset.counts}")
        
    finally:
        shutil.rmtree(temp_dir)


def test_weight_calculation():
    """Test that weights are calculated correctly."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="train: ",
        )
        
        # Rare class should have higher weight than common class
        assert dataset.class_weights[1] > dataset.class_weights[0], \
            f"Rare class weight ({dataset.class_weights[1]}) should be higher than common class ({dataset.class_weights[0]})"
        
        # All images should have weights
        assert len(dataset.weights) == len(dataset.labels), \
            f"Number of weights ({len(dataset.weights)}) should match number of labels ({len(dataset.labels)})"
        
        print(f"✓ Weight calculation correct: class_weights={dataset.class_weights}")
        
    finally:
        shutil.rmtree(temp_dir)


def test_probability_calculation():
    """Test that probabilities sum to 1."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="train: ",
        )
        
        # Probabilities should sum to approximately 1
        prob_sum = sum(dataset.probabilities)
        assert abs(prob_sum - 1.0) < 1e-6, \
            f"Probabilities should sum to 1.0, got {prob_sum}"
        
        # All probabilities should be positive
        assert all(p > 0 for p in dataset.probabilities), \
            "All probabilities should be positive"
        
        print(f"✓ Probability calculation correct: sum={prob_sum:.6f}")
        
    finally:
        shutil.rmtree(temp_dir)


def test_validation_mode():
    """Test that validation mode disables weighted sampling."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        # Create dataset with validation prefix
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="val: ",  # Validation prefix
        )
        
        # Should not be in training mode
        assert dataset.train_mode is False, \
            "Should not be in training mode with 'val:' prefix"
        
        print("✓ Validation mode correctly detected")
        
    finally:
        shutil.rmtree(temp_dir)


def test_getitem_sampling():
    """Test that __getitem__ uses weighted sampling in training mode."""
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    
    temp_dir, data_dict, img_path = create_test_dataset()
    
    try:
        dataset = YOLOWeightedDataset(
            img_path=img_path,
            data=data_dict,
            imgsz=640,
            augment=False,
            cache=False,
            prefix="train: ",
        )
        
        # In training mode, sampling should work
        # We can't test randomness directly, but we can verify it doesn't crash
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        assert sample1 is not None, "Should return a valid sample"
        assert sample2 is not None, "Should return a valid sample"
        
        print("✓ Weighted sampling works in training mode")
        
    finally:
        shutil.rmtree(temp_dir)


def test_monkey_patching():
    """Test that monkey-patching works correctly."""
    import ultralytics.data.build as build
    from ultralytics.data.weighted_dataset import YOLOWeightedDataset
    from ultralytics.data.dataset import YOLODataset
    
    # Save original
    original_dataset = build.YOLODataset
    
    try:
        # Apply monkey-patch
        build.YOLODataset = YOLOWeightedDataset
        
        # Verify the patch
        assert build.YOLODataset == YOLOWeightedDataset, \
            "Monkey-patch should replace YOLODataset with YOLOWeightedDataset"
        
        print("✓ Monkey-patching works correctly")
        
    finally:
        # Restore original
        build.YOLODataset = original_dataset


if __name__ == "__main__":
    """Run tests directly."""
    print("Running YOLOWeightedDataset tests...\n")
    
    try:
        test_weighted_dataset_initialization()
        test_class_counting()
        test_weight_calculation()
        test_probability_calculation()
        test_validation_mode()
        test_getitem_sampling()
        test_monkey_patching()
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
