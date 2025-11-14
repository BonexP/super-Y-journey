How to apply CutOut augmentation?
🔗
Albumentations provides the CoarseDropout transform, which is a generalization of the CutOut and Random Erasing augmentation techniques. If you are looking for CutOut or Random Erasing, you should use CoarseDropout.

CoarseDropout generalizes these techniques by allowing:

Variable number of holes: You specify a range for the number of holes (num_holes_range) instead of a fixed number or just one.
Variable hole size: Holes can be rectangular, with ranges for height (hole_height_range) and width (hole_width_range). Sizes can be specified in pixels (int) or as fractions of image dimensions (float).
Flexible fill values: You can fill the holes with a constant value (int/float), per-channel values (tuple), random noise per pixel ('random'), a single random color per hole ('random_uniform'), or using OpenCV inpainting methods ('inpaint_telea', 'inpaint_ns').
Optional mask filling: You can specify a separate fill_mask value to fill corresponding areas in the mask, or leave the mask unchanged (None).
Example using random uniform fill:


import albumentations as A
import numpy as np

image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

# Apply CoarseDropout with 3-6 holes, each 10-20 pixels in size,
# filled with a random uniform color.
transform = A.CoarseDropout(
    num_holes_range=(3, 6),
    hole_height_range=(10, 20),
    hole_width_range=(10, 20),
    fill="random_uniform",
    p=1.0
)

augmented_image = transform(image=image)['image']
This transform randomly removes rectangular regions from an image, similar to CutOut, but with more configuration options.

To specifically mimic the original CutOut behavior, you can configure CoarseDropout as follows:

Set num_holes_range=(1, 1) to always create exactly one hole.
Set hole_height_range and hole_width_range to the same fixed value (e.g., (16, 16) for a 16x16 pixel square or (0.1, 0.1) for a square 10% of the image size).
Set fill=0 to fill the hole with black.
Example mimicking CutOut with a fixed 16x16 hole:


cutout_transform = A.CoarseDropout(
    num_holes_range=(1, 1),
    hole_height_range=(16, 16),
    hole_width_range=(16, 16),
    fill=0,
    p=1.0 # Apply always, or adjust probability as needed
)