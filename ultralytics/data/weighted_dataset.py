# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Weighted Dataset for YOLO to handle class imbalance.

This module provides a custom dataset class that extends YOLODataset with weighted sampling
capabilities to address data imbalance issues during training.
"""

import numpy as np

from ultralytics.data.dataset import YOLODataset


class YOLOWeightedDataset(YOLODataset):
    """
    YOLO dataset with weighted sampling to handle class imbalance.

    This dataset class extends YOLODataset to provide weighted sampling based on class frequencies.
    During training, images are sampled with probabilities inversely proportional to their class
    frequencies, helping to balance the distribution of classes seen during training.

    Attributes:
        train_mode (bool): Whether the dataset is in training mode.
        counts (np.ndarray): Count of instances per class.
        class_weights (np.ndarray): Weight for each class (inverse frequency).
        agg_func (callable): Function to aggregate weights for images with multiple objects.
        weights (list): Aggregated weight for each image.
        probabilities (list): Sampling probability for each image.

    Methods:
        count_instances: Count the number of instances per class.
        calculate_weights: Calculate aggregated weight for each image.
        calculate_probabilities: Calculate sampling probabilities.
        __getitem__: Return transformed label information, sampled by probability during training.

    Examples:
        >>> from ultralytics.data.weighted_dataset import YOLOWeightedDataset
        >>> dataset = YOLOWeightedDataset(img_path="path/to/images", data={"names": {0: "person"}})
        >>> # Use with monkey-patching
        >>> import ultralytics.data.build as build
        >>> build.YOLODataset = YOLOWeightedDataset
    """

    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the YOLOWeightedDataset.

        Args:
            *args (Any): Positional arguments passed to YOLODataset.
            mode (str): Dataset mode, used to determine if in training mode.
            **kwargs (Any): Keyword arguments passed to YOLODataset.
        """
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        # Determine if we're in training mode based on prefix
        self.train_mode = "train" in self.prefix

        # Calculate class weights and sampling probabilities
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function to combine weights for images with multiple objects
        # Can be changed to np.max, np.min, etc. for different weighting strategies
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class in the dataset.

        This method iterates through all labels and counts how many instances of each
        class are present. Classes with zero instances are set to 1 to avoid division by zero.

        Returns:
            None: Updates self.counts attribute.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        # Avoid division by zero for classes with no instances
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each image based on class weights.

        For images with multiple objects, the weights of all objects are aggregated
        using the aggregation function (default: np.mean). Images with no objects
        receive a default weight of 1.

        Returns:
            list: A list of aggregated weights corresponding to each image.
        """
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            # Give a default weight to images with no objects (background)
            if cls.size == 0:
                weights.append(1)
                continue

            # Aggregate weights for all objects in the image
            # Default: mean of weights, but can be changed via self.agg_func
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Normalizes the weights to create a probability distribution that sums to 1.

        Returns:
            list: A list of sampling probabilities corresponding to each image.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.

        During training, images are sampled according to their probabilities to balance
        class distribution. During validation, images are accessed sequentially.

        Args:
            index (int): Index of the image (ignored during training with weighted sampling).

        Returns:
            dict: Transformed label information for the sampled image.
        """
        # Don't use weighted sampling for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            # Sample an index based on calculated probabilities
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))
