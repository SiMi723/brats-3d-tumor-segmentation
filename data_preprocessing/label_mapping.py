import numpy as np


LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
}


INVERSE_LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
}


def remap_labels(segmentation):
    """
    Convert BraTS labels to consecutive training labels.
    """

    mapped = np.zeros_like(
        segmentation,
        dtype=np.int64
    )

    for original_label, training_label in LABEL_MAP.items():

        mapped[segmentation == original_label] = (
            training_label
        )

    return mapped


def restore_labels(segmentation):
    """
    Convert consecutive training labels
    back to original BraTS labels.
    """

    restored = np.zeros_like(
        segmentation,
        dtype=np.int64
    )

    for training_label, original_label in INVERSE_LABEL_MAP.items():

        restored[segmentation == training_label] = (
            original_label
        )

    return restored