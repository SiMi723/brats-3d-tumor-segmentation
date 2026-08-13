import numpy as np


def zscore_normalize(volume):
    """
    Z-score normalize non-background voxels.

    Background voxels remain zero.
    """

    volume = volume.astype(np.float32, copy=True)

    mask = volume != 0

    if not np.any(mask):
        return volume

    mean = volume[mask].mean()
    std = volume[mask].std()

    if std == 0:
        return volume

    volume[mask] = (
        volume[mask] - mean
    ) / std

    return volume


def normalize_case(volumes):
    """
    Normalize MRI modalities.

    The segmentation mask is NOT normalized.
    """

    normalized_volumes = {}

    for name, volume in volumes.items():

        if name == "seg":
            normalized_volumes[name] = volume
        else:
            normalized_volumes[name] = zscore_normalize(
                volume
            )

    return normalized_volumes