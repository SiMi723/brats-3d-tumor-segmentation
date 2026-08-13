import numpy as np


def get_brain_bbox(reference_volume):
    """
    Find the bounding box of the non-zero brain region.

    The reference volume will be FLAIR.
    """

    mask = reference_volume != 0

    if not np.any(mask):
        raise ValueError("No non-zero voxels found.")

    coordinates = np.argwhere(mask)

    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)

    return min_coords, max_coords


def crop_volume(volume, min_coords, max_coords):
    """
    Apply the same bounding box to a volume.
    """

    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords

    return volume[
        x_min:x_max + 1,
        y_min:y_max + 1,
        z_min:z_max + 1,
    ]


def crop_case(volumes):
    """
    Crop all modalities and the segmentation mask
    using one common brain bounding box.

    FLAIR is used as the reference volume.
    """

    min_coords, max_coords = get_brain_bbox(
        volumes["flair"]
    )

    cropped_volumes = {}

    for name, volume in volumes.items():

        cropped_volumes[name] = crop_volume(
            volume,
            min_coords,
            max_coords,
        )

    return cropped_volumes