import numpy as np


PATCH_SIZE = (96, 96, 96)
TUMOR_PATCH_RATIO = 0.70


def get_tumor_coordinates(segmentation):
    """
    Get coordinates of all tumor voxels.
    """

    tumor_mask = segmentation > 0

    coordinates = np.argwhere(tumor_mask)

    return coordinates


def get_random_center(volume_shape):
    """
    Select a random voxel from the volume.
    """

    return np.array([
        np.random.randint(0, volume_shape[0]),
        np.random.randint(0, volume_shape[1]),
        np.random.randint(0, volume_shape[2]),
    ])


def get_valid_start(center, volume_shape):
    """
    Convert a patch center into a valid patch starting position.

    The complete patch must remain inside the volume.
    """

    start = []

    for i in range(3):

        half_size = PATCH_SIZE[i] // 2

        min_start = 0
        max_start = volume_shape[i] - PATCH_SIZE[i]

        desired_start = center[i] - half_size

        desired_start = max(
            min_start,
            desired_start
        )

        desired_start = min(
            max_start,
            desired_start
        )

        start.append(desired_start)

    return np.array(start)


def extract_patch(volume, start):
    """
    Extract a 3D patch from a volume.
    """

    x, y, z = start

    px, py, pz = PATCH_SIZE

    patch = volume[
        x:x + px,
        y:y + py,
        z:z + pz,
    ]

    return patch


def sample_patch(volumes):
    """
    Sample one 3D training patch.

    Most patches are centered around tumor voxels,
    while some are sampled randomly.
    """

    segmentation = volumes["seg"]

    tumor_coordinates = get_tumor_coordinates(
        segmentation
    )

    volume_shape = segmentation.shape

    # Decide whether this patch should be tumor-centered
    if (
        len(tumor_coordinates) > 0
        and np.random.random() < TUMOR_PATCH_RATIO
    ):

        random_index = np.random.randint(
            len(tumor_coordinates)
        )

        center = tumor_coordinates[
            random_index
        ]

    else:

        center = get_random_center(
            volume_shape
        )

    # Make sure the patch stays inside the volume
    start = get_valid_start(
        center,
        volume_shape
    )

    # Extract MRI patches
    image_patches = []

    for modality in ["flair", "t1", "t1ce", "t2"]:

        patch = extract_patch(
            volumes[modality],
            start
        )

        image_patches.append(patch)

    # Convert four modalities into channels
    image_patch = np.stack(
        image_patches,
        axis=0
    )

    # Extract segmentation patch
    mask_patch = extract_patch(
        segmentation,
        start
    )

    return image_patch, mask_patch