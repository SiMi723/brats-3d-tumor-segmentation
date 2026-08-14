from pathlib import Path

import numpy as np
import nibabel as nib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "BraTS2020_TrainingData"
    / "MICCAI_BraTS2020_TrainingData"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.txt"
)


def get_brain_shape(case_id):
    """
    Load FLAIR and find the shape of its non-zero brain region.
    """

    flair_path = (
        DATA_DIR
        / case_id
        / f"{case_id}_flair.nii"
    )

    flair = nib.load(flair_path).get_fdata()

    mask = flair != 0

    coordinates = np.argwhere(mask)

    min_coords = coordinates.min(axis=0)
    max_coords = coordinates.max(axis=0)

    shape = max_coords - min_coords + 1

    return shape


def load_case_ids():
    """Read patient IDs from train.txt."""

    with open(SPLIT_FILE, "r") as f:
        case_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return case_ids


if __name__ == "__main__":

    case_ids = load_case_ids()

    print(f"Analyzing {len(case_ids)} training cases...")

    shapes = []

    for i, case_id in enumerate(case_ids):

        shape = get_brain_shape(case_id)

        shapes.append(shape)

        print(
            f"{i + 1}/{len(case_ids)} "
            f"{case_id}: {tuple(shape)}"
        )

    shapes = np.array(shapes)

    minimum = shapes.min(axis=0)
    mean = shapes.mean(axis=0)
    maximum = shapes.max(axis=0)

    print("\n" + "=" * 50)
    print("CROPPED VOLUME STATISTICS")
    print("=" * 50)

    print(f"Number of cases: {len(shapes)}")

    print("\nMinimum dimensions:")
    print(tuple(minimum))

    print("\nMean dimensions:")
    print(tuple(np.round(mean, 2)))

    print("\nMaximum dimensions:")
    print(tuple(maximum))

    print("\nAnalysis completed.")