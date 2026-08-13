from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ["flair", "t1", "t1ce", "t2"]


def load_nifti(file_path):
    """Load a NIfTI file and return its voxel data and affine."""

    image = nib.load(file_path)

    data = image.get_fdata(dtype=np.float32)

    return data, image.affine


def load_case(case_dir):
    """
    Load all four MRI modalities and the segmentation
    mask for one BraTS case.
    """

    case_dir = Path(case_dir)
    case_id = case_dir.name

    volumes = {}
    affines = {}

    # Load MRI modalities
    for modality in MODALITIES:

        file_path = (
            case_dir / f"{case_id}_{modality}.nii"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing {modality} file: {file_path}"
            )

        volumes[modality], affines[modality] = load_nifti(
            str(file_path)
        )

    # Load segmentation
    seg_path = case_dir / f"{case_id}_seg.nii"

    if not seg_path.exists():
        raise FileNotFoundError(
            f"Missing segmentation file: {seg_path}"
        )

    volumes["seg"], affines["seg"] = load_nifti(
        str(seg_path)
    )

    # Check shapes
    reference_shape = volumes["flair"].shape

    for name, volume in volumes.items():

        if volume.shape != reference_shape:
            raise ValueError(
                f"Shape mismatch in {case_id}: "
                f"{name} has shape {volume.shape}, "
                f"expected {reference_shape}"
            )

    # Check affine matrices
    reference_affine = affines["flair"]

    for name, affine in affines.items():

        if not np.allclose(
            affine,
            reference_affine
        ):
            raise ValueError(
                f"Affine mismatch in {case_id}: "
                f"{name} does not match FLAIR"
            )

    return volumes, reference_affine

if __name__ == "__main__":

    case_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "BraTS2020_TrainingData"
        / "MICCAI_BraTS2020_TrainingData"
        / "BraTS20_Training_001"
    )

    volumes, affine = load_case(case_path)

    print("Case loaded successfully.")

    for name, volume in volumes.items():
        print(f"{name}: {volume.shape}")

    print("\nAffine:")
    print(affine)